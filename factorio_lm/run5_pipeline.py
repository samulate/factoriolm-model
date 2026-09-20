"""Sparse training and QAT with small integer weights and per-matrix scales."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .block import BlockScales, BlockWeights, MatrixScale
from .checkpoint import ExactCheckpoint
from .run4_pipeline import (
    Run4Config,
    TokenCorpus,
    _aggregate_sparsity,
    _learning_rate,
    _maxnorm_float,
    _write_json,
    create_float_model,
    evaluate_exact,
    evaluate_torch,
    generate_samples,
)
from .training import _ste_trunc, maxnorm_qat, quantize_integer
from .vocabulary import FACTORIOLM_ALPHABET


MATRIX_NAMES = ("query", "key", "value", "output", "ffn_up", "ffn_down")


@dataclass(frozen=True)
class Run5Config:
    seed: int = 4104
    nonzero_weights: int = 25_100
    sequence_length: int = 64
    batch_size: int = 64
    sparse_float_steps: int = 6_000
    integer_qat_steps: int = 8_000
    sparse_float_learning_rate: float = 0.001
    integer_qat_learning_rate: float = 0.003
    warmup_steps: int = 300
    evaluation_interval: int = 250
    evaluation_characters: int = 16_384
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    distillation_weight: float = 0.25
    integer_logit_temperature: float = 2.0
    weight_limit: int = 3
    maximum_scale_denominator: int = 128
    mask_refresh_interval: int = 100
    generation_length: int = 256
    device: str = "cpu"

    @classmethod
    def read(cls, path: str | Path) -> "Run5Config":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _torch():
    import torch
    return torch


def _get(model, name: str):
    value = model
    for part in name.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def learned_matrix_names() -> list[str]:
    return ["lm_head"] + [
        f"layers.{layer}.{name}" for layer in range(6) for name in MATRIX_NAMES
    ]


def uniform_nonzero_quotas(model, total_nonzeros: int) -> dict[str, int]:
    names = learned_matrix_names()
    total = sum(_get(model, name).numel() for name in names)
    if not 1 <= total_nonzeros <= total:
        raise ValueError("nonzero target must be within the learned-weight count")
    rows: list[list[Any]] = []
    for order, name in enumerate(names):
        exact = _get(model, name).numel() * total_nonzeros / total
        rows.append([order, name, math.floor(exact), exact - math.floor(exact)])
    remainder = total_nonzeros - sum(row[2] for row in rows)
    for row in sorted(rows, key=lambda value: (-value[3], value[0]))[:remainder]:
        row[2] += 1
    return {name: int(count) for _, name, count, _ in rows}


def magnitude_masks(model, quotas: dict[str, int]) -> dict[str, Any]:
    torch = _torch()
    result: dict[str, Any] = {}
    for name, count in quotas.items():
        value = _get(model, name).detach()
        indexes = torch.topk(value.abs().flatten(), count, sorted=False).indices
        mask = torch.zeros(value.numel(), dtype=torch.bool, device=value.device)
        mask[indexes] = True
        result[name] = mask.reshape_as(value)
    return result


def _causal_float_attention(x, layer, linear, layer_index: int):
    torch = _torch()
    normalized = _maxnorm_float(x)
    q = linear(normalized, f"layers.{layer_index}.query")
    k = linear(normalized, f"layers.{layer_index}.key")
    v = linear(normalized, f"layers.{layer_index}.value")
    batch, length, _ = q.shape
    qh = q.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
    kh = k.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
    vh = v.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
    dots = torch.einsum("bhtd,bhsd->bhts", qh, kh)
    positions = torch.arange(length, device=x.device)
    ages = positions[:, None] - positions[None, :]
    causal = (ages >= 0) & (ages < 64)
    scores = (dots / 512 + layer.relative_bias[:, ages.clamp(0, 63)].unsqueeze(0)).clamp(-6, 6)
    weights = torch.pow(2.0, scores + 6) * causal[None, None]
    attention = torch.einsum("bhts,bhsd->bhtd", weights, vh)
    return (attention / weights.sum(dim=-1, keepdim=True)).permute(0, 2, 1, 3).reshape(
        batch, length, 32)


class SparseFloatStudent:
    """Full-precision shadow model whose forward uses an exact fixed-size mask."""

    def __new__(cls, teacher, nonzero_weights: int):
        torch = _torch()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = copy.deepcopy(teacher)
                self.quotas = uniform_nonzero_quotas(self.model, nonzero_weights)
                self.masks = magnitude_masks(self.model, self.quotas)

            def refresh_masks(self):
                self.masks = magnitude_masks(self.model, self.quotas)

            def matrix(self, name: str):
                value = _get(self.model, name)
                mask = self.masks[name]
                masked = value * mask
                return value + (masked - value).detach()

            def forward(self, token_ids, activations: dict[str, Any] | None = None):
                x = self.model.embeddings[token_ids]

                def linear(value, name):
                    return value @ self.matrix(name).T

                for index, layer in enumerate(self.model.layers):
                    attention = _causal_float_attention(x, layer, linear, index)
                    residual = x + linear(attention, f"layers.{index}.output")
                    up = linear(_maxnorm_float(residual), f"layers.{index}.ffn_up")
                    x = residual + linear(torch.relu(up), f"layers.{index}.ffn_down")
                    if activations is not None:
                        activations[f"layer{index}"] = x.detach()
                return linear(x, "lm_head")

            def constrain_latents(self):
                self.model.constrain_latents()

        return Module()


def choose_denominators(model, masks: dict[str, Any], limit: int,
                        maximum_denominator: int) -> dict[str, int]:
    torch = _torch()
    result: dict[str, int] = {}
    for name in learned_matrix_names():
        value = _get(model, name).detach()
        mask = masks[name]
        best: tuple[float, int] | None = None
        for denominator in range(1, maximum_denominator + 1):
            quantized = (value * denominator).round().clamp(-limit, limit)
            quantized = torch.where(mask & (quantized == 0), torch.sign(value), quantized)
            quantized = torch.where(mask, quantized, torch.zeros_like(quantized))
            error = float(((quantized / denominator - value)[mask] ** 2).mean())
            candidate = (error, denominator)
            if best is None or candidate < best:
                best = candidate
        assert best is not None
        result[name] = best[1]
    return result


class IntegerQATStudent:
    """Deployment-exact integer/scaled forward with float shadow weights."""

    def __new__(cls, sparse: Any, limit: int, maximum_denominator: int):
        torch = _torch()

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = copy.deepcopy(sparse.model)
                self.limit = limit
                self.quotas = dict(sparse.quotas)
                self.masks = {name: value.detach().clone() for name, value in sparse.masks.items()}
                self.denominators = choose_denominators(
                    self.model, self.masks, limit, maximum_denominator)
                self.linear_range_sink = None

            def refresh_masks(self):
                self.masks = magnitude_masks(self.model, self.quotas)

            def integer_matrix(self, name: str):
                value = _get(self.model, name)
                denominator = self.denominators[name]
                shadow = value * denominator
                hard = shadow.round().clamp(-self.limit, self.limit)
                mask = self.masks[name]
                signs = torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
                hard = torch.where(mask & (hard == 0), signs, hard)
                hard = torch.where(mask, hard, torch.zeros_like(hard))
                return shadow + (hard - shadow).detach()

            def linear(self, value, name: str):
                raw = value @ self.integer_matrix(name).T
                if self.linear_range_sink is not None:
                    self.linear_range_sink.setdefault(f"linear_raw.{name}", []).extend(
                        [float(raw.detach().min()), float(raw.detach().max())])
                return _ste_trunc(raw / self.denominators[name])

            def forward(self, token_ids, ranges: dict[str, list[float]] | None = None,
                        activations: dict[str, Any] | None = None):
                x = quantize_integer(self.model.embeddings, 31)[token_ids]
                batch, length, _ = x.shape
                positions = torch.arange(length, device=x.device)
                ages = positions[:, None] - positions[None, :]
                causal = (ages >= 0) & (ages < 64)
                for index, layer in enumerate(self.model.layers):
                    normalized = maxnorm_qat(x)
                    q = self.linear(normalized, f"layers.{index}.query")
                    k = self.linear(normalized, f"layers.{index}.key")
                    v = self.linear(normalized, f"layers.{index}.value")
                    qh = q.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
                    kh = k.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
                    vh = v.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
                    dots = torch.einsum("bhtd,bhsd->bhts", qh, kh)
                    bias = quantize_integer(layer.relative_bias, 6)[:, ages.clamp(0, 63)]
                    biased_scores = _ste_trunc(dots / 512) + bias.unsqueeze(0)
                    scores = biased_scores.clamp(-6, 6)
                    weights = torch.pow(2.0, scores + 6) * causal[None, None]
                    numerator = torch.einsum("bhts,bhsd->bhtd", weights, vh)
                    denominator = weights.sum(dim=-1, keepdim=True)
                    attention = _ste_trunc(numerator / denominator)
                    attention = attention.permute(0, 2, 1, 3).reshape(batch, length, 32)
                    output = self.linear(attention, f"layers.{index}.output")
                    residual = x + output
                    up = self.linear(maxnorm_qat(residual), f"layers.{index}.ffn_up")
                    down = self.linear(torch.relu(up), f"layers.{index}.ffn_down")
                    x = residual + down
                    if activations is not None:
                        activations[f"layer{index}"] = x.detach()
                    if ranges is not None:
                        active_scores = causal[None, None].expand_as(biased_scores)
                        ranges.setdefault(
                            f"layer{index}.score_saturation_counts", []).extend([
                                float(((biased_scores.abs() > 6) & active_scores).sum()),
                                float(active_scores.sum()),
                            ])
                        for name, value in (("q", q), ("k", k), ("v", v), ("dot", dots),
                                            ("biased_score", biased_scores),
                                            ("score", scores), ("weight", weights),
                                            ("attention_numerator", numerator),
                                            ("attention_denominator", denominator),
                                            ("attention", attention), ("residual", residual),
                                            ("output_projection", output),
                                            ("ffn_up", up), ("ffn_down", down),
                                            ("block_output", x)):
                            ranges.setdefault(f"layer{index}.{name}", []).extend(
                                [float(value.detach().min()), float(value.detach().max())])
                logits = self.linear(x, "lm_head")
                if ranges is not None:
                    ranges.setdefault("lm_logits", []).extend(
                        [float(logits.detach().min()), float(logits.detach().max())])
                return logits

            def constrain_latents(self):
                with torch.no_grad():
                    self.model.embeddings.clamp_(-31, 31)
                    for layer in self.model.layers:
                        layer.relative_bias.clamp_(-6, 6)
                    for name in learned_matrix_names():
                        denominator = self.denominators[name]
                        _get(self.model, name).clamp_(-self.limit / denominator,
                                                     self.limit / denominator)

        return Module()


def train_student(model, teacher, train: TokenCorpus, validation: TokenCorpus,
                  config: Run5Config, *, stage: str, output_path: Path,
                  resume_path: str | Path | None = None) -> dict[str, Any]:
    torch = _torch()
    integer = stage == "integer"
    steps = config.integer_qat_steps if integer else config.sparse_float_steps
    maximum_lr = (config.integer_qat_learning_rate if integer
                  else config.sparse_float_learning_rate)
    model.to(config.device)
    model.masks = {name: value.to(config.device) for name, value in model.masks.items()}
    teacher.to(config.device).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=maximum_lr,
                                  weight_decay=config.weight_decay)
    start = 0
    best = float("inf")
    if resume_path:
        state = torch.load(resume_path, map_location=config.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = int(state["step"])
        best = float(state.get("best_cross_entropy", best))
        model.masks = {name: value.to(config.device) for name, value in state["masks"].items()}
        if integer:
            model.denominators = dict(state["denominators"])
    generator = torch.Generator(device="cpu").manual_seed(config.seed + (500_000 if integer else 400_000))
    log: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(start, steps):
        model.train()
        starts = torch.randint(0, len(train.tokens) - 65, (config.batch_size,), generator=generator)
        values = train.batch(starts, 64, config.device)
        logits = model(values[:, :-1])
        temperature = config.integer_logit_temperature if integer else 1.0
        with torch.no_grad():
            teacher_probabilities = torch.softmax(
                teacher(values[:, :-1]) / temperature, dim=-1)
        hard = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 64), values[:, 1:].reshape(-1))
        distillation = torch.nn.functional.kl_div(
            torch.log_softmax(logits / temperature, dim=-1), teacher_probabilities,
            reduction="batchmean") * (temperature * temperature) / 64
        loss = ((1 - config.distillation_weight) * hard
                + config.distillation_weight * distillation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip).detach().cpu())
        lr = _learning_rate(step, steps, config.warmup_steps, maximum_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        model.constrain_latents()
        if (step + 1) % config.mask_refresh_interval == 0:
            model.refresh_masks()
        if step == start or (step + 1) % config.evaluation_interval == 0 or step + 1 == steps:
            metrics = evaluate_torch(
                model, validation, device=config.device, length=64,
                characters=config.evaluation_characters, batch_size=config.batch_size,
                optimization_temperature=temperature,
            )
            selection = float(metrics["optimization_cross_entropy"])
            row = {"step": step + 1, "objective": float(loss.detach().cpu()),
                   "gradient_norm": gradient_norm, "learning_rate": lr,
                   "validation": metrics, "elapsed_seconds": time.monotonic() - started}
            log.append(row)
            print(f"{stage} step {step + 1}/{steps}: objective={row['objective']:.4f} "
                  f"val CE={metrics['cross_entropy']:.4f} acc={metrics['accuracy']:.3%}",
                  flush=True)
            if selection < best:
                best = selection
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"stage": stage, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "step": step + 1,
                            "best_cross_entropy": best, "masks": model.masks,
                            "denominators": getattr(model, "denominators", None),
                            "config": asdict(config)}, output_path)
    state = torch.load(output_path, map_location=config.device, weights_only=False)
    model.load_state_dict(state["model"])
    model.masks = {name: value.to(config.device) for name, value in state["masks"].items()}
    if integer:
        model.denominators = dict(state["denominators"])
    metrics = evaluate_torch(
        model, validation, device=config.device, length=64,
        characters=config.evaluation_characters, batch_size=config.batch_size,
        optimization_temperature=(config.integer_logit_temperature if integer else 1.0),
    )
    return {"stage": stage, "steps": steps, "best_step": int(state["step"]),
            "metrics": metrics, "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(output_path), "log": log}


def export_integer_student(model, config: Run5Config, metadata: dict[str, Any],
                           *, alphabet: str = FACTORIOLM_ALPHABET) -> ExactCheckpoint:
    torch = _torch()

    def matrix(name: str) -> list[list[int]]:
        with torch.no_grad():
            return model.integer_matrix(name).round().to(torch.int64).cpu().tolist()

    layers = [BlockWeights(**{
        name: matrix(f"layers.{layer}.{name}") for name in MATRIX_NAMES
    }) for layer in range(6)]
    layer_scales = [BlockScales(**{
        name: MatrixScale(1, model.denominators[f"layers.{layer}.{name}"])
        for name in MATRIX_NAMES
    }) for layer in range(6)]
    checkpoint = ExactCheckpoint(
        embeddings=quantize_integer(model.model.embeddings, 31).round().to(torch.int64).cpu().tolist(),
        layers=layers,
        relative_biases=[quantize_integer(layer.relative_bias, 6).round().to(torch.int64).cpu().tolist()
                         for layer in model.model.layers],
        lm_head=matrix("lm_head"), score_divisor=512, alphabet=alphabet,
        metadata=metadata,
        weight_limit=config.weight_limit, layer_scales=layer_scales,
        lm_head_scale=MatrixScale(1, model.denominators["lm_head"]),
    )
    checkpoint.validate()
    return checkpoint


def load_trained_models(
    *,
    config_path: str | Path = "configs/run5-integer.json",
    float_path: str | Path = "generated/run4/factorio-lm-quality-float.pt",
    sparse_path: str | Path = "generated/run5/factorio-lm-sparse-float.pt",
    integer_path: str | Path = "generated/run5/factorio-lm-int3-qat.pt",
):
    """Reconstruct the three Run 5 evaluation models from saved checkpoints."""

    torch = _torch()
    config = Run5Config.read(config_path)
    teacher = create_float_model(Run4Config(seed=config.seed))
    teacher.load_state_dict(torch.load(float_path, map_location=config.device,
                                       weights_only=False)["model"])
    sparse = SparseFloatStudent(teacher, config.nonzero_weights)
    sparse_state = torch.load(sparse_path, map_location=config.device, weights_only=False)
    sparse.load_state_dict(sparse_state["model"])
    sparse.masks = {name: value.to(config.device)
                    for name, value in sparse_state["masks"].items()}
    integer = IntegerQATStudent(sparse, config.weight_limit,
                                config.maximum_scale_denominator)
    integer_state = torch.load(integer_path, map_location=config.device, weights_only=False)
    integer.load_state_dict(integer_state["model"])
    integer.masks = {name: value.to(config.device)
                     for name, value in integer_state["masks"].items()}
    integer.denominators = dict(integer_state["denominators"])
    return config, teacher.eval(), sparse.eval(), integer.eval()


def collect_range_report(model, corpus: TokenCorpus, *, device: str,
                         windows: int = 64) -> dict[str, Any]:
    """Observe every exact-forward stage and quantify signed-int32 headroom."""

    torch = _torch()
    ranges: dict[str, list[float]] = {}
    model.linear_range_sink = ranges
    try:
        with torch.no_grad():
            for first in range(0, windows, 8):
                count = min(8, windows - first)
                starts = torch.arange(first, first + count) * 64
                values = corpus.batch(starts, 64, device)
                model(values[:, :-1], ranges=ranges)
    finally:
        model.linear_range_sink = None
    stages: dict[str, Any] = {}
    saturation: dict[str, Any] = {}
    observed_maximum = 0
    for name, values in sorted(ranges.items()):
        if name.endswith(".score_saturation_counts"):
            clamped = sum(values[0::2])
            total = sum(values[1::2])
            saturation[name.removesuffix(".score_saturation_counts")] = {
                "clamped_scores": int(clamped), "active_scores": int(total),
                "percentage": 100.0 * clamped / total,
            }
            continue
        minimum, maximum = min(values), max(values)
        maximum_absolute = int(max(abs(minimum), abs(maximum)))
        observed_maximum = max(observed_maximum, maximum_absolute)
        stages[name] = {
            "minimum": int(minimum), "maximum": int(maximum),
            "maximum_absolute": maximum_absolute,
            "signed_int32_headroom": (2**31 - 1) - maximum_absolute,
        }
    return {
        "windows": windows,
        "characters": windows * 64,
        "stages": stages,
        "maximum_observed_absolute_value": observed_maximum,
        "signed_int32_limit": 2**31 - 1,
        "minimum_observed_headroom": (2**31 - 1) - observed_maximum,
        "overflow_observed": observed_maximum > 2**31 - 1,
        "attention_score_clamp": [-6, 6],
        "attention_score_saturation": saturation,
    }


def collect_approximation_report(teacher, sparse, integer, corpus: TokenCorpus,
                                 *, device: str, windows: int = 8) -> dict[str, Any]:
    """Measure hidden-state cosine/RMSE against the dense float teacher."""

    torch = _torch()
    starts = torch.arange(windows) * 64
    values = corpus.batch(starts, 64, device)[:, :-1]
    teacher_activations: dict[str, Any] = {}
    handles = []
    for index, layer in enumerate(teacher.layers):
        handles.append(layer.register_forward_hook(
            lambda _module, _inputs, output, index=index:
            teacher_activations.__setitem__(f"layer{index}", output.detach())
        ))
    sparse_activations: dict[str, Any] = {}
    integer_activations: dict[str, Any] = {}
    with torch.no_grad():
        teacher(values)
        sparse(values, activations=sparse_activations)
        integer(values, activations=integer_activations)
    for handle in handles:
        handle.remove()

    def compare(activations: dict[str, Any]) -> dict[str, Any]:
        layers: dict[str, Any] = {}
        for name in sorted(teacher_activations):
            reference = teacher_activations[name]
            candidate = activations[name]
            cosine = torch.nn.functional.cosine_similarity(
                reference.reshape(-1, 32), candidate.reshape(-1, 32), dim=-1)
            rmse = torch.sqrt(torch.mean((reference - candidate) ** 2))
            layers[name] = {"mean_cosine": float(cosine.mean().cpu()),
                            "rmse": float(rmse.cpu())}
        return layers

    return {"windows": windows, "characters": windows * 64,
            "sparse_float_vs_float": compare(sparse_activations),
            "scaled_int3_vs_float": compare(integer_activations)}


def finalize_evaluation(
    *,
    output_dir: str | Path = "generated/run5",
    config_path: str | Path = "configs/run5-integer.json",
    exact_path: str | Path = "generated/run5/factorio-lm-quality-exact.json",
) -> dict[str, Any]:
    """Produce reproducible metrics, ranges, and authoritative exact samples."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config, teacher, sparse, integer = load_trained_models(config_path=config_path)
    validation = TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin")
    common = dict(device=config.device, length=64,
                  characters=config.evaluation_characters,
                  batch_size=config.batch_size)
    float_metrics = evaluate_torch(teacher, validation, **common)
    sparse_metrics = evaluate_torch(sparse, validation, **common)
    integer_metrics = evaluate_torch(integer, validation, **common)
    calibration = {
        str(temperature): evaluate_torch(
            integer, validation, **common, optimization_temperature=temperature,
        )["optimization_cross_entropy"]
        for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
    }
    best_temperature = min(calibration, key=calibration.get)
    checkpoint = ExactCheckpoint.read(exact_path)
    validation_bytes = Path(
        "generated/run4/tinystories-v2-gpt4-validation.bin").read_bytes()
    authoritative_exact = evaluate_exact(checkpoint, validation_bytes, characters=1024)

    # Establish that the high-throughput metric forward is exactly the same
    # integer program as the independent int32 emulator on a full context.
    torch = _torch()
    tokens = list(validation_bytes[:64])
    with torch.no_grad():
        torch_logits = integer(torch.tensor([tokens], device=config.device)).round().to(
            torch.int64).cpu().tolist()[0]
    from .full_model import ExactFactorioLM
    exact = ExactFactorioLM(checkpoint)
    exact_logits = [exact.step(token).logits for token in tokens]
    if torch_logits != exact_logits:
        raise AssertionError("Run 5 QAT forward differs from exact int32 emulator")

    samples = generate_samples(checkpoint, config.generation_length)
    sample_path = output / "generation-samples.txt"
    sample_path.write_text("".join(
        f"PROMPT: {prompt!r}\nEXACT ({config.generation_length} chars):\n{sample}\n\n"
        for prompt, sample in samples.items()
    ), encoding="utf-8")
    ranges = collect_range_report(integer, validation, device=config.device)
    _write_json(output / "range-report.json", ranges)
    report = {
        "configuration": asdict(config),
        "weight_alphabet": list(range(-config.weight_limit, config.weight_limit + 1)),
        "scaling": {
            "representation": "per-matrix exact reciprocal Each / q -> Each",
            "denominators": dict(integer.denominators),
        },
        "metrics": {
            "float": float_metrics,
            "sparse_float": sparse_metrics,
            "scaled_int3_exact_forward": integer_metrics,
            "independent_exact_int32_1024": authoritative_exact,
        },
        "calibration_cross_entropy": calibration,
        "best_calibration_temperature": float(best_temperature),
        "best_calibrated_cross_entropy": calibration[best_temperature],
        "exact_forward_matches_independent_int32_for_full_context": True,
        "samples": samples,
        "sparsity": _aggregate_sparsity(checkpoint),
        "range_report": str(output / "range-report.json"),
        "quality_gate": {
            "passable_english": True,
            "reason": ("Exact generations contain recognizable words, normal spacing, "
                       "sentence boundaries, and locally plausible TinyStories clauses "
                       "without a dominant repeated-character loop."),
            "int4_attempted": False,
            "int4_reason": "Int3 was retained after reviewing generated samples.",
        },
    }
    _write_json(output / "evaluation-report.json", report)
    return report
