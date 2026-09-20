"""TinyStories pretraining, quantization-aware fine-tuning, and checkpoint export."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import ExactCheckpoint
from .full_model import ExactFactorioLM
from .training import TrainingConfig, create_model, export_model, select_device, set_deterministic
from .vocabulary import FACTORIOLM_ALPHABET


PROMPTS = (
    "Once upon a time", "One day ", "There was ", "The little ", "The boy ",
    "The girl ", "He was ", "She was ", "Today ", "Factorio ",
)


@dataclass(frozen=True)
class Run4Config:
    seed: int = 4104
    train_characters: int = 24_000_000
    validation_characters: int = 2_000_000
    sequence_length: int = 64
    batch_size: int = 64
    float_steps: int = 6_000
    qat_steps: int = 4_000
    float_learning_rate: float = 0.002
    qat_learning_rate: float = 0.0005
    qat_logit_temperature: float = 256.0
    qat_distillation_weight: float = 0.25
    warmup_steps: int = 300
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    evaluation_interval: int = 250
    evaluation_characters: int = 16_384
    exact_evaluation_characters: int = 1_024
    ternary_threshold: float = 0.55
    target_sparsity: float = 0.51
    embedding_limit: int = 31
    relative_bias_limit: int = 6
    generation_length: int = 128
    device: str = "auto"

    @classmethod
    def read(cls, path: str | Path) -> "Run4Config":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-", "\u2026": "...",
    "\u00a0": " ", "\ufeff": "",
})

_DOCUMENT_SEPARATOR = re.compile(r"<\|endoftext\|>|<endoftext>", re.IGNORECASE)


def normalize_tinystories(text: str) -> tuple[str, dict[str, int]]:
    """Normalize source text into the immutable 64-character hardware ABI.

    The ABI has no punctuation tokens. Sentence punctuation therefore becomes
    a newline, apostrophes/quotes are removed, and other punctuation becomes a
    word-separating space.  NFKD retains ASCII base letters where possible.
    """

    source_characters = len(text)
    # TinyStories' serialized document boundary is metadata, not story text.
    # Remove the complete token before character-level punctuation handling;
    # otherwise ``<|endoftext|>`` becomes the learned literal ``endoftext``.
    text, document_separators = _DOCUMENT_SEPARATOR.subn("\n", text)
    text = unicodedata.normalize("NFKD", text.translate(_TYPOGRAPHY))
    allowed = set(FACTORIOLM_ALPHABET)
    output: list[str] = []
    unsupported = punctuation = typography = 0
    previous = ""
    for character in text:
        if character in allowed:
            rendered = character
        elif unicodedata.combining(character):
            unsupported += 1
            continue
        elif character in ".!?":
            rendered = "\n"
            punctuation += 1
        elif character in "'\"`":
            unsupported += 1
            continue
        elif character.isspace() or unicodedata.category(character).startswith(("P", "S")):
            rendered = " "
            punctuation += 1
        else:
            unsupported += 1
            continue
        if rendered == " " and previous in {"", " ", "\n"}:
            continue
        if rendered == "\n" and previous == "\n":
            continue
        output.append(rendered)
        previous = rendered
    normalized = "".join(output).strip(" ")
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    typography = sum(text.count(value) for value in ("'", '"', "-"))
    return normalized, {
        "source_characters": source_characters,
        "normalized_characters": len(normalized),
        "document_separators_mapped_to_boundaries": document_separators,
        "unsupported_removed": unsupported,
        "punctuation_or_symbols_mapped_to_boundaries": punctuation,
        "ascii_typography_after_translation": typography,
    }


def _normalized_prefix(source: Path, count: int) -> tuple[str, dict[str, int]]:
    raw = source.read_text(encoding="utf-8", errors="ignore")
    normalized, stats = normalize_tinystories(raw)
    if len(normalized) < count:
        raise ValueError(f"{source} normalized to {len(normalized):,} characters; need {count:,}")
    value = normalized[:count]
    # A deterministic prefix may end inside a story, but never inside UTF-8.
    stats["selected_characters"] = len(value)
    return value, stats


def _encode_binary(text: str) -> bytes:
    lookup = {ord(character): index for index, character in enumerate(FACTORIOLM_ALPHABET)}
    return bytes(lookup[ord(character)] for character in text)


def prepare_corpus(train_source: str | Path, validation_source: str | Path,
                   output_dir: str | Path, config: Run4Config) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: dict[str, Any] = {}
    for split, source_value, count in (
        ("train", train_source, config.train_characters),
        ("validation", validation_source, config.validation_characters),
    ):
        source = Path(source_value)
        normalized, statistics = _normalized_prefix(source, count)
        text_path = output / f"tinystories-v2-gpt4-{split}.txt"
        binary_path = output / f"tinystories-v2-gpt4-{split}.bin"
        text_path.write_text(normalized, encoding="ascii")
        binary_path.write_bytes(_encode_binary(normalized))
        rows[split] = {
            "source": str(source), "text": str(text_path), "tokens": str(binary_path),
            "text_sha256": _sha256(text_path), "token_sha256": _sha256(binary_path),
            **statistics,
        }
    report = {
        "dataset": "roneneldan/TinyStories TinyStoriesV2-GPT4",
        "source_revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "selection": "deterministic normalized prefix of the official train and validation files",
        "alphabet": FACTORIOLM_ALPHABET,
        "punctuation_constraint": (
            "The fixed 64-symbol hardware ABI has no punctuation tokens; sentence punctuation maps "
            "to newline, other punctuation to space, and quotes/apostrophes are removed."
        ),
        "splits": rows,
    }
    _write_json(output / "data-report.json", report)
    return report


def _torch():
    import torch
    return torch


def _maxnorm_float(value):
    return 31.0 * value / value.abs().amax(dim=-1, keepdim=True).clamp(min=1.0e-4)


def _build_float_model_class():
    torch = _torch()

    class FloatBlock(torch.nn.Module):
        def __init__(self, generator) -> None:
            super().__init__()

            def matrix(rows: int, columns: int, scale: float):
                return torch.nn.Parameter(torch.randn(rows, columns, generator=generator) * scale)

            self.query = matrix(32, 32, 0.05)
            self.key = matrix(32, 32, 0.05)
            self.value = matrix(32, 32, 0.05)
            self.output = matrix(32, 32, 0.02)
            self.ffn_up = matrix(64, 32, 0.05)
            self.ffn_down = matrix(32, 64, 0.02)
            initial_bias = torch.zeros(4, 64)
            for head in range(4):
                for age in range(64):
                    initial_bias[head, age] = max(-2, 1 - age // (9 + head))
            self.relative_bias = torch.nn.Parameter(initial_bias)

        def forward(self, x, score_divisor: int):
            normalized = _maxnorm_float(x)
            q = normalized @ self.query.T
            k = normalized @ self.key.T
            v = normalized @ self.value.T
            batch, length, _ = q.shape
            qh = q.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
            kh = k.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
            vh = v.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
            dots = torch.einsum("bhtd,bhsd->bhts", qh, kh)
            positions = torch.arange(length, device=x.device)
            ages = positions[:, None] - positions[None, :]
            causal = (ages >= 0) & (ages < 64)
            bias = self.relative_bias[:, ages.clamp(0, 63)]
            scores = (dots / score_divisor + bias.unsqueeze(0)).clamp(-6, 6)
            weights = torch.pow(2.0, scores + 6) * causal[None, None]
            attention = torch.einsum("bhts,bhsd->bhtd", weights, vh)
            attention = attention / weights.sum(dim=-1, keepdim=True)
            attention = attention.permute(0, 2, 1, 3).reshape(batch, length, 32)
            residual = x + attention @ self.output.T
            return residual + torch.relu(_maxnorm_float(residual) @ self.ffn_up.T) @ self.ffn_down.T

    class FloatFactorioLM(torch.nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.score_divisor = 512
            self.embeddings = torch.nn.Parameter(
                torch.empty(64, 32).uniform_(-8.0, 8.0, generator=generator)
            )
            self.layers = torch.nn.ModuleList([FloatBlock(generator) for _ in range(6)])
            self.lm_head = torch.nn.Parameter(torch.randn(64, 32, generator=generator) * 0.05)

        def forward(self, token_ids):
            x = self.embeddings[token_ids]
            for layer in self.layers:
                x = layer(x, self.score_divisor)
            return x @ self.lm_head.T

        def constrain_latents(self) -> None:
            with torch.no_grad():
                self.embeddings.clamp_(-31, 31)
                self.lm_head.clamp_(-2, 2)
                for layer in self.layers:
                    layer.relative_bias.clamp_(-6, 6)
                    for name in ("query", "key", "value", "output", "ffn_up", "ffn_down"):
                        getattr(layer, name).clamp_(-2, 2)

    return FloatFactorioLM


def create_float_model(config: Run4Config):
    set_deterministic(config.seed)
    return _build_float_model_class()(config.seed)


def _matrix_parameters(model) -> Iterable[Any]:
    yield model.lm_head
    for layer in model.layers:
        for name in ("query", "key", "value", "output", "ffn_up", "ffn_down"):
            yield getattr(layer, name)


def project_qat_sparsity(model, target_sparsity: float) -> None:
    """Keep one uniform sparsity target without per-layer optimization."""

    torch = _torch()
    with torch.no_grad():
        for matrix in _matrix_parameters(model):
            flat = matrix.flatten()
            nonzero = max(1, int(round(flat.numel() * (1.0 - target_sparsity))))
            chosen = torch.topk(flat.abs(), nonzero, sorted=False).indices
            mask = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
            mask[chosen] = True
            signs = torch.where(flat >= 0, torch.ones_like(flat), -torch.ones_like(flat))
            flat[mask] = signs[mask] * flat[mask].abs().clamp(min=model.threshold + 0.04, max=1.5)
            flat[~mask] = flat[~mask].clamp(min=-model.threshold + 0.04,
                                           max=model.threshold - 0.04)


def initialize_qat_from_float(float_model, config: Run4Config):
    torch = _torch()
    training_config = TrainingConfig(
        seed=config.seed, sequence_length=64, batch_size=config.batch_size,
        ternary_threshold=config.ternary_threshold, embedding_limit=config.embedding_limit,
        relative_bias_limit=config.relative_bias_limit, device=config.device,
    )
    qat = create_model(training_config, 512)
    with torch.no_grad():
        qat.embeddings.copy_(float_model.embeddings.round().clamp(-31, 31))
        qat.lm_head.copy_(float_model.lm_head)
        for float_layer, qat_layer in zip(float_model.layers, qat.layers, strict=True):
            qat_layer.relative_bias.copy_(float_layer.relative_bias.round().clamp(-6, 6))
            for name in ("query", "key", "value", "output", "ffn_up", "ffn_down"):
                source = getattr(float_layer, name)
                target = getattr(qat_layer, name)
                quantile = torch.quantile(source.abs().flatten(), config.target_sparsity).clamp(min=1.0e-6)
                target.copy_((source * (config.ternary_threshold / quantile)).clamp(-1.5, 1.5))
        quantile = torch.quantile(float_model.lm_head.abs().flatten(), config.target_sparsity).clamp(min=1.0e-6)
        qat.lm_head.copy_((float_model.lm_head * (config.ternary_threshold / quantile)).clamp(-1.5, 1.5))
    project_qat_sparsity(qat, config.target_sparsity)
    return qat


class TokenCorpus:
    def __init__(self, path: str | Path) -> None:
        torch = _torch()
        raw = bytearray(Path(path).read_bytes())
        self.tokens = torch.frombuffer(raw, dtype=torch.uint8).clone()
        if self.tokens.numel() < 128:
            raise ValueError("token corpus must contain at least 128 tokens")

    def batch(self, starts, length: int, device: str):
        torch = _torch()
        indexes = starts[:, None] + torch.arange(length + 1)
        return self.tokens[indexes].to(dtype=torch.long, device=device)


def evaluate_torch(model, corpus: TokenCorpus, *, device: str, length: int,
                   characters: int, batch_size: int,
                   optimization_temperature: float = 1.0) -> dict[str, float | int]:
    torch = _torch()
    model.eval()
    total_loss = optimization_loss = total_correct = total = 0
    windows = max(1, math.ceil(characters / length))
    with torch.no_grad():
        for first in range(0, windows, batch_size):
            count = min(batch_size, windows - first)
            starts = (torch.arange(first, first + count) * length) % (len(corpus.tokens) - length - 1)
            values = corpus.batch(starts, length, device)
            logits = model(values[:, :-1])
            targets = values[:, 1:]
            total_loss += float(torch.nn.functional.cross_entropy(
                logits.reshape(-1, 64), targets.reshape(-1), reduction="sum").cpu())
            optimization_loss += float(torch.nn.functional.cross_entropy(
                (logits / optimization_temperature).reshape(-1, 64),
                targets.reshape(-1), reduction="sum").cpu())
            total_correct += int((logits.argmax(dim=-1) == targets).sum().cpu())
            total += targets.numel()
    cross_entropy = total_loss / total
    return {
        "cross_entropy": cross_entropy,
        "perplexity": math.exp(cross_entropy) if cross_entropy < 709.0 else float("inf"),
        "accuracy": total_correct / total,
        "characters": total,
        "optimization_cross_entropy": optimization_loss / total,
        "optimization_temperature": optimization_temperature,
    }


def _learning_rate(step: int, steps: int, warmup: int, maximum: float) -> float:
    if step < warmup:
        return maximum * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, steps - warmup - 1)
    return maximum * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_stage(model, train: TokenCorpus, validation: TokenCorpus, config: Run4Config,
                *, stage: str, device: str, checkpoint_path: Path,
                resume_path: str | Path | None = None, teacher=None) -> dict[str, Any]:
    torch = _torch()
    if stage not in {"float", "qat"}:
        raise ValueError("stage must be float or qat")
    steps = config.float_steps if stage == "float" else config.qat_steps
    maximum_lr = config.float_learning_rate if stage == "float" else config.qat_learning_rate
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=maximum_lr,
                                  weight_decay=config.weight_decay)
    start_step = 0
    best_ce = float("inf")
    if resume_path:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
        best_ce = float(state.get("best_cross_entropy", best_ce))
        previous_config = state.get("config", {})
        if (stage == "qat"
                and float(previous_config.get("qat_logit_temperature", config.qat_logit_temperature))
                != config.qat_logit_temperature):
            # A changed temperature changes the validation objective's scale;
            # keep the weights/optimizer but establish a fresh comparable best.
            best_ce = float("inf")
    generator = torch.Generator(device="cpu").manual_seed(config.seed + (0 if stage == "float" else 100_000))
    log: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(start_step, steps):
        model.train()
        starts = torch.randint(0, len(train.tokens) - config.sequence_length - 1,
                               (config.batch_size,), generator=generator)
        values = train.batch(starts, config.sequence_length, device)
        logits = model(values[:, :-1])
        if stage == "qat":
            temperature = config.qat_logit_temperature
            hard_loss = torch.nn.functional.cross_entropy(
                (logits / temperature).reshape(-1, 64), values[:, 1:].reshape(-1))
            loss = hard_loss
            if teacher is not None and config.qat_distillation_weight:
                teacher.eval()
                with torch.no_grad():
                    teacher_probabilities = torch.softmax(teacher(values[:, :-1]), dim=-1)
                distillation = torch.nn.functional.kl_div(
                    torch.log_softmax(logits / temperature, dim=-1),
                    teacher_probabilities, reduction="batchmean") / config.sequence_length
                loss = ((1.0 - config.qat_distillation_weight) * hard_loss
                        + config.qat_distillation_weight * distillation)
        else:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), values[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip).cpu())
        lr = _learning_rate(step, steps, config.warmup_steps, maximum_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        model.constrain_latents()
        if stage == "qat":
            project_qat_sparsity(model, config.target_sparsity)
        evaluate = step == start_step or (step + 1) % config.evaluation_interval == 0 or step + 1 == steps
        if evaluate:
            metrics = evaluate_torch(
                model, validation, device=device, length=config.sequence_length,
                characters=config.evaluation_characters, batch_size=config.batch_size,
                optimization_temperature=(config.qat_logit_temperature if stage == "qat" else 1.0),
            )
            row = {"step": step + 1, "train_cross_entropy": float(loss.detach().cpu()),
                   "gradient_norm_before_clip": gradient_norm, "learning_rate": lr,
                   "validation": metrics, "elapsed_seconds": time.monotonic() - started}
            log.append(row)
            print(f"{stage} step {step + 1}/{steps}: train CE={row['train_cross_entropy']:.4f} "
                  f"val CE={metrics['cross_entropy']:.4f} acc={metrics['accuracy']:.3%} lr={lr:.6g}",
                  flush=True)
            selection_ce = float(metrics["optimization_cross_entropy"])
            if selection_ce < best_ce:
                best_ce = selection_ce
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"stage": stage, "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(), "step": step + 1,
                            "best_cross_entropy": best_ce, "config": asdict(config)}, checkpoint_path)
    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    final = evaluate_torch(model, validation, device=device, length=config.sequence_length,
                           characters=config.evaluation_characters,
                           batch_size=config.batch_size,
                           optimization_temperature=(config.qat_logit_temperature if stage == "qat" else 1.0))
    return {"stage": stage, "configured_steps": steps, "best_step": int(best["step"]),
            "best_validation": final, "elapsed_seconds": time.monotonic() - started,
            "checkpoint": str(checkpoint_path), "log": log}


def evaluate_exact(checkpoint: ExactCheckpoint, tokens: bytes, *, characters: int) -> dict[str, Any]:
    count = min(characters, len(tokens) - 1)
    loss = 0.0
    correct = 0
    evaluated = 0
    # Match evaluate_torch: non-overlapping ordinary 64-character windows,
    # with cache state reset at each training-window boundary.
    for start in range(0, count, 64):
        model = ExactFactorioLM(checkpoint)
        window_count = min(64, count - start)
        for token, target in zip(tokens[start:start + window_count],
                                 tokens[start + 1:start + window_count + 1], strict=True):
            logits = model.step(token).logits
            maximum = max(logits)
            loss += maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[target]
            correct += target == max(range(64), key=lambda index: (logits[index], -index))
            evaluated += 1
    cross_entropy = loss / evaluated
    return {"cross_entropy": cross_entropy,
            "perplexity": math.exp(cross_entropy) if cross_entropy < 709.0 else float("inf"),
            "accuracy": correct / evaluated, "characters": evaluated}


def generate_samples(checkpoint: ExactCheckpoint, length: int) -> dict[str, str]:
    return {prompt: ExactFactorioLM(checkpoint).generate(prompt, length)[0] for prompt in PROMPTS}


def _aggregate_sparsity(checkpoint: ExactCheckpoint) -> dict[str, Any]:
    report = checkpoint.sparsity_report()
    total = sum(int(row["total"]) for row in report.values())
    zero = sum(int(row["zero"]) for row in report.values())
    return {"zero": zero, "nonzero": total - zero, "total": total,
            "sparsity_percentage": 100.0 * zero / total, "matrices": report}


def run_training(*, config_path: str | Path, train_source: str | Path,
                 validation_source: str | Path, output_dir: str | Path,
                 resume_float: str | Path | None = None,
                 resume_qat: str | Path | None = None) -> dict[str, Any]:
    config = Run4Config.read(config_path)
    if config.sequence_length != 64:
        raise ValueError("Run 4 must train ordinary 64-character windows")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_deterministic(config.seed)
    random.seed(config.seed)
    data = prepare_corpus(train_source, validation_source, output, config)
    train = TokenCorpus(data["splits"]["train"]["tokens"])
    validation = TokenCorpus(data["splits"]["validation"]["tokens"])
    device = select_device(config.device)

    float_model = create_float_model(config)
    float_path = output / "factorio-lm-quality-float.pt"
    float_training = train_stage(float_model, train, validation, config, stage="float",
                                 device=device, checkpoint_path=float_path,
                                 resume_path=resume_float)
    float_metrics = evaluate_torch(float_model, validation, device=device,
                                   length=64, characters=config.evaluation_characters,
                                   batch_size=config.batch_size)

    qat_model = initialize_qat_from_float(float_model, config)
    qat_path = output / "factorio-lm-quality-qat.pt"
    qat_training = train_stage(qat_model, train, validation, config, stage="qat",
                               device=device, checkpoint_path=qat_path,
                               resume_path=resume_qat, teacher=float_model)
    qat_metrics = evaluate_torch(qat_model, validation, device=device,
                                 length=64, characters=config.evaluation_characters,
                                 batch_size=config.batch_size,
                                 optimization_temperature=config.qat_logit_temperature)
    metadata = {
        "run": 4, "dataset": "TinyStoriesV2-GPT4", "seed": config.seed,
        "float_checkpoint": str(float_path), "qat_checkpoint": str(qat_path),
        "score_divisor": 512, "training_characters": config.train_characters,
        "latent_parameters_required_for_inference": False,
    }
    exact_checkpoint = export_model(qat_model, metadata=metadata)
    exact_path = exact_checkpoint.write(output / "factorio-lm-quality-exact.json")
    validation_bytes = Path(data["splits"]["validation"]["tokens"]).read_bytes()
    exact_metrics = evaluate_exact(exact_checkpoint, validation_bytes,
                                   characters=config.exact_evaluation_characters)
    run3_path = Path("generated/run3/factorio-lm-smoke-exact.json")
    run3_metrics = None
    if run3_path.exists():
        run3_metrics = evaluate_exact(ExactCheckpoint.read(run3_path), validation_bytes,
                                      characters=config.exact_evaluation_characters)
    samples = generate_samples(exact_checkpoint, config.generation_length)
    run3_samples = generate_samples(ExactCheckpoint.read(run3_path), config.generation_length) if run3_path.exists() else {}
    # A numerical improvement over the smoke checkpoint is necessary but not
    # sufficient. Below 30% next-character accuracy these exact generations
    # are empirically dominated by fragments/repetition rather than simple
    # grammatical English, so they do not pass the language-quality gate.
    clearly_better = bool(
        run3_metrics
        and exact_metrics["accuracy"] > run3_metrics["accuracy"] + 0.02
        and exact_metrics["cross_entropy"] < run3_metrics["cross_entropy"]
        and exact_metrics["accuracy"] >= 0.30
    )
    sample_path = output / "generation-samples.txt"
    sample_path.write_text("".join(
        f"PROMPT: {prompt!r}\nEXACT: {sample}\nRUN 3: {run3_samples.get(prompt, '')}\n\n"
        for prompt, sample in samples.items()
    ), encoding="utf-8")
    report = {
        "configuration": asdict(config), "device": device, "data": data,
        "training": {"float": float_training, "qat": qat_training},
        "metrics": {"float": float_metrics, "qat": qat_metrics, "exact": exact_metrics,
                    "run3_exact_same_validation_slice": run3_metrics},
        "samples": samples, "run3_samples": run3_samples,
        "sparsity": _aggregate_sparsity(exact_checkpoint),
        "clearly_better_than_run3": clearly_better,
        "quality_gate": {
            "minimum_exact_accuracy": 0.30,
            "numerically_better_than_run3": bool(
                run3_metrics
                and exact_metrics["accuracy"] > run3_metrics["accuracy"] + 0.02
                and exact_metrics["cross_entropy"] < run3_metrics["cross_entropy"]
            ),
            "recognizably_english": clearly_better,
        },
        "exact_candidate_checkpoint": str(exact_path),
        "winning_checkpoint": str(exact_path) if clearly_better else None,
    }
    _write_json(output / "training-report.json", report)
    return {"checkpoint": exact_checkpoint, "report": report, "output": output}
