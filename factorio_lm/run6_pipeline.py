"""Search sparse int4 weights and scales, then export the selected checkpoint."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import re
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .checkpoint import ExactCheckpoint
from .full_model import ExactFactorioLM
from .run4_pipeline import (
    TokenCorpus,
    _aggregate_sparsity,
    _write_json,
    evaluate_exact,
    evaluate_torch,
)
from .run5_pipeline import (
    IntegerQATStudent,
    Run5Config,
    _get,
    learned_matrix_names,
    load_trained_models,
    export_integer_student,
    collect_range_report,
    train_student,
)
from .training import maxnorm_qat, quantize_integer
from .vocabulary import Vocabulary


RUN6_CONFIG = Path("configs/run6-int4.json")
RUN6_REFINE_CONFIG = Path("configs/run6-int4-refine.json")
RUN6_OUTPUT = Path("generated/run6")
RUN6_INT4_RUN5_MASK = RUN6_OUTPUT / "factorio-lm-int4-run5-mask.pt"
RUN6_IMPORTANCE = RUN6_OUTPUT / "importance-scores.pt"
RUN6_SPARSE_IMPORTANCE = RUN6_OUTPUT / "factorio-lm-sparse-importance.pt"
RUN6_INT4_IMPORTANCE = RUN6_OUTPUT / "factorio-lm-int4-importance.pt"

GENERAL_PROMPTS = (
    "Once upon a time", "One day ", "There was ", "The little ",
    "The boy ", "The girl ", "He was ", "She was ", "Today ",
    "A little ", "The dog ", "The cat ", "Mom said ", "She said ",
    "He wanted ", "The house ",
)


class IncrementalIntegerBatch:
    """Batched integer inference with live K/V caches for fast prompt searches."""

    def __init__(self, model, batch_size: int, device: str = "cpu") -> None:
        import torch

        self.torch = torch
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.embeddings = quantize_integer(
            model.model.embeddings, 31).detach().to(device)
        self.matrices = {
            name: model.integer_matrix(name).detach().to(device)
            for name in learned_matrix_names()
        }
        self.denominators = dict(model.denominators)
        self.biases = [quantize_integer(layer.relative_bias, 6).detach().to(device)
                       for layer in model.model.layers]
        self.keys: list[Any | None] = [None] * 6
        self.values: list[Any | None] = [None] * 6

    def linear(self, value, name: str):
        return self.torch.trunc(
            (value @ self.matrices[name].T) / self.denominators[name])

    def step(self, token_ids):
        torch = self.torch
        token_ids = torch.as_tensor(token_ids, dtype=torch.long, device=self.device)
        if token_ids.shape != (self.batch_size,):
            raise ValueError("token batch does not match recurrent batch size")
        x = self.embeddings[token_ids]
        for index, _layer in enumerate(self.model.model.layers):
            normalized = maxnorm_qat(x)
            q = self.linear(normalized, f"layers.{index}.query").reshape(
                self.batch_size, 4, 8)
            k = self.linear(normalized, f"layers.{index}.key").reshape(
                self.batch_size, 4, 8)
            v = self.linear(normalized, f"layers.{index}.value").reshape(
                self.batch_size, 4, 8)
            current_k = k.unsqueeze(2)
            current_v = v.unsqueeze(2)
            if self.keys[index] is None:
                keys = current_k
                values = current_v
            else:
                keys = torch.cat((current_k, self.keys[index]), dim=2)[:, :, :64]
                values = torch.cat((current_v, self.values[index]), dim=2)[:, :, :64]
            self.keys[index] = keys
            self.values[index] = values
            dots = torch.einsum("bhd,bhsd->bhs", q, keys)
            ages = torch.arange(keys.shape[2], device=self.device)
            bias = self.biases[index][:, ages]
            scores = (torch.trunc(dots / 512) + bias.unsqueeze(0)).clamp(-6, 6)
            weights = torch.pow(2.0, scores + 6)
            numerator = torch.einsum("bhs,bhsd->bhd", weights, values)
            attention = torch.trunc(numerator / weights.sum(dim=2, keepdim=True))
            attention = attention.reshape(self.batch_size, 32)
            projected = self.linear(attention, f"layers.{index}.output")
            residual = x + projected
            up = self.linear(maxnorm_qat(residual), f"layers.{index}.ffn_up")
            down = self.linear(torch.relu(up), f"layers.{index}.ffn_down")
            x = residual + down
        return self.linear(x, "lm_head")


def generate_incremental_integer(model, prompts: list[str], count: int,
                                 *, device: str = "cpu") -> list[str]:
    """Generate an equal-length prompt batch with live recurrent K/V caches."""

    import torch

    if not prompts or len({len(prompt) for prompt in prompts}) != 1:
        raise ValueError("batched prompts must be nonempty and have equal length")
    vocabulary = Vocabulary()
    encoded = [vocabulary.encode(prompt) for prompt in prompts]
    recurrent = IncrementalIntegerBatch(model, len(prompts), device)
    logits = None
    with torch.no_grad():
        for position in range(len(encoded[0])):
            logits = recurrent.step([tokens[position] for tokens in encoded])
        assert logits is not None
        generated: list[list[int]] = [[] for _ in prompts]
        tokens = logits.argmax(dim=-1)
        for step in range(count):
            values = tokens.to(torch.int64).cpu().tolist()
            for row, value in zip(generated, values, strict=True):
                row.append(value)
            if step + 1 < count:
                logits = recurrent.step(tokens)
                tokens = logits.argmax(dim=-1)
    return [vocabulary.decode(row) for row in generated]


def generate_incremental_scored(model, prompts: list[str], count: int,
                                *, device: str = "cpu") -> list[dict[str, Any]]:
    """Generate a prompt batch and retain exact greedy confidence margins."""

    import torch

    if not prompts or len({len(prompt) for prompt in prompts}) != 1:
        raise ValueError("batched prompts must be nonempty and have equal length")
    vocabulary = Vocabulary()
    encoded = [vocabulary.encode(prompt) for prompt in prompts]
    recurrent = IncrementalIntegerBatch(model, len(prompts), device)
    logits = None
    with torch.no_grad():
        for position in range(len(encoded[0])):
            logits = recurrent.step([tokens[position] for tokens in encoded])
        assert logits is not None
        generated: list[list[int]] = [[] for _ in prompts]
        margins: list[list[int]] = [[] for _ in prompts]
        for step in range(count):
            tokens, margin_tensor = factorio_greedy_tokens_and_margins(logits)
            step_margins = margin_tensor.to(torch.int64).cpu().tolist()
            values = tokens.to(torch.int64).cpu().tolist()
            for row, row_margins, value, margin in zip(
                    generated, margins, values, step_margins, strict=True):
                row.append(value)
                row_margins.append(margin)
            if step + 1 < count:
                logits = recurrent.step(tokens)
    return [
        {
            "generated": vocabulary.decode(row),
            "mean_argmax_margin": sum(row_margins) / len(row_margins),
            "minimum_argmax_margin": min(row_margins),
        }
        for row, row_margins in zip(generated, margins, strict=True)
    ]


def generate_incremental_scored_ragged(
    model,
    prompts: list[str],
    count: int,
    *,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Generate mixed-length prompt batches without padding any K/V history.

    Every row advances once per recurrent step.  Shorter prompts begin their
    greedy continuation while longer rows are still consuming prompt tokens;
    rows never observe one another, so this is exactly equivalent to grouping
    by prompt length but makes large offline prompt searches much faster.
    """

    import torch

    if not prompts or count < 1 or any(not prompt for prompt in prompts):
        raise ValueError("prompts must be nonempty and count must be positive")
    vocabulary = Vocabulary()
    encoded = [vocabulary.encode(prompt) for prompt in prompts]
    recurrent = IncrementalIntegerBatch(model, len(prompts), device)
    generated: list[list[int]] = [[] for _ in prompts]
    margins: list[list[int]] = [[] for _ in prompts]
    logits = None
    with torch.no_grad():
        for tick in range(max(map(len, encoded)) + count):
            if logits is None:
                choices = margin_tensor = None
            else:
                choices, margin_tensor = factorio_greedy_tokens_and_margins(logits)
            step_tokens: list[int] = []
            for row, prompt_tokens in enumerate(encoded):
                if tick < len(prompt_tokens):
                    step_tokens.append(prompt_tokens[tick])
                elif len(generated[row]) < count:
                    assert choices is not None and margin_tensor is not None
                    token = int(choices[row])
                    generated[row].append(token)
                    margins[row].append(int(margin_tensor[row]))
                    step_tokens.append(token)
                else:
                    # This row is complete. Its subsequent state is ignored,
                    # but a token is required to keep the dense batch rectangular.
                    step_tokens.append(0)
            if all(len(row) == count for row in generated):
                break
            logits = recurrent.step(step_tokens)
    if any(len(row) != count for row in generated):
        raise AssertionError("ragged generation ended before every row completed")
    return [
        {
            "generated": vocabulary.decode(row),
            "mean_argmax_margin": sum(row_margins) / len(row_margins),
            "minimum_argmax_margin": min(row_margins),
        }
        for row, row_margins in zip(generated, margins, strict=True)
    ]


def factorio_greedy_tokens_and_margins(logits):
    """Return lowest-index greedy tokens and top-one/top-two logit margins."""

    import torch

    if logits.ndim != 2 or logits.shape[1] != 64:
        raise ValueError("logits must have shape batch x 64")
    top_values = torch.topk(logits, 2, dim=-1).values
    # torch.topk does not promise a lowest-index tie break.  Factorio's
    # tournament and ExactFactorioLM do, while torch.argmax returns the first
    # (therefore lowest) index of an equal maximum.
    tokens = logits.argmax(dim=-1)
    return tokens, top_values[:, 0] - top_values[:, 1]


def generate_prompt_set(model, prompts: tuple[str, ...] | list[str], count: int,
                        *, device: str = "cpu") -> dict[str, str]:
    """Batch a prompt set by length without inserting synthetic padding."""

    grouped: dict[int, list[str]] = {}
    for prompt in prompts:
        grouped.setdefault(len(prompt), []).append(prompt)
    result: dict[str, str] = {}
    for _length, group in sorted(grouped.items()):
        generated = generate_incremental_integer(model, group, count, device=device)
        result.update(zip(group, generated, strict=True))
    return result


def validate_incremental_integer(model, checkpoint: ExactCheckpoint,
                                 *, prompt: str = "Once upon a time",
                                 count: int = 96, device: str = "cpu") -> None:
    """Require recurrent torch generation to equal the independent emulator."""

    fast = generate_incremental_integer(model, [prompt], count, device=device)[0]
    exact = ExactFactorioLM(checkpoint).generate(prompt, count)[0]
    if fast != exact:
        first = next(index for index, pair in enumerate(zip(fast, exact, strict=True))
                     if pair[0] != pair[1])
        raise AssertionError(
            f"incremental exact forward differs at generated character {first}: "
            f"{fast[first]!r} != {exact[first]!r}")


def mask_summary(masks: dict[str, Any]) -> dict[str, Any]:
    """Return deterministic per-matrix and per-module nonzero accounting."""

    per_matrix = {name: int(mask.sum().item())
                  for name, mask in sorted(masks.items())}
    per_module: dict[str, int] = {
        "query": 0, "key": 0, "value": 0, "output": 0,
        "ffn_up": 0, "ffn_down": 0, "lm_head": 0,
    }
    for name, count in per_matrix.items():
        module = "lm_head" if name == "lm_head" else name.rsplit(".", 1)[1]
        per_module[module] += count
    digest = hashlib.sha256()
    for name in sorted(masks):
        digest.update(name.encode("utf-8"))
        digest.update(masks[name].detach().to("cpu").numpy().tobytes())
    return {
        "total_nonzeros": sum(per_matrix.values()),
        "per_matrix": per_matrix,
        "per_module": per_module,
        "mask_sha256": digest.hexdigest(),
    }


def module_sparsity_summary(masks: dict[str, Any]) -> dict[str, Any]:
    """Aggregate nonzeros and sparsity by hardware projection family."""

    totals = {
        "query": 0, "key": 0, "value": 0, "output": 0,
        "ffn_up": 0, "ffn_down": 0, "lm_head": 0,
    }
    nonzeros = dict(totals)
    for name, mask in masks.items():
        module = "lm_head" if name == "lm_head" else name.rsplit(".", 1)[1]
        totals[module] += mask.numel()
        nonzeros[module] += int(mask.sum().item())
    return {
        module: {
            "nonzeros": nonzeros[module],
            "total": total,
            "sparsity_percentage": 100.0 * (total - nonzeros[module]) / total,
        }
        for module, total in totals.items()
    }


def collect_importance_scores(
    sparse,
    corpus: TokenCorpus,
    *,
    batches: int = 24,
    batch_size: int = 32,
    device: str = "cpu",
    seed: int = 4104,
) -> dict[str, Any]:
    """Accumulate first-order ``abs(weight * gradient)`` pruning saliency.

    ``SparseFloatStudent`` uses a straight-through mask, so the
    loss supplies gradients for both present and absent connections.  This
    makes the resulting score useful for swaps as well as removals.
    """

    import torch

    model = copy.deepcopy(sparse).to(device).train()
    model.masks = {name: value.to(device) for name, value in model.masks.items()}
    result = {name: torch.zeros_like(_get(model.model, name), device="cpu")
              for name in learned_matrix_names()}
    generator = torch.Generator(device="cpu").manual_seed(seed + 600_000)
    mean_losses: list[float] = []
    for _ in range(batches):
        starts = torch.randint(0, len(corpus.tokens) - 65, (batch_size,),
                               generator=generator)
        values = corpus.batch(starts, 64, device)
        model.zero_grad(set_to_none=True)
        logits = model(values[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 64), values[:, 1:].reshape(-1))
        loss.backward()
        mean_losses.append(float(loss.detach().cpu()))
        for name in learned_matrix_names():
            weight = _get(model.model, name)
            if weight.grad is None:
                raise AssertionError(f"missing saliency gradient for {name}")
            result[name].add_((weight.detach() * weight.grad.detach()).abs().cpu())
    for name in result:
        result[name].div_(batches)
    return {
        "scores": result,
        "batches": batches,
        "batch_size": batch_size,
        "mean_cross_entropy": sum(mean_losses) / len(mean_losses),
    }


def importance_masks(
    model,
    scores: dict[str, Any],
    total_nonzeros: int,
    *,
    minimum_fraction: float = 0.25,
    maximum_fraction: float = 0.75,
) -> dict[str, Any]:
    """Allocate a global saliency budget with conservative matrix guardrails."""

    import torch

    names = learned_matrix_names()
    learned_total = sum(_get(model, name).numel() for name in names)
    if not 1 <= total_nonzeros <= learned_total:
        raise ValueError("nonzero target must be within learned-weight count")
    selected: dict[str, Any] = {}
    maximums: dict[str, int] = {}
    used = 0
    for name in names:
        size = _get(model, name).numel()
        minimum = min(size, math.floor(size * minimum_fraction))
        maximums[name] = max(minimum, math.ceil(size * maximum_fraction))
        indexes = torch.topk(scores[name].flatten(), minimum,
                             sorted=False).indices
        mask = torch.zeros(size, dtype=torch.bool)
        mask[indexes] = True
        selected[name] = mask
        used += minimum
    if used > total_nonzeros:
        raise ValueError("minimum fractions exceed nonzero target")

    # Fifty-one thousand learned weights is small enough that an explicit,
    # deterministically tie-broken global ranking is clearer than an opaque
    # per-matrix heuristic.
    candidates: list[tuple[float, int, int, str]] = []
    for order, name in enumerate(names):
        flat = scores[name].flatten().cpu()
        for index in (~selected[name]).nonzero(as_tuple=False).flatten().tolist():
            candidates.append((-float(flat[index]), order, index, name))
    candidates.sort()
    counts = {name: int(mask.sum().item()) for name, mask in selected.items()}
    for _negative_score, _order, index, name in candidates:
        if used == total_nonzeros:
            break
        if counts[name] >= maximums[name]:
            continue
        selected[name][index] = True
        counts[name] += 1
        used += 1
    if used != total_nonzeros:
        raise AssertionError("importance allocation could not fill budget")
    return {name: mask.reshape_as(_get(model, name)).to(_get(model, name).device)
            for name, mask in selected.items()}


def limited_importance_swap_masks(
    base_masks: dict[str, Any],
    scores: dict[str, Any],
    swaps: int,
    *,
    minimum_fraction: float = 0.40,
    maximum_fraction: float = 0.60,
) -> dict[str, Any]:
    """Make a bounded saliency swap while retaining most recovered topology."""

    masks = {name: value.detach().cpu().clone()
             for name, value in base_masks.items()}
    names = learned_matrix_names()
    counts = {name: int(masks[name].sum().item()) for name in names}
    minimums = {name: math.floor(masks[name].numel() * minimum_fraction)
                for name in names}
    maximums = {name: math.ceil(masks[name].numel() * maximum_fraction)
                for name in names}
    removals: list[tuple[float, int, int, str]] = []
    additions: list[tuple[float, int, int, str]] = []
    for order, name in enumerate(names):
        flat_mask = masks[name].flatten()
        flat_score = scores[name].flatten().cpu()
        for index in flat_mask.nonzero(as_tuple=False).flatten().tolist():
            removals.append((float(flat_score[index]), order, index, name))
        for index in (~flat_mask).nonzero(as_tuple=False).flatten().tolist():
            additions.append((-float(flat_score[index]), order, index, name))
    removals.sort()
    additions.sort()
    removed = 0
    for _score, _order, index, name in removals:
        if removed == swaps:
            break
        if counts[name] <= minimums[name]:
            continue
        masks[name].view(-1)[index] = False
        counts[name] -= 1
        removed += 1
    added = 0
    for _negative_score, _order, index, name in additions:
        if added == removed:
            break
        if counts[name] >= maximums[name]:
            continue
        masks[name].view(-1)[index] = True
        counts[name] += 1
        added += 1
    if removed != swaps or added != swaps:
        raise AssertionError("limited importance swap could not fill its budget")
    return {name: masks[name].to(base_masks[name].device) for name in names}


def screen_limited_swaps(
    *,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
    swap_counts: tuple[int, ...] = (128, 256, 512, 1024, 2048),
) -> dict[str, Any]:
    """Successive-halving first stage: cheap no-recovery mask screening."""

    import torch

    config = Run5Config.read(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _, _, sparse, _ = load_trained_models()
    saliency = torch.load(output / RUN6_IMPORTANCE.name, map_location="cpu",
                          weights_only=False)
    validation = TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin")
    rows: list[dict[str, Any]] = []
    for swaps in swap_counts:
        candidate = copy.deepcopy(sparse)
        candidate.masks = limited_importance_swap_masks(
            sparse.masks, saliency["scores"], swaps)
        candidate.quotas = {name: int(value.sum().item())
                            for name, value in candidate.masks.items()}
        metrics = evaluate_torch(
            candidate.eval(), validation, device=config.device, length=64,
            characters=config.evaluation_characters,
            batch_size=config.batch_size)
        rows.append({"swaps": swaps, "metrics": metrics,
                     "mask": mask_summary(candidate.masks)})
    report = {"stage": "unrecovered-screen", "rows": rows}
    _write_json(output / "limited-swap-screen.json", report)
    return report


def build_limited_swap_candidate(
    swaps: int,
    *,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
):
    """Build a stable saliency swap, initializing additions from the teacher."""

    import torch

    config = Run5Config.read(config_path)
    output = Path(output_dir)
    _, teacher, sparse, _ = load_trained_models()
    saliency = torch.load(output / RUN6_IMPORTANCE.name, map_location="cpu",
                          weights_only=False)
    candidate = copy.deepcopy(sparse)
    masks = limited_importance_swap_masks(sparse.masks, saliency["scores"], swaps)
    with torch.no_grad():
        for name, mask in masks.items():
            added = mask & ~sparse.masks[name]
            _get(candidate.model, name)[added] = _get(teacher, name)[added]
    candidate.masks = masks
    candidate.quotas = {name: int(value.sum().item())
                        for name, value in masks.items()}
    return config, teacher, candidate


def train_limited_swap_sparse(
    swaps: int,
    *,
    steps: int = 250,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    """Short recovery used by the focused mask-search successive halving."""

    config, teacher, candidate = build_limited_swap_candidate(
        swaps, config_path=config_path, output_dir=output_dir)
    search_config = replace(
        config, sparse_float_steps=steps,
        evaluation_interval=min(125, steps),
        warmup_steps=min(100, max(1, steps // 3)))
    output = Path(output_dir)
    checkpoint_path = output / f"factorio-lm-sparse-swap-{swaps}-{steps}.pt"
    training = train_student(
        candidate, teacher,
        TokenCorpus("generated/run4/tinystories-v2-gpt4-train.bin"),
        TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin"),
        search_config, stage="sparse", output_path=checkpoint_path,
        resume_path=resume_path)
    report = {
        "candidate": f"limited-saliency-swap-{swaps}",
        "swaps": swaps,
        "mask": mask_summary(candidate.masks),
        "configuration": asdict(search_config),
        "training": training,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(output / f"sparse-swap-{swaps}-{steps}-training.json", report)
    return report


def load_limited_swap_sparse(
    swaps: int,
    *,
    sparse_steps: int = 750,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
):
    """Load the fully recovered limited-swap sparse-float checkpoint."""

    import torch

    config, teacher, candidate = build_limited_swap_candidate(
        swaps, config_path=config_path, output_dir=output_dir)
    state = torch.load(
        Path(output_dir) / f"factorio-lm-sparse-swap-{swaps}-{sparse_steps}.pt",
        map_location=config.device, weights_only=False)
    candidate.load_state_dict(state["model"])
    candidate.masks = {name: value.to(config.device)
                       for name, value in state["masks"].items()}
    candidate.quotas = {name: int(value.sum().item())
                        for name, value in candidate.masks.items()}
    return config, teacher.eval(), candidate.eval()


def train_limited_swap_int4(
    swaps: int = 64,
    *,
    sparse_steps: int = 750,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train int4 on the best sensitivity-aware, fixed-budget topology."""

    config, teacher, sparse = load_limited_swap_sparse(
        swaps, sparse_steps=sparse_steps, config_path=config_path,
        output_dir=output_dir)
    model = IntegerQATStudent(
        sparse, config.weight_limit, config.maximum_scale_denominator)
    output = Path(output_dir)
    checkpoint_path = output / f"factorio-lm-int4-swap-{swaps}.pt"
    training = train_student(
        model, teacher,
        TokenCorpus("generated/run4/tinystories-v2-gpt4-train.bin"),
        TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin"),
        config, stage="integer", output_path=checkpoint_path,
        resume_path=resume_path)
    report = {
        "candidate": f"scaled-int4-limited-swap-{swaps}",
        "configuration": asdict(config),
        "mask": mask_summary(model.masks),
        "training": training,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(output / f"int4-swap-{swaps}-training.json", report)
    return report


def load_limited_swap_int4(
    swaps: int = 64,
    *,
    sparse_steps: int = 750,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
):
    """Reconstruct the trained limited-swap int4 candidate."""

    import torch

    config, teacher, sparse = load_limited_swap_sparse(
        swaps, sparse_steps=sparse_steps, config_path=config_path,
        output_dir=output_dir)
    model = IntegerQATStudent(
        sparse, config.weight_limit, config.maximum_scale_denominator)
    state = torch.load(
        Path(output_dir) / f"factorio-lm-int4-swap-{swaps}.pt",
        map_location=config.device, weights_only=False)
    model.load_state_dict(state["model"])
    model.masks = {name: value.to(config.device)
                   for name, value in state["masks"].items()}
    model.denominators = dict(state["denominators"])
    return config, teacher.eval(), sparse.eval(), model.eval()


def _validation_lexicon(minimum_count: int = 2) -> set[str]:
    vocabulary = Vocabulary()
    ids = Path("generated/run4/tinystories-v2-gpt4-validation.bin").read_bytes()
    text = vocabulary.decode(ids)
    counts: dict[str, int] = {}
    for word in re.findall(r"[A-Za-z]+", text.lower()):
        counts[word] = counts.get(word, 0) + 1
    return {word for word, count in counts.items() if count >= minimum_count}


def generation_quality_metrics(text: str, lexicon: set[str]) -> dict[str, Any]:
    """Simple deterministic lexical and degeneration diagnostics."""

    words = re.findall(r"[A-Za-z]+", text.lower())
    dictionary_words = sum(word in lexicon for word in words)
    adjacent_repeats = sum(a == b for a, b in zip(words, words[1:]))
    maximum_run = 0
    current_run = 0
    previous = None
    for character in text:
        if character == previous:
            current_run += 1
        else:
            previous = character
            current_run = 1
        maximum_run = max(maximum_run, current_run)

    def repeated_ngram_fraction(n: int) -> float:
        grams = [text[index:index + n] for index in range(max(0, len(text) - n + 1))]
        return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)

    return {
        "characters": len(text),
        "words": len(words),
        "dictionary_like_word_fraction": dictionary_words / max(1, len(words)),
        "unique_word_fraction": len(set(words)) / max(1, len(words)),
        "adjacent_repeated_word_fraction": adjacent_repeats / max(1, len(words) - 1),
        "maximum_repeated_character_run": maximum_run,
        "space_fraction": text.count(" ") / max(1, len(text)),
        "sentence_boundaries": text.count("\n"),
        "repeated_bigram_fraction": repeated_ngram_fraction(2),
        "repeated_trigram_fraction": repeated_ngram_fraction(3),
        "repeated_fourgram_fraction": repeated_ngram_fraction(4),
    }


def _heldout_prefix_candidates(limit: int, *, seed: int = 4104) -> list[dict[str, Any]]:
    """Select deterministic, natural, incomplete line prefixes from held-out text."""

    text = Vocabulary().decode(
        Path("generated/run4/tinystories-v2-gpt4-validation.bin").read_bytes())
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_index, raw_line in enumerate(text.splitlines()):
        line = raw_line.strip()
        if len(line) < 16 or "endoftext" in line.lower():
            continue
        for match in re.finditer(r" ", line):
            end = match.end()
            if not 15 <= end <= 55:
                continue
            prompt = line[:end]
            if prompt in seen or len(re.findall(r"[A-Za-z]+", prompt)) < 3:
                continue
            seen.add(prompt)
            pool.append({"prompt": prompt, "heldout_line_index": line_index,
                         "prefix_characters": len(prompt)})
    generator = random.Random(seed + 700_000)
    generator.shuffle(pool)
    if len(pool) < limit:
        raise ValueError(f"held-out corpus yielded only {len(pool)} prefixes")
    return pool[:limit]


def _prompt_score(quality: dict[str, Any], mean_margin: float) -> float:
    return (
        4.0 * quality["dictionary_like_word_fraction"]
        + 1.25 * quality["unique_word_fraction"]
        - 4.0 * quality["adjacent_repeated_word_fraction"]
        - 2.0 * quality["repeated_fourgram_fraction"]
        - 0.75 * max(0, quality["maximum_repeated_character_run"] - 2)
        - 1.5 * abs(quality["space_fraction"] - 0.17)
        + 0.15 * min(2, quality["sentence_boundaries"])
        + 0.00025 * min(mean_margin, 400.0)
    )


def mine_demo_prompts(
    *,
    candidates: int = 5_000,
    generation_length: int = 128,
    output_dir: str | Path = RUN6_OUTPUT,
) -> dict[str, Any]:
    """Rank natural held-out prefixes with the selected exact integer model."""

    config, _, _, model = load_int4_run5_mask()
    prefix_rows = _heldout_prefix_candidates(candidates, seed=config.seed)
    lexicon = _validation_lexicon()
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in prefix_rows:
        groups.setdefault(len(row["prompt"]), []).append(row)
    scored: list[dict[str, Any]] = []
    for length, rows in sorted(groups.items()):
        # Limit batch memory while retaining vectorized exact recurrent state.
        for first in range(0, len(rows), 256):
            chunk = rows[first:first + 256]
            generations = generate_incremental_scored(
                model, [row["prompt"] for row in chunk], generation_length,
                device=config.device)
            for row, generated in zip(chunk, generations, strict=True):
                quality = generation_quality_metrics(generated["generated"], lexicon)
                scored.append({
                    **row,
                    **generated,
                    "quality": quality,
                    "score": _prompt_score(
                        quality, generated["mean_argmax_margin"]),
                })
    scored.sort(key=lambda row: (-row["score"], row["prompt"]))

    # Re-rank the most promising inputs on the full 256-character
    # showcase horizon.  The top 20 are retained for transparent selection.
    finalists: list[dict[str, Any]] = []
    finalist_groups: dict[int, list[dict[str, Any]]] = {}
    for row in scored[:40]:
        finalist_groups.setdefault(len(row["prompt"]), []).append(row)
    for _length, rows in sorted(finalist_groups.items()):
        generations = generate_incremental_scored(
            model, [row["prompt"] for row in rows], 256,
            device=config.device)
        for row, generated in zip(rows, generations, strict=True):
            quality = generation_quality_metrics(generated["generated"], lexicon)
            finalists.append({
                **{key: value for key, value in row.items()
                   if key not in {"generated", "quality", "score",
                                  "mean_argmax_margin", "minimum_argmax_margin"}},
                **generated,
                "quality": quality,
                "score": _prompt_score(quality, generated["mean_argmax_margin"]),
            })
    finalists.sort(key=lambda row: (-row["score"], row["prompt"]))
    report = {
        "source": "deterministic natural prefixes from held-out TinyStories validation text",
        "candidate_count": candidates,
        "candidate_generation_characters": generation_length,
        "ranking": {
            "lexical": "held-out word-frequency lexicon (minimum count 2)",
            "features": [
                "dictionary-like word rate", "unique word rate", "whitespace",
                "sentence boundaries", "repeated word/character/four-gram penalties",
                "exact greedy argmax margin",
            ],
        },
        "top_20": finalists[:20],
        "initial_top_100": scored[:100],
    }
    _write_json(Path(output_dir) / "prompt-mining.json", report)
    return report


def finalize_demo_prompt(
    prompt: str = "He loved to play with it every ",
    *,
    output_dir: str | Path = RUN6_OUTPUT,
) -> dict[str, Any]:
    """Combine 128/256 scores and record a demo prompt from the finalists."""

    output = Path(output_dir)
    report = json.loads((output / "prompt-mining.json").read_text(encoding="utf-8"))
    config, _, _, model = load_int4_run5_mask()
    lexicon = _validation_lexicon()
    rows = report["initial_top_100"][:40]
    groups: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(len(row["prompt"]), []).append(row)
    finalists: list[dict[str, Any]] = []
    for _length, group in sorted(groups.items()):
        generations = generate_incremental_scored(
            model, [row["prompt"] for row in group], 256,
            device=config.device)
        for row, generated in zip(group, generations, strict=True):
            quality = generation_quality_metrics(generated["generated"], lexicon)
            long_score = _prompt_score(quality, generated["mean_argmax_margin"])
            early_score = float(row["score"])
            finalists.append({
                "prompt": row["prompt"],
                "heldout_line_index": row["heldout_line_index"],
                "prefix_characters": row["prefix_characters"],
                **generated,
                "quality": quality,
                "score_128": early_score,
                "score_256": long_score,
                "score": 0.6 * early_score + 0.4 * long_score,
            })
    finalists.sort(key=lambda row: (-row["score"], row["prompt"]))
    report["top_20"] = finalists[:20]
    report["ranking"]["final_score"] = "60% 128-character score + 40% 256-character score"
    selected = next((row for row in finalists if row["prompt"] == prompt), None)
    if selected is None:
        raise ValueError("selected demo prompt was not a top-40 mined finalist")
    report["selected_demo"] = {
        **selected,
        "selection_reason": (
            f"Selected from {len(finalists)} held-out finalists; "
            f"prompt length {len(prompt)}, score {selected['score']:.4f}."
        ),
    }
    _write_json(output / "prompt-mining.json", report)
    return report


def _compare_general_quality(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    baseline_quality: dict[str, dict[str, Any]],
    candidate_quality: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compare validation metrics and average word quality across sample prompts."""

    def average(rows: dict[str, dict[str, Any]], key: str) -> float:
        return sum(row[key] for row in rows.values()) / len(rows)

    evidence = {
        "exact_accuracy_absolute_gain": (
            candidate_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "raw_cross_entropy_reduction": (
            baseline_metrics["cross_entropy"] - candidate_metrics["cross_entropy"]),
        "dictionary_word_rate_gain": (
            average(candidate_quality, "dictionary_like_word_fraction")
            - average(baseline_quality, "dictionary_like_word_fraction")),
        "adjacent_word_repetition_reduction": (
            average(baseline_quality, "adjacent_repeated_word_fraction")
            - average(candidate_quality, "adjacent_repeated_word_fraction")),
    }
    return {
        "run5_generation_quality": baseline_quality,
        "run6_generation_quality": candidate_quality,
        "clearly_better_than_run5": (
            evidence["exact_accuracy_absolute_gain"] > 0
            and evidence["raw_cross_entropy_reduction"] > 0
            and evidence["dictionary_word_rate_gain"] >= 0
            and evidence["adjacent_word_repetition_reduction"] >= 0
        ),
        "evidence": evidence,
    }


def finalize_run6_release(
    *,
    output_dir: str | Path = RUN6_OUTPUT,
) -> dict[str, Any]:
    """Export the selected model and its evaluation reports."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config, teacher, sparse, model = load_int4_run5_mask()
    candidate_report = json.loads(
        (output / "candidate-evaluation.json").read_text(encoding="utf-8"))
    prompt_report = json.loads(
        (output / "prompt-mining.json").read_text(encoding="utf-8"))
    demo = prompt_report["selected_demo"]
    checkpoint = export_integer_student(
        model, config,
        {
            "run": 6,
            "candidate": "scaled-int4-run5-mask",
            "selection": "best general exact prose after focused mask/QAT search",
            "demo_prompt": demo["prompt"],
        })
    quality_path = output / "factorio-lm-quality-exact.json"
    general_path = output / "factorio-lm-general-exact.json"
    checkpoint.write(quality_path)
    checkpoint.write(general_path)

    validation = TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin")
    validation_bytes = Path(
        "generated/run4/tinystories-v2-gpt4-validation.bin").read_bytes()
    independent_1024 = evaluate_exact(checkpoint, validation_bytes, characters=1024)
    ranges = collect_range_report(model, validation, device=config.device)
    _write_json(output / "range-report.json", ranges)

    selected = candidate_report["candidates"]["run5-mask"]
    run5_evaluation = json.loads(
        Path("generated/run5/evaluation-report.json").read_text(encoding="utf-8"))
    run5_exact = run5_evaluation["metrics"]["scaled_int3_exact_forward"]
    run5_calibrated = run5_evaluation["best_calibrated_cross_entropy"]
    _, _, _, run5_model = load_trained_models()
    run5_generations = generate_prompt_set(
        run5_model, GENERAL_PROMPTS, 256, device=config.device)
    lexicon = _validation_lexicon()
    run5_generation_quality = {
        prompt: generation_quality_metrics(text, lexicon)
        for prompt, text in run5_generations.items()
    }

    generations = {
        prompt: row["generated"] for prompt, row in selected["samples"].items()
    }
    (output / "generation-samples.txt").write_text(
        "".join(
            f"PROMPT: {prompt!r}\nEXACT (256 chars):\n{text}\n\n"
            for prompt, text in generations.items()),
        encoding="utf-8")

    search_rows = {
        "A_run5_scaled_int3": {
            "metrics": run5_exact,
            "best_calibrated_cross_entropy": run5_calibrated,
            "nonzeros": 25_100,
        },
        "B_scaled_int4_run5_mask": candidate_report["candidates"]["run5-mask"],
        "C_scaled_int4_improved_mask": candidate_report["candidates"]["swap-64"],
        "D_scaled_int4_refined": candidate_report["candidates"]["refined"],
    }
    quality_comparison = _compare_general_quality(
        run5_exact, selected["metrics"], run5_generation_quality,
        {prompt: row["quality"] for prompt, row in selected["samples"].items()},
    )
    accuracy_gain = quality_comparison["evidence"]["exact_accuracy_absolute_gain"]
    evaluation = {
        "selected_candidate": "B_scaled_int4_run5_mask",
        "selection_reason": (
            "Run 5 mask with int4 weights; exact accuracy change from Run 5: "
            f"{100 * accuracy_gain:+.2f} percentage points."
        ),
        "serious_candidates": search_rows,
        "selected_metrics": selected["metrics"],
        "selected_best_temperature": selected["best_temperature"],
        "selected_best_calibrated_cross_entropy": selected["best_calibrated_cross_entropy"],
        "selected_independent_exact_int32_1024": independent_1024,
        "selected_sparsity": _aggregate_sparsity(checkpoint),
        "selected_mask": mask_summary(model.masks),
        "general_quality_gate": quality_comparison,
        "range_report": str(output / "range-report.json"),
        "prompt_mining": str(output / "prompt-mining.json"),
        "specialization_used": False,
        "specialization_reason": "No specialized checkpoint was used.",
    }
    _write_json(output / "evaluation-report.json", evaluation)

    training_artifacts = [
        "int4-run5-mask-training.json", "importance-sparse-25100-training.json",
        "sparse-swap-32-250-training.json", "sparse-swap-64-250-training.json",
        "sparse-swap-128-250-training.json", "sparse-swap-64-750-training.json",
        "int4-swap-64-training.json", "int4-refined-training.json",
    ]
    elapsed = 0.0
    for name in training_artifacts:
        path = output / name
        if path.exists():
            elapsed += float(json.loads(path.read_text(encoding="utf-8"))
                             ["training"]["elapsed_seconds"])
    training_report = {
        "run": 6,
        "source_float_checkpoint": "generated/run4/factorio-lm-quality-float.pt",
        "source_sparse_checkpoint": "generated/run5/factorio-lm-sparse-float.pt",
        "selected_qat_checkpoint": str(RUN6_INT4_RUN5_MASK),
        "winning_exact_checkpoint": str(quality_path),
        "dataset": json.loads(Path("generated/run4/data-report.json").read_text(
            encoding="utf-8")),
        "focused_search": training_artifacts,
        "measured_optimizer_elapsed_seconds": elapsed,
        "selected_training": json.loads(
            (output / "int4-run5-mask-training.json").read_text(encoding="utf-8")),
        "refinement_rejected_for_repetition": True,
        "mask_search": {
            "winner_changed_mask": False,
            "best_sparse_float_candidate": "64 sensitivity-aware swaps",
            "best_sparse_float_metrics": json.loads(
                (output / "sparse-swap-64-750-training.json").read_text(encoding="utf-8"))
                ["training"]["metrics"],
            "integer_result": "did not beat the original mask on calibrated CE or prose",
        },
    }
    _write_json(output / "training-report.json", training_report)

    return {"checkpoint": checkpoint, "evaluation": evaluation, "demo": demo}


def export_and_evaluate_candidates(
    *,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
) -> dict[str, Any]:
    """Export both serious int4 finalists and evaluate exact integer behavior."""

    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config, _, _, baseline = load_int4_run5_mask(config_path=config_path)
    _, _, _, swapped = load_limited_swap_int4(
        config_path=config_path, output_dir=output)
    _, _, _, refined = load_refined_int4(output_dir=output)
    validation = TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin")
    validation_bytes = Path(
        "generated/run4/tinystories-v2-gpt4-validation.bin").read_bytes()
    lexicon = _validation_lexicon()
    rows: dict[str, Any] = {}
    for name, model in (("run5-mask", baseline), ("swap-64", swapped),
                        ("refined", refined)):
        checkpoint = export_integer_student(
            model, config,
            {"run": 6, "candidate": name, "training_representation": "scaled-int4"})
        checkpoint_path = output / f"candidate-{name}-exact.json"
        checkpoint.write(checkpoint_path)
        common = dict(device=config.device, length=64,
                      characters=config.evaluation_characters,
                      batch_size=config.batch_size)
        metrics = evaluate_torch(model, validation, **common)
        calibration = {
            str(temperature): evaluate_torch(
                model, validation, **common,
                optimization_temperature=temperature,
            )["optimization_cross_entropy"]
            for temperature in (0.75, 1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 2.0)
        }
        best_temperature = min(calibration, key=calibration.get)
        independent = evaluate_exact(checkpoint, validation_bytes, characters=128)

        tokens = list(validation_bytes[:64])
        with torch.no_grad():
            torch_logits = model(torch.tensor([tokens], device=config.device)).round().to(
                torch.int64).cpu().tolist()[0]
        exact_model = ExactFactorioLM(checkpoint)
        exact_logits = [exact_model.step(token).logits for token in tokens]
        if torch_logits != exact_logits:
            raise AssertionError(f"{name} QAT forward differs from exact int32 emulator")
        validate_incremental_integer(
            model, checkpoint, count=96, device=config.device)

        samples: dict[str, Any] = {}
        generated_set = generate_prompt_set(
            model, GENERAL_PROMPTS, 256, device=config.device)
        for prompt, generated in generated_set.items():
            samples[prompt] = {
                "generated": generated,
                "quality": generation_quality_metrics(generated, lexicon),
            }
        rows[name] = {
            "checkpoint": str(checkpoint_path),
            "metrics": metrics,
            "calibration_cross_entropy": calibration,
            "best_temperature": float(best_temperature),
            "best_calibrated_cross_entropy": calibration[best_temperature],
            "independent_exact_int32_128": independent,
            "exact_forward_matches_independent_int32": True,
            "incremental_forward_matches_independent_int32_96_characters": True,
            "sparsity": _aggregate_sparsity(checkpoint),
            "mask": mask_summary(model.masks),
            "samples": samples,
        }
    report = {"candidates": rows}
    _write_json(output / "candidate-evaluation.json", report)
    return report


def train_int4_run5_mask(
    *,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train the mandatory scaled-int4 baseline from recovered sparse float."""

    config = Run5Config.read(config_path)
    if config.weight_limit != 7:
        raise ValueError("Run 6 scaled-int4 requires weight_limit=7")
    _, teacher, sparse, _ = load_trained_models()
    model = IntegerQATStudent(
        sparse, config.weight_limit, config.maximum_scale_denominator)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / RUN6_INT4_RUN5_MASK.name
    training = train_student(
        model,
        teacher,
        TokenCorpus("generated/run4/tinystories-v2-gpt4-train.bin"),
        TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin"),
        config,
        stage="integer",
        output_path=checkpoint_path,
        resume_path=resume_path,
    )
    report = {
        "candidate": "scaled-int4-run5-mask",
        "configuration": asdict(config),
        "training": training,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(output / "int4-run5-mask-training.json", report)
    return report


def load_int4_run5_mask(
    *,
    config_path: str | Path = RUN6_CONFIG,
    checkpoint_path: str | Path = RUN6_INT4_RUN5_MASK,
):
    """Reconstruct the trained Run 6 baseline without mutating Run 5 files."""

    import torch

    config = Run5Config.read(config_path)
    _, teacher, sparse, _ = load_trained_models()
    model = IntegerQATStudent(
        sparse, config.weight_limit, config.maximum_scale_denominator)
    state = torch.load(checkpoint_path, map_location=config.device,
                       weights_only=False)
    model.load_state_dict(state["model"])
    model.masks = {name: value.to(config.device)
                   for name, value in state["masks"].items()}
    model.denominators = dict(state["denominators"])
    return config, teacher.eval(), sparse.eval(), model.eval()


def train_refined_int4(
    *,
    config_path: str | Path = RUN6_REFINE_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    """Targeted low-LR QAT/distillation refinement of the winning topology."""

    config = Run5Config.read(config_path)
    _, teacher, sparse, model = load_int4_run5_mask()
    output = Path(output_dir)
    checkpoint_path = output / "factorio-lm-int4-refined.pt"
    training = train_student(
        model, teacher,
        TokenCorpus("generated/run4/tinystories-v2-gpt4-train.bin"),
        TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin"),
        config, stage="integer", output_path=checkpoint_path,
        resume_path=resume_path)
    report = {
        "candidate": "scaled-int4-refined",
        "initialization": str(RUN6_INT4_RUN5_MASK),
        "configuration": asdict(config),
        "mask": mask_summary(model.masks),
        "training": training,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(output / "int4-refined-training.json", report)
    return report


def load_refined_int4(
    *,
    config_path: str | Path = RUN6_REFINE_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
):
    """Load the targeted Run 6 QAT refinement."""

    import torch

    config = Run5Config.read(config_path)
    _, teacher, sparse, model = load_int4_run5_mask()
    state = torch.load(Path(output_dir) / "factorio-lm-int4-refined.pt",
                       map_location=config.device, weights_only=False)
    model.load_state_dict(state["model"])
    model.masks = {name: value.to(config.device)
                   for name, value in state["masks"].items()}
    model.denominators = dict(state["denominators"])
    return config, teacher.eval(), sparse.eval(), model.eval()


def evaluate_int4_run5_mask(
    *,
    config_path: str | Path = RUN6_CONFIG,
    checkpoint_path: str | Path = RUN6_INT4_RUN5_MASK,
) -> dict[str, Any]:
    """Evaluate the literal integer forward and its deterministic calibration."""

    config, _, _, model = load_int4_run5_mask(
        config_path=config_path, checkpoint_path=checkpoint_path)
    validation = TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin")
    common = dict(device=config.device, length=64,
                  characters=config.evaluation_characters,
                  batch_size=config.batch_size)
    raw = evaluate_torch(model, validation, **common)
    calibration = {
        str(temperature): evaluate_torch(
            model, validation, **common,
            optimization_temperature=temperature,
        )["optimization_cross_entropy"]
        for temperature in (0.75, 1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 2.0)
    }
    best_temperature = min(calibration, key=calibration.get)
    return {
        "candidate": "scaled-int4-run5-mask",
        "metrics": raw,
        "calibration_cross_entropy": calibration,
        "best_temperature": float(best_temperature),
        "best_calibrated_cross_entropy": calibration[best_temperature],
        "denominators": dict(model.denominators),
    }


def build_importance_candidate(
    *,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
    total_nonzeros: int | None = None,
) -> dict[str, Any]:
    """Score connections and persist a deterministic improved-mask candidate."""

    import torch

    config = Run5Config.read(config_path)
    total_nonzeros = total_nonzeros or config.nonzero_weights
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _, _, sparse, _ = load_trained_models()
    score_path = output / RUN6_IMPORTANCE.name
    if score_path.exists():
        saliency = torch.load(score_path, map_location="cpu", weights_only=False)
    else:
        saliency = collect_importance_scores(
            sparse,
            TokenCorpus("generated/run4/tinystories-v2-gpt4-train.bin"),
            device=config.device,
            seed=config.seed,
        )
        torch.save(saliency, score_path)
    masks = importance_masks(sparse.model, saliency["scores"], total_nonzeros)
    candidate = copy.deepcopy(sparse)
    candidate.masks = masks
    candidate.quotas = {name: int(mask.sum().item())
                        for name, mask in masks.items()}
    validation = TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin")
    metrics = evaluate_torch(
        candidate.eval(), validation, device=config.device, length=64,
        characters=config.evaluation_characters, batch_size=config.batch_size)
    summary = mask_summary(masks)
    report = {
        "method": "global-abs-weight-times-gradient-with-25-to-75-percent-guards",
        "saliency_batches": saliency["batches"],
        "saliency_batch_size": saliency["batch_size"],
        "saliency_mean_cross_entropy": saliency["mean_cross_entropy"],
        "target_metrics_before_recovery": metrics,
        "mask": summary,
        "run5_mask": mask_summary(sparse.masks),
        "score_checkpoint": str(score_path),
    }
    _write_json(output / f"importance-mask-{total_nonzeros}.json", report)
    return {"candidate": candidate, "report": report}


def train_importance_sparse(
    *,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
    total_nonzeros: int | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recover the improved topology in float before integer QAT."""

    config = Run5Config.read(config_path)
    total_nonzeros = total_nonzeros or config.nonzero_weights
    output = Path(output_dir)
    built = build_importance_candidate(
        config_path=config_path, output_dir=output, total_nonzeros=total_nonzeros)
    _, teacher, _, _ = load_trained_models()
    checkpoint_path = output / f"factorio-lm-sparse-importance-{total_nonzeros}.pt"
    training = train_student(
        built["candidate"], teacher,
        TokenCorpus("generated/run4/tinystories-v2-gpt4-train.bin"),
        TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin"),
        config, stage="sparse", output_path=checkpoint_path,
        resume_path=resume_path)
    report = {
        "candidate": f"importance-sparse-{total_nonzeros}",
        "configuration": asdict(config),
        "allocation": built["report"],
        "training": training,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(output / f"importance-sparse-{total_nonzeros}-training.json", report)
    return report


def load_importance_sparse(
    *,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
    total_nonzeros: int | None = None,
):
    """Reconstruct a recovered importance allocation for int4 initialization."""

    import torch

    config = Run5Config.read(config_path)
    total_nonzeros = total_nonzeros or config.nonzero_weights
    _, teacher, sparse, _ = load_trained_models()
    state = torch.load(
        Path(output_dir) / f"factorio-lm-sparse-importance-{total_nonzeros}.pt",
        map_location=config.device, weights_only=False)
    sparse.load_state_dict(state["model"])
    sparse.masks = {name: value.to(config.device)
                    for name, value in state["masks"].items()}
    sparse.quotas = {name: int(value.sum().item())
                     for name, value in sparse.masks.items()}
    return config, teacher.eval(), sparse.eval()


def train_importance_int4(
    *,
    config_path: str | Path = RUN6_CONFIG,
    output_dir: str | Path = RUN6_OUTPUT,
    total_nonzeros: int | None = None,
    resume_path: str | Path | None = None,
) -> dict[str, Any]:
    """Train deployment int4 on the recovered importance allocation."""

    config, teacher, sparse = load_importance_sparse(
        config_path=config_path, output_dir=output_dir,
        total_nonzeros=total_nonzeros)
    total_nonzeros = total_nonzeros or config.nonzero_weights
    model = IntegerQATStudent(
        sparse, config.weight_limit, config.maximum_scale_denominator)
    output = Path(output_dir)
    checkpoint_path = output / f"factorio-lm-int4-importance-{total_nonzeros}.pt"
    training = train_student(
        model, teacher,
        TokenCorpus("generated/run4/tinystories-v2-gpt4-train.bin"),
        TokenCorpus("generated/run4/tinystories-v2-gpt4-validation.bin"),
        config, stage="integer", output_path=checkpoint_path,
        resume_path=resume_path)
    report = {
        "candidate": f"scaled-int4-importance-{total_nonzeros}",
        "configuration": asdict(config),
        "mask": mask_summary(model.masks),
        "training": training,
        "checkpoint": str(checkpoint_path),
    }
    _write_json(output / f"int4-importance-{total_nonzeros}-training.json", report)
    return report
