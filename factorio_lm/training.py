"""Quantization-aware training. PyTorch is only needed for training, not inference."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .checkpoint import ExactCheckpoint, SCORE_DIVISORS
from .vocabulary import Vocabulary


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised by CLI failure path
        raise RuntimeError("training requires PyTorch; install factorio-lm[train]") from exc
    return torch


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 7321
    sequence_length: int = 24
    batch_size: int = 4
    candidate_steps: int = 16
    final_steps: int = 96
    learning_rate: float = 0.012
    gradient_clip: float = 1.0
    ternary_threshold: float = 0.55
    embedding_limit: int = 31
    relative_bias_limit: int = 6
    validation_fraction: float = 0.15
    sample_prompt: str = "Factorio "
    sample_length: int = 24
    device: str = "auto"

    @classmethod
    def read(cls, path: str | Path) -> "TrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**value)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)


def select_device(requested: str = "auto") -> str:
    torch = _torch()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_corpus(path: str | Path, vocabulary: Vocabulary | None = None) -> tuple[list[int], list[int], dict[str, Any]]:
    vocabulary = vocabulary or Vocabulary()
    text = Path(path).read_text(encoding="utf-8")
    if len(text) < 128:
        raise ValueError("training corpus must contain at least 128 characters")
    tokens = vocabulary.encode(text)
    split = max(2, min(len(tokens) - 2, int(len(tokens) * 0.85)))
    return tokens[:split], tokens[split:], {
        "path": str(path), "characters": len(text), "tokens": len(tokens),
        "train_tokens": split, "validation_tokens": len(tokens) - split,
        "unsupported_characters_mapped_to_space": sum(character not in vocabulary.alphabet for character in text),
    }


def _ste_round(value):
    rounded = value.round()
    return value + (rounded - value).detach()


def _ste_trunc(value):
    truncated = value.trunc()
    return value + (truncated - value).detach()


def quantize_ternary(value, threshold: float):
    torch = _torch()
    quantized = torch.where(value > threshold, torch.ones_like(value),
                            torch.where(value < -threshold, -torch.ones_like(value), torch.zeros_like(value)))
    return value + (quantized - value).detach()


def quantize_integer(value, limit: int):
    return value + (value.round().clamp(-limit, limit) - value).detach()


def maxnorm_qat(value):
    torch = _torch()
    maximum = value.abs().amax(dim=-1, keepdim=True).clamp(min=1)
    # Forward exactly matches trunc(31*x/max); backward treats truncation as identity.
    return _ste_trunc(31.0 * value / maximum)


class QATLinear:
    """Namespace for serializable latent matrix helpers."""

    @staticmethod
    def create(torch, rows: int, columns: int, generator):
        return torch.nn.Parameter(torch.randn(rows, columns, generator=generator) * 0.82)


def _build_model_class():
    torch = _torch()

    class TorchQATBlock(torch.nn.Module):
        def __init__(self, generator, threshold: float, bias_limit: int) -> None:
            super().__init__()
            self.threshold = threshold
            self.bias_limit = bias_limit
            self.query = QATLinear.create(torch, 32, 32, generator)
            self.key = QATLinear.create(torch, 32, 32, generator)
            self.value = QATLinear.create(torch, 32, 32, generator)
            self.output = QATLinear.create(torch, 32, 32, generator)
            self.ffn_up = QATLinear.create(torch, 64, 32, generator)
            self.ffn_down = QATLinear.create(torch, 32, 64, generator)
            initial_bias = torch.zeros(4, 64)
            for head in range(4):
                for age in range(64):
                    initial_bias[head, age] = max(-2, 1 - age // (9 + head))
            self.relative_bias = torch.nn.Parameter(initial_bias)

        def matrix(self, value):
            return quantize_ternary(value, self.threshold)

        def forward(self, x, score_divisor: int, statistics: dict[str, list] | None = None):
            q = maxnorm_qat(x) @ self.matrix(self.query).T
            k = maxnorm_qat(x) @ self.matrix(self.key).T
            v = maxnorm_qat(x) @ self.matrix(self.value).T
            bias = quantize_integer(self.relative_bias, self.bias_limit)
            # All positions are evaluated together.  This is algebraically the
            # same causal computation as the old Python position loop, but it
            # makes 64-character training windows practical on CPU and MPS.
            batch, length, _ = q.shape
            qh = q.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
            kh = k.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
            vh = v.reshape(batch, length, 4, 8).permute(0, 2, 1, 3)
            dots = torch.einsum("bhtd,bhsd->bhts", qh, kh)
            positions = torch.arange(length, device=x.device)
            ages = positions[:, None] - positions[None, :]
            causal = (ages >= 0) & (ages < 64)
            relative = bias[:, ages.clamp(0, 63)]
            biased = _ste_trunc(dots / score_divisor) + relative.unsqueeze(0)
            scores = biased.clamp(-6, 6)
            weights = torch.pow(2.0, scores + 6) * causal[None, None]
            numerator = torch.einsum("bhts,bhsd->bhtd", weights, vh)
            denominator = weights.sum(dim=-1, keepdim=True)
            attention = _ste_trunc(numerator / denominator).permute(0, 2, 1, 3).reshape(batch, length, 32)
            if statistics is not None:
                statistics.setdefault("biased_scores", []).append(biased.detach().cpu())
                statistics.setdefault("scores", []).append(scores.detach().cpu())
                statistics.setdefault("weights", []).append(weights.detach().cpu())
            residual1 = x + attention @ self.matrix(self.output).T
            up = maxnorm_qat(residual1) @ self.matrix(self.ffn_up).T
            return residual1 + torch.relu(up) @ self.matrix(self.ffn_down).T

    class TorchFactorioLM(torch.nn.Module):
        def __init__(self, *, seed: int, score_divisor: int, threshold: float,
                     embedding_limit: int, bias_limit: int) -> None:
            super().__init__()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            self.score_divisor = score_divisor
            self.threshold = threshold
            self.embedding_limit = embedding_limit
            self.bias_limit = bias_limit
            self.embeddings = torch.nn.Parameter(
                torch.empty(64, 32).uniform_(-8.0, 8.0, generator=generator)
            )
            self.layers = torch.nn.ModuleList([
                TorchQATBlock(generator, threshold, bias_limit) for _ in range(6)
            ])
            self.lm_head = QATLinear.create(torch, 64, 32, generator)

        def forward(self, token_ids, statistics: dict[str, list] | None = None):
            x = quantize_integer(self.embeddings, self.embedding_limit)[token_ids]
            for layer in self.layers:
                x = layer(x, self.score_divisor, statistics)
            return x @ quantize_ternary(self.lm_head, self.threshold).T

        def constrain_latents(self) -> None:
            with torch.no_grad():
                self.embeddings.clamp_(-self.embedding_limit, self.embedding_limit)
                self.lm_head.clamp_(-1.5, 1.5)
                for layer in self.layers:
                    layer.relative_bias.clamp_(-self.bias_limit, self.bias_limit)
                    for name in ("query", "key", "value", "output", "ffn_up", "ffn_down"):
                        getattr(layer, name).clamp_(-1.5, 1.5)

    return TorchFactorioLM


def create_model(config: TrainingConfig, score_divisor: int):
    if score_divisor not in SCORE_DIVISORS:
        raise ValueError(f"score divisor must be one of {SCORE_DIVISORS}")
    set_deterministic(config.seed)
    model_class = _build_model_class()
    return model_class(seed=config.seed, score_divisor=score_divisor,
                       threshold=config.ternary_threshold,
                       embedding_limit=config.embedding_limit,
                       bias_limit=config.relative_bias_limit)


def _batch(tokens: list[int], *, batch_size: int, length: int, generator: random.Random):
    torch = _torch()
    if len(tokens) < length + 2:
        repeated = (tokens * (math.ceil((length + 2) / len(tokens)) + 1))
        tokens = repeated
    starts = [generator.randrange(0, len(tokens) - length - 1) for _ in range(batch_size)]
    return torch.tensor([tokens[start:start + length + 1] for start in starts], dtype=torch.long)


def evaluate_loss(model, tokens: list[int], config: TrainingConfig, device: str) -> float:
    torch = _torch()
    model.eval()
    generator = random.Random(config.seed + 991)
    with torch.no_grad():
        values = _batch(tokens, batch_size=min(config.batch_size, 4),
                        length=min(config.sequence_length, max(2, len(tokens) - 2)), generator=generator).to(device)
        logits = model(values[:, :-1])
        return float(torch.nn.functional.cross_entropy(logits.reshape(-1, 64), values[:, 1:].reshape(-1)).cpu())


def train_model(model, train_tokens: list[int], validation_tokens: list[int], config: TrainingConfig,
                *, steps: int, device: str, resume_path: str | Path | None = None,
                latent_path: str | Path | None = None) -> dict[str, Any]:
    torch = _torch()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.0005)
    start_step = 0
    if resume_path:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = int(state["step"])
    generator = random.Random(config.seed + start_step)
    initial_train = evaluate_loss(model, train_tokens, config, device)
    losses: list[dict[str, float | int]] = []
    model.train()
    for step in range(start_step, start_step + steps):
        values = _batch(train_tokens, batch_size=config.batch_size,
                        length=config.sequence_length, generator=generator).to(device)
        logits = model(values[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 64), values[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip).detach().cpu())
        optimizer.step()
        model.constrain_latents()
        if step == start_step or (step + 1) % max(1, steps // 8) == 0 or step + 1 == start_step + steps:
            losses.append({"step": step + 1, "train_loss": float(loss.detach().cpu()),
                           "gradient_norm_before_clip": gradient_norm})
    final_train = evaluate_loss(model, train_tokens, config, device)
    validation = evaluate_loss(model, validation_tokens, config, device)
    if latent_path:
        target = Path(latent_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": start_step + steps, "config": asdict(config),
                    "score_divisor": model.score_divisor}, target)
    return {"start_step": start_step, "steps_this_run": steps,
            "completed_steps": start_step + steps, "initial_train_loss": initial_train,
            "final_train_loss": final_train, "validation_loss": validation,
            "loss_log": losses, "device": device}


def export_model(model, *, metadata: dict[str, Any] | None = None) -> ExactCheckpoint:
    from .block import BlockWeights
    torch = _torch()

    def integer_list(value, *, ternary: bool = False):
        with torch.no_grad():
            quantized = quantize_ternary(value, model.threshold) if ternary else quantize_integer(value, model.embedding_limit)
            return quantized.round().to(dtype=torch.int64).cpu().tolist()

    layers: list[BlockWeights] = []
    biases: list[list[list[int]]] = []
    for layer in model.layers:
        layers.append(BlockWeights(**{
            name: integer_list(getattr(layer, name), ternary=True)
            for name in ("query", "key", "value", "output", "ffn_up", "ffn_down")
        }))
        with torch.no_grad():
            biases.append(quantize_integer(layer.relative_bias, layer.bias_limit).round()
                          .to(dtype=torch.int64).cpu().tolist())
    checkpoint = ExactCheckpoint(
        embeddings=integer_list(model.embeddings),
        layers=layers,
        relative_biases=biases,
        lm_head=integer_list(model.lm_head, ternary=True),
        score_divisor=int(model.score_divisor),
        metadata=metadata,
    )
    checkpoint.validate()
    return checkpoint


def exact_attention_statistics(checkpoint: ExactCheckpoint, tokens: Iterable[int], *, limit: int = 96) -> dict[str, Any]:
    from collections import Counter
    from .full_model import ExactFactorioLM

    model = ExactFactorioLM(checkpoint)
    scores: Counter[int] = Counter()
    weights: Counter[int] = Counter()
    clamped_low = clamped_high = total = 0
    maximum = 0
    for token in list(tokens)[:limit]:
        trace = model.step(token)
        for layer in trace.layer_traces:
            for cell in layer.attention.cells:
                if not cell.valid:
                    continue
                for biased, score, weight in zip(cell.biased_scores, cell.scores, cell.weights, strict=True):
                    scores[score] += 1
                    weights[weight] += 1
                    clamped_low += biased < -6
                    clamped_high += biased > 6
                    total += 1
                maximum = max(maximum, *(abs(value) for value in layer.block_output),
                              *(abs(value) for value in layer.attention.numerators),
                              *(abs(value) for value in layer.attention.denominators))
    return {
        "valid_head_scores": total,
        "clamped_at_minus_6_percentage": 100.0 * clamped_low / max(total, 1),
        "clamped_at_plus_6_percentage": 100.0 * clamped_high / max(total, 1),
        "score_distribution": {str(key): scores[key] for key in sorted(scores)},
        "weight_distribution": {str(key): weights[key] for key in sorted(weights)},
        "maximum_absolute_observed_integer": maximum,
        "int32_range_safe": maximum < 2 ** 31,
    }
