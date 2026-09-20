"""Stable exact-discrete FactorioLM checkpoint format."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .attention import CONTEXT_DEPTH, HEADS, MODEL_WIDTH
from .block import BlockScales, BlockWeights, MatrixScale
from .int32 import INT32_MAX, INT32_MIN
from .vocabulary import FACTORIOLM_ALPHABET, Vocabulary


CHECKPOINT_FORMAT = "factorio-lm-exact"
CHECKPOINT_VERSION = 2
LAYERS = 6
FFN_WIDTH = 64
SCORE_DIVISORS = (16, 32, 64, 128, 256, 512)


def _integer_matrix(value: Any, rows: int, columns: int, name: str, *, ternary: bool) -> list[list[int]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must have shape {rows}x{columns}")
    result: list[list[int]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != columns:
            raise ValueError(f"{name} row {row_index} must have width {columns}")
        if any(isinstance(item, bool) or not isinstance(item, int) for item in row):
            raise ValueError(f"{name} must contain integers")
        if ternary and any(item not in {-1, 0, 1} for item in row):
            raise ValueError(f"{name} must be ternary")
        if any(not INT32_MIN <= item <= INT32_MAX for item in row):
            raise ValueError(f"{name} contains an out-of-int32 value")
        result.append(list(row))
    return result


@dataclass(frozen=True)
class ExactCheckpoint:
    embeddings: list[list[int]]
    layers: list[BlockWeights]
    relative_biases: list[list[list[int]]]
    lm_head: list[list[int]]
    score_divisor: int
    alphabet: str = FACTORIOLM_ALPHABET
    metadata: dict[str, Any] | None = None
    weight_limit: int = 1
    layer_scales: list[BlockScales] | None = None
    lm_head_scale: MatrixScale = MatrixScale()

    def normalized_layer_scales(self) -> list[BlockScales]:
        return list(self.layer_scales or [BlockScales() for _ in range(LAYERS)])

    def uses_version_2(self) -> bool:
        return self.weight_limit != 1 or any(
            scale != BlockScales() for scale in self.normalized_layer_scales()
        ) or self.lm_head_scale != MatrixScale()

    def validate(self) -> None:
        Vocabulary(self.alphabet)
        _integer_matrix(self.embeddings, 64, MODEL_WIDTH, "embeddings", ternary=False)
        if len(self.layers) != LAYERS:
            raise ValueError(f"checkpoint must contain exactly {LAYERS} layers")
        if isinstance(self.weight_limit, bool) or not 1 <= self.weight_limit <= 127:
            raise ValueError("weight_limit must be an integer in [1, 127]")
        for index, layer in enumerate(self.layers):
            try:
                layer.validate(self.weight_limit)
            except ValueError as exc:
                raise ValueError(f"invalid layer {index}: {exc}") from exc
        if len(self.relative_biases) != LAYERS:
            raise ValueError(f"checkpoint must contain {LAYERS} relative-bias tables")
        for index, table in enumerate(self.relative_biases):
            _integer_matrix(table, HEADS, CONTEXT_DEPTH, f"relative_biases[{index}]", ternary=False)
        _integer_matrix(self.lm_head, 64, MODEL_WIDTH, "lm_head",
                        ternary=self.weight_limit == 1)
        if any(abs(value) > self.weight_limit for row in self.lm_head for value in row):
            raise ValueError("lm_head exceeds configured weight_limit")
        scales = self.normalized_layer_scales()
        if len(scales) != LAYERS:
            raise ValueError(f"checkpoint must contain {LAYERS} layer scale tables")
        for scale in scales:
            scale.validate()
        self.lm_head_scale.validate()
        if self.score_divisor not in SCORE_DIVISORS:
            raise ValueError(f"score_divisor must be one of {SCORE_DIVISORS}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        version = 2 if self.uses_version_2() else 1
        result = {
            "format": CHECKPOINT_FORMAT,
            "version": version,
            "architecture": {
                "vocab_size": 64,
                "context_length": CONTEXT_DEPTH,
                "layers": LAYERS,
                "d_model": MODEL_WIDTH,
                "heads": HEADS,
                "head_width": 8,
                "ffn_width": FFN_WIDTH,
                "attention_score_clamp": [-6, 6],
                "attention_weight_rule": "1 << (score + 6)",
                "normalization": "MaxNorm(31)",
                "activation": "ReLU",
                "argmax_tie_break": "lowest-token-index",
            },
            "vocabulary": {"alphabet": self.alphabet, "unknown_id": self.alphabet.index(" ")},
            "hardware": {"score_divisor": self.score_divisor, "integer_bits": 32, "maxnorm_scale": 31},
            "embeddings": self.embeddings,
            "layers": [
                {
                    "query": layer.query,
                    "key": layer.key,
                    "value": layer.value,
                    "output": layer.output,
                    "ffn_up": layer.ffn_up,
                    "ffn_down": layer.ffn_down,
                    "relative_bias": bias,
                }
                for layer, bias in zip(self.layers, self.relative_biases, strict=True)
            ],
            "lm_head": self.lm_head,
            "metadata": dict(self.metadata or {}),
        }
        if version == 2:
            result["hardware"]["weight_limit"] = self.weight_limit
            for row, scales in zip(result["layers"], self.normalized_layer_scales(), strict=True):
                row["scales"] = scales.to_dict()
            result["lm_head_scale"] = self.lm_head_scale.to_dict()
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExactCheckpoint":
        version = value.get("version")
        if value.get("format") != CHECKPOINT_FORMAT or version not in {1, 2}:
            raise ValueError("unsupported FactorioLM checkpoint format/version")
        vocabulary = value.get("vocabulary")
        hardware = value.get("hardware")
        raw_layers = value.get("layers")
        if not isinstance(vocabulary, Mapping) or not isinstance(hardware, Mapping) or not isinstance(raw_layers, list):
            raise ValueError("checkpoint is missing vocabulary, hardware, or layers")
        weight_limit = 1 if version == 1 else int(hardware.get("weight_limit", 0))
        if not 1 <= weight_limit <= 127:
            raise ValueError("invalid checkpoint weight_limit")
        layers: list[BlockWeights] = []
        biases: list[list[list[int]]] = []
        scales: list[BlockScales] = []
        def scale(raw: Any, name: str) -> MatrixScale:
            if version == 1:
                return MatrixScale()
            if not isinstance(raw, Mapping):
                raise ValueError(f"{name} must be a scale object")
            return MatrixScale(int(raw.get("numerator", 0)), int(raw.get("denominator", 0)))
        for index, raw in enumerate(raw_layers):
            if not isinstance(raw, Mapping):
                raise ValueError(f"layer {index} must be an object")
            layers.append(BlockWeights(
                query=_integer_matrix(raw.get("query"), 32, 32, f"layers[{index}].query", ternary=weight_limit == 1),
                key=_integer_matrix(raw.get("key"), 32, 32, f"layers[{index}].key", ternary=weight_limit == 1),
                value=_integer_matrix(raw.get("value"), 32, 32, f"layers[{index}].value", ternary=weight_limit == 1),
                output=_integer_matrix(raw.get("output"), 32, 32, f"layers[{index}].output", ternary=weight_limit == 1),
                ffn_up=_integer_matrix(raw.get("ffn_up"), 64, 32, f"layers[{index}].ffn_up", ternary=weight_limit == 1),
                ffn_down=_integer_matrix(raw.get("ffn_down"), 32, 64, f"layers[{index}].ffn_down", ternary=weight_limit == 1),
            ))
            biases.append(_integer_matrix(raw.get("relative_bias"), 4, 64,
                                           f"layers[{index}].relative_bias", ternary=False))
            raw_scales = raw.get("scales", {})
            if version == 2 and not isinstance(raw_scales, Mapping):
                raise ValueError(f"layers[{index}].scales must be an object")
            scales.append(BlockScales(**{
                name: scale(raw_scales.get(name), f"layers[{index}].scales.{name}")
                for name in ("query", "key", "value", "output", "ffn_up", "ffn_down")
            }) if version == 2 else BlockScales())
        checkpoint = cls(
            embeddings=_integer_matrix(value.get("embeddings"), 64, 32, "embeddings", ternary=False),
            layers=layers,
            relative_biases=biases,
            lm_head=_integer_matrix(value.get("lm_head"), 64, 32, "lm_head", ternary=weight_limit == 1),
            score_divisor=int(hardware.get("score_divisor", 0)),
            alphabet=str(vocabulary.get("alphabet", "")),
            metadata=dict(value.get("metadata") or {}),
            weight_limit=weight_limit,
            layer_scales=scales,
            lm_head_scale=(scale(value.get("lm_head_scale"), "lm_head_scale")
                           if version == 2 else MatrixScale()),
        )
        checkpoint.validate()
        return checkpoint

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: str | Path) -> "ExactCheckpoint":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def sparsity_report(self) -> dict[str, dict[str, int | float]]:
        matrices: dict[str, Sequence[Sequence[int]]] = {"lm_head": self.lm_head}
        for layer_index, layer in enumerate(self.layers):
            for name in ("query", "key", "value", "output", "ffn_up", "ffn_down"):
                matrices[f"layer{layer_index}.{name}"] = getattr(layer, name)
        result: dict[str, dict[str, int | float]] = {}
        for name, matrix in matrices.items():
            flat = [item for row in matrix for item in row]
            negative = sum(item < 0 for item in flat)
            zero = flat.count(0)
            positive = sum(item > 0 for item in flat)
            result[name] = {"negative": negative, "zero": zero, "positive": positive,
                            "total": len(flat), "sparsity_percentage": 100.0 * zero / len(flat)}
        return result
