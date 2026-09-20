"""Complete exact-int32 six-layer autoregressive FactorioLM emulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from .block import BlockTrace, TransformerBlockEmulator
from .checkpoint import ExactCheckpoint
from .emulator import integer_linear
from .vocabulary import Vocabulary


def greedy_argmax(values: Sequence[int]) -> int:
    """Return the lowest token index having the maximum signed int32 logit."""

    if len(values) != 64:
        raise ValueError("argmax requires exactly 64 logits")
    return max(range(64), key=lambda index: (values[index], -index))


@dataclass(frozen=True)
class ModelStepTrace:
    step: int
    token_id: int
    token: str
    embedding: list[int]
    layer_inputs: list[list[int]]
    layer_outputs: list[list[int]]
    layer_traces: list[BlockTrace]
    logits: list[int]
    chosen_token_id: int
    chosen_token: str

    def to_dict(self, *, include_attention_cells: bool = False) -> dict[str, Any]:
        traces: list[dict[str, Any]] = []
        for trace in self.layer_traces:
            value = trace.to_dict()
            if not include_attention_cells:
                value["attention"].pop("cells", None)
                # Keep K/V vectors to check cache aging and validity.
            traces.append(value)
        return {
            "step": self.step,
            "token_id": self.token_id,
            "token": self.token,
            "embedding": self.embedding,
            "layer_inputs": self.layer_inputs,
            "layer_outputs": self.layer_outputs,
            "layer_traces": traces,
            "logits": self.logits,
            "chosen_token_id": self.chosen_token_id,
            "chosen_token": self.chosen_token,
        }


class ExactFactorioLM:
    """Authoritative exported-checkpoint inference implementation."""

    def __init__(self, checkpoint: ExactCheckpoint) -> None:
        checkpoint.validate()
        self.checkpoint = checkpoint
        self.vocabulary = Vocabulary(checkpoint.alphabet)
        self.layers = [
            TransformerBlockEmulator(
                weights=weights,
                relative_biases=biases,
                score_divisor=checkpoint.score_divisor,
                scales=scales,
                maximum_absolute_weight=checkpoint.weight_limit,
            )
            for weights, biases, scales in zip(
                checkpoint.layers, checkpoint.relative_biases,
                checkpoint.normalized_layer_scales(), strict=True)
        ]
        self.steps = 0

    def reset(self) -> None:
        for layer in self.layers:
            layer.reset()
        self.steps = 0

    def step(self, token_id: int) -> ModelStepTrace:
        if not 0 <= token_id < 64:
            raise ValueError("token ID must be in [0, 63]")
        vector = list(self.checkpoint.embeddings[token_id])
        embedding = list(vector)
        layer_inputs: list[list[int]] = []
        layer_outputs: list[list[int]] = []
        layer_traces: list[BlockTrace] = []
        for layer in self.layers:
            layer_inputs.append(list(vector))
            trace = layer.step(vector)
            vector = list(trace.block_output)
            layer_outputs.append(list(vector))
            layer_traces.append(trace)
        logits = self.checkpoint.lm_head_scale.apply(integer_linear(
            self.checkpoint.lm_head, vector,
            maximum_absolute_weight=self.checkpoint.weight_limit,
        ))
        chosen = greedy_argmax(logits)
        trace = ModelStepTrace(
            step=self.steps,
            token_id=token_id,
            token=self.vocabulary.decode([token_id]),
            embedding=embedding,
            layer_inputs=layer_inputs,
            layer_outputs=layer_outputs,
            layer_traces=layer_traces,
            logits=logits,
            chosen_token_id=chosen,
            chosen_token=self.vocabulary.decode([chosen]),
        )
        self.steps += 1
        return trace

    def prime(self, prompt: str, *, reset: bool = True) -> list[ModelStepTrace]:
        if not prompt:
            raise ValueError("prompt cannot be empty")
        if reset:
            self.reset()
        return [self.step(token) for token in self.vocabulary.encode(prompt)]

    def generate(self, prompt: str, count: int, *, reset: bool = True) -> tuple[str, list[ModelStepTrace]]:
        if count < 0:
            raise ValueError("generation count cannot be negative")
        traces = self.prime(prompt, reset=reset)
        if count == 0:
            return "", traces
        token = traces[-1].chosen_token_id
        generated = [token]
        for _ in range(count - 1):
            trace = self.step(token)
            traces.append(trace)
            token = trace.chosen_token_id
            generated.append(token)
        return self.vocabulary.decode(generated), traces

    def cache_summary(self) -> list[dict[str, Any]]:
        return [
            {
                "layer": index,
                "valid_count": sum(layer.cache.validity),
                "validity": list(layer.cache.validity),
                "newest_key": list(layer.cache.keys.slots[0]),
                "oldest_key": list(layer.cache.keys.slots[-1]),
            }
            for index, layer in enumerate(self.layers)
        ]


def first_generation_trace(checkpoint: ExactCheckpoint, prompt: str = "Factorio ") -> dict[str, Any]:
    """Return the compact exact oracle for the prompt's first visible token."""

    trace = ExactFactorioLM(checkpoint).prime(prompt)[-1]
    return {
        "step": trace.step,
        "input_token": trace.token,
        "input_token_id": trace.token_id,
        "chosen_token": trace.chosen_token,
        "chosen_token_id": trace.chosen_token_id,
        "embedding": trace.embedding,
        "layers": [
            {"block_input": block.block_input, "block_output": block.block_output}
            for block in trace.layer_traces
        ],
        "final_layer_6_output": trace.layer_outputs[-1],
        "logits": trace.logits,
        "argmax_token": trace.chosen_token,
    }
