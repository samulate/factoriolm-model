"""Transaction-level contract for prompt priming and autoregressive feedback."""

from __future__ import annotations

from dataclasses import dataclass, field

from .full_model import ExactFactorioLM


@dataclass
class AutoregressiveController:
    model: ExactFactorioLM
    prompt: str
    prompt_index: int = 0
    generated_tokens: list[int] = field(default_factory=list)
    last_prediction: int | None = None
    transaction: int = 0

    def reset(self) -> None:
        self.model.reset()
        self.prompt_index = 0
        self.generated_tokens = []
        self.last_prediction = None
        self.transaction = 0

    @property
    def priming(self) -> bool:
        return self.prompt_index < len(self.model.vocabulary.encode(self.prompt))

    def step(self) -> dict[str, object]:
        prompt_tokens = self.model.vocabulary.encode(self.prompt)
        if self.priming:
            token = prompt_tokens[self.prompt_index]
            source = "prompt-rom"
            self.prompt_index += 1
        else:
            if self.last_prediction is None:
                raise RuntimeError("generation cannot begin before prompt priming")
            token = self.last_prediction
            source = "autoregressive-feedback"
        trace = self.model.step(token)
        self.last_prediction = trace.chosen_token_id
        # The final prompt prediction is generated character zero;
        # earlier prompt predictions are discarded.
        displayed = self.prompt_index >= len(prompt_tokens)
        if displayed:
            self.generated_tokens.append(trace.chosen_token_id)
        result = {
            "transaction": self.transaction,
            "input_token_id": token,
            "input_character": self.model.vocabulary.decode([token]),
            "source": source,
            "input_is_displayed_generated_character": False,
            "predicted_token_is_displayed": displayed,
            "displayed_generated_token_id": trace.chosen_token_id if displayed else None,
            "displayed_generated_character": trace.chosen_token if displayed else None,
            "predicted_token_id": trace.chosen_token_id,
            "predicted_character": trace.chosen_token,
            "trace": trace,
        }
        self.transaction += 1
        return result

    def run(self, generated_count: int) -> list[dict[str, object]]:
        if generated_count < 0:
            raise ValueError("generated_count cannot be negative")
        if generated_count == 0:
            return []
        events: list[dict[str, object]] = []
        while self.priming:
            events.append(self.step())
        while len(self.generated_tokens) < generated_count:
            events.append(self.step())
        return events

    @property
    def visible_generated_text(self) -> str:
        return self.model.vocabulary.decode(self.generated_tokens)
