"""The fixed 64-token character vocabulary used by training and hardware."""

from __future__ import annotations

from dataclasses import dataclass


# Two layout-friendly whitespace tokens, then 26 lower-case, 26 upper-case,
# and ten digits.  The order is an exported hardware ABI: never sort it.
FACTORIOLM_ALPHABET = "\n " + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + "0123456789"
assert len(FACTORIOLM_ALPHABET) == 64 and len(set(FACTORIOLM_ALPHABET)) == 64


@dataclass(frozen=True)
class Vocabulary:
    alphabet: str = FACTORIOLM_ALPHABET

    def __post_init__(self) -> None:
        if len(self.alphabet) != 64 or len(set(self.alphabet)) != 64:
            raise ValueError("FactorioLM vocabulary must contain 64 unique characters")

    @property
    def unknown_id(self) -> int:
        """Unsupported input maps deterministically to ASCII space."""

        return self.alphabet.index(" ")

    def encode(self, text: str) -> list[int]:
        lookup = {character: index for index, character in enumerate(self.alphabet)}
        return [lookup.get(character, self.unknown_id) for character in text]

    def decode(self, tokens: list[int] | tuple[int, ...]) -> str:
        if any(not 0 <= token < 64 for token in tokens):
            raise ValueError("token IDs must be in [0, 63]")
        return "".join(self.alphabet[token] for token in tokens)

    def to_dict(self) -> dict[str, object]:
        return {"alphabet": self.alphabet, "unknown_id": self.unknown_id}
