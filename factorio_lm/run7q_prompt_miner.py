"""Exhaustively mine natural Run 7P TinyChat prompts for a surprising demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time
from typing import Any

from .checkpoint import ExactCheckpoint
from .run4_pipeline import _write_json
from .run7p_pipeline import (
    RUN7P_OUTPUT,
    _diverse_topic_names,
    _load_run7_integer,
    _semantic_demo_score,
    _word_counter,
    generate_punctuation_prompt_set,
    validate_incremental_punctuation,
)


RUN7P_INTEGER = RUN7P_OUTPUT / "factorio-lm-tinychat-punctuation-aligned-int4.pt"
RUN7P_EXACT = RUN7P_OUTPUT / "factorio-lm-tinychat-punctuation-exact.json"
_MOOD = re.compile(
    r"\b(?:happy|feel|feeling|sad|thank|lovely day|mood|cheer)\b", re.IGNORECASE)
_SURPRISING = re.compile(
    r"\b(?:alien|ghost|monster|dinosaur|dragon|space|planet|galaxy|rocket|"
    r"universe|moon|machine|robot|treasure|castle|mystery|secret|strange|"
    r"scary|ocean|storm|animal|snake|fish|bird|horse|dream)\b", re.IGNORECASE)


def is_non_mood_prompt(question: str) -> bool:
    return not _MOOD.search(question)


def mine_all_run7p_prompts(*, output_dir: str | Path = "generated/run7q",
                           batch_size: int = 512) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch size must be positive")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    source = RUN7P_OUTPUT / "tinychat-punctuation-train.jsonl"
    records = [json.loads(line) for line in source.read_text(
        encoding="utf-8").splitlines()]
    records = [row for row in records if is_non_mood_prompt(row["question"])]
    work = output / "prompt-mining-exhaustive-work.jsonl"
    completed: dict[tuple[int, int], dict[str, Any]] = {}
    if work.exists():
        for line in work.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[(int(row["source_row"]), int(row["turn"]))] = row
    pending = [row for row in records if (
        int(row["source_row"]), int(row["turn"])) not in completed]

    model, _ = _load_run7_integer(RUN7P_INTEGER, "cpu")
    exact = ExactCheckpoint.read(RUN7P_EXACT)
    validation_prompt = f"Q {records[0]['question']}\nA "
    validate_incremental_punctuation(
        model, exact, prompt=validation_prompt, count=8, device="cpu")
    known_words = _word_counter(source)
    started = time.monotonic()
    for first in range(0, len(pending), batch_size):
        chunk = pending[first:first + batch_size]
        prompts = [f"Q {row['question']}\nA " for row in chunk]
        generated = generate_punctuation_prompt_set(
            model, prompts, 64, device="cpu")
        new_rows: list[dict[str, Any]] = []
        for record, prompt in zip(chunk, prompts, strict=True):
            trajectory = generated[prompt]
            answer = trajectory.split("\n", 1)[0].strip()
            score, quality = _semantic_demo_score(
                record["question"], record["answer"], answer, known_words)
            generic = bool(re.search(
                r"\b(?:thank you for asking|lovely day|happy today|feel a bit)\b",
                answer, re.IGNORECASE))
            topics = _diverse_topic_names(record["question"])
            surprising = bool(_SURPRISING.search(record["question"]))
            row = {
                "source_row": int(record["source_row"]),
                "turn": int(record["turn"]),
                "question": record["question"],
                "reference_answer": record["answer"],
                "prompt": prompt, "generated_answer": answer,
                "generated_64": trajectory,
                "topics": topics, "surprising_prompt": surprising,
                "generic_output": generic,
                "score": score + 8.0 * surprising + 2.0 * min(2, len(topics))
                         - 60.0 * generic,
                **quality,
            }
            new_rows.append(row)
            completed[(row["source_row"], row["turn"])] = row
        with work.open("a", encoding="utf-8") as stream:
            for row in new_rows:
                stream.write(json.dumps(row, sort_keys=True) + "\n")
        print(
            f"Run7Q exhaustive prompts {len(completed):,}/{len(records):,}",
            flush=True)

    ranked = sorted(completed.values(), key=lambda row: (
        -row["expanded_hard_filter_pass"], -row["surprising_prompt"],
        -row["score"], row["source_row"], row["turn"]))
    strict = [row for row in ranked if row["expanded_hard_filter_pass"]
              and not row["generic_output"]]
    surprising_strict = [row for row in strict if row["surprising_prompt"]]
    report = {
        "run": "7Q", "variant": "exhaustive-frozen-run7p-prompt-mining",
        "model_changed": False, "hardware_changed": False,
        "source_records": len(records), "candidates_evaluated": len(completed),
        "generation_characters": 64,
        "exact_recurrent_path_validated_against_checkpoint": True,
        "strict_non_generic_passes": len(strict),
        "surprising_strict_passes": len(surprising_strict),
        "top_100": ranked[:100],
        "top_surprising": [row for row in ranked if row["surprising_prompt"]][
            :100],
        "elapsed_seconds": time.monotonic() - started,
        "work_file": str(work),
    }
    _write_json(output / "prompt-mining-exhaustive.json", report)
    lines = [
        "# Exhaustive Run 7P prompt search", "",
        f"Tried {len(completed):,} non-mood training prompts with the int4 model.", "",
        f"Strict non-generic passes: {len(strict):,}.",
        f"Strict surprising-topic passes: {len(surprising_strict):,}.", "",
    ]
    display_rows = (surprising_strict or strict or [
        row for row in ranked if row["surprising_prompt"]])
    if not strict:
        lines.extend([
            "None passed the filters. These are the highest-scoring topic prompts,",
            "but their answers still failed.",
            "",
        ])
    for index, row in enumerate(display_rows[:20], 1):
        lines.extend([
            f"## {index}. score {row['score']:.3f}", "",
            f"`Q {row['question']}`", "",
            f"`A {row['generated_answer']}`", "",
            f"Reference: `{row['reference_answer']}`", "",
        ])
    (output / "PROMPT-EXHAUSTIVE-FINALISTS.md").write_text(
        "\n".join(lines), encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="generated/run7q")
    parser.add_argument("--batch-size", type=int, default=512)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    mine_all_run7p_prompts(
        output_dir=arguments.output, batch_size=arguments.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
