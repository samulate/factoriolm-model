"""Scheduled-sampling refinement of Run 7P using TinyChat answers.

Some training steps feed predictions back as context to practice recovering
from generation errors.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time
from typing import Any

from .run4_pipeline import _learning_rate, _write_json
from .run6_pipeline import mask_summary
from .run7_pipeline import evaluate_aligned_torch
from .run7p_pipeline import (
    RUN7P_ALPHABET,
    PunctuationAlignedChatCorpus,
    _load_run7_integer,
    _semantic_demo_score,
    _word_counter,
    generate_punctuation_prompt_set,
)
from .run7q_pipeline import Run7QConfig, prepare_run7q_records
from .training import set_deterministic
from .vocabulary import Vocabulary


RUN7P_INTEGER = Path(
    "generated/run7p/factorio-lm-tinychat-punctuation-aligned-int4.pt")


class SequenceChatCorpus:
    """Natural Q/A records bucketed by equal prompt length for free-running batches."""

    def __init__(self, records_path: str | Path, *, rollout: int,
                 maximum_answer: int, minimum_examples: int) -> None:
        vocabulary = Vocabulary(RUN7P_ALPHABET)
        buckets: dict[int, list[tuple[tuple[int, ...], tuple[int, ...]]]] = {}
        for line in Path(records_path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            answer = str(row["answer"]) + "\n"
            if not rollout <= len(answer) <= maximum_answer + 1:
                continue
            prompt = tuple(vocabulary.encode(f"Q {row['question']}\nA "))
            targets = tuple(vocabulary.encode(answer[:rollout]))
            buckets.setdefault(len(prompt), []).append((prompt, targets))
        self.buckets = {length: rows for length, rows in buckets.items() if rows}
        self.lengths = sorted(self.buckets)
        self.examples = sum(map(len, self.buckets.values()))
        if self.examples < minimum_examples:
            raise ValueError("too few sequence-aware examples")
        self.cumulative: list[int] = []
        total = 0
        for length in self.lengths:
            total += len(self.buckets[length])
            self.cumulative.append(total)

    def batch(self, generator, batch_size: int):
        import torch

        draw = int(torch.randint(0, self.examples, (1,), generator=generator))
        bucket_index = next(
            index for index, end in enumerate(self.cumulative) if draw < end)
        bucket = self.buckets[self.lengths[bucket_index]]
        indexes = torch.randint(
            0, len(bucket), (batch_size,), generator=generator).tolist()
        return [bucket[index][0] for index in indexes], [
            bucket[index][1] for index in indexes]


def _rank_probe(model, records_path: Path, *, device: str,
                known_words: Counter[str]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    prompts = [f"Q {row['question']}\nA " for row in rows]
    generations = generate_punctuation_prompt_set(
        model, prompts, 64, device=device)
    result: list[dict[str, Any]] = []
    for row, prompt in zip(rows, prompts, strict=True):
        trajectory = generations[prompt]
        answer = trajectory.split("\n", 1)[0].strip()
        score, quality = _semantic_demo_score(
            row["question"], row["answer"], answer, known_words)
        generic = any(phrase in answer.lower() for phrase in (
            "thank you for asking", "lovely day", "happy today"))
        result.append({
            "source_row": int(row["source_row"]), "turn": int(row["turn"]),
            "question": row["question"], "reference_answer": row["answer"],
            "generated_answer": answer, "generated_64": trajectory,
            "generic_greeting_output": generic,
            "score": score - 50.0 * generic, **quality,
        })
    result.sort(key=lambda row: (
        -row["expanded_hard_filter_pass"], -row["score"],
        row["source_row"], row["turn"]))
    return result


def train_sequence_refinement(*,
                              config_path: str | Path =
                              "configs/run7q-cosmic-mystery.json",
                              output_dir: str | Path =
                              "generated/run7q/sequence-mystery",
                              steps: int = 100, rollout: int = 16,
                              topic_probability: float = 0.5,
                              maximum_lr: float = 0.00005) -> dict[str, Any]:
    import torch

    if steps < 1 or rollout < 4 or not 0.0 < topic_probability < 1.0:
        raise ValueError("invalid sequence refinement parameters")
    config = Run7QConfig.read(config_path)
    config.validate()
    set_deterministic(config.seed + 71)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_run7q_records(
        output, topic_name=config.topic_name, topic_regex=config.topic_regex)
    ordinary = SequenceChatCorpus(
        Path("generated/run7p/tinychat-punctuation-train.jsonl"),
        rollout=rollout, maximum_answer=config.aligned_maximum_answer_characters,
        minimum_examples=100)
    thematic = SequenceChatCorpus(
        Path(data["splits"]["train"]["records"]), rollout=rollout,
        maximum_answer=config.aligned_maximum_answer_characters,
        minimum_examples=config.thematic_minimum_examples)
    ordinary_validation = PunctuationAlignedChatCorpus(
        "generated/run7p/tinychat-punctuation-validation.jsonl",
        maximum_answer=config.aligned_maximum_answer_characters)
    thematic_validation = PunctuationAlignedChatCorpus(
        data["splits"]["validation"]["records"],
        maximum_answer=config.aligned_maximum_answer_characters,
        minimum_examples=config.thematic_minimum_examples)
    model, _ = _load_run7_integer(RUN7P_INTEGER, config.device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=maximum_lr, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 720_000)
    known_words = _word_counter(
        Path("generated/run7p/tinychat-punctuation-train.jsonl"))
    validation_records = Path(data["splits"]["validation"]["records"])
    log: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(steps):
        use_topic = float(torch.rand((), generator=generator)) < topic_probability
        prompts, targets = (thematic if use_topic else ordinary).batch(
            generator, config.batch_size)
        contexts = [list(prompt) for prompt in prompts]
        target_tensor = torch.tensor(targets, dtype=torch.long, device=config.device)
        teacher_probability = 0.85 - 0.35 * step / max(1, steps - 1)
        optimizer.zero_grad(set_to_none=True)
        objective = 0.0
        for offset in range(rollout):
            context = torch.tensor(
                [row[-64:] for row in contexts], dtype=torch.long,
                device=config.device)
            logits = model(context)[:, -1] / config.integer_logit_temperature
            loss = torch.nn.functional.cross_entropy(
                logits, target_tensor[:, offset]) / rollout
            loss.backward()
            objective += float(loss.detach().cpu())
            predicted = logits.detach().argmax(dim=-1).cpu()
            use_teacher = torch.rand(
                config.batch_size, generator=generator) < teacher_probability
            supplied = torch.where(
                use_teacher, target_tensor[:, offset].detach().cpu(), predicted)
            for row, token in zip(contexts, supplied.tolist(), strict=True):
                row.append(int(token))
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        lr = _learning_rate(
            step, steps, min(20, max(1, steps // 5)), maximum_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        model.constrain_latents()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == steps:
            general_metrics = evaluate_aligned_torch(
                model, ordinary_validation, device=config.device,
                characters=2048, batch_size=config.batch_size,
                temperature=config.integer_logit_temperature)
            topic_metrics = evaluate_aligned_torch(
                model, thematic_validation, device=config.device,
                characters=2048, batch_size=config.batch_size,
                temperature=config.integer_logit_temperature)
            probes = _rank_probe(
                model, validation_records, device=config.device,
                known_words=known_words)
            row = {
                "step": step + 1, "objective": objective,
                "teacher_probability": teacher_probability,
                "general_validation": general_metrics,
                "thematic_validation": topic_metrics,
                "hard_filter_passes": sum(
                    item["expanded_hard_filter_pass"] for item in probes),
                "top_probes": probes[:10],
                "elapsed_seconds": time.monotonic() - started,
            }
            log.append(row)
            snapshot = output / f"sequence-int4-step-{step + 1:04d}.pt"
            torch.save({
                "run": "7Q", "stage": "scheduled-sampling",
                "model": model.state_dict(), "masks": model.masks,
                "denominators": model.denominators, "step": step + 1,
                "config": {
                    "base": config_path, "steps": steps, "rollout": rollout,
                    "topic_probability": topic_probability,
                    "maximum_lr": maximum_lr,
                }, "log": log,
            }, snapshot)
            _write_json(output / "progress.json", {
                "run": "7Q", "stage": "scheduled-sampling",
                "source_checkpoint": str(RUN7P_INTEGER),
                "architecture_changed": False, "mask_changed": False,
                "steps": steps, "rollout": rollout,
                "topic_probability": topic_probability,
                "ordinary_examples": ordinary.examples,
                "thematic_examples": thematic.examples,
                "log": log,
            })
            print(
                f"Run7Q sequence {step + 1}/{steps}: general "
                f"CE={general_metrics['cross_entropy']:.4f}; thematic "
                f"CE={topic_metrics['cross_entropy']:.4f}; demo passes="
                f"{row['hard_filter_passes']}; top="
                f"{probes[0]['generated_answer']!r}", flush=True)
    sparsity = mask_summary(model.masks)
    if sparsity["total_nonzeros"] != 25_100:
        raise AssertionError("sequence refinement changed the fixed Run 7P mask")
    report = {
        "run": "7Q", "variant": "scheduled-sampling-cosmic-mystery",
        "source_checkpoint": str(RUN7P_INTEGER),
        "architecture_changed": False, "mask_changed": False,
        "weight_representation_changed": False, "sparsity": sparsity,
        "dataset": data, "log": log,
        "quality_gate": {
            "passed": any(row["hard_filter_passes"] for row in log),
            "requires_manual_generation_review": True,
        },
    }
    _write_json(output / "training-report.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/run7q-cosmic-mystery.json")
    parser.add_argument("--output", default="generated/run7q/sequence-mystery")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--rollout", type=int, default=16)
    parser.add_argument("--topic-probability", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    train_sequence_refinement(
        config_path=arguments.config, output_dir=arguments.output,
        steps=arguments.steps, rollout=arguments.rollout,
        topic_probability=arguments.topic_probability,
        maximum_lr=arguments.learning_rate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
