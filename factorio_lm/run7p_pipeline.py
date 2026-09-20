"""TinyChat fine-tuning with punctuation in place of the ten digit tokens."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any
import unicodedata

from .checkpoint import ExactCheckpoint
from .full_model import ExactFactorioLM
from .run4_pipeline import Run4Config, _learning_rate, _write_json, create_float_model
from .run5_pipeline import collect_range_report, export_integer_student
from .run6_pipeline import IncrementalIntegerBatch, mask_summary
from .run6d_pipeline import load_clean_int4
from .run7_pipeline import (
    EVALUATION_QUESTIONS,
    ChatExample,
    MaskedTokenCorpus,
    Run7Config,
    _PAIR,
    _answer_quality,
    _first_answer_sentence,
    _last_question,
    _run5_config,
    _split_for,
    evaluate_aligned_torch,
    evaluate_masked_exact,
    evaluate_masked_torch,
    train_aligned_model,
    train_chat_model,
)
from .training import set_deterministic
from .vocabulary import Vocabulary


RUN7P_OUTPUT = Path("generated/run7p")
RUN7_OUTPUT = Path("generated/run7")
RUN7P_ALPHABET = (
    "\n " + "abcdefghijklmnopqrstuvwxyz" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + ".,?!;'\"-()"
)
RUN7P_PUNCTUATION = ".,?!;'\"-()"
RUN7P_SOURCE = RUN7_OUTPUT / "tinychat-source-rows.jsonl"
RUN7_FLOAT = RUN7_OUTPUT / "factorio-lm-tinychat-float.pt"
RUN7_INTEGER = RUN7_OUTPUT / "factorio-lm-tinychat-aligned-int4.pt"
RUN7_EXACT = RUN7_OUTPUT / "factorio-lm-tinychat-exact.json"
assert len(RUN7P_ALPHABET) == 64 and len(set(RUN7P_ALPHABET)) == 64
assert RUN7P_ALPHABET[:54] == Vocabulary().alphabet[:54]

_DOCUMENT_SEPARATOR = re.compile(r"<\|endoftext\|>|<endoftext>", re.IGNORECASE)
_TYPOGRAPHY = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-", "\u2026": "...",
    "\u00a0": " ", "\ufeff": "",
})


@dataclass(frozen=True)
class Run7PConfig(Run7Config):
    punctuation_refinement_steps: int = 750
    punctuation_target_weight: float = 1.5
    heldout_questions: int = 600

    def validate(self) -> None:
        super().validate()
        if self.aligned_float_steps != 0:
            raise ValueError("Run 7P keeps the trained punctuation float teacher fixed during alignment")
        if self.punctuation_refinement_steps < 0:
            raise ValueError("punctuation refinement steps cannot be negative")
        if self.punctuation_target_weight < 1.0:
            raise ValueError("punctuation target weight must be at least one")
        if self.heldout_questions < 600:
            raise ValueError("Run 7P must evaluate at least 600 held-out questions")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_tinychat_punctuation(text: str) -> tuple[str, dict[str, int]]:
    """Normalize typography while retaining the Run 7P punctuation ABI."""

    source_characters = len(text)
    text, separators = _DOCUMENT_SEPARATOR.subn("\n", text)
    text = unicodedata.normalize("NFKD", text.translate(_TYPOGRAPHY))
    allowed = set(RUN7P_ALPHABET)
    output: list[str] = []
    unsupported = mapped_boundaries = 0
    previous = ""
    for character in text:
        if character in allowed:
            rendered = character
        elif unicodedata.combining(character):
            unsupported += 1
            continue
        elif character.isspace():
            rendered = " "
        elif unicodedata.category(character).startswith(("P", "S")):
            rendered = " "
            mapped_boundaries += 1
        else:
            unsupported += 1
            continue
        if rendered == " " and previous in {"", " ", "\n", "("}:
            continue
        if rendered == "\n" and previous == "\n":
            continue
        output.append(rendered)
        previous = rendered
    normalized = "".join(output).strip(" ")
    normalized = re.sub(r" +([.,?!;)])", r"\1", normalized)
    normalized = re.sub(r"\( +", "(", normalized)
    return normalized, {
        "source_characters": source_characters,
        "normalized_characters": len(normalized),
        "document_separators_mapped_to_boundaries": separators,
        "unsupported_removed": unsupported,
        "unsupported_punctuation_or_symbols_mapped_to_boundaries": mapped_boundaries,
    }


def _normalize_turn(text: str) -> str:
    normalized, _ = normalize_tinychat_punctuation(text)
    return " ".join(normalized.split())


def _extract_examples(text: str, source_row: int,
                      config: Run7PConfig) -> tuple[list[ChatExample], int]:
    result: list[ChatExample] = []
    digit_rejections = 0
    for turn, (user, assistant) in enumerate(_PAIR.findall(text)):
        raw_question = _last_question(user)
        if raw_question is None:
            continue
        raw_answer = _first_answer_sentence(assistant)
        if any(character.isascii() and character.isdigit()
               for character in raw_question + raw_answer):
            digit_rejections += 1
            continue
        question = _normalize_turn(raw_question)
        answer = _normalize_turn(raw_answer)
        if not (config.minimum_question_characters <= len(question)
                <= config.maximum_question_characters):
            continue
        if not (config.minimum_answer_characters <= len(answer)
                <= config.maximum_answer_characters):
            continue
        example = ChatExample(source_row, turn, question, answer)
        if len(example.prompt) > 64:
            raise AssertionError("accepted punctuation prompt exceeds hardware context")
        if any(character not in RUN7P_ALPHABET for character in example.record):
            raise AssertionError("punctuation normalization emitted a character outside its ABI")
        result.append(example)
    return result, digit_rejections


def extract_examples_punctuation(text: str, source_row: int,
                                 config: Run7PConfig) -> list[ChatExample]:
    return _extract_examples(text, source_row, config)[0]


def _serialize_split(examples: list[ChatExample], prefix: Path) -> dict[str, Any]:
    vocabulary = Vocabulary(RUN7P_ALPHABET)
    text = "".join(example.record for example in examples)
    mask = bytearray()
    for example in examples:
        mask.extend(b"\0" * len(example.prompt))
        mask.extend(b"\1" * len(example.answer))
        mask.extend(b"\0")
    tokens = bytes(vocabulary.encode(text))
    text_path = prefix.with_suffix(".txt")
    token_path = prefix.with_suffix(".bin")
    mask_path = prefix.with_suffix(".mask.bin")
    records_path = prefix.with_suffix(".jsonl")
    text_path.write_text(text, encoding="ascii")
    token_path.write_bytes(tokens)
    mask_path.write_bytes(mask)
    records_path.write_text("".join(
        json.dumps(asdict(example), sort_keys=True) + "\n" for example in examples
    ), encoding="utf-8")
    punctuation = Counter(character for character in text if character in RUN7P_PUNCTUATION)
    return {
        "examples": len(examples), "characters": len(text),
        "answer_characters": sum(mask),
        "text": str(text_path), "tokens": str(token_path),
        "answer_mask": str(mask_path), "records": str(records_path),
        "token_sha256": _sha256(token_path), "mask_sha256": _sha256(mask_path),
        "punctuation_counts": {character: punctuation[character]
                               for character in RUN7P_PUNCTUATION},
    }


def prepare_punctuation_corpus(output_dir: str | Path,
                               config: Run7PConfig) -> dict[str, Any]:
    """Rebuild the deterministic Run 7 sample without touching Run 7 files."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not RUN7P_SOURCE.exists():
        raise FileNotFoundError(
            f"cached Run 7 TinyChat source is required: {RUN7P_SOURCE}")
    splits: dict[str, list[ChatExample]] = {"train": [], "validation": []}
    source_pairs = digit_rejections = source_rows = 0
    for line in RUN7P_SOURCE.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        source_rows += 1
        source_pairs += len(_PAIR.findall(row["text"]))
        examples, rejected = _extract_examples(
            row["text"], int(row["row_idx"]), config)
        digit_rejections += rejected
        for example in examples:
            splits[_split_for(example, config.validation_fraction)].append(example)
    if source_rows != config.requested_source_rows:
        raise ValueError(
            f"cached source has {source_rows:,} rows; expected {config.requested_source_rows:,}")
    for values in splits.values():
        values.sort(key=lambda item: (item.source_row, item.turn))
    if min(map(len, splits.values())) < 100:
        raise ValueError("punctuation extraction produced too few examples")
    report = {
        "run": "7P", "alphabet": RUN7P_ALPHABET,
        "digit_policy": "reject the complete question/answer example before normalization",
        "source": {
            "path": str(RUN7P_SOURCE), "rows": source_rows,
            "sha256": _sha256(RUN7P_SOURCE),
            "selection": "deterministic 200000-row Run 7 TinyChat sample",
        },
        "format": "Q question-with-punctuation\\nA answer-with-punctuation\\n",
        "source_conversation_pairs": source_pairs,
        "digit_rejected_examples": digit_rejections,
        "accepted_examples": sum(map(len, splits.values())),
        "splits": {
            name: _serialize_split(values, output / f"tinychat-punctuation-{name}")
            for name, values in splits.items()
        },
    }
    _write_json(output / "dataset-report.json", report)
    return report


class PunctuationAlignedChatCorpus:
    """Prompt-aligned answer targets encoded with the Run 7P vocabulary."""

    def __init__(self, records_path: str | Path, *, maximum_answer: int,
                 minimum_examples: int = 100) -> None:
        vocabulary = Vocabulary(RUN7P_ALPHABET)
        buckets: dict[int, list[tuple[tuple[int, ...], int]]] = {}
        examples = 0
        for line in Path(records_path).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            answer = str(row["answer"])
            if len(answer) > maximum_answer:
                continue
            prompt_tokens = vocabulary.encode(f"Q {row['question']}\nA ")
            answer_tokens = vocabulary.encode(answer + "\n")
            for offset, target in enumerate(answer_tokens):
                context = tuple((prompt_tokens + answer_tokens[:offset])[-64:])
                repeats = 4 if offset < 16 else 3 if offset < 32 else 2 if offset < 48 else 1
                if target == 0:
                    repeats = 4
                buckets.setdefault(len(context), []).extend([(context, target)] * repeats)
            examples += 1
        if examples < minimum_examples or not buckets:
            raise ValueError("too few aligned punctuation examples")
        self.buckets = buckets
        self.examples = examples
        self.items = sum(map(len, buckets.values()))
        self.lengths = sorted(buckets)
        self.cumulative: list[int] = []
        total = 0
        for length in self.lengths:
            total += len(buckets[length])
            self.cumulative.append(total)

    def batch(self, generator, batch_size: int, device: str):
        torch = _torch()
        draw = int(torch.randint(0, self.items, (1,), generator=generator))
        bucket_index = next(index for index, end in enumerate(self.cumulative) if draw < end)
        bucket = self.buckets[self.lengths[bucket_index]]
        indexes = torch.randint(0, len(bucket), (batch_size,), generator=generator).tolist()
        contexts = torch.tensor([bucket[index][0] for index in indexes],
                                dtype=torch.long, device=device)
        targets = torch.tensor([bucket[index][1] for index in indexes],
                               dtype=torch.long, device=device)
        return contexts, targets


def _torch():
    import torch
    return torch


def generate_incremental_punctuation(model, prompts: list[str], count: int,
                                     *, device: str = "cpu") -> list[str]:
    """Run the bit-exact recurrent torch path with the Run 7P alphabet."""

    torch = _torch()
    if not prompts or len({len(prompt) for prompt in prompts}) != 1:
        raise ValueError("batched punctuation prompts must be nonempty and equal length")
    vocabulary = Vocabulary(RUN7P_ALPHABET)
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


def generate_punctuation_prompt_set(model, prompts: list[str], count: int,
                                    *, device: str = "cpu") -> dict[str, str]:
    grouped: dict[int, list[str]] = {}
    for prompt in prompts:
        grouped.setdefault(len(prompt), []).append(prompt)
    result: dict[str, str] = {}
    for _, group in sorted(grouped.items()):
        result.update(zip(
            group,
            generate_incremental_punctuation(model, group, count, device=device),
            strict=True,
        ))
    return result


def validate_incremental_punctuation(model, checkpoint: ExactCheckpoint, *,
                                     prompt: str, count: int,
                                     device: str) -> None:
    fast = generate_incremental_punctuation(model, [prompt], count, device=device)[0]
    exact = ExactFactorioLM(checkpoint).generate(prompt, count)[0]
    if fast != exact:
        first = next(index for index, pair in enumerate(zip(fast, exact, strict=True))
                     if pair[0] != pair[1])
        raise AssertionError(
            f"Run 7P recurrent path differs at character {first}: "
            f"{fast[first]!r} != {exact[first]!r}")


def _symbol_report(targets: list[int], predictions: list[int]) -> dict[str, Any]:
    if len(targets) != len(predictions):
        raise ValueError("target and prediction lengths differ")
    vocabulary = Vocabulary(RUN7P_ALPHABET)
    punctuation_ids = {vocabulary.alphabet.index(character)
                       for character in RUN7P_PUNCTUATION}
    letters_and_space = set(range(1, 54))
    rows: dict[str, Any] = {}
    for character in RUN7P_PUNCTUATION:
        token = vocabulary.alphabet.index(character)
        target_count = sum(value == token for value in targets)
        prediction_count = sum(value == token for value in predictions)
        true_positive = sum(target == token and prediction == token
                            for target, prediction in zip(targets, predictions, strict=True))
        rows[character] = {
            "targets": target_count, "predictions": prediction_count,
            "correct": true_positive,
            "precision": true_positive / max(1, prediction_count),
            "recall": true_positive / max(1, target_count),
        }
    punctuation_targets = sum(target in punctuation_ids for target in targets)
    punctuation_predictions = sum(prediction in punctuation_ids for prediction in predictions)
    punctuation_correct = sum(
        target == prediction and target in punctuation_ids
        for target, prediction in zip(targets, predictions, strict=True))
    letter_targets = [(target, prediction) for target, prediction
                      in zip(targets, predictions, strict=True)
                      if target in letters_and_space]
    return {
        "characters": rows,
        "micro": {
            "targets": punctuation_targets,
            "predictions": punctuation_predictions,
            "correct": punctuation_correct,
            "precision": punctuation_correct / max(1, punctuation_predictions),
            "recall": punctuation_correct / max(1, punctuation_targets),
        },
        "letter_space": {
            "targets": len(letter_targets),
            "accuracy": sum(target == prediction for target, prediction in letter_targets)
                        / max(1, len(letter_targets)),
        },
    }


def evaluate_symbol_torch(model, corpus: MaskedTokenCorpus, *, config: Run7PConfig,
                          temperature: float = 1.0) -> dict[str, Any]:
    torch = _torch()
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    windows = max(1, math.ceil(config.evaluation_characters / 64))
    with torch.no_grad():
        for first in range(0, windows, config.batch_size):
            count = min(config.batch_size, windows - first)
            starts = (torch.arange(first, first + count) * 64) % (
                len(corpus.tokens) - 65)
            values, answer = corpus.weighted_batch(starts, 64, config.device)
            logits = model(values[:, :-1]) / temperature
            expected = values[:, 1:]
            predicted = logits.argmax(dim=-1)
            active = answer.bool()
            targets.extend(expected[active].to(torch.int64).cpu().tolist())
            predictions.extend(predicted[active].to(torch.int64).cpu().tolist())
    return _symbol_report(targets, predictions)


def evaluate_symbol_exact(checkpoint: ExactCheckpoint, corpus: MaskedTokenCorpus,
                          *, characters: int) -> dict[str, Any]:
    tokens = bytes(corpus.tokens.tolist())
    mask = bytes(corpus.answer_mask.tolist())
    count = min(characters, len(tokens) - 1)
    targets: list[int] = []
    predictions: list[int] = []
    for start in range(0, count, 64):
        model = ExactFactorioLM(checkpoint)
        for offset in range(min(64, count - start)):
            trace = model.step(tokens[start + offset])
            if mask[start + offset + 1]:
                targets.append(tokens[start + offset + 1])
                predictions.append(trace.chosen_token_id)
    return _symbol_report(targets, predictions)


def _load_run7_integer(path: Path, device: str):
    torch = _torch()
    _, _, _, model = load_clean_int4()
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.masks = {name: value.to(device) for name, value in state["masks"].items()}
    model.denominators = dict(state["denominators"])
    return model, state


def _export_checkpoint(model, config: Run7PConfig, path: Path, *,
                       stage: str) -> ExactCheckpoint:
    metadata = {
        "run": 7, "variant": "7P", "dataset": "starhopp3r/TinyChat",
        "task": "punctuation-capable compact question answering",
        "seed": config.seed, "initialization": str(RUN7_INTEGER),
        "fixed_run7_mask": True, "nonzero_weights": config.nonzero_weights,
        "weight_alphabet": list(range(-7, 8)), "score_divisor": 512,
        "prompt_format": "Q question-with-punctuation\\nA ",
        "training_stage": stage,
    }
    checkpoint = export_integer_student(
        model, _run5_config(config), metadata, alphabet=RUN7P_ALPHABET)
    checkpoint.write(path)
    return checkpoint


def train_punctuation_refinement(model, teacher,
                                 train: PunctuationAlignedChatCorpus,
                                 validation: PunctuationAlignedChatCorpus,
                                 config: Run7PConfig, *,
                                 output_path: Path) -> dict[str, Any]:
    """One conditional QAT pass with punctuation targets weighted 1.5x."""

    torch = _torch()
    model.to(config.device)
    teacher.to(config.device).eval()
    model.masks = {name: value.to(config.device) for name, value in model.masks.items()}
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.aligned_integer_learning_rate,
        weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 930_000)
    punctuation_ids = torch.tensor(
        [RUN7P_ALPHABET.index(character) for character in RUN7P_PUNCTUATION],
        device=config.device)
    best = float("inf")
    log: list[dict[str, Any]] = []
    started = time.monotonic()
    steps = config.punctuation_refinement_steps
    for step in range(steps):
        model.train()
        contexts, targets = train.batch(generator, config.batch_size, config.device)
        logits = model(contexts)[:, -1]
        temperature = config.integer_logit_temperature
        target_weights = torch.where(
            (targets[:, None] == punctuation_ids[None, :]).any(dim=1),
            torch.full_like(targets, config.punctuation_target_weight,
                            dtype=torch.float32),
            torch.ones_like(targets, dtype=torch.float32))
        hard_rows = torch.nn.functional.cross_entropy(
            logits / temperature, targets, reduction="none")
        hard = (hard_rows * target_weights).sum() / target_weights.sum()
        with torch.no_grad():
            probabilities = torch.softmax(teacher(contexts)[:, -1] / temperature, dim=-1)
        kl_rows = torch.nn.functional.kl_div(
            torch.log_softmax(logits / temperature, dim=-1), probabilities,
            reduction="none").sum(dim=-1) * temperature * temperature
        distillation = (kl_rows * target_weights).sum() / target_weights.sum()
        loss = ((1.0 - config.distillation_weight) * hard
                + config.distillation_weight * distillation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip).detach().cpu())
        lr = _learning_rate(
            step, steps, min(config.warmup_steps, max(1, steps // 5)),
            config.aligned_integer_learning_rate)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        model.constrain_latents()
        if (step == 0 or (step + 1) % config.evaluation_interval == 0
                or step + 1 == steps):
            metrics = evaluate_aligned_torch(
                model, validation, device=config.device,
                characters=config.evaluation_characters,
                batch_size=config.batch_size,
                temperature=config.integer_logit_temperature)
            row = {
                "step": step + 1, "objective": float(loss.detach().cpu()),
                "gradient_norm": gradient_norm, "learning_rate": lr,
                "validation": metrics,
                "elapsed_seconds": time.monotonic() - started,
            }
            log.append(row)
            print(f"punctuation refinement {step + 1}/{steps}: "
                  f"CE={metrics['cross_entropy']:.4f} "
                  f"acc={metrics['accuracy']:.3%}", flush=True)
            if metrics["cross_entropy"] < best:
                best = float(metrics["cross_entropy"])
                torch.save({
                    "stage": "punctuation-refinement", "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(), "step": step + 1,
                    "best_cross_entropy": best, "masks": model.masks,
                    "denominators": model.denominators,
                    "config": asdict(config), "log": log,
                }, output_path)
    state = torch.load(output_path, map_location=config.device, weights_only=False)
    model.load_state_dict(state["model"])
    model.masks = {name: value.to(config.device) for name, value in state["masks"].items()}
    model.denominators = dict(state["denominators"])
    return {
        "stage": "punctuation-refinement", "steps": steps,
        "best_step": int(state["step"]), "best_cross_entropy": best,
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(output_path), "log": state.get("log", log),
        "punctuation_target_weight": config.punctuation_target_weight,
    }


def _candidate_metrics(checkpoint: ExactCheckpoint, model, float_model,
                       validation: MaskedTokenCorpus,
                       config: Run7PConfig) -> dict[str, Any]:
    return {
        "float": {
            **evaluate_masked_torch(float_model, validation, config=config),
            "symbol_classes": evaluate_symbol_torch(
                float_model, validation, config=config),
        },
        "qat": {
            **evaluate_masked_torch(
                model, validation, config=config,
                temperature=config.integer_logit_temperature),
            "symbol_classes": evaluate_symbol_torch(
                model, validation, config=config,
                temperature=config.integer_logit_temperature),
        },
        "exact": {
            **evaluate_masked_exact(
                checkpoint, validation,
                characters=config.exact_evaluation_characters),
            "symbol_classes": evaluate_symbol_exact(
                checkpoint, validation,
                characters=config.exact_evaluation_characters),
        },
    }


def _legacy_comparison(checkpoint: ExactCheckpoint, config: Run7PConfig) -> dict[str, Any]:
    legacy_report = json.loads(
        (RUN7_OUTPUT / "dataset-report.json").read_text(encoding="utf-8"))
    legacy = MaskedTokenCorpus(
        legacy_report["splits"]["validation"]["tokens"],
        legacy_report["splits"]["validation"]["answer_mask"])
    baseline = ExactCheckpoint.read(RUN7_EXACT)
    run7 = evaluate_masked_exact(
        baseline, legacy, characters=config.exact_evaluation_characters)
    run7p = evaluate_masked_exact(
        checkpoint, legacy, characters=config.exact_evaluation_characters)
    return {
        "source": "Run 7 punctuation-free validation tokens 0..53",
        "run7": run7, "run7p": run7p,
        "letter_space_answer_accuracy": {
            "run7": run7["answer_characters"]["accuracy"],
            "run7p": run7p["answer_characters"]["accuracy"],
        },
    }


def _evaluation_generations(model, config: Run7PConfig) -> dict[str, Any]:
    prompts = [f"Q {question}?\nA " for question in EVALUATION_QUESTIONS]
    generated = generate_punctuation_prompt_set(
        model, prompts, config.generation_length, device=config.device)
    rows: dict[str, Any] = {}
    for question, prompt in zip(EVALUATION_QUESTIONS, prompts, strict=True):
        text = generated[prompt]
        answer = text.split("\n", 1)[0].strip()
        rows[question] = {
            "prompt": prompt, "generated": text,
            **_answer_quality(answer),
            "punctuation_count": sum(character in RUN7P_PUNCTUATION
                                     for character in answer),
            "terminal_punctuation": bool(answer) and answer[-1] in ".!?",
        }
    return rows


def train_run7p(*, config_path: str | Path = "configs/run7p-punctuation.json",
                output_dir: str | Path = RUN7P_OUTPUT) -> dict[str, Any]:
    """Train the standard Run 7P recipe and its conditional punctuation pass."""

    torch = _torch()
    config = Run7PConfig.read(config_path)
    config.validate()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    set_deterministic(config.seed)
    random.seed(config.seed)
    data = prepare_punctuation_corpus(output, config)
    train = MaskedTokenCorpus(
        data["splits"]["train"]["tokens"],
        data["splits"]["train"]["answer_mask"])
    validation = MaskedTokenCorpus(
        data["splits"]["validation"]["tokens"],
        data["splits"]["validation"]["answer_mask"])

    float_model = create_float_model(Run4Config(seed=config.seed, device=config.device))
    float_state = torch.load(RUN7_FLOAT, map_location=config.device, weights_only=False)
    float_model.load_state_dict(float_state["model"])
    float_path = output / "factorio-lm-tinychat-punctuation-float.pt"
    float_training = train_chat_model(
        float_model, None, train, validation, config, stage="float",
        output_path=float_path)

    integer_model, _ = _load_run7_integer(RUN7_INTEGER, config.device)
    integer_path = output / "factorio-lm-tinychat-punctuation-int4.pt"
    integer_training = train_chat_model(
        integer_model, float_model, train, validation, config, stage="integer",
        output_path=integer_path)

    train_aligned = PunctuationAlignedChatCorpus(
        data["splits"]["train"]["records"],
        maximum_answer=config.aligned_maximum_answer_characters)
    validation_aligned = PunctuationAlignedChatCorpus(
        data["splits"]["validation"]["records"],
        maximum_answer=config.aligned_maximum_answer_characters)
    aligned_path = output / "factorio-lm-tinychat-punctuation-aligned-int4.pt"
    aligned_training = train_aligned_model(
        integer_model, float_model, train_aligned, validation_aligned, config,
        stage="integer", output_path=aligned_path)

    standard_exact_path = output / "factorio-lm-tinychat-punctuation-standard-exact.json"
    standard = _export_checkpoint(
        integer_model, config, standard_exact_path, stage="standard-aligned-QAT")
    standard_metrics = _candidate_metrics(
        standard, integer_model, float_model, validation, config)
    float_recall = standard_metrics["float"]["symbol_classes"]["micro"]["recall"]
    standard_recall = standard_metrics["exact"]["symbol_classes"]["micro"]["recall"]
    refine_required = (
        config.punctuation_refinement_steps > 0
        and float_recall >= 0.50
        and standard_recall + 0.15 < float_recall
    )

    refinement_training: dict[str, Any] | None = None
    refinement_metrics: dict[str, Any] | None = None
    selected_stage = "standard-aligned-QAT"
    selected_state_path = aligned_path
    selected_checkpoint = standard
    if refine_required:
        refinement_path = output / "factorio-lm-tinychat-punctuation-refined-int4.pt"
        refinement_training = train_punctuation_refinement(
            integer_model, float_model, train_aligned, validation_aligned, config,
            output_path=refinement_path)
        refined_exact_path = output / "factorio-lm-tinychat-punctuation-refined-exact.json"
        refined = _export_checkpoint(
            integer_model, config, refined_exact_path,
            stage="punctuation-weighted-QAT")
        refinement_metrics = _candidate_metrics(
            refined, integer_model, float_model, validation, config)
        refined_exact = refinement_metrics["exact"]
        standard_exact = standard_metrics["exact"]
        recall_gain = (
            refined_exact["symbol_classes"]["micro"]["recall"]
            - standard_exact["symbol_classes"]["micro"]["recall"])
        letter_loss = (
            standard_exact["symbol_classes"]["letter_space"]["accuracy"]
            - refined_exact["symbol_classes"]["letter_space"]["accuracy"])
        answer_gain = (
            refined_exact["answer_characters"]["accuracy"]
            - standard_exact["answer_characters"]["accuracy"])
        if ((recall_gain >= 0.05 and letter_loss <= 0.02)
                or (answer_gain > 0 and recall_gain >= 0)):
            selected_stage = "punctuation-weighted-QAT"
            selected_state_path = refinement_path
            selected_checkpoint = refined
        else:
            integer_model, _ = _load_run7_integer(aligned_path, config.device)

    final_path = output / "factorio-lm-tinychat-punctuation-exact.json"
    selected_checkpoint.write(final_path)
    selected_metrics = (refinement_metrics if selected_stage == "punctuation-weighted-QAT"
                        else standard_metrics)
    assert selected_metrics is not None
    validate_incremental_punctuation(
        integer_model, selected_checkpoint,
        prompt="Q How are you today?\nA ", count=32, device=config.device)
    generations = _evaluation_generations(integer_model, config)
    legacy = _legacy_comparison(selected_checkpoint, config)
    ranges = collect_range_report(integer_model, validation, device=config.device, windows=64)
    _write_json(output / "range-report.json", ranges)
    exact_answer = selected_metrics["exact"]["answer_characters"]
    accuracy_gate = exact_answer["accuracy"] >= 0.5140442986054142
    report = {
        "run": "7P", "configuration": asdict(config), "data": data,
        "alphabet": RUN7P_ALPHABET,
        "initialization": {
            "float": str(RUN7_FLOAT), "integer": str(RUN7_INTEGER),
            "mask_changed": False, "architecture_changed": False,
        },
        "training": {
            "float": float_training, "integer_qat": integer_training,
            "aligned_integer_qat": aligned_training,
            "conditional_punctuation_refinement": refinement_training,
            "conditional_refinement_triggered": refine_required,
        },
        "candidates": {
            "standard": standard_metrics,
            "refined": refinement_metrics,
            "selected_stage": selected_stage,
            "selected_state": str(selected_state_path),
        },
        "metrics": selected_metrics,
        "legacy_run7_letter_space_comparison": legacy,
        "generations": generations,
        "sparsity": mask_summary(integer_model.masks),
        "range_report": str(output / "range-report.json"),
        "quality_gate": {
            "minimum_exact_answer_accuracy": 0.5140442986054142,
            "exact_accuracy_within_two_points_of_run7": accuracy_gate,
            "heldout_punctuation_demo_selected": False,
        },
        "exact_checkpoint": str(final_path),
    }
    _write_json(output / "training-report.json", report)
    _write_json(output / "evaluation-report.json", {
        "metrics": selected_metrics,
        "legacy_run7_letter_space_comparison": legacy,
        "generations": generations,
        "quality_gate": report["quality_gate"],
    })
    (output / "generation-samples.txt").write_text("".join(
        f"QUESTION: {question}\nPROMPT: {row['prompt']!r}\n"
        f"EXACT: {row['generated']}\n\n"
        for question, row in generations.items()), encoding="utf-8")
    _write_json(output / "manifest.json", {
        "run": "7P", "checkpoint": str(final_path),
        "dataset_report": str(output / "dataset-report.json"),
        "training_report": str(output / "training-report.json"),
        "evaluation_report": str(output / "evaluation-report.json"),
        "heldout_mining": None, "selected_demo": None,
    })
    return {"checkpoint": selected_checkpoint, "model": integer_model,
            "report": report, "output": output}


def _load_selected_model(output: Path, config: Run7PConfig):
    report = json.loads((output / "training-report.json").read_text(encoding="utf-8"))
    return _load_run7_integer(Path(report["candidates"]["selected_state"]), config.device)[0]


def _word_counter(records: Path) -> Counter[str]:
    words: Counter[str] = Counter()
    for line in records.read_text(encoding="utf-8").splitlines():
        answer = str(json.loads(line)["answer"])
        words.update(word.lower() for word in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", answer))
    return words


def _mine_score(answer: str, known_words: Counter[str]) -> tuple[float, dict[str, Any]]:
    words = re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", answer)
    lower = [word.lower() for word in words]
    recognized = sum(known_words[word] >= 3 for word in lower)
    adjacent = sum(left == right for left, right in zip(lower, lower[1:]))
    unique = len(set(lower)) / max(1, len(lower))
    word_rate = recognized / max(1, len(lower))
    terminal = bool(answer) and answer[-1] in ".!?"
    punctuation_count = sum(character in RUN7P_PUNCTUATION for character in answer)
    repeated_ngram = 0
    for size in (2, 3):
        ngrams = [tuple(lower[index:index + size])
                  for index in range(max(0, len(lower) - size + 1))]
        repeated_ngram += len(ngrams) - len(set(ngrams))
    maximum_run = max(
        (len(match.group(0)) for match in re.finditer(r"(.)\1+", answer)),
        default=1)
    score = (
        24.0 * word_rate
        + 16.0 * unique
        + 18.0 * terminal
        + 1.4 * min(len(words), 12)
        + 2.0 * min(punctuation_count, 3)
        - 18.0 * adjacent
        - 8.0 * repeated_ngram
        - 12.0 * max(0, maximum_run - 2)
        - 0.4 * max(0, len(answer) - 100)
    )
    hard_pass = (
        5 <= len(words) <= 24 and len(answer) <= 110 and terminal
        and word_rate >= 0.80 and adjacent == 0 and repeated_ngram == 0
        and maximum_run <= 2 and 1 <= punctuation_count <= 6
    )
    return score, {
        "words": len(words), "word_rate": word_rate,
        "unique_word_ratio": unique, "adjacent_duplicate_words": adjacent,
        "repeated_word_ngrams": repeated_ngram,
        "maximum_character_run": maximum_run,
        "punctuation_count": punctuation_count,
        "terminal_punctuation": terminal, "hard_filter_pass": hard_pass,
    }


_DEMO_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
    "from", "had", "has", "have", "he", "her", "him", "his", "i", "if",
    "in", "is", "it", "me", "my", "not", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "why", "will", "with",
    "you", "your",
}

RUN7P_DIVERSE_THEMES = {
    "animals": (
        r"\b(?:animal|animals|dog|cat|bird|horse|fish|rat|worm|bear|lion|"
        r"mouse|rabbit|pet|pets|snake|monkey|zoo)\b"),
    "space_mystery": (
        r"\b(?:space|moon|star|stars|universe|planet|sky|mystery|mysterious|"
        r"strange|secret)\b"),
    "weather_nature": (
        r"\b(?:storm|rain|snow|fog|mist|thunder|lightning|sun|beach|ocean|"
        r"forest|river|tree|flower|flowers|nature)\b"),
    "machines_adventure": (
        r"\b(?:machine|train|engine|ship|car|adventure|treasure|castle|"
        r"explore|discover)\b"),
    "fear_horror": (
        r"\b(?:scary|scared|fear|afraid|dark|dangerous|ghost|monster|creepy|"
        r"spooky)\b"),
    "food": (
        r"\b(?:food|soup|cheese|cookie|cookies|breakfast|cook|cooking|fruit|"
        r"cake|bread|dinner|eat)\b"),
    "books_creativity": (
        r"\b(?:book|books|story|stories|read|reading|music|song|paint|"
        r"painting|draw|drawing|art)\b"),
}
_DIVERSE_MOOD_PATTERN = re.compile(
    r"\b(?:happy|feel|feeling|sad|thank|lovely day)\b", re.IGNORECASE)
_GENERIC_DEPLOYMENT_PATTERN = re.compile(
    r"\b(?:happy|thank you for asking|lovely day|beautiful day|feel a bit)\b",
    re.IGNORECASE)


def _demo_words(text: str) -> list[str]:
    return [word.lower() for word in re.findall(
        r"[A-Za-z]+(?:'[A-Za-z]+)?", text)]


def _diverse_topic_names(question: str) -> list[str]:
    return [
        name for name, pattern in RUN7P_DIVERSE_THEMES.items()
        if re.search(pattern, question, re.IGNORECASE)
    ]


def _set_f1(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def _semantic_demo_score(question: str, reference: str, answer: str,
                         known_words: Counter[str]) -> tuple[float, dict[str, Any]]:
    """Rank answers by fluency, reference similarity, and question-word overlap.

    Fluency alone can give irrelevant stock replies a high score.
    """

    surface_score, surface = _mine_score(answer, known_words)
    answer_words = _demo_words(answer)
    reference_words = _demo_words(reference)
    question_words = _demo_words(question)
    answer_set = set(answer_words)
    reference_set = set(reference_words)
    answer_content = answer_set - _DEMO_STOP_WORDS
    reference_content = reference_set - _DEMO_STOP_WORDS
    question_content = set(question_words) - _DEMO_STOP_WORDS
    sequence_similarity = SequenceMatcher(
        None, " ".join(answer_words), " ".join(reference_words)).ratio()
    word_f1 = _set_f1(answer_set, reference_set)
    content_f1 = _set_f1(answer_content, reference_content)
    question_content_recall = (
        len(answer_content & question_content) / max(1, len(question_content)))
    begins_uppercase = bool(answer) and answer[0].isupper()
    useful_length = 24 <= len(answer) <= 96
    malformed_fragment = bool(re.search(
        r"\b(?:a|an|the|to|with|for|and|but|because)\s*[.!?]$", answer,
        flags=re.IGNORECASE))
    semantic_score = (
        surface_score
        + 38.0 * sequence_similarity
        + 18.0 * word_f1
        + 16.0 * content_f1
        + 6.0 * min(question_content_recall, 0.5) / 0.5
        + 5.0 * begins_uppercase
        + 6.0 * useful_length
        - 24.0 * malformed_fragment
        - 1.25 * max(0, 20 - len(answer))
    )
    expanded_pass = bool(
        surface["hard_filter_pass"] and begins_uppercase and useful_length
        and sequence_similarity >= 0.28 and word_f1 >= 0.30
        and not malformed_fragment)
    return semantic_score, {
        **surface,
        "surface_score": surface_score,
        "reference_sequence_similarity": sequence_similarity,
        "reference_word_f1": word_f1,
        "reference_content_word_f1": content_f1,
        "question_content_recall": question_content_recall,
        "begins_uppercase": begins_uppercase,
        "useful_demo_length": useful_length,
        "malformed_terminal_fragment": malformed_fragment,
        "expanded_hard_filter_pass": expanded_pass,
    }


def _write_expanded_finalists(output: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Run 7P expanded prompt finalists",
        "",
        "Held-out TinyChat questions, with 128 characters of exact int32 generation.",
        "Ranked by fluency and reference similarity. Check relevance before choosing a demo.",
        "",
    ]
    for index, row in enumerate(rows[:30], 1):
        lines.extend([
            f"## {index}. score {row['expanded_score']:.3f}",
            "",
            f"Source: row {row['source_row']}, turn {row['turn']}",
            "",
            "Question:",
            "",
            "```text",
            row["question"],
            "```",
            "",
            "Exact generated answer:",
            "",
            "```text",
            row["generated_answer"],
            "```",
            "",
            "Held-out reference:",
            "",
            "```text",
            row["reference_answer"],
            "```",
            "",
            (f"Reference similarity {row['reference_sequence_similarity']:.3f}, "
             f"word F1 {row['reference_word_f1']:.3f}, "
             f"surface score {row['surface_score']:.3f}."),
            "",
        ])
    (output / "PROMPT-FINALISTS-EXPANDED.md").write_text(
        "\n".join(lines), encoding="utf-8")


def mine_expanded_run7p(*,
                        config_path: str | Path = "configs/run7p-punctuation.json",
                        output_dir: str | Path = RUN7P_OUTPUT) -> dict[str, Any]:
    """Rank the complete held-out split for coherent live-demo answers."""

    config = Run7PConfig.read(config_path)
    config.validate()
    output = Path(output_dir)
    data = json.loads((output / "dataset-report.json").read_text(encoding="utf-8"))
    records_path = Path(data["splits"]["validation"]["records"])
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    prompts = [f"Q {row['question']}\nA " for row in records]
    model = _load_selected_model(output, config)
    started = time.monotonic()
    generations = generate_punctuation_prompt_set(
        model, prompts, 128, device=config.device)
    known_words = _word_counter(Path(data["splits"]["train"]["records"]))
    rows: list[dict[str, Any]] = []
    for record, prompt in zip(records, prompts, strict=True):
        generated = generations[prompt]
        answer = generated.split("\n", 1)[0].strip()
        score, quality = _semantic_demo_score(
            record["question"], record["answer"], answer, known_words)
        rows.append({
            "source_row": int(record["source_row"]),
            "turn": int(record["turn"]),
            "question": record["question"],
            "reference_answer": record["answer"],
            "prompt": prompt,
            "prompt_length": len(prompt),
            "generated_128": generated,
            "generated_answer": answer,
            "first_32": generated[:32],
            "first_64": generated[:64],
            "expanded_score": score,
            **quality,
        })
    surface_rows = sorted(rows, key=lambda row: (
        -row["hard_filter_pass"], -row["surface_score"],
        row["source_row"], row["turn"]))
    rows.sort(key=lambda row: (
        -row["expanded_hard_filter_pass"], -row["expanded_score"],
        row["source_row"], row["turn"]))
    retained = rows[:100]
    checkpoint_path = output / "factorio-lm-tinychat-punctuation-exact.json"
    report = {
        "run": "7P",
        "purpose": "expanded semantic live-demo prompt mining",
        "model_sha256": _sha256(checkpoint_path),
        "validation_records_sha256": _sha256(records_path),
        "candidates_evaluated": len(rows),
        "inference": "bit-exact recurrent signed-integer model",
        "generation_characters_per_candidate": 128,
        "elapsed_seconds": time.monotonic() - started,
        "scoring": {
            "surface": "word rate, uniqueness, terminal punctuation, length, and repetition",
            "semantic": "held-out reference sequence similarity, word/content F1, and question-content continuity",
            "selection": "automatic top 100 followed by manual semantic review",
        },
        "expanded_hard_filter_passes": sum(
            row["expanded_hard_filter_pass"] for row in rows),
        "top_100": retained,
        "top_100_by_surface_quality": surface_rows[:100],
        "current_demo_preserved": json.loads(
            (output / "selected-demo.json").read_text(encoding="utf-8")),
        "selection_status": "ranked alternatives; not selected for release",
    }
    _write_json(output / "prompt-mining-expanded.json", report)
    _write_expanded_finalists(output, retained)
    return report


def mine_training_patterns_run7p(*,
                                 config_path: str | Path = "configs/run7p-punctuation.json",
                                 output_dir: str | Path = RUN7P_OUTPUT) -> dict[str, Any]:
    """Probe the strongest learned chat family in the training split.

    This is explicitly a diagnostic supplement, not a held-out benchmark. It
    helps determine whether training-set prompt patterns reveal a materially
    stronger public demo than the complete held-out search.
    """

    config = Run7PConfig.read(config_path)
    config.validate()
    output = Path(output_dir)
    data = json.loads((output / "dataset-report.json").read_text(encoding="utf-8"))
    records_path = Path(data["splits"]["train"]["records"])
    all_records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    mood_pattern = re.compile(
        r"(?:how (?:are|do).*you|how about you|feel(?:ing)? "
        r"(?:today|happy|sad)|your day)", re.IGNORECASE)
    records = [
        row for row in all_records
        if mood_pattern.search(row["question"])
        and 24 <= len(row["answer"]) <= 96
        and row["answer"][-1:] in ".!?"
        and len(_demo_words(row["answer"])) >= 5
    ]
    prompts = [f"Q {row['question']}\nA " for row in records]
    model = _load_selected_model(output, config)
    started = time.monotonic()
    generations = generate_punctuation_prompt_set(
        model, prompts, 128, device=config.device)
    known_words = _word_counter(records_path)
    rows: list[dict[str, Any]] = []
    for record, prompt in zip(records, prompts, strict=True):
        generated = generations[prompt]
        answer = generated.split("\n", 1)[0].strip()
        score, quality = _semantic_demo_score(
            record["question"], record["answer"], answer, known_words)
        rows.append({
            "source_row": int(record["source_row"]),
            "turn": int(record["turn"]),
            "question": record["question"],
            "reference_answer": record["answer"],
            "prompt": prompt,
            "prompt_length": len(prompt),
            "generated_128": generated,
            "generated_answer": answer,
            "first_32": generated[:32],
            "first_64": generated[:64],
            "expanded_score": score,
            **quality,
        })
    rows.sort(key=lambda row: (
        -row["expanded_hard_filter_pass"], -row["expanded_score"],
        row["source_row"], row["turn"]))
    report = {
        "run": "7P",
        "purpose": "training-pattern diagnostic for demo prompt discovery",
        "benchmark_warning": (
            "These are training-split prompts and must not be described as held-out."),
        "source_training_records": len(all_records),
        "pattern_candidates_evaluated": len(rows),
        "pattern": mood_pattern.pattern,
        "inference": "bit-exact recurrent signed-integer model",
        "generation_characters_per_candidate": 128,
        "elapsed_seconds": time.monotonic() - started,
        "expanded_hard_filter_passes": sum(
            row["expanded_hard_filter_pass"] for row in rows),
        "top_100": rows[:100],
    }
    _write_json(output / "prompt-mining-training-patterns.json", report)
    return report


def mine_diverse_run7p(*,
                       config_path: str | Path = "configs/run7p-punctuation.json",
                       output_dir: str | Path = RUN7P_OUTPUT) -> dict[str, Any]:
    """Search prominent non-mood TinyChat themes for an unusual exact demo."""

    config = Run7PConfig.read(config_path)
    config.validate()
    output = Path(output_dir)
    data = json.loads((output / "dataset-report.json").read_text(encoding="utf-8"))
    records_path = Path(data["splits"]["train"]["records"])
    all_records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    themed: list[tuple[dict[str, Any], list[str]]] = []
    raw_theme_counts = Counter()
    non_mood_theme_counts = Counter()
    for row in all_records:
        names = _diverse_topic_names(row["question"])
        raw_theme_counts.update(names)
        if names and not _DIVERSE_MOOD_PATTERN.search(row["question"]):
            themed.append((row, names))
            non_mood_theme_counts.update(names)

    model = _load_selected_model(output, config)
    known_words = _word_counter(records_path)
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    batch = 512
    for first in range(0, len(themed), batch):
        chunk = themed[first:first + batch]
        prompts = [f"Q {row['question']}\nA " for row, _ in chunk]
        generated = generate_punctuation_prompt_set(
            model, prompts, 128, device=config.device)
        for (record, themes), prompt in zip(chunk, prompts, strict=True):
            trajectory = generated[prompt]
            answer = trajectory.split("\n", 1)[0].strip()
            score, quality = _semantic_demo_score(
                record["question"], record["answer"], answer, known_words)
            generic = bool(_GENERIC_DEPLOYMENT_PATTERN.search(answer))
            strict_pass = bool(
                quality["hard_filter_pass"] and not generic
                and quality["reference_sequence_similarity"] >= 0.45
                and quality["reference_content_word_f1"] >= 0.35)
            diverse_score = (
                score - 80.0 * generic
                + 16.0 * quality["reference_content_word_f1"])
            rows.append({
                "source_row": int(record["source_row"]),
                "turn": int(record["turn"]),
                "themes": themes,
                "question": record["question"],
                "reference_answer": record["answer"],
                "prompt": prompt,
                "generated_answer": answer,
                "generated_128": trajectory,
                "diverse_score": diverse_score,
                "generic_mood_output": generic,
                "strict_non_mood_pass": strict_pass,
                **quality,
            })
    rows.sort(key=lambda row: (
        -row["strict_non_mood_pass"], -row["diverse_score"],
        row["source_row"], row["turn"]))
    report = {
        "run": "7P",
        "purpose": "find an unusual, coherent demo outside the greeting/mood basin",
        "split": "training (prompt-discovery diagnostic, not held-out evaluation)",
        "training_records": len(all_records),
        "theme_counts": dict(sorted(raw_theme_counts.items())),
        "non_mood_theme_counts": dict(sorted(non_mood_theme_counts.items())),
        "themed_non_mood_candidates_evaluated": len(rows),
        "generation_characters_per_candidate": 128,
        "inference": "bit-exact recurrent signed-integer model",
        "elapsed_seconds": time.monotonic() - started,
        "strict_filter": (
            "surface hard pass; no generic mood phrase; reference sequence "
            "similarity >= 0.45; reference content-word F1 >= 0.35"),
        "strict_non_mood_passes": sum(row["strict_non_mood_pass"] for row in rows),
        "top_100": rows[:100],
        "selection": None,
        "selection_status": "automatic search complete; manual semantic review required",
    }
    _write_json(output / "prompt-mining-diverse.json", report)
    return report


def mine_heldout_run7p(*,
                       config_path: str | Path = "configs/run7p-punctuation.json",
                       output_dir: str | Path = RUN7P_OUTPUT) -> dict[str, Any]:
    """Rank 600 natural validation questions using exact recurrent inference."""

    config = Run7PConfig.read(config_path)
    config.validate()
    output = Path(output_dir)
    data = json.loads((output / "dataset-report.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in Path(
        data["splits"]["validation"]["records"]).read_text(encoding="utf-8").splitlines()]
    records = records[:config.heldout_questions]
    if len(records) < config.heldout_questions:
        raise ValueError("not enough held-out punctuation records")
    prompts = [f"Q {row['question']}\nA " for row in records]
    model = _load_selected_model(output, config)
    generations = generate_punctuation_prompt_set(
        model, prompts, 128, device=config.device)
    known_words = _word_counter(Path(data["splits"]["train"]["records"]))
    rows: list[dict[str, Any]] = []
    for record, prompt in zip(records, prompts, strict=True):
        generated = generations[prompt]
        answer = generated.split("\n", 1)[0].strip()
        score, quality = _mine_score(answer, known_words)
        rows.append({
            "source_row": int(record["source_row"]), "turn": int(record["turn"]),
            "question": record["question"], "reference_answer": record["answer"],
            "prompt": prompt, "generated_128": generated,
            "generated_answer": answer, "score": score, **quality,
        })
    rows.sort(key=lambda row: (-row["hard_filter_pass"], -row["score"],
                               row["source_row"], row["turn"]))
    report = {
        "run": "7P", "candidates_evaluated": len(rows),
        "inference": "bit-exact recurrent signed-integer model",
        "hard_filter_passes": sum(row["hard_filter_pass"] for row in rows),
        "top_60": rows[:60],
        "selection_status": "manual semantic review required",
    }
    _write_json(output / "heldout-punctuation-mining.json", report)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    manifest["heldout_mining"] = str(output / "heldout-punctuation-mining.json")
    _write_json(output / "manifest.json", manifest)
    return report


def select_demo_run7p(source_row: int, turn: int, *,
                      output_dir: str | Path = RUN7P_OUTPUT) -> dict[str, Any]:
    """Record one reviewed, unedited held-out exact trajectory."""

    output = Path(output_dir)
    mining = json.loads(
        (output / "heldout-punctuation-mining.json").read_text(encoding="utf-8"))
    try:
        candidate = next(
            row for row in mining["top_60"]
            if row["source_row"] == source_row and row["turn"] == turn)
    except StopIteration as exc:
        raise ValueError("selection must be one of the retained top 60 candidates") from exc
    if not candidate["hard_filter_pass"]:
        raise ValueError("selected demo did not pass the early quality filter")
    checkpoint = ExactCheckpoint.read(
        output / "factorio-lm-tinychat-punctuation-exact.json")
    exact = ExactFactorioLM(checkpoint).generate(candidate["prompt"], 128)[0]
    if exact != candidate["generated_128"]:
        raise AssertionError("mined recurrent generation differs from independent emulator")
    answer = exact.split("\n", 1)[0].strip()
    if not answer or answer[-1] not in ".!?":
        raise ValueError("selected answer lacks terminal punctuation")
    selected = {
        "source_row": source_row, "turn": turn,
        "prompt": candidate["prompt"], "prompt_length": len(candidate["prompt"]),
        "question": candidate["question"],
        "reference_answer": candidate["reference_answer"],
        "exact_answer": answer, "exact_128": exact,
        "first_32": exact[:32], "first_64": exact[:64],
        "score": candidate["score"],
        "selection": (
            "Held-out TinyChat question selected after reviewing the generated answer."
        ),
        "exact_coherent_heldout_answer": True,
    }
    _write_json(output / "selected-demo.json", selected)
    report_path = output / "training-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["quality_gate"]["heldout_punctuation_demo_selected"] = True
    _write_json(report_path, report)
    _refresh_release_documents(output)
    return selected


def _write_demo(output: Path, selected: dict[str, Any]) -> None:
    text = f"""# FactorioLM Run 7P punctuation demo

## Input

```text
{selected['prompt']}
```

## Exact answer

```text
{selected['exact_answer']}
```

## Exact first 128 generated characters

```text
{selected['exact_128']}
```

Held-out TinyChat question, generated with greedy decoding from the integer model.
"""
    (output / "DEMO.md").write_text(text, encoding="utf-8")


def _refresh_release_documents(output: Path) -> None:
    """Synchronize reports after semantic selection."""

    report_path = output / "training-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    legacy = report["legacy_run7_letter_space_comparison"]
    if "letter_space_answer_accuracy" not in legacy:
        legacy["letter_space_answer_accuracy"] = {
            "run7": legacy["run7"]["answer_characters"]["accuracy"],
            "run7p": legacy["run7p"]["answer_characters"]["accuracy"],
        }
    _write_json(report_path, report)
    evaluation_path = output / "evaluation-report.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation["metrics"] = report["metrics"]
    evaluation["legacy_run7_letter_space_comparison"] = legacy
    evaluation["quality_gate"] = report["quality_gate"]
    _write_json(evaluation_path, evaluation)
    selected_path = output / "selected-demo.json"
    selected = (json.loads(selected_path.read_text(encoding="utf-8"))
                if selected_path.exists() else None)
    mining_path = output / "heldout-punctuation-mining.json"
    if mining_path.exists() and selected is not None:
        mining = json.loads(mining_path.read_text(encoding="utf-8"))
        mining["selection_status"] = "complete after manual semantic review"
        mining["selected_demo"] = {
            key: selected[key] for key in (
                "source_row", "turn", "prompt", "exact_answer", "score")
        }
        _write_json(mining_path, mining)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "run": "7P",
        "checkpoint": str(output / "factorio-lm-tinychat-punctuation-exact.json"),
        "dataset_report": str(output / "dataset-report.json"),
        "training_report": str(report_path),
        "evaluation_report": str(evaluation_path),
        "generation_samples": str(output / "generation-samples.txt"),
        "heldout_mining": str(mining_path) if mining_path.exists() else None,
        "selected_demo": str(selected_path) if selected is not None else None,
    })
    if selected is not None:
        manifest["demo_prompt"] = selected["prompt"]
        _write_demo(output, selected)
    _write_json(manifest_path, manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser("data", help="build the punctuation corpus")
    data.add_argument("--config", default="configs/run7p-punctuation.json")
    data.add_argument("--output", default=str(RUN7P_OUTPUT))
    train = commands.add_parser("train", help="run the complete Run 7P recipe")
    train.add_argument("--config", default="configs/run7p-punctuation.json")
    train.add_argument("--output", default=str(RUN7P_OUTPUT))
    mine = commands.add_parser("mine", help="rank held-out punctuation demos")
    mine.add_argument("--config", default="configs/run7p-punctuation.json")
    mine.add_argument("--output", default=str(RUN7P_OUTPUT))
    expanded = commands.add_parser(
        "mine-expanded", help="rank the complete held-out split semantically")
    expanded.add_argument("--config", default="configs/run7p-punctuation.json")
    expanded.add_argument("--output", default=str(RUN7P_OUTPUT))
    training_patterns = commands.add_parser(
        "mine-training-patterns",
        help="diagnose promising prompt families in the training split")
    training_patterns.add_argument(
        "--config", default="configs/run7p-punctuation.json")
    training_patterns.add_argument("--output", default=str(RUN7P_OUTPUT))
    diverse = commands.add_parser(
        "mine-diverse", help="search prominent non-mood themes for an unusual demo")
    diverse.add_argument("--config", default="configs/run7p-punctuation.json")
    diverse.add_argument("--output", default=str(RUN7P_OUTPUT))
    select = commands.add_parser("select", help="record a reviewed held-out demo")
    select.add_argument("source_row", type=int)
    select.add_argument("turn", type=int)
    select.add_argument("--output", default=str(RUN7P_OUTPUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "data":
        config = Run7PConfig.read(arguments.config)
        config.validate()
        prepare_punctuation_corpus(arguments.output, config)
    elif arguments.command == "train":
        train_run7p(config_path=arguments.config, output_dir=arguments.output)
    elif arguments.command == "mine":
        mine_heldout_run7p(config_path=arguments.config, output_dir=arguments.output)
    elif arguments.command == "mine-expanded":
        mine_expanded_run7p(
            config_path=arguments.config, output_dir=arguments.output)
    elif arguments.command == "mine-training-patterns":
        mine_training_patterns_run7p(
            config_path=arguments.config, output_dir=arguments.output)
    elif arguments.command == "mine-diverse":
        mine_diverse_run7p(
            config_path=arguments.config, output_dir=arguments.output)
    elif arguments.command == "select":
        select_demo_run7p(arguments.source_row, arguments.turn,
                          output_dir=arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
