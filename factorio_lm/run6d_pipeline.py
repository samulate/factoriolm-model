"""Audit TinyStories separators, retrain on cleaned text, and search demo prompts."""

from __future__ import annotations

import argparse
from collections import Counter
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
    Run4Config,
    TokenCorpus,
    _write_json,
    create_float_model,
    evaluate_exact,
    evaluate_torch,
    prepare_corpus,
    train_stage,
)
from .run5_pipeline import Run5Config, export_integer_student, train_student
from .run6_pipeline import (
    GENERAL_PROMPTS,
    generation_quality_metrics,
    generate_incremental_scored,
    generate_incremental_scored_ragged,
    generate_prompt_set,
    load_int4_run5_mask,
    mask_summary,
    validate_incremental_integer,
)
from .vocabulary import Vocabulary


RUN6D_OUTPUT = Path("generated/run6d")
TRAIN_SOURCE = Path("/private/tmp/TinyStoriesV2-GPT4-train-prefix.txt")
VALIDATION_SOURCE = Path("/private/tmp/TinyStoriesV2-GPT4-valid-prefix.txt")
RUN4_TRAIN_TEXT = Path("generated/run4/tinystories-v2-gpt4-train.txt")
RUN4_VALIDATION_TEXT = Path("generated/run4/tinystories-v2-gpt4-validation.txt")
RUN4_TRAIN_TOKENS = Path("generated/run4/tinystories-v2-gpt4-train.bin")
RUN4_VALIDATION_TOKENS = Path("generated/run4/tinystories-v2-gpt4-validation.bin")
RUN6D_FLOAT = RUN6D_OUTPUT / "factorio-lm-cleaned-float.pt"
RUN6D_INT4 = RUN6D_OUTPUT / "factorio-lm-cleaned-int4.pt"
RUN6D_EXACT = RUN6D_OUTPUT / "factorio-lm-cleaned-exact.json"

MARKERS = (
    "endoftext",
    "<|endoftext|>",
    "<endoftext>",
    "end of text",
    "ENDOFTEXT",
)
_CONTEXT_MARKER = re.compile(
    r"<\|endoftext\|>|<endoftext>|end of text|endoftext", re.IGNORECASE
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _marker_counts(text: str) -> dict[str, dict[str, int]]:
    lowered = text.lower()
    return {
        marker: {
            "case_sensitive": text.count(marker),
            "case_insensitive": lowered.count(marker.lower()),
        }
        for marker in MARKERS
    }


def _marker_contexts(text: str, *, limit: int = 5, radius: int = 100) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in _CONTEXT_MARKER.finditer(text):
        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)
        rows.append({
            "matched": match.group(0),
            "character_offset": match.start(),
            "context": text[start:end],
        })
        if len(rows) == limit:
            break
    return rows


def _text_stage(path: Path, *, representation: str) -> dict[str, Any]:
    raw = path.read_bytes()
    text = raw.decode("utf-8", errors="ignore")
    return {
        "path": str(path),
        "representation": representation,
        "bytes": len(raw),
        "characters": len(text),
        "sha256": _sha256_bytes(raw),
        "counts": _marker_counts(text),
        "example_contexts": _marker_contexts(text),
    }


def _token_stage(path: Path, *, representation: str) -> dict[str, Any]:
    raw = path.read_bytes()
    text = Vocabulary().decode(raw)
    return {
        "path": str(path),
        "representation": representation,
        "tokens": len(raw),
        "decoded_characters": len(text),
        "sha256": _sha256_bytes(raw),
        "counts_after_decoding": _marker_counts(text),
        "example_contexts_after_decoding": _marker_contexts(text),
    }


def audit_endoftext(output_dir: str | Path = RUN6D_OUTPUT) -> dict[str, Any]:
    """Audit the literal separator artifact at every materialized corpus stage."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for path in (
        TRAIN_SOURCE, VALIDATION_SOURCE, RUN4_TRAIN_TEXT,
        RUN4_VALIDATION_TEXT, RUN4_TRAIN_TOKENS, RUN4_VALIDATION_TOKENS,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    original_train = _text_stage(
        TRAIN_SOURCE, representation="cached official TinyStories text prefix")
    original_validation = _text_stage(
        VALIDATION_SOURCE, representation="cached official TinyStories text prefix")
    stages: dict[str, Any] = {
        "A_original_cached_tinystories": {
            "train": original_train,
            "validation": original_validation,
        },
        "B_after_loading_and_parsing": {
            "description": (
                "Run 4 reads each cached UTF-8 text file directly with errors=ignore; "
                "there is no record parser or separator transform before normalization."
            ),
            "byte_and_character_identical_to_stage_A": True,
            "train": original_train,
            "validation": original_validation,
        },
        "C_normalized_training_text_used_by_runs_4_5_6": _text_stage(
            RUN4_TRAIN_TEXT,
            representation="selected normalized ASCII training characters"),
        "D_normalized_validation_text_used_by_runs_4_5_6": _text_stage(
            RUN4_VALIDATION_TEXT,
            representation="selected normalized ASCII validation characters"),
        "E_materialized_training_token_stream": _token_stage(
            RUN4_TRAIN_TOKENS,
            representation="one vocabulary ID per selected normalized character"),
        "E_materialized_validation_token_stream": _token_stage(
            RUN4_VALIDATION_TOKENS,
            representation="one vocabulary ID per selected normalized character"),
        "E_training_windows": {
            "materialized": False,
            "description": (
                "Individual 64-character windows are sampled contiguously from the "
                "materialized token stream at training time; therefore their source "
                "occurrence counts are the decoded token-stream counts above."
            ),
        },
    }
    clean_train_text = output / "tinystories-v2-gpt4-train.txt"
    clean_validation_text = output / "tinystories-v2-gpt4-validation.txt"
    clean_train_tokens = output / "tinystories-v2-gpt4-train.bin"
    clean_validation_tokens = output / "tinystories-v2-gpt4-validation.bin"
    if all(path.exists() for path in (
            clean_train_text, clean_validation_text,
            clean_train_tokens, clean_validation_tokens)):
        stages["F_cleaned_normalized_training_text"] = _text_stage(
            clean_train_text, representation="whole-token separator fix")
        stages["F_cleaned_normalized_validation_text"] = _text_stage(
            clean_validation_text, representation="whole-token separator fix")
        stages["F_cleaned_training_token_stream"] = _token_stage(
            clean_train_tokens, representation="whole-token separator fix")
        stages["F_cleaned_validation_token_stream"] = _token_stage(
            clean_validation_tokens, representation="whole-token separator fix")

    train_source_markers = original_train["counts"]["<|endoftext|>"]["case_sensitive"]
    train_normalized_literals = stages[
        "C_normalized_training_text_used_by_runs_4_5_6"]["counts"][
            "endoftext"]["case_sensitive"]
    validation_source_markers = original_validation[
        "counts"]["<|endoftext|>"]["case_sensitive"]
    validation_normalized_literals = stages[
        "D_normalized_validation_text_used_by_runs_4_5_6"]["counts"][
            "endoftext"]["case_sensitive"]
    report = {
        "run": "6D",
        "markers_audited": list(MARKERS),
        "stages": stages,
        "finding": {
            "preprocessing_bug": True,
            "exact_source": (
                "The official cached text uses <|endoftext|> between stories. "
                "The former character normalizer mapped '<', '|', and '>' to spaces "
                "but retained the ASCII letters, producing literal 'endoftext'."
            ),
            "observed_selected_training_collapses": train_normalized_literals,
            "observed_selected_validation_collapses": validation_normalized_literals,
            "cached_train_separator_tokens": train_source_markers,
            "cached_validation_separator_tokens": validation_source_markers,
            "exact_fix": (
                "Recognize <|endoftext|> and <endoftext> case-insensitively as "
                "whole document-separator tokens before character normalization, and "
                "map each complete separator to a newline story boundary."
            ),
            "runtime_blacklist_or_argmax_change": False,
            "retraining_required_by_run6d_gate": True,
        },
    }
    _write_json(output / "endoftext-audit.json", report)
    return report


def rebuild_cleaned_corpus(output_dir: str | Path = RUN6D_OUTPUT) -> dict[str, Any]:
    """Rebuild the same deterministic 24M/2M selection with fixed normalization."""

    report = prepare_corpus(
        TRAIN_SOURCE,
        VALIDATION_SOURCE,
        output_dir,
        Run4Config(),
    )
    audit_endoftext(output_dir)
    return report


def _clean_paths(output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    return (
        output / "tinystories-v2-gpt4-train.bin",
        output / "tinystories-v2-gpt4-validation.bin",
    )


def train_clean_float_teacher(output_dir: str | Path = RUN6D_OUTPUT) -> dict[str, Any]:
    """Adapt the existing strong float teacher to the corrected corpus."""

    import torch

    output = Path(output_dir)
    train_path, validation_path = _clean_paths(output)
    config = replace(
        Run4Config(),
        float_steps=1_000,
        float_learning_rate=0.0005,
        warmup_steps=100,
        evaluation_interval=125,
        device="cpu",
    )
    model = create_float_model(config)
    source = torch.load(
        "generated/run4/factorio-lm-quality-float.pt",
        map_location=config.device,
        weights_only=False,
    )
    model.load_state_dict(source["model"])
    checkpoint_path = output / RUN6D_FLOAT.name
    training = train_stage(
        model,
        TokenCorpus(train_path),
        TokenCorpus(validation_path),
        config,
        stage="float",
        device=config.device,
        checkpoint_path=checkpoint_path,
    )
    report = {
        "stage": "cleaned-data float-teacher adaptation",
        "initialization": "generated/run4/factorio-lm-quality-float.pt",
        "configuration": asdict(config),
        "training": training,
    }
    _write_json(output / "cleaned-float-training.json", report)
    return report


def load_clean_float_teacher(output_dir: str | Path = RUN6D_OUTPUT):
    import torch

    output = Path(output_dir)
    config = replace(Run4Config(), device="cpu")
    model = create_float_model(config)
    state = torch.load(
        output / RUN6D_FLOAT.name,
        map_location=config.device,
        weights_only=False,
    )
    model.load_state_dict(state["model"])
    return model.eval()


def train_clean_int4(output_dir: str | Path = RUN6D_OUTPUT) -> dict[str, Any]:
    """Warm-start from Run 6 int4 and fine-tune on cleaned text."""

    output = Path(output_dir)
    train_path, validation_path = _clean_paths(output)
    config = Run5Config.read("configs/run6-int4.json")
    teacher = load_clean_float_teacher(output)
    _, _, _, model = load_int4_run5_mask()
    checkpoint_path = output / RUN6D_INT4.name
    training = train_student(
        model,
        teacher,
        TokenCorpus(train_path),
        TokenCorpus(validation_path),
        config,
        stage="integer",
        output_path=checkpoint_path,
    )
    report = {
        "stage": "cleaned-data scaled-int4 recovery",
        "initialization": "generated/run6/factorio-lm-int4-run5-mask.pt",
        "teacher": str(output / RUN6D_FLOAT.name),
        "fixed_run6_mask": True,
        "configuration": asdict(config),
        "training": training,
    }
    _write_json(output / "cleaned-int4-training.json", report)
    return report


def load_clean_int4(output_dir: str | Path = RUN6D_OUTPUT):
    import torch

    output = Path(output_dir)
    config, teacher, sparse, model = load_int4_run5_mask()
    state = torch.load(
        output / RUN6D_INT4.name,
        map_location=config.device,
        weights_only=False,
    )
    model.load_state_dict(state["model"])
    model.masks = {
        name: value.to(config.device) for name, value in state["masks"].items()
    }
    model.denominators = dict(state["denominators"])
    return config, teacher.eval(), sparse.eval(), model.eval()


def _calibrated_metrics(model, corpus: TokenCorpus, config: Run5Config) -> dict[str, Any]:
    common = dict(
        device=config.device,
        length=64,
        characters=config.evaluation_characters,
        batch_size=config.batch_size,
    )
    raw = evaluate_torch(model, corpus, **common)
    temperatures = (0.75, 1.0, 1.125, 1.25, 1.375, 1.5, 1.75, 2.0)
    calibration = {
        str(temperature): evaluate_torch(
            model,
            corpus,
            **common,
            optimization_temperature=temperature,
        )["optimization_cross_entropy"]
        for temperature in temperatures
    }
    best = min(calibration, key=calibration.get)
    return {
        **raw,
        "calibration_cross_entropy": calibration,
        "best_calibration_temperature": float(best),
        "best_calibrated_cross_entropy": calibration[best],
        "best_calibrated_perplexity": math.exp(calibration[best]),
    }


def evaluate_clean_model(output_dir: str | Path = RUN6D_OUTPUT) -> dict[str, Any]:
    """Compare Run 6 and cleaned int4 fairly on the corrected validation set."""

    output = Path(output_dir)
    _, validation_path = _clean_paths(output)
    validation = TokenCorpus(validation_path)
    validation_bytes = validation_path.read_bytes()
    config, _, _, baseline = load_int4_run5_mask()
    _, _, _, cleaned = load_clean_int4(output)
    teacher = load_clean_float_teacher(output)
    lexicon = {
        word for word, count in _word_counts(Vocabulary().decode(validation_bytes)).items()
        if count >= 2
    }
    baseline_metrics = _calibrated_metrics(baseline, validation, config)
    cleaned_metrics = _calibrated_metrics(cleaned, validation, config)
    teacher_metrics = evaluate_torch(
        teacher,
        validation,
        device=config.device,
        length=64,
        characters=config.evaluation_characters,
        batch_size=config.batch_size,
    )
    checkpoint = export_integer_student(
        cleaned,
        config,
        {
            "run": 6,
            "subrun": "6D",
            "training_representation": "scaled-int4",
            "corpus_fix": "whole-token TinyStories document separators",
            "initialization": "Run 6 selected exact/QAT model",
        },
    )
    checkpoint.write(output / RUN6D_EXACT.name)
    validate_incremental_integer(
        cleaned, checkpoint, count=128, device=config.device)
    independent = evaluate_exact(checkpoint, validation_bytes, characters=1_024)
    prompts = tuple(dict.fromkeys((*GENERAL_PROMPTS, *PREVIOUS_DEMO_PROMPTS)))
    baseline_samples = generate_prompt_set(
        baseline, prompts, 256, device=config.device)
    cleaned_samples = generate_prompt_set(
        cleaned, prompts, 256, device=config.device)

    def sample_rows(samples: dict[str, str]) -> dict[str, Any]:
        return {
            prompt: {
                "generated": generated,
                "quality": generation_quality_metrics(generated, lexicon),
            }
            for prompt, generated in samples.items()
        }

    baseline_rows = sample_rows(baseline_samples)
    cleaned_rows = sample_rows(cleaned_samples)

    def aggregate(rows: dict[str, Any]) -> dict[str, float]:
        return {
            key: sum(row["quality"][key] for row in rows.values()) / len(rows)
            for key in (
                "dictionary_like_word_fraction",
                "unique_word_fraction",
                "adjacent_repeated_word_fraction",
                "repeated_fourgram_fraction",
            )
        }

    published = json.loads(
        Path("generated/run6/evaluation-report.json").read_text(encoding="utf-8"))
    report = {
        "run": "6D",
        "evaluation_corpus": str(validation_path),
        "characters": config.evaluation_characters,
        "published_run6_metrics_on_original_validation": {
            "cross_entropy": 1.4909652769565582,
            "calibrated_cross_entropy": 1.4749764502048492,
            "accuracy": 0.5462646484375,
        },
        "fair_cleaned_validation_comparison": {
            "run6_baseline": baseline_metrics,
            "cleaned_data_model": cleaned_metrics,
            "cleaned_float_teacher": teacher_metrics,
        },
        "independent_exact_int32_1024": independent,
        "exact_checkpoint": str(output / RUN6D_EXACT.name),
        "exact_recurrent_matches_independent_emulator_128": True,
        "mask": mask_summary(cleaned.masks),
        "generation_quality": {
            "run6_baseline": aggregate(baseline_rows),
            "cleaned_data_model": aggregate(cleaned_rows),
        },
        "samples": {
            "run6_baseline": baseline_rows,
            "cleaned_data_model": cleaned_rows,
        },
        "replacement_gate": {
            "accuracy_within_two_percentage_points": (
                cleaned_metrics["accuracy"] >= baseline_metrics["accuracy"] - 0.02
            ),
            "cross_entropy_not_materially_worse": (
                cleaned_metrics["cross_entropy"] <= baseline_metrics["cross_entropy"] + 0.08
            ),
        },
        "source_run6_evaluation_path": "generated/run6/evaluation-report.json",
        "source_run6_model_preserved": True,
        "source_run6_report_loaded": bool(published),
    }
    report["replacement_gate"]["passed"] = all(
        report["replacement_gate"][key]
        for key in (
            "accuracy_within_two_percentage_points",
            "cross_entropy_not_materially_worse",
        )
    )
    _write_json(output / "cleaned-model-evaluation.json", report)
    return report


def _word_counts(text: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for word in re.findall(r"[A-Za-z]+", text.lower()):
        result[word] = result.get(word, 0) + 1
    return result


PREVIOUS_DEMO_PROMPTS = (
    "He liked to jump run and ",
    "Tim went to the store with his mom to ",
    "He loved to play with it every ",
    "He said to his friend ",
    "Tom thanked the bird for helping them and they all ",
    "Tim was very eager ",
    "He saw a large mud ",
    "The bird lived in a big tree ",
    "One day Maxs friend a little ",
    "He wanted to find some dinner to ",
)


def heldout_prefix_candidates(
    limit: int,
    *,
    output_dir: str | Path = RUN6D_OUTPUT,
    seed: int = 4104,
) -> list[dict[str, Any]]:
    """Select prefixes from cleaned validation text."""

    text = (Path(output_dir) / "tinystories-v2-gpt4-validation.txt").read_text(
        encoding="ascii"
    )
    pool: list[dict[str, Any]] = []
    seen: set[str] = set()
    character_offset = 0
    for line_index, raw_line in enumerate(text.splitlines(keepends=True)):
        line = raw_line.rstrip("\n")
        if len(line) < 16:
            character_offset += len(raw_line)
            continue
        for match in re.finditer(r" ", line):
            end = match.end()
            if not 15 <= end <= 55:
                continue
            prompt = line[:end]
            if prompt in seen or len(re.findall(r"[A-Za-z]+", prompt)) < 3:
                continue
            if not prompt[-1].isspace() or "endoftext" in prompt.lower():
                continue
            seen.add(prompt)
            pool.append({
                "prompt": prompt,
                "source_split": "cleaned TinyStories validation",
                "source_line_index": line_index,
                "source_character_offset": character_offset,
                "prefix_characters": len(prompt),
            })
        character_offset += len(raw_line)
    generator = random.Random(seed + 760_000)
    generator.shuffle(pool)
    if len(pool) < limit:
        raise ValueError(f"held-out corpus yielded only {len(pool)} prefixes")
    return pool[:limit]


def corpus_prefix_candidates(
    path: str | Path,
    limit: int,
    *,
    source_split: str,
    seed: int,
    contextual: bool = False,
) -> list[dict[str, Any]]:
    """Reservoir-sample line prefixes without materializing the pool.

    Unlike :func:`heldout_prefix_candidates`, this is practical for the much
    larger training split.  A prompt always begins at a normalized TinyStories
    line boundary and ends immediately after a space, so it remains an
    understandable natural prefix rather than a model-crafted suffix.
    """

    text = Path(path).read_text(encoding="ascii")
    generator = random.Random(seed)
    reservoir: list[dict[str, Any]] = []
    seen: set[str] = set()
    eligible = 0
    character_offset = 0
    for line_index, raw_line in enumerate(text.splitlines(keepends=True)):
        line = raw_line.rstrip("\n")
        for match in re.finditer(r" ", line):
            end = match.end()
            global_end = character_offset + end
            if contextual:
                search_start = max(0, global_end - 64)
                boundary = text.find("\n", search_start, global_end - 14)
                prompt_start = boundary + 1 if boundary >= 0 else character_offset
                prompt = text[prompt_start:global_end]
            else:
                prompt = line[:end]
            reference = text[global_end:global_end + 128]
            if (
                not 15 <= end <= 64
                or len(reference) != 128
                or prompt in seen
                or len(re.findall(r"[A-Za-z]+", prompt)) < 3
                or "endoftext" in prompt.lower()
            ):
                continue
            seen.add(prompt)
            eligible += 1
            row = {
                "prompt": prompt,
                "source_split": source_split,
                "source_line_index": line_index,
                "source_character_offset": character_offset,
                "prefix_characters": len(prompt),
                "contextual_multiline_prompt": contextual,
                "natural_reference_128": reference,
            }
            if len(reservoir) < limit:
                reservoir.append(row)
            else:
                replacement = generator.randrange(eligible)
                if replacement < limit:
                    reservoir[replacement] = row
        character_offset += len(raw_line)
    if len(reservoir) < limit:
        raise ValueError(f"corpus yielded only {len(reservoir)} prompt prefixes")
    return reservoir


def _corpus_word_ngrams(path: str | Path):
    """Count normalized word n-grams used only to rank exact generations."""

    unigrams: Counter[str] = Counter()
    bigrams: Counter[tuple[str, str]] = Counter()
    trigrams: Counter[tuple[str, str, str]] = Counter()
    for line in Path(path).read_text(encoding="ascii").splitlines():
        words = ["<B>", *re.findall(r"[A-Za-z]+", line.lower()), "<E>"]
        unigrams.update(words)
        bigrams.update(zip(words, words[1:]))
        trigrams.update(zip(words, words[1:], words[2:]))
    return unigrams, bigrams, trigrams


def coherent_quality_metrics(
    prompt: str,
    generated: str,
    ngrams,
) -> dict[str, Any]:
    """Whole-continuation diagnostics resistant to fluent-looking loops."""

    unigrams, bigrams, trigrams = ngrams
    prompt_words = re.findall(r"[A-Za-z]+", prompt.lower())
    generated_words = re.findall(r"[A-Za-z]+", generated.lower())
    words = [*prompt_words[-2:], *generated_words]
    generated_start = len(words) - len(generated_words)
    seen_bigrams = 0
    seen_trigrams = 0
    conditional_log_probability = 0.0
    evaluated = 0
    total_unigrams = sum(unigrams.values())
    vocabulary_size = len(unigrams)
    for index in range(max(2, generated_start), len(words)):
        first, second, third = words[index - 2:index + 1]
        trigram_count = trigrams[(first, second, third)]
        prefix_count = bigrams[(first, second)]
        bigram_count = bigrams[(second, third)]
        second_count = unigrams[second]
        seen_bigrams += int(bigram_count > 0)
        seen_trigrams += int(trigram_count > 0)
        evaluated += 1
        if trigram_count:
            probability = (trigram_count + 0.2) / (prefix_count + 20.0)
        elif bigram_count:
            probability = (bigram_count + 0.1) / (second_count + 20.0)
            probability *= math.exp(-1.0)
        else:
            probability = (
                (unigrams[third] + 0.1) /
                (total_unigrams + vocabulary_size)
            ) * math.exp(-3.0)
        conditional_log_probability += math.log(max(probability, 1.0e-12))

    word_bigrams = list(zip(generated_words, generated_words[1:]))
    word_trigrams = list(zip(
        generated_words, generated_words[1:], generated_words[2:]))
    repeated_bigrams = len(word_bigrams) - len(set(word_bigrams))
    repeated_trigrams = len(word_trigrams) - len(set(word_trigrams))
    adjacent_duplicates = sum(
        first == second
        for first, second in zip(generated_words, generated_words[1:])
    )
    syntax_smells = len(re.findall(
        r"\b(?:and but|but and|he with|she but|they but|was fun was|"
        r"wanted to play with (?:the|his|her) \w+ wanted to play)\b",
        generated.lower(),
    ))
    return {
        "words": len(generated_words),
        "unique_word_fraction": (
            len(set(generated_words)) / max(1, len(generated_words))
        ),
        "seen_corpus_bigram_fraction": seen_bigrams / max(1, evaluated),
        "seen_corpus_trigram_fraction": seen_trigrams / max(1, evaluated),
        "mean_corpus_log_probability": (
            conditional_log_probability / max(1, evaluated)
        ),
        "adjacent_duplicate_words": adjacent_duplicates,
        "repeated_word_bigrams": repeated_bigrams,
        "repeated_word_trigrams": repeated_trigrams,
        "syntax_smells": syntax_smells,
        "newlines": generated.count("\n"),
    }


def coherent_quality_score(metrics: dict[str, Any]) -> float:
    """Rank whole 128-character readability; human review remains final."""

    return (
        2.0 * metrics["seen_corpus_bigram_fraction"]
        + 5.0 * metrics["seen_corpus_trigram_fraction"]
        + 0.65 * metrics["mean_corpus_log_probability"]
        + 1.5 * metrics["unique_word_fraction"]
        - 3.0 * metrics["adjacent_duplicate_words"]
        - 1.6 * metrics["repeated_word_bigrams"]
        - 2.3 * metrics["repeated_word_trigrams"]
        - 2.5 * metrics["syntax_smells"]
        - 0.25 * max(0, metrics["newlines"] - 4)
    )


def _reference_prefix_match(generated: str, reference: str) -> int:
    result = 0
    for actual, expected in zip(generated, reference):
        if actual != expected:
            break
        result += 1
    return result


def early_quality_metrics(text: str, lexicon: set[str]) -> dict[str, Any]:
    """Diagnostics tuned to what a viewer sees during the early live reveal."""

    words = re.findall(r"[A-Za-z]+", text.lower())
    plausible = sum(word in lexicon for word in words)
    adjacent_duplicates = sum(a == b for a, b in zip(words, words[1:]))

    def repeated_word_ngram_fraction(size: int) -> float:
        grams = [tuple(words[index:index + size])
                 for index in range(max(0, len(words) - size + 1))]
        return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)

    maximum_run = 0
    run = 0
    previous = None
    for character in text:
        if character == previous:
            run += 1
        else:
            previous = character
            run = 1
        maximum_run = max(maximum_run, run)
    return {
        "characters": len(text),
        "words": len(words),
        "plausible_words": plausible,
        "dictionary_like_word_fraction": plausible / max(1, len(words)),
        "unique_word_fraction": len(set(words)) / max(1, len(words)),
        "adjacent_duplicate_words": adjacent_duplicates,
        "adjacent_repeated_word_fraction": (
            adjacent_duplicates / max(1, len(words) - 1)
        ),
        "repeated_word_bigram_fraction": repeated_word_ngram_fraction(2),
        "repeated_word_trigram_fraction": repeated_word_ngram_fraction(3),
        "maximum_repeated_character_run": maximum_run,
        "space_fraction": text.count(" ") / max(1, len(text)),
        "newlines": text.count("\n"),
        "malformed_newline_runs": len(re.findall(r"\n{2,}", text)),
        "double_spaces": text.count("  "),
        "contains_endoftext": "endoftext" in text.lower(),
    }


def early_window_score(metrics: dict[str, Any]) -> float:
    """Score one horizon; larger horizon weights are applied separately."""

    return (
        4.5 * metrics["dictionary_like_word_fraction"]
        + 1.15 * metrics["unique_word_fraction"]
        + 0.10 * min(metrics["plausible_words"], 10)
        - 7.0 * metrics["adjacent_repeated_word_fraction"]
        - 4.0 * metrics["repeated_word_bigram_fraction"]
        - 2.5 * metrics["repeated_word_trigram_fraction"]
        - 1.5 * max(0, metrics["maximum_repeated_character_run"] - 2)
        - 1.75 * metrics["malformed_newline_runs"]
        - 0.35 * metrics["double_spaces"]
        - 1.25 * abs(metrics["space_fraction"] - 0.17)
        - 100.0 * int(metrics["contains_endoftext"])
        + 0.12 * min(metrics["newlines"], 1)
    )


def hard_early_quality_pass(generated: str, lexicon: set[str]) -> tuple[bool, list[str]]:
    """Reject visibly degenerate first-32/48 behavior before scalar ranking."""

    first32 = early_quality_metrics(generated[:32], lexicon)
    first48 = early_quality_metrics(generated[:48], lexicon)
    reasons: list[str] = []
    if first48["plausible_words"] < 4:
        reasons.append("fewer than four plausible words in first 48")
    if first32["maximum_repeated_character_run"] > 2:
        reasons.append("repeated-character run in first 32")
    if first48["adjacent_duplicate_words"]:
        reasons.append("adjacent duplicate word in first 48")
    if first48["contains_endoftext"]:
        reasons.append("endoftext in first 48")
    if first48["repeated_word_bigram_fraction"] > 0.20:
        reasons.append("short word-loop in first 48")
    if first48["malformed_newline_runs"]:
        reasons.append("malformed newline run in first 48")
    if first48["newlines"] > 2:
        reasons.append("too many newlines in first 48")
    if first48["double_spaces"]:
        reasons.append("double space in first 48")
    return not reasons, reasons


def _score_generation(generated: str, lexicon: set[str], *, include_256: bool) -> dict[str, Any]:
    horizons = (32, 64, 128, 256) if include_256 else (32, 64, 128)
    weights = {32: 5.0, 64: 3.0, 128: 1.0, 256: 0.25}
    metrics = {
        str(horizon): early_quality_metrics(generated[:horizon], lexicon)
        for horizon in horizons
    }
    scores = {
        str(horizon): early_window_score(metrics[str(horizon)])
        for horizon in horizons
    }
    passed, rejection_reasons = hard_early_quality_pass(generated, lexicon)
    return {
        "first_32": generated[:32],
        "first_64": generated[:64],
        "first_128": generated[:128],
        **({"generated_256": generated[:256]} if include_256 else {}),
        "horizon_metrics": metrics,
        "horizon_scores": scores,
        "demo_score": sum(weights[horizon] * scores[str(horizon)] for horizon in horizons),
        "hard_early_quality_pass": passed,
        "hard_filter_reasons": rejection_reasons,
    }


def _clean_validation_lexicon(output_dir: str | Path) -> set[str]:
    text = (Path(output_dir) / "tinystories-v2-gpt4-validation.txt").read_text(
        encoding="ascii"
    )
    return {word for word, count in _word_counts(text).items() if count >= 2}


def _generate_rows(
    model,
    config: Run5Config,
    rows: list[dict[str, Any]],
    count: int,
    *,
    batch_size: int = 256,
):
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(len(row["prompt"]), []).append(row)
    completed = 0
    for length, group in sorted(grouped.items()):
        for first in range(0, len(group), batch_size):
            chunk = group[first:first + batch_size]
            generations = generate_incremental_scored(
                model,
                [row["prompt"] for row in chunk],
                count,
                device=config.device,
            )
            for row, generated in zip(chunk, generations, strict=True):
                yield row, generated
            completed += len(chunk)
            if completed % 1_000 < len(chunk):
                print(f"exact prompt mining: {completed:,} trajectories", flush=True)


def _generate_rows_ragged(
    model,
    config: Run5Config,
    rows: list[dict[str, Any]],
    count: int,
    *,
    batch_size: int,
):
    completed = 0
    for first in range(0, len(rows), batch_size):
        chunk = rows[first:first + batch_size]
        generations = generate_incremental_scored_ragged(
            model,
            [row["prompt"] for row in chunk],
            count,
            device=config.device,
        )
        for row, generated in zip(chunk, generations, strict=True):
            yield row, generated
        completed += len(chunk)
        print(f"exact coherent mining: {completed:,} trajectories", flush=True)


def mine_coherent_prompts(
    *,
    validation_candidates: int = 50_000,
    training_candidates: int = 100_000,
    finalists: int = 5_000,
    output_dir: str | Path = RUN6D_OUTPUT,
    contextual: bool = False,
) -> dict[str, Any]:
    """Search natural corpus prefixes for a coherent full 128-character demo.

    The inexpensive discovery pass generates 64 exact characters for a broad
    deterministic sample.  Only non-degenerate finalists receive the full 128
    character pass.  Training prompts are allowed because this is explicitly a
    showcase search, but provenance is retained and held-out prompts win ties.
    """

    output = Path(output_dir)
    train_path = output / "tinystories-v2-gpt4-train.txt"
    validation_path = output / "tinystories-v2-gpt4-validation.txt"
    config, _, _, model = load_clean_int4(output)
    checkpoint = ExactCheckpoint.read(output / RUN6D_EXACT.name)
    validate_incremental_integer(model, checkpoint, count=128, device=config.device)
    ngrams = _corpus_word_ngrams(train_path)
    lexicon = _clean_validation_lexicon(output)
    sources = [
        *corpus_prefix_candidates(
            validation_path,
            validation_candidates,
            source_split="cleaned TinyStories validation",
            seed=config.seed + 61_001,
            contextual=contextual,
        ),
        *corpus_prefix_candidates(
            train_path,
            training_candidates,
            source_split="cleaned TinyStories training",
            seed=config.seed + 61_002,
            contextual=contextual,
        ),
    ]

    discovery: list[dict[str, Any]] = []
    rejected = 0
    for source, result in _generate_rows_ragged(
            model, config, sources, 64, batch_size=8_192):
        generated = result["generated"]
        early = early_quality_metrics(generated, lexicon)
        coherent = coherent_quality_metrics(source["prompt"], generated, ngrams)
        natural_match = _reference_prefix_match(
            generated, source["natural_reference_128"])
        acceptable = (
            not early["contains_endoftext"]
            and not early["adjacent_duplicate_words"]
            and early["maximum_repeated_character_run"] <= 2
            and early["dictionary_like_word_fraction"] >= 0.94
            and coherent["repeated_word_bigrams"] <= 1
            and coherent["repeated_word_trigrams"] == 0
            and coherent["syntax_smells"] == 0
        )
        if not acceptable:
            rejected += 1
            continue
        discovery.append({
            **source,
            **result,
            "discovery_64": generated,
            "discovery_metrics": coherent,
            "natural_reference_prefix_match": natural_match,
            "discovery_score": (
                coherent_quality_score(coherent)
                + early_window_score(early)
                + 0.08 * min(natural_match, 32)
            ),
        })
    discovery.sort(key=lambda row: (
        -row["discovery_score"],
        row["source_split"] != "cleaned TinyStories validation",
        row["prompt"],
    ))
    if len(discovery) < finalists:
        raise AssertionError(
            f"only {len(discovery)} coherent discovery rows for {finalists} finalists")

    finalist_sources = [
        {key: row[key] for key in (
            "prompt", "source_split", "source_line_index",
            "source_character_offset", "prefix_characters",
            "natural_reference_128", "natural_reference_prefix_match",
            "contextual_multiline_prompt", "discovery_64", "discovery_score",
        )}
        for row in discovery[:finalists]
    ]
    completed: list[dict[str, Any]] = []
    for source, result in _generate_rows_ragged(
            model, config, finalist_sources, 128, batch_size=4_096):
        generated = result["generated"]
        early = early_quality_metrics(generated, lexicon)
        coherent = coherent_quality_metrics(source["prompt"], generated, ngrams)
        natural_match = _reference_prefix_match(
            generated, source["natural_reference_128"])
        completed.append({
            **source,
            **result,
            "first_32": generated[:32],
            "first_64": generated[:64],
            "first_128": generated,
            "natural_reference_prefix_match": natural_match,
            "coherence_metrics": coherent,
            "early_metrics": early,
            "coherence_score": (
                coherent_quality_score(coherent)
                + early_window_score(early)
                + 0.08 * min(natural_match, 32)
            ),
        })
    completed.sort(key=lambda row: (
        -row["coherence_score"],
        row["source_split"] != "cleaned TinyStories validation",
        row["prompt"],
    ))
    report = {
        "run": (
            "6D contextual coherence search" if contextual
            else "6D coherence search"
        ),
        "model": str(output / RUN6D_EXACT.name),
        "output_modified_or_postprocessed": False,
        "search": {
            "validation_prefixes": validation_candidates,
            "training_prefixes": training_candidates,
            "total_prefixes": len(sources),
            "discovery_generation_characters": 64,
            "discovery_rejected": rejected,
            "full_128_finalists": finalists,
            "contextual_multiline_prompts": contextual,
            "selection_warning": (
                "Training-split prompts may benefit from memorization. Review finalists for relevance."
            ),
        },
        "ranking": (
            "Corpus word-bigram/trigram support plus lexical quality and diversity, "
            "with strong penalties for repeated n-grams, adjacent duplicates, "
            "syntax-smell patterns, and malformed character runs. Natural-reference "
            "prefix matches receive a small bounded bonus."
        ),
        "top_200": completed[:200],
    }
    stem = "contextual" if contextual else "coherent"
    report_path = output / f"prompt-mining-{stem}.json"
    _write_json(report_path, report)
    _write_coherent_finalists_markdown(
        output / f"PROMPT-{stem.upper()}-FINALISTS.md", report)
    return report


def mine_contextual_prompts(
    *,
    output_dir: str | Path = RUN6D_OUTPUT,
) -> dict[str, Any]:
    """Focused follow-up using natural multi-sentence 64-character contexts."""

    return mine_coherent_prompts(
        validation_candidates=20_000,
        training_candidates=40_000,
        finalists=3_000,
        output_dir=output_dir,
        contextual=True,
    )


def _write_coherent_finalists_markdown(
    path: Path,
    report: dict[str, Any],
) -> None:
    lines = [
        "# Run 6D full-128 coherence finalists",
        "",
        (
            f"Tried {report['search']['total_prefixes']:,} corpus prefixes with the "
            f"int32 model, then generated 128 characters for "
            f"{report['search']['full_128_finalists']:,} finalists."
        ),
        "",
        "Output uses greedy decoding. The text below is shown as generated.",
        "",
    ]
    for index, row in enumerate(report["top_200"][:30], 1):
        lines.extend([
            f"## {index}. Score {row['coherence_score']:.4f}",
            "",
            (
                f"Source: {row['source_split']}, line "
                f"{row['source_line_index']}, character offset "
                f"{row['source_character_offset']}"
            ),
            "",
            f"Natural-reference prefix match: {row['natural_reference_prefix_match']} characters",
            "",
            "INPUT:",
            "",
            "```text",
            row["prompt"],
            "```",
            "",
            "EXACT FIRST 128:",
            "",
            "```text",
            row["first_128"],
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def mine_early_prompts(
    *,
    candidates: int = 10_000,
    output_dir: str | Path = RUN6D_OUTPUT,
) -> dict[str, Any]:
    """Mine exact int32 trajectories with first-32/64-dominant ranking."""

    output = Path(output_dir)
    config, _, _, model = load_clean_int4(output)
    checkpoint = ExactCheckpoint.read(output / RUN6D_EXACT.name)
    checkpoint = replace(
        checkpoint,
        metadata={**(checkpoint.metadata or {}), "run": 6, "subrun": "6D"},
    )
    validate_incremental_integer(model, checkpoint, count=128, device=config.device)
    lexicon = _clean_validation_lexicon(output)
    prefix_rows = heldout_prefix_candidates(
        candidates, output_dir=output, seed=config.seed)
    scored: list[dict[str, Any]] = []
    for source, generated in _generate_rows(model, config, prefix_rows, 128):
        text = generated["generated"]
        scored.append({
            **source,
            **generated,
            **_score_generation(text, lexicon, include_256=False),
        })
    passed = [row for row in scored if row["hard_early_quality_pass"]]
    passed.sort(key=lambda row: (-row["demo_score"], row["prompt"]))
    if len(passed) < 60:
        raise AssertionError(f"only {len(passed)} candidates passed the early filter")

    finalists: list[dict[str, Any]] = []
    finalist_sources = [
        {key: row[key] for key in (
            "prompt", "source_split", "source_line_index",
            "source_character_offset", "prefix_characters",
        )}
        for row in passed[:80]
    ]
    for source, generated in _generate_rows(model, config, finalist_sources, 256):
        text = generated["generated"]
        finalists.append({
            **source,
            **generated,
            **_score_generation(text, lexicon, include_256=True),
        })
    finalists.sort(key=lambda row: (-row["demo_score"], row["prompt"]))

    explicit_sources = [
        {
            "prompt": prompt,
            "source_split": "Run 6 demo prompt",
            "source_line_index": None,
            "source_character_offset": None,
            "prefix_characters": len(prompt),
        }
        for prompt in PREVIOUS_DEMO_PROMPTS
    ]
    previous_rows: list[dict[str, Any]] = []
    for source, generated in _generate_rows(model, config, explicit_sources, 256):
        text = generated["generated"]
        previous_rows.append({
            **source,
            **generated,
            **_score_generation(text, lexicon, include_256=True),
        })
    previous_rows.sort(key=lambda row: (-row["demo_score"], row["prompt"]))

    # Preserve all 128-character generations for auditability; only the top 80
    # incur the extra 256-character exact recurrent pass.
    scored.sort(key=lambda row: (-row["demo_score"], row["prompt"]))
    report = {
        "run": "6D",
        "model": str(output / RUN6D_EXACT.name),
        "source": "prefixes from cleaned TinyStories validation text",
        "candidate_count": candidates,
        "hard_filter_pass_count": len(passed),
        "candidate_generation_characters": 128,
        "finalist_generation_characters": 256,
        "scoring": {
            "formula": (
                "5.0*quality(first32) + 3.0*quality(first64) + "
                "1.0*quality(first128) + 0.25*quality(first256)"
            ),
            "early_output_dominates": True,
            "hard_filter": (
                "at least four plausible words in first 48; no endoftext, "
                "adjacent duplicate word, repeated-character run, immediate "
                "word-bigram loop, malformed newline run, or double space"
            ),
        },
        "top_30": finalists[:30],
        "top_80_full_horizon": finalists,
        "required_previous_prompts": previous_rows,
        "all_candidates_128": scored,
    }
    _write_json(output / "prompt-mining-early.json", report)
    _write_prompt_finalists_markdown(output / "PROMPT-FINALISTS.md", report)
    return report


def _display_text(value: str) -> str:
    return value.replace("\n", "<NL>\n")


def _write_prompt_finalists_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Run 6D early-output prompt finalists",
        "",
        f"Evaluated {report['candidate_count']:,} held-out prefixes with the int32 model.",
        "Ranking weights first 32/64/128/256 characters by 5.0/3.0/1.0/0.25.",
        "",
    ]
    if "selected_demo" in report:
        selected = report["selected_demo"]
        lines.extend([
            "## Selected demo",
            "",
            f"INPUT: `{selected['prompt']}`",
            "",
            "FIRST 64:",
            "",
            "```text",
            selected["first_64"],
            "```",
            "",
            selected["selection_reason"].replace("; ", ", "),
            "",
        ])
    for index, row in enumerate(report["top_30"][:20], 1):
        lines.extend([
            f"## {index}. Score {row['demo_score']:.4f}",
            "",
            f"Source line `{row['source_line_index']}`, offset `{row['source_character_offset']}`",
            "",
            "INPUT:",
            "",
            "```text",
            row["prompt"],
            "```",
            "",
            "FIRST 64:",
            "",
            "```text",
            row["first_64"],
            "```",
            "",
        ])
    lines.extend(["# Earlier demo prompts", ""])
    for row in report["required_previous_prompts"]:
        lines.extend([
            f"## {row['prompt']!r} — score {row['demo_score']:.4f}",
            "",
            "```text",
            row["first_128"],
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def finalize_demo(
    prompt: str = "Sam decided to ",
    *,
    output_dir: str | Path = RUN6D_OUTPUT,
) -> dict[str, Any]:
    """Record a finalist's output and export the exact checkpoint."""

    output = Path(output_dir)
    mining_path = output / "prompt-mining-early.json"
    mining = json.loads(mining_path.read_text(encoding="utf-8"))
    source = next(
        (row for row in mining["all_candidates_128"] if row["prompt"] == prompt),
        None,
    )
    if source is None:
        raise ValueError("selected prompt is not one of the mined held-out prefixes")
    config, _, _, model = load_clean_int4(output)
    generated_row = generate_incremental_scored(
        model, [prompt], 256, device=config.device)[0]
    generated = generated_row["generated"]
    lexicon = _clean_validation_lexicon(output)
    selection = {
        **{key: source[key] for key in (
            "prompt", "source_split", "source_line_index",
            "source_character_offset", "prefix_characters",
        )},
        **generated_row,
        **_score_generation(generated, lexicon, include_256=True),
        "selection_method": "manual review of ranked finalists",
        "selection_reason": (
            f"Selected from {len(mining['all_candidates_128']):,} held-out candidates, "
            f"prompt length {len(prompt)}."
        ),
        "output_modified_or_postprocessed": False,
    }
    checkpoint = ExactCheckpoint.read(output / RUN6D_EXACT.name)
    checkpoint = replace(
        checkpoint,
        metadata={**(checkpoint.metadata or {}), "run": 6, "subrun": "6D"},
    )
    independent = ExactFactorioLM(checkpoint).generate(prompt, 256)[0]
    if independent != generated:
        raise AssertionError("selected recurrent generation differs from exact emulator")
    checkpoint.write(output / "factorio-lm-quality-exact.json")
    checkpoint.write(output / "factorio-lm-general-exact.json")
    mining["selected_demo"] = selection
    _write_json(mining_path, mining)
    _write_prompt_finalists_markdown(output / "PROMPT-FINALISTS.md", mining)

    comparison = json.loads(
        (output / "cleaned-model-evaluation.json").read_text(encoding="utf-8"))
    float_training = json.loads(
        (output / "cleaned-float-training.json").read_text(encoding="utf-8"))
    int4_training = json.loads(
        (output / "cleaned-int4-training.json").read_text(encoding="utf-8"))
    training_report = {
        "run": "6D",
        "reason_for_retraining": "confirmed TinyStories separator normalization bug",
        "source_run6_artifacts_preserved": True,
        "data": json.loads((output / "data-report.json").read_text(encoding="utf-8")),
        "float_adaptation": float_training,
        "int4_recovery": int4_training,
        "evaluation": comparison,
        "selected_prompt": selection,
    }
    _write_json(output / "training-report.json", training_report)
    (output / "generation-samples.txt").write_text(
        "".join(
            f"PROMPT: {prompt_value!r}\nEXACT (256 chars):\n"
            f"{row['generated']}\n\n"
            for prompt_value, row in comparison["samples"][
                "cleaned_data_model"].items()
        )
        + f"SELECTED DEMO: {prompt!r}\nEXACT (256 chars):\n{generated}\n",
        encoding="utf-8",
    )

    selection_report = {"selected_demo": selection}
    _write_json(output / "selection-report.json", selection_report)
    return selection_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FactorioLM Run 6D pipeline")
    parser.add_argument(
        "command",
        choices=(
            "audit",
            "prepare-clean-data",
            "train-clean-float",
            "train-clean-int4",
            "evaluate-clean",
            "mine-early-prompts",
            "mine-coherent-prompts",
            "mine-contextual-prompts",
            "finalize-demo",
        ),
    )
    parser.add_argument("--output-dir", default=str(RUN6D_OUTPUT))
    args = parser.parse_args(argv)
    if args.command == "audit":
        audit_endoftext(args.output_dir)
    elif args.command == "prepare-clean-data":
        rebuild_cleaned_corpus(args.output_dir)
    elif args.command == "train-clean-float":
        train_clean_float_teacher(args.output_dir)
    elif args.command == "train-clean-int4":
        train_clean_int4(args.output_dir)
    elif args.command == "evaluate-clean":
        evaluate_clean_model(args.output_dir)
    elif args.command == "mine-early-prompts":
        mine_early_prompts(output_dir=args.output_dir)
    elif args.command == "mine-coherent-prompts":
        mine_coherent_prompts(output_dir=args.output_dir)
    elif args.command == "mine-contextual-prompts":
        mine_contextual_prompts(output_dir=args.output_dir)
    elif args.command == "finalize-demo":
        finalize_demo(output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
