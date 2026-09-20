"""Fine-tune Run 7P on a mix of general and topic-specific TinyChat answers."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Any

from .checkpoint import ExactCheckpoint
from .run4_pipeline import Run4Config, _learning_rate, _write_json, create_float_model
from .run5_pipeline import export_integer_student
from .run6_pipeline import mask_summary
from .run7_pipeline import (
    Run7Config,
    _run5_config,
    evaluate_aligned_torch,
    evaluate_masked_exact,
)
from .run7p_pipeline import (
    RUN7P_ALPHABET,
    RUN7P_DIVERSE_THEMES,
    RUN7P_OUTPUT,
    MaskedTokenCorpus,
    PunctuationAlignedChatCorpus,
    Run7PConfig,
    _diverse_topic_names,
    _load_run7_integer,
    _semantic_demo_score,
    _sha256,
    _word_counter,
    generate_punctuation_prompt_set,
)
from .training import set_deterministic
from .vocabulary import Vocabulary


RUN7Q_OUTPUT = Path("generated/run7q")
RUN7P_FLOAT = RUN7P_OUTPUT / "factorio-lm-tinychat-punctuation-float.pt"
RUN7P_INTEGER = RUN7P_OUTPUT / "factorio-lm-tinychat-punctuation-aligned-int4.pt"
RUN7P_EXACT = RUN7P_OUTPUT / "factorio-lm-tinychat-punctuation-exact.json"


@dataclass(frozen=True)
class Run7QConfig(Run7PConfig):
    thematic_batch_probability: float = 0.60
    thematic_selection_weight: float = 0.65
    topic_name: str = "broad-diverse"
    topic_regex: str | None = None
    thematic_minimum_examples: int = 50
    save_integer_snapshots: bool = False

    def validate(self) -> None:
        Run7Config.validate(self)
        if self.float_steps or self.integer_qat_steps:
            raise ValueError("Run 7Q performs aligned refinement only")
        if self.aligned_float_steps < 1 or self.aligned_integer_steps < 1:
            raise ValueError("Run 7Q needs float and integer aligned refinement")
        if not 0.0 < self.thematic_batch_probability < 1.0:
            raise ValueError("thematic batch probability must be between zero and one")
        if not 0.0 < self.thematic_selection_weight < 1.0:
            raise ValueError("thematic selection weight must be between zero and one")
        if self.thematic_minimum_examples < 1:
            raise ValueError("thematic minimum examples must be positive")
        if self.topic_regex is not None:
            re.compile(self.topic_regex)


def _torch():
    import torch
    return torch


def prepare_run7q_records(output_dir: str | Path = RUN7Q_OUTPUT, *,
                          topic_name: str = "broad-diverse",
                          topic_regex: str | None = None) -> dict[str, Any]:
    """Materialize deterministic thematic views without copying the full corpus."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "source_run": "7P",
        "topic_name": topic_name,
        "topic_regex": topic_regex,
        "themes": RUN7P_DIVERSE_THEMES if topic_regex is None else None,
        "mood_exclusion": r"\b(?:happy|feel|feeling|sad|thank|lovely day)\b",
        "splits": {},
    }
    mood = re.compile(report["mood_exclusion"], re.IGNORECASE)
    for split in ("train", "validation"):
        source = RUN7P_OUTPUT / f"tinychat-punctuation-{split}.jsonl"
        rows = [json.loads(line) for line in source.read_text(
            encoding="utf-8").splitlines()]
        topic = re.compile(topic_regex, re.IGNORECASE) if topic_regex else None
        thematic = []
        for row in rows:
            matches = (bool(topic.search(row["question"])) if topic is not None
                       else bool(_diverse_topic_names(row["question"])))
            if matches and not mood.search(row["question"]):
                thematic.append(row)
        destination = output / f"tinychat-diverse-{split}.jsonl"
        destination.write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in thematic),
            encoding="utf-8")
        report["splits"][split] = {
            "ordinary_records": len(rows),
            "thematic_records": len(thematic),
            "source": str(source),
            "source_sha256": _sha256(source),
            "records": str(destination),
            "sha256": _sha256(destination),
        }
    _write_json(output / "dataset-report.json", report)
    return report


def _float_generate(model, prompts: list[str], count: int,
                    *, device: str) -> dict[str, str]:
    torch = _torch()
    vocabulary = Vocabulary(RUN7P_ALPHABET)
    result: dict[str, str] = {}
    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            tokens = vocabulary.encode(prompt)
            generated: list[int] = []
            for _ in range(count):
                context = torch.tensor(
                    [tokens[-64:]], dtype=torch.long, device=device)
                token = int(model(context)[0, -1].argmax())
                tokens.append(token)
                generated.append(token)
            result[prompt] = vocabulary.decode(generated)
    return result


def _balanced_train(model, teacher,
                    ordinary_train: PunctuationAlignedChatCorpus,
                    thematic_train: PunctuationAlignedChatCorpus,
                    ordinary_validation: PunctuationAlignedChatCorpus,
                    thematic_validation: PunctuationAlignedChatCorpus,
                    config: Run7QConfig, *, stage: str,
                    output_path: Path) -> dict[str, Any]:
    torch = _torch()
    integer = stage == "integer"
    if stage not in {"float", "integer"}:
        raise ValueError("stage must be float or integer")
    steps = config.aligned_integer_steps if integer else config.aligned_float_steps
    maximum_lr = (config.aligned_integer_learning_rate if integer
                  else config.aligned_float_learning_rate)
    temperature = config.integer_logit_temperature if integer else 1.0
    model.to(config.device)
    if hasattr(model, "masks"):
        model.masks = {name: value.to(config.device)
                       for name, value in model.masks.items()}
    if teacher is not None:
        teacher.to(config.device).eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=maximum_lr, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(
        config.seed + (970_000 if integer else 960_000))
    best = float("inf")
    log: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(steps):
        model.train()
        use_thematic = float(torch.rand((), generator=generator)) < (
            config.thematic_batch_probability)
        corpus = thematic_train if use_thematic else ordinary_train
        contexts, targets = corpus.batch(
            generator, config.batch_size, config.device)
        logits = model(contexts)[:, -1]
        hard = torch.nn.functional.cross_entropy(logits / temperature, targets)
        loss = hard
        if teacher is not None and config.distillation_weight:
            with torch.no_grad():
                probabilities = torch.softmax(
                    teacher(contexts)[:, -1] / temperature, dim=-1)
            distillation = torch.nn.functional.kl_div(
                torch.log_softmax(logits / temperature, dim=-1), probabilities,
                reduction="batchmean") * temperature * temperature
            loss = ((1.0 - config.distillation_weight) * hard
                    + config.distillation_weight * distillation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.gradient_clip).detach().cpu())
        lr = _learning_rate(
            step, steps, min(config.warmup_steps, max(1, steps // 5)), maximum_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if hasattr(model, "constrain_latents"):
            model.constrain_latents()
        if (step == 0 or (step + 1) % config.evaluation_interval == 0
                or step + 1 == steps):
            ordinary = evaluate_aligned_torch(
                model, ordinary_validation, device=config.device,
                characters=config.evaluation_characters,
                batch_size=config.batch_size, temperature=temperature)
            thematic = evaluate_aligned_torch(
                model, thematic_validation, device=config.device,
                characters=config.evaluation_characters,
                batch_size=config.batch_size, temperature=temperature)
            selection = (
                config.thematic_selection_weight * thematic["cross_entropy"]
                + (1.0 - config.thematic_selection_weight)
                * ordinary["cross_entropy"])
            row = {
                "step": step + 1, "objective": float(loss.detach().cpu()),
                "gradient_norm": gradient_norm, "learning_rate": lr,
                "ordinary_validation": ordinary,
                "thematic_validation": thematic,
                "selection_cross_entropy": selection,
                "elapsed_seconds": time.monotonic() - started,
            }
            log.append(row)
            print(
                f"Run7Q {stage} {step + 1}/{steps}: general "
                f"CE={ordinary['cross_entropy']:.4f} acc={ordinary['accuracy']:.2%}; "
                f"thematic CE={thematic['cross_entropy']:.4f} "
                f"acc={thematic['accuracy']:.2%}", flush=True)
            if selection < best:
                best = selection
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "run": "7Q", "stage": f"balanced-{stage}",
                    "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step + 1, "best_cross_entropy": best,
                    "masks": getattr(model, "masks", None),
                    "denominators": getattr(model, "denominators", None),
                    "config": asdict(config), "log": log,
                }, output_path)
            if integer and config.save_integer_snapshots:
                snapshot = output_path.with_name(
                    f"{output_path.stem}-step-{step + 1:04d}{output_path.suffix}")
                torch.save({
                    "run": "7Q", "stage": "balanced-integer-snapshot",
                    "model": model.state_dict(), "step": step + 1,
                    "selection_cross_entropy": selection,
                    "masks": getattr(model, "masks", None),
                    "denominators": getattr(model, "denominators", None),
                    "config": asdict(config), "log": log,
                }, snapshot)
    state = torch.load(output_path, map_location=config.device, weights_only=False)
    model.load_state_dict(state["model"])
    if hasattr(model, "masks"):
        model.masks = {name: value.to(config.device)
                       for name, value in state["masks"].items()}
    if integer:
        model.denominators = dict(state["denominators"])
    return {
        "stage": stage, "steps": steps, "best_step": int(state["step"]),
        "best_selection_cross_entropy": float(state["best_cross_entropy"]),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint": str(output_path), "log": state.get("log", log),
    }


def _export(model, config: Run7QConfig, output: Path) -> ExactCheckpoint:
    checkpoint = export_integer_student(
        model, _run5_config(config), {
            "run": "7Q", "variant": "balanced-thematic-recovery",
            "initialization": str(RUN7P_INTEGER),
            "float_teacher_initialization": str(RUN7P_FLOAT),
            "fixed_run7p_mask": True,
            "nonzero_weights": config.nonzero_weights,
            "weight_alphabet": list(range(-7, 8)),
            "score_divisor": 512,
        }, alphabet=RUN7P_ALPHABET)
    checkpoint.write(output / "factorio-lm-tinychat-diverse-exact.json")
    return checkpoint


def _rank_exact(model, records_path: Path, config: Run7QConfig,
                known_words, *, limit: int | None = None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in records_path.read_text(
        encoding="utf-8").splitlines()]
    if limit is not None:
        records = records[:limit]
    rows: list[dict[str, Any]] = []
    batch = 512
    for first in range(0, len(records), batch):
        chunk = records[first:first + batch]
        prompts = [f"Q {row['question']}\nA " for row in chunk]
        generated = generate_punctuation_prompt_set(
            model, prompts, 128, device=config.device)
        for record, prompt in zip(chunk, prompts, strict=True):
            trajectory = generated[prompt]
            answer = trajectory.split("\n", 1)[0].strip()
            score, quality = _semantic_demo_score(
                record["question"], record["answer"], answer, known_words)
            generic = bool(re.search(
                r"\b(?:thank you for asking|lovely day)\b", answer,
                re.IGNORECASE))
            rows.append({
                "source_row": int(record["source_row"]),
                "turn": int(record["turn"]),
                "question": record["question"],
                "reference_answer": record["answer"],
                "prompt": prompt, "generated_answer": answer,
                "generated_128": trajectory,
                "generic_greeting_output": generic,
                "score": score - 50.0 * generic,
                **quality,
            })
    rows.sort(key=lambda row: (
        -row["expanded_hard_filter_pass"], -row["score"],
        row["source_row"], row["turn"]))
    return rows


def train_run7q(*,
                config_path: str | Path = "configs/run7q-diverse.json",
                output_dir: str | Path = RUN7Q_OUTPUT) -> dict[str, Any]:
    torch = _torch()
    config = Run7QConfig.read(config_path)
    config.validate()
    set_deterministic(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_run7q_records(
        output, topic_name=config.topic_name, topic_regex=config.topic_regex)

    ordinary_train = PunctuationAlignedChatCorpus(
        RUN7P_OUTPUT / "tinychat-punctuation-train.jsonl",
        maximum_answer=config.aligned_maximum_answer_characters)
    ordinary_validation = PunctuationAlignedChatCorpus(
        RUN7P_OUTPUT / "tinychat-punctuation-validation.jsonl",
        maximum_answer=config.aligned_maximum_answer_characters)
    thematic_train = PunctuationAlignedChatCorpus(
        data["splits"]["train"]["records"],
        maximum_answer=config.aligned_maximum_answer_characters)
    thematic_validation = PunctuationAlignedChatCorpus(
        data["splits"]["validation"]["records"],
        maximum_answer=config.aligned_maximum_answer_characters,
        minimum_examples=config.thematic_minimum_examples)

    float_model = create_float_model(Run4Config(
        seed=config.seed, device=config.device))
    float_source = torch.load(
        RUN7P_FLOAT, map_location=config.device, weights_only=False)
    float_model.load_state_dict(float_source["model"])
    float_path = output / "factorio-lm-tinychat-diverse-float.pt"
    float_training = _balanced_train(
        float_model, None, ordinary_train, thematic_train,
        ordinary_validation, thematic_validation, config, stage="float",
        output_path=float_path)

    integer_model, _ = _load_run7_integer(RUN7P_INTEGER, config.device)
    integer_path = output / "factorio-lm-tinychat-diverse-int4.pt"
    integer_training = _balanced_train(
        integer_model, float_model, ordinary_train, thematic_train,
        ordinary_validation, thematic_validation, config, stage="integer",
        output_path=integer_path)
    checkpoint = _export(integer_model, config, output)

    validation = MaskedTokenCorpus(
        RUN7P_OUTPUT / "tinychat-punctuation-validation.bin",
        RUN7P_OUTPUT / "tinychat-punctuation-validation.mask.bin")
    exact_metrics = evaluate_masked_exact(
        checkpoint, validation, characters=config.exact_evaluation_characters)
    baseline_metrics = evaluate_masked_exact(
        ExactCheckpoint.read(RUN7P_EXACT), validation,
        characters=config.exact_evaluation_characters)
    known_words = _word_counter(
        RUN7P_OUTPUT / "tinychat-punctuation-train.jsonl")
    validation_rows = _rank_exact(
        integer_model, Path(data["splits"]["validation"]["records"]),
        config, known_words)
    _write_json(output / "prompt-mining-diverse-validation.json", {
        "candidates_evaluated": len(validation_rows),
        "top_100": validation_rows[:100],
    })
    mask = mask_summary(integer_model.masks)
    if mask["total_nonzeros"] != 25_100:
        raise AssertionError("Run 7Q changed the fixed sparse mask budget")
    report = {
        "run": "7Q", "configuration": asdict(config), "data": data,
        "initialization": {
            "float": str(RUN7P_FLOAT), "integer": str(RUN7P_INTEGER),
            "run7p_exact": str(RUN7P_EXACT), "run7p_preserved": True,
        },
        "training": {"float": float_training, "integer": integer_training},
        "exact_validation": {
            "run7p": baseline_metrics, "run7q": exact_metrics,
            "accuracy_loss": (
                baseline_metrics["answer_characters"]["accuracy"]
                - exact_metrics["answer_characters"]["accuracy"]),
        },
        "sparsity": mask,
        "thematic_validation_candidates": len(validation_rows),
        "top_thematic_candidates": validation_rows[:20],
        "exact_checkpoint": str(
            output / "factorio-lm-tinychat-diverse-exact.json"),
        "quality_gate": {
            "general_accuracy_within_two_points_of_run7p": (
                exact_metrics["answer_characters"]["accuracy"]
                >= baseline_metrics["answer_characters"]["accuracy"] - 0.02),
            "coherent_non_greeting_candidate_found": False,
        },
    }
    _write_json(output / "training-report.json", report)
    _write_json(output / "manifest.json", {
        "run": "7Q",
        "checkpoint": report["exact_checkpoint"],
        "training_report": str(output / "training-report.json"),
        "prompt_mining": str(output / "prompt-mining-diverse-validation.json"),
    })
    return {"model": integer_model, "checkpoint": checkpoint, "report": report}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/run7q-diverse.json")
    parser.add_argument("--output", default=str(RUN7Q_OUTPUT))
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    train_run7q(config_path=arguments.config, output_dir=arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
