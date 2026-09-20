"""Deterministic Run 3 training, divisor selection, export, and golden traces."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .checkpoint import ExactCheckpoint, SCORE_DIVISORS
from .full_model import ExactFactorioLM
from .training import (
    TrainingConfig,
    create_model,
    exact_attention_statistics,
    export_model,
    load_corpus,
    select_device,
    train_model,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exact_loss(checkpoint: ExactCheckpoint, tokens: list[int], *, limit: int = 96) -> float:
    model = ExactFactorioLM(checkpoint)
    losses: list[float] = []
    sequence = tokens[:limit + 1]
    for token, target in zip(sequence, sequence[1:]):
        logits = model.step(token).logits
        maximum = max(logits)
        losses.append(maximum + math.log(sum(math.exp(value - maximum) for value in logits)) - logits[target])
    return sum(losses) / max(1, len(losses))


def sample(checkpoint: ExactCheckpoint, prompt: str, length: int) -> str:
    return ExactFactorioLM(checkpoint).generate(prompt, length)[0]


def compare_qat_to_exact(model, checkpoint: ExactCheckpoint, tokens: list[int], device: str) -> dict[str, Any]:
    import torch

    sequence = tokens[:8]
    model.eval()
    with torch.no_grad():
        qat = model(torch.tensor([sequence], dtype=torch.long, device=device)).round().to(torch.int64).cpu().tolist()[0]
    exact_model = ExactFactorioLM(checkpoint)
    exact = [exact_model.step(token).logits for token in sequence]
    differences = [abs(left - right) for qat_row, exact_row in zip(qat, exact, strict=True)
                   for left, right in zip(qat_row, exact_row, strict=True)]
    return {"tokens_compared": len(sequence), "logits_compared": len(differences),
            "maximum_absolute_difference": max(differences, default=0),
            "all_logits_exact": all(difference == 0 for difference in differences)}


def golden_traces(checkpoint: ExactCheckpoint) -> dict[str, Any]:
    prompts = ("Factorio ", "Signals ", "Build ")
    result: dict[str, Any] = {"checkpoint_score_divisor": checkpoint.score_divisor, "prompts": []}
    for prompt in prompts:
        model = ExactFactorioLM(checkpoint)
        generated, traces = model.generate(prompt, 8)
        # Last three priming/generation transactions keep the artifact useful
        # without duplicating several megabytes of all-zero early cache slots.
        selected = traces[-3:]
        result["prompts"].append({
            "prompt": prompt,
            "expected_generated_text": generated,
            "expected_generated_token_ids": model.vocabulary.encode(generated),
            "selected_step_traces": [trace.to_dict() for trace in selected],
            "final_cache_summary": model.cache_summary(),
        })
    return result


def run_training(*, config_path: str | Path, corpus_path: str | Path,
                 output_dir: str | Path, resume_path: str | Path | None = None) -> dict[str, Any]:
    config = TrainingConfig.read(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_tokens, validation_tokens, corpus = load_corpus(corpus_path)
    device = select_device(config.device)
    comparison_steps = config.candidate_steps + config.final_steps
    divisor_rows: list[dict[str, Any]] = []
    for divisor in SCORE_DIVISORS:
        model = create_model(config, divisor).to(device)
        metrics = train_model(model, train_tokens, validation_tokens, config,
                              steps=comparison_steps, device=device)
        checkpoint = export_model(model, metadata={"stage": "score-divisor-candidate"})
        row = {
            "score_divisor": divisor,
            "training": metrics,
            "exact_validation_loss": exact_loss(checkpoint, validation_tokens),
            "attention": exact_attention_statistics(checkpoint, validation_tokens),
            "generated_sample": sample(checkpoint, config.sample_prompt, config.sample_length),
        }
        divisor_rows.append(row)
    selected = min(divisor_rows, key=lambda row: (row["exact_validation_loss"], row["score_divisor"]))
    selected_divisor = int(selected["score_divisor"])

    model = create_model(config, selected_divisor).to(device)
    latent_path = output / "factorio-lm-smoke-latent.pt"
    final_training = train_model(model, train_tokens, validation_tokens, config,
                                 steps=config.final_steps if resume_path else comparison_steps,
                                 device=device,
                                 resume_path=resume_path, latent_path=latent_path)
    metadata = {
        "quality": "deterministic smoke-trained model; not a production language model",
        "seed": config.seed,
        "training_steps": final_training["completed_steps"],
        "corpus": corpus,
        "selected_score_divisor": selected_divisor,
        "latent_parameters_required_for_inference": False,
    }
    checkpoint = export_model(model, metadata=metadata)
    checkpoint_path = checkpoint.write(output / "factorio-lm-smoke-exact.json")
    comparison = compare_qat_to_exact(model, checkpoint, validation_tokens, device)
    final_metrics = {
        "configuration": asdict(config),
        "corpus": corpus,
        "accelerator": device,
        "divisor_comparison_steps_per_candidate": comparison_steps,
        "final_training": final_training,
        "exact_validation_loss": exact_loss(checkpoint, validation_tokens),
        "attention": exact_attention_statistics(checkpoint, validation_tokens),
        "qat_vs_exact": comparison,
        "embedding_range": [min(min(row) for row in checkpoint.embeddings),
                            max(max(row) for row in checkpoint.embeddings)],
        "relative_bias_range": [min(item for table in checkpoint.relative_biases for row in table for item in row),
                                max(item for table in checkpoint.relative_biases for row in table for item in row)],
        "sparsity": checkpoint.sparsity_report(),
        "sample_prompt": config.sample_prompt,
        "sample_generated": sample(checkpoint, config.sample_prompt, config.sample_length),
        "checkpoint": str(checkpoint_path),
        "latent_checkpoint": str(latent_path),
    }
    divisor_report = {
        "candidate_values": list(SCORE_DIVISORS),
        "selection_rule": "minimum exact-integer validation cross entropy; lowest divisor breaks an exact tie",
        "selected_production_score_divisor": selected_divisor,
        "candidates": divisor_rows,
    }
    golden = golden_traces(checkpoint)
    _write_json(output / "training-config.json", asdict(config))
    _write_json(output / "training-summary.json", final_metrics)
    _write_json(output / "score-divisor-evaluation.json", divisor_report)
    _write_json(output / "golden-traces.json", golden)
    return {"checkpoint": checkpoint, "metrics": final_metrics,
            "divisor_report": divisor_report, "golden": golden}
