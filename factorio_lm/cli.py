"""Command-line interface for training, evaluation, and exact inference."""

from __future__ import annotations

import argparse
import json

from .checkpoint import ExactCheckpoint


def _train_run3(args: argparse.Namespace) -> int:
    from .run3_pipeline import run_training

    result = run_training(config_path=args.config, corpus_path=args.corpus,
                          output_dir=args.output_dir, resume_path=args.resume)
    metrics = result["metrics"]
    print(f"selected SCORE_DIVISOR={result['checkpoint'].score_divisor}")
    print(f"train loss {metrics['final_training']['initial_train_loss']:.6f} -> "
          f"{metrics['final_training']['final_train_loss']:.6f}")
    print(f"exact validation loss={metrics['exact_validation_loss']:.6f}")
    print(f"sample={metrics['sample_generated']!r}")
    return 0


def _emulate_run3(args: argparse.Namespace) -> int:
    from .full_model import ExactFactorioLM

    checkpoint = ExactCheckpoint.read(args.checkpoint)
    model = ExactFactorioLM(checkpoint)
    generated, traces = model.generate(args.prompt, args.count)
    print(json.dumps({"prompt": args.prompt, "generated": generated,
                      "generated_token_ids": model.vocabulary.encode(generated),
                      "transactions": len(traces), "cache": model.cache_summary()},
                     indent=2, sort_keys=True))
    return 0


def _train_run4(args: argparse.Namespace) -> int:
    from .run4_pipeline import run_training

    result = run_training(
        config_path=args.config, train_source=args.train_source,
        validation_source=args.validation_source, output_dir=args.output_dir,
        resume_float=args.resume_float, resume_qat=args.resume_qat,
    )
    report = result["report"]
    for name in ("float", "qat", "exact"):
        metrics = report["metrics"][name]
        print(f"{name}: CE={metrics['cross_entropy']:.6f} "
              f"perplexity={metrics['perplexity']:.6f} accuracy={metrics['accuracy']:.3%}")
    print(f"clearly better than Run 3: {report['clearly_better_than_run3']}")
    return 0


def _evaluate_run5(args: argparse.Namespace) -> int:
    from .run5_pipeline import finalize_evaluation

    report = finalize_evaluation(
        output_dir=args.output_dir, config_path=args.config,
        exact_path=args.checkpoint,
    )
    metrics = report["metrics"]["scaled_int3_exact_forward"]
    print(f"exact int3: CE={metrics['cross_entropy']:.6f} "
          f"perplexity={metrics['perplexity']:.6f} accuracy={metrics['accuracy']:.3%}")
    print(f"samples: {args.output_dir}/generation-samples.txt")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="factoriolm-model", description="FactorioLM model training and exact integer reference inference",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_run3 = subparsers.add_parser("train-run3", help="run deterministic QAT and export")
    train_run3.add_argument("--config", default="configs/run3-smoke.json")
    train_run3.add_argument("--corpus", default="data/smoke_corpus.txt")
    train_run3.add_argument("--output-dir", default="generated/run3")
    train_run3.add_argument("--resume")
    train_run3.set_defaults(func=_train_run3)

    emulate_run3 = subparsers.add_parser("emulate-run3", help="run exact six-layer autoregressive inference")
    emulate_run3.add_argument("--checkpoint", default="generated/run3/factorio-lm-smoke-exact.json")
    emulate_run3.add_argument("--prompt", default="Factorio ")
    emulate_run3.add_argument("--count", type=int, default=16)
    emulate_run3.set_defaults(func=_emulate_run3)

    train_run4 = subparsers.add_parser("train-run4", help="float-pretrain and QAT-fine-tune TinyStories")
    train_run4.add_argument("--config", default="configs/run4-tinystories.json")
    train_run4.add_argument("--train-source", required=True)
    train_run4.add_argument("--validation-source", required=True)
    train_run4.add_argument("--output-dir", default="generated/run4")
    train_run4.add_argument("--resume-float")
    train_run4.add_argument("--resume-qat")
    train_run4.set_defaults(func=_train_run4)

    evaluate_run5 = subparsers.add_parser("evaluate-run5", help="evaluate scaled-int3 exact generation and ranges")
    evaluate_run5.add_argument("--config", default="configs/run5-integer.json")
    evaluate_run5.add_argument("--checkpoint", default="generated/run5/factorio-lm-quality-exact.json")
    evaluate_run5.add_argument("--output-dir", default="generated/run5")
    evaluate_run5.set_defaults(func=_evaluate_run5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
