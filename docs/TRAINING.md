# Training

Training keeps float weights for the optimizer and uses quantized weights in
the forward pass. Straight-through gradients let the weights learn through
quantization and truncation. The result is exported as integer JSON for the
[emulator](MODEL.md).

## Start with a small run

```bash
.venv/bin/factoriolm-model train-run3 \
  --config configs/run3-smoke.json \
  --corpus data/smoke_corpus.txt \
  --output-dir generated/run3
```

This uses the small included corpus to check that training and export work.
It tries score divisors 16, 32, 64, 128, 256, and 512, then compares validation
loss, integer ranges, attention stats, and sample text. To resume a run, pass
`--resume generated/run3/factorio-lm-smoke-latent.pt`.

## Train on TinyStories

```bash
.venv/bin/factoriolm-model train-run4 \
  --config configs/run4-tinystories.json \
  --train-source /path/to/train.txt \
  --validation-source /path/to/validation.txt \
  --output-dir generated/run4
```

This cleans up the text, trains a float model, then fine-tunes it with
quantization-aware training (QAT) and exports an integer checkpoint. Use
`--resume-float` or `--resume-qat` to pick up either stage where it left off.

## Later experiments

| Run | What it tries |
|---|---|
| Run 5 | Sparse weights and scaled-int3 QAT |
| Run 6 | Scaled-int4 weights and a search for better sparsity masks |
| Run 6D | Retraining after fixing TinyStories separators |
| Run 7 | Fine-tuning on TinyChat |
| Run 7P | Adding punctuation and aligning answer targets with questions |
| Run 7Q | Topic balancing and training with predicted text as context |

Settings live in `configs/`, and outputs go in the matching `generated/run*`
folders. TinyChat downloads and processed data stay local and aren't tracked
in Git. Later runs need some model and data files from earlier runs.

For the available commands, run `python -m factorio_lm.run6d_pipeline --help`,
`python -m factorio_lm.run7_pipeline --help`,
`python -m factorio_lm.run7p_pipeline --help`, and
`python -m factorio_lm.run7q_pipeline --help`.

Run 5 and Run 6 experiments are called as Python functions in `run5_pipeline.py`
and `run6_pipeline.py`. You can also use `evaluate-run5` from the main CLI to
evaluate a scaled-int3 checkpoint.
