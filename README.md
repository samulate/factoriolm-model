# factoriolm-model

A tiny Transformer designed to run in vanilla Factorio circuits.

This repo has the training code, quantization, and integer emulator used to
test the model. It's a character-level model with six layers, hidden width 32,
four attention heads, FFN width 64, and a 64-character context.

## Setup

You'll need Python 3.11 or newer. Running an exported model only needs the
standard library. Training also needs PyTorch.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[train]'
```

## Try the model

```bash
.venv/bin/factoriolm-model emulate-run3 \
  --checkpoint generated/run7p/factorio-lm-tinychat-punctuation-exact.json \
  --prompt $'Q How are you today?\nA ' --count 64
```

`emulate-run3` works with all checkpoint versions, despite the name. Leave out
`--checkpoint` to use the small Run 3 model used by the tests.

## What's here

- [How the model works](docs/MODEL.md)
- [Training commands and experiments](docs/TRAINING.md)
- `configs/`: settings for each experiment.
- `generated/`: checkpoints, results, and sample output. Downloaded TinyChat data stays local.

The Python package is called `factorio_lm`.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests -v
```
