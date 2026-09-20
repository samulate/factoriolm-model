# How the model works

## Architecture

| Property | Value |
|---|---|
| Vocabulary | 64 characters, with token order saved in each checkpoint |
| Context | 64 tokens |
| Layers | 6 |
| Hidden width | 32 |
| Attention heads | 4, each of width 8 |
| FFN width | 64 |
| Normalization | MaxNorm with scale 31 |
| FFN activation | ReLU |
| Token selection | Greedy argmax, with ties going to the lowest token index |

Run 3 uses ternary weights. Later runs use sparse integer weights with a scale
for each matrix. Run 7P uses weights in `[-7, 7]` and swaps the ten digit tokens
for punctuation. Read the alphabet, weights, and scales from the checkpoint
you're using, since they differ between runs.

## Arithmetic

The math uses signed 32-bit integers with wraparound. Division truncates toward
zero, and dividing by zero gives zero. The helpers are in `factorio_lm/int32.py`
and `factorio_lm/emulator.py`.

Matrices are stored as `[output][input]`. Multiplying a matrix `W` by a vector `x` looks like this:

```text
z[i] = sum_j(W[i][j] * x[j])
y[i] = trunc((z[i] * scale.numerator) / scale.denominator)
```

Products and sums wrap as int32, including multiplication by the scale numerator.
Apply the scale after finishing the sum. Dividing each term or rounding instead
of truncating changes the result.

MaxNorm is:

```text
d = max(max_i(abs_int32(x[i])), 1)
y[i] = trunc((31 * x[i]) / d)
```

Negating `INT32_MIN` overflows, so `abs_int32(INT32_MIN)` stays `INT32_MIN`.
See `abs_int32` in the emulator.

## Attention and state

Each layer keeps its own key and value history. Reset clears the values and
validity flags. A new token's K/V pair goes into age zero before attention runs.
Older entries shift one slot, and entries past age 63 are dropped. Invalid slots
don't count toward either the numerator or denominator.

For each head and valid age slot:

```text
dot = sum_j(query[j] * key[j])
s = clamp(trunc(dot / score_divisor) + relative_bias[head][age], -6, 6)
w = 1 << (s + 6)
output[j] = trunc(sum_age(w * value[j]) / sum_age(w))
```

This math also wraps as int32. The powers of two take the place of softmax.

Each block computes:

```text
n = MaxNorm(x)
q, k, v = scaled_linear_projections(n)
a = attention(q, updated_kv_history)
r = x + scaled_output_projection(a)
y = r + scaled_ffn_down(ReLU(scaled_ffn_up(MaxNorm(r))))
```

After six blocks, the scaled LM head gives 64 logits. Feeding in the prompt
fills the caches. The last prompt token predicts the first output character.
That character is then fed back in to predict the next one.

## Exact checkpoints

The `factorio-lm-exact` JSON files hold the architecture, vocabulary, embeddings,
six sets of projection and FFN matrices, relative-age biases, score divisor, and
LM head. Version 2 adds the weight limit and matrix scales. Version 1 uses
ternary weights and a scale of one.

`ExactCheckpoint.read()` checks the shapes and integer values. You don't need
optimizer state or PyTorch to run one of these checkpoints. The format is
defined in `factorio_lm/checkpoint.py`.

`ExactFactorioLM.step()` returns the embeddings, intermediate layer values,
cache contents, logits, and chosen token. Use these traces to check another
implementation. They don't describe circuit wiring or game-tick timing.
