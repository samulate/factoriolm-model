# Run 7P expanded demo alternatives

Searched all 1,629 held-out validation questions at 128 characters each, plus
3,082 training questions similar to the successful greeting prompts.

## Recommended held-out alternative

Input:

```text
Q How are you feeling today on this wonderful day?
A 
```

Exact generated answer:

```text
I am feeling happy today, thank you for asking, it is a beautiful day outside.
```

Exact 128-character trajectory:

```text
I am feeling happy today, thank you for asking, it is a beautiful day outside.
Q I am also feel happy today, thank you for askin
```

The answer stays coherent for 78 characters and ends with a period. After the
newline, generation drifts into another chat record.

## Shortlist

1. **Held-out, source 393283:0**

   Input: `How are you feeling today on this wonderful day?`

   Output: `I am feeling happy today, thank you for asking, it is a beautiful day outside.`

2. **Held-out, source 23849:0 — current released demo**

   Input: `Hi there, I feel really happy today, how about you?`

   Output: `I am feeling happy too, thank you for asking.`

3. **Training split, source 641075:0**

   Input: `How are you feeling today on this clear and bright morning?`

   Output: `I am feeling happy today, thank you for asking.`

Training examples are search diagnostics, not held-out evaluation. The first
two examples above are held out, the third is from training.

For in-game use, rebuild the prompt ROM and compare the machine's output with
the exact model. The alternative uses the same model weights.
