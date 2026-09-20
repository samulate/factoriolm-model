# Run 7Q results

Run 7Q did not produce a usable non-greeting answer. Run 7P is still the release model.

| Experiment | Result |
|---|---|
| Balanced topics: 3,059 training questions | Better teacher-forced accuracy, but broken greedy answers |
| Cosmic/mystery: 146 training, 18 validation questions | Some topical words, no coherent held-out answers |
| Scheduled sampling: 60 steps | No acceptable answers, general cross-entropy got worse |
| Specialist probe: 99% topical batches | Step 1 was best, further training made validation worse |
| All 13,503 non-mood Run 7P training questions | 274 seconds, zero strict non-generic or surprising-topic passes |

## Examples

```text
Q Have you ever seen a machine that can create art or music?
A I am feeling that is a little can be a bit and about the see tha...
```

```text
Q What if we see strange shadows or hear noises in the dark?
A I am all feeling that is a beautiful day to can feel happy to ca...
```

```text
Q Did they say what kind of snake it was?
A I hope a beautiful day in the seem to feel happy too, it can be...
```

The examples above failed review. Both the float teacher and int4 model tend
to fall back on greetings regardless of the question. Teacher-forced accuracy
isn't a good measure of answer quality here.

## Next attempt

Train a new float teacher with balanced topics and use greedy answer quality
to select checkpoints. More mining of the same greeting-heavy model looks
unlikely to help.

No Run 7Q candidate reached hardware compilation. The recorded Run 7P machine
counts are 79,037 inference entities and 205,101 wires.
