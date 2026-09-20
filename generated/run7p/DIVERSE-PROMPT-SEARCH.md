# Run 7P diverse prompt search

Of 3,059 non-mood training questions, none produced an answer that passed
the grammar, relevance, repetition, and non-generic-output filters. Each was
evaluated for 128 characters with the exact int32 model.

The data has varied topics, but this model mostly answers with greetings.

## Prominent corpus themes

Theme counts overlap when a question belongs to more than one category.

| Theme | All training questions | Non-mood questions searched |
|---|---:|---:|
| Weather and nature | 1,040 | 823 |
| Fear, darkness, and horror | 989 | 508 |
| Food and cooking | 622 | 554 |
| Books, music, and art | 528 | 438 |
| Space and mystery | 519 | 362 |
| Animals | 489 | 389 |
| Machines and adventure | 184 | 154 |

Some questions from the source data:

- `Did you know that some fish can glow in the dark?`
- `Do you know that the fat cat can fly in a funny dream?`
- `Have you ever seen a machine that can create art or music?`
- `What do you think will be the next big discovery in space?`
- `What do you think would happen if a horse tried to swim?`
- `What do you think causes those strange sounds in the dark?`

The reference answers cover these topics. Generated answers mostly repeat
`happy`, `beautiful day`, `feel a bit`, and `thank you for asking`.

## Representative exact failures

Input:

```text
Q Did you know that some fish can glow in the dark?
A 
```

Exact output begins:

```text
I am feeling that a bit can be a beautiful day to seem that the see the see...
```

Input:

```text
Q What do you think will be the next big discovery in space?
A 
```

Exact output begins:

```text
I also feel a bit seem to feel happy today, it is a lovely day...
```

Input:

```text
Q What do you think causes those strange sounds in the dark?
A 
```

Exact output begins:

```text
I also feel happy today, it is a love to hear and can help...
```

The closest grammatical non-mood output was:

```text
Q What do you do to have fun when the sun is shining bright?
A Yes, it is a beautiful day.
```

The sentence is grammatical but doesn't answer the question.

## Follow-up

The greeting demo and the longer `wonderful day` alternative are the useful
results from this checkpoint. A broader Q/A demo needs topic-balanced training
and evaluation of complete greedy answers, rather than more prompt searches.
