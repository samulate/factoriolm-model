# Run 6D coherence search

No prompt gave a clean, non-repetitive 128-character continuation.

Search coverage:

- 10,000 held-out prefixes, each generated for 128 characters.
- 50,000 held-out and 100,000 training prefixes at 64 characters, 5,000 finalists at 128.
- 20,000 held-out and 40,000 training prefixes with prior-sentence context, 3,000 finalists at 128.
- 220,000 discovery trajectories and 18,000 full-length evaluations in total.

The best candidates were also checked with the independent exact emulator.

## Best short continuation

Cleaned TinyStories validation, line 19,645, character offset 828,921.

INPUT:

```text
One day he was playing with his 
```

EXACT FIRST 32:

```text
friends
He was very happy
He wan
```

EXACT FIRST 64:

```text
friends
He was very happy
He wanted to play with his bear and ha
```

EXACT FIRST 128:

```text
friends
He was very happy
He wanted to play with his bear and have fun
He was happy and happy and said the bear friends
He liked
```

The first 71 characters read well. Repetition starts with `He was happy and
happy` after that. This works as a 64-character demo, but fails the full
128-character target. Further improvement needs training work.
