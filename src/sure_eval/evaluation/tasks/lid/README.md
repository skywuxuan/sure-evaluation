# LID Task

LID evaluates labels produced by an external spoken-language identification
system. The evaluator aligns `key<TAB>label` reference and hypothesis files,
canonicalizes both label streams, and computes utterance-level accuracy.

```text
lid.any.accuracy.lid_label_canonical_v1.classify_v1
```

The evaluator does not run a LID model or read audio. Label production remains
the responsibility of the system under evaluation.
