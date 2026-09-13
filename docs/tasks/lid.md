# LID - Spoken Language Identification

LID evaluates utterance-level labels produced by an external
spoken-language identification system. The evaluator does not run a LID model
or read audio.

```bash
sure-eval metric routes lid --metric accuracy
```

## Metric And Route

| Metric | Pipeline ID | Required input | Nodes |
|:-------|:------------|:---------------|:------|
| `accuracy` | `lid.any.accuracy.lid_label_canonical_v1.classify_v1` | `ref`, `hyp` | `normalization/lid_label` -> `scoring/classify` |

Accuracy is the number of correctly identified reference utterances divided by
all reference utterances. Missing, empty, special-token, and unknown
predictions count as incorrect.

## Input Contract

References and hypotheses are aligned `key<TAB>label` files. The label
inventory is dataset-defined:

```text
utt-zh<TAB>zh-mandarin
utt-en<TAB>en
```

Both files must use unique keys. The scoring loader reports missing and extra
hypothesis keys in the result details.

## Label Normalization

Labels are lowercased and spaces, underscores, or slashes become hyphens. For
example, `zh mandarin`, `zh_mandarin`, and `zh-mandarin` all normalize to
`zh-mandarin`. A label is valid when it remains non-empty after normalization.

## CLI Usage

```bash
sure-eval metric describe lid \
  --pipeline-id lid.any.accuracy.lid_label_canonical_v1.classify_v1 \
  --output /tmp/lid.json
sure-eval metric run --pipeline /tmp/lid.json \
  --ref-file ref.txt --hyp-file hyp.txt --output-dir /tmp/lid-eval
```

## Output

`report.json` contains aggregate accuracy, correct/total/valid counts,
alignment diagnostics, and per-sample normalized labels.
`pipeline_description.json` preserves the exact route, input roles, and
computation nodes.
