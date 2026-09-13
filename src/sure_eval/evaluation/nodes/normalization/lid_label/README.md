# normalization/lid_label

This node converts spoken-language labels to lowercase, hyphen-separated
codes before accuracy scoring. For example, `zh mandarin`, `zh_mandarin`, and
`zh-mandarin` become the same canonical label. ISO language codes such as
`en`, `ar`, and `bo` are unchanged. Empty and special-token labels normalize
to an invalid empty value.

The LID route accepts any non-empty dataset-defined label after normalization.
