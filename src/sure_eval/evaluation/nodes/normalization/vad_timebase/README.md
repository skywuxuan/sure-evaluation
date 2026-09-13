# normalization/vad_timebase

Loads VAD reference and prediction JSONL, enforces their input contract, and
normalizes rows on the reference seconds timebase.

This task version supports only `profile: strict` with zero collar and zero
boundary exclusion. JSONL parsing, key alignment, required-field checks, and
interval checks are internal parts of this node rather than a separate pipeline
stage. Invalid, out-of-range, and overlapping intervals are rejected before
timebase conversion. The node keeps a stable order and records scored-region
summaries on the reference `duration` timebase.
