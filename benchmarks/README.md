# Public benchmark evidence

`trail.json` is the archived report from the April 2026 Pisama heuristic run
over TRAIL. `trail_llm_baselines.json` contains the archived comparison runs.

Verify every published per-category precision, recall, and F1 value, plus the
aggregate macro-F1 and micro-F1:

```bash
python benchmarks/verify_report.py
```

The verifier uses only Python's standard library. CI runs it on every pull
request.

## Reproducibility boundary

The checked-in confusion counts fully reproduce the category metrics and
aggregate F1 values. The archived 59.9% joint-accuracy value requires
per-annotation predicted spans and categories. Those predictions were not
included in the original public artifact, so the verifier reports that value
as archived rather than claiming to recompute it.

The archived top-level metadata says 813 annotations were mapped, while the
14 published category summaries contain support for 808. The verifier exposes
this five-annotation metadata gap rather than silently treating the values as
equivalent.

This limitation is intentional and visible. A future benchmark release must
include sanitized per-annotation predictions and a dataset fingerprint before
its joint-accuracy result is described as independently reproducible.
