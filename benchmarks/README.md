# Archived benchmark evidence

`trail.json` is the archived report from the April 2026 Pisama heuristic run
over TRAIL. It is evidence for an archived Pisama platform run, not an
evaluation of the current `pisama-detectors` package release.
`trail_llm_baselines.json` contains the archived comparison runs.
`evidence.json` pins the reviewed report by SHA-256 and records its claim
boundary.

Verify every published per-category precision, recall, and F1 value, plus the
aggregate macro-F1 and micro-F1:

```bash
python benchmarks/verify_report.py
```

The verifier uses only Python's standard library. CI runs it on every pull
request.

## Reproducibility boundary

The checked-in confusion counts reproduce the category metrics and aggregate
F1 arithmetic. They do not constitute held-out evidence. A later provenance
audit found that 144 of the 148 TRAIL traces appeared in calibration material,
so this result is in-distribution. The result must not be used as evidence of
generalization to unseen traces.

The archived 59.9% joint-accuracy value requires per-annotation predicted spans
and categories. Those predictions were not included in the original public
artifact, so the verifier reports that value as archived rather than claiming
to recompute it. The archive also lacks a negative candidate set, which means
its zero false-positive count cannot independently establish production
precision.

The archived top-level metadata says 813 annotations were mapped, while the
14 published category summaries contain support for 808. The verifier exposes
this five-annotation metadata gap rather than silently treating the values as
equivalent.

These limitations are intentional and visible. A future package-level
benchmark must use a dataset excluded from calibration and include a dataset
fingerprint, sanitized per-example predictions, negative examples, the exact
package version, and a frozen evaluation command before it is described as
held out or independently reproducible.
