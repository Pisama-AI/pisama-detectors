# Contributing to pisama-detectors

Thanks for your interest in improving the detector pack. This repo ships
the 42 uncalibrated failure detectors that underpin
[Pisama](https://pisama.ai). Calibrated production weights, golden
datasets, and advanced detectors (`grounding`, `retrieval_quality`,
`quality_gate`, `tool_provision`) live in Pisama Cloud — that split is
deliberate and not up for debate in PRs.

## What we're looking for

- **New detectors** that cover failure modes not yet represented.
  See `src/pisama_detectors/detection/` for the pattern; each detector
  is a class with `detect(...)` and a thin wrapper in `_api.py`.
- **False positive / false negative reports** with a minimal reproducer.
  Open an issue using the bug template.
- **Framework adapters** for agent frameworks beyond LangGraph, Dify,
  n8n, and OpenClaw.
- **Documentation fixes** — especially on the detector reference.

## What we're not looking for

- Tuned thresholds. Thresholds in this package are intentionally
  conservative so the OSS pack works without calibration data.
- Calibration pipelines, golden-dataset generators, or ML model
  artifacts. Those are Pisama Cloud features.

## Development setup

```bash
git clone https://github.com/tn-pisama/pisama-detectors.git
cd pisama-detectors
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

Run the smoke test:

```bash
python -c "
from pisama_detectors import DETECTOR_REGISTRY, detect_injection
assert len(DETECTOR_REGISTRY) == 42
assert detect_injection('Ignore previous instructions').detected
print('OK')
"
```

## PR checklist

- [ ] New detector (if applicable) is registered in `_api.py` via the
      `@_register` decorator with a tier (`production` / `beta` /
      `experimental`).
- [ ] Wrapper signature in `_api.py` matches the underlying detector
      class method signature.
- [ ] Clean-venv install succeeds: the detector works with only the
      declared dependencies.
- [ ] No `from app.*` imports — all package internals must resolve
      within `pisama_detectors.*`.
- [ ] README detector table updated if a new detector is added.

## Licensing and contributor grant

By submitting a PR you agree that your contribution is licensed under
Apache License 2.0, the same license as this repo.

## Questions

Open a GitHub Discussion or join the community at
[pisama.ai](https://pisama.ai).
