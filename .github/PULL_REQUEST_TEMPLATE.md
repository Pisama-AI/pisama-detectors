## What this changes

<!-- 1–3 sentences. What's different after this PR and why. -->

## Type

- [ ] New detector
- [ ] Bug fix (false positive / false negative / crash)
- [ ] Framework adapter
- [ ] API change
- [ ] Docs

## Checklist

- [ ] Clean-venv install works: `pip install .` in a fresh env, import
      and run the affected detector.
- [ ] `DETECTOR_REGISTRY` length matches README (currently 42).
- [ ] If a new detector: registered via `@_register` with a tier, and
      the wrapper signature in `_api.py` matches the class method.
- [ ] No `from app.*` imports — all package internals resolve under
      `pisama_detectors.*`.
- [ ] README detector table updated if a new detector is added.

## Reproducer or before/after (for bug fixes)

```python
# Before: ...
# After: ...
```
