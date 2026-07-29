# Changelog

All notable changes to `pisama-detectors` are documented here.

## [Unreleased]

## [0.3.4] - 2026-07-29

### Fixed

- Correct the README license badge, which read `License: BSL 1.1`. That is not a
  valid SPDX identifier, and the nearest match a scanner lands on is `BSL-1.0`, the
  unrelated permissive Boost Software License. The badge now reads `BUSL 1.1`,
  matching the `License-Expression: BUSL-1.1` already carried in package metadata.
  The README is the PyPI long description, so the wrong badge was rendering on the
  public project page. The prose reference to "BSL-licensed" detector families is
  corrected the same way. The LICENSE file itself is unchanged and was always
  correct.

### Added

- `CODE_OF_CONDUCT.md`, the Contributor Covenant 2.1 text already used across the
  sibling Pisama repositories, reported to conduct@pisama.ai.

## [0.3.3] - 2026-07-29

### Fixed

- Declare the license as `BUSL-1.1`, the real SPDX identifier for the Business
  Source License 1.1. The previous `BSL-1.1` is not a valid SPDX identifier at all,
  and the nearest match a scanner lands on is `BSL-1.0`, the permissive Boost
  Software License. Package metadata now carries a machine-readable
  `License-Expression: BUSL-1.1`. The LICENSE file itself is unchanged.
- Repoint the TRAIL citation at the arXiv paper and the HuggingFace dataset. The
  previously cited `github.com/PatronusAI/trail` is a 404.
- Make the README's evidence links absolute. They were relative, so on the
  PyPI-rendered page they resolved to the project page rather than to
  `benchmarks/evidence.json` and `benchmarks/README.md`.
- Remove the per-category precision column from the README's TRAIL table. The
  archive records `fp = 0` in 14 of 14 categories because it scored only annotated
  errors, so precision was 1.000 by construction rather than by measurement and F1
  reduced to `2R/(1+R)`. The table is regenerated from `benchmarks/trail.json` and
  now reports F1, recall and support, with the constraint stated inline.

## [0.3.2] - 2026-07-26

### Added

- Accept structured grounding sources with citation labels in
  `detect_hallucination`.
- Accept provider-reported token counts in `detect_overflow` for callers that
  cannot rely on an offline estimate.
- Exercise statement and branch coverage in CI, including a pinned real-MiniLM
  semantic regression job.

### Changed

- Route `run_all_detectors` to the selected LangGraph, Dify, n8n, or OpenClaw
  adapters when framework metadata is present.
- Pin the optional MiniLM model to an immutable reviewed revision and download
  only its required safe artifacts.
- Count long or special-token text with bounded offline tokenization.
- Require `anthropic>=0.41.0` in the `full` extra so the documented
  `messages.count_tokens` API is available.

### Fixed

- Honor public loop window and similarity options across pairwise and
  clustering paths, validate invalid detector settings, and report correct
  loop origins.
- Count separately supplied output during overflow detection and recognize
  Claude Sonnet 4.6 pricing and context limits.
- Ground claims against matched source clauses, reject unsupported related
  claims, and validate every numbered or named citation.
- Detect direct polarity, numeric, language, and audience-scope conflicts
  without cross-pairing unrelated requirements.

## [0.3.1] - 2026-07-26

### Changed

- Measure coverage across every Python module shipped in the wheel, including
  the frozen turn-aware compatibility namespace.
- Resolve public return annotations at runtime and make the strict type
  contract analyze imported result types.
- Add failure and healthy-path contracts for the complete supported detector
  surface and compatibility checks for every turn-aware failure mode.
- Ship a PEP 561 marker and type-check the supported top-level API in CI.
- Document the legacy turn-aware namespace as frozen and outside the supported
  top-level API.
- Add CodeQL, dependency review, Dependabot, and a coverage regression gate.
- Update pinned CI and trusted-publishing actions.
- Require release tags to match both the package version and the current `main`
  commit before trusted publishing.
- Audit the resolved core dependencies and exercise the documented minimum core
  dependency versions in CI.
- Add positive public-contract tests for every Dify and OpenClaw detector.
- Correct the public cost result and decomposition input type annotations.

### Fixed

- Preserve every item in mixed-marker decomposition lists.
- Allow the benchmark verifier to validate renamed copies by their reviewed
  SHA-256 digest instead of requiring a specific filename.
- Coerce tenant threshold overrides to their declared numeric types.
- Fix all Dify and OpenClaw public functions so they invoke their real
  workflow and session analyzers instead of returning the empty metadata path.

## [0.3.0] - 2026-07-23

### Changed

- Moved embedding and clustering dependencies to the optional `semantic`
  extra, reducing the default installation footprint.
- Deferred scikit-learn loading until semantic clustering is actually needed.
- Reported the installed package version from distribution metadata.
- Added Python 3.13 and complete project URL metadata.

### Fixed

- Replaced the stale CI wheel-content assertion with installed public API and
  documented quickstart checks.
