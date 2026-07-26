# Changelog

All notable changes to `pisama-detectors` are documented here.

## [Unreleased]

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
