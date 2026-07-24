# Changelog

All notable changes to `pisama-detectors` are documented here.

## [Unreleased]

## [0.3.0] - 2026-07-23

### Changed

- Ship a PEP 561 marker and type-check the supported top-level API in CI.
- Document the legacy turn-aware namespace as frozen and outside the supported
  top-level API.
- Add CodeQL, dependency review, Dependabot, and a coverage regression gate.
- Update pinned CI and trusted-publishing actions.
- Add positive public-contract tests for every Dify and OpenClaw detector.
- Correct the public cost result and decomposition input type annotations.
- Moved embedding and clustering dependencies to the optional `semantic`
  extra, reducing the default installation footprint.
- Deferred scikit-learn loading until semantic clustering is actually needed.
- Reported the installed package version from distribution metadata.
- Added Python 3.13 and complete project URL metadata.

### Fixed

- Coerce tenant threshold overrides to their declared numeric types.
- Fix all Dify and OpenClaw public functions so they invoke their real
  workflow and session analyzers instead of returning the empty metadata path.
- Replaced the stale CI wheel-content assertion with installed public API and
  documented quickstart checks.
