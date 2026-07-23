# Changelog

All notable changes to `pisama-detectors` are documented here.

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
