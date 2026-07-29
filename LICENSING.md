# Licensing

This document is for anyone doing license diligence on `pisama-detectors`,
an acquirer, counsel, or a downstream user auditing their dependency tree.
It explains what this package is licensed under, how it relates to the
separate `pisama-core` package, and where the boundary of the license sits.

## What license is this package under

`pisama-detectors` is licensed under the Business Source License 1.1
(BUSL). See [`LICENSE`](LICENSE) for the exact text. In short: it is
source-available, not open source. You may copy, modify, and make
non-production use of it freely, and the Additional Use Grant in the
LICENSE file permits production use as well, as long as that use does not
offer this package's functionality to third parties on a hosted or
embedded basis in competition with Pisama's paid products. On the Change
Date (2030-06-08), the license converts to Apache License 2.0 for the code
as it stood at each released version. For anything not covered by the
Additional Use Grant, a commercial license is available; contact
[team@pisama.ai](mailto:team@pisama.ai).

This document is a plain-language summary for orientation. The LICENSE
file is the controlling legal text; read it directly for exact terms.

## What this package is

`pisama-detectors` ships 42 failure detectors for LLM agent systems: loops,
hallucinations, prompt injection, state corruption, coordination failures,
persona drift, workflow execution bugs, and framework-specific detector
families for LangGraph, Dify, n8n, and OpenClaw. The detectors are
independently implemented, calibrated, and production-tuned, distinct from
the simpler heuristic versions in `pisama-core`.

## How `pisama-core` relates to this package

`Pisama-AI/pisama-core` is a separate package covering much of the same
failure-mode taxonomy (loops, hallucinations, coordination breakdowns, and
more) with simpler, independently implemented heuristic detectors. It is
licensed under MIT, permanently free and permanently open, and is meant to
be a baseline tier anyone can use without restriction.

The two packages are commonly confused because they sound related and
cover the same taxonomy. They are not the same code under two licenses.
They are two independent implementations that happen to detect the same
failure modes. This package does not depend on `pisama-core`, and neither
package's source imports from the other; you can verify this yourself by
grepping either package's source tree for an import of the other, or by
checking that `pisama-core` does not appear in this package's dependency
list in [`pyproject.toml`](pyproject.toml).

### Evidence: independent implementations, not shared code

Line counts below are from a fresh clone of both repositories at their
current `main` branch heads, for every detector module that exists under
the same name in both packages:

| Detector | `pisama-detectors` (BUSL, this package) | `pisama-core` (MIT) |
|---|---|---|
| `hallucination.py` | 1,028 lines | 72 lines |
| `loop.py` | 1,006 lines | 219 lines |
| `coordination.py` | 1,445 lines | 103 lines |
| `persona.py` | 601 lines | 689 lines |
| `specification.py` | 1,594 lines | 1,012 lines |
| `withholding.py` | 711 lines | 619 lines |

These are not a fork with edits, and not one package being a superset of
the other. Reading the two `hallucination.py` files side by side shows
entirely different approaches: this package's version does embedding-based
grounding scoring against retrieved source documents, with citation-pattern
regexes and confidence calibration; `pisama-core`'s version is a short
heuristic over tool-call error rates on an ATIF trace. Neither is a bigger
or smaller version of the other; they solve the same detection problem with
different code. The size relationship is not even consistently
one-directional across the taxonomy (`persona.py` is smaller here than in
`pisama-core`), which is further evidence these are independent
implementations rather than a copy in either direction.

Across the full source trees, `pisama-detectors/src` is about 40,000 lines
and `pisama-core/src` is about 27,000 lines, covering different code paths
throughout, not a shared core with a thin diff.

## Where the BUSL boundary sits

The obligations in this package's LICENSE apply only to this package's own
code: the source under `pisama-detectors/src`, as distributed in this
repository and on PyPI. They do not extend to `pisama-core`, do not apply
retroactively to any earlier or later `pisama-core` release, and do not
apply by association just because both packages are published by Pisama
LLC and cover related ground. If you use `pisama-core` on its own, the
BUSL Additional Use Grant, the non-compete condition, and the Change Date
are irrelevant to you; the only license that applies is `pisama-core`'s own
MIT license.

## Questions

If you are doing diligence and something here does not match what you find
in the source, or you need a commercial license for a use case the
Additional Use Grant does not cover, email
[team@pisama.ai](mailto:team@pisama.ai).
