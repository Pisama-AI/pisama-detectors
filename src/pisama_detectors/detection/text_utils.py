"""Text-normalization helpers shared across detectors.

These utilities existed as private copies in four separate detector
modules (completion, decomposition, derailment). Consolidating avoids
drift when one variant gets fixed and the others don't.

Each helper takes the caller's suffix list as a parameter so detectors
can keep their tuned lists — the point is to share the loop, not to
impose a single canonical suffix set.
"""

from __future__ import annotations

from typing import Sequence


def strip_suffix(
    word: str,
    suffixes: Sequence[str],
    *,
    min_remainder: int = 3,
) -> str:
    """Strip the first matching suffix if the remainder is long enough.

    Suffixes are tried in the order given — put longer/more-specific
    suffixes first ("ation" before "tion") to avoid partial matches.
    """
    for sfx in suffixes:
        if word.endswith(sfx) and len(word) - len(sfx) >= min_remainder:
            return word[: -len(sfx)]
    return word


def strip_plural_s(word: str, *, min_length: int = 4) -> str:
    """Drop a trailing 's' if the word is long enough and not a 'ss' word.

    Mirrors the behavior most detectors want: 'cats' -> 'cat' but
    'class' stays 'class'. 'processes' is handled separately via
    strip_ses_plural — don't call both without checking order.
    """
    if len(word) >= min_length and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def strip_ses_plural(word: str, *, min_length: int = 6) -> str:
    """Turn '…ses' into '…s' (e.g. 'processes' -> 'process').

    Only fires when the full length is >= min_length so short words
    ending in 'ses' (like 'uses') aren't over-stemmed.
    """
    if len(word) >= min_length and word.endswith("ses"):
        return word[:-2]
    return word
