"""Behavior tests for the corruption detector's reconciled fixes.

Covers two families of behavior ported from the production backend:

* lossless type coercions ("68.00" -> 68.0) are normalisation, not corruption,
  while coercions that change the value still fire as ``type_drift``;
* the v1.2 information-contamination-propagation signal, which flags a
  downstream agent that strips the hedges, contradictions, or low-confidence
  markers an upstream agent attached to a claim.

Every test drives the real ``SemanticCorruptionDetector`` (or the public
``pisama_detectors.detect_corruption`` entry point) with hand-written payloads.
"""

from __future__ import annotations

from typing import Any

import pytest

import pisama_detectors as pd
from pisama_detectors.detection.corruption import (
    ContaminationProfile,
    CorruptionResult,
    SemanticCorruptionDetector,
    StateSnapshot,
)

HEDGED = (
    "The migration might have succeeded, but the result is unverified and "
    "preliminary. Server status is possibly online."
)
STRIPPED = "The migration succeeded. Server status is online."


def _issues(previous: dict[str, Any], current: dict[str, Any]) -> list[tuple[str, Any]]:
    detector = SemanticCorruptionDetector()
    found = detector.detect_corruption(
        StateSnapshot(state_delta=previous, agent_id="etl"),
        StateSnapshot(state_delta=current, agent_id="etl"),
    )
    return [(issue.issue_type, issue.field) for issue in found]


# ---------------------------------------------------------------------------
# Lossless coercion is normalisation, value-changing coercion is corruption
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        ({"price": "68.00"}, {"price": 68.0}),
        ({"qty": "2"}, {"qty": 2}),
        ({"flag": "true"}, {"flag": True}),
    ],
)
def test_value_preserving_type_change_is_not_type_drift(
    previous: dict[str, Any], current: dict[str, Any]
) -> None:
    assert ("type_drift", next(iter(previous))) not in _issues(previous, current)
    assert not pd.detect_corruption(previous, current).detected


@pytest.mark.parametrize(
    ("previous", "current", "field"),
    [
        # Same type change as above, but the value is altered on the way.
        ({"price": "68.00"}, {"price": 0}, "price"),
        ({"price": "68.00"}, {"price": True}, "price"),
        ({"x": "abc"}, {"x": 5}, "x"),
        # Zero-padded strings are identifiers, so dropping the padding is lossy.
        ({"zip": "02134"}, {"zip": 2134}, "zip"),
    ],
)
def test_value_changing_type_change_still_reports_type_drift(
    previous: dict[str, Any], current: dict[str, Any], field: str
) -> None:
    assert ("type_drift", field) in _issues(previous, current)
    assert pd.detect_corruption(previous, current).detected


def test_nested_value_preserving_coercion_is_silent_but_structural_change_fires() -> None:
    benign = _issues({"json": {"qty": "2"}}, {"json": {"qty": 2}})
    corrupt = _issues({"json": {"qty": "2"}}, {"json": {"qty": [1]}})

    assert benign == []
    assert ("type_drift", "json.qty") in corrupt


def test_type_change_is_reported_once_per_field() -> None:
    result = pd.detect_corruption({"x": "abc"}, {"x": 5})

    drift = [i for i in result.issues if i.issue_type == "type_drift" and i.field == "x"]
    assert len(drift) == 1


# ---------------------------------------------------------------------------
# Contamination text extraction
# ---------------------------------------------------------------------------


def test_contamination_text_joins_only_non_empty_string_content_keys() -> None:
    detector = SemanticCorruptionDetector()

    text = detector._extract_contamination_text(
        {
            "summary": "third",
            "output": "first",
            "content": "",
            "text": 42,
            "unrelated": "ignored",
            "answer": "second",
        }
    )

    # Order follows the detector's key priority, not dict insertion order.
    assert text == "first\nsecond\nthird"
    assert detector._extract_contamination_text({"unrelated": "x", "text": None}) == ""


def test_contamination_text_reads_every_content_key_in_priority_order() -> None:
    detector = SemanticCorruptionDetector()
    priority = [
        "output",
        "content",
        "text",
        "answer",
        "response",
        "summary",
        "current_state",
        "prev_state",
    ]

    # Insert in reverse so the result can only come from the key priority.
    text = detector._extract_contamination_text({key: f"<{key}>" for key in reversed(priority)})

    assert text == "\n".join(f"<{key}>" for key in priority)


# ---------------------------------------------------------------------------
# Contamination profile
# ---------------------------------------------------------------------------


def test_profile_of_empty_text_is_all_zero() -> None:
    profile = SemanticCorruptionDetector().compute_contamination_profile("")

    assert profile == ContaminationProfile(0, 0, 0, 0.0)


def test_profile_counts_hedges_and_normalizes_by_token_count() -> None:
    detector = SemanticCorruptionDetector()

    profile = detector.compute_contamination_profile(HEDGED)

    # might, unverified, preliminary, possibly
    assert profile.hedge_count == 4
    assert profile.contradiction_count == 0
    assert profile.low_confidence_count == 0
    assert profile.normalized_score == pytest.approx(4 / 17)
    assert detector.compute_contamination_profile(STRIPPED).hedge_count == 0


def test_profile_counts_low_confidence_sources_separately_from_hedges() -> None:
    profile = SemanticCorruptionDetector().compute_contamination_profile(
        "The unverified sources say the vendor was allegedly breached."
    )

    # "unverified sources" and "allegedly" are low-confidence sourcing markers.
    assert profile.low_confidence_count == 2
    # "unverified" is also a plain hedge word, counted in its own bucket.
    assert profile.hedge_count == 1
    assert profile.contradiction_count == 0


def test_profile_weights_contradictions_double_in_normalized_score() -> None:
    detector = SemanticCorruptionDetector()

    profile = detector.compute_contamination_profile(
        "The service is online. Later the service is offline."
    )

    assert profile.contradiction_count == 1
    assert profile.hedge_count == 0
    tokens = 9  # the service is online later the service is offline
    assert profile.normalized_score == pytest.approx(2 / tokens)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The job succeeded and then the job failed.", 1),
        ("The claim was confirmed and later denied.", 1),
        ("The claim is true. The claim is false.", 1),
        ("The node is online. The node is offline.", 1),
        ("The API is available. The API is unavailable.", 1),
        ("The build is not ready. The build is ready.", 1),
        # Look-alikes that are not contradictions.
        ("The build is not ready.", 0),
        ("The API is unavailable.", 0),
        ("The job succeeded.", 0),
    ],
)
def test_contradiction_counting_requires_both_sides(text: str, expected: int) -> None:
    assert SemanticCorruptionDetector()._count_contradictions(text) == expected


def test_contradiction_counting_handles_was_true_false_and_bare_not_targets() -> None:
    detector = SemanticCorruptionDetector()

    # The "was true / was false" pair is its own template, distinct from "is".
    assert detector._count_contradictions("The claim was true. Later the claim was false.") == 1
    # "not X" plus a separate bare "X" is one contradiction per distinct target.
    assert detector._count_contradictions("Not ready for release. Later it is ready.") == 1
    assert detector._count_contradictions("Not ready. Not stable. Ready and stable.") == 2
    # "not X" with no bare X elsewhere stays silent.
    assert detector._count_contradictions("The build is not ready and not stable.") == 0
    # The same negated target repeated is still one contradiction.
    assert detector._count_contradictions("Not ready now. Not ready still. Ready later.") == 1
    # Only content words of four letters or more are matched; "use" is too short.
    assert detector._count_contradictions("It is not done. Later it is done.") == 1
    assert detector._count_contradictions("Do not use it. Then use it.") == 0


def test_hedge_and_low_confidence_counters_are_case_insensitive() -> None:
    detector = SemanticCorruptionDetector()

    assert detector._count_hedges("MAYBE this is PROBABLY fine") == 2
    assert detector._count_hedges("This is fine.") == 0
    assert detector._count_low_confidence("Anonymous sources say so; ALLEGEDLY.") == 2
    assert detector._count_low_confidence("The audit log says so.") == 0
    # Contradiction matching is case-insensitive on both sides of the pair.
    assert detector._count_contradictions("The job SUCCEEDED, then it FAILED.") == 1
    assert detector._count_contradictions("NOT READY. Later READY.") == 1


@pytest.mark.parametrize(
    "phrase",
    [
        "It might work.",
        "Maybe it works.",
        "Perhaps it works.",
        "It works possibly.",
        "It probably works.",
        "The figure is unverified.",
        "A preliminary figure.",
        "It could be fine.",
        "It appears to be fine.",
        "They appear to be fine.",
        "It seems to work.",
        "They seem to work.",
        "Approximately 40 units.",
        "Roughly 40 units.",
        "As far as I can tell, it works.",
        "As far as we can tell, it works.",
        "I am not entirely sure it works.",
        "I am not fully certain it works.",
        "The cause is uncertain.",
        "A tentative plan.",
        "A suspected leak.",
        "The cause is unclear.",
        "The wording is ambiguous.",
    ],
)
def test_each_hedge_phrase_counts_exactly_once(phrase: str) -> None:
    detector = SemanticCorruptionDetector()

    assert detector._count_hedges(phrase) == 1
    # Hedges are a separate signal from low-confidence sourcing.
    assert detector._count_low_confidence(phrase) == 0


@pytest.mark.parametrize(
    "phrase",
    [
        "He could not go.",
        "It is certain and clear.",
        "A mighty result.",
        "The seam is tight.",
        "Approximate values.",
        "It appears fine.",
    ],
)
def test_hedge_look_alikes_are_not_counted(phrase: str) -> None:
    assert SemanticCorruptionDetector()._count_hedges(phrase) == 0


@pytest.mark.parametrize(
    "phrase",
    [
        "Unverified sources say so.",
        "An unverified source says so.",
        "It was rumored.",
        "Rumor has it.",
        "Rumors have it.",
        "Allegedly true.",
        "It has been suggested.",
        "It is suggested.",
        "Anonymous source says so.",
        "Anonymous sources say so.",
        "Unconfirmed reports say so.",
        "An unconfirmed report says so.",
        "According to leaked memos.",
        "According to a leaked memo.",
        "Word on the street is so.",
    ],
)
def test_each_low_confidence_phrase_counts_exactly_once(phrase: str) -> None:
    assert SemanticCorruptionDetector()._count_low_confidence(phrase) == 1


@pytest.mark.parametrize(
    "phrase",
    ["Sources say so.", "The report was confirmed.", "The memo leaked."],
)
def test_low_confidence_look_alikes_are_not_counted(phrase: str) -> None:
    assert SemanticCorruptionDetector()._count_low_confidence(phrase) == 0


# ---------------------------------------------------------------------------
# Content overlap ignores the hedge vocabulary itself
# ---------------------------------------------------------------------------


def test_content_jaccard_ignores_hedge_words() -> None:
    detector = SemanticCorruptionDetector()

    # Only hedge words differ, so the content is identical.
    assert detector._content_jaccard(
        "Rollout completed successfully, possibly, perhaps.",
        "Rollout completed successfully.",
    ) == 1.0
    # The stripped output is a strict subset of the hedged content tokens.
    assert detector._content_jaccard(HEDGED, STRIPPED) == pytest.approx(7 / 11)


def test_content_jaccard_is_zero_for_disjoint_or_hedge_only_text() -> None:
    detector = SemanticCorruptionDetector()

    assert detector._content_jaccard(HEDGED, "Quarterly revenue grew twelve percent.") == 0.0
    assert detector._content_jaccard("maybe perhaps possibly", STRIPPED) == 0.0
    assert detector._content_jaccard("", STRIPPED) == 0.0


# ---------------------------------------------------------------------------
# Pairwise propagation score
# ---------------------------------------------------------------------------


def test_stripping_all_markers_from_hedged_input_is_high_severity() -> None:
    detector = SemanticCorruptionDetector()

    score, issues = detector._compute_contamination_propagation_score(HEDGED, STRIPPED)

    assert score == pytest.approx(0.8182)
    assert len(issues) == 1
    issue = issues[0]
    assert issue.issue_type == "contamination_stripping"
    assert issue.field is None
    assert issue.severity == "high"
    assert "input had 4 markers" in issue.message
    assert "output has 0" in issue.message
    assert "drop 100%" in issue.message
    assert "content overlap 64%" in issue.message


def test_identical_content_with_markers_removed_scores_one() -> None:
    score, issues = SemanticCorruptionDetector()._compute_contamination_propagation_score(
        "Rollout completed successfully, possibly, perhaps.",
        "Rollout completed successfully.",
    )

    assert score == 1.0
    assert issues[0].severity == "high"


def test_single_marker_stripped_is_medium_severity_not_high() -> None:
    detector = SemanticCorruptionDetector()

    score, issues = detector._compute_contamination_propagation_score(
        "The deploy might have finished. Server status is online.",
        "The deploy finished. Server status is online.",
    )

    # High severity needs at least two markers in the input.
    assert score == pytest.approx(0.9375)
    assert [issue.severity for issue in issues] == ["medium"]
    assert "input had 1 markers" in issues[0].message


def test_partial_strip_at_half_drop_is_medium_with_lower_score() -> None:
    detector = SemanticCorruptionDetector()
    half_stripped = (
        "The migration might have succeeded, but the result is unverified. "
        "Server status is online."
    )

    score, issues = detector._compute_contamination_propagation_score(HEDGED, half_stripped)

    # drop 0.5 * (0.5 + 0.5 * overlap 10/11): exactly the 50% threshold still fires.
    assert score == pytest.approx(0.4773)
    assert issues[0].severity == "medium"
    assert "output has 2" in issues[0].message
    assert "drop 50%" in issues[0].message


def test_upstream_contradiction_stripped_downstream_is_flagged() -> None:
    detector = SemanticCorruptionDetector()

    score, issues = detector._compute_contamination_propagation_score(
        "The service is online. Later the service is offline.",
        "The service is online.",
    )

    # drop 100% * (0.5 + 0.5 * overlap 4/6)
    assert score == pytest.approx(0.8333)
    assert issues[0].issue_type == "contamination_stripping"
    # A single upstream marker is never high severity.
    assert issues[0].severity == "medium"
    assert "input had 1 markers (h=0, c=1, lc=0)" in issues[0].message
    assert "output has 0" in issues[0].message


def test_low_confidence_sources_stripped_downstream_are_flagged() -> None:
    detector = SemanticCorruptionDetector()

    score, issues = detector._compute_contamination_propagation_score(
        "Anonymous sources say the vendor was breached.",
        "The vendor was breached.",
    )

    assert score == pytest.approx(0.8333)
    assert issues[0].issue_type == "contamination_stripping"
    assert issues[0].severity == "medium"
    assert "input had 1 markers (h=0, c=0, lc=1)" in issues[0].message
    assert "output has 0" in issues[0].message


def test_content_overlap_bar_is_forty_percent_inclusive() -> None:
    detector = SemanticCorruptionDetector()

    # Content tokens {alpha, beta} vs {alpha, gamma}: overlap 1/3, below the bar,
    # so the output is treated as different content rather than stripped input.
    assert detector._compute_contamination_propagation_score(
        "Perhaps alpha beta.", "Alpha gamma."
    ) == (0.0, [])

    # {alpha, beta, gamma, delta} vs {alpha, beta, epsilon}: overlap 2/5, exactly the bar.
    score, issues = detector._compute_contamination_propagation_score(
        "Perhaps alpha beta gamma delta.", "Alpha beta epsilon."
    )
    assert score == pytest.approx(0.7)  # drop 100% * (0.5 + 0.5 * 0.4)
    assert [issue.severity for issue in issues] == ["medium"]
    assert "content overlap 40%" in issues[0].message


TEN_HEDGES = (
    "The rollout finished. It might be maybe perhaps possibly probably "
    "unverified preliminary approximately roughly tentative."
)


@pytest.mark.parametrize(
    ("current", "remaining", "drop", "severity", "score"),
    [
        # 10 markers down to 1 is a 90% drop: exactly the high-severity bar.
        ("The rollout finished. It might be.", 1, "90%", "high", 0.9),
        # 10 down to 2 is an 80% drop: stripped, but only medium severity.
        ("The rollout finished. It might be maybe.", 2, "80%", "medium", 0.8),
    ],
)
def test_high_severity_needs_a_drop_of_at_least_ninety_percent(
    current: str, remaining: int, drop: str, severity: str, score: float
) -> None:
    detector = SemanticCorruptionDetector()

    assert detector._count_hedges(TEN_HEDGES) == 10
    assert detector._count_hedges(current) == remaining

    got, issues = detector._compute_contamination_propagation_score(TEN_HEDGES, current)

    assert got == pytest.approx(score)
    assert [issue.severity for issue in issues] == [severity]
    assert f"output has {remaining} (drop {drop}" in issues[0].message


def test_retained_contradictions_count_as_surviving_uncertainty() -> None:
    detector = SemanticCorruptionDetector()
    upstream = (
        "The service might be online, but later the service is offline. "
        "The job succeeded, then the job failed."
    )
    keeps_contradictions = (
        "The service is online, but later the service is offline. "
        "The job succeeded, then the job failed."
    )
    drops_everything = "The service is online. The job succeeded."

    # 1 hedge + 2 contradictions upstream; only the hedge is dropped (33%).
    assert detector.compute_contamination_profile(upstream).contradiction_count == 2
    assert detector._compute_contamination_propagation_score(upstream, keeps_contradictions) == (
        0.0,
        [],
    )
    # Dropping every marker from the same upstream text is flagged.
    score, issues = detector._compute_contamination_propagation_score(
        upstream, drops_everything
    )
    assert score == pytest.approx(0.75)
    assert [issue.severity for issue in issues] == ["high"]
    assert "input had 3 markers (h=1, c=2, lc=0), output has 0" in issues[0].message


def test_retained_low_confidence_sources_count_as_surviving_uncertainty() -> None:
    detector = SemanticCorruptionDetector()
    upstream = "Anonymous sources say the vendor might have been breached, allegedly."
    keeps_sourcing = "Anonymous sources say the vendor was breached, allegedly."
    drops_everything = "Sources say the vendor was breached."

    # 1 hedge + 2 low-confidence markers upstream; only the hedge is dropped (33%).
    assert detector.compute_contamination_profile(upstream).low_confidence_count == 2
    assert detector._compute_contamination_propagation_score(upstream, keeps_sourcing) == (
        0.0,
        [],
    )
    # Dropping every marker from the same upstream text is flagged.
    score, issues = detector._compute_contamination_propagation_score(
        upstream, drops_everything
    )
    assert score == pytest.approx(0.8125)
    assert [issue.severity for issue in issues] == ["high"]
    assert "input had 3 markers (h=1, c=0, lc=2), output has 0" in issues[0].message


@pytest.mark.parametrize(
    ("previous", "current"),
    [
        # Nothing to propagate: the input carried no uncertainty.
        ("The migration succeeded.", "The migration succeeded."),
        ("The migration succeeded.", "The migration might have succeeded."),
        # Uncertainty survives (drop below 50%).
        (
            HEDGED,
            "The migration might have succeeded, but the result is unverified and "
            "preliminary. Server status is online.",
        ),
        # A one-in-three drop (3 markers down to 2) is still below the 50% bar.
        (
            "The deploy might have finished, and the result is unverified and "
            "preliminary. Server status is online.",
            "The deploy might have finished, and the result is unverified. "
            "Server status is online.",
        ),
        # The output is about something else, so nothing was "stripped".
        (HEDGED, "Quarterly revenue grew twelve percent across all regions."),
        # Missing sides.
        ("", STRIPPED),
        (HEDGED, ""),
    ],
)
def test_propagation_score_stays_silent_when_nothing_was_stripped(
    previous: str, current: str
) -> None:
    score, issues = SemanticCorruptionDetector()._compute_contamination_propagation_score(
        previous, current
    )

    assert score == 0.0
    assert issues == []


# ---------------------------------------------------------------------------
# Chain detection
# ---------------------------------------------------------------------------


def test_chain_needs_at_least_two_states() -> None:
    detector = SemanticCorruptionDetector()

    for states in ([], [HEDGED]):
        result = detector.detect_contamination_propagation(states)
        assert isinstance(result, CorruptionResult)
        assert not result.detected
        assert result.confidence == 0.0
        assert result.issues == []
        assert result.issue_count == 0
        assert result.max_severity == "none"


def test_chain_with_uncertainty_preserved_end_to_end_is_clean() -> None:
    result = SemanticCorruptionDetector().detect_contamination_propagation(
        [HEDGED, HEDGED, HEDGED]
    )

    assert not result.detected
    assert result.confidence == 0.0
    assert result.max_severity == "none"


def test_chain_reports_the_hop_that_strips_uncertainty() -> None:
    result = SemanticCorruptionDetector().detect_contamination_propagation(
        [HEDGED, STRIPPED, STRIPPED]
    )

    # Hop 2 -> 3 has no markers on the input side and adds nothing.
    assert result.detected
    assert result.issue_count == 1
    assert result.max_severity == "high"
    assert result.raw_score == pytest.approx(0.8182)
    assert result.confidence == pytest.approx(0.8182)


def test_chain_takes_worst_severity_and_best_score_across_hops() -> None:
    hedged_once = "The deploy might have finished. Server status is online."
    stripped_once = "The deploy finished. Server status is online."

    result = SemanticCorruptionDetector().detect_contamination_propagation(
        [hedged_once, stripped_once, HEDGED, STRIPPED]
    )

    # Hops: medium (score 0.9375), skipped (no markers on input), high (0.8182).
    assert result.issue_count == 2
    assert [issue.severity for issue in result.issues] == ["medium", "high"]
    assert result.max_severity == "high"
    assert result.raw_score == pytest.approx(0.9375)


def test_chain_best_score_does_not_depend_on_hop_order() -> None:
    hedged_once = "The deploy might have finished. Server status is online."
    stripped_once = "The deploy finished. Server status is online."
    detector = SemanticCorruptionDetector()

    # Weaker hop first (0.8182), stronger hop last (0.9375): the max is not the first hop.
    ascending = detector.detect_contamination_propagation(
        [HEDGED, STRIPPED, hedged_once, stripped_once]
    )
    # Stronger hop first, weaker hop last: the max is not the last hop either.
    descending = detector.detect_contamination_propagation(
        [hedged_once, stripped_once, HEDGED, STRIPPED]
    )

    assert ascending.raw_score == pytest.approx(0.9375)
    assert descending.raw_score == pytest.approx(0.9375)
    assert [i.severity for i in ascending.issues] == ["high", "medium"]
    assert ascending.max_severity == descending.max_severity == "high"
    assert ascending.confidence == pytest.approx(0.9375)


def test_chain_confidence_follows_scaling_and_is_capped() -> None:
    perfect_overlap = [
        "Rollout completed successfully, possibly, perhaps.",
        "Rollout completed successfully.",
    ]

    unscaled = SemanticCorruptionDetector().detect_contamination_propagation(perfect_overlap)
    halved = SemanticCorruptionDetector(
        confidence_scaling=0.5
    ).detect_contamination_propagation(perfect_overlap)
    doubled = SemanticCorruptionDetector(
        confidence_scaling=2.0
    ).detect_contamination_propagation(perfect_overlap)

    assert unscaled.raw_score == 1.0
    assert unscaled.confidence == 0.99
    assert halved.confidence == 0.5
    assert doubled.confidence == 0.99
    assert halved.raw_score == unscaled.raw_score

    # Confidence is rounded to four places: 0.8182 * 0.3 = 0.24546.
    odd = SemanticCorruptionDetector(
        confidence_scaling=0.3
    ).detect_contamination_propagation([HEDGED, STRIPPED])
    assert odd.raw_score == pytest.approx(0.8182)
    assert odd.confidence == 0.2455


# ---------------------------------------------------------------------------
# End-to-end through the public API
# ---------------------------------------------------------------------------


def test_public_detector_flags_hedges_stripped_between_agent_outputs() -> None:
    result = pd.detect_corruption({"output": HEDGED}, {"output": STRIPPED})

    assert result.detected
    assert result.max_severity == "high"
    stripping = [i for i in result.issues if i.issue_type == "contamination_stripping"]
    assert len(stripping) == 1
    assert stripping[0].field is None
    assert result.confidence > 0.0


def test_public_detector_is_silent_when_downstream_keeps_the_hedges() -> None:
    result = pd.detect_corruption({"output": HEDGED}, {"output": HEDGED})

    assert not result.detected
    assert result.issues == []
    assert result.max_severity == "none"
    assert result.confidence == 0.0


def test_public_detector_reads_text_from_any_contamination_key() -> None:
    result = pd.detect_corruption({"summary": HEDGED}, {"answer": STRIPPED})

    assert any(issue.issue_type == "contamination_stripping" for issue in result.issues)


def test_public_detector_ignores_uncertainty_outside_text_keys() -> None:
    result = pd.detect_corruption({"notes": HEDGED}, {"notes": STRIPPED})

    assert not any(issue.issue_type == "contamination_stripping" for issue in result.issues)
