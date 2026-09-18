"""Behavioral contracts for the loop detector fixes reconciled from the backend.

Every case drives the real ``MultiLevelLoopDetector`` (or one of its pure
helpers) with hand-written snapshots. Nothing is patched and no model output is
simulated. Cases are paired: the risky trace fires and its benign look-alike
stays silent, so a regression in either direction fails a test.

CI installs only ``.[dev]``, so the semantic embedder is normally absent and the
empty-delta structural tier falls back to word overlap. The one test that needs
the embedder is isolated and skipped when the extra is missing.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest

from pisama_detectors.detection.loop import (
    LoopDetectionResult,
    MultiLevelLoopDetector,
    StateSnapshot,
    _has_terminal_progress_after,
    _repeats_driven_by_distinct_inputs,
    _strip_non_input_fields,
    _structured_external_input_fingerprint,
)
from pisama_detectors.detection.shared_embedder import get_shared_embedder

Spec = tuple  # (agent_id, state_delta, content)

SCREEN = (
    "Screen the attached contract for indemnification clauses and report "
    "the verdict to the legal team."
)
INTAKE_PROSE = [
    "Received the first document from the client portal.",
    "Second upload arrived through the email gateway.",
    "A scanned bundle landed via courier scan.",
]
DIVERSE_CONTENT = [
    "Outline the database schema for orders",
    "Choose queue technology for ingestion",
    "Draft rollout schedule for August",
    "Estimate hosting budget for the launch",
    "Review the incident postmortem template",
    "Rotate the expiring TLS certificates tonight",
]


def snaps(*specs: Spec) -> List[StateSnapshot]:
    return [
        StateSnapshot(agent_id=agent, state_delta=delta, content=content, sequence_num=index)
        for index, (agent, delta, content) in enumerate(specs)
    ]


def detect(states: List[StateSnapshot], **kwargs: Any) -> LoopDetectionResult:
    return MultiLevelLoopDetector(**kwargs).detect_loop(states)


def screening_trace(
    inputs: List[Dict[str, Any]],
    tail: Optional[Spec] = None,
) -> List[StateSnapshot]:
    """A screener that repeats one request, with one intake state between repeats."""
    specs: List[Spec] = []
    for index, intake_input in enumerate(inputs):
        specs.append(("screener", {}, SCREEN))
        specs.append(("intake", intake_input, INTAKE_PROSE[index]))
    specs.append(tail or ("archivist", {}, "Filing the paperwork into the shared drive."))
    return snaps(*specs)


# --------------------------------------------------------------------------
# Input-payload helpers behind the fan-out exemption
# --------------------------------------------------------------------------


def test_strip_non_input_fields_drops_coordination_telemetry_and_counters() -> None:
    payload = {
        "query": "inventory SKU-1001",
        "Round": 3,
        "handoff_target": "reviewer",
        "call_id": "c-17",
        "attempt_num": 5,
        "message": "passing this along",
        "nested": {"turn_count": 2, "doc": "d1", "content": "chatter"},
        "empty_after_strip": {"round": 1},
        # Counter-shaped names that are not in the fixed key list are dropped too.
        "fetch_attempt": 2,
        "review_round": 4,
        "page_step_num": 3,
        "retry_budget": 3,
        # Look-alikes that merely start with a counter word are real input.
        "turnover": 7,
        "stepwise": "yes",
        "roundtrip_ms": 40,
    }

    assert _strip_non_input_fields(payload) == {
        "query": "inventory SKU-1001",
        "nested": {"doc": "d1"},
        # "retry_budget" is configuration, not a counter, so it is real input.
        "retry_budget": 3,
        "turnover": 7,
        "stepwise": "yes",
        "roundtrip_ms": 40,
    }


def test_strip_non_input_fields_recurses_lists_and_keeps_original_key_spelling() -> None:
    payload = {"Docs": [{"round": 1}, {"path": "a.txt"}, None, "", {}], "Speaker": "x"}

    assert _strip_non_input_fields(payload) == {"Docs": [{"path": "a.txt"}]}
    assert _strip_non_input_fields(["kept", None, ""]) == ["kept"]
    assert _strip_non_input_fields(42) == 42


def test_external_input_fingerprint_is_canonical_json_of_the_work_fields() -> None:
    first = StateSnapshot("intake", {"round": 1, "page": 2, "doc": "b", "status": "ok"}, "", 0)
    reordered = StateSnapshot("intake", {"doc": "b", "turn": 9, "page": 2}, "no failures reported", 1)
    expected = json.dumps({"doc": "b", "page": 2})

    # Keys come out sorted, so insertion order never changes the fingerprint.
    assert _structured_external_input_fingerprint(first) == expected
    assert _structured_external_input_fingerprint(reordered) == expected


@pytest.mark.parametrize(
    ("delta", "content"),
    [
        ({}, "peer prose only"),
        ({"round": 1, "note": "hand-off"}, ""),
        ({"doc": "b", "status": "failed"}, ""),
        ({"doc": "b"}, "ValueError: could not parse the payload"),
        ({"doc": "request failed with a timeout"}, ""),
    ],
    ids=["empty-delta", "only-coordination", "failed-status", "failure-content", "failure-in-input"],
)
def test_external_input_fingerprint_is_none_without_trustworthy_work(
    delta: Dict[str, Any], content: str
) -> None:
    assert _structured_external_input_fingerprint(StateSnapshot("intake", delta, content, 0)) is None


def test_terminal_progress_follows_latest_verdict_of_the_repeating_agent() -> None:
    succeeded_last = snaps(
        ("a", {}, "x"), ("a", {"status": "failed"}, "y"), ("a", {"status": "success"}, "z")
    )
    failed_last = snaps(
        ("a", {}, "x"), ("a", {"status": "success"}, "y"), ("a", {"status": "failed"}, "z")
    )

    assert _has_terminal_progress_after(succeeded_last, 0, "a") is True
    assert _has_terminal_progress_after(failed_last, 0, "a") is False


def test_terminal_progress_ignores_other_agents_success() -> None:
    other_agent_ok = snaps(("a", {}, "x"), ("b", {"status": "success"}, "All 5 files processed"))

    assert _has_terminal_progress_after(other_agent_ok, 0, "a") is False


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("All 5 contracts reviewed and cleared.", True),
        ("three findings submitted to legal", True),
        ("5 contracts reviewed, then a ValueError was raised", False),
        ("The contract was reviewed by legal", False),
        ("Still working, making good progress here", False),
        ("The peer has 4 items screened", False),
    ],
)
def test_terminal_progress_text_fallback_needs_quantified_completion(
    content: str, expected: bool
) -> None:
    agent = "b" if content.startswith("The peer") else "a"
    states = snaps(("a", {}, "x"), (agent, {}, content))

    assert _has_terminal_progress_after(states, 0, "a") is expected


def test_repeats_need_two_indices_to_count_as_distinct_inputs() -> None:
    states = snaps(("a", {}, SCREEN), ("b", {"doc": "x"}, "input"))

    assert _repeats_driven_by_distinct_inputs(states, [0]) is False
    assert _repeats_driven_by_distinct_inputs(states, []) is False


def test_repeats_driven_by_distinct_structured_inputs_in_every_interval() -> None:
    states = snaps(
        ("a", {}, SCREEN),
        ("b", {"doc": "alpha", "round": 1}, "one"),
        ("a", {}, SCREEN),
        ("b", {"doc": "beta", "round": 2}, "two"),
        ("a", {}, SCREEN),
    )

    assert _repeats_driven_by_distinct_inputs(states, [0, 2, 4]) is True


def test_every_interval_needs_its_own_input_not_just_some_variety() -> None:
    def four_repeats(docs: List[str]) -> List[StateSnapshot]:
        specs: List[Spec] = []
        for doc in docs:
            specs.append(("a", {}, SCREEN))
            specs.append(("b", {"doc": doc}, f"prose about {doc}"))
        specs.append(("a", {}, SCREEN))
        return snaps(*specs)

    all_distinct = four_repeats(["alpha", "beta", "gamma"])
    revisited = four_repeats(["alpha", "beta", "alpha"])

    assert _repeats_driven_by_distinct_inputs(all_distinct, [0, 2, 4, 6]) is True
    # Input "alpha" arrived before two different repeats: that is a revisit, not fan-out.
    assert _repeats_driven_by_distinct_inputs(revisited, [0, 2, 4, 6]) is False


@pytest.mark.parametrize(
    "intakes",
    [
        [{"doc": "alpha", "round": 1}, {"doc": "alpha", "round": 2}],
        [{"round": 1}, {"round": 2}],
        [{"doc": "alpha"}, {"round": 2}],
        [{"doc": "alpha"}, {"doc": "beta", "status": "failed"}],
    ],
    ids=["identical-input", "counters-only", "one-interval-empty", "failed-input"],
)
def test_repeats_without_distinct_trustworthy_inputs_stay_loop_evidence(
    intakes: List[Dict[str, Any]],
) -> None:
    states = snaps(
        ("a", {}, SCREEN),
        ("b", intakes[0], "prose one"),
        ("a", {}, SCREEN),
        ("b", intakes[1], "different prose two"),
        ("a", {}, SCREEN),
    )

    assert _repeats_driven_by_distinct_inputs(states, [0, 2, 4]) is False


def test_repeating_agents_own_delta_is_not_an_external_input() -> None:
    states = snaps(
        ("a", {}, SCREEN),
        ("a", {"doc": "alpha"}, "own note"),
        ("a", {}, SCREEN),
        ("a", {"doc": "beta"}, "own note two"),
        ("a", {}, SCREEN),
    )

    assert _repeats_driven_by_distinct_inputs(states, [0, 2, 4]) is False


def test_terminal_progress_after_last_repeat_counts_as_distinct_work() -> None:
    states = snaps(("a", {}, SCREEN), ("a", {}, SCREEN), ("a", {"status": "completed"}, "Wrapped."))

    assert _repeats_driven_by_distinct_inputs(states, [0, 1]) is True


# --------------------------------------------------------------------------
# v1.9 content fingerprint tier
# --------------------------------------------------------------------------


def test_content_fingerprint_flags_an_agent_repeating_identical_output() -> None:
    states = snaps(
        ("reviewer", {}, SCREEN),
        ("programmer", {}, "Patched the currency handling in payments.py"),
        ("reviewer", {}, SCREEN),
        ("programmer", {}, "Added a null guard around the currency lookup."),
        ("reviewer", {}, SCREEN),
        ("tester", {}, "Running the whole suite again to see what is left."),
    )

    result = detect(states)

    assert result.detected
    assert result.method == "content_fingerprint"
    assert result.loop_start_index == 0
    assert result.loop_length == 4
    assert result.raw_score == pytest.approx(0.9)
    assert result.confidence == pytest.approx(0.925)
    assert result.cost == 0.0
    assert result.evidence == {
        "exact_content_matches": 3,
        "content_preview": SCREEN[:40],
        "loop_agent": "reviewer",
        "total_states": 6,
    }


def test_content_fingerprint_confidence_grows_with_each_exact_repeat() -> None:
    detector = MultiLevelLoopDetector()

    def trace(repeats: int, third_prefix_twin: bool = False) -> List[StateSnapshot]:
        specs: List[Spec] = []
        for index in range(repeats):
            specs.append(("a", {}, SCREEN))
            specs.append(("b", {}, f"unrelated line number {index}"))
        if third_prefix_twin:
            specs.append(("c", {}, SCREEN + " Amended copy."))
        return snaps(*specs)

    two = detector._detect_content_fingerprint_loop(trace(2, third_prefix_twin=True))
    three = detector._detect_content_fingerprint_loop(trace(3))
    four = detector._detect_content_fingerprint_loop(trace(4))

    assert two is not None and three is not None and four is not None
    assert [r.evidence["exact_content_matches"] for r in (two, three, four)] == [2, 3, 4]
    assert two.raw_score == pytest.approx(0.7)
    assert four.raw_score == 1.0
    # base 0.96*0.5 + raw*0.25 + min(1, span/5)*0.15 + min(1, matches/3)*0.1
    assert [r.confidence for r in (two, three, four)] == pytest.approx(
        [0.7817, 0.925, 0.98], abs=1e-4
    )
    assert [r.loop_length for r in (two, three, four)] == [2, 4, 6]


def test_content_fingerprint_prefix_is_case_and_whitespace_insensitive() -> None:
    twin = SCREEN.upper().replace("CONTRACT FOR", "CONTRACT\nFOR") + " Amended."
    states = snaps(
        ("a", {}, SCREEN),
        ("b", {}, "unrelated line one"),
        ("a", {}, SCREEN),
        ("b", {}, "unrelated line two"),
        ("c", {}, twin),
        ("d", {}, "unrelated line three"),
    )
    without_twin = snaps(
        ("a", {}, SCREEN),
        ("b", {}, "unrelated line one"),
        ("a", {}, SCREEN),
        ("b", {}, "unrelated line two"),
        ("c", {}, "Something very different from the earlier content in this one"),
        ("d", {}, "unrelated line three"),
    )

    detector = MultiLevelLoopDetector()
    fired = detector._detect_content_fingerprint_loop(states)

    assert fired is not None
    assert fired.evidence["loop_agent"] == "a"
    assert fired.evidence["exact_content_matches"] == 2
    # Two exact repeats alone are below the three-matching-prefix bar.
    assert detector._detect_content_fingerprint_loop(without_twin) is None


def test_content_fingerprint_needs_one_agent_to_repeat_itself() -> None:
    three_agents_same_text = snaps(
        ("a", {}, SCREEN),
        ("b", {}, "unrelated line one"),
        ("c", {}, SCREEN),
        ("b", {}, "unrelated line two"),
        ("d", {}, SCREEN),
        ("e", {}, "unrelated line three"),
    )

    assert MultiLevelLoopDetector()._detect_content_fingerprint_loop(three_agents_same_text) is None


def test_content_fingerprint_ignores_short_repeated_commands() -> None:
    short = "Please retry the failed step again now"
    assert 20 < len(short) <= 50
    states = snaps(
        *[
            ("a", {}, short) if index % 2 == 0 else ("b", {}, f"different {index} text")
            for index in range(6)
        ]
    )

    assert MultiLevelLoopDetector()._detect_content_fingerprint_loop(states) is None


def test_content_fingerprint_needs_at_least_four_states() -> None:
    detector = MultiLevelLoopDetector()
    # Three identical outputs from one agent would qualify on content alone;
    # only the trace length keeps this tier out of very short traces.
    three = snaps(("a", {}, SCREEN), ("a", {}, SCREEN), ("a", {}, SCREEN))
    four = snaps(*[("a", {}, SCREEN)] * 3, ("b", {}, "Filing the paperwork"))

    assert detector._detect_content_fingerprint_loop(three) is None
    fired = detector._detect_content_fingerprint_loop(four)
    assert fired is not None
    assert fired.evidence["exact_content_matches"] == 3


def test_fanout_over_distinct_structured_inputs_is_not_a_loop() -> None:
    distinct = screening_trace(
        [
            {"contract": "alpha.pdf", "round": 1},
            {"contract": "beta.pdf", "round": 2},
            {"contract": "gamma.pdf", "round": 3},
        ]
    )

    assert MultiLevelLoopDetector()._detect_content_fingerprint_loop(distinct) is None
    assert not detect(distinct).detected


@pytest.mark.parametrize(
    "inputs",
    [
        [{"contract": "alpha.pdf", "round": n} for n in (1, 2, 3)],
        [{"round": 1}, {"round": 2}, {"round": 3}],
        [{"contract": "alpha.pdf"}, {"contract": "beta.pdf", "status": "failed"}, {"contract": "gamma.pdf"}],
    ],
    ids=["same-contract-changing-counter", "counters-only", "one-input-failed"],
)
def test_repeat_with_untrustworthy_inputs_between_requests_is_still_a_loop(
    inputs: List[Dict[str, Any]],
) -> None:
    result = detect(screening_trace(inputs))

    assert result.detected
    assert result.method == "content_fingerprint"
    assert result.evidence["loop_agent"] == "screener"
    assert result.evidence["exact_content_matches"] == 3


def _repeat_then_close(closing: Spec) -> List[StateSnapshot]:
    return snaps(
        ("screener", {}, SCREEN),
        ("intake", {}, INTAKE_PROSE[0]),
        ("screener", {}, SCREEN),
        ("intake", {}, INTAKE_PROSE[1]),
        ("screener", {}, SCREEN),
        closing,
    )


@pytest.mark.parametrize(
    "closing",
    [
        ("screener", {}, "All 3 contracts screened and cleared for signature."),
        ("screener", {"status": "completed"}, "Wrapped."),
    ],
    ids=["quantified-completion-text", "terminal-status"],
)
def test_explicit_terminal_progress_after_the_last_repeat_suppresses_the_loop(
    closing: Spec,
) -> None:
    assert not detect(_repeat_then_close(closing)).detected


@pytest.mark.parametrize(
    "closing",
    [
        ("screener", {"status": "failed"}, "Wrapped."),
        ("screener", {}, "Still working on it and making progress here."),
        ("intake", {"status": "success"}, "Someone else finished something unrelated."),
    ],
    ids=["failed-status", "generic-progress-prose", "peer-success"],
)
def test_failure_or_vague_progress_after_the_last_repeat_keeps_the_loop(closing: Spec) -> None:
    result = detect(_repeat_then_close(closing))

    assert result.detected
    assert result.method == "content_fingerprint"


# --------------------------------------------------------------------------
# v2.3 no-signal guard
# --------------------------------------------------------------------------


@pytest.mark.parametrize("placeholder", ["", "None", "null", " N/A ", "{}", "[]", "-"])
def test_trace_of_placeholder_only_snapshots_is_not_a_loop(placeholder: str) -> None:
    result = detect(snaps(*[("a", {}, placeholder)] * 5))

    assert not result.detected
    assert result.method is None
    assert result.evidence == {"no_signal": True}


def test_snapshots_whose_delta_only_holds_channel_labels_and_nulls_carry_no_signal() -> None:
    result = detect(
        snaps(*[("a", {"content": None, "type": "llm", "step": 3}, "")] * 5)
    )

    assert not result.detected
    assert result.evidence == {"no_signal": True}


def test_no_signal_guard_is_whole_trace_not_per_snapshot() -> None:
    one_real_message = snaps(
        *[("a", {}, "")] * 4,
        ("a", {}, "The deployment target is missing so I cannot proceed"),
    )
    real_payload_no_text = snaps(*[("a", {"query": "inventory SKU-1001"}, "")] * 5)
    real_text_repeating = snaps(
        *[("a", {"type": "llm"}, "Retrying the identical upload request to the bucket now")] * 5
    )

    assert "no_signal" not in (detect(one_real_message).evidence or {})
    stuck_query = detect(real_payload_no_text)
    assert stuck_query.detected
    assert stuck_query.method == "structural"
    assert detect(real_text_repeating).detected


def test_content_is_informative_rejects_absent_data_placeholders() -> None:
    detector = MultiLevelLoopDetector()

    assert detector._content_is_informative("Searching the repository for config.yaml")
    for placeholder in ("", "None", "  NULL ", "n/a", "{}", "[]", "()", "-"):
        assert not detector._content_is_informative(placeholder)


# --------------------------------------------------------------------------
# v2.0 empty-delta structural tier (word-overlap fallback without the embedder)
# --------------------------------------------------------------------------


def test_empty_delta_structural_match_needs_the_content_to_repeat() -> None:
    restated = snaps(
        ("agent", {}, "I could not find the file config.yaml in the repository, retrying the search now"),
        ("agent", {}, "I could not find the file config.yaml in the repository, retrying the search now again"),
        ("agent", {}, "I could not find the file config.yaml in the repository, retrying the search again"),
    )
    topic_switching = snaps(
        ("agent", {}, "Summarize the quarterly revenue figures for the northern region"),
        ("agent", {}, "Translate the onboarding handbook into Spanish for new hires"),
        ("agent", {}, "Debug the memory leak inside the image resizing worker pool"),
        ("agent", {}, "Draft a press release announcing the partnership with Contoso"),
    )
    detector = MultiLevelLoopDetector()

    stuck = detector.detect_loop(restated)
    assert stuck.detected
    assert stuck.method == "structural"
    assert stuck.loop_start_index == 0
    assert stuck.evidence["structural_matches"] == 2
    assert detector._detect_structural_loop(
        topic_switching[-1], topic_switching[:-1], topic_switching
    ) is None
    assert not detector.detect_loop(topic_switching).detected


def test_empty_delta_word_overlap_fallback_bar_is_half_the_words() -> None:
    if get_shared_embedder() is not None:
        pytest.skip("with the semantic model installed, cosine replaces word overlap")

    states = snaps(
        ("agent", {}, "check inventory reports tomorrow"),  # 2 of 6 distinct words shared
        ("agent", {}, "check inventory levels tomorrow"),  # 3 of 5 distinct words shared
        ("agent", {}, "check inventory levels today"),
    )

    result = MultiLevelLoopDetector()._detect_structural_loop(states[-1], states[:-1], states)

    assert result is not None
    assert result.evidence["structural_matches"] == 1
    assert result.loop_start_index == 1
    assert result.loop_length == 1
    assert result.raw_score == pytest.approx(0.5)


def test_empty_delta_structural_tier_uses_embedder_cosine_when_available() -> None:
    pytest.importorskip("sentence_transformers")
    if get_shared_embedder() is None:
        pytest.skip("semantic model is unavailable")

    # No word overlap to speak of, so the Jaccard fallback would reject both
    # earlier snapshots. Only the closer paraphrase clears the cosine bar.
    states = snaps(
        ("agent", {}, "I could not find the configuration file anywhere in the repository."),
        ("agent", {}, "The config file is missing from the repo, and my search turned up nothing."),
        ("agent", {}, "Unable to locate the settings file in the codebase after searching everywhere."),
    )
    detector = MultiLevelLoopDetector()

    result = detector._detect_structural_loop(states[-1], states[:-1], states)

    assert result is not None
    assert result.method == "structural"
    assert result.evidence["structural_matches"] == 1
    assert result.loop_start_index == 0


# --------------------------------------------------------------------------
# v2.1 structural recurrence and v1.7 status progress
# --------------------------------------------------------------------------


def test_structural_tier_requires_a_genuine_return_to_the_same_state() -> None:
    contents = [
        "Outlining the database schema for the orders subsystem",
        "Comparing queue technologies for the ingestion layer",
        "Drafting the rollout schedule across regions in August",
        "Estimating the hosting budget needed for the launch",
    ]
    responses = [
        "Outline the database schema for orders",
        "Choose queue technology for ingestion",
        "Draft rollout schedule for August",
        "Estimate hosting budget for the launch",
    ]
    # "owner" repeats, so the batch-iteration guard cannot be what rejects this:
    # only the return-to-the-same-state requirement separates it from a loop.
    distinct_work = snaps(
        *[("planner", {"owner": "planner", "response": r}, c) for r, c in zip(responses, contents)]
    )
    same_work = snaps(
        ("planner", {"response": "Outline schema"}, contents[0]),
        ("planner", {"response": "Outline schema"}, contents[0] + " now"),
        ("planner", {"response": "Outline schema"}, contents[0] + " again"),
    )
    detector = MultiLevelLoopDetector()

    assert detector._detect_structural_loop(
        distinct_work[-1], distinct_work[:-1], distinct_work
    ) is None
    assert not detector.detect_loop(distinct_work).detected
    repeated = detector.detect_loop(same_work)
    assert repeated.detected
    assert repeated.method == "structural"
    assert repeated.evidence["structural_matches"] == 2


def test_trivial_payload_recurrence_needs_similar_content_to_be_a_loop() -> None:
    identity_only = {"agent": "planner", "status": "ok"}
    different_work = snaps(
        *[("planner", identity_only, text) for text in DIVERSE_CONTENT[:4]]
    )
    same_work = snaps(
        *[
            ("planner", identity_only, "Outline the database schema for orders" + suffix)
            for suffix in ("", " again", " once more", " yet again")
        ]
    )

    assert not detect(different_work).detected
    stuck = detect(same_work)
    assert stuck.detected
    assert stuck.method == "structural"


def test_structural_tier_does_not_flag_strictly_advancing_single_key_batches() -> None:
    detector = MultiLevelLoopDetector()
    text = "Processing the intake document again"
    batch = snaps(*[("a", {"doc": n}, text) for n in range(1, 5)])
    cycling = snaps(*[("a", {"doc": n % 2}, text) for n in range(1, 5)])

    assert detector._detect_structural_loop(batch[-1], batch[:-1], batch) is None
    fired = detector._detect_structural_loop(cycling[-1], cycling[:-1], cycling)
    assert fired is not None
    assert fired.evidence["structural_matches"] == 3


def test_multi_key_state_with_a_strictly_advancing_work_key_is_iteration() -> None:
    text = "Processing the intake document again"
    iterating = snaps(*[("a", {"doc": f"d{n}", "owner": "x"}, text) for n in range(1, 6)])
    stuck = snaps(*[("a", {"doc": "d1", "owner": "x"}, text) for _ in range(5)])

    result = detect(iterating)

    assert not result.detected
    assert result.evidence == {"iterating_signal": True}
    assert detect(stuck).detected


def test_isolated_recap_at_the_end_is_not_a_loop_but_a_recap_after_a_loop_is() -> None:
    clarifying = [
        ("a", {}, "Clarifying the deployment target please"),
        ("a", {}, "Could you clarify the deployment target"),
        ("a", {}, "Need clarification about the deployment target"),
    ]
    plain = detect(snaps(*clarifying, ("a", {}, "Still waiting on the deployment target clarification")))
    recap = detect(
        snaps(*clarifying, ("a", {}, "To summarize, still waiting on the deployment target clarification"))
    )
    recap_after_repeats = detect(
        snaps(*[("a", {}, SCREEN)] * 3, ("a", {}, "To summarize: " + SCREEN))
    )

    assert plain.detected
    assert not recap.detected
    assert recap.evidence == {"summary_short_circuit": True}
    # When the earlier states already repeat themselves, the recap excuses nothing.
    assert recap_after_repeats.detected


def test_structural_loop_start_is_reported_in_full_trace_coordinates() -> None:
    states = snaps(
        *[("w", {"tool": "fetch", "attempt": n}, "fetching the page again") for n in range(1, 9)]
    )

    result = detect(states, window_size=4)

    assert result.method == "structural"
    # Only the last four states are examined: three window states before the current one.
    assert result.loop_start_index == 4
    assert result.loop_length == 3
    assert result.evidence["window_size"] == 3
    assert result.evidence["structural_matches"] == 3


def test_is_loop_repeat_rules() -> None:
    detector = MultiLevelLoopDetector()
    identity = {"agent": "p", "status": "ok"}

    a = StateSnapshot("a", {**identity, "round": 1}, "Reviewing the alpha document and writing notes", 0)
    b = StateSnapshot("a", {**identity, "round": 2}, "Reviewing the alpha document and writing notes", 1)
    c = StateSnapshot("a", {**identity, "round": 3}, "Completely different subject about budgets", 2)
    blank = StateSnapshot("a", dict(identity), "", 3)
    # Trivial payload: content decides. A blank side cannot contradict it.
    assert detector._is_loop_repeat(a, b) is True
    assert detector._is_loop_repeat(a, c) is False
    assert detector._is_loop_repeat(a, blank) is True

    # Substantive payload recurs: a genuine repeat regardless of content drift.
    query_1 = StateSnapshot("a", {"query": "inventory", "round": 1}, "Looking up inventory", 4)
    query_2 = StateSnapshot("a", {"query": "inventory", "round": 2}, "Something else entirely here", 5)
    assert detector._is_loop_repeat(query_1, query_2) is True

    # Differing payloads: only repeated content makes it a return to the same state.
    orders = StateSnapshot("a", {"query": "orders"}, "Looking up the inventory levels now please", 6)
    inventory = StateSnapshot("a", {"query": "inventory"}, "Looking up the inventory levels now please", 7)
    unrelated = StateSnapshot("a", {"query": "orders"}, "Zzz qqq", 8)
    assert detector._is_loop_repeat(orders, inventory) is True
    assert detector._is_loop_repeat(inventory, unrelated) is False

    # The content bar is half the words: 3 of 5 shared clears it, 2 of 6 does not.
    base = StateSnapshot("a", {"query": "orders"}, "check inventory levels today", 9)
    near = StateSnapshot("a", {"query": "inventory"}, "check inventory levels tomorrow", 10)
    far = StateSnapshot("a", {"query": "inventory"}, "check inventory reports tomorrow", 11)
    assert detector._is_loop_repeat(base, near) is True
    assert detector._is_loop_repeat(base, far) is False


def test_meaningful_delta_hash_ignores_only_bookkeeping_keys() -> None:
    detector = MultiLevelLoopDetector()

    def digest(delta: Dict[str, Any]) -> str:
        return detector._meaningful_delta_hash(StateSnapshot("a", delta, "", 0))

    assert digest({"attempt_num": 1, "iter": 2, "cycle_count": 3, "tick": 4}) == ""
    assert digest({"tool": "fetch", "attempt_num": 1}) == digest({"tool": "fetch", "attempt_num": 9})
    assert digest({"tool": "fetch", "attempt_num": 1}) != digest({"tool": "post", "attempt_num": 1})
    # Work payloads that only differ in list order are the same work.
    forward = {"ids": [{"id": "a"}, {"id": "b"}], "attempt_num": 1}
    backward = {"ids": [{"id": "b"}, {"id": "a"}], "attempt_num": 2}
    assert digest(forward) == digest(backward)


def test_delta_is_substantive_only_when_it_carries_a_work_value() -> None:
    substantive = MultiLevelLoopDetector()._delta_is_substantive

    for work in (
        {"query": "inventory"},
        {"Query": "inventory"},
        {"action": "check_inventory"},
        # "source" names the retrieved document in RAG traces, so it is work.
        {"source": "doc-1", "agent": "planner"},
        {"tool": "fetch", "attempt": 3},
    ):
        assert substantive(work), work
    for no_work in (
        {},
        {"agent": "planner", "status": "ok"},
        {"Agent": "planner"},
        {"Event_Type": "agent.message"},
        {"STATUS": "ok"},
        {"attempt": 3, "round": 1},
        {"body": None},
        {"prompt": ""},
        {"payload": {}},
        {"items": []},
    ):
        assert not substantive(no_work), no_work


@pytest.mark.parametrize(
    "content",
    [
        "Copy completed cleanly",
        "Copy finished cleanly",
        "All done here",
        "Trying the next host",
        "Running step two",
        "Moving on to the fallback host",
        "COMPLETED the copy",
    ],
)
def test_one_changed_work_value_is_progress_only_when_the_text_says_work_moved(
    content: str,
) -> None:
    detector = MultiLevelLoopDetector()
    base = {"task": "sync", "target": "db1", "round": 1}
    moved = {**base, "target": "db2", "round": 2}

    def progress(text: str) -> bool:
        return detector._has_meaningful_progress(
            StateSnapshot("a", base, "", 0), StateSnapshot("a", moved, text, 1)
        )

    assert progress(content)
    assert not progress("Still trying the same thing")
    assert not progress("")


def test_meaningful_progress_needs_new_work_keys_or_two_changed_values() -> None:
    detector = MultiLevelLoopDetector()
    base = {"task": "sync", "target": "db1", "round": 1}

    def progress(current: Dict[str, Any], content: str = "") -> bool:
        return detector._has_meaningful_progress(
            StateSnapshot("a", base, "", 0), StateSnapshot("a", current, content, 1)
        )

    # A newly appearing work key is progress; a newly appearing counter is not.
    assert progress({**base, "artifact": "report.pdf"})
    assert not progress({**base, "attempt": 2})
    # A ticking counter is not progress, however the text reads.
    assert not progress({**base, "round": 2}, "Moving on to the next host")
    # Two work values changing together is progress; one alone is not.
    assert progress({**base, "task": "backup", "target": "db2"})
    assert not progress({**base, "target": "db2"})


@pytest.mark.parametrize(
    "counter_key",
    ["attempt_num", "iter_num", "round_num", "cycle_count", "loop_index", "epoch", "tick", "step_num"],
)
def test_stuck_retry_with_any_reconciled_counter_key_is_still_a_loop(counter_key: str) -> None:
    states = snaps(
        *[("worker", {"tool": "fetch", counter_key: n}, "fetching the page again") for n in range(1, 5)]
    )

    result = detect(states)

    assert result.detected
    assert result.method == "structural"


def test_status_reaching_terminal_success_is_progress_but_constant_success_is_not() -> None:
    detector = MultiLevelLoopDetector()

    def progress(previous: Dict[str, Any], current: Dict[str, Any]) -> bool:
        return detector._has_meaningful_progress(
            StateSnapshot("a", previous, "", 0), StateSnapshot("a", current, "", 1)
        )

    assert progress({"status": "pending", "task": "sync"}, {"status": "success", "task": "sync"})
    assert progress({"Phase": "build"}, {"Phase": "DONE"})
    assert progress({"task": "sync"}, {"status": "completed", "task": "sync"})
    assert not progress({"status": "success", "task": "sync"}, {"status": "success", "task": "sync"})
    assert not progress({"status": "pending", "task": "sync"}, {"status": "retry", "task": "sync"})
    # Only status-like keys count; a free-form note saying "done" does not.
    assert not progress({"note": "x", "task": "s"}, {"note": "done", "task": "s"})


def test_retry_sequence_that_resolves_is_not_flagged_but_a_stuck_one_is() -> None:
    def retries(final_status: str) -> List[StateSnapshot]:
        statuses = ["pending", "retry", "retry", final_status]
        return snaps(
            *[
                ("a", {"status": s, "tool": "fetch", "attempt": n}, "fetching the page again")
                for n, s in enumerate(statuses, start=1)
            ]
        )

    assert not detect(retries("success")).detected
    stuck = detect(retries("retry"))
    assert stuck.detected
    assert stuck.method == "structural"


# --------------------------------------------------------------------------
# Structural key matching against the configured threshold
# --------------------------------------------------------------------------


def _keyed(agent: str, keys: str) -> StateSnapshot:
    return StateSnapshot(agent, {key: 1 for key in keys}, "", 0)


@pytest.mark.parametrize(
    ("threshold", "keys_a", "keys_b", "expected"),
    [
        (1.0, "abc", "abcd", False),
        (0.95, "abc", "abcd", False),
        (0.75, "abc", "abcd", True),
        (0.75, "ab", "abc", False),
        (0.5, "ab", "abc", True),
        (0.5, "ab", "cd", False),
        (0.5, "", "a", False),
    ],
)
def test_structural_match_relaxes_key_overlap_to_the_threshold(
    threshold: float, keys_a: str, keys_b: str, expected: bool
) -> None:
    detector = MultiLevelLoopDetector(structural_threshold=threshold)

    assert detector._structural_match(_keyed("x", keys_a), _keyed("x", keys_b)) is expected


def test_structural_match_always_requires_the_same_agent_and_accepts_equal_key_sets() -> None:
    detector = MultiLevelLoopDetector(structural_threshold=0.0)

    assert detector._structural_match(_keyed("x", "ab"), _keyed("x", "ab"))
    assert detector._structural_match(_keyed("x", ""), _keyed("x", ""))
    assert not detector._structural_match(_keyed("x", "ab"), _keyed("y", "ab"))


def test_structural_tier_tolerates_key_drift_only_under_a_lenient_threshold() -> None:
    # The later snapshot drops one key. (A newly added key would itself count
    # as progress, so drift in that direction is never a repeat.)
    states = snaps(
        ("w", {"tool": "fetch", "url": "u", "auth": "t", "trace": "x"}, "fetching the page again"),
        ("w", {"tool": "fetch", "url": "u", "auth": "t"}, "fetching the page again"),
    )
    window, current = states[:-1], states[-1]

    strict = MultiLevelLoopDetector(structural_threshold=1.0)
    lenient = MultiLevelLoopDetector(structural_threshold=0.7)

    assert strict._detect_structural_loop(current, window, states) is None
    relaxed = lenient._detect_structural_loop(current, window, states)
    assert relaxed is not None
    assert relaxed.method == "structural"
    assert relaxed.evidence["structural_threshold"] == 0.7


# --------------------------------------------------------------------------
# Content similarity helper
# --------------------------------------------------------------------------


def test_content_similar_rules() -> None:
    detector = MultiLevelLoopDetector()
    same_prefix_a = "Connection to the inventory service timed out after thirty seconds while fetching SKU data"
    same_prefix_b = "Connection to the inventory service timed out after thirty seconds; giving up entirely"
    assert same_prefix_a[:50] == same_prefix_b[:50]

    assert not detector._content_similar("", "text")
    assert not detector._content_similar("text", "")
    assert not detector._content_similar("   ", "   x")
    # Two blank sides are absence of content, not repeated content.
    assert not detector._content_similar("", "")
    assert not detector._content_similar("   ", "  ")
    assert detector._content_similar("identical", "identical")
    assert detector._content_similar("Retry The Upload Now", "retry the upload now")
    assert detector._content_similar(same_prefix_a, same_prefix_b)
    assert detector._content_similar("the cat sat on the mat", "the cat sat on a mat")
    assert not detector._content_similar("alpha beta gamma", "delta epsilon zeta")
    # Jaccard is 3/5 = 0.6: it clears the default bar but not a stricter one.
    assert detector._content_similar("a b c d", "a b c e", threshold=0.6)
    assert not detector._content_similar("a b c d", "a b c e", threshold=0.7)


# --------------------------------------------------------------------------
# State hashing and the hash tier
# --------------------------------------------------------------------------

ITEM_A = {"id": "a", "v": 1}
ITEM_B = {"id": "b", "v": 2}


def test_deep_canonicalize_makes_list_order_irrelevant() -> None:
    canon = MultiLevelLoopDetector._deep_canonicalize

    assert canon([ITEM_B, ITEM_A]) == [ITEM_A, ITEM_B]
    assert canon((3, 1, 2)) == [1, 2, 3]
    assert canon({"z": [ITEM_B, ITEM_A], "a": 1}) == {"a": 1, "z": [ITEM_A, ITEM_B]}
    assert canon("scalar") == "scalar"


def test_deep_canonicalize_keeps_order_when_elements_cannot_be_sorted() -> None:
    unsortable = [{(1, 2): "a"}, {(0, 1): "b"}]

    assert MultiLevelLoopDetector._deep_canonicalize(unsortable) == unsortable


def test_state_hash_ignores_list_order_but_not_list_content() -> None:
    detector = MultiLevelLoopDetector()

    def digest(items: List[Dict[str, Any]]) -> str:
        return detector._compute_state_hash(StateSnapshot("a", {"items": items}, "", 0))

    assert digest([ITEM_A, ITEM_B]) == digest([ITEM_B, ITEM_A])
    assert digest([ITEM_A, ITEM_B]) != digest([ITEM_B, {"id": "c"}])


def test_hash_tier_matches_states_whose_lists_only_differ_in_order() -> None:
    reordered = snaps(
        ("w1", {"items": [ITEM_A, ITEM_B]}, "first pass over records"),
        ("w2", {"items": [ITEM_B, ITEM_A]}, "second worker handling batch"),
        ("w3", {"items": [ITEM_A, ITEM_B]}, "third worker on data"),
        ("w4", {"items": [ITEM_B, ITEM_A]}, "fourth worker doing thing"),
    )
    different = snaps(
        ("w1", {"items": [ITEM_A, ITEM_B]}, "first pass over records"),
        ("w2", {"items": [ITEM_B, {"id": "c"}]}, "second worker handling batch"),
        ("w3", {"items": [ITEM_A, {"id": "d"}]}, "third worker on data"),
        ("w4", {"items": [{"id": "e"}, ITEM_B]}, "fourth worker doing thing"),
    )

    caught = detect(reordered)

    assert caught.detected
    assert caught.method == "hash"
    assert caught.evidence["hash_matches"] == 3
    assert not detect(different).detected


def test_hash_tier_does_not_trust_channel_labels_or_nulled_content() -> None:
    channel_only = snaps(
        *[
            (f"w{index % 2}", {"event_type": "agent.message", "category": "chat"}, text)
            for index, text in enumerate(DIVERSE_CONTENT)
        ]
    )
    step_label = snaps(*[("w", {"step": "work"}, text) for text in DIVERSE_CONTENT])
    nulled = snaps(
        *[("w", {"body": None, "prompt": "", "type": "llm"}, text) for text in DIVERSE_CONTENT]
    )
    real_work_key = snaps(
        *[("w", {"query": "inventory SKU-1001", "type": "llm"}, text) for text in DIVERSE_CONTENT]
    )
    emptied = snaps(
        *[("w", {"extras": {}, "tags": [], "type": "llm"}, text) for text in DIVERSE_CONTENT]
    )

    for trace in (channel_only, step_label, nulled, emptied):
        assert not detect(trace).detected
    assert detect(real_work_key).detected


def test_hash_tier_confirms_workless_deltas_with_same_agent_and_repeated_content() -> None:
    detector = MultiLevelLoopDetector()
    label = {"type": "llm"}  # a channel label only, so the delta carries no work
    text = "Uploading the quarterly export to the archive bucket"

    def run(*specs: Spec) -> Optional[LoopDetectionResult]:
        states = snaps(*specs)
        return detector._detect_hash_loop(states[-1], states[:-1], states)

    identical = run(*[("a", label, text)] * 4)
    assert identical is not None
    assert identical.method == "hash"
    assert identical.evidence == {"hash_matches": 3, "window_size": 3}
    assert identical.confidence == pytest.approx(0.84, abs=1e-4)

    same_opening = run(*[("a", label, f"{text} (try {n})") for n in range(4)])
    assert same_opening is not None
    assert same_opening.evidence["hash_matches"] == 3

    short_exact = run(*[("a", label, "Retrying upload")] * 4)
    assert short_exact is not None
    assert short_exact.evidence["hash_matches"] == 3

    # Same label on every snapshot is not enough: the agent or the words must repeat.
    assert run(*[(f"agent{n}", label, text) for n in range(4)]) is None
    # Sharing an opening word is not the same as repeating: the first 40 characters must match.
    topics = ("quarterly", "annual", "monthly", "weekly")
    assert run(*[("a", label, f"Uploading {topic} export to the archive bucket") for topic in topics]) is None

    partial = run(("a", {"type": "other"}, "Something unrelated entirely"), *[("a", label, text)] * 3)
    assert partial is not None
    assert partial.evidence == {"hash_matches": 2, "window_size": 3}
    assert partial.loop_start_index == 1
    assert partial.loop_length == 2
    assert partial.raw_score == pytest.approx(2 / 3)
    assert partial.confidence == pytest.approx(0.7267, abs=1e-4)


# --------------------------------------------------------------------------
# Cluster batch-iteration veto helpers (no clustering backend needed)
# --------------------------------------------------------------------------

CYCLE_EVIDENCE = {"type": "cluster_cycle", "cycle_length": 2, "cluster_distribution": {0: 2, 1: 2}}


def _one_cluster(values: List[Any], key: str = "k") -> List[StateSnapshot]:
    return snaps(*[("a", {key: value}, "") for value in values])


def test_cluster_veto_for_dominance_accepts_any_strictly_distinct_work_value() -> None:
    detector = MultiLevelLoopDetector()
    evidence = {"dominant_cluster": 0}
    free_text = ["approach A is more suitable", "approach B is more suitable", "approach C is better"]

    for values in (["5001", "5002", "5003"], free_text):
        assert detector._cluster_is_batch_iteration(_one_cluster(values), [0, 0, 0], evidence)
    assert not detector._cluster_is_batch_iteration(_one_cluster(["x", "x", "y"]), [0, 0, 0], evidence)


def test_cluster_veto_hashes_unhashable_values_by_their_json() -> None:
    detector = MultiLevelLoopDetector()
    evidence = {"dominant_cluster": 0}

    assert detector._cluster_is_batch_iteration(
        _one_cluster([[1, 2], [3], [4, 5]]), [0, 0, 0], evidence
    )
    assert not detector._cluster_is_batch_iteration(
        _one_cluster([[1, 2], [1, 2], [3]]), [0, 0, 0], evidence
    )
    assert not detector._cluster_is_batch_iteration(
        _one_cluster([{"a": 1}, {"a": 1}, {"a": 2}]), [0, 0, 0], evidence
    )


def test_cluster_veto_ignores_bookkeeping_missing_keys_and_singletons() -> None:
    detector = MultiLevelLoopDetector()
    evidence = {"dominant_cluster": 0}

    counters = snaps(*[("a", {"iteration": n}, "") for n in range(3)])
    assert not detector._cluster_is_batch_iteration(counters, [0, 0, 0], evidence)

    gap = snaps(("a", {"k": 1}, ""), ("a", {}, ""), ("a", {"k": 3}, ""))
    assert not detector._cluster_is_batch_iteration(gap, [0, 0, 0], evidence)

    pair = snaps(("a", {"k": 1}, ""), ("a", {"k": 2}, ""))
    assert not detector._cluster_is_batch_iteration(pair, [0, 1], evidence)
    assert not detector._cluster_is_batch_iteration(pair, [0, 0], {})


def test_cluster_cycle_veto_needs_identifier_like_values_not_paraphrase() -> None:
    detector = MultiLevelLoopDetector()
    record_ids = snaps(*[("a", {"rec": f"r{n}"}, "") for n in range(4)])
    restated = snaps(
        *[("a", {"rec": f"approach {c} wins the comparison"}, "") for c in "ABCD"]
    )
    labels = [0, 1, 0, 1]

    def veto(states: List[StateSnapshot], require: bool) -> bool:
        return detector._cluster_is_batch_iteration(
            states, labels, CYCLE_EVIDENCE, require_identifier_like=require
        )

    # Loose (dominance) rule vetoes both; the cycle rule vetoes only real ids.
    assert veto(record_ids, False) and veto(restated, False)
    assert veto(record_ids, True)
    assert not veto(restated, True)


def test_cluster_cycle_veto_recognises_a_cycle_from_either_marker() -> None:
    detector = MultiLevelLoopDetector()
    record_ids = snaps(*[("a", {"rec": f"r{n}"}, "") for n in range(4)])
    labels = [0, 1, 0, 1]
    distribution = {0: 2, 1: 2}

    def veto(evidence: Dict[str, Any]) -> bool:
        return detector._cluster_is_batch_iteration(
            record_ids, labels, evidence, require_identifier_like=True
        )

    assert veto({"type": "cluster_cycle", "cluster_distribution": distribution})
    assert veto({"cycle_length": 2, "cluster_distribution": distribution})
    # Neither a cycle marker nor a dominant cluster: nothing to inspect, no veto.
    assert not veto({"cluster_distribution": distribution})


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], False),
        ([1, 2.5], True),
        (["a-1", "b-2"], True),
        ([True, False], False),
        (["approach A wins", "approach B wins"], False),
        (["step 0", "step 1", "step 2"], True),
        (["page 2", "page 3"], True),
        (["step 1", "step 1"], False),
        (["a b", "a b"], False),
        (["x 1", "y 2"], False),
        ([1, "two words"], False),
        (["step 1", 5], False),
        ([[1], [2]], False),
    ],
)
def test_values_are_identifier_like(values: List[Any], expected: bool) -> None:
    assert MultiLevelLoopDetector._values_are_identifier_like(values) is expected


# --------------------------------------------------------------------------
# detect_loop_enhanced is now an alias
# --------------------------------------------------------------------------


def test_detect_loop_enhanced_returns_exactly_what_detect_loop_returns() -> None:
    detector = MultiLevelLoopDetector()
    looping = snaps(*[("a", {"query": "inventory"}, "Looking up inventory")] * 5)
    progressing = snaps(*[("a", {}, text) for text in DIVERSE_CONTENT[:4]])

    for states in (looping, progressing):
        assert detector.detect_loop_enhanced(states) == detector.detect_loop(states)
    assert detector.detect_loop_enhanced(looping).detected
    assert not detector.detect_loop_enhanced(progressing).detected

