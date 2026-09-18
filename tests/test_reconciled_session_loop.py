"""Behavioral contracts for the OpenClaw session-loop detector.

Covers the behavior the backend reconciliation added: outcome-aware polling
suppression, nested ``data`` tool fields, the distinct-input fuzzy tool guard,
and the fuzzy (appended-text) message loop.
"""

from typing import Any, Dict, List, Optional

import pytest

import pisama_detectors as pd
from pisama_detectors.detection.openclaw import OpenClawSessionLoopDetector
from pisama_detectors.detection.openclaw.session_loop_detector import (
    _common_prefix_len,
    _common_suffix_len,
    _hash_input,
    _message_content,
    _structural_hash,
    _variation_is_appended,
)
from pisama_detectors.detection.turn_aware._base import TurnAwareSeverity


def _call(name: str, tool_input: Any = None) -> Dict[str, Any]:
    return {"type": "tool.call", "tool_name": name, "tool_input": tool_input}


def _result(payload: Any) -> Dict[str, Any]:
    return {"type": "tool.result", "tool_result": payload}


def _msg(text: str) -> Dict[str, Any]:
    return {"type": "message.sent", "content": text}


def _detect(events: List[Dict[str, Any]]) -> Any:
    return pd.detect_openclaw_session_loop({"events": events})


def _issue(result: Any, issue_type: str) -> Optional[Dict[str, Any]]:
    for issue in result.evidence.get("issues", []):
        if issue["type"] == issue_type:
            return issue
    return None


def _poll(payloads: List[Any], name: str = "get_export_status") -> List[Dict[str, Any]]:
    """Identical-input tool calls, each followed by its own result."""
    events: List[Dict[str, Any]] = []
    for payload in payloads:
        events.append(_call(name, {"job_id": "job-7"}))
        events.append(_result(payload))
    return events


# --- base interface and trivial sessions -----------------------------------


def test_detect_base_interface_reads_session_from_metadata() -> None:
    detector = OpenClawSessionLoopDetector()
    events = [_call("fetch", {"u": 1}) for _ in range(3)]

    hit = detector.detect([], {"session": {"events": events}})
    assert hit.detected
    assert hit.failure_mode == "F8"
    assert detector.supported_failure_modes == ["F8"]

    for metadata in (None, {}, {"session": {}}):
        miss = detector.detect([], metadata)
        assert not miss.detected
        assert miss.explanation == "No events in session"


def test_benign_session_reports_no_loop_with_zero_confidence() -> None:
    result = _detect(
        [
            _call("read_file", {"path": "a.txt"}),
            _call("write_file", {"path": "b.txt"}),
            _call("search", {"q": "pisama"}),
            _msg("Done"),
        ]
    )

    assert not result.detected
    assert result.explanation == "No loop patterns detected"
    assert result.severity == TurnAwareSeverity.NONE
    assert result.confidence == 0.0
    assert result.failure_mode is None


# --- exact tool-call loop ---------------------------------------------------


@pytest.mark.parametrize(
    ("repeats", "severity", "confidence"),
    [
        (3, TurnAwareSeverity.MINOR, 0.6),
        (4, TurnAwareSeverity.MODERATE, 0.8),
        (6, TurnAwareSeverity.SEVERE, 1.0),
    ],
)
def test_identical_tool_calls_scale_severity_and_confidence(
    repeats: int, severity: TurnAwareSeverity, confidence: float
) -> None:
    result = _detect([_call("fetch", {"url": "x"}) for _ in range(repeats)])

    assert result.detected
    assert result.severity == severity
    assert result.confidence == pytest.approx(confidence)
    assert result.failure_mode == "F8"
    assert result.detector_name == "OpenClawSessionLoopDetector"
    assert result.suggested_fix
    # Identical calls also match the fuzzy tier's structure test at 5+ repeats;
    # the exact tier fired, so only one issue may be reported.
    assert [i["type"] for i in result.evidence["issues"]] == ["tool_call_loop"]
    assert result.evidence["issues"][0]["repeat_count"] == repeats
    assert result.affected_turns == list(range(repeats))


def test_two_identical_calls_are_below_the_loop_bar() -> None:
    result = _detect([_call("fetch", {"url": "x"}), _call("fetch", {"url": "x"})])

    assert not result.detected


def test_longest_run_wins_and_is_reported_with_event_indices() -> None:
    events = [
        _call("alpha", {"k": 1}),
        {"type": "agent.turn"},
        _call("alpha", {"k": 1}),
        _call("beta", {"k": 2}),
        {"type": "agent.turn"},
        _call("beta", {"k": 2}),
        _call("beta", {"k": 2}),
    ]
    result = _detect(events)

    issue = _issue(result, "tool_call_loop")
    assert issue is not None
    assert issue["tool_name"] == "beta"
    assert issue["repeat_count"] == 3
    assert issue["turns"] == [3, 5, 6]


def test_run_at_start_survives_a_different_trailing_call() -> None:
    result = _detect(
        [_call("alpha", {"k": 1}) for _ in range(3)] + [_call("beta", {"k": 2})]
    )

    issue = _issue(result, "tool_call_loop")
    assert issue is not None
    assert issue["tool_name"] == "alpha"
    assert issue["turns"] == [0, 1, 2]


def test_identical_calls_separated_by_many_events_are_not_consecutive() -> None:
    filler = [{"type": "agent.turn"}] * 4  # cur_idx - prev_idx == 5
    far = [_call("fetch", {"u": 1}), *filler, _call("fetch", {"u": 1}), *filler,
           _call("fetch", {"u": 1})]
    near = [_call("fetch", {"u": 1}), *filler[:3], _call("fetch", {"u": 1}), *filler[:3],
            _call("fetch", {"u": 1})]

    assert not _detect(far).detected
    assert _detect(near).detected


def test_input_key_order_does_not_hide_an_identical_call() -> None:
    result = _detect(
        [
            _call("fetch", {"a": 1, "b": 2}),
            _call("fetch", {"b": 2, "a": 1}),
            _call("fetch", {"a": 1, "b": 2}),
        ]
    )

    issue = _issue(result, "tool_call_loop")
    assert issue is not None
    assert issue["repeat_count"] == 3


def test_same_tool_with_different_input_breaks_the_exact_run() -> None:
    result = _detect(
        [
            _call("fetch", {"u": 1}),
            _call("fetch", {"u": 2}),
            _call("fetch", {"u": 1}),
        ]
    )

    assert not result.detected


# --- nested `data` tool name / input ---------------------------------------


@pytest.mark.parametrize(
    "data",
    [
        {"name": "web_search", "input": {"q": "same"}},
        {"tool_name": "web_search", "tool_input": {"q": "same"}},
        {"name": "web_search", "args": {"q": "same"}},
    ],
)
def test_tool_name_and_input_are_read_from_nested_data(data: Dict[str, Any]) -> None:
    result = _detect([{"type": "tool.call", "data": dict(data)} for _ in range(3)])

    issue = _issue(result, "tool_call_loop")
    assert issue is not None
    assert issue["tool_name"] == "web_search"
    assert "web_search" in issue["description"]


def test_distinct_nested_tool_names_are_not_one_giant_loop() -> None:
    # Every event nests its name under data. Reading only the top level would
    # resolve all of them to None and report a session-spanning run.
    names = ["read", "write", "search", "email", "plan", "summarize"]
    result = _detect([{"type": "tool.call", "data": {"name": n}} for n in names])

    assert not result.detected


@pytest.mark.parametrize("input_key", ["tool_input", "input", "args"])
def test_nested_calls_with_different_inputs_do_not_form_exact_loop(input_key: str) -> None:
    # If the nested input were not read, all three would hash as "no input"
    # and look identical.
    result = _detect(
        [
            {"type": "tool.call", "data": {"name": "web_search", input_key: {"q": str(i)}}}
            for i in range(3)
        ]
    )

    assert not result.detected


def test_calls_without_a_readable_name_do_not_group_with_named_calls() -> None:
    nameless = {"type": "tool.call", "data": "raw string payload"}
    named = {"type": "tool.call", "tool_name": "fetch"}

    assert not _detect([nameless, named, nameless]).detected


def test_absent_tool_input_repeats_but_differs_from_a_real_input() -> None:
    bare = {"type": "tool.call", "tool_name": "fetch", "data": "not a dict"}
    result = _detect([bare, bare, bare, _call("fetch", {"u": 1})])

    issue = _issue(result, "tool_call_loop")
    assert issue is not None
    assert issue["repeat_count"] == 3
    assert issue["turns"] == [0, 1, 2]


# --- polling: results decide whether repetition is progress ----------------


def test_poll_that_walks_to_completion_is_not_a_loop() -> None:
    result = _detect(
        _poll(
            [
                {"status": "queued"},
                {"status": "running", "note": "working"},
                {"status": "running", "note": "almost"},
                {"status": "completed"},
            ]
        )
    )

    assert not result.detected


def test_identical_poll_results_are_still_a_loop() -> None:
    result = _detect(_poll([{"status": "queued"}] * 4))

    assert result.detected
    issue = _issue(result, "tool_call_loop")
    assert issue is not None
    assert issue["tool_name"] == "get_export_status"
    assert issue["repeat_count"] == 4


def test_calls_with_no_results_between_them_are_still_a_loop() -> None:
    result = _detect([_call("get_export_status", {"job_id": "job-7"}) for _ in range(4)])

    assert result.detected
    assert _issue(result, "tool_call_loop") is not None


def test_monotonic_progress_counters_suppress_the_loop() -> None:
    result = _detect(
        _poll(
            [
                {"progress": 10},
                {"progress": 40},
                {"job": {"Percent": 40.0}},
                {"progress": 90},
            ]
        )
    )

    assert not result.detected


@pytest.mark.parametrize(
    "key",
    [
        "progress",
        "percent",
        "percentage",
        "pct",
        "position",
        "completed",
        "processed",
        "current",
        "offset",
        "Progress",
    ],
)
def test_each_recognised_counter_name_counts_as_progress(key: str) -> None:
    advancing = _detect(_poll([{key: 10}, {key: 20}, {key: 30}]))
    assert not advancing.detected

    # Same counter name, but it never moves: still the same answer every time
    # apart from noise, so it must not be treated as progress.
    stuck = _detect(_poll([{key: 10, "n": 1}, {key: 10, "n": 2}, {key: 10, "n": 3}]))
    assert stuck.detected


def test_a_non_numeric_progress_field_is_not_a_counter() -> None:
    # Distinct strings under a counter-like key must neither crash nor count
    # as measurable advancement.
    result = _detect(
        _poll([{"progress": "starting"}, {"progress": "running"}, {"progress": "finishing"}])
    )

    assert result.detected
    assert _issue(result, "tool_call_loop") is not None


@pytest.mark.parametrize(
    "payloads",
    [
        # regresses in the middle
        [{"progress": 50}, {"progress": 20}, {"progress": 60}],
        # never advances, only the free-text note changes
        [
            {"progress": 30, "note": "a"},
            {"progress": 30, "note": "b"},
            {"progress": 30, "note": "c"},
        ],
        # one poll reports no counter at all
        [{"progress": 10}, {"note": "hello"}, {"progress": 90}],
    ],
)
def test_progress_that_is_not_monotonic_advancement_stays_a_loop(
    payloads: List[Any],
) -> None:
    result = _detect(_poll(payloads))

    assert result.detected
    assert _issue(result, "tool_call_loop") is not None


def test_failures_with_changing_payloads_stay_a_loop() -> None:
    # Distinct payload text, but every attempt explicitly failed.
    result = _detect(
        _poll(
            [
                {"exit_code": 1, "attempt": 1},
                {"exit_code": 1, "attempt": 2},
                {"exit_code": 1, "attempt": 3},
            ],
            name="run_migration",
        )
    )

    assert result.detected
    assert _issue(result, "tool_call_loop")["tool_name"] == "run_migration"


def test_failing_attempts_are_not_progress_even_if_a_counter_advances() -> None:
    result = _detect(
        _poll(
            [
                {"exit_code": 1, "progress": 10},
                {"exit_code": 1, "progress": 40},
                {"exit_code": 1, "progress": 90},
            ],
            name="upload_chunk",
        )
    )

    assert result.detected
    assert _issue(result, "tool_call_loop")["tool_name"] == "upload_chunk"


def test_error_prose_that_changes_each_time_stays_a_loop() -> None:
    result = _detect(
        _poll(
            [
                {"message": "request 41 failed after 30s"},
                {"message": "request 42 failed after 31s"},
                {"message": "request 43 failed after 32s"},
            ]
        )
    )

    assert result.detected


def test_repeated_terminal_successes_with_new_ids_are_not_progress() -> None:
    # Each attempt was already terminal; there was nothing to advance.
    result = _detect(
        _poll(
            [
                {"status": "success", "request_id": "r-1"},
                {"status": "success", "request_id": "r-2"},
                {"status": "success", "request_id": "r-3"},
            ]
        )
    )

    assert result.detected


def test_terminal_success_after_failed_attempts_counts_as_progress() -> None:
    result = _detect(
        _poll(
            [
                {"exit_code": 1, "attempt": 1},
                {"status": "running", "attempt": 2},
                {"exit_code": 0, "attempt": 3},
            ]
        )
    )

    assert not result.detected


def test_text_only_results_with_no_structured_signal_stay_a_loop() -> None:
    events: List[Dict[str, Any]] = []
    for text in ("looking into it", "still looking", "checking further"):
        events.append(_call("get_status", {"id": 1}))
        events.append({"type": "tool.result", "content": text})

    result = _detect(events)

    assert result.detected
    assert _issue(result, "tool_call_loop") is not None


def test_events_between_call_and_result_are_skipped_when_pairing() -> None:
    events: List[Dict[str, Any]] = []
    for payload in (
        {"status": "queued"},
        {"status": "running", "note": "half"},
        {"status": "completed"},
    ):
        events.append(_call("get_export_status", {"job_id": "job-7"}))
        events.append({"type": "agent.turn", "content": "waiting"})
        events.append(_result(payload))

    assert not _detect(events).detected


def test_a_later_calls_result_is_not_evidence_for_the_last_poll() -> None:
    poll = _poll([{"status": "queued"}, {"status": "running", "note": "x"}])
    unanswered = _call("get_export_status", {"job_id": "job-7"})
    # A terminal success that belongs to a DIFFERENT tool call. If it were
    # borrowed as the last poll's answer, the poll would look finished.
    other_tool = [_call("send_email", {"to": "a@b.c"}), _result({"status": "completed"})]

    result = _detect(poll + [unanswered] + other_tool)

    assert result.detected
    assert _issue(result, "tool_call_loop")["repeat_count"] == 3

    # Paired control: the same terminal result, but returned to the final poll.
    answered = _detect(poll + [unanswered, _result({"status": "completed"})] + other_tool)
    assert not answered.detected


def test_a_poll_with_no_result_in_the_middle_is_still_a_loop() -> None:
    call = _call("get_export_status", {"job_id": "job-7"})

    unanswered_middle = _detect(
        [call, _result({"status": "queued"}), call, call, _result({"status": "completed"})]
    )
    assert unanswered_middle.detected
    assert _issue(unanswered_middle, "tool_call_loop")["repeat_count"] == 3

    # Paired control: every poll answered, and the answers walk to completion.
    answered = _detect(
        [
            call,
            _result({"status": "queued"}),
            call,
            _result({"status": "running", "note": "half"}),
            call,
            _result({"status": "completed"}),
        ]
    )
    assert not answered.detected


def test_each_poll_is_paired_with_only_its_nearest_result() -> None:
    # Every poll's own answer is "queued". The extra result events after it
    # (one of them a terminal success) belong to something else and must not
    # make the polls look like they advanced.
    events: List[Dict[str, Any]] = []
    for extra in ({"log": "a"}, {"log": "b"}, {"status": "completed"}):
        events.append(_call("get_export_status", {"job_id": "job-7"}))
        events.append(_result({"status": "queued"}))
        events.append(_result(extra))

    result = _detect(events)

    assert result.detected
    assert _issue(result, "tool_call_loop")["repeat_count"] == 3


def test_result_fields_carried_on_the_event_itself_are_read() -> None:
    # Results that have no `tool_result` payload keep their fields on the
    # event. The text differs each time, and the status field walks to done.
    events: List[Dict[str, Any]] = []
    for text, status in (("queued", "queued"), ("running now", "running"), ("all done", "completed")):
        events.append(_call("get_status", {"id": 1}))
        events.append({"type": "tool.result", "content": text, "status": status})

    assert not _detect(events).detected

    # Paired control: the same changing text with no status signal at all.
    no_signal: List[Dict[str, Any]] = []
    for text in ("queued", "running now", "all done"):
        no_signal.append(_call("get_status", {"id": 1}))
        no_signal.append({"type": "tool.result", "content": text})
    assert _detect(no_signal).detected


# --- fuzzy tool loop --------------------------------------------------------


def test_fuzzy_tool_loop_fires_on_same_shape_with_recurring_inputs() -> None:
    queries = ["pisama", "pisama", "detectors", "detectors", "pisama"]
    result = _detect([_call("search", {"q": q}) for q in queries])

    assert result.detected
    assert _issue(result, "tool_call_loop") is None
    issue = _issue(result, "fuzzy_tool_loop")
    assert issue is not None
    assert issue["tool_name"] == "search"
    assert issue["repeat_count"] == 5
    assert issue["turns"] == [0, 1, 2, 3, 4]
    assert result.confidence == 1.0
    assert result.severity == TurnAwareSeverity.MODERATE


def test_fuzzy_tool_loop_run_ends_at_a_different_tool() -> None:
    queries = ["a", "a", "b", "b", "a"]
    events = [_call("search", {"q": q}) for q in queries] + [_call("close", {"id": 1})]
    issue = _issue(_detect(events), "fuzzy_tool_loop")

    assert issue is not None
    assert issue["tool_name"] == "search"
    assert issue["repeat_count"] == 5
    assert issue["turns"] == [0, 1, 2, 3, 4]


def test_fuzzy_tool_loop_run_that_starts_mid_session_is_reported_from_its_start() -> None:
    queries = ["a", "a", "b", "b", "a"]
    events = [_call("open", {"id": 1})] + [_call("search", {"q": q}) for q in queries]
    issue = _issue(_detect(events), "fuzzy_tool_loop")

    assert issue is not None
    assert issue["tool_name"] == "search"
    assert issue["repeat_count"] == 5
    assert issue["turns"] == [1, 2, 3, 4, 5]


def test_fuzzy_tool_loop_needs_the_same_argument_shape() -> None:
    # Same tool and recurring inputs, but the argument keys alternate, so no
    # five calls share a structure.
    inputs = [{"a": 1}, {"b": 1}, {"a": 1}, {"b": 1}, {"a": 1}]

    assert not _detect([_call("search", i) for i in inputs]).detected


def test_fuzzy_tool_loop_needs_the_same_tool() -> None:
    names = ["search", "search", "lookup", "search", "search"]

    assert not _detect([_call(n, {"q": "same-shape"}) for n in names]).detected


def test_fuzzy_tool_loop_respects_the_event_gap_limit() -> None:
    queries = ["a", "a", "b", "b", "a"]

    def with_gap(fillers: int) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for q in queries:
            events.append(_call("search", {"q": q}))
            events.extend({"type": "agent.turn"} for _ in range(fillers))
        return events

    assert _issue(_detect(with_gap(3)), "fuzzy_tool_loop") is not None  # gap of 4
    assert not _detect(with_gap(4)).detected  # gap of 5


def test_fuzzy_tool_loop_stays_silent_when_every_input_is_distinct() -> None:
    result = _detect([_call("search", {"q": f"topic-{i}"}) for i in range(6)])

    assert not result.detected


def test_fuzzy_tool_loop_uses_nested_data_fields() -> None:
    events = [
        {"type": "tool.call", "data": {"name": "search", "input": {"q": q}}}
        for q in ("a", "a", "b", "b", "a")
    ]
    result = _detect(events)

    issue = _issue(result, "fuzzy_tool_loop")
    assert issue is not None
    assert issue["tool_name"] == "search"


# --- spawn / send ping-pong -------------------------------------------------


def test_repeated_sends_to_one_target_are_a_ping_pong() -> None:
    result = _detect(
        [{"type": "session.send", "target_session": "worker-1"} for _ in range(3)]
    )

    issue = _issue(result, "spawn_ping_pong")
    assert issue is not None
    assert issue["target"] == "worker-1"
    assert issue["repeat_count"] == 3
    assert result.affected_turns == [0, 1, 2]


def test_alternating_targets_are_an_abab_ping_pong_with_nested_targets() -> None:
    events = [
        {"type": "session.spawn", "data": {"recipient": who}}
        for who in ("planner", "coder", "planner", "coder")
    ]
    result = _detect(events)

    issue = _issue(result, "abab_ping_pong")
    assert issue is not None
    assert issue["targets"] == ["planner", "coder"]
    assert issue["repeat_count"] == 4
    assert result.severity == TurnAwareSeverity.MODERATE


def test_ping_pong_run_in_the_middle_is_reported_with_its_own_events() -> None:
    events: List[Dict[str, Any]] = [
        {"type": "session.send", "target_session": target}
        for target in ("a", "b", "b", "b", "c")
    ]
    events.insert(2, {"type": "agent.turn"})
    result = _detect(events)

    issue = _issue(result, "spawn_ping_pong")
    assert issue is not None
    assert issue["target"] == "b"
    assert issue["repeat_count"] == 3
    assert issue["turns"] == [1, 3, 4]


def test_ping_pong_run_at_the_end_does_not_absorb_earlier_targets() -> None:
    result = _detect(
        [
            {"type": "session.spawn", "target_session": target}
            for target in ("a", "b", "b", "b")
        ]
    )

    issue = _issue(result, "spawn_ping_pong")
    assert issue is not None
    assert issue["target"] == "b"
    assert issue["turns"] == [1, 2, 3]


def test_abab_run_that_starts_mid_session_is_reported_from_its_start() -> None:
    who = ["setup", "planner", "coder", "planner", "coder"]
    events: List[Dict[str, Any]] = [
        {"type": "session.send", "target_agent": w} for w in who
    ]
    events.insert(3, {"type": "agent.turn"})
    issue = _issue(_detect(events), "abab_ping_pong")

    assert issue is not None
    assert issue["targets"] == ["planner", "coder"]
    assert issue["repeat_count"] == 4
    assert issue["turns"] == [1, 2, 4, 5]


def test_three_alternating_sends_are_below_the_abab_bar() -> None:
    # A-B-A then a new target: only one and a half cycles.
    events = [
        {"type": "session.send", "target_session": who} for who in ("a", "b", "a", "c")
    ]

    assert not _detect(events).detected


def test_spawn_and_send_events_without_a_target_are_not_a_ping_pong() -> None:
    events = [{"type": "session.send", "content": "hello"} for _ in range(4)]

    assert not _detect(events).detected


@pytest.mark.parametrize(
    "event",
    [
        {"target_session": "w-1"},
        {"target_agent": "w-1"},
        {"spawned_session_id": "w-1"},
        {"data": {"target_agent": "w-1"}},
        {"data": {"target_session": "w-1"}},
        {"data": {"recipient": "w-1"}},
        {"data": {"recipient_agent": "w-1"}},
        {"data": {"target": "w-1"}},
        {"data": {"source_agent": "w-1"}},
        {"data": {"spawned_session_id": "w-1"}},
        {"data": {"agent": "w-1"}},
    ],
)
def test_every_supported_target_field_identifies_the_ping_pong_target(
    event: Dict[str, Any],
) -> None:
    result = _detect([{"type": "session.spawn", **event} for _ in range(3)])

    issue = _issue(result, "spawn_ping_pong")
    assert issue is not None
    assert issue["target"] == "w-1"


def test_top_level_target_wins_over_nested_target() -> None:
    event = {
        "type": "session.send",
        "target_session": "top",
        "data": {"target_agent": "nested"},
    }
    issue = _issue(_detect([dict(event) for _ in range(3)]), "spawn_ping_pong")

    assert issue is not None
    assert issue["target"] == "top"


def test_distinct_spawn_targets_are_not_a_ping_pong() -> None:
    events = [
        {"type": "session.spawn", "target_agent": who}
        for who in ("a", "b", "c", "d", "e")
    ]

    assert not _detect(events).detected


# --- exact message loop -----------------------------------------------------


def test_identical_messages_are_a_message_loop_with_preview() -> None:
    text = "Still waiting for the upstream service to answer."
    result = _detect([_msg(text) for _ in range(3)])

    issue = _issue(result, "message_sent_loop")
    assert issue is not None
    assert issue["repeat_count"] == 3
    assert issue["content_preview"] == text


@pytest.mark.parametrize("event_type", ["message.sent", "message.send"])
@pytest.mark.parametrize(
    "event",
    [
        {"content": "  Still polling the queue.  "},
        {"message": "Still polling the queue."},
        {"text": "Still polling the queue."},
        {"data": {"content": "Still polling the queue."}},
        {"data": {"message": "Still polling the queue."}},
        {"data": {"text": "Still polling the queue."}},
    ],
)
def test_message_loop_reads_every_content_field_for_both_event_types(
    event_type: str, event: Dict[str, Any]
) -> None:
    result = _detect([{"type": event_type, **event} for _ in range(3)])

    issue = _issue(result, "message_sent_loop")
    assert issue is not None
    assert issue["content_preview"] == "Still polling the queue."


def test_message_loop_preview_is_truncated_to_100_characters() -> None:
    text = "x" * 150
    issue = _issue(_detect([_msg(text) for _ in range(3)]), "message_sent_loop")

    assert issue is not None
    assert issue["content_preview"] == "x" * 100


def test_message_loop_ignores_surrounding_whitespace() -> None:
    result = _detect([_msg("Working on it"), _msg("  Working on it "), _msg("Working on it\n")])

    assert _issue(result, "message_sent_loop") is not None


def test_empty_messages_are_not_an_exact_loop() -> None:
    assert not _detect([_msg("") for _ in range(4)]).detected


def test_message_loop_run_ends_at_a_different_message() -> None:
    same = "Retrying the upload now."
    result = _detect([_msg(same)] * 3 + [_msg("Upload finished, moving on.")])

    issue = _issue(result, "message_sent_loop")
    assert issue is not None
    assert issue["repeat_count"] == 3
    assert issue["turns"] == [0, 1, 2]


def test_message_loop_run_that_starts_mid_session_is_reported_from_its_start() -> None:
    repeated = "Retrying the upload now."
    result = _detect([_msg("Starting the upload.")] + [_msg(repeated)] * 3)

    issue = _issue(result, "message_sent_loop")
    assert issue is not None
    assert issue["turns"] == [1, 2, 3]
    assert issue["content_preview"] == repeated


# --- fuzzy message loop -----------------------------------------------------

_STEM = "The export is still not ready, so I am going to check the job status again"


def test_appended_variation_repeats_are_a_fuzzy_message_loop() -> None:
    result = _detect([_msg(f"{_STEM} (try {i})") for i in range(1, 5)])

    assert result.detected
    assert _issue(result, "message_sent_loop") is None
    issue = _issue(result, "fuzzy_message_loop")
    assert issue is not None
    assert issue["repeat_count"] == 4
    assert issue["turns"] == [0, 1, 2, 3]
    assert result.confidence == pytest.approx(0.8)
    assert result.severity == TurnAwareSeverity.MODERATE
    assert "near-identical" in issue["description"]


def test_exact_message_loop_suppresses_a_separate_fuzzy_message_report() -> None:
    events = [_msg("Retrying the upload now.")] * 3 + [
        _msg(f"{_STEM} (try {i})") for i in range(1, 5)
    ]
    result = _detect(events)

    assert [i["type"] for i in result.evidence["issues"]] == ["message_sent_loop"]


def test_fuzzy_message_run_that_starts_mid_session_is_reported_from_its_start() -> None:
    events = [_msg("Kicking off the export.")] + [
        _msg(f"{_STEM} (try {i})") for i in range(1, 5)
    ]
    issue = _issue(_detect(events), "fuzzy_message_loop")

    assert issue is not None
    assert issue["repeat_count"] == 4
    assert issue["turns"] == [1, 2, 3, 4]


def test_fuzzy_message_run_followed_by_unrelated_messages_is_still_reported() -> None:
    events = [_msg(f"{_STEM} (try {i})") for i in range(1, 5)] + [
        _msg("Completely different note about the weather forecast today."),
        _msg("Another unrelated remark regarding lunch plans for tomorrow."),
    ]
    issue = _issue(_detect(events), "fuzzy_message_loop")

    assert issue is not None
    assert issue["turns"] == [0, 1, 2, 3]


def test_fuzzy_message_loop_reads_nested_data_content() -> None:
    events = [
        {"type": "message.send", "data": {"text": f"{_STEM} (try {i})"}}
        for i in range(1, 5)
    ]

    assert _issue(_detect(events), "fuzzy_message_loop") is not None


def test_three_appended_variations_are_below_the_fuzzy_message_bar() -> None:
    result = _detect([_msg(f"{_STEM} (try {i})") for i in range(1, 4)])

    assert not result.detected


def test_embedded_variation_is_legitimate_repetitive_work() -> None:
    # Per-item status lines share text on BOTH sides of the varying part.
    result = _detect(
        [_msg(f"Processed item {i} of 40 successfully and stored it.") for i in range(1, 7)]
    )

    assert not result.detected


def test_empty_messages_never_form_a_fuzzy_run() -> None:
    result = _detect([_msg("") for _ in range(5)])

    assert not result.detected


def test_fuzzy_message_run_is_broken_by_an_unrelated_message() -> None:
    events = [
        _msg(f"{_STEM} (try 1)"),
        _msg(f"{_STEM} (try 2)"),
        _msg("Completely different note about the weather forecast today."),
        _msg(f"{_STEM} (try 3)"),
        _msg(f"{_STEM} (try 4)"),
    ]

    assert not _detect(events).detected


# --- combined reporting -----------------------------------------------------


def test_multiple_loop_patterns_are_reported_together() -> None:
    # Messages come FIRST in the session but are reported LAST, so the
    # affected turns only come out ordered if they are sorted.
    events = [
        _msg(f"{_STEM} (try 1)"),
        _msg(f"{_STEM} (try 2)"),
        _msg(f"{_STEM} (try 3)"),
        _msg(f"{_STEM} (try 4)"),
        _call("fetch", {"u": 1}),
        _call("fetch", {"u": 1}),
        _call("fetch", {"u": 1}),
        {"type": "session.send", "target_session": "w"},
        {"type": "session.send", "target_session": "w"},
        {"type": "session.send", "target_session": "w"},
        {"type": "session.send", "target_session": "w"},
    ]
    result = _detect(events)

    kinds = [issue["type"] for issue in result.evidence["issues"]]
    assert kinds == ["tool_call_loop", "spawn_ping_pong", "fuzzy_message_loop"]
    assert result.evidence["total_events"] == len(events)
    assert result.affected_turns == list(range(len(events)))
    assert result.explanation == "Session loop detected: 3 pattern(s), max 4 repeats"
    assert result.severity == TurnAwareSeverity.MODERATE
    assert result.confidence == pytest.approx(0.8)


# --- pure helpers -----------------------------------------------------------


def test_common_prefix_and_suffix_lengths() -> None:
    assert _common_prefix_len("abcdef", "abcxyz") == 3
    assert _common_prefix_len("abc", "xyz") == 0
    assert _common_prefix_len("abc", "abcdef") == 3
    assert _common_suffix_len("abcxyz", "defxyz") == 3
    assert _common_suffix_len("abc", "xyz") == 0
    assert _common_suffix_len("xyz", "abcxyz") == 3


def test_variation_is_appended_distinguishes_tacked_on_from_embedded() -> None:
    stem = "the job is still running please wait"
    assert _variation_is_appended(stem + " (1)", stem + " (2)")
    assert _variation_is_appended(stem, stem + " and also this extra tail text")
    # Embedded difference: long shared text on both sides.
    assert not _variation_is_appended(
        "item 1 of 40 was processed successfully",
        "item 2 of 40 was processed successfully",
    )
    # Different messages: no shared stem.
    assert not _variation_is_appended("alpha beta gamma", "zzz yyy xxx")
    # Identical messages share everything, which is the exact tier's job.
    assert not _variation_is_appended(stem, stem)
    # A short shared opening is not a shared stem when the rest is long and different.
    assert not _variation_is_appended("hello", "hello" + " and then a long unrelated tail" * 4)
    # Nothing to compare.
    assert not _variation_is_appended("", "")


def test_message_content_prefers_top_level_then_nested_data() -> None:
    assert _message_content({"content": "  hi  "}) == "hi"
    assert _message_content({"message": "m", "text": "t"}) == "m"
    assert _message_content({"text": "t"}) == "t"
    assert _message_content({"data": {"content": "c"}}) == "c"
    assert _message_content({"data": {"message": "m"}}) == "m"
    assert _message_content({"data": {"text": "t"}}) == "t"
    assert _message_content({"data": "not a dict"}) == ""
    assert _message_content({}) == ""


def test_hash_input_is_key_order_independent_but_value_sensitive() -> None:
    assert _hash_input({"a": 1, "b": 2}) == _hash_input({"b": 2, "a": 1})
    assert _hash_input({"a": 1, "b": 2}) != _hash_input({"a": 1, "b": 3})
    assert _hash_input(None) != _hash_input({})


def test_structural_hash_ignores_values_but_not_keys_or_types() -> None:
    assert _structural_hash({"a": 1, "b": "x"}) == _structural_hash({"b": "y", "a": 2})
    assert _structural_hash({"q": "a"}) == _structural_hash({"q": "b"})
    assert _structural_hash({"q": "a"}) != _structural_hash({"r": "a"})
    assert _structural_hash({"q": "a"}) != _structural_hash({"q": 1})
    # Non-dict inputs hash by type name only.
    assert _structural_hash("abc") == _structural_hash("def")
    assert _structural_hash("abc") != _structural_hash(["abc"])
    assert _structural_hash(None) == _structural_hash(None)
