"""Behavioral tests for the coordination detector fixes ported from the backend.

Each block covers one behavior added by the reconciliation: conflicting file
edits, priority inversion, unacknowledged deadlock chains, repeat-request loops,
group-bus stuck loops, hierarchical-topology suppression, telemetry abstention,
and the confidence rules layered on top. Every test drives the real
``CoordinationAnalyzer`` (or the public ``pd.detect_coordination`` wrapper) with
hand-written payloads and pairs the risky input with a benign look-alike.
"""

from __future__ import annotations

from typing import Any

import pytest

import pisama_detectors as pd
from pisama_detectors.detection.coordination import (
    CoordinationAnalysisResult,
    CoordinationAnalyzer,
    Message,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64


def _analyze(
    messages: list[Message],
    agent_ids: list[str],
    **kwargs: Any,
) -> CoordinationAnalysisResult:
    return CoordinationAnalyzer().analyze_coordination_with_confidence(
        messages, agent_ids, **kwargs
    )


def _types(result: CoordinationAnalysisResult) -> list[str]:
    return [issue.issue_type for issue in result.issues]


def _issues(result: CoordinationAnalysisResult, issue_type: str) -> list[Any]:
    return [issue for issue in result.issues if issue.issue_type == issue_type]


def _edit(
    old_start: int,
    old_count: int,
    *,
    added: str = HASH_A,
    deleted: str = HASH_B,
    added_count: int = 1,
    deleted_count: int | None = None,
) -> dict[str, Any]:
    return {
        "old_start": old_start,
        "old_count": old_count,
        "added_sha256": added,
        "deleted_sha256": deleted,
        "added_count": added_count,
        "deleted_count": old_count if deleted_count is None else deleted_count,
    }


def _change(agent_id: str, path: str, *edits: dict[str, Any]) -> dict[str, Any]:
    return {"agent_id": agent_id, "files": [{"path": path, "edit_blocks": list(edits)}]}


# ---------------------------------------------------------------------------
# Conflicting resource changes
# ---------------------------------------------------------------------------


def test_overlapping_incompatible_edits_are_critical_and_lift_confidence() -> None:
    changes = [
        _change("agent_b", "src/app.py", _edit(10, 5, added=HASH_A)),
        _change("agent_a", "src/app.py", _edit(12, 4, added=HASH_C)),
    ]

    result = _analyze(
        [], ["agent_a", "agent_b"], resource_changes=changes, message_telemetry_present=False
    )

    conflicts = _issues(result, "conflicting_resource_change")
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.severity == "critical"
    # Agents are reported in sorted order regardless of input order.
    assert conflict.agents_involved == ["agent_a", "agent_b"]
    assert "src/app.py" in conflict.message
    assert result.detected
    assert not result.healthy
    # Structural evidence is pinned to the 0.96 floor, above any threshold in use.
    assert result.confidence == 0.96


def test_resource_conflict_confidence_beats_comparable_conversational_critical() -> None:
    conflict = _analyze(
        [],
        ["agent_a", "agent_b"],
        resource_changes=[
            _change("agent_a", "f.py", _edit(1, 3, added=HASH_A)),
            _change("agent_b", "f.py", _edit(2, 3, added=HASH_C)),
        ],
        message_telemetry_present=False,
    )
    # A lone conversational critical issue in an otherwise healthy, acknowledged exchange.
    inversion = _analyze(
        [
            Message(
                "planner",
                "worker",
                "priority urgent task blocked by low priority report job",
                0.0,
                True,
            ),
            Message("worker", "planner", "Understood, escalating the report job now.", 1.0, True),
        ],
        ["planner", "worker"],
    )

    assert _types(inversion) == ["priority_inversion"]
    assert inversion.confidence < 0.96
    assert conflict.confidence == 0.96


def test_resource_conflict_confidence_respects_scaling() -> None:
    analyzer = CoordinationAnalyzer(confidence_scaling=0.5)
    result = analyzer.analyze_coordination_with_confidence(
        [],
        ["agent_a", "agent_b"],
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", "f.py", _edit(1, 3, added=HASH_A)),
            _change("agent_b", "f.py", _edit(2, 3, added=HASH_C)),
        ],
    )

    assert result.confidence == 0.48


def test_legacy_analyze_coordination_reports_resource_conflicts() -> None:
    result = CoordinationAnalyzer().analyze_coordination(
        [],
        ["agent_a", "agent_b"],
        resource_changes=[
            _change("agent_a", "f.py", _edit(20, 2, added=HASH_A)),
            _change("agent_b", "f.py", _edit(21, 2, added=HASH_C)),
        ],
    )

    conflicts = [i for i in result.issues if i.issue_type == "conflicting_resource_change"]
    assert len(conflicts) == 1
    assert conflicts[0].agents_involved == ["agent_a", "agent_b"]
    assert not result.healthy


def test_identical_edits_from_two_agents_are_not_a_conflict() -> None:
    same = _edit(10, 5, added=HASH_A, deleted=HASH_B)
    result = _analyze(
        [],
        ["agent_a", "agent_b"],
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", "src/app.py", same),
            _change("agent_b", "src/app.py", dict(same)),
        ],
    )

    assert not result.detected
    assert result.confidence == 0.0


def test_same_hashes_with_different_line_counts_still_conflict() -> None:
    result = _analyze(
        [],
        ["agent_a", "agent_b"],
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", "f.py", _edit(5, 4, added_count=2)),
            _change("agent_b", "f.py", _edit(5, 4, added_count=7)),
        ],
    )

    assert _types(result) == ["conflicting_resource_change"]


@pytest.mark.parametrize(
    "variant",
    [_edit(5, 4, deleted=HASH_C), _edit(5, 4, deleted_count=1)],
    ids=["deleted-hash", "deleted-count"],
)
def test_edits_differing_only_on_the_deleted_side_still_conflict(variant: dict[str, Any]) -> None:
    result = _analyze(
        [],
        ["agent_a", "agent_b"],
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", "f.py", _edit(5, 4)),
            _change("agent_b", "f.py", variant),
        ],
    )

    assert _types(result) == ["conflicting_resource_change"]


def test_missing_deleted_count_falls_back_to_the_old_range_length() -> None:
    explicit = _edit(5, 4)  # deleted_count == old_count == 4
    implicit = {k: v for k, v in explicit.items() if k != "deleted_count"}
    agents = ["agent_a", "agent_b"]

    same_edit = _analyze(
        [],
        agents,
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", "f.py", explicit),
            _change("agent_b", "f.py", implicit),
        ],
    )
    different_edit = _analyze(
        [],
        agents,
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", "f.py", _edit(5, 4, deleted_count=9)),
            _change("agent_b", "f.py", implicit),
        ],
    )

    # The two spellings of the same edit are identical, so not a conflict.
    assert not same_edit.detected
    assert _types(different_edit) == ["conflicting_resource_change"]


@pytest.mark.parametrize(
    ("first", "second", "overlaps"),
    [
        # Disjoint and merely adjacent ranges are healthy parallel work.
        ((10, 5), (30, 5), False),
        ((10, 5), (15, 5), False),
        # Partial and nested overlaps conflict.
        ((10, 5), (14, 5), True),
        ((10, 20), (15, 2), True),
        # A pure insertion (count 0) conflicts when it lands inside the other
        # range and is clear of it when it lands beyond, in either argument order.
        ((10, 0), (5, 10), True),
        ((5, 10), (10, 0), True),
        ((5, 5), (11, 0), False),
        ((11, 0), (5, 5), False),
        # ...and equally when it lands clear of the range on the low side.
        ((3, 0), (5, 5), False),
        ((5, 5), (3, 0), False),
        # Two insertions conflict only at the same anchor line.
        ((7, 0), (7, 0), True),
        ((7, 0), (8, 0), False),
    ],
)
def test_range_overlap_rules(
    first: tuple[int, int], second: tuple[int, int], overlaps: bool
) -> None:
    result = _analyze(
        [],
        ["agent_a", "agent_b"],
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", "f.py", _edit(*first, added=HASH_A)),
            _change("agent_b", "f.py", _edit(*second, added=HASH_C)),
        ],
    )

    assert bool(_issues(result, "conflicting_resource_change")) is overlaps


def test_same_agent_edits_never_conflict_with_themselves() -> None:
    result = _analyze(
        [],
        ["agent_a"],
        message_telemetry_present=False,
        resource_changes=[
            _change(
                "agent_a",
                "f.py",
                _edit(10, 5, added=HASH_A),
                _edit(11, 5, added=HASH_C),
            )
        ],
    )

    assert not result.detected


def test_edits_in_different_files_do_not_conflict() -> None:
    result = _analyze(
        [],
        ["agent_a", "agent_b"],
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", "src/a.py", _edit(10, 5, added=HASH_A)),
            _change("agent_b", "src/b.py", _edit(10, 5, added=HASH_C)),
        ],
    )

    assert not result.detected


@pytest.mark.parametrize(
    "path_a, path_b",
    [
        ("a/src/app.py", "b/src/app.py"),
        ("./src/app.py", "src/app.py"),
        ("src\\app.py", "src/app.py"),
        ("  ././a/src/app.py  ", "src/app.py"),
    ],
)
def test_paths_are_normalized_before_comparison(path_a: str, path_b: str) -> None:
    result = _analyze(
        [],
        ["agent_a", "agent_b"],
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", path_a, _edit(10, 5, added=HASH_A)),
            _change("agent_b", path_b, _edit(10, 5, added=HASH_C)),
        ],
    )

    conflicts = _issues(result, "conflicting_resource_change")
    assert len(conflicts) == 1
    assert "'src/app.py'" in conflicts[0].message


def test_each_agent_pair_is_reported_once_per_file() -> None:
    result = _analyze(
        [],
        ["agent_a", "agent_b", "agent_c"],
        message_telemetry_present=False,
        resource_changes=[
            _change(
                "agent_a",
                "f.py",
                _edit(10, 5, added=HASH_A),
                _edit(12, 2, added=HASH_D),
            ),
            _change("agent_b", "f.py", _edit(11, 5, added=HASH_C)),
            _change("agent_c", "f.py", _edit(12, 3, added=HASH_B, deleted=HASH_D)),
        ],
    )

    pairs = sorted(tuple(i.agents_involved) for i in _issues(result, "conflicting_resource_change"))
    assert pairs == [
        ("agent_a", "agent_b"),
        ("agent_a", "agent_c"),
        ("agent_b", "agent_c"),
    ]


@pytest.mark.parametrize(
    "thin_edit",
    [
        {"old_start": 10, "old_count": 5},
        {**_edit(10, 5), "added_sha256": "short"},
        {**_edit(10, 5), "deleted_sha256": "short"},
        {**_edit(10, 5), "added_sha256": None},
        {**_edit(10, 5), "deleted_sha256": None},
        {**_edit(10, 5), "added_count": "many"},
    ],
    ids=[
        "no-hashes",
        "short-added-hash",
        "short-deleted-hash",
        "null-added-hash",
        "null-deleted-hash",
        "non-numeric-count",
    ],
)
def test_thin_or_malformed_edit_telemetry_abstains(thin_edit: dict[str, Any]) -> None:
    for first, second in (
        (thin_edit, _edit(10, 5, added=HASH_C)),
        (_edit(10, 5, added=HASH_C), thin_edit),
    ):
        result = _analyze(
            [],
            ["agent_a", "agent_b"],
            message_telemetry_present=False,
            resource_changes=[
                _change("agent_a", "f.py", first),
                _change("agent_b", "f.py", second),
            ],
        )
        assert not result.detected


@pytest.mark.parametrize(
    "bad_edit",
    [
        {**_edit(10, 5), "old_start": "ten"},
        {**_edit(10, 5), "old_start": None},
        {k: v for k, v in _edit(10, 5).items() if k != "old_start"},
        # A range that would span lines 10-14 if the negative start were trusted.
        _edit(-3, 20),
    ],
    ids=["text-start", "null-start", "missing-start", "negative-start"],
)
def test_unusable_line_ranges_never_produce_a_conflict(bad_edit: dict[str, Any]) -> None:
    result = _analyze(
        [],
        ["agent_a", "agent_b"],
        message_telemetry_present=False,
        resource_changes=[
            _change("agent_a", "f.py", bad_edit),
            _change("agent_b", "f.py", _edit(10, 5, added=HASH_C)),
        ],
    )

    assert not result.detected


def test_malformed_resource_change_records_are_skipped() -> None:
    valid_a = _change("agent_a", "f.py", _edit(10, 5, added=HASH_A))
    valid_b = _change("agent_b", "f.py", _edit(10, 5, added=HASH_C))
    clashing = _edit(10, 5, added=HASH_D)
    junk: list[Any] = [
        "not-a-dict",
        None,
        {"files": [{"path": "f.py", "edit_blocks": [clashing]}]},  # no agent id
        {
            "agent_id": "  ",
            "files": [{"path": "f.py", "edit_blocks": [clashing]}],
        },  # blank agent id
        {"agent_id": "agent_z", "files": "f.py"},  # files is not a list
        {"agent_id": "agent_z", "files": ["f.py", None]},  # file entries are not dicts
        {
            "agent_id": "agent_z",
            "files": [{"path": "f.py", "edit_blocks": "oops"}],
        },  # blocks not a list
        {
            "agent_id": "agent_z",
            "files": [{"path": "f.py", "edit_blocks": ["x", 3, None]}],
        },  # blocks not dicts
        # Two agents clashing on an empty path or /dev/null (file creation or
        # deletion) is not an overlapping edit of a shared base file.
        _change("agent_y", "", _edit(1, 2, added=HASH_A)),
        _change("agent_z", "", _edit(1, 2, added=HASH_C)),
        _change("agent_y", "/dev/null", _edit(1, 2, added=HASH_A)),
        _change("agent_z", "/dev/null", _edit(1, 2, added=HASH_C)),
    ]
    agents = ["agent_a", "agent_b", "agent_y", "agent_z"]

    only_junk = _analyze([], agents, resource_changes=junk, message_telemetry_present=False)
    mixed = _analyze(
        [], agents, resource_changes=[*junk, valid_a, valid_b], message_telemetry_present=False
    )

    assert not only_junk.detected
    assert [i.agents_involved for i in _issues(mixed, "conflicting_resource_change")] == [
        ["agent_a", "agent_b"]
    ]


def test_omitted_resource_changes_leave_message_only_analysis_unchanged() -> None:
    messages = [Message("planner", "worker", "Draft the release notes for version two.", 0.0, True)]

    omitted = _analyze(messages, ["planner", "worker"])
    empty = _analyze(messages, ["planner", "worker"], resource_changes=[])

    # The worker never spoke, which is the only finding; no file conflict is invented.
    assert _types(omitted) == ["silent_agent"]
    assert _types(empty) == _types(omitted)
    assert empty.confidence == omitted.confidence
    assert 0.0 < omitted.confidence <= 0.45


# ---------------------------------------------------------------------------
# Priority inversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        '{"priority":"URGENT","status":"blocked","blocked_by":"low_priority_report"}',
        "Our P0 payment task is blocked by a low priority reporting job.",
        "critical ingest is waiting on lower priority batch export to release the queue",
        "High priority migration is blocked, sitting behind low-priority cleanup",
        # Each urgency spelling, and the "waiting for" blocked spelling.
        "Urgent deploy is blocked by a low priority lint job.",
        "P1 checkout is waiting for low priority backfill to release the lock",
        '{"priority":"high","status":"blocked","blocked_by":"low_priority_cleanup"}',
        "task tagged priority_urgent is blocked by low priority export",
        "task tagged priority_critical is blocked by low priority export",
    ],
)
def test_priority_inversion_fires_when_urgent_work_is_blocked_by_low_priority_work(
    content: str,
) -> None:
    result = _analyze([Message("scheduler", "worker", content, 1.0)], ["scheduler", "worker"])

    inversions = _issues(result, "priority_inversion")
    assert len(inversions) == 1
    assert inversions[0].severity == "critical"
    assert inversions[0].agents_involved == ["scheduler", "worker"]
    assert content[:40] in inversions[0].message
    assert not result.healthy


@pytest.mark.parametrize(
    "content",
    [
        # Urgent but not blocked.
        '{"priority":"URGENT","status":"running"}',
        # Blocked but not urgent.
        "The nightly report is blocked by a low priority export.",
        # Urgent and blocked, but by something that is not lower priority.
        "URGENT deploy is blocked by a failing database migration.",
        # Urgent and waiting, but the blocker is explicitly high priority.
        "Urgent task is waiting on high priority approval.",
        # Urgent and queued behind low priority work, but never reported as blocked.
        "Urgent P0 task sits behind low priority cleanup in the queue.",
        # Urgent and blocked, but the low priority job is not what blocks it.
        "Urgent invoice job is blocked by the approval queue. Separately, the weekly "
        "cleanup of the low priority archive job also finished.",
    ],
)
def test_priority_inversion_needs_urgency_blocking_and_a_lower_priority_blocker(
    content: str,
) -> None:
    result = _analyze([Message("scheduler", "worker", content, 1.0)], ["scheduler", "worker"])

    assert "priority_inversion" not in _types(result)


def test_priority_inversion_reaches_the_public_wrapper() -> None:
    result = pd.detect_coordination(
        [
            {
                "sender": "scheduler",
                "receiver": "executor",
                "content": '{"priority":"URGENT","status":"blocked","blocked_by":"low_priority_report"}',
                "timestamp": 1,
            }
        ],
        ["scheduler", "executor"],
    )

    assert result.detected
    assert "priority_inversion" in {i.issue_type for i in result.issues}
    assert result.confidence >= 0.5


# ---------------------------------------------------------------------------
# Unacknowledged deadlock chain
# ---------------------------------------------------------------------------


def _blocked(sender: str, receiver: str, resource: str, ts: float, *, ack: bool = False) -> Message:
    return Message(sender, receiver, f"access {resource} (blocked)", ts, ack)


def test_three_unacked_blocking_messages_form_a_deadlock_chain() -> None:
    messages = [
        _blocked("agent_a", "agent_b", "resource_a", 0.0),
        _blocked("agent_b", "agent_c", "resource_b", 1.0),
        _blocked("agent_c", "coordinator", "resource_c", 2.0),
    ]

    result = _analyze(messages, ["agent_a", "agent_b", "agent_c", "coordinator"])

    chains = _issues(result, "unacked_deadlock_chain")
    assert len(chains) == 1
    assert chains[0].severity == "high"
    assert chains[0].agents_involved == ["agent_a", "agent_b", "agent_c", "coordinator"]
    assert "3 unacknowledged blocking messages across 4 agents" in chains[0].message
    assert not result.healthy


def test_deadlock_chain_reaches_the_public_wrapper() -> None:
    result = pd.detect_coordination(
        [
            {"sender": "a", "receiver": "b", "content": "awaiting lock on table_x", "timestamp": 0},
            {
                "sender": "b",
                "receiver": "c",
                "content": "queued behind a stalled job",
                "timestamp": 1,
            },
            {
                "sender": "c",
                "receiver": "a",
                "content": "cannot proceed without table_y",
                "timestamp": 2,
            },
        ],
        ["a", "b", "c"],
    )

    assert "unacked_deadlock_chain" in {i.issue_type for i in result.issues}
    assert result.detected


@pytest.mark.parametrize(
    "phrase",
    [
        "blocked",
        "waiting for",
        "wait for",
        "queued",
        "pending",
        "cannot proceed",
        "stalled",
        "awaiting",
    ],
)
def test_every_blocking_phrase_counts_toward_the_chain(phrase: str) -> None:
    messages = [
        Message(f"agent_{i}", f"agent_{i + 1}", f"job {i} is {phrase} on the lock", float(i))
        for i in range(3)
    ]

    assert "unacked_deadlock_chain" in _types(_analyze(messages, [f"agent_{i}" for i in range(4)]))


def test_deadlock_chain_needs_at_least_three_messages_and_three_blocked() -> None:
    two = [
        _blocked("agent_a", "agent_b", "resource_a", 0.0),
        _blocked("agent_b", "agent_c", "resource_b", 1.0),
    ]
    two_blocked_one_normal = [
        *two,
        Message("agent_c", "agent_a", "Sharing the weekly numbers.", 2.0),
    ]

    assert "unacked_deadlock_chain" not in _types(_analyze(two, ["agent_a", "agent_b", "agent_c"]))
    assert "unacked_deadlock_chain" not in _types(
        _analyze(two_blocked_one_normal, ["agent_a", "agent_b", "agent_c"])
    )


def test_acknowledged_blocking_messages_are_not_a_deadlock() -> None:
    messages = [
        _blocked("agent_a", "agent_b", "resource_a", 0.0, ack=True),
        _blocked("agent_b", "agent_c", "resource_b", 1.0, ack=True),
        _blocked("agent_c", "coordinator", "resource_c", 2.0, ack=True),
    ]

    assert "unacked_deadlock_chain" not in _types(
        _analyze(messages, ["agent_a", "agent_b", "agent_c", "coordinator"])
    )


def test_deadlock_chain_ignores_messages_without_blocking_language() -> None:
    messages = [
        Message("agent_a", "agent_b", "Here is the first draft of the summary.", 0.0),
        Message("agent_b", "agent_c", "Forwarding the summary for review.", 1.0),
        Message("agent_c", "coordinator", "Review notes attached below.", 2.0),
    ]

    assert "unacked_deadlock_chain" not in _types(
        _analyze(messages, ["agent_a", "agent_b", "agent_c", "coordinator"])
    )


def test_deadlock_chain_ratio_guard_when_some_messages_were_acknowledged() -> None:
    blocked = [
        _blocked("agent_a", "agent_b", "resource_a", 0.0),
        _blocked("agent_b", "agent_c", "resource_b", 1.0),
        _blocked("agent_c", "agent_a", "resource_c", 2.0),
    ]
    # 3 of 6 messages are unacked and blocked (50% < 60%): unrelated notices.
    diluted = [
        *blocked,
        Message("agent_a", "agent_b", "Thanks for the update.", 3.0, True),
        Message("agent_b", "agent_c", "Sending the invoice totals.", 4.0, True),
        Message("agent_c", "agent_a", "Invoice totals received.", 5.0, True),
    ]
    # 3 of 5 (exactly 60%) is enough once the ratio bar is met.
    dense = [
        *blocked,
        Message("agent_a", "agent_b", "Thanks for the update.", 3.0, True),
        Message("agent_b", "agent_c", "Sending the invoice totals.", 4.0, False),
    ]
    agents = ["agent_a", "agent_b", "agent_c"]

    assert "unacked_deadlock_chain" not in _types(_analyze(diluted, agents))
    assert "unacked_deadlock_chain" in _types(_analyze(dense, agents))


def test_deadlock_chain_fires_when_nothing_at_all_was_acknowledged() -> None:
    # With ack_rate == 0 the ratio guard does not apply: every message is unacked.
    messages = [_blocked("agent_a", "agent_b", "resource_a", float(i)) for i in range(3)]
    messages += [
        Message("agent_b", "agent_a", f"Status note number {i} for the log.", 10.0 + i)
        for i in range(7)
    ]

    result = _analyze(messages, ["agent_a", "agent_b", "agent_c"])

    chain = _issues(result, "unacked_deadlock_chain")
    assert len(chain) == 1
    assert chain[0].agents_involved == ["agent_a", "agent_b"]


# ---------------------------------------------------------------------------
# Repeat-request loops
# ---------------------------------------------------------------------------

REQUEST = "Please collect the quarterly sales figures for the northern region"


def _repeat(sender: str, receiver: str, content: str = REQUEST, count: int = 3) -> list[Message]:
    return [Message(sender, receiver, content, float(i), True) for i in range(count)]


def test_same_sender_repeating_a_request_is_a_repeat_request_loop() -> None:
    result = _analyze(_repeat("planner", "collector", count=3), ["planner", "collector"])

    loops = _issues(result, "repeat_request_loop")
    assert len(loops) == 1
    assert loops[0].severity == "high"
    assert sorted(loops[0].agents_involved) == ["collector", "planner"]
    assert "'planner' sent 3 near-duplicate messages" in loops[0].message
    assert "Mutual-request" not in loops[0].message
    assert not result.healthy


def test_two_agents_echoing_the_same_request_is_a_mutual_request_deadlock() -> None:
    messages = [
        Message(
            "planner",
            "collector",
            "Please send the updated inventory spreadsheet immediately",
            0.0,
            True,
        ),
        Message(
            "collector",
            "planner",
            "Please send the updated inventory spreadsheet immediately",
            1.0,
            True,
        ),
    ]

    result = _analyze(messages, ["planner", "collector"])

    loops = _issues(result, "repeat_request_loop")
    assert len(loops) == 1
    assert loops[0].message.startswith("Mutual-request deadlock between")
    assert "2 near-duplicate exchanges" in loops[0].message


def test_paraphrased_re_request_clusters_below_the_old_seventy_percent_gate() -> None:
    messages = [
        Message(
            "planner", "uploader", "please retry uploading quarterly report archive", 0.0, True
        ),
        Message(
            "planner", "uploader", "retry uploading quarterly report archive today soon", 1.0, True
        ),
    ]

    assert "repeat_request_loop" in _types(_analyze(messages, ["planner", "uploader"]))


def _two_requests(first: str, second: str) -> list[Message]:
    return [
        Message("planner", "uploader", first, 0.0, True),
        Message("planner", "uploader", second, 1.0, True),
    ]


def test_similarity_gate_is_sixty_percent_and_inclusive() -> None:
    agents = ["planner", "uploader"]
    # Stem overlap 3 of 5 (0.60): the boundary counts as a repeat.
    at_gate = _two_requests("retry uploading quarterly report", "retry uploading quarterly summary")
    # Stem overlap 2 of 4 (0.50): two related but different requests.
    below_gate = _two_requests("retry uploading report", "retry uploading summary")

    assert "repeat_request_loop" in _types(_analyze(at_gate, agents))
    assert "repeat_request_loop" not in _types(_analyze(below_gate, agents))


def test_filler_words_and_word_endings_do_not_hide_a_repeated_request() -> None:
    agents = ["planner", "uploader"]
    # Only words of 4+ characters carry meaning; the short filler differs completely.
    filler = _two_requests("retry the quarterly report now", "retry a quarterly report do it")
    # Word endings are absorbed: collect/collected and regions/region share a stem.
    endings = _two_requests(
        "Please collect quarterly figures northern regions",
        "Please collected quarterly figures northern region",
    )

    assert "repeat_request_loop" in _types(_analyze(filler, agents))
    assert "repeat_request_loop" in _types(_analyze(endings, agents))


def test_a_resolved_cluster_does_not_hide_an_unresolved_one_and_the_largest_is_reported() -> None:
    resolved = "Please collect the quarterly sales figures"
    small = "Schedule the offsite venue tour for October"
    large = "Translate the onboarding emails into German"
    contents = [
        resolved,
        small,
        large,
        resolved,
        small,
        large,
        f"{resolved} (completed)",
        large,
        resolved,
    ]
    messages = [
        Message("planner", "collector", text, float(i), True) for i, text in enumerate(contents)
    ]

    loops = _issues(_analyze(messages, ["planner", "collector"]), "repeat_request_loop")

    # The largest cluster (4 collection requests) shows progress, so it is skipped;
    # the 3-message translation cluster is the largest stuck one.
    assert len(loops) == 1
    assert "'planner' sent 3 near-duplicate messages" in loops[0].message


@pytest.mark.parametrize(
    ("length", "is_loop"),
    [(9, False), (10, True), (300, True), (301, False)],
)
def test_repeat_loop_payload_length_bounds_are_inclusive(length: int, is_loop: bool) -> None:
    content = ("retry " * 100)[:length]
    if content.endswith(" "):
        content = content[:-1] + "x"
    assert len(content) == length

    result = _analyze(_repeat("planner", "collector", content, count=2), ["planner", "collector"])

    assert ("repeat_request_loop" in _types(result)) is is_loop


def test_unrelated_messages_between_a_pair_are_not_a_loop() -> None:
    messages = [
        Message("planner", "collector", "Please collect the quarterly sales figures", 0.0, True),
        Message("planner", "collector", "Schedule the offsite venue tour for October", 1.0, True),
        Message(
            "collector", "planner", "Understood, translating onboarding emails first", 2.0, True
        ),
    ]

    assert "repeat_request_loop" not in _types(_analyze(messages, ["planner", "collector"]))


def test_repeats_split_across_different_pairs_are_not_a_pair_loop() -> None:
    messages = [
        Message("planner", "collector", REQUEST, 0.0, True),
        Message("reviewer", "publisher", REQUEST, 1.0, True),
    ]

    assert "repeat_request_loop" not in _types(
        _analyze(messages, ["planner", "collector", "reviewer", "publisher"])
    )


@pytest.mark.parametrize(
    "progress_word",
    ["completed", "created", "delivered", "resolved", "applied", "ok"],
)
def test_progress_marker_in_the_cluster_means_the_loop_is_resolving(progress_word: str) -> None:
    messages = [
        Message("planner", "collector", REQUEST, 0.0, True),
        Message("planner", "collector", f"{REQUEST} ({progress_word})", 1.0, True),
        Message("planner", "collector", REQUEST, 2.0, True),
    ]

    assert "repeat_request_loop" not in _types(_analyze(messages, ["planner", "collector"]))


def test_single_message_and_empty_traces_are_never_loops() -> None:
    assert "repeat_request_loop" not in _types(_analyze([], ["planner", "collector"]))
    assert "repeat_request_loop" not in _types(
        _analyze([Message("planner", "collector", REQUEST, 0.0, True)], ["planner", "collector"])
    )


@pytest.mark.parametrize(
    ("sender", "receiver"),
    [
        ("web_tool", "planner"),
        ("planner", "code_executor"),
        ("user", "planner"),
        ("planner", "System"),
        ("planner", "unknown"),
        ("orchestrator", "group"),
        ("planner", "all"),
        ("Broadcast", "planner"),
        ("planner", "planner"),
    ],
)
def test_repeat_loop_skips_tool_protocol_broadcast_and_self_edges(
    sender: str, receiver: str
) -> None:
    result = _analyze(_repeat(sender, receiver, count=4), ["planner", "collector", "web_tool"])

    assert "repeat_request_loop" not in _types(result)


def test_repeat_loop_ignores_very_short_and_very_long_payloads() -> None:
    short = _repeat("planner", "collector", "retry it", count=4)
    long_payload = "Collect the quarterly sales figures and summarise regional variance. " * 6
    assert len(long_payload) > 300
    long_messages = _repeat("planner", "collector", long_payload, count=4)

    assert "repeat_request_loop" not in _types(_analyze(short, ["planner", "collector"]))
    assert "repeat_request_loop" not in _types(_analyze(long_messages, ["planner", "collector"]))


# ---------------------------------------------------------------------------
# Group-bus stuck loops
# ---------------------------------------------------------------------------

STUCK_BASE = "Retrying the search for the latest pricing table entries again before we continue"


def _bus(
    contents: list[str], senders: list[str] | None = None, channel: str = "group"
) -> list[Message]:
    senders = senders or ["orchestrator"]
    return [
        Message(senders[i % len(senders)], channel, text, float(i))
        for i, text in enumerate(contents)
    ]


def test_single_sender_recycling_the_same_broadcast_is_a_group_bus_loop() -> None:
    messages = _bus([STUCK_BASE] * 5)

    result = _analyze(messages, ["orchestrator", "searcher", "writer"])

    loops = _issues(result, "group_bus_loop")
    assert len(loops) == 1
    assert loops[0].severity == "high"
    assert loops[0].agents_involved == ["orchestrator"]
    assert "'orchestrator' broadcast 5 near-duplicate messages to 'group'" in loops[0].message
    assert "bus novelty 0%" in loops[0].message
    assert not result.healthy


def test_multi_sender_paraphrased_bus_loop_is_caught() -> None:
    messages = _bus(
        [
            "Retrying the search for the latest pricing table entries again",
            "Retrying the search for latest pricing table entries again please",
            "Retrying search for the latest pricing table entries again now",
            "Retrying the search for the latest pricing table entries again today",
        ],
        senders=["planner", "critic"],
        channel="all",
    )

    result = _analyze(messages, ["planner", "critic", "writer"])

    loops = _issues(result, "group_bus_loop")
    assert len(loops) == 1
    assert loops[0].agents_involved == ["critic", "planner"]
    assert (
        "Group-bus stuck loop: 2 agents broadcast 4 near-duplicate messages to 'all'"
        in loops[0].message
    )


def test_group_bus_loop_covers_what_the_pairwise_loop_detector_skips() -> None:
    result = _analyze(_bus([STUCK_BASE] * 5), ["orchestrator", "searcher", "writer"])

    # Broadcast edges are skipped by the bilateral detector; the bus detector owns them.
    assert "repeat_request_loop" not in _types(result)
    assert "group_bus_loop" in _types(result)


def test_group_bus_that_keeps_introducing_new_content_is_benign() -> None:
    fresh = [
        "Search returned three vendor contracts mentioning indemnification caps",
        "Extracting renewal dates from the second contract shows a march deadline",
        "Compliance folder holds an outdated privacy addendum requiring redlines",
    ]
    messages = _bus([STUCK_BASE, fresh[0], STUCK_BASE, fresh[1], STUCK_BASE, fresh[2]])

    result = _analyze(messages, ["orchestrator", "searcher", "writer"])

    # Three near-duplicates exist, but the bus is advancing, so no loop.
    assert "group_bus_loop" not in _types(result)
    # Recycling the same content instead of advancing is what makes it a loop.
    stuck = _analyze(_bus([STUCK_BASE] * 6), ["orchestrator", "searcher", "writer"])
    assert "group_bus_loop" in _types(stuck)


_BUS_WORDS = [
    "pricing",
    "vendor",
    "contract",
    "renewal",
    "deadline",
    "compliance",
    "addendum",
    "privacy",
    "indemnity",
    "warranty",
    "liability",
    "termination",
]
_BUS_AGENTS = ["orchestrator", "searcher", "writer"]


def _bus_words(start: int, stop: int) -> str:
    return " ".join(_BUS_WORDS[start:stop])


def test_group_bus_needs_three_near_duplicates_of_the_same_message() -> None:
    # The kickoff seeds the whole vocabulary, so nothing later reads as new content
    # (novelty is zero throughout); only the near-duplicate count differs. Each half
    # shares 6 of the kickoff's 12 stems (0.50), which is not a near-duplicate of it.
    kickoff = _bus_words(0, 12)
    first_half = _bus_words(0, 6)
    second_half = _bus_words(6, 12)

    twice = _bus([kickoff, first_half, first_half, second_half])
    thrice = _bus([kickoff, first_half, first_half, first_half, second_half])

    assert "group_bus_loop" not in _types(_analyze(twice, _BUS_AGENTS))
    loops = _issues(_analyze(thrice, _BUS_AGENTS), "group_bus_loop")
    assert len(loops) == 1
    assert "broadcast 3 near-duplicate messages" in loops[0].message
    assert "bus novelty 0%" in loops[0].message


def test_group_bus_clusters_paraphrases_that_swap_one_word() -> None:
    base = _bus_words(0, 6)
    # Each variant swaps one already-seen word for another: 5 of 7 stems shared (0.71).
    variants = [f"{_bus_words(0, 5)} {_BUS_WORDS[extra]}" for extra in (6, 7, 8)]

    messages = _bus([_bus_words(0, 12), base, *variants])

    loops = _issues(_analyze(messages, _BUS_AGENTS), "group_bus_loop")
    assert len(loops) == 1
    assert "broadcast 4 near-duplicate messages" in loops[0].message


def test_group_bus_absorbs_word_endings_when_clustering() -> None:
    forms = [
        ("collecting", "searching", "compiling", "validating", "reviewing"),
        ("collected", "searched", "compiled", "validated", "reviewed"),
        ("collects", "searches", "compiles", "validates", "reviews"),
    ]
    contents = [" ".join([*forms[i % 3], "pricing table entries again"]) for i in range(5)]

    loops = _issues(_analyze(_bus(contents), _BUS_AGENTS), "group_bus_loop")

    assert len(loops) == 1
    assert "broadcast 5 near-duplicate messages" in loops[0].message


def test_group_bus_progress_marker_means_the_loop_is_resolving() -> None:
    contents = [STUCK_BASE] * 4 + [f"{STUCK_BASE} and the lookup has completed"]

    assert "group_bus_loop" not in _types(
        _analyze(_bus(contents), ["orchestrator", "searcher", "writer"])
    )


def test_group_bus_needs_a_real_conversation_of_four_messages() -> None:
    result = _analyze(_bus([STUCK_BASE] * 3), ["orchestrator", "searcher", "writer"])

    assert "group_bus_loop" not in _types(result)


def test_group_bus_ignores_short_protocol_pings() -> None:
    # Six distinct stems, but well under 40 characters.
    ping = "next step plan work task item"
    assert len(ping) < 40

    result = _analyze(_bus([ping] * 6), ["orchestrator", "coder", "writer"])

    assert "group_bus_loop" not in _types(result)


def test_group_bus_ignores_low_information_messages() -> None:
    # Over 40 characters but only four distinct 4+ character stems.
    filler = "well this that then this that then this that then this that then"
    assert len(filler) >= 40

    assert "group_bus_loop" not in _types(
        _analyze(_bus([filler] * 6), ["orchestrator", "coder", "writer"])
    )


def test_group_bus_ignores_broadcasts_from_tool_agents() -> None:
    messages = _bus([STUCK_BASE] * 5, senders=["search_tool"])

    assert "group_bus_loop" not in _types(_analyze(messages, ["search_tool", "planner", "writer"]))


def test_group_bus_ignores_messages_addressed_to_named_agents() -> None:
    messages = _bus([STUCK_BASE] * 5, channel="reviewer")

    result = _analyze(messages, ["orchestrator", "reviewer", "writer"])

    assert "group_bus_loop" not in _types(result)


def test_group_bus_tolerates_unparseable_timestamps() -> None:
    messages = _bus([STUCK_BASE] * 5)
    for message in messages:
        message.timestamp = "not-a-time"  # type: ignore[assignment]

    issues = CoordinationAnalyzer()._detect_group_bus_loop(messages)

    assert [i.issue_type for i in issues] == ["group_bus_loop"]


def test_group_bus_novelty_is_measured_in_time_order_not_list_order() -> None:
    extras = (
        "vendor contracts renewal deadlines compliance addenda privacy redlines "
        "indemnification warranty liability termination"
    )
    broad_kickoff = f"{STUCK_BASE} while also tracking {extras}"

    def build(kickoff_ts: float) -> list[Message]:
        # The list order is identical in both traces; only the kickoff's time differs.
        return [
            Message("orchestrator", "group", STUCK_BASE, 1.0),
            Message("orchestrator", "group", STUCK_BASE, 2.0),
            Message("orchestrator", "group", STUCK_BASE, 3.0),
            Message("orchestrator", "group", broad_kickoff, kickoff_ts),
        ]

    agents = ["orchestrator", "searcher", "writer"]
    # Kickoff first: every later broadcast repeats it, so the bus is stuck.
    assert "group_bus_loop" in _types(_analyze(build(0.0), agents))
    # Kickoff last: it introduces plenty of new vocabulary, so the bus advanced.
    assert "group_bus_loop" not in _types(_analyze(build(9.0), agents))


# ---------------------------------------------------------------------------
# Hierarchical topology and telemetry abstention
# ---------------------------------------------------------------------------

_TOPICS = [
    "invoice reconciliation for the northern region",
    "schema migration plan for the billing database",
    "customer churn cohort analysis for spring",
    "latency regression triage on the checkout service",
    "warehouse inventory forecast for december",
    "security audit of the payment gateway",
    "translation review of onboarding emails",
    "kubernetes upgrade rehearsal in staging",
    "marketing attribution model comparison",
    "fraud rule tuning for card transactions",
    "support ticket taxonomy cleanup",
    "quarterly tax provisioning worksheet",
    "mobile crash report grouping",
    "vendor contract clause extraction",
    "data retention policy mapping",
    "search ranking experiment readout",
    "sla breach root cause writeup",
    "hiring pipeline funnel dashboard",
]
_PEER_CHATTER = [
    ("s1", "s2"),
    ("s2", "s3"),
    ("s3", "s1"),
    ("s1", "s3"),
    ("s2", "s1"),
    ("s3", "s2"),
]


def _flow(pairs: list[tuple[str, str]]) -> list[Message]:
    return [
        Message(sender, receiver, f"{_TOPICS[i % len(_TOPICS)]} step {i}", float(i), True)
        for i, (sender, receiver) in enumerate(pairs)
    ]


def _hub_fanning_out_to_three() -> list[Message]:
    # hub sends 12 of 18 messages (67%) to three distinct specialists who
    # also chat among themselves, so this is neither a pipeline nor a peer team.
    pairs = [("hub", s) for _ in range(4) for s in ("s1", "s2", "s3")] + _PEER_CHATTER
    return _flow(pairs)


def _hub_talking_to_two() -> list[Message]:
    # Same 67% volume dominance, but the hub only addresses two recipients.
    pairs = [("hub", s) for _ in range(6) for s in ("s1", "s2")] + _PEER_CHATTER
    return _flow(pairs)


AGENTS = ["hub", "s1", "s2", "s3"]


def test_hub_fanning_out_to_three_specialists_is_hierarchical_not_hoarding() -> None:
    messages = _hub_fanning_out_to_three()
    analyzer = CoordinationAnalyzer()

    assert analyzer._is_hierarchical_topology(messages, AGENTS)
    assert not analyzer._is_pipeline_topology(messages, AGENTS)
    # The raw volume check would call this lead hoarding...
    assert [i.issue_type for i in analyzer._detect_lead_hoarding(messages, AGENTS)] == [
        "lead_hoarding"
    ]
    # ...but the full analysis treats the hub as a legitimate coordinator.
    assert "lead_hoarding" not in _types(_analyze(messages, AGENTS))


def test_same_volume_dominance_toward_two_recipients_is_still_lead_hoarding() -> None:
    messages = _hub_talking_to_two()

    assert not CoordinationAnalyzer()._is_hierarchical_topology(messages, AGENTS)
    result = _analyze(messages, AGENTS)

    hoarding = _issues(result, "lead_hoarding")
    assert len(hoarding) == 1
    assert hoarding[0].agents_involved == ["hub"]
    assert "12/18" in hoarding[0].message


def test_hierarchical_suppression_also_covers_back_and_forth_and_stale_handoffs() -> None:
    # 16 hub<->s1 exchanges (well over the 8-message back-and-forth bar) while
    # the hub also fans out to s2 and s3.
    pairs = [("hub", "s1"), ("s1", "hub")] * 8 + [("hub", "s2"), ("hub", "s3")] * 3 + _PEER_CHATTER
    messages = _flow(pairs)
    # The first three hub<->s1 handoffs repeat the same content: a stale loop.
    for message in messages[:3]:
        message.content = "please rework the invoice reconciliation totals again"
    analyzer = CoordinationAnalyzer()
    assert analyzer._is_hierarchical_topology(messages, AGENTS)
    assert not analyzer._is_pipeline_topology(messages, AGENTS)
    assert analyzer._detect_excessive_back_forth(messages)
    assert [i.issue_type for i in analyzer._detect_stale_handoff_loop(messages)] == [
        "stale_handoff_loop"
    ]

    result = _analyze(messages, AGENTS)

    # The raw detectors would fire, but a hub legitimately iterates with a specialist.
    assert "excessive_back_forth" not in _types(result)
    assert "stale_handoff_loop" not in _types(result)


def test_back_and_forth_is_still_flagged_outside_hierarchical_topologies() -> None:
    # Two dominant senders, none reaching 50% of traffic: not a hub.
    pairs = (
        [("hub", "s1"), ("s1", "hub")] * 8
        + [("s2", "s3"), ("s3", "s2")] * 4
        + [("hub", "s2"), ("hub", "s3"), ("s1", "s2")]
    )
    messages = _flow(pairs)
    assert not CoordinationAnalyzer()._is_hierarchical_topology(messages, AGENTS)

    assert "excessive_back_forth" in _types(_analyze(messages, AGENTS))


def test_back_and_forth_needs_more_than_eight_messages_and_skips_tool_executors() -> None:
    analyzer = CoordinationAnalyzer()

    def chat(first: str, second: str, count: int) -> list[Message]:
        return _flow([(first, second) if i % 2 == 0 else (second, first) for i in range(count)])

    assert analyzer._detect_excessive_back_forth(chat("planner", "coder", 8)) == []
    nine = analyzer._detect_excessive_back_forth(chat("planner", "coder", 9))
    assert [i.issue_type for i in nine] == ["excessive_back_forth"]
    assert nine[0].agents_involved == ["coder", "planner"]
    assert "exchanged 9 messages" in nine[0].message
    # Tool round-trips are expected to be chatty, however many there are.
    assert analyzer._detect_excessive_back_forth(chat("planner", "tool_executor", 20)) == []


@pytest.mark.parametrize(
    ("messages", "agent_ids", "expected"),
    [
        # Fewer than three agents can never be a hub-and-spoke system.
        (_flow([("hub", "s1"), ("hub", "s2"), ("hub", "s3")] * 3), ["hub", "s1"], False),
        # No traffic, or only self-messages, gives no evidence of a hub.
        ([], ["hub", "s1", "s2"], False),
        (
            [Message("hub", "hub", "Noting my own progress on the plan", 0.0)],
            ["hub", "s1", "s2"],
            False,
        ),
        # Exactly half of outbound traffic to three recipients is a hub.
        (
            _flow(
                [
                    ("hub", "s1"),
                    ("hub", "s2"),
                    ("hub", "s3"),
                    ("s1", "s2"),
                    ("s2", "s3"),
                    ("s3", "s1"),
                ]
            ),
            AGENTS,
            True,
        ),
        # Below half of outbound traffic is not.
        (
            _flow(
                [("hub", "s1"), ("hub", "s2"), ("hub", "s3")]
                + [("s1", "s2"), ("s2", "s3"), ("s3", "s1"), ("s1", "s3")]
            ),
            AGENTS,
            False,
        ),
        # A dominant sender with only two recipients is not a hub.
        (_flow([("hub", "s1"), ("hub", "s2")] * 5 + [("s3", "s1")]), AGENTS, False),
        # Messages a sender addresses to itself are not outbound traffic: the hub
        # sends 3 of 7 real messages (43%), and self-notes must not lift it to half.
        (
            _flow(
                [
                    ("hub", "s1"),
                    ("hub", "s2"),
                    ("hub", "s3"),
                    ("s1", "s2"),
                    ("s2", "s3"),
                    ("s3", "s1"),
                    ("s1", "s3"),
                ]
            )
            + [
                Message("hub", "hub", "Noting my own progress on the plan", 8.0),
                Message("hub", "hub", "Noting more of my own progress", 9.0),
            ],
            AGENTS,
            False,
        ),
    ],
    ids=[
        "two-agents",
        "no-messages",
        "self-only",
        "exactly-half",
        "below-half",
        "two-recipients",
        "self-notes-ignored",
    ],
)
def test_hierarchical_topology_rules(
    messages: list[Message], agent_ids: list[str], expected: bool
) -> None:
    assert CoordinationAnalyzer()._is_hierarchical_topology(messages, agent_ids) is expected


def test_unavailable_message_telemetry_suppresses_volume_checks() -> None:
    messages = _hub_talking_to_two()

    observed = _analyze(messages, AGENTS)
    edits_only = _analyze(messages, AGENTS, message_telemetry_present=False)

    assert "lead_hoarding" in _types(observed)
    assert "lead_hoarding" not in _types(edits_only)


def test_empty_message_stream_reads_as_silence_unless_telemetry_is_marked_unavailable() -> None:
    changes = [
        _change("agent_a", "f.py", _edit(4, 2, added=HASH_A)),
        _change("agent_b", "f.py", _edit(5, 2, added=HASH_C)),
    ]

    observed_empty = _analyze([], ["agent_a", "agent_b"], resource_changes=changes)
    edits_only = _analyze(
        [], ["agent_a", "agent_b"], resource_changes=changes, message_telemetry_present=False
    )

    # An observed-but-empty stream keeps the legacy reading: everyone was silent.
    assert "silent_agent" in _types(observed_empty)
    # A structural-only source reports just the file conflict.
    assert _types(edits_only) == ["conflicting_resource_change"]
    assert _issues(edits_only, "conflicting_resource_change")[0].agents_involved == [
        "agent_a",
        "agent_b",
    ]


# ---------------------------------------------------------------------------
# Lead hoarding: only real collaborating agents count
# ---------------------------------------------------------------------------


def _dominated_chat(dominant: str, others: list[str]) -> list[Message]:
    pairs = [(dominant, others[i % len(others)]) for i in range(8)] + [(others[0], dominant)] * 2
    return _flow(pairs)


@pytest.mark.parametrize(
    "agent_ids",
    [
        ["assistant", "user"],
        ["assistant", "user", "system"],
        ["assistant", "human", "team", "broadcast"],
    ],
)
def test_assistant_user_chat_is_not_lead_hoarding(agent_ids: list[str]) -> None:
    messages = _dominated_chat("assistant", ["user"])

    assert "lead_hoarding" not in _types(_analyze(messages, agent_ids))


def test_three_real_agents_with_one_dominant_sender_is_lead_hoarding() -> None:
    analyzer = CoordinationAnalyzer()
    messages = _dominated_chat("planner", ["coder", "reviewer"])

    issues = analyzer._detect_lead_hoarding(messages, ["planner", "coder", "reviewer"])

    assert len(issues) == 1
    assert issues[0].agents_involved == ["planner"]
    assert "8/10" in issues[0].message
    assert issues[0].severity == "medium"


@pytest.mark.parametrize(
    "pseudo",
    [
        "user",
        "human",
        "system",
        "lead",
        "group",
        "all",
        "everyone",
        "everybody",
        "team",
        "channel",
        "broadcast",
    ],
)
def test_each_pseudo_role_is_excluded_from_the_team_count(pseudo: str) -> None:
    analyzer = CoordinationAnalyzer()
    messages = _dominated_chat("planner", ["coder"])

    # planner + coder + a pseudo-role is a two-agent team, in any letter case...
    assert analyzer._detect_lead_hoarding(messages, ["planner", "coder", pseudo]) == []
    assert analyzer._detect_lead_hoarding(messages, ["planner", "coder", pseudo.upper()]) == []
    # ...while a genuine third agent makes the same volume lead hoarding.
    real_team = analyzer._detect_lead_hoarding(messages, ["planner", "coder", "reviewer"])
    assert [i.agents_involved for i in real_team] == [["planner"]]


@pytest.mark.parametrize(
    ("planner_sent", "others_sent", "fires"),
    [
        # Under five messages there is too little traffic to call anything hoarding.
        (4, 0, False),
        (5, 0, True),
        # The lead must send MORE than 60% of the messages, not exactly 60%.
        (6, 4, False),
        (7, 3, True),
    ],
)
def test_lead_hoarding_volume_and_share_thresholds(
    planner_sent: int, others_sent: int, fires: bool
) -> None:
    messages = _flow([("planner", "coder")] * planner_sent + [("coder", "reviewer")] * others_sent)

    issues = CoordinationAnalyzer()._detect_lead_hoarding(
        messages, ["planner", "coder", "reviewer"]
    )

    assert bool(issues) is fires


def test_pseudo_agents_do_not_pad_out_a_team_to_three() -> None:
    analyzer = CoordinationAnalyzer()
    messages = _dominated_chat("planner", ["coder", "reviewer"])

    # Two real agents plus a human, the system and a broadcast bus are still two.
    assert (
        analyzer._detect_lead_hoarding(
            messages, ["planner", "coder", "User", "SYSTEM", "lead", "group"]
        )
        == []
    )


# ---------------------------------------------------------------------------
# Abstention band and ignored-message gating
# ---------------------------------------------------------------------------


def test_medium_only_traces_abstain_below_the_serving_threshold() -> None:
    result = _analyze(_hub_talking_to_two(), AGENTS)

    severities = {issue.severity for issue in result.issues}
    assert result.detected
    assert severities <= {"low", "medium"}
    assert result.confidence <= 0.45


def test_a_high_severity_issue_lifts_confidence_above_the_abstention_cap() -> None:
    result = _analyze(_repeat("planner", "collector", count=3), ["planner", "collector"])

    assert "high" in {issue.severity for issue in result.issues}
    assert result.confidence > 0.45


def test_ignored_message_needs_reliable_ack_data() -> None:
    silent_recipient = Message(
        "planner", "reviewer", "Please review the draft contract", 0.0, False
    )

    without_ack_data = _analyze([silent_recipient], ["planner", "reviewer"])
    with_ack_data = _analyze(
        [
            Message("planner", "publisher", "Publishing schedule confirmed for Friday", 0.0, True),
            Message("planner", "reviewer", "Please review the draft contract", 1.0, False),
        ],
        ["planner", "reviewer", "publisher"],
    )

    assert "ignored_message" not in _types(without_ack_data)
    ignored = _issues(with_ack_data, "ignored_message")
    assert len(ignored) == 1
    assert ignored[0].severity == "medium"
    assert ignored[0].agents_involved == ["planner", "reviewer"]


def test_ignored_message_is_reported_once_per_pair_and_only_for_silent_recipients() -> None:
    acked = Message("planner", "publisher", "Publishing schedule confirmed for Friday", 0.0, True)
    agents = ["planner", "reviewer", "publisher", "system"]

    def flagged(*extra: Message) -> list[list[str]]:
        result = _analyze([acked, *extra], agents)
        return [i.agents_involved for i in _issues(result, "ignored_message")]

    # Two unacknowledged requests to the same silent reviewer are one finding.
    assert flagged(
        Message("planner", "reviewer", "Please review the draft contract", 1.0, False),
        Message("planner", "reviewer", "Please review the draft contract today", 2.0, False),
    ) == [["planner", "reviewer"]]
    # A recipient who spoke afterwards did not ignore the message.
    assert "planner" not in {
        pair[0]
        for pair in flagged(
            Message("planner", "reviewer", "Please review the draft contract", 1.0, False),
            Message("reviewer", "publisher", "Contract review finished, sending notes", 2.0, True),
        )
    }
    # Receive-only sinks and messages to oneself never need a reply.
    assert (
        flagged(
            Message("planner", "system", "Logging the release checklist state", 1.0, False),
            Message("planner", "planner", "Noting the release checklist state", 2.0, False),
        )
        == []
    )
