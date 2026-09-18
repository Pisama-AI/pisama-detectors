"""Behavioral contracts for the Dify iteration-escape detector.

Covers the reconciled behavior: the config/DSL-shape check
(``detect_workflow_config``), its ``_is_positive_bound`` helper, and the
evidence contract of the run-shape check (real node IDs in evidence, no fake
turn indices). The real detector class runs against hand-written payloads.
"""

from typing import Any, Dict, List, Optional

import pytest

import pisama_detectors as pd
from pisama_detectors.detection.dify import DifyIterationEscapeDetector
from pisama_detectors.detection.turn_aware._base import TurnAwareSeverity


def _config_node(
    node_id: Optional[str],
    node_type: str = "iteration",
    title: Optional[str] = None,
    **data_fields: Any,
) -> Dict[str, Any]:
    data: Dict[str, Any] = {"type": node_type, **data_fields}
    if title is not None:
        data["title"] = title
    node: Dict[str, Any] = {"data": data}
    if node_id is not None:
        node["id"] = node_id
    return node


def _run_children(
    parent_id: str, count: int, extra: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Child execution records with contiguous indices and distinct outputs."""
    children = []
    for i in range(count):
        child: Dict[str, Any] = {
            "node_id": f"{parent_id}-step-{i}",
            "parent_node_id": parent_id,
            "iteration_index": i,
            "outputs": {"value": f"result-{i}"},
        }
        if extra:
            child.update(extra)
        children.append(child)
    return children


def _issue_types(result: Any) -> List[str]:
    return [issue["type"] for issue in result.evidence["issues"]]


# ---------------------------------------------------------------------------
# _is_positive_bound
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [1, 25, 100, "5", " 7 ", "1000"])
def test_is_positive_bound_accepts_finite_positive_caps(value: Any) -> None:
    assert DifyIterationEscapeDetector._is_positive_bound(value) is True


@pytest.mark.parametrize(
    "value",
    [0, -1, -50, "0", "-3", "", "   ", "unlimited", "inf", "nan", None, True, False, [], {}],
)
def test_is_positive_bound_rejects_missing_or_non_finite_caps(value: Any) -> None:
    # True/False are ints in Python; they must not pass as "a cap of 1".
    assert DifyIterationEscapeDetector._is_positive_bound(value) is False


# ---------------------------------------------------------------------------
# detect_workflow_config: unbounded vs bounded
# ---------------------------------------------------------------------------


def test_config_flags_iteration_node_without_max_iterations() -> None:
    result = DifyIterationEscapeDetector().detect_workflow_config(
        {"nodes": [_config_node("loop-1", title="Process items")]}
    )

    assert result.detected is True
    assert result.failure_mode == "F11"
    assert result.severity == TurnAwareSeverity.MODERATE
    assert result.confidence == pytest.approx(0.8)
    assert result.affected_turns == []
    assert result.detector_name == "DifyIterationEscapeDetector"
    assert result.evidence["unbounded_nodes"] == ["loop-1"]
    assert result.evidence["total_iteration_nodes"] == 1
    assert result.evidence["issues"] == [
        {"type": "no_iteration_bound", "node_id": "loop-1", "potentially_infinite": True}
    ]
    assert "1 of 1" in result.explanation
    assert result.suggested_fix is not None
    assert "max_iterations" in result.suggested_fix


def test_config_is_quiet_when_every_iteration_node_has_a_finite_bound() -> None:
    result = DifyIterationEscapeDetector().detect_workflow_config(
        {
            "nodes": [
                _config_node("loop-1", "iteration", max_iterations=10),
                _config_node("loop-2", "loop", max_iterations="25"),
            ]
        }
    )

    assert result.detected is False
    assert result.severity == TurnAwareSeverity.NONE
    assert result.confidence == 0.0
    assert result.failure_mode is None
    assert "finite max_iterations" in result.explanation


@pytest.mark.parametrize("node_type", ["iteration", "loop"])
def test_config_checks_both_iteration_and_loop_node_types(node_type: str) -> None:
    detector = DifyIterationEscapeDetector()

    unbounded = detector.detect_workflow_config({"nodes": [_config_node("n1", node_type)]})
    bounded = detector.detect_workflow_config(
        {"nodes": [_config_node("n1", node_type, max_iterations=5)]}
    )

    assert unbounded.detected is True
    assert bounded.detected is False


@pytest.mark.parametrize(
    "bogus_bound",
    [0, -5, "0", "abc", "", "unlimited", None, True, False],
)
def test_config_treats_non_finite_or_fake_bounds_as_unbounded(bogus_bound: Any) -> None:
    detector = DifyIterationEscapeDetector()
    node = _config_node("loop-1", "iteration", max_iterations=bogus_bound)

    result = detector.detect_workflow_config({"nodes": [node]})

    assert result.detected is True
    assert result.evidence["unbounded_nodes"] == ["loop-1"]


def test_config_accepts_max_attempts_as_the_bound_when_max_iterations_absent() -> None:
    detector = DifyIterationEscapeDetector()

    with_attempts = detector.detect_workflow_config(
        {"nodes": [_config_node("retry", "loop", max_attempts=3)]}
    )
    bad_attempts = detector.detect_workflow_config(
        {"nodes": [_config_node("retry", "loop", max_attempts=0)]}
    )

    assert with_attempts.detected is False
    assert bad_attempts.detected is True
    assert bad_attempts.evidence["unbounded_nodes"] == ["retry"]


# ---------------------------------------------------------------------------
# detect_workflow_config: graph shape, node selection, identifiers
# ---------------------------------------------------------------------------


def test_config_reads_nodes_from_nested_graph_shape() -> None:
    detector = DifyIterationEscapeDetector()

    unbounded = detector.detect_workflow_config(
        {"graph": {"nodes": [_config_node("loop-1")]}}
    )
    bounded = detector.detect_workflow_config(
        {"graph": {"nodes": [_config_node("loop-1", max_iterations=4)]}}
    )

    assert unbounded.detected is True
    assert unbounded.evidence["unbounded_nodes"] == ["loop-1"]
    assert bounded.detected is False


@pytest.mark.parametrize("config", [{}, {"nodes": []}, {"graph": {}}, {"graph": {"nodes": []}}])
def test_config_without_nodes_is_not_a_detection(config: Dict[str, Any]) -> None:
    result = DifyIterationEscapeDetector().detect_workflow_config(config)

    assert result.detected is False
    assert result.explanation == "No iteration/loop nodes in config"


def test_config_ignores_non_iteration_nodes_even_without_bounds() -> None:
    result = DifyIterationEscapeDetector().detect_workflow_config(
        {
            "nodes": [
                _config_node("llm-1", "llm"),
                _config_node("code-1", "code"),
                {"id": "no-data"},
                {"id": "null-data", "data": None},
            ]
        }
    )

    assert result.detected is False
    assert result.explanation == "No iteration/loop nodes in config"


def test_config_tolerates_nodes_with_null_data_alongside_real_iteration_nodes() -> None:
    result = DifyIterationEscapeDetector().detect_workflow_config(
        {"nodes": [{"id": "broken", "data": None}, _config_node("loop-1")]}
    )

    assert result.detected is True
    assert result.evidence["total_iteration_nodes"] == 1
    assert result.evidence["unbounded_nodes"] == ["loop-1"]


def test_config_reports_only_unbounded_nodes_and_counts_all_iteration_nodes() -> None:
    result = DifyIterationEscapeDetector().detect_workflow_config(
        {
            "nodes": [
                _config_node("ok-loop", "loop", max_iterations=10),
                _config_node("bad-iter", "iteration"),
                _config_node("llm-1", "llm"),
                _config_node("bad-loop", "loop", max_iterations=0),
            ]
        }
    )

    assert result.detected is True
    assert result.evidence["unbounded_nodes"] == ["bad-iter", "bad-loop"]
    assert result.evidence["total_iteration_nodes"] == 3
    assert "2 of 3" in result.explanation
    assert [i["node_id"] for i in result.evidence["issues"]] == ["bad-iter", "bad-loop"]
    assert all(i["potentially_infinite"] is True for i in result.evidence["issues"])


def test_config_falls_back_to_title_then_empty_string_for_unbounded_node_identity() -> None:
    result = DifyIterationEscapeDetector().detect_workflow_config(
        {
            "nodes": [
                _config_node("with-id", title="Has both"),
                _config_node(None, title="Title only"),
                _config_node(None),
            ]
        }
    )

    assert result.evidence["unbounded_nodes"] == ["with-id", "Title only", ""]


def test_config_goes_quiet_only_after_a_real_bound_is_applied() -> None:
    """A circuit-breaker fix must set a genuine cap; a fake one keeps it firing."""
    detector = DifyIterationEscapeDetector()
    node = _config_node("loop-1", "iteration", title="Crawl")
    config = {"nodes": [node]}

    assert detector.detect_workflow_config(config).detected is True

    node["data"]["max_iterations"] = "unlimited"
    assert detector.detect_workflow_config(config).detected is True

    node["data"]["max_iterations"] = 0
    assert detector.detect_workflow_config(config).detected is True

    node["data"]["max_iterations"] = 20
    assert detector.detect_workflow_config(config).detected is False


# ---------------------------------------------------------------------------
# Run-shape vs config-shape are distinct counterparts
# ---------------------------------------------------------------------------


def test_config_variant_does_not_read_run_shape_traces() -> None:
    run_trace = {
        "nodes": [
            {"node_id": "loop", "node_type": "iteration", "status": "failed"},
            *_run_children("loop", 3),
        ]
    }
    detector = DifyIterationEscapeDetector()

    assert detector.detect_workflow_run(run_trace).detected is True
    config_result = detector.detect_workflow_config(run_trace)
    assert config_result.detected is False
    assert config_result.explanation == "No iteration/loop nodes in config"


def test_run_variant_does_not_read_config_shape_graphs() -> None:
    result = DifyIterationEscapeDetector().detect_workflow_run(
        {"nodes": [_config_node("loop-1")]}
    )

    assert result.detected is False
    assert result.explanation == "No iteration/loop nodes found"


# ---------------------------------------------------------------------------
# detect(): delegation through conversation metadata
# ---------------------------------------------------------------------------


def test_detect_delegates_to_workflow_run_from_metadata() -> None:
    workflow_run = {
        "nodes": [
            {"node_id": "loop", "node_type": "iteration", "status": "failed"},
            *_run_children("loop", 2),
        ]
    }
    detector = DifyIterationEscapeDetector()

    via_detect = detector.detect([], {"workflow_run": workflow_run})
    direct = detector.detect_workflow_run(workflow_run)

    assert via_detect.detected is True
    assert via_detect.confidence == direct.confidence
    assert via_detect.severity == direct.severity
    assert _issue_types(via_detect) == _issue_types(direct)


@pytest.mark.parametrize("metadata", [None, {}, {"workflow_run": {}}, {"other": 1}])
def test_detect_without_workflow_run_is_not_a_detection(metadata: Optional[Dict[str, Any]]) -> None:
    result = DifyIterationEscapeDetector().detect([], metadata)

    assert result.detected is False
    assert result.explanation == "No workflow_run data provided"


# ---------------------------------------------------------------------------
# Run-shape evidence contract
# ---------------------------------------------------------------------------


def test_run_evidence_lists_real_node_ids_and_no_fake_turn_indices() -> None:
    nodes: List[Dict[str, Any]] = [
        {"node_id": "loop-b", "node_type": "iteration", "status": "failed", "title": "B"},
        *_run_children("loop-b", 2),
        {"node_id": "loop-a", "node_type": "loop", "inputs": {"max_iterations": 2}},
        *_run_children("loop-a", 4),
        {"node_id": "loop-ok", "node_type": "iteration", "status": "succeeded"},
        *_run_children("loop-ok", 2),
    ]

    result = DifyIterationEscapeDetector().detect_workflow_run({"nodes": nodes})

    assert result.detected is True
    # affected_turns used to be a count masquerading as indices; it must stay empty.
    assert result.affected_turns == []
    # IDs are unique and sorted; the healthy loop is not implicated.
    assert result.evidence["affected_node_ids"] == ["loop-a", "loop-b"]
    assert result.evidence["total_iteration_nodes"] == 3


def test_run_evidence_deduplicates_node_with_multiple_issues() -> None:
    # One node trips both iteration_failure and scope_leak_duplicate_outputs.
    children = _run_children("loop", 3)
    for child in children:
        child["outputs"] = {"value": "same"}
    nodes = [{"node_id": "loop", "node_type": "iteration", "status": "failed"}, *children]

    result = DifyIterationEscapeDetector().detect_workflow_run({"nodes": nodes})

    assert _issue_types(result) == ["iteration_failure", "scope_leak_duplicate_outputs"]
    assert result.evidence["affected_node_ids"] == ["loop"]
    # Structural finding: confidence grows 0.1 per issue from 0.6 (2 issues -> 0.8).
    assert result.confidence == pytest.approx(0.8)
    assert result.severity == TurnAwareSeverity.MODERATE


# ---------------------------------------------------------------------------
# Run-shape checks with paired benign inputs
# ---------------------------------------------------------------------------


def test_run_excessive_iterations_with_failure_is_severe_and_confidence_scales() -> None:
    nodes = [
        {
            "node_id": "loop",
            "node_type": "iteration",
            "status": "failed",
            "outputs": {"note": "stop"},
        },
        *_run_children("loop", 150),
    ]

    result = DifyIterationEscapeDetector().detect_workflow_run({"nodes": nodes})

    assert result.detected is True
    assert result.severity == TurnAwareSeverity.SEVERE
    assert result.confidence == pytest.approx(0.7 + 50 / 500)
    assert result.evidence["max_iteration_count"] == 150
    assert {"excessive_iterations", "iteration_failure"} <= set(_issue_types(result))


def test_run_excessive_iterations_confidence_is_capped() -> None:
    nodes = [
        {"node_id": "loop", "node_type": "iteration", "outputs": {"note": "stop"}},
        *_run_children("loop", 400),
    ]

    result = DifyIterationEscapeDetector().detect_workflow_run({"nodes": nodes})

    assert result.detected is True
    assert _issue_types(result) == ["excessive_iterations"]
    assert result.severity == TurnAwareSeverity.MODERATE
    assert result.confidence == pytest.approx(0.95)


def test_run_many_iterations_without_exit_signal_is_minor_but_exit_signal_silences_it() -> None:
    detector = DifyIterationEscapeDetector()
    silent_loop = {"node_id": "loop", "node_type": "iteration", "status": "succeeded"}
    guarded_loop = {**silent_loop, "outputs": {"stop_reason": "max_iterations reached"}}

    flagged = detector.detect_workflow_run({"nodes": [silent_loop, *_run_children("loop", 60)]})
    guarded = detector.detect_workflow_run({"nodes": [guarded_loop, *_run_children("loop", 60)]})

    assert flagged.detected is True
    assert _issue_types(flagged) == ["no_break_condition"]
    assert flagged.severity == TurnAwareSeverity.MINOR
    assert flagged.confidence == pytest.approx(0.6 + (60 - 50) / (100 - 50) * 0.15)
    assert guarded.detected is False
    assert guarded.explanation == "No iteration escape issues found"


def test_run_loop_overrun_flags_more_iterations_than_configured_cap() -> None:
    detector = DifyIterationEscapeDetector()

    over = detector.detect_workflow_run(
        {
            "nodes": [
                {"node_id": "loop", "node_type": "loop", "inputs": {"max_iterations": 3}},
                *_run_children("loop", 5),
            ]
        }
    )
    within = detector.detect_workflow_run(
        {
            "nodes": [
                {"node_id": "loop", "node_type": "loop", "inputs": {"max_iterations": 5}},
                *_run_children("loop", 5),
            ]
        }
    )
    via_attempts = detector.detect_workflow_run(
        {
            "nodes": [
                {"node_id": "loop", "node_type": "loop", "inputs": {"max_attempts": 2}},
                *_run_children("loop", 4),
            ]
        }
    )
    empty_items = detector.detect_workflow_run(
        {
            "nodes": [
                {
                    "node_id": "loop",
                    "node_type": "loop",
                    "inputs": {"items": [], "max_iterations": 2},
                },
                *_run_children("loop", 4),
            ]
        }
    )

    assert over.detected is True
    assert _issue_types(over) == ["loop_overrun"]
    assert over.evidence["issues"][0]["max_configured"] == 3
    assert over.evidence["issues"][0]["iteration_count"] == 5
    assert over.severity == TurnAwareSeverity.MODERATE
    assert over.confidence == pytest.approx(0.7)
    assert within.detected is False
    assert _issue_types(via_attempts) == ["loop_overrun"]
    assert via_attempts.evidence["issues"][0]["max_configured"] == 2
    # An empty items list is not an item bound; the configured cap still applies.
    assert _issue_types(empty_items) == ["loop_overrun"]


def test_run_iteration_overrun_flags_more_iterations_than_input_items() -> None:
    detector = DifyIterationEscapeDetector()
    parent = {"node_id": "loop", "node_type": "iteration", "inputs": {"items": ["a", "b"]}}

    over = detector.detect_workflow_run({"nodes": [parent, *_run_children("loop", 4)]})
    exact = detector.detect_workflow_run({"nodes": [parent, *_run_children("loop", 2)]})

    assert over.detected is True
    assert _issue_types(over) == ["iteration_overrun"]
    assert over.evidence["issues"][0]["input_count"] == 2
    assert over.evidence["issues"][0]["iteration_count"] == 4
    assert over.severity == TurnAwareSeverity.MODERATE
    assert over.confidence == pytest.approx(0.7)
    assert exact.detected is False


def test_run_index_gap_is_a_structural_moderate_finding() -> None:
    detector = DifyIterationEscapeDetector()
    gapped = _run_children("loop", 3)
    gapped[1]["iteration_index"] = 2
    gapped[2]["iteration_index"] = 3
    parent = {"node_id": "loop", "node_type": "iteration"}

    corrupted = detector.detect_workflow_run({"nodes": [parent, *gapped]})
    contiguous = detector.detect_workflow_run({"nodes": [parent, *_run_children("loop", 3)]})
    # Only gaps between observed indices count; a run that starts above 0 is not a gap.
    offset = _run_children("loop", 3)
    for child in offset:
        child["iteration_index"] += 1
    offset_run = detector.detect_workflow_run({"nodes": [parent, *offset]})

    assert corrupted.detected is True
    assert _issue_types(corrupted) == ["index_corruption"]
    assert corrupted.evidence["issues"][0]["missing_indices"] == [1]
    assert corrupted.evidence["issues"][0]["actual_indices"] == [0, 2, 3]
    # Iteration count is highest index + 1 (4), not the number of child records (3).
    assert corrupted.evidence["issues"][0]["iteration_count"] == 4
    assert corrupted.severity == TurnAwareSeverity.MODERATE
    assert corrupted.confidence == pytest.approx(0.7)
    assert contiguous.detected is False
    assert offset_run.detected is False


def test_run_child_referencing_parent_scope_is_minor_low_confidence() -> None:
    detector = DifyIterationEscapeDetector()
    parent = {"node_id": "loop", "node_type": "iteration"}
    leaky = [
        {
            "node_id": "step-0",
            "parent_node_id": "loop",
            "iteration_index": 0,
            "title": "Writer",
            "outputs": {"assign": "Parent.counter += 1"},
        }
    ]
    clean = _run_children("loop", 1)

    flagged = detector.detect_workflow_run({"nodes": [parent, *leaky]})
    benign = detector.detect_workflow_run({"nodes": [parent, *clean]})

    assert flagged.detected is True
    assert _issue_types(flagged) == ["parent_scope_modification"]
    ref = flagged.evidence["issues"][0]["children_with_parent_refs"][0]
    assert ref == {"child_node_id": "step-0", "child_title": "Writer", "keyword": "parent"}
    assert flagged.severity == TurnAwareSeverity.MINOR
    assert flagged.confidence == pytest.approx(0.5)
    assert benign.detected is False


@pytest.mark.parametrize("keyword", ["parent", "global", "workflow_var", "sys."])
def test_run_parent_scope_reference_is_found_in_child_inputs_for_each_keyword(
    keyword: str,
) -> None:
    result = DifyIterationEscapeDetector().detect_workflow_run(
        {
            "nodes": [
                {"node_id": "loop", "node_type": "iteration"},
                {
                    "node_id": "step-0",
                    "parent_node_id": "loop",
                    "iteration_index": 0,
                    "inputs": {"source": f"reads {keyword.upper()}query"},
                    "outputs": {"value": "ok"},
                },
            ]
        }
    )

    assert result.detected is True
    refs = result.evidence["issues"][0]["children_with_parent_refs"]
    assert [r["keyword"] for r in refs] == [keyword]
    assert result.severity == TurnAwareSeverity.MINOR


def test_run_parent_scope_reference_reports_each_child_once_with_first_matching_keyword() -> None:
    children = _run_children("loop", 2)
    children[0]["inputs"] = {"source": "parent and global values"}
    children[1]["inputs"] = {"source": "workflow_var only"}

    result = DifyIterationEscapeDetector().detect_workflow_run(
        {"nodes": [{"node_id": "loop", "node_type": "iteration"}, *children]}
    )

    assert _issue_types(result) == ["parent_scope_modification"]
    refs = result.evidence["issues"][0]["children_with_parent_refs"]
    assert [(r["child_node_id"], r["keyword"]) for r in refs] == [
        ("loop-step-0", "parent"),
        ("loop-step-1", "workflow_var"),
    ]


def _loop_with_children(count: int, **parent_fields: Any) -> Dict[str, Any]:
    parent = {"node_id": "loop", "node_type": "iteration", **parent_fields}
    return {"nodes": [parent, *_run_children("loop", count)]}


@pytest.mark.parametrize("count, flagged", [(100, False), (101, True)])
def test_run_excessive_iteration_threshold_is_strictly_above_100(count: int, flagged: bool) -> None:
    run = _loop_with_children(count, outputs={"note": "stop"})

    result = DifyIterationEscapeDetector().detect_workflow_run(run)

    assert result.detected is flagged
    if flagged:
        assert _issue_types(result) == ["excessive_iterations"]
        assert result.evidence["issues"][0]["threshold"] == 100
        assert result.confidence == pytest.approx(0.7 + 1 / 500)


@pytest.mark.parametrize("count, flagged", [(50, False), (51, True)])
def test_run_missing_exit_signal_threshold_is_strictly_above_50(count: int, flagged: bool) -> None:
    result = DifyIterationEscapeDetector().detect_workflow_run(_loop_with_children(count))

    assert result.detected is flagged
    if flagged:
        assert _issue_types(result) == ["no_break_condition"]
        assert result.confidence == pytest.approx(0.6 + 1 / 50 * 0.15)


@pytest.mark.parametrize(
    "signal", ["break", "exit", "stop", "terminate", "max_iterations", "limit"]
)
def test_run_each_exit_signal_keyword_in_parent_outputs_silences_no_break_condition(
    signal: str,
) -> None:
    detector = DifyIterationEscapeDetector()

    unguarded = detector.detect_workflow_run(_loop_with_children(60, outputs={"note": "done"}))
    # Upper-case on purpose: the exit-signal search is case-insensitive.
    guarded = detector.detect_workflow_run(
        _loop_with_children(60, outputs={"note": f"loop {signal.upper()} hit"})
    )

    assert _issue_types(unguarded) == ["no_break_condition"]
    assert guarded.detected is False


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_run_failed_or_stopped_multi_iteration_node_is_a_moderate_finding(status: str) -> None:
    detector = DifyIterationEscapeDetector()

    multi = detector.detect_workflow_run(_loop_with_children(2, status=status))
    single = detector.detect_workflow_run(_loop_with_children(1, status=status))
    healthy = detector.detect_workflow_run(_loop_with_children(2, status="succeeded"))

    assert multi.detected is True
    assert _issue_types(multi) == ["iteration_failure"]
    assert multi.evidence["issues"][0]["status"] == status
    assert multi.severity == TurnAwareSeverity.MODERATE
    assert multi.confidence == pytest.approx(0.5)
    # A failure on the very first iteration is not an escape; a healthy loop is quiet.
    assert single.detected is False
    assert healthy.detected is False


def test_run_identical_outputs_across_iterations_are_detected_regardless_of_key_order() -> None:
    detector = DifyIterationEscapeDetector()
    parent = {"node_id": "loop", "node_type": "iteration"}
    same = _run_children("loop", 2)
    same[0]["outputs"] = {"a": 1, "b": 2}
    same[1]["outputs"] = {"b": 2, "a": 1}
    different = _run_children("loop", 2)
    different[0]["outputs"] = {"a": 1, "b": 2}
    different[1]["outputs"] = {"a": 1, "b": 3}

    leaked = detector.detect_workflow_run({"nodes": [parent, *same]})
    distinct = detector.detect_workflow_run({"nodes": [parent, *different]})

    assert _issue_types(leaked) == ["scope_leak_duplicate_outputs"]
    assert leaked.evidence["issues"][0]["child_count"] == 2
    assert leaked.severity == TurnAwareSeverity.MODERATE
    assert leaked.confidence == pytest.approx(0.7)
    assert distinct.detected is False


def test_run_compound_structural_findings_cap_confidence_at_point_nine() -> None:
    children = _run_children("loop", 3)
    children[1]["iteration_index"] = 3  # indices 0, 3, 2 -> gap at 1
    for child in children:
        child["outputs"] = {"value": "same"}
    parent = {
        "node_id": "loop",
        "node_type": "iteration",
        "status": "failed",
        "inputs": {"items": ["only-one"]},
    }

    result = DifyIterationEscapeDetector().detect_workflow_run({"nodes": [parent, *children]})

    assert set(_issue_types(result)) == {
        "iteration_failure",
        "index_corruption",
        "scope_leak_duplicate_outputs",
        "iteration_overrun",
    }
    # 0.6 + 4 * 0.1 would be 1.0; structural confidence is capped at 0.90.
    assert result.confidence == pytest.approx(0.90)
    assert result.severity == TurnAwareSeverity.MODERATE


@pytest.mark.parametrize("run", [{}, {"nodes": []}])
def test_run_without_nodes_is_not_a_detection(run: Dict[str, Any]) -> None:
    result = DifyIterationEscapeDetector().detect_workflow_run(run)

    assert result.detected is False
    assert result.explanation == "No nodes in workflow run"


def test_run_iteration_node_with_no_child_records_is_quiet() -> None:
    # Nothing executed under the node, so there is no iteration count to escape.
    run = {"nodes": [{"node_id": "loop", "node_type": "iteration", "status": "failed"}]}

    result = DifyIterationEscapeDetector().detect_workflow_run(run)

    assert result.detected is False
    assert result.explanation == "No iteration escape issues found"


def test_public_api_entry_point_matches_the_class_on_the_same_run() -> None:
    run = {
        "nodes": [
            {"node_id": "loop", "node_type": "loop", "inputs": {"max_iterations": 1}},
            *_run_children("loop", 3),
        ]
    }

    via_api = pd.detect_dify_iteration_escape(run)
    via_class = DifyIterationEscapeDetector().detect_workflow_run(run)

    assert via_api.detected is True
    assert via_api.evidence["affected_node_ids"] == ["loop"]
    assert via_api.evidence == via_class.evidence
