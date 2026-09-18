"""Behavioral contracts for the reconciled LangGraph state-corruption detector.

Covers the outcome-awareness and schema-awareness added when the detector was
reconciled with the production-validated version:

* a value that satisfies its DECLARED type is the schema working, not corruption;
* a successful run may stall a retry counter, but a list shrink is only excused
  when the schema explicitly declares replacement / top-k semantics for that key;
* affected steps and confidence are derived from the final evidence set.
"""

from typing import Any, Dict, List, Optional

import pytest

import pisama_detectors as pd
from pisama_detectors.detection.langgraph.state_corruption_detector import (
    _declared_field_type,
    _field_declaration,
    _list_replacement_is_declared,
)
from pisama_detectors.detection.turn_aware._base import TurnAwareSeverity


def _snapshots(*states: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{"superstep": index, "state": state} for index, state in enumerate(states)]


def _run(
    *states: Dict[str, Any],
    status: Optional[str] = None,
    schema: Any = None,
    nodes: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    trace: Dict[str, Any] = {
        "state_snapshots": _snapshots(*states),
        "nodes": nodes or [],
    }
    if status is not None:
        trace["status"] = status
    if schema is not None:
        trace["state_schema"] = schema
    return pd.detect_langgraph_state_corruption(trace)


def _types(result: Any) -> List[str]:
    return [signal["type"] for signal in result.evidence["signals"]]


# --- Declared-type awareness: None -> value ---------------------------------


@pytest.mark.parametrize(
    "schema",
    [
        {"selected_docs": "Optional[list]"},
        {"selected_docs": "list | None"},
        {"selected_docs": "list"},
        {"selected_docs": {"type": "Optional[list]"}},
        {"selected_docs": {"annotation": "array"}},
        {"selected_docs": {"declared_type": "Optional[List]"}},
        {"fields": {"selected_docs": {"type": "Optional[list]"}}},
        {"properties": {"selected_docs": "sequence"}},
        {"channels": {"selected_docs": {"annotation": "list"}}},
    ],
)
def test_filling_an_optional_field_with_its_declared_type_is_not_corruption(
    schema: Dict[str, Any],
) -> None:
    result = _run(
        {"selected_docs": None, "query": "q"},
        {"selected_docs": ["doc-1"], "query": "q"},
        schema=schema,
    )

    assert not result.detected
    assert result.severity == TurnAwareSeverity.NONE
    assert result.confidence == pytest.approx(0.10)
    assert not result.evidence


def test_same_none_to_value_transition_is_flagged_without_a_declaration() -> None:
    result = _run(
        {"selected_docs": None, "query": "q"},
        {"selected_docs": ["doc-1"], "query": "q"},
    )

    assert result.detected
    assert _types(result) == ["type_change"]
    signal = result.evidence["signals"][0]
    assert signal["key"] == "selected_docs"
    assert signal["previous_type"] == "null"
    assert signal["current_type"] == "list"


@pytest.mark.parametrize(
    "schema",
    [
        # The filled-in value violates the declared type.
        {"selected_docs": "Optional[dict]"},
        {"selected_docs": {"type": "int"}},
        # No parseable type: reducer metadata must not be mistaken for a type.
        {"selected_docs": {"reducer": "top_k"}},
        {"selected_docs": {"type": ["list"]}},
        # A type form the guard does not parse gives no evidence either way.
        {"selected_docs": "Optional[SomeCustomModel]"},
        # Declared for a different key only.
        {"other_field": "Optional[list]"},
        # Non-dict containers are ignored instead of trusted.
        {"fields": ["selected_docs"]},
    ],
)
def test_declaration_that_does_not_cover_the_value_still_reports_type_change(
    schema: Dict[str, Any],
) -> None:
    result = _run(
        {"selected_docs": None, "query": "q"},
        {"selected_docs": ["doc-1"], "query": "q"},
        schema=schema,
    )

    assert result.detected
    assert _types(result) == ["type_change"]


def test_declared_type_only_excuses_the_declared_key() -> None:
    result = _run(
        {"selected_docs": None, "score": None},
        {"selected_docs": ["doc-1"], "score": "high"},
        schema={"selected_docs": "Optional[list]", "score": "Optional[int]"},
    )

    assert result.detected
    flagged = [s["key"] for s in result.evidence["signals"] if s["type"] == "type_change"]
    assert flagged == ["score"]


def test_a_bool_does_not_satisfy_a_declared_int() -> None:
    states = ({"flag": None, "q": 1}, {"flag": True, "q": 1})

    as_int = _run(*states, schema={"flag": "Optional[int]"})
    as_bool = _run(*states, schema={"flag": "Optional[bool]"})

    # bool is an int subclass in Python, but a declared int field holding True
    # is a genuine type mismatch, so only the bool declaration excuses it.
    assert as_int.detected
    assert _types(as_int) == ["type_change"]
    assert not as_bool.detected


def test_identity_field_first_assignment_is_never_a_type_change() -> None:
    result = _run(
        {"session_id": None, "q": 1},
        {"session_id": "s1", "q": 1},
    )

    assert not result.detected
    assert result.evidence == {}


@pytest.mark.parametrize("status", ["running", "completed"])
def test_none_schema_is_treated_as_no_declaration(status: str) -> None:
    # A null state_schema must behave like an absent one on both schema paths:
    # the declared-type lookup and the success-time replacement lookup.
    result = pd.detect_langgraph_state_corruption(
        {
            "status": status,
            "state_snapshots": _snapshots(
                {"docs": None, "notes": ["a", "b", "c"]},
                {"docs": ["a"], "notes": ["a"]},
            ),
            "nodes": [],
            "state_schema": None,
        }
    )

    assert result.detected
    assert sorted((s["type"], s["key"]) for s in result.evidence["signals"]) == [
        ("list_shrinkage", "notes"),
        ("type_change", "docs"),
    ]


# --- Outcome awareness: successful runs -------------------------------------


def test_successful_run_may_stall_a_retry_counter_but_a_running_one_may_not() -> None:
    states = ({"retry_count": 3, "q": 1}, {"retry_count": 3, "q": 1})

    completed = _run(*states, status="completed")
    running = _run(*states, status="running")
    no_status = _run(*states)

    assert not completed.detected
    assert completed.severity == TurnAwareSeverity.NONE
    assert completed.confidence == pytest.approx(0.10)
    assert not completed.evidence
    for unfinished in (running, no_status):
        assert unfinished.detected
        assert _types(unfinished) == ["counter_stall"]
        assert unfinished.severity == TurnAwareSeverity.MODERATE
        signal = unfinished.evidence["signals"][0]
        assert signal["key"] == "retry_count"
        assert signal["value"] == 3


@pytest.mark.parametrize("node_status", ["failed", "FAILED", "error", "Errored"])
def test_a_failed_node_voids_the_success_exemption_for_counter_stall(
    node_status: str,
) -> None:
    result = _run(
        {"retry_count": 3, "q": 1},
        {"retry_count": 3, "q": 1},
        status="completed",
        nodes=[{"node_id": "tool", "status": node_status, "superstep": 1}],
    )

    assert result.detected
    assert "counter_stall" in _types(result)


def test_successful_run_does_not_excuse_a_counter_that_went_backwards() -> None:
    result = _run(
        {"step_count": 5, "q": 1},
        {"step_count": 2, "q": 1},
        status="completed",
    )

    assert result.detected
    assert _types(result) == ["counter_decrease"]


def test_unrecognised_status_gives_no_success_exemption() -> None:
    result = _run(
        {"retry_count": 3, "q": 1},
        {"retry_count": 3, "q": 1},
        status="wibble",
    )

    assert result.detected
    assert _types(result) == ["counter_stall"]


def test_failed_status_gives_no_success_exemption() -> None:
    result = _run(
        {"documents": ["a", "b", "c"]},
        {"documents": ["a"]},
        status="failed",
        schema={"documents": "top_k"},
    )

    assert result.detected
    assert _types(result) == ["list_shrinkage"]


# --- Outcome awareness: list shrinkage needs a declared replacement ---------


_SHRINK = ({"documents": ["a", "b", "c"]}, {"documents": ["a"]})


def test_successful_run_still_reports_an_undeclared_list_shrink() -> None:
    result = _run(*_SHRINK, status="completed")

    assert result.detected
    assert _types(result) == ["list_shrinkage"]
    signal = result.evidence["signals"][0]
    assert signal["key"] == "documents"
    assert signal["previous_length"] == 3
    assert signal["current_length"] == 1
    assert result.severity == TurnAwareSeverity.MODERATE


@pytest.mark.parametrize(
    "schema",
    [
        # Compact string declarations, normalised for case, dashes and spaces.
        {"documents": "top_k"},
        {"documents": "Top-K"},
        {"documents": " last value "},
        {"documents": "OVERWRITE"},
        # Per-field dict declarations under each recognised semantic key.
        {"documents": {"reducer": "replace"}},
        {"documents": {"merge_strategy": "replacement"}},
        {"documents": {"update": "lastvalue"}},
        {"documents": {"semantics": "topk"}},
        {"documents": {"mode": "overwrite"}},
        # Nested and list-valued semantics are searched.
        {"documents": {"reducer": {"kind": "top_k", "k": 3}}},
        {"documents": {"reducer": ["append", "replace"]}},
        # Nested field containers.
        {"fields": {"documents": {"reducer": "replace"}}},
        {"channels": {"documents": "last_value"}},
        # Separate reducer tables.
        {"reducers": {"documents": "replace"}},
        {"field_reducers": {"documents": "overwrite"}},
        {"merge_strategies": {"documents": {"strategy": "top_k"}}},
        {"update_strategies": {"documents": ["last_value"]}},
    ],
)
def test_successful_run_excuses_a_list_shrink_when_replacement_is_declared(
    schema: Dict[str, Any],
) -> None:
    result = _run(*_SHRINK, status="completed", schema=schema)

    assert not result.detected
    assert result.severity == TurnAwareSeverity.NONE
    assert result.confidence == pytest.approx(0.10)
    assert not result.evidence


@pytest.mark.parametrize(
    "schema",
    [
        # Append-style reducers are the opposite of replacement.
        {"documents": "append"},
        {"documents": {"reducer": "add_messages"}},
        {"reducers": {"documents": "append"}},
        # A type annotation is not a semantic declaration.
        {"documents": {"type": "list"}},
        {"documents": "list"},
        # Only the recognised semantic keys count; other metadata is ignored
        # even when its text happens to spell a replacement word.
        {"documents": {"description": "top_k"}},
        {"documents": {"type": "list", "note": "replace"}},
        # Replacement is declared for a different field only.
        {"notes": "top_k"},
        {"reducers": {"notes": "replace"}},
        # A substring is not an explicit declaration.
        {"documents": "replace_if_empty"},
        # Semantic keys inside non-dict containers are not read.
        {"reducers": ["documents"]},
        {"documents": 3},
    ],
)
def test_successful_run_keeps_a_list_shrink_when_replacement_is_not_declared(
    schema: Dict[str, Any],
) -> None:
    result = _run(*_SHRINK, status="completed", schema=schema)

    assert result.detected
    assert _types(result) == ["list_shrinkage"]


def test_replacement_declaration_is_ignored_when_the_run_did_not_succeed() -> None:
    # Only a successful outcome earns the exemption; a declaration alone does not.
    result = _run(*_SHRINK, status="running", schema={"documents": "top_k"})

    assert result.detected
    assert _types(result) == ["list_shrinkage"]


def test_only_the_declared_list_is_excused_and_the_other_still_fires() -> None:
    result = _run(
        {"documents": ["a", "b", "c"], "audit_log": ["e1", "e2", "e3"], "retry_count": 2},
        {"documents": ["a"], "audit_log": ["e1"], "retry_count": 2},
        status="completed",
        schema={"documents": "top_k"},
    )

    assert result.detected
    assert [(s["type"], s["key"]) for s in result.evidence["signals"]] == [
        ("list_shrinkage", "audit_log")
    ]
    assert result.evidence["signal_count"] == 1


# --- Confidence and affected steps come from the final evidence set ---------


def test_confidence_counts_only_signals_that_survive_the_success_filter() -> None:
    states = (
        {"audit_log": ["e1", "e2", "e3"], "retry_count": 2},
        {"audit_log": ["e1"], "retry_count": 2},
    )

    running = _run(*states, status="running")
    completed = _run(*states, status="completed")

    assert sorted(_types(running)) == ["counter_stall", "list_shrinkage"]
    assert running.confidence == pytest.approx(0.8)
    assert _types(completed) == ["list_shrinkage"]
    assert completed.confidence == pytest.approx(0.7)
    assert completed.confidence < running.confidence
    assert running.evidence["signal_count"] == 2
    assert completed.evidence["signal_count"] == 1


def test_suppressed_signals_do_not_leak_their_supersteps_into_affected_turns() -> None:
    states = (
        {"documents": ["a", "b", "c"], "session_id": "s1"},
        {"documents": ["a"], "session_id": "s1"},
        {"documents": ["a"], "session_id": "s2"},
    )

    excused = _run(*states, status="completed", schema={"documents": "top_k"})
    unexcused = _run(*states, status="completed")

    assert _types(excused) == ["identity_mutation"]
    assert excused.affected_turns == [2]
    assert _types(unexcused) == ["list_shrinkage", "identity_mutation"]
    assert unexcused.affected_turns == [1, 2]


def test_node_error_supersteps_are_included_in_affected_turns() -> None:
    result = _run(
        {"score": 1, "q": 1},
        {"score": "one", "q": 1},
        status="failed",
        nodes=[
            {
                "node_id": "validate",
                "node_type": "tool",
                "status": "failed",
                "superstep": 4,
                "error": "Schema violation: state mismatch",
            }
        ],
    )

    assert result.detected
    assert result.severity == TurnAwareSeverity.SEVERE
    assert sorted(_types(result)) == ["node_error", "type_change"]
    assert result.affected_turns == [1, 4]
    node_signal = next(s for s in result.evidence["signals"] if s["type"] == "node_error")
    assert node_signal["node_id"] == "validate"
    assert node_signal["superstep"] == 4
    assert {"schema violation", "state mismatch"} <= set(node_signal["error_keywords"])


def test_node_error_without_a_superstep_adds_no_affected_turn() -> None:
    result = _run(
        {"score": 1},
        {"score": 1},
        status="failed",
        nodes=[
            {
                "node_id": "validate",
                "status": "failed",
                "error": "corrupt checkpoint",
            }
        ],
    )

    assert result.detected
    assert _types(result) == ["node_error"]
    assert result.affected_turns == []


def test_a_failed_node_without_corruption_keywords_is_not_evidence() -> None:
    result = _run(
        {"retry_count": 3, "q": 1},
        {"retry_count": 3, "q": 1},
        status="completed",
        nodes=[{"node_id": "tool", "status": "failed", "error": "connection timed out"}],
    )

    # The failure voids the success exemption, so the stall is reported, but the
    # unrelated error text contributes no node_error signal of its own.
    assert _types(result) == ["counter_stall"]


# --- Pure helpers ------------------------------------------------------------


def test_field_declaration_reads_flat_then_nested_containers() -> None:
    assert _field_declaration({"docs": "list"}, "docs") == "list"
    assert _field_declaration({"fields": {"docs": "list"}}, "docs") == "list"
    assert _field_declaration({"properties": {"docs": "dict"}}, "docs") == "dict"
    assert _field_declaration({"channels": {"docs": "int"}}, "docs") == "int"
    # Flat declarations win over nested ones.
    assert _field_declaration({"docs": "flat", "fields": {"docs": "nested"}}, "docs") == "flat"
    assert _field_declaration({"fields": {"other": "list"}}, "docs") is None
    assert _field_declaration({"fields": "not-a-dict"}, "docs") is None
    assert _field_declaration({}, "docs") is None


def test_declared_field_type_only_returns_string_type_annotations() -> None:
    assert _declared_field_type({"docs": "Optional[list]"}, "docs") == "Optional[list]"
    assert _declared_field_type({"docs": {"type": "list"}}, "docs") == "list"
    assert _declared_field_type({"docs": {"annotation": "dict"}}, "docs") == "dict"
    assert _declared_field_type({"docs": {"declared_type": "int"}}, "docs") == "int"
    # "type" wins over the other annotation keys when several are present.
    assert _declared_field_type({"docs": {"type": "list", "annotation": "dict"}}, "docs") == "list"
    # A non-string "type" falls through to the next recognised key.
    assert _declared_field_type({"docs": {"type": 3, "annotation": "dict"}}, "docs") == "dict"
    # Reducer metadata is not a type.
    assert _declared_field_type({"docs": {"reducer": "top_k"}}, "docs") is None
    assert _declared_field_type({"docs": ["list"]}, "docs") is None
    assert _declared_field_type({}, "docs") is None


def test_list_replacement_is_declared_is_scoped_to_the_exact_key() -> None:
    schema = {
        "documents": {"reducer": "top_k"},
        "history": "append",
        "reducers": {"scratch": "overwrite"},
    }

    assert _list_replacement_is_declared(schema, "documents") is True
    assert _list_replacement_is_declared(schema, "scratch") is True
    assert _list_replacement_is_declared(schema, "history") is False
    assert _list_replacement_is_declared(schema, "unknown") is False
    assert _list_replacement_is_declared({}, "documents") is False
