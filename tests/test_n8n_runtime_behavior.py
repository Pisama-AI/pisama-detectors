"""Behavioral contracts for the n8n runtime-trace detector lane."""

from collections.abc import Callable
from typing import Any

import pytest

from pisama_detectors.detection.n8n import (
    N8NComplexityDetector,
    N8NCycleDetector,
    N8NErrorDetector,
    N8NResourceDetector,
    N8NSchemaDetector,
    N8NTimeoutDetector,
)
from pisama_detectors.detection.turn_aware._base import TurnSnapshot


def _turn(
    number: int,
    node: str,
    content: str,
    **metadata: Any,
) -> TurnSnapshot:
    return TurnSnapshot(
        turn_number=number,
        participant_type="node",
        participant_id=node,
        content=content,
        turn_metadata=metadata,
    )


RUNTIME_CASES: dict[
    str,
    tuple[
        Callable[[], Any],
        list[TurnSnapshot],
        dict[str, Any] | None,
        list[TurnSnapshot],
        dict[str, Any] | None,
    ],
] = {
    "cycle": (
        N8NCycleDetector,
        [
            _turn(
                index,
                "A" if index % 2 == 0 else "B",
                "same A output" if index % 2 == 0 else "same B output",
            )
            for index in range(10)
        ],
        None,
        [_turn(index, f"Node {index}", f"unique output {index}") for index in range(5)],
        None,
    ),
    "error": (
        N8NErrorDetector,
        [
            _turn(
                0,
                "API Call",
                "Error: HTTP 500 Server Error",
                has_error=True,
                continue_on_fail=True,
                node_type="n8n-nodes-base.httpRequest",
            ),
            _turn(1, "Next Node", "Continued processing", has_error=False),
        ],
        {"workflow_status": "success"},
        [
            _turn(0, "Start", "Workflow started successfully", has_error=False),
            _turn(1, "Process", "Data processed successfully", has_error=False),
        ],
        {"workflow_status": "success"},
    ),
    "timeout": (
        lambda: N8NTimeoutDetector(max_workflow_duration_ms=60_000),
        [
            _turn(
                index,
                f"Node {index}",
                f"Processing step {index}",
                execution_time_ms=1_000,
            )
            for index in range(10)
        ],
        {"workflow_duration_ms": 400_000},
        [
            _turn(0, "Start", "Started", execution_time_ms=100),
            _turn(1, "Process", "Completed", execution_time_ms=500),
        ],
        {"workflow_duration_ms": 5_000},
    ),
    "complexity": (
        lambda: N8NComplexityDetector(max_node_count=20),
        [
            _turn(
                index,
                f"Node {index}",
                f"Step {index}",
                node_type="n8n-nodes-base.function",
            )
            for index in range(25)
        ],
        None,
        [
            _turn(
                index,
                f"Node {index}",
                f"Step {index}",
                node_type="n8n-nodes-base.function",
            )
            for index in range(5)
        ],
        {"workflow_duration_ms": 5_000},
    ),
    "schema": (
        N8NSchemaDetector,
        [
            _turn(0, "Producer", '{"id": 1}'),
            _turn(1, "Send Email", "cannot read property $json.to of undefined"),
        ],
        None,
        [
            _turn(
                0,
                "Producer",
                '{"to": "operator@example.com", "subject": "Status", "body": "Ready"}',
            ),
            _turn(1, "Send Email", "sent"),
        ],
        None,
    ),
    "resource": (
        N8NResourceDetector,
        [
            _turn(0, "Input", "x" * 10),
            _turn(1, "Expand", "x" * 100),
        ],
        None,
        [
            _turn(0, "Input", "x" * 100),
            _turn(1, "Output", "x" * 110),
        ],
        None,
    ),
}


@pytest.mark.parametrize(
    ("factory", "failing_turns", "failing_metadata", "healthy_turns", "healthy_metadata"),
    RUNTIME_CASES.values(),
    ids=RUNTIME_CASES.keys(),
)
def test_runtime_detector_separates_failure_from_healthy_trace(
    factory: Callable[[], Any],
    failing_turns: list[TurnSnapshot],
    failing_metadata: dict[str, Any] | None,
    healthy_turns: list[TurnSnapshot],
    healthy_metadata: dict[str, Any] | None,
) -> None:
    detector = factory()
    failure = detector.detect(failing_turns, failing_metadata)
    healthy = detector.detect(healthy_turns, healthy_metadata)

    assert failure.detected, failure.explanation
    assert failure.confidence >= 0.6
    assert not healthy.detected, healthy.explanation
