"""Behavioral contracts for n8n and OpenClaw detector failure modes."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import pisama_detectors as pd
from pisama_detectors.detection.n8n import N8NErrorDetector, N8NTimeoutDetector
from pisama_detectors.detection.turn_aware._base import TurnSnapshot


def _turn(number: int, node: str, content: str, **metadata: Any) -> TurnSnapshot:
    return TurnSnapshot(
        turn_number=number,
        participant_type="node",
        participant_id=node,
        content=content,
        turn_metadata=metadata,
    )


def _issue_types(result: Any) -> set[str]:
    return {issue["type"] for issue in result.evidence["issues"]}


def test_n8n_timeout_runtime_reports_webhook_node_and_stall_signals() -> None:
    started = datetime(2026, 7, 26, tzinfo=timezone.utc)
    result = N8NTimeoutDetector(
        max_workflow_duration_ms=30_000,
        max_webhook_wait_ms=10_000,
        stall_threshold_ms=5_000,
    ).detect(
        [
            _turn(
                0,
                "Webhook",
                "started",
                node_type="n8n-nodes-base.webhook",
                execution_time_ms=100,
                timestamp=started.isoformat(),
            ),
            _turn(
                1,
                "HTTP",
                "done",
                node_type="n8n-nodes-base.httpRequest",
                execution_time_ms=130_000,
                timestamp=(started + timedelta(seconds=40)).isoformat(),
            ),
        ],
        {"workflow_duration_ms": 40_000, "workflow_mode": "webhook"},
    )

    assert result.detected
    assert {
        "workflow_timeout",
        "webhook_timeout",
        "node_timeout",
        "stalled_execution",
    } <= _issue_types(result)


def test_n8n_error_runtime_reports_invalid_propagation_and_error_rate() -> None:
    result = N8NErrorDetector().detect(
        [
            _turn(0, "API", "invalid data", has_error=True, invalid_data=True),
            _turn(1, "Transform", "undefined value", has_error=True, invalid_data=True),
            _turn(2, "Write", "failed", has_error=True),
            _turn(3, "End", "success", has_error=False),
        ],
        {"workflow_status": "success"},
    )

    assert result.detected
    assert {"invalid_data_propagation", "high_error_rate"} <= _issue_types(result)


def test_n8n_timeout_reports_every_static_unbounded_wait_shape() -> None:
    result = pd.detect_n8n_timeout(
        {
            "nodes": [
                {"name": "Hook", "type": "n8n-nodes-base.webhook", "parameters": {}},
                {
                    "name": "HTTP",
                    "type": "n8n-nodes-base.httpRequest",
                    "parameters": {},
                },
                {
                    "name": "LLM",
                    "type": "@n8n/n8n-nodes-langchain.lmChatOpenAi",
                    "parameters": {},
                },
                {
                    "name": "Wait",
                    "type": "n8n-nodes-base.wait",
                    "parameters": {"mode": "waitForAllBranches"},
                },
            ],
            "connections": {},
        }
    )

    assert result.detected
    assert {
        "missing_workflow_timeout",
        "webhook_no_response_timeout",
        "http_no_timeout",
        "ai_no_timeout",
        "merge_wait_stall_risk",
    } <= _issue_types(result)


@pytest.mark.parametrize(
    ("session", "detected"),
    [
        (
            {
                "elevated_mode": False,
                "events": [
                    {
                        "type": "tool.call",
                        "tool_name": "search",
                        "tool_input": {"query": "status"},
                    }
                ],
            },
            False,
        ),
        (
            {
                "elevated_mode": True,
                "events": [
                    {
                        "type": "tool.call",
                        "tool_name": "search",
                        "tool_input": {"query": "status"},
                    }
                ],
            },
            True,
        ),
        (
            {
                "elevated_mode": False,
                "events": [
                    {
                        "type": "tool.call",
                        "tool_name": "delete_user",
                        "tool_input": {},
                    }
                ],
            },
            True,
        ),
        (
            {
                "elevated_mode": False,
                "events": [
                    {
                        "type": "tool.call",
                        "tool_name": "helper",
                        "tool_input": {"command": "rm /tmp/export -rf"},
                    }
                ],
            },
            True,
        ),
    ],
)
def test_openclaw_elevated_risk_obeys_privilege_and_input_boundaries(
    session: dict[str, Any], detected: bool
) -> None:
    assert pd.detect_openclaw_elevated_risk(session).detected is detected


def test_openclaw_tool_abuse_distinguishes_normal_high_error_and_sensitive_calls() -> None:
    safe = pd.detect_openclaw_tool_abuse(
        {
            "events": [
                {"type": "tool.call", "tool_name": "search", "tool_input": {"query": "x"}},
                {"type": "tool.result", "tool_result": {"status": "ok"}},
            ]
        }
    )
    high_error = pd.detect_openclaw_tool_abuse(
        {
            "events": [
                {"type": "tool.call", "tool_name": f"search_{index}", "tool_input": {}}
                for index in range(3)
            ]
            + [
                {"type": "tool.result", "tool_result": {"status": "failed"}}
                for _ in range(2)
            ]
        }
    )
    sensitive = pd.detect_openclaw_tool_abuse(
        {"events": [{"type": "tool.call", "tool_name": "delete_user", "tool_input": {}}]}
    )

    assert not safe.detected
    assert high_error.detected
    assert sensitive.detected
