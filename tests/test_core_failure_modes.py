"""Behavioral contracts for distinct core detector failure modes.

Every case runs the real public detector implementation with deterministic
inputs. No detector internals, pricing tables, or model responses are replaced.
"""

from typing import Any

import pytest

import pisama_detectors as pd
from pisama_detectors.detection.cost import CostCalculator
from pisama_detectors.detection.overflow import ContextOverflowDetector, OverflowSeverity


def test_loop_preserves_iteration_and_summary_false_positive_guards() -> None:
    progress = pd.detect_loop(
        [{"output": f"Process item {index}", "item": index} for index in range(6)]
    )
    summary = pd.detect_loop(
        [{"output": f"Unique task {index}"} for index in range(5)]
        + [{"output": "In summary, all requested items are complete."}]
    )
    bookkeeping_only = pd.detect_loop(
        [
            {"output": "Retry database connection", "retry_count": index}
            for index in range(6)
        ]
    )

    assert not progress.detected
    assert not summary.detected
    assert bookkeeping_only.detected


@pytest.mark.parametrize(
    ("output", "expected_type"),
    [
        (
            (
                "According to a 2025 study, 93% of developers agreed. "
                "Dr. Ada Smith published in the Future Systems Journal. "
                "Scientists predict this directly caused guaranteed 300% growth."
            ),
            "fabricated_fact",
        ),
        (
            (
                "There is a critical bug, missing validation, and an error. "
                "But this is minor and not a big deal. Score: 9/10."
            ),
            "self_negotiation",
        ),
    ],
)
def test_hallucination_classifies_strong_nonsemantic_failure_modes(
    output: str, expected_type: str
) -> None:
    result = pd.detect_hallucination(output)

    assert result.detected
    assert result.hallucination_type == expected_type


@pytest.mark.parametrize(
    ("metrics", "expected_type"),
    [
        ([1, 2, 3, 4, 5], "regression"),
        ([1, 3, 1, 3, 1, 3, 1], "thrashing"),
        ([1, 2, 4, 8, 16, 32], "divergence"),
    ],
)
def test_convergence_reports_each_optimization_failure_mode(
    metrics: list[float], expected_type: str
) -> None:
    result = pd.detect_convergence(metrics, direction="minimize")

    assert result.detected
    assert expected_type in {issue.failure_type.value for issue in result.issues}


@pytest.mark.parametrize(
    ("messages", "expected_type"),
    [
        (
            [
                {
                    "sender": "A",
                    "receiver": "B",
                    "content": "Please handle this task",
                    "timestamp": 1,
                    "acknowledged": True,
                },
                {
                    "sender": "B",
                    "receiver": "A",
                    "content": "I will hand off this task back to you",
                    "timestamp": 2,
                    "acknowledged": True,
                },
            ],
            "circular_delegation",
        ),
        (
            [
                {
                    "sender": "A",
                    "receiver": "worker",
                    "content": "enable feature",
                    "timestamp": 1,
                    "acknowledged": True,
                },
                {
                    "sender": "B",
                    "receiver": "worker",
                    "content": "disable feature",
                    "timestamp": 2,
                    "acknowledged": True,
                },
            ],
            "conflicting_instructions",
        ),
        (
            [
                {
                    "sender": "lead",
                    "receiver": "A",
                    "content": "Review the release test evidence now",
                    "timestamp": 1,
                    "acknowledged": True,
                },
                {
                    "sender": "lead",
                    "receiver": "B",
                    "content": "Review the release test evidence now",
                    "timestamp": 2,
                    "acknowledged": True,
                },
            ],
            "duplicate_dispatch",
        ),
        (
            [
                {
                    "sender": "A",
                    "receiver": "db",
                    "content": "acquire resource database",
                    "timestamp": 1,
                    "acknowledged": True,
                },
                {
                    "sender": "B",
                    "receiver": "db",
                    "content": "acquire resource database",
                    "timestamp": 2,
                    "acknowledged": True,
                },
            ],
            "resource_contention",
        ),
        (
            [
                {
                    "sender": "lead",
                    "receiver": "team",
                    "content": "Progress: 0/7 steps completed",
                    "timestamp": 1,
                    "acknowledged": True,
                }
            ],
            "stalled_progress",
        ),
    ],
)
def test_coordination_reports_distinct_multi_agent_failure_modes(
    messages: list[dict[str, Any]], expected_type: str
) -> None:
    result = pd.detect_coordination(messages)

    assert result.detected
    assert expected_type in {issue.issue_type for issue in result.issues}


@pytest.mark.parametrize(
    ("previous", "current", "expected_type"),
    [
        ({"profile": {"age": 20}}, {"profile": {"age": "twenty"}}, "type_drift"),
        ({"balance": 100}, {"balance": 1_000_000}, "extreme_magnitude_change"),
        (
            {"a": "unique-value", "b": "other"},
            {"a": "unique-value", "b": "unique-value"},
            "suspicious_value_copy",
        ),
        (
            {"current_task_id": "known"},
            {"current_task_id": "missing"},
            "hallucinated_reference",
        ),
    ],
)
def test_corruption_reports_distinct_state_integrity_failures(
    previous: dict[str, Any], current: dict[str, Any], expected_type: str
) -> None:
    result = pd.detect_corruption(previous, current)

    assert result.detected
    assert expected_type in {issue.issue_type for issue in result.issues}


def test_overflow_engine_covers_message_accounting_memory_growth_and_remediation() -> None:
    detector = ContextOverflowDetector()
    messages = [
        {"role": "system", "content": "policy " * 500},
        {"role": "user", "content": [{"text": "question " * 100}]},
        {
            "role": "assistant",
            "content": "calling tool",
            "tool_calls": [{"function": {"arguments": "query " * 100}}],
        },
        {"role": "tool", "content": "result " * 500},
    ]

    result = detector.detect_overflow(
        current_tokens=7_800,
        model="gpt-4",
        messages=messages,
        expected_output_tokens=512,
    )
    leak = detector.detect_memory_leak([100, 120, 150, 190, 240], "gpt-4")
    suggestions = detector.suggest_remediation(result)

    assert result.severity in {OverflowSeverity.CRITICAL, OverflowSeverity.OVERFLOW}
    assert result.details["token_breakdown"]["system"] > 0
    assert result.details["token_breakdown"]["tools"] > 0
    assert leak is not None and leak["leak_detected"]
    assert suggestions


def test_cost_engine_aggregates_aliases_and_multiple_providers() -> None:
    calculator = CostCalculator()
    result = calculator.calculate_trace_cost(
        [
            {
                "model": "gpt-4o-2024-11-20",
                "prompt_tokens": 1_000_000,
                "completion_tokens": 1_000_000,
            },
            {
                "model": "gemini-2.5-pro",
                "input_tokens": 1_000_000,
                "output_tokens": 1_000_000,
            },
        ]
    )

    assert result.total_tokens == 4_000_000
    assert result.total_cost_usd > 0
    assert {"openai", "google"} == set(result.provider.split(","))
