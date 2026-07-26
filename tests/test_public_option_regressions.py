"""Regression coverage for configurable public detector helpers.

The cases exercise the real public API and detector implementations. They do
not replace detector classes, tokenizers, or pricing data with test doubles.
"""

from __future__ import annotations

import pytest

import pisama_detectors as pd
from pisama_detectors.detection.loop import MultiLevelLoopDetector, StateSnapshot
from pisama_detectors.detection.overflow import ContextOverflowDetector
from pisama_detectors.detection.shared_embedder import get_shared_embedder

LOOP_STATES = [
    {"agent_id": "worker-a", "output": "Repeated deployment request."},
    {"agent_id": "worker-b", "output": "Database migration completed."},
    {"agent_id": "worker-c", "output": "Security audit finished."},
    {"agent_id": "worker-a", "output": "Repeated deployment request."},
]


def test_detect_loop_honors_window_size() -> None:
    narrow = pd.detect_loop(LOOP_STATES, window_size=2)
    broad = pd.detect_loop(LOOP_STATES, window_size=4)

    assert not narrow.detected
    assert broad.detected
    assert broad.loop_start_index == 0


def test_detect_loop_reports_origin_of_actual_prior_window() -> None:
    states = [
        {"agent_id": "outside", "output": "Unrelated earlier state."},
        {"agent_id": "worker-a", "output": "Repeated deployment request."},
        {"agent_id": "worker-b", "output": "Database migration completed."},
        {"agent_id": "worker-c", "output": "Security audit finished."},
        {"agent_id": "worker-d", "output": "Documentation published."},
        {"agent_id": "worker-a", "output": "Repeated deployment request."},
    ]

    result = pd.detect_loop(states, window_size=5)

    assert result.detected
    assert result.method == "structural"
    assert result.loop_start_index == 1


@pytest.mark.parametrize("window_size", [0, -1])
def test_detect_loop_rejects_non_positive_window_size(window_size: int) -> None:
    with pytest.raises(ValueError, match="window_size must be at least 1"):
        pd.detect_loop(LOOP_STATES, window_size=window_size)


@pytest.mark.parametrize("similarity_threshold", [-0.01, 1.01, float("nan")])
def test_detect_loop_rejects_invalid_similarity_threshold(
    similarity_threshold: float,
) -> None:
    with pytest.raises(ValueError, match="similarity_threshold must be between 0 and 1"):
        pd.detect_loop(LOOP_STATES, similarity_threshold=similarity_threshold)


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        ("window_size", 0, "window_size"),
        ("window_size", float("nan"), "window_size"),
        ("min_matches_for_loop", 0, "min_matches_for_loop"),
        ("min_matches_for_loop", float("nan"), "min_matches_for_loop"),
        ("structural_threshold", float("nan"), "structural_threshold"),
        ("semantic_threshold", float("nan"), "semantic_threshold"),
        ("confidence_scaling", 0.0, "confidence_scaling"),
        ("confidence_scaling", float("nan"), "confidence_scaling"),
    ],
)
def test_loop_engine_rejects_invalid_configuration(
    option: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MultiLevelLoopDetector(**{option: value})


def test_loop_engine_accepts_zero_similarity_thresholds() -> None:
    detector = MultiLevelLoopDetector(
        structural_threshold=0.0,
        semantic_threshold=0.0,
    )

    assert detector.structural_threshold == 0.0
    assert detector.semantic_threshold == 0.0


def test_detect_loop_honors_similarity_threshold_with_semantic_extra() -> None:
    pytest.importorskip("sentence_transformers")
    if get_shared_embedder() is None:
        pytest.skip("semantic model is unavailable")

    states = [
        {
            "agent_id": "worker-z",
            "output": "The deployment destination is missing.",
        },
        {
            "agent_id": "worker-a",
            "output": "Database migration completed successfully.",
        },
        {
            "agent_id": "worker-b",
            "output": "The deployment destination is missing.",
        },
        {
            "agent_id": "worker-c",
            "output": "Cannot deploy because no destination was specified.",
        },
        {
            "agent_id": "worker-d",
            "output": "Security audit completed successfully.",
        },
        {
            "agent_id": "worker-z",
            "output": "Deployment is blocked until a destination is provided.",
        },
    ]

    permissive = pd.detect_loop(states, similarity_threshold=0.1)
    strict = pd.detect_loop(states, similarity_threshold=1.0)

    assert permissive.detected
    assert permissive.method == "semantic_clustering"
    assert permissive.evidence is not None
    assert permissive.evidence["type"] == "cluster_dominance"
    assert not strict.detected


def test_semantic_cycle_only_path_honors_similarity_threshold() -> None:
    pytest.importorskip("sentence_transformers")
    if get_shared_embedder() is None:
        pytest.skip("semantic model is unavailable")

    contents = [
        ("worker-z", "Quantum physics particle wave electron photon laboratory " * 5),
        ("worker-q", "Cooking recipe kitchen bread flour oven dinner " * 5),
        (
            "worker-a",
            "Financial invoice payment bank account balance transaction settled " * 8,
        ),
        (
            "worker-b",
            "Network server cable router connection offline unreachable datacenter " * 8,
        ),
        (
            "worker-c",
            "Banking transaction settled the account invoice and payment balance " * 8,
        ),
        (
            "worker-z",
            "Datacenter connectivity remains unreachable while the router cable "
            "and server are offline " * 8,
        ),
    ]
    states = [
        StateSnapshot(
            agent_id=agent_id,
            state_delta={},
            content=content,
            sequence_num=index,
        )
        for index, (agent_id, content) in enumerate(contents)
    ]

    permissive = MultiLevelLoopDetector(
        window_size=4,
        semantic_threshold=0.8,
    ).detect_loop(states)
    strict = MultiLevelLoopDetector(
        window_size=4,
        semantic_threshold=1.0,
    ).detect_loop(states)

    assert permissive.detected
    assert permissive.method == "semantic_clustering"
    assert permissive.evidence is not None
    assert permissive.evidence["type"] == "cluster_cycle"
    assert not strict.detected


def test_detect_overflow_counts_context_and_output_tokens() -> None:
    context = "System: Review the attached release checklist carefully."
    output = "Assistant: The release checklist is complete and verified."
    detector = ContextOverflowDetector()

    result = pd.detect_overflow(context, output, model="claude-sonnet-4-6")
    context_only = pd.detect_overflow(context, "", model="claude-sonnet-4-6")

    expected_tokens = detector.count_tokens(context, "claude-sonnet-4-6")
    expected_tokens += detector.count_tokens(output, "claude-sonnet-4-6")

    assert result.current_tokens == expected_tokens
    assert result.current_tokens > context_only.current_tokens


def test_detect_overflow_always_counts_nonempty_separately_supplied_output() -> None:
    output = "The release checklist is complete and verified."
    context = f"System: Review the release checklist.\nAssistant: {output}"
    detector = ContextOverflowDetector()

    separate = pd.detect_overflow(context, output, model="claude-sonnet-4-6")
    already_included = pd.detect_overflow(context, "", model="claude-sonnet-4-6")

    context_tokens = detector.count_tokens(context, "claude-sonnet-4-6")
    output_tokens = detector.count_tokens(output, "claude-sonnet-4-6")

    assert separate.current_tokens == context_tokens + output_tokens
    assert already_included.current_tokens == context_tokens


def test_detect_overflow_accepts_exact_provider_token_count() -> None:
    result = pd.detect_overflow(
        "context",
        "output",
        model="claude-sonnet-4-6",
        provider_token_count=987_654,
    )

    assert result.current_tokens == 987_654
    assert result.details["token_count_source"] == "provider"


@pytest.mark.parametrize("provider_token_count", [-1, 1.5, True])
def test_detect_overflow_rejects_invalid_provider_token_count(
    provider_token_count: object,
) -> None:
    with pytest.raises(ValueError, match="provider_token_count"):
        pd.detect_overflow(
            "",
            "",
            provider_token_count=provider_token_count,
        )


def test_overflow_estimator_handles_special_tokens_and_long_repetition() -> None:
    detector = ContextOverflowDetector()
    special = "<|endoftext|> content <|fim_prefix|>"
    repeated = "x" * 1_000_000

    special_tokens = detector.count_tokens(special, "claude-sonnet-4-6")
    repeated_tokens = detector.count_tokens(repeated, "claude-sonnet-4-6")
    result = pd.detect_overflow(special, repeated, model="claude-sonnet-4-6")

    assert special_tokens > 0
    assert repeated_tokens > 0
    assert result.current_tokens == special_tokens + repeated_tokens
    assert result.details["token_count_source"] == "offline_estimate"


def test_claude_sonnet_4_6_has_current_pricing_and_context_window() -> None:
    cost = pd.calculate_cost("claude-sonnet-4-6", 1_000_000, 1_000_000)
    overflow = pd.detect_overflow("", "", model="claude-sonnet-4-6")

    assert cost.provider == "anthropic"
    assert cost.model == "claude-sonnet-4-6"
    assert cost.input_cost_usd == 3.0
    assert cost.output_cost_usd == 15.0
    assert cost.total_cost_usd == 18.0
    assert overflow.context_window == 1_000_000
