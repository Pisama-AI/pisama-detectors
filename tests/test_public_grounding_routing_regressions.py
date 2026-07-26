"""Regression contracts for public grounding, routing, and specification APIs."""

from typing import Any

import pytest

import pisama_detectors as pd
from pisama_detectors.detection.hallucination import (
    HallucinationDetector,
    SourceDocument,
)
from pisama_detectors.detection.specification import MismatchSeverity, MismatchType

FRAMEWORK_DETECTORS = {
    "langgraph_recursion",
    "langgraph_state_corruption",
    "langgraph_edge_misroute",
    "langgraph_checkpoint_corruption",
    "langgraph_parallel_sync",
    "langgraph_tool_failure",
    "dify_classifier_drift",
    "dify_iteration_escape",
    "dify_rag_poisoning",
    "dify_tool_schema_mismatch",
    "dify_variable_leak",
    "dify_model_fallback",
    "n8n_cycle",
    "n8n_error",
    "n8n_timeout",
    "n8n_complexity",
    "n8n_schema",
    "n8n_resource",
    "openclaw_session_loop",
    "openclaw_sandbox_escape",
    "openclaw_tool_abuse",
    "openclaw_spawn_chain",
    "openclaw_channel_mismatch",
    "openclaw_elevated_risk",
}
N8N_DETECTORS = {name for name in FRAMEWORK_DETECTORS if name.startswith("n8n_")}


def _empty_workflow() -> dict[str, Any]:
    return {"nodes": [], "connections": {}}


def test_public_hallucination_sources_ground_and_validate_numbered_citations() -> None:
    result = pd.detect_hallucination(
        "The API requires TLS 1.3 for all requests [1].",
        ["The API requires TLS 1.3 for all requests."],
    )

    assert not result.detected
    assert result.grounding_score >= 0.8
    assert "source_grounding_score" in result.details
    assert "context_consistency_score" not in result.details
    assert "citation_issues" not in result.details


def test_run_all_detectors_routes_a_known_framework_to_its_adapters() -> None:
    results = pd.run_all_detectors(
        {
            "framework": "n8n",
            "trace": _empty_workflow(),
        }
    )

    assert set(results) == N8N_DETECTORS
    assert all(not isinstance(result, dict) or "error" not in result for result in results.values())


def test_run_all_detectors_routes_nested_framework_metadata() -> None:
    results = pd.run_all_detectors(
        {
            "trace": {
                "framework": "n8n",
                **_empty_workflow(),
            },
        }
    )

    assert set(results) == N8N_DETECTORS


def test_run_all_detectors_preserves_legacy_fanout_without_framework() -> None:
    results = pd.run_all_detectors({"trace": _empty_workflow()})

    assert set(results) == FRAMEWORK_DETECTORS


def test_run_all_detectors_preserves_legacy_fanout_for_unknown_framework() -> None:
    results = pd.run_all_detectors(
        {
            "trace": {
                "framework": "custom",
                **_empty_workflow(),
            },
        }
    )

    assert set(results) == FRAMEWORK_DETECTORS


@pytest.mark.parametrize(
    ("user_intent", "task_specification", "evidence"),
    [
        (
            "All API requests must use TLS.",
            "All API requests must not use TLS.",
            "polarity reversal",
        ),
        (
            "The worker must process at least 100 records.",
            "The worker must process at most 10 records.",
            "numeric reversal",
        ),
        (
            "Implement the service in Python.",
            "Implement the service in TypeScript.",
            "language reversal",
        ),
        (
            "The dashboard must be available to all users.",
            "The dashboard is available only to administrators.",
            "scope reversal",
        ),
        ("The service must not log PII.", "The service must log PII.", "polarity reversal"),
    ],
)
def test_specification_detects_direct_reversals(
    user_intent: str,
    task_specification: str,
    evidence: str,
) -> None:
    result = pd.detect_specification(user_intent, task_specification)

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC
    assert result.severity is MismatchSeverity.SEVERE
    assert any(evidence in item for item in result.missing_requirements)


def test_compatible_constraints_are_not_reversals() -> None:
    stronger_minimum = pd.detect_specification(
        "The worker must process at least 100 records.",
        "The worker must process at least 200 records.",
    )
    overlapping_bounds = pd.detect_specification(
        "The worker must process at least 100 records.",
        "The worker must process at most 200 records.",
    )
    typescript_superset = pd.detect_specification(
        "Implement the service in JavaScript.",
        "Implement the service in TypeScript.",
    )
    unrelated_only_qualifier = pd.detect_specification(
        "The dashboard must be available to all users.",
        "The dashboard is available to users only on weekdays.",
    )
    compatible_tls_exclusion = pd.detect_specification(
        "All API requests must use TLS 1.3.",
        "API requests must not use plaintext and must use TLS 1.3.",
    )

    assert not stronger_minimum.detected
    assert not overlapping_bounds.detected
    assert not typescript_superset.detected
    assert not unrelated_only_qualifier.detected
    assert not compatible_tls_exclusion.detected


@pytest.mark.parametrize(
    ("user_intent", "task_specification"),
    [
        (
            "The service must encrypt records.",
            "The service must encrypt records. The service must not encrypt records.",
        ),
        (
            "Implement the service in Python.",
            "Implement the service in Python. Implement the service in TypeScript.",
        ),
        (
            "The dashboard must be available to all users.",
            (
                "The dashboard must be available to all users. "
                "The dashboard is available only to administrators."
            ),
        ),
        (
            "The worker must process at least 100 records.",
            (
                "The worker must process at least 100 records. "
                "The worker must process at most 10 records."
            ),
        ),
    ],
)
def test_direct_reversals_are_checked_before_extension_shortcut(
    user_intent: str,
    task_specification: str,
) -> None:
    result = pd.detect_specification(user_intent, task_specification)

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC


def test_identical_specification_remains_clean() -> None:
    text = "The service must encrypt records."

    result = pd.detect_specification(text, text)

    assert not result.detected


@pytest.mark.parametrize(
    ("user_intent", "task_specification"),
    [
        (
            "The service must encrypt records.",
            "The service must not delete records. The service must encrypt records.",
        ),
        (
            "The API must retain audit logs.",
            "The API must not delete audit logs. The API must retain audit logs.",
        ),
        (
            (
                "All users can view the dashboard. "
                "Only administrators can edit the dashboard."
            ),
            (
                "Only administrators can edit the dashboard. "
                "All users can view the dashboard."
            ),
        ),
    ],
)
def test_unrelated_actions_on_same_subject_do_not_cross_pair(
    user_intent: str,
    task_specification: str,
) -> None:
    result = pd.detect_specification(user_intent, task_specification)

    assert not result.detected


def test_reordered_scope_constraints_remain_equivalent() -> None:
    result = pd.detect_specification(
        (
            "The dashboard is available to all users. "
            "The audit log is available only to administrators."
        ),
        (
            "The audit log is available only to administrators. "
            "The dashboard is available to all users."
        ),
    )

    assert not result.detected


def test_paraphrased_audience_reversal_is_detected() -> None:
    result = pd.detect_specification(
        "The dashboard must be available to all users.",
        "Only administrators may view the dashboard.",
    )

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC
    assert any("scope reversal" in item for item in result.missing_requirements)


def test_reordered_same_unit_numeric_constraints_remain_equivalent() -> None:
    result = pd.detect_specification(
        "Use a 5 minute timeout and a 60 minute cache.",
        "Use a 60 minute cache and a 5 minute timeout.",
    )

    assert not result.detected


def test_unrelated_same_unit_constraints_are_not_paired() -> None:
    result = pd.detect_specification(
        "Use exactly 5 seconds for the retry delay.",
        "Use exactly 10 seconds for the cache refresh.",
    )

    assert not result.detected


@pytest.mark.parametrize(
    "task_specification",
    [
        (
            "The worker must process at least 100 records and "
            "must process at most 10 records."
        ),
        (
            "The worker must process at most 10 records and "
            "must process at least 100 records."
        ),
    ],
)
def test_every_relevant_same_unit_constraint_is_compared(
    task_specification: str,
) -> None:
    result = pd.detect_specification(
        "The worker must process at least 100 records.",
        task_specification,
    )

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC


def test_percent_constraints_are_compared() -> None:
    result = pd.detect_specification(
        "The service must achieve at least 99% uptime.",
        "The service must achieve at most 50% uptime.",
    )

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC


def test_language_directives_are_paired_by_clause_subject() -> None:
    result = pd.detect_specification(
        "Implement the backend in Python. Implement the frontend in TypeScript.",
        "Implement the backend in TypeScript. Implement the frontend in Python.",
    )

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC


def test_javascript_to_typescript_compatibility_is_preserved_per_clause() -> None:
    result = pd.detect_specification(
        "Implement the backend in JavaScript. Implement the frontend in JavaScript.",
        "Implement the backend in TypeScript. Implement the frontend in TypeScript.",
    )

    assert not result.detected


@pytest.mark.parametrize(
    ("user_intent", "task_specification"),
    [
        ("Process more than 5 records.", "Process at most 5 records."),
        ("Process at least 5 records.", "Process fewer than 5 records."),
        ("Use at most 5 retries.", "Use at least 10 retries."),
        (
            "The timeout must be at most 5 minutes.",
            "The timeout must be at least 10 minutes.",
        ),
        ("Implement the service in C++.", "Implement the service in Python."),
        ("Implement the service in C#.", "Implement the service in Python."),
    ],
)
def test_strict_numeric_plural_and_language_reversals_are_detected(
    user_intent: str,
    task_specification: str,
) -> None:
    result = pd.detect_specification(user_intent, task_specification)

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC


def test_equivalent_negative_requirements_remain_equivalent() -> None:
    result = pd.detect_specification(
        "The service must not log PII or tokens.",
        "The service should never log tokens or PII.",
    )

    assert not result.detected


@pytest.mark.parametrize(
    ("user_intent", "task_specification"),
    [
        ("Never remove audit logs.", "Remove audit logs after 30 days."),
        ("Do not use plaintext.", "Use plaintext for requests."),
    ],
)
def test_common_negation_reversals_are_detected(
    user_intent: str,
    task_specification: str,
) -> None:
    result = pd.detect_specification(user_intent, task_specification)

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC
    assert any("polarity reversal" in item for item in result.missing_requirements)


def test_cannot_negation_reversal_is_detected() -> None:
    result = pd.detect_specification(
        "The service cannot log PII.",
        "The service logs PII.",
    )

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC


def test_postfixed_only_scope_reversal_is_detected() -> None:
    result = pd.detect_specification(
        "All users can view the dashboard.",
        "The dashboard is available to administrators only.",
    )

    assert result.detected
    assert result.mismatch_type is MismatchType.CONFLICTING_SPEC


@pytest.mark.parametrize(
    ("output", "source"),
    [
        (
            "The API requires TLS 1.3 for all requests [1].",
            "The API does not require TLS 1.3 for any requests.",
        ),
        (
            "The service is available only to administrators.",
            "The service is available to all users.",
        ),
    ],
)
def test_lexical_grounding_rejects_opposite_claims(output: str, source: str) -> None:
    result = pd.detect_hallucination(output, [source])

    assert result.detected
    assert result.grounding_score < 0.65
    assert any("contradicts" in item.lower() for item in result.evidence)


def test_high_overlap_does_not_mask_a_novel_number() -> None:
    result = pd.detect_hallucination(
        "The system processed 1000 records.",
        ["The system processed 10 records."],
    )

    assert result.detected
    assert result.grounding_score < 0.65
    assert "high_source_overlap" not in result.details


def test_supporting_source_clause_is_not_poisoned_by_unrelated_negation() -> None:
    result = pd.detect_hallucination(
        "The API encrypts records.",
        ["The API encrypts records. The API must not delete records."],
    )

    assert not result.detected
    assert result.grounding_score >= 0.8


def test_best_supporting_source_wins_over_a_conflicting_source() -> None:
    result = pd.detect_hallucination(
        "The API requires TLS.",
        [
            "The API does not require TLS.",
            "TLS is required by the API.",
        ],
    )

    assert not result.detected
    assert result.grounding_score >= 0.8


@pytest.mark.parametrize(
    ("output", "source"),
    [
        (
            "The API needs TLS for every request.",
            "TLS is required by the API for all requests.",
        ),
        (
            "Paris is located in France.",
            "Paris is in France.",
        ),
    ],
)
def test_supported_paraphrases_remain_grounded(output: str, source: str) -> None:
    result = pd.detect_hallucination(output, [source])

    assert not result.detected
    assert result.grounding_score >= 0.8


@pytest.mark.parametrize("position", ["start", "end"])
def test_exact_fact_in_long_source_is_grounded(position: str) -> None:
    claim = "The launch date is October 15, 2026."
    filler = (
        "This section discusses routine maintenance procedures and operational background. "
        * 120
    )
    source = f"{claim} {filler}" if position == "start" else f"{filler} {claim}"

    result = pd.detect_hallucination(claim, [source])

    assert not result.detected
    assert result.grounding_score >= 0.8


@pytest.mark.parametrize(
    ("output", "sources"),
    [
        ("The Moon is made of cheese.", ["The Moon is rocky."]),
        ("Revenue increased this quarter.", ["Revenue decreased this quarter."]),
        ("Cats are mammals.", ["Dogs are mammals."]),
        (
            "The request timeout is 10 seconds.",
            ["The request timeout is 5 seconds. The cache TTL is 10 seconds."],
        ),
    ],
)
def test_semantic_relatedness_alone_does_not_ground_claims(
    output: str,
    sources: list[str],
) -> None:
    result = pd.detect_hallucination(output, sources)

    assert result.detected
    assert result.grounding_score < 0.65


@pytest.mark.parametrize(
    "output",
    [
        "The API requires TLS 1.3 (source: fabricated).",
        "The API requires TLS 1.3 {{cite:fabricated}}.",
    ],
)
def test_named_citations_must_resolve_to_source_metadata(output: str) -> None:
    detector = HallucinationDetector()
    result = detector.detect_hallucination(
        output,
        [SourceDocument("The API requires TLS 1.3.", {"id": "official"})],
    )

    assert result.detected
    assert result.hallucination_type == "invalid_citation"
    assert "citation_issues" in result.details


@pytest.mark.parametrize(
    "citation",
    ["[1]", "(source: 1)", "{{cite:1}}", "(source: official)", "{{cite:official}}"],
)
def test_all_supported_citation_forms_resolve(citation: str) -> None:
    detector = HallucinationDetector()
    result = detector.detect_hallucination(
        f"The API requires TLS 1.3 {citation}.",
        [SourceDocument("The API requires TLS 1.3.", {"id": "official"})],
    )

    assert "citation_issues" not in result.details


@pytest.mark.parametrize(
    "source",
    [
        {
            "content": "The API requires TLS.",
            "metadata": {"title": "Official Guide"},
        },
        {
            "content": "The API requires TLS.",
            "title": "Official Guide",
        },
    ],
)
def test_public_source_mappings_resolve_named_citations(source: dict[str, Any]) -> None:
    result = pd.detect_hallucination(
        "The API requires TLS (source: Official Guide).",
        [source],
    )

    assert not result.detected
    assert result.grounding_score >= 0.8
    assert "citation_issues" not in result.details


def test_any_invalid_citation_is_reported() -> None:
    result = pd.detect_hallucination(
        "The API requires TLS [1] [1] [1] [1] [9].",
        ["The API requires TLS."],
    )

    assert result.detected
    assert result.hallucination_type == "invalid_citation"
    assert "citation_issues" in result.details


def test_short_factual_claims_are_grounded() -> None:
    result = pd.detect_hallucination("The total is 30.", ["The total is 20."])

    assert result.detected
    assert result.grounding_score < 0.65
