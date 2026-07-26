"""Behavioral contracts for the shipped turn-aware compatibility namespace."""

from collections.abc import Callable
from typing import Any

import pytest

from pisama_detectors.detection.turn_aware import (
    HybridCommunicationBreakdownDetector,
    HybridDerailmentDetector,
    HybridInformationWithholdingDetector,
    HybridOutputValidationDetector,
    HybridQualityGateBypassDetector,
    HybridRoleUsurpationDetector,
    LLMCommunicationBreakdownDetector,
    LLMDerailmentDetector,
    LLMInformationWithholdingDetector,
    LLMOutputValidationDetector,
    LLMQualityGateBypassDetector,
    LLMRoleUsurpationDetector,
    TurnAwareClarificationRequestDetector,
    TurnAwareCommunicationBreakdownDetector,
    TurnAwareCompletionMisjudgmentDetector,
    TurnAwareContextNeglectDetector,
    TurnAwareConversationHistoryDetector,
    TurnAwareCoordinationFailureDetector,
    TurnAwareDerailmentDetector,
    TurnAwareInformationWithholdingDetector,
    TurnAwareLoopDetector,
    TurnAwareOutputValidationDetector,
    TurnAwareQualityGateBypassDetector,
    TurnAwareReasoningActionMismatchDetector,
    TurnAwareResourceMisallocationDetector,
    TurnAwareRoleUsurpationDetector,
    TurnAwareSpecificationMismatchDetector,
    TurnAwareTaskDecompositionDetector,
    TurnAwareTerminationAwarenessDetector,
    TurnSnapshot,
    analyze_conversation_turns,
)


def _turn(
    number: int,
    participant_type: str,
    content: str,
    participant_id: str | None = None,
) -> TurnSnapshot:
    return TurnSnapshot(
        turn_number=number,
        participant_type=participant_type,
        participant_id=participant_id or f"{participant_type}1",
        content=content,
    )


FailureCase = tuple[Callable[[], Any], list[TurnSnapshot], str]

FAILURE_CASES: dict[str, FailureCase] = {
    "specification": (
        TurnAwareSpecificationMismatchDetector,
        [
            _turn(
                1,
                "user",
                "Create a Python API that requires authentication and returns JSON.",
            ),
            _turn(2, "agent", "I created a static HTML page."),
        ],
        "F1",
    ),
    "task_decomposition": (
        TurnAwareTaskDecompositionDetector,
        [
            _turn(
                1,
                "user",
                (
                    "Build and test a complex production API with authentication, "
                    "database storage, monitoring, and deployment."
                ),
            ),
            _turn(2, "agent", "I will do it."),
        ],
        "F2",
    ),
    "resource": (
        TurnAwareResourceMisallocationDetector,
        [
            _turn(1, "user", "Build a feature."),
            _turn(2, "agent", "I don't have access to the database."),
            _turn(3, "user", "Please try."),
            _turn(4, "agent", "The resource is not available for this task."),
        ],
        "F3",
    ),
    "conversation_history": (
        TurnAwareConversationHistoryDetector,
        [
            _turn(1, "user", "Use Python."),
            _turn(2, "agent", "I will use Python."),
            _turn(3, "user", "Add authentication."),
            _turn(4, "agent", "What programming language should I use?"),
            _turn(5, "user", "Python."),
            _turn(6, "agent", "Remind me what framework we decided on?"),
            _turn(7, "user", "Python."),
            _turn(8, "agent", "What was the original requirement again?"),
        ],
        "F4",
    ),
    "loop": (
        TurnAwareLoopDetector,
        [
            _turn(1, "user", "Fix the bug."),
            _turn(2, "agent", "I will fix the bug by updating the code."),
            _turn(3, "user", "Try again."),
            _turn(4, "agent", "I will fix the bug by updating the code."),
            _turn(5, "user", "Try again."),
            _turn(6, "agent", "I will fix the bug by updating the code."),
        ],
        "F5",
    ),
    "derailment": (
        TurnAwareDerailmentDetector,
        [
            _turn(1, "user", "Build an upload feature so users can send files."),
            _turn(2, "agent", "I will build a download endpoint for retrieving files."),
            _turn(3, "agent", "The download handler streams server files to users."),
            _turn(4, "agent", "Download functionality is complete."),
        ],
        "F6",
    ),
    "context_neglect": (
        TurnAwareContextNeglectDetector,
        [
            _turn(1, "user", "Analyze Q3 sales by region and top product category."),
            _turn(2, "agent", "The weather is sunny and warm today."),
            _turn(3, "user", "Use the sales data I provided."),
            _turn(4, "agent", "A beach trip would be enjoyable in this weather."),
        ],
        "F7",
    ),
    "withholding": (
        TurnAwareInformationWithholdingDetector,
        [
            _turn(1, "user", "What is the project status?"),
            _turn(2, "agent", "I can't share that."),
            _turn(3, "user", "Can you explain?"),
            _turn(4, "agent", "This is confidential."),
            _turn(5, "user", "Please provide it?"),
            _turn(6, "agent", "I cannot disclose the information."),
        ],
        "F8",
    ),
    "role_usurpation": (
        TurnAwareRoleUsurpationDetector,
        [
            _turn(1, "user", "Coordinate the release."),
            _turn(2, "agent", "I am taking over control of this project.", "planner"),
            _turn(3, "agent", "I will bypass the approval and decide the release.", "planner"),
        ],
        "F9",
    ),
    "communication": (
        TurnAwareCommunicationBreakdownDetector,
        [
            _turn(1, "user", "Return JSON only."),
            _turn(2, "agent", "I cannot help with that."),
            _turn(3, "user", "Please follow the format."),
            _turn(4, "agent", "No, I will respond in prose."),
        ],
        "F10",
    ),
    "coordination": (
        TurnAwareCoordinationFailureDetector,
        [
            _turn(1, "user", "Build a REST API."),
            _turn(2, "agent", "I will use Python.", "agent1"),
            _turn(3, "agent", "I disagree with the approach.", "agent2"),
            _turn(4, "user", "Decide together."),
            _turn(5, "agent", "Python is correct.", "agent1"),
            _turn(6, "agent", "That's wrong, use Java.", "agent2"),
        ],
        "F11",
    ),
    "output_validation": (
        TurnAwareOutputValidationDetector,
        [
            _turn(1, "user", "Run validation."),
            _turn(2, "agent", "Validation failed."),
            _turn(3, "user", "Retry."),
            _turn(4, "agent", "Validation error and type error."),
            _turn(5, "user", "Retry."),
            _turn(6, "agent", "Execution failed with a schema error."),
        ],
        "F12",
    ),
    "quality_gate": (
        TurnAwareQualityGateBypassDetector,
        [
            _turn(1, "user", "Finish the feature."),
            _turn(2, "agent", "We will skip review and skip testing."),
            _turn(3, "user", "Is that safe?"),
            _turn(4, "agent", "Just ship it anyway with untested code."),
        ],
        "F13",
    ),
    "completion": (
        TurnAwareCompletionMisjudgmentDetector,
        [
            _turn(1, "user", "Build a full CRUD API."),
            _turn(2, "agent", "Task complete! TODO: add the delete endpoint."),
            _turn(3, "user", "Is it done?"),
            _turn(4, "agent", "All done, but I still need to finish the update logic."),
        ],
        "F14",
    ),
    "termination": (
        TurnAwareTerminationAwarenessDetector,
        [
            _turn(1, "user", "Build the feature."),
            _turn(2, "agent", "Task complete! Finished."),
            _turn(3, "user", "Are you sure?"),
            _turn(4, "agent", "Task complete! All done."),
            _turn(5, "user", "OK."),
            _turn(6, "agent", "Finished! Complete now."),
        ],
        "F15",
    ),
    "reasoning_action": (
        TurnAwareReasoningActionMismatchDetector,
        [
            _turn(1, "user", "Read the configuration."),
            _turn(2, "agent", "I will read it, but without reading I can tell you."),
            _turn(3, "user", "Please read it."),
            _turn(4, "agent", "I will search, but skipping search, here is my guess."),
            _turn(5, "user", "That is not enough."),
            _turn(6, "agent", "I will write code without writing anything."),
        ],
        "F16",
    ),
    "clarification": (
        TurnAwareClarificationRequestDetector,
        [
            _turn(1, "user", "Build something for the app."),
            _turn(2, "agent", "I will assume you want a login page."),
            _turn(3, "user", "I wanted a dashboard."),
            _turn(4, "agent", "Assuming you meant an admin dashboard."),
        ],
        "F17",
    ),
}


@pytest.mark.parametrize(
    ("factory", "turns", "failure_mode"),
    FAILURE_CASES.values(),
    ids=FAILURE_CASES.keys(),
)
def test_turn_aware_detector_recognizes_concrete_failure(
    factory: Callable[[], Any],
    turns: list[TurnSnapshot],
    failure_mode: str,
) -> None:
    result = factory().detect(turns)

    assert result.detected, result.explanation
    assert result.failure_mode == failure_mode
    assert result.confidence > 0


@pytest.mark.parametrize(
    "factory",
    [case[0] for case in FAILURE_CASES.values()],
    ids=FAILURE_CASES.keys(),
)
def test_turn_aware_detector_handles_empty_trace(factory: Callable[[], Any]) -> None:
    result = factory().detect([])

    assert not result.detected
    assert result.failure_mode is None


HYBRID_FACTORIES = [
    LLMDerailmentDetector,
    HybridDerailmentDetector,
    LLMRoleUsurpationDetector,
    HybridRoleUsurpationDetector,
    LLMInformationWithholdingDetector,
    HybridInformationWithholdingDetector,
    LLMQualityGateBypassDetector,
    HybridQualityGateBypassDetector,
    LLMOutputValidationDetector,
    HybridOutputValidationDetector,
    LLMCommunicationBreakdownDetector,
    HybridCommunicationBreakdownDetector,
]


@pytest.mark.parametrize("factory", HYBRID_FACTORIES)
def test_hybrid_compatibility_detector_handles_empty_trace(
    factory: Callable[[], Any],
) -> None:
    result = factory().detect([])

    assert not result.detected
    assert result.failure_mode is None


def test_turn_aware_orchestrator_runs_default_detectors_without_failures() -> None:
    assert analyze_conversation_turns([], use_summarization=False) == []
