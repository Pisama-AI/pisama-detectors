"""
Turn-Aware Detection Package
============================

Provides turn-aware detection algorithms that analyze entire conversation traces
rather than single states. Designed for MAST-Data and similar multi-turn
conversation benchmarks.

Key differences from state-based detectors:
1. Analyze accumulated context across turns
2. Track topic/intent drift over conversation
3. Detect patterns that emerge across multiple turns
4. Support participant-aware analysis (user vs agent vs tool)

Version History:
- v1.0: Initial implementation with turn-aware context neglect and derailment
- v1.1: Added sliding window support for long conversations
- v2.0: Refactored to package structure with lazy loading
"""

# Base classes (always loaded - they're small)
from ._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
    MODULE_VERSION,
    MAX_TURNS_BEFORE_SUMMARIZATION,
    MAX_TOKENS_BEFORE_SUMMARIZATION,
    EMBEDDING_SIMILARITY_THRESHOLD,
)
from ._embedding_mixin import (
    EmbeddingMixin,
    _check_embedding_available,
)

# Detector module mapping for lazy loading
_DETECTOR_MODULES = {
    'TurnAwareContextNeglectDetector': '.context_neglect',
    'TurnAwareDerailmentDetector': '.derailment',
    'TurnAwareLoopDetector': '.loop',
    'TurnAwareSpecificationMismatchDetector': '.specification',
    'TurnAwareOutputValidationDetector': '.output_validation',
    'TurnAwareQualityGateBypassDetector': '.quality_gate',
    'TurnAwareCoordinationFailureDetector': '.coordination',
    'TurnAwareCommunicationBreakdownDetector': '.communication',
    'TurnAwareResourceMisallocationDetector': '.resource',
    'TurnAwareRoleUsurpationDetector': '.role_usurpation',
    'TurnAwareInformationWithholdingDetector': '.withholding',
    'TurnAwareCompletionMisjudgmentDetector': '.completion',
    'TurnAwareTerminationAwarenessDetector': '.termination',
    'TurnAwareTaskDecompositionDetector': '.task_decomposition',
    'TurnAwareConversationHistoryDetector': '.conversation',
    'TurnAwareReasoningActionMismatchDetector': '.reasoning_action',
    'TurnAwareClarificationRequestDetector': '.clarification',
    'LLMDerailmentDetector': '.hybrid_derailment',
    'HybridDerailmentDetector': '.hybrid_derailment',
    'LLMRoleUsurpationDetector': '.hybrid_role_usurpation',
    'HybridRoleUsurpationDetector': '.hybrid_role_usurpation',
    'LLMInformationWithholdingDetector': '.hybrid_withholding',
    'HybridInformationWithholdingDetector': '.hybrid_withholding',
    'LLMQualityGateBypassDetector': '.hybrid_quality_gate',
    'HybridQualityGateBypassDetector': '.hybrid_quality_gate',
    'LLMOutputValidationDetector': '.hybrid_output_validation',
    'HybridOutputValidationDetector': '.hybrid_output_validation',
    'LLMCommunicationBreakdownDetector': '.hybrid_communication',
    'HybridCommunicationBreakdownDetector': '.hybrid_communication',
    'analyze_conversation_turns': '.hybrid_orchestrator',
}


def __getattr__(name):
    """Lazy load detectors on first access."""
    if name in _DETECTOR_MODULES:
        import importlib
        module = importlib.import_module(_DETECTOR_MODULES[name], __package__)
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Return all available names for tab completion."""
    return list(__all__)


# For backward compatibility with `from turn_aware import *`
__all__ = [
    # Base classes
    'TurnSnapshot',
    'TurnAwareDetector',
    'TurnAwareDetectionResult',
    'TurnAwareSeverity',
    'EmbeddingMixin',
    # Constants
    'MODULE_VERSION',
    'MAX_TURNS_BEFORE_SUMMARIZATION',
    'MAX_TOKENS_BEFORE_SUMMARIZATION',
    'EMBEDDING_SIMILARITY_THRESHOLD',
    # Helper functions
    '_check_embedding_available',
    # Detectors (lazy loaded)
    'TurnAwareContextNeglectDetector',
    'TurnAwareDerailmentDetector',
    'TurnAwareLoopDetector',
    'TurnAwareSpecificationMismatchDetector',
    'TurnAwareOutputValidationDetector',
    'TurnAwareQualityGateBypassDetector',
    'TurnAwareCoordinationFailureDetector',
    'TurnAwareCommunicationBreakdownDetector',
    'TurnAwareResourceMisallocationDetector',
    'TurnAwareRoleUsurpationDetector',
    'TurnAwareInformationWithholdingDetector',
    'TurnAwareCompletionMisjudgmentDetector',
    'TurnAwareTerminationAwarenessDetector',
    'TurnAwareTaskDecompositionDetector',
    'TurnAwareConversationHistoryDetector',
    'TurnAwareReasoningActionMismatchDetector',
    'TurnAwareClarificationRequestDetector',
    'LLMDerailmentDetector',
    'HybridDerailmentDetector',
    'LLMInformationWithholdingDetector',
    'HybridInformationWithholdingDetector',
    'LLMRoleUsurpationDetector',
    'HybridRoleUsurpationDetector',
    'LLMQualityGateBypassDetector',
    'HybridQualityGateBypassDetector',
    'LLMOutputValidationDetector',
    'HybridOutputValidationDetector',
    'LLMCommunicationBreakdownDetector',
    'HybridCommunicationBreakdownDetector',
    'analyze_conversation_turns',
]
