"""
Hybrid Orchestrator
====================

Convenience function to run all turn-aware detectors on a conversation.
"""

import logging
from typing import List, Optional, Dict, Any

from ._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    MAX_TURNS_BEFORE_SUMMARIZATION,
    MAX_TOKENS_BEFORE_SUMMARIZATION,
)

logger = logging.getLogger(__name__)


# Convenience function to run all turn-aware detectors
def analyze_conversation_turns(
    turns: List[TurnSnapshot],
    conversation_metadata: Optional[Dict[str, Any]] = None,
    detectors: Optional[List[TurnAwareDetector]] = None,
    use_summarization: bool = True,
) -> List[TurnAwareDetectionResult]:
    """Run multiple turn-aware detectors on a conversation.

    Args:
        turns: List of conversation turns
        conversation_metadata: Optional metadata
        detectors: Optional list of detectors to run. If None, runs all defaults.
        use_summarization: Whether to use summarization for long conversations

    Returns:
        List of detection results (only those with detected=True)
    """
    if detectors is None:
        # Import all detector classes
        from .specification import TurnAwareSpecificationMismatchDetector
        from .task_decomposition import TurnAwareTaskDecompositionDetector
        from .resource import TurnAwareResourceMisallocationDetector
        from .conversation import TurnAwareConversationHistoryDetector
        from .loop import TurnAwareLoopDetector
        from .derailment import TurnAwareDerailmentDetector
        from .context_neglect import TurnAwareContextNeglectDetector
        from .withholding import TurnAwareInformationWithholdingDetector
        from .role_usurpation import TurnAwareRoleUsurpationDetector
        from .communication import TurnAwareCommunicationBreakdownDetector
        from .coordination import TurnAwareCoordinationFailureDetector
        from .output_validation import TurnAwareOutputValidationDetector
        from .quality_gate import TurnAwareQualityGateBypassDetector
        from .completion import TurnAwareCompletionMisjudgmentDetector
        from .termination import TurnAwareTerminationAwarenessDetector
        from .reasoning_action import TurnAwareReasoningActionMismatchDetector
        from .clarification import TurnAwareClarificationRequestDetector

        detectors = [
            TurnAwareSpecificationMismatchDetector(),  # F1
            TurnAwareTaskDecompositionDetector(),  # F2
            TurnAwareResourceMisallocationDetector(),  # F3
            TurnAwareConversationHistoryDetector(),  # F4
            TurnAwareLoopDetector(),  # F5
            TurnAwareDerailmentDetector(),  # F6
            TurnAwareContextNeglectDetector(),  # F7
            TurnAwareInformationWithholdingDetector(),  # F8
            TurnAwareRoleUsurpationDetector(),  # F9
            TurnAwareCommunicationBreakdownDetector(),  # F10
            TurnAwareCoordinationFailureDetector(),  # F11
            TurnAwareOutputValidationDetector(),  # F12
            TurnAwareQualityGateBypassDetector(),  # F13
            TurnAwareCompletionMisjudgmentDetector(),  # F14
            # New MAST-aligned detectors (NeurIPS 2025)
            TurnAwareTerminationAwarenessDetector(),  # F15 (FM-1.5, 40% of FC1)
            TurnAwareReasoningActionMismatchDetector(),  # F16 (FM-2.6, 26% of FC2)
            TurnAwareClarificationRequestDetector(),  # F17 (FM-2.2, 18% of FC2)
        ]

    # Check if conversation is long enough to need summarization
    working_turns = turns
    summarization_applied = False

    if use_summarization and len(turns) > MAX_TURNS_BEFORE_SUMMARIZATION:
        try:
            from app.core.summarizer import SlidingWindowManager, count_tokens

            # Check total tokens
            total_content = " ".join(t.content for t in turns)
            total_tokens = count_tokens(total_content)

            if total_tokens > MAX_TOKENS_BEFORE_SUMMARIZATION:
                logger.info(
                    f"Long conversation detected ({len(turns)} turns, ~{total_tokens} tokens). "
                    "Using sliding window for detection."
                )

                # For long conversations, we analyze in chunks
                # and aggregate results
                window_manager = SlidingWindowManager()

                # Convert TurnSnapshots to dicts for the window manager
                turn_dicts = [
                    {
                        "turn_number": t.turn_number,
                        "role": t.participant_type,
                        "participant_id": t.participant_id,
                        "content": t.content,
                        "content_hash": t.content_hash,
                    }
                    for t in turns
                ]

                # Get chunks for batch detection
                chunks = window_manager.chunk_for_batch_detection(turn_dicts)
                summarization_applied = True

                # For now, we'll just use the last chunk (most recent context)
                # Future enhancement: aggregate results from all chunks
                if chunks:
                    # Get turns covered by the last chunk
                    last_chunk = chunks[-1]
                    start_turn, end_turn = last_chunk.recent_turns

                    # Filter to recent turns
                    working_turns = [
                        t for t in turns
                        if start_turn <= t.turn_number <= end_turn
                    ]

                    # Always include first turn (task) if not already
                    first_turn = next((t for t in turns if t.turn_number == 1), None)
                    if first_turn and first_turn not in working_turns:
                        working_turns = [first_turn] + working_turns

        except ImportError as e:
            logger.warning(f"Summarizer not available, using full conversation: {e}")
        except Exception as e:
            logger.warning(f"Summarization failed, using full conversation: {e}")

    results = []
    for detector in detectors:
        try:
            result = detector.detect(working_turns, conversation_metadata)
            if result.detected:
                # Add summarization info to evidence if applicable
                if summarization_applied:
                    result.evidence["summarization_applied"] = True
                    result.evidence["original_turns"] = len(turns)
                    result.evidence["analyzed_turns"] = len(working_turns)
                results.append(result)
        except Exception as e:
            logger.error(f"Detector {detector.name} failed: {e}")

    return results
