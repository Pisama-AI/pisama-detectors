"""
Hybrid Derailment Detection (F6)
=================================

LLM-based and hybrid detectors for F6 (Task Derailment):
1. LLMDerailmentDetector - Pure LLM-based F6 detection using Claude
2. HybridDerailmentDetector - Pattern-first, LLM-escalation approach

The hybrid approach achieves ~80% cost savings vs LLM-only while maintaining accuracy.
"""

import logging
from typing import List, Optional, Dict, Any, TYPE_CHECKING

from ._base import (
    TurnSnapshot,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
    MODULE_VERSION,
)

if TYPE_CHECKING:
    from ..task_extractors import ConversationTurn

logger = logging.getLogger(__name__)


class LLMDerailmentDetector:
    """
    LLM-based F6 (Task Derailment) detector using Claude Opus 4.5.

    Unlike the pattern-based TurnAwareDerailmentDetector which uses keyword drift,
    this detector uses semantic understanding to detect when agents diverge from
    the original task. Required for MAST benchmark where agents stay on-topic
    but derail semantically.

    Usage:
        detector = LLMDerailmentDetector()
        result = detector.detect(snapshots, metadata)
    """

    name = "LLMDerailmentDetector"
    version = "1.0"
    supported_failure_modes = ["F6"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        confidence_threshold: float = 0.6,
    ):
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        self._judge = None

    @property
    def judge(self):
        """Lazy-load the LLM judge."""
        if self._judge is None:
            from ..llm_judge import MASTLLMJudge
            self._judge = MASTLLMJudge(api_key=self.api_key)
        return self._judge

    def _convert_to_conversation_turns(
        self,
        snapshots: List[TurnSnapshot],
    ) -> List["ConversationTurn"]:
        """Convert TurnSnapshots to ConversationTurns for task extraction."""
        from ..task_extractors import ConversationTurn

        turns = []
        for snapshot in snapshots:
            turns.append(ConversationTurn(
                role=snapshot.participant_type,
                content=snapshot.content,
                participant_id=snapshot.participant_id,
                metadata=snapshot.turn_metadata or {},
            ))
        return turns

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """
        Detect F6 (Task Derailment) using LLM verification.

        Args:
            turns: List of turn snapshots
            conversation_metadata: Optional metadata about the conversation

        Returns:
            TurnAwareDetectionResult with LLM-based detection
        """
        if len(turns) < 3:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few turns for LLM derailment analysis",
                detector_name=self.name,
                detector_version=MODULE_VERSION,
            )

        metadata = conversation_metadata or {}

        # Convert to ConversationTurns and extract task
        conv_turns = self._convert_to_conversation_turns(turns)

        try:
            from ..task_extractors import extract_task
            extraction = extract_task(conv_turns, metadata)
        except Exception as e:
            logger.warning(f"Task extraction failed: {e}")
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation=f"Task extraction failed: {e}",
                detector_name=self.name,
                detector_version=MODULE_VERSION,
            )

        if not extraction.task:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No task found in conversation",
                detector_name=self.name,
                detector_version=MODULE_VERSION,
            )

        # Enhance agent output summary if it's too sparse
        # MAST trajectories often have sparse parsed content but rich raw data
        agent_summary = extraction.agent_output_summary
        if len(agent_summary) < 200:
            # Build summary from actual turn content
            agent_content = []
            for t in turns:
                if t.participant_type == "agent" and len(t.content) > 30:
                    # Skip metadata-only content
                    if not t.content.startswith("[") or len(t.content) > 100:
                        agent_content.append(t.content[:300])
            if agent_content:
                agent_summary = "\n---\n".join(agent_content[:5])

        # Call LLM judge
        try:
            from ..llm_judge import MASTFailureMode

            result = self.judge.evaluate(
                failure_mode=MASTFailureMode.F6,
                task=extraction.task[:2000],  # Truncate for token limits
                trace_summary=agent_summary[:2000],  # Use enhanced summary
                key_events=extraction.key_events,
            )
        except Exception as e:
            logger.error(f"LLM judge call failed: {e}")
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation=f"LLM verification failed: {e}",
                detector_name=self.name,
                detector_version=MODULE_VERSION,
            )

        # Convert LLM verdict to detection result
        detected = result.verdict == "YES" and result.confidence >= self.confidence_threshold

        if detected:
            if result.confidence >= 0.85:
                severity = TurnAwareSeverity.SEVERE
            elif result.confidence >= 0.7:
                severity = TurnAwareSeverity.MODERATE
            else:
                severity = TurnAwareSeverity.MINOR
        else:
            severity = TurnAwareSeverity.NONE

        return TurnAwareDetectionResult(
            detected=detected,
            severity=severity,
            confidence=result.confidence,
            failure_mode="F6" if detected else None,
            explanation=result.reasoning if result.reasoning else (
                "LLM detected task derailment" if detected else "LLM found no task derailment"
            ),
            evidence={
                "llm_verdict": result.verdict,
                "llm_confidence": result.confidence,
                "llm_reasoning": result.reasoning,
                "task_extracted": extraction.task[:500],
                "framework_detected": extraction.framework,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "cost_usd": result.cost_usd,
            },
            suggested_fix=(
                "Review agent focus and add task reminders. Consider: "
                "'Stay focused on: [ORIGINAL_TASK]. Do not address unrelated topics.'"
            ),
            detector_name=self.name,
            detector_version=MODULE_VERSION,
        )


class HybridDerailmentDetector:
    """
    Hybrid F6 (Task Derailment) detector: pattern-first, LLM-escalation.

    Cost optimization strategy:
    1. Run fast pattern detector first (free, <50ms)
    2. If pattern confidence >= 0.7, use pattern result (clear detection or clear negative)
    3. If pattern confidence < 0.7 (ambiguous), escalate to LLM (~$0.03/trace)

    This achieves ~80% cost savings vs LLM-only while maintaining accuracy.

    Usage:
        detector = HybridDerailmentDetector()
        result = detector.detect(snapshots, metadata)
    """

    name = "HybridDerailmentDetector"
    version = "1.0"
    supported_failure_modes = ["F6"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        escalation_threshold: float = 0.7,  # Below this confidence, escalate to LLM
        confidence_threshold: float = 0.6,  # Min confidence for LLM to detect
    ):
        self.api_key = api_key
        self.escalation_threshold = escalation_threshold
        self.confidence_threshold = confidence_threshold
        self._pattern_detector = None
        self._llm_detector = None

    @property
    def pattern_detector(self):
        """Lazy-load pattern detector."""
        if self._pattern_detector is None:
            from .derailment import TurnAwareDerailmentDetector
            self._pattern_detector = TurnAwareDerailmentDetector(
                drift_threshold=0.5,  # Sensitive for initial detection
                require_strong_evidence=False,
            )
        return self._pattern_detector

    @property
    def llm_detector(self):
        """Lazy-load LLM detector."""
        if self._llm_detector is None:
            self._llm_detector = LLMDerailmentDetector(
                api_key=self.api_key,
                confidence_threshold=self.confidence_threshold,
            )
        return self._llm_detector

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """
        Detect F6 (Task Derailment) using hybrid approach.

        Strategy:
        - Pattern detector first (fast, free)
        - Escalate to LLM only on ambiguous cases

        Args:
            turns: List of turn snapshots
            conversation_metadata: Optional metadata about the conversation

        Returns:
            TurnAwareDetectionResult with detection decision
        """
        if len(turns) < 3:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few turns for derailment analysis",
                detector_name=self.name,
                detector_version=MODULE_VERSION,
            )

        # Step 1: Run pattern detector
        pattern_result = self.pattern_detector.detect(turns, conversation_metadata)

        # Step 2: Decide if we need LLM escalation
        should_escalate = (
            pattern_result.confidence < self.escalation_threshold
            or (pattern_result.detected and pattern_result.confidence < 0.8)
        )

        if not should_escalate:
            # High confidence from pattern detector - use its result
            return TurnAwareDetectionResult(
                detected=pattern_result.detected,
                severity=pattern_result.severity,
                confidence=pattern_result.confidence,
                failure_mode=pattern_result.failure_mode,
                explanation=f"[Pattern] {pattern_result.explanation}",
                affected_turns=pattern_result.affected_turns,
                evidence={
                    **(pattern_result.evidence or {}),
                    "detection_method": "pattern",
                    "llm_escalated": False,
                },
                suggested_fix=pattern_result.suggested_fix,
                detector_name=self.name,
                detector_version=MODULE_VERSION,
            )

        # Step 3: Escalate to LLM for borderline cases
        try:
            llm_result = self.llm_detector.detect(turns, conversation_metadata)

            # Combine evidence from both detectors
            combined_evidence = {
                **(pattern_result.evidence or {}),
                **(llm_result.evidence or {}),
                "detection_method": "hybrid",
                "llm_escalated": True,
                "pattern_confidence": pattern_result.confidence,
                "pattern_detected": pattern_result.detected,
            }

            return TurnAwareDetectionResult(
                detected=llm_result.detected,
                severity=llm_result.severity,
                confidence=llm_result.confidence,
                failure_mode=llm_result.failure_mode,
                explanation=f"[LLM-verified] {llm_result.explanation}",
                affected_turns=llm_result.affected_turns or pattern_result.affected_turns,
                evidence=combined_evidence,
                suggested_fix=llm_result.suggested_fix,
                detector_name=self.name,
                detector_version=MODULE_VERSION,
            )

        except Exception as e:
            # LLM failed - fall back to pattern result
            logger.warning(f"LLM escalation failed, using pattern result: {e}")
            return TurnAwareDetectionResult(
                detected=pattern_result.detected,
                severity=pattern_result.severity,
                confidence=pattern_result.confidence,
                failure_mode=pattern_result.failure_mode,
                explanation=f"[Pattern-fallback] {pattern_result.explanation}",
                affected_turns=pattern_result.affected_turns,
                evidence={
                    **(pattern_result.evidence or {}),
                    "detection_method": "pattern_fallback",
                    "llm_escalated": True,
                    "llm_error": str(e),
                },
                suggested_fix=pattern_result.suggested_fix,
                detector_name=self.name,
                detector_version=MODULE_VERSION,
            )
