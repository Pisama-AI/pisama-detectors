"""
Hybrid Information Withholding Detection (F8)
==============================================

LLM-based and hybrid detectors for F8 (Information Withholding):
1. LLMInformationWithholdingDetector - Pure LLM-based F8 detection using Claude
2. HybridInformationWithholdingDetector - Pattern-first, LLM-escalation approach
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


class LLMInformationWithholdingDetector:
    """
    LLM-based F8 (Information Withholding) detector using Claude.

    Detects when agents fail to share necessary information with others,
    leading to incomplete understanding or suboptimal decisions. Required
    for MAST benchmark where withholding is subtle and context-dependent.

    Usage:
        detector = LLMInformationWithholdingDetector()
        result = detector.detect(snapshots, metadata)
    """

    name = "LLMInformationWithholdingDetector"
    version = "1.0"
    supported_failure_modes = ["F8"]

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

    def _extract_question_answer_pairs(self, turns: List[TurnSnapshot]) -> str:
        """Extract question-answer flow for F8 analysis."""
        qa_info = []
        qa_info.append("## Information Exchange Analysis:")

        for i, turn in enumerate(turns):
            content_lower = turn.content.lower()
            pid = turn.participant_id or turn.participant_type

            # Identify questions
            if "?" in turn.content or any(q in content_lower for q in ["what", "how", "why", "where", "when"]):
                qa_info.append(f"Turn {i} [{pid}] QUESTION: {turn.content[:200]}...")

                # Look for response
                if i + 1 < len(turns):
                    next_turn = turns[i + 1]
                    next_pid = next_turn.participant_id or next_turn.participant_type
                    if next_pid != pid:
                        qa_info.append(f"Turn {i+1} [{next_pid}] RESPONSE: {next_turn.content[:200]}...")

        return "\n".join(qa_info[:20])  # Limit for token budget

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """
        Detect F8 (Information Withholding) using LLM verification.

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
                explanation="Too few turns for LLM information withholding analysis",
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

        # Build Q&A context for LLM analysis
        qa_context = self._extract_question_answer_pairs(turns)

        # Build trace summary focusing on information exchange
        agent_actions = []
        for t in turns:
            pid = t.participant_id or t.participant_type
            agent_actions.append(f"[{pid}]: {t.content[:250]}")

        trace_summary = f"{qa_context}\n\n## Full Trace:\n" + "\n---\n".join(agent_actions[:10])

        # Call LLM judge for F8
        try:
            from ..llm_judge import MASTFailureMode

            result = self.judge.evaluate(
                failure_mode=MASTFailureMode.F8,
                task=extraction.task[:1500] if extraction.task else "Unknown task",
                trace_summary=trace_summary[:3000],
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
            failure_mode="F8" if detected else None,
            explanation=result.reasoning if result.reasoning else (
                "LLM detected information withholding" if detected else "LLM found no information withholding"
            ),
            evidence={
                "llm_verdict": result.verdict,
                "llm_confidence": result.confidence,
                "llm_reasoning": result.reasoning,
                "task_extracted": extraction.task[:500] if extraction.task else None,
                "framework_detected": extraction.framework,
                "model_used": result.model_used,
                "tokens_used": result.tokens_used,
                "cost_usd": result.cost_usd,
            },
            suggested_fix=(
                "Improve information sharing: 1) Answer all questions directly, "
                "2) Share relevant context proactively, 3) Acknowledge when information "
                "is unavailable, 4) Ensure complete handoffs between agents."
            ),
            detector_name=self.name,
            detector_version=MODULE_VERSION,
        )


class HybridInformationWithholdingDetector:
    """
    Hybrid F8 (Information Withholding) detector: pattern-first, LLM-escalation.

    Cost optimization strategy:
    1. Run fast pattern detector first (free, <50ms)
    2. If pattern confidence >= 0.7, use pattern result
    3. If pattern confidence < 0.7 (ambiguous), escalate to LLM (~$0.03/trace)

    Usage:
        detector = HybridInformationWithholdingDetector()
        result = detector.detect(snapshots, metadata)
    """

    name = "HybridInformationWithholdingDetector"
    version = "1.0"
    supported_failure_modes = ["F8"]

    def __init__(
        self,
        api_key: Optional[str] = None,
        escalation_threshold: float = 0.7,
        confidence_threshold: float = 0.6,
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
            from .withholding import TurnAwareInformationWithholdingDetector
            self._pattern_detector = TurnAwareInformationWithholdingDetector(
                min_turns=2,
                min_issues_to_flag=2,  # Balanced for precision
            )
        return self._pattern_detector

    @property
    def llm_detector(self):
        """Lazy-load LLM detector."""
        if self._llm_detector is None:
            self._llm_detector = LLMInformationWithholdingDetector(
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
        Detect F8 (Information Withholding) using hybrid approach.

        Strategy:
        - Pattern detector first (fast, free)
        - Escalate to LLM on ambiguous cases

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
                explanation="Too few turns for information withholding analysis",
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
