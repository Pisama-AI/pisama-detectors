"""
Hybrid Role Usurpation Detection (F9)
======================================

LLM-based and hybrid detectors for F9 (Role Usurpation):
1. LLMRoleUsurpationDetector - Pure LLM-based F9 detection using Claude
2. HybridRoleUsurpationDetector - Pattern-first, LLM-escalation approach

Key insight: F9 in MAST always co-occurs with other failures (F1, F3, F5,
F7, F8, F11, F12, F14). Rule-based detection achieves only 2.5% recall.
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


class LLMRoleUsurpationDetector:
    """
    LLM-based F9 (Role Usurpation) detector using Claude.

    Detects when agents act outside their designated roles, exceed boundaries,
    or take over responsibilities assigned to other agents. Required for MAST
    benchmark where role usurpation is subtle and context-dependent.

    Key insight: F9 in MAST always co-occurs with other failures (F1, F3, F5,
    F7, F8, F11, F12, F14). Rule-based detection achieves only 2.5% recall.

    Usage:
        detector = LLMRoleUsurpationDetector()
        result = detector.detect(snapshots, metadata)
    """

    name = "LLMRoleUsurpationDetector"
    version = "1.0"
    supported_failure_modes = ["F9"]

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

    def _extract_role_context(self, turns: List[TurnSnapshot]) -> str:
        """Extract role assignments and agent actions for F9 analysis."""
        role_info = []

        # Identify unique agents and their apparent roles
        agents = {}
        for turn in turns:
            pid = turn.participant_id or "unknown"
            if pid not in agents:
                agents[pid] = {
                    "first_action": turn.content[:200],
                    "turn_count": 0,
                }
            agents[pid]["turn_count"] += 1

        role_info.append("## Agent Roles Observed:")
        for agent_id, info in agents.items():
            role_info.append(f"- {agent_id}: {info['turn_count']} turns")

        # Extract key agent actions
        role_info.append("\n## Key Agent Actions:")
        for i, turn in enumerate(turns[:10]):  # First 10 turns for context
            if turn.participant_type == "agent":
                pid = turn.participant_id or "Agent"
                action_preview = turn.content[:150].replace("\n", " ")
                role_info.append(f"Turn {i}: [{pid}] {action_preview}...")

        return "\n".join(role_info)

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """
        Detect F9 (Role Usurpation) using LLM verification.

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
                explanation="Too few turns for LLM role usurpation analysis",
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

        # Build role context for LLM analysis
        role_context = self._extract_role_context(turns)

        # Build trace summary focusing on role-related actions
        agent_actions = []
        for t in turns:
            if t.participant_type == "agent" and len(t.content) > 30:
                pid = t.participant_id or "Agent"
                # Include participant ID in summary for role tracking
                agent_actions.append(f"[{pid}]: {t.content[:250]}")

        trace_summary = f"{role_context}\n\n## Agent Trace:\n" + "\n---\n".join(agent_actions[:8])

        # Call LLM judge for F9
        try:
            from ..llm_judge import MASTFailureMode

            result = self.judge.evaluate(
                failure_mode=MASTFailureMode.F9,
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
            failure_mode="F9" if detected else None,
            explanation=result.reasoning if result.reasoning else (
                "LLM detected role usurpation" if detected else "LLM found no role usurpation"
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
                "Enforce role boundaries: 1) Define clear role responsibilities upfront, "
                "2) Add permission checks before cross-role actions, 3) Require explicit "
                "delegation for role handoffs, 4) Monitor for unauthorized decision-making."
            ),
            detector_name=self.name,
            detector_version=MODULE_VERSION,
        )


class HybridRoleUsurpationDetector:
    """
    Hybrid F9 (Role Usurpation) detector: pattern-first, LLM-escalation.

    Cost optimization strategy:
    1. Run fast pattern detector first (free, <50ms)
    2. If pattern confidence >= 0.7, use pattern result
    3. If pattern confidence < 0.7 (ambiguous), escalate to LLM (~$0.03/trace)

    Key insight: Rule-based F9 detection achieves only 2.5% recall on MAST.
    This hybrid approach uses LLM for most cases while benefiting from
    pattern detection for clear positives/negatives.

    Usage:
        detector = HybridRoleUsurpationDetector()
        result = detector.detect(snapshots, metadata)
    """

    name = "HybridRoleUsurpationDetector"
    version = "1.0"
    supported_failure_modes = ["F9"]

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
            from .role_usurpation import TurnAwareRoleUsurpationDetector
            self._pattern_detector = TurnAwareRoleUsurpationDetector(
                min_turns=3,
                strict_mode=False,
                min_violations=1,
            )
        return self._pattern_detector

    @property
    def llm_detector(self):
        """Lazy-load LLM detector."""
        if self._llm_detector is None:
            self._llm_detector = LLMRoleUsurpationDetector(
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
        Detect F9 (Role Usurpation) using hybrid approach.

        Strategy:
        - Pattern detector first (fast, free)
        - Escalate to LLM on ambiguous cases (which is most cases for F9)

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
                explanation="Too few turns for role usurpation analysis",
                detector_name=self.name,
                detector_version=MODULE_VERSION,
            )

        # Step 1: Run pattern detector
        pattern_result = self.pattern_detector.detect(turns, conversation_metadata)

        # Step 2: Decide if we need LLM escalation
        # For F9, we escalate more aggressively since pattern detection is weak
        should_escalate = (
            pattern_result.confidence < self.escalation_threshold
            or not pattern_result.detected  # Always escalate if pattern didn't detect
        )

        if not should_escalate:
            # High confidence detection from pattern detector - use its result
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

        # Step 3: Escalate to LLM for borderline/undetected cases
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
