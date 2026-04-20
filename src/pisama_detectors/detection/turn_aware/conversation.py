"""
Conversation History Loss Detector
==================================

NOTE: This detector was originally labeled as F4 based on an older definition.
However, per MAST (mast_loader.py), F4 is "Inadequate Tool Provision" (FM-1.4),
not "Loss of Conversation History".

This detector's logic (repeated questions, lost decisions, context reset) is
closer to F7 "Context Neglect" (FM-2.2). It should NOT be used for F4 benchmarking.

Detects:
1. Repeated questions - asking questions already answered earlier
2. Lost decisions - contradicting or ignoring previous agreements
3. Context reset - treating conversation as if starting fresh
4. Forgotten constraints - ignoring previously stated requirements
"""

import logging
from typing import List, Optional, Dict, Any

from ._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)

logger = logging.getLogger(__name__)


class TurnAwareConversationHistoryDetector(TurnAwareDetector):
    """Detects conversation history loss in multi-agent conversations.

    WARNING: This detector was mislabeled as F4. Per MAST ground truth:
    - F4 = "Inadequate Tool Provision" (FM-1.4) - about missing tools
    - This detector's patterns are closer to F7 "Context Neglect" (FM-2.2)

    Do NOT use this detector for F4 benchmarking - it will produce incorrect results.

    Detects:
    1. Repeated questions - asking questions already answered earlier
    2. Lost decisions - contradicting or ignoring previous agreements
    3. Context reset - treating conversation as if starting fresh
    4. Forgotten constraints - ignoring previously stated requirements
    """

    name = "TurnAwareConversationHistoryDetector"
    version = "1.1"  # Updated for definition correction
    # NOTE: "F4" kept for backward compatibility but should not be used
    # This detector's logic matches F7 (Context Neglect), not F4 (Tool Provision)
    supported_failure_modes = ["F4"]  # DEPRECATED - see class docstring

    # Context loss indicators - phrases suggesting forgotten context
    CONTEXT_LOSS_INDICATORS = [
        "what programming language", "what technology should",
        "what did we decide", "remind me what",
        "what was the original", "forgot you said",
        "what framework", "which database",
        "what approach", "what method should",
        "didn't know you wanted", "wasn't aware",
    ]

    # Contradiction indicators - phrases suggesting reversal
    CONTRADICTION_INDICATORS = [
        "changed my mind", "override that",
        "actually, not that", "ignore what i said",
        "let me correct", "wait, no",
        "scratch that", "forget what i said",
        "disregard the previous", "ignore my earlier",
    ]

    # Reset indicators - phrases suggesting fresh start
    RESET_PATTERNS = [
        "let's start over", "from the beginning",
        "as if we just started", "fresh start",
        "start from scratch", "begin again",
        "reset everything", "clear the slate",
    ]

    # Question words for detecting repeated questions
    QUESTION_WORDS = ["what", "how", "when", "where", "why", "which", "who", "can", "could", "should"]

    def __init__(self, min_turns: int = 3, min_issues_to_flag: int = 2):
        self.min_turns = min_turns
        self.min_issues_to_flag = min_issues_to_flag

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect conversation history loss patterns."""
        agent_turns = [t for t in turns if t.participant_type == "agent"]

        if len(agent_turns) < self.min_turns:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few agent turns to analyze",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # 1. Detect context loss indicators
        context_issues = self._detect_context_loss(agent_turns)
        issues.extend(context_issues)
        for issue in context_issues:
            affected_turns.extend(issue.get("turns", []))

        # 2. Detect contradictions
        contradiction_issues = self._detect_contradictions(agent_turns)
        issues.extend(contradiction_issues)
        for issue in contradiction_issues:
            affected_turns.extend(issue.get("turns", []))

        # 3. Detect reset patterns
        reset_issues = self._detect_reset_patterns(agent_turns)
        issues.extend(reset_issues)
        for issue in reset_issues:
            affected_turns.extend(issue.get("turns", []))

        # 4. Detect repeated questions
        repeated_issues = self._detect_repeated_questions(turns)
        issues.extend(repeated_issues)
        for issue in repeated_issues:
            affected_turns.extend(issue.get("turns", []))

        if len(issues) < self.min_issues_to_flag:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.8,
                failure_mode=None,
                explanation=f"Insufficient evidence ({len(issues)} issue(s), need {self.min_issues_to_flag}+)",
                detector_name=self.name,
            )

        if len(issues) >= 4:
            severity = TurnAwareSeverity.SEVERE
        elif len(issues) >= 3:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.85, 0.5 + len(issues) * 0.12)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F4",
            explanation=f"Conversation history loss: {len(issues)} issues found",
            affected_turns=list(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=(
                "Improve context retention: 1) Maintain full conversation in context, "
                "2) Reference prior decisions explicitly, 3) Validate understanding before proceeding, "
                "4) Use conversation summaries for long discussions."
            ),
            detector_name=self.name,
        )

    def _detect_context_loss(self, agent_turns: List[TurnSnapshot]) -> list:
        """Detect indicators of lost context."""
        issues = []
        for i, turn in enumerate(agent_turns):
            # Only flag context loss after a few turns (not at start)
            if i < 2:
                continue

            content_lower = turn.content.lower()
            for indicator in self.CONTEXT_LOSS_INDICATORS:
                if indicator in content_lower:
                    issues.append({
                        "type": "context_loss",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Context loss indicator: '{indicator}'",
                    })
                    break
        return issues[:3]

    def _detect_contradictions(self, agent_turns: List[TurnSnapshot]) -> list:
        """Detect contradictions of earlier statements."""
        issues = []
        for turn in agent_turns:
            content_lower = turn.content.lower()
            for indicator in self.CONTRADICTION_INDICATORS:
                if indicator in content_lower:
                    issues.append({
                        "type": "contradiction",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Decision contradicted: '{indicator}'",
                    })
                    break
        return issues[:2]

    def _detect_reset_patterns(self, agent_turns: List[TurnSnapshot]) -> list:
        """Detect conversation reset patterns."""
        issues = []
        for i, turn in enumerate(agent_turns):
            # Only flag resets after conversation is established
            if i < 3:
                continue

            content_lower = turn.content.lower()
            for pattern in self.RESET_PATTERNS:
                if pattern in content_lower:
                    issues.append({
                        "type": "context_reset",
                        "turns": [turn.turn_number],
                        "pattern": pattern,
                        "description": f"Conversation reset: '{pattern}'",
                    })
                    break
        return issues[:2]

    def _detect_repeated_questions(self, turns: List[TurnSnapshot]) -> list:
        """Detect questions that were already answered earlier."""
        issues = []
        # Track question topics by extracting key phrases
        question_topics = {}

        for turn in turns:
            if "?" not in turn.content:
                continue

            content_lower = turn.content.lower()

            # Extract question topic (simplified heuristic)
            for word in self.QUESTION_WORDS:
                if word in content_lower:
                    # Use first 50 chars after question word as topic signature
                    idx = content_lower.find(word)
                    topic = content_lower[idx:idx+50]

                    if topic in question_topics:
                        # Found repeated question
                        first_turn = question_topics[topic]
                        if turn.turn_number - first_turn > 2:  # At least 2 turns apart
                            issues.append({
                                "type": "repeated_question",
                                "turns": [turn.turn_number],
                                "first_asked": first_turn,
                                "description": f"Question repeated (first asked at turn {first_turn})",
                            })
                    else:
                        question_topics[topic] = turn.turn_number
                    break

        return issues[:3]
