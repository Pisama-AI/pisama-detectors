"""
F17: Clarification Request Detector
====================================

Detects FM-2.2: Failure to Ask for Clarification.

Based on MAST research (NeurIPS 2025): 18% of FC2 failures.
Agents proceed with ambiguous instructions without seeking clarification.

Detects:
1. Ambiguous task with no clarification request
2. Proceeding despite uncertainty
3. Making assumptions without verification
4. Missing clarification in multi-step tasks

Reference: https://arxiv.org/abs/2503.13657
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


class TurnAwareClarificationRequestDetector(TurnAwareDetector):
    """Detects FM-2.2: Failure to Ask for Clarification.

    Based on MAST research (NeurIPS 2025): 18% of FC2 failures.
    Agents proceed with ambiguous instructions without seeking clarification.

    Detects:
    1. Ambiguous task with no clarification request
    2. Proceeding despite uncertainty
    3. Making assumptions without verification
    4. Missing clarification in multi-step tasks

    Reference: https://arxiv.org/abs/2503.13657
    """

    name = "TurnAwareClarificationRequestDetector"
    version = "1.0"
    supported_failure_modes = ["F17"]  # FM-2.2 maps to new F17

    # Ambiguity indicators in user/task messages
    AMBIGUITY_MARKERS = [
        "maybe", "perhaps", "could be", "either", "or",
        "not sure", "unclear", "ambiguous", "vague",
        "depending", "depends on", "if needed",
        "something like", "kind of", "sort of",
        "whatever", "anything", "some", "any",
    ]

    # Assumption indicators without clarification
    ASSUMPTION_WITHOUT_CLARIFICATION = [
        "i'll assume", "assuming", "i assume",
        "let me assume", "i'm assuming", "assuming that",
        "i'll go with", "i'll use", "defaulting to",
        "probably means", "likely means", "must mean",
        "interpreting as", "taking this to mean",
    ]

    # Proper clarification request indicators
    CLARIFICATION_REQUESTS = [
        "could you clarify", "can you clarify", "please clarify",
        "what do you mean", "could you explain", "can you specify",
        "which one", "do you mean", "are you referring to",
        "to clarify", "just to confirm", "to make sure",
        "?",  # Questions in general
    ]

    def __init__(self, min_turns: int = 2):
        self.min_turns = min_turns

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect failure to ask for clarification."""
        if len(turns) < self.min_turns:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few turns to analyze clarification patterns",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # 1. Check for assumptions without clarification
        assumptions = self._detect_assumptions_without_clarification(turns)
        issues.extend(assumptions)
        for issue in assumptions:
            affected_turns.extend(issue.get("turns", []))

        # 2. Check for proceeding with ambiguous input
        ambiguous = self._detect_proceeding_with_ambiguity(turns)
        issues.extend(ambiguous)
        for issue in ambiguous:
            affected_turns.extend(issue.get("turns", []))

        # 3. Check for missing clarification in complex tasks
        complex_task = self._detect_complex_task_without_clarification(turns)
        issues.extend(complex_task)
        for issue in complex_task:
            affected_turns.extend(issue.get("turns", []))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.80,
                failure_mode=None,
                explanation="No clarification request failures detected",
                detector_name=self.name,
            )

        if len(issues) >= 3:
            severity = TurnAwareSeverity.SEVERE
        elif len(issues) >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.85, 0.50 + len(issues) * 0.12)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F17",
            explanation=f"Clarification request failure: {len(issues)} issues found",
            affected_turns=list(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=(
                "Ask for clarification when needed: 1) Identify ambiguous requirements, "
                "2) Ask specific clarifying questions before proceeding, "
                "3) Confirm assumptions with user, 4) Don't proceed on uncertain paths."
            ),
            detector_name=self.name,
        )

    def _detect_assumptions_without_clarification(self, turns: List[TurnSnapshot]) -> list:
        """Detect when agent makes assumptions without asking."""
        issues = []
        agent_turns = [t for t in turns if t.participant_type == "agent"]

        for turn in agent_turns:
            content_lower = turn.content.lower()

            # Check for assumption markers
            has_assumption = any(marker in content_lower for marker in self.ASSUMPTION_WITHOUT_CLARIFICATION)

            if has_assumption:
                # Check if there was a prior clarification request
                prior_turns = [t for t in turns if t.turn_number < turn.turn_number]
                asked_clarification = any(
                    any(req in t.content.lower() for req in self.CLARIFICATION_REQUESTS)
                    for t in prior_turns
                    if t.participant_type == "agent"
                )

                if not asked_clarification:
                    issues.append({
                        "type": "assumption_without_clarification",
                        "turns": [turn.turn_number],
                        "description": "Made assumption without asking for clarification",
                    })

        return issues[:2]

    def _detect_proceeding_with_ambiguity(self, turns: List[TurnSnapshot]) -> list:
        """Detect when agent proceeds despite ambiguous input."""
        issues = []

        # Find user turns with ambiguity
        for i, turn in enumerate(turns):
            if turn.participant_type in ("user", "system"):
                content_lower = turn.content.lower()
                has_ambiguity = any(marker in content_lower for marker in self.AMBIGUITY_MARKERS)

                if has_ambiguity:
                    # Check if next agent turn asks for clarification
                    next_agent_turns = [
                        t for t in turns[i+1:]
                        if t.participant_type == "agent"
                    ][:2]

                    asks_clarification = any(
                        any(req in t.content.lower() for req in self.CLARIFICATION_REQUESTS)
                        for t in next_agent_turns
                    )

                    if not asks_clarification and next_agent_turns:
                        issues.append({
                            "type": "proceeding_with_ambiguity",
                            "turns": [turn.turn_number, next_agent_turns[0].turn_number],
                            "description": "Proceeded with ambiguous input without clarification",
                        })

        return issues[:2]

    def _detect_complex_task_without_clarification(self, turns: List[TurnSnapshot]) -> list:
        """Detect complex multi-part tasks without clarification."""
        issues = []

        # Check first few turns for complex task indicators
        early_turns = turns[:3]
        for turn in early_turns:
            if turn.participant_type in ("user", "system"):
                content = turn.content

                # Indicators of complex/multi-part task
                complex_indicators = [
                    " and ", " then ", " also ", " additionally ",
                    "1.", "2.", "first", "second", "multiple",
                    "several", "various", "different",
                ]

                has_complexity = sum(1 for ind in complex_indicators if ind in content.lower())

                if has_complexity >= 2 and len(content) > 200:
                    # Check if agent asks clarifying questions
                    next_agent_turns = [t for t in turns if t.participant_type == "agent"][:2]
                    asks_questions = any(
                        "?" in t.content
                        for t in next_agent_turns
                    )

                    if not asks_questions:
                        issues.append({
                            "type": "complex_task_no_clarification",
                            "turns": [turn.turn_number],
                            "description": "Complex multi-part task without clarifying questions",
                        })

        return issues[:1]
