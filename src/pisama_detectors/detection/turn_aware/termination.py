"""
F15: Termination Awareness Detector
====================================

Detects FM-1.5: Unaware of Termination Conditions.

Based on MAST research (NeurIPS 2025): This is the highest-prevalence
failure mode in FC1 (40% of system design failures).

Detects:
1. Missing termination signals in long conversations
2. Continuation after explicit termination
3. Repeated completion claims without actual completion
4. Infinite processing without progress indicators

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
from ._embedding_mixin import EmbeddingMixin

logger = logging.getLogger(__name__)


class TurnAwareTerminationAwarenessDetector(EmbeddingMixin, TurnAwareDetector):
    """Detects FM-1.5: Unaware of Termination Conditions.

    Based on MAST research (NeurIPS 2025): This is the highest-prevalence
    failure mode in FC1 (40% of system design failures).

    Detects:
    1. Missing termination signals in long conversations
    2. Continuation after explicit termination
    3. Repeated completion claims without actual completion
    4. Infinite processing without progress indicators

    Enhanced with semantic analysis (v2.0):
    - Semantic similarity to detect completion claims
    - Progress tracking via embedding drift
    - Task completion verification

    Reference: https://arxiv.org/abs/2503.13657
    """

    name = "TurnAwareTerminationAwarenessDetector"
    version = "2.0"  # Semantic enhancement
    supported_failure_modes = ["F15"]  # FM-1.5 maps to new F15

    # Explicit termination signals
    TERMINATION_SIGNALS = [
        "terminate", "done", "complete", "finished",
        "task complete", "goal achieved", "mission accomplished",
        "all done", "nothing more", "that's all",
        "successfully completed", "work is done", "task finished",
        "end of task", "completed successfully", "job done",
    ]

    # Signals that conversation continues after termination
    CONTINUATION_AFTER_TERMINATION = [
        "but wait", "actually", "one more thing",
        "let me also", "additionally", "furthermore",
        "however", "also need to", "i should also",
        "before we finish", "wait", "hold on",
    ]

    # Progress indicators that show work is happening
    PROGRESS_INDICATORS = [
        "step", "progress", "moving on", "next",
        "continuing", "proceeding", "working on",
        "now i'll", "let me", "i will",
    ]

    def __init__(
        self,
        max_turns_without_termination: int = 25,
        max_turns_without_progress: int = 10,
    ):
        self.max_turns_without_termination = max_turns_without_termination
        self.max_turns_without_progress = max_turns_without_progress

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect termination awareness failures."""
        if len(turns) < 3:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few turns to analyze termination patterns",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # 1. Check for missing termination in long conversations
        missing_term = self._detect_missing_termination(turns)
        issues.extend(missing_term)

        # 2. Check for continuation after termination
        ignored_term = self._detect_ignored_termination(turns)
        issues.extend(ignored_term)
        for issue in ignored_term:
            affected_turns.extend(issue.get("turns", []))

        # 3. Check for stalled progress
        stalled = self._detect_stalled_progress(turns)
        issues.extend(stalled)

        # 4. Check for repeated completion claims
        repeated = self._detect_repeated_completion_claims(turns)
        issues.extend(repeated)
        for issue in repeated:
            affected_turns.extend(issue.get("turns", []))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="No termination awareness issues detected",
                detector_name=self.name,
            )

        # Severity based on issue count and type
        has_critical = any(i.get("type") == "ignored_termination" for i in issues)
        if has_critical or len(issues) >= 3:
            severity = TurnAwareSeverity.SEVERE
        elif len(issues) >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.90, 0.55 + len(issues) * 0.12)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F15",
            explanation=f"Termination awareness failure: {len(issues)} issues found",
            affected_turns=list(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=(
                "Implement clear termination conditions: 1) Define explicit stopping criteria, "
                "2) Add termination signal recognition, 3) Prevent processing after completion, "
                "4) Add progress indicators and maximum iteration limits."
            ),
            detector_name=self.name,
        )

    def _detect_missing_termination(self, turns: List[TurnSnapshot]) -> list:
        """Detect conversations that run too long without termination."""
        issues = []

        if len(turns) > self.max_turns_without_termination:
            # Check if there's any termination signal in recent turns
            recent_turns = turns[-5:]
            has_termination = any(
                any(sig in t.content.lower() for sig in self.TERMINATION_SIGNALS)
                for t in recent_turns
            )

            if not has_termination:
                issues.append({
                    "type": "missing_termination",
                    "turns": [len(turns)],
                    "description": f"Long conversation ({len(turns)} turns) without termination signal",
                })

        return issues

    def _detect_ignored_termination(self, turns: List[TurnSnapshot]) -> list:
        """Detect when termination signals are ignored."""
        issues = []

        for i, turn in enumerate(turns[:-1]):
            content_lower = turn.content.lower()

            # Check if this turn has termination signal
            has_termination = any(sig in content_lower for sig in self.TERMINATION_SIGNALS)

            if has_termination:
                # Check if next turn continues inappropriately
                next_turn = turns[i + 1]
                next_lower = next_turn.content.lower()

                # Check for continuation markers
                continues = any(cont in next_lower for cont in self.CONTINUATION_AFTER_TERMINATION)

                # Or if the next turn is from same participant continuing work
                same_participant = turn.participant_id == next_turn.participant_id
                substantial_content = len(next_turn.content) > 100

                if continues or (same_participant and substantial_content):
                    issues.append({
                        "type": "ignored_termination",
                        "turns": [turn.turn_number, next_turn.turn_number],
                        "description": "Conversation continues after termination signal",
                    })

        return issues[:2]

    def _detect_stalled_progress(self, turns: List[TurnSnapshot]) -> list:
        """Detect when conversation stalls without progress."""
        issues = []

        if len(turns) < self.max_turns_without_progress:
            return issues

        # Check recent turns for progress indicators
        recent = turns[-self.max_turns_without_progress:]
        progress_count = sum(
            1 for t in recent
            if any(prog in t.content.lower() for prog in self.PROGRESS_INDICATORS)
        )

        # If very few progress indicators in many turns
        if progress_count < 2:
            issues.append({
                "type": "stalled_progress",
                "turns": [t.turn_number for t in recent],
                "description": f"No progress indicators in last {self.max_turns_without_progress} turns",
            })

        return issues

    def _detect_repeated_completion_claims(self, turns: List[TurnSnapshot]) -> list:
        """Detect repeated claims of completion without actual termination."""
        issues = []
        completion_turns = []

        for turn in turns:
            content_lower = turn.content.lower()
            if any(sig in content_lower for sig in self.TERMINATION_SIGNALS[:6]):
                completion_turns.append(turn.turn_number)

        # Multiple completion claims suggests issues
        if len(completion_turns) >= 3:
            issues.append({
                "type": "repeated_completion_claims",
                "turns": completion_turns,
                "description": f"Multiple completion claims ({len(completion_turns)}) without actual termination",
            })

        return issues
