"""
F16: Reasoning-Action Mismatch Detector
=======================================

Detects FM-2.6: Reasoning-Action Mismatch.

Based on MAST research (NeurIPS 2025): This is the highest-prevalence
failure mode in FC2 (26% of inter-agent misalignment failures).

Detects discrepancy between stated reasoning and actual actions:
1. Intent expressed but different action taken
2. Reasoning suggests one approach, execution uses another
3. Chain-of-thought diverges from final action
4. Stated goals don't match actions taken

Reference: https://arxiv.org/abs/2503.13657
ReAct Framework: https://arxiv.org/abs/2210.03629
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


class TurnAwareReasoningActionMismatchDetector(TurnAwareDetector):
    """Detects FM-2.6: Reasoning-Action Mismatch.

    Based on MAST research (NeurIPS 2025): This is the highest-prevalence
    failure mode in FC2 (26% of inter-agent misalignment failures).

    Detects discrepancy between stated reasoning and actual actions:
    1. Intent expressed but different action taken
    2. Reasoning suggests one approach, execution uses another
    3. Chain-of-thought diverges from final action
    4. Stated goals don't match actions taken

    Reference: https://arxiv.org/abs/2503.13657
    ReAct Framework: https://arxiv.org/abs/2210.03629
    """

    name = "TurnAwareReasoningActionMismatchDetector"
    version = "1.0"
    supported_failure_modes = ["F16"]  # FM-2.6 maps to new F16

    # Intent markers in reasoning
    INTENT_MARKERS = {
        "search": ["will search", "going to search", "let me search", "searching for", "i'll look up"],
        "write": ["will write", "going to create", "let me write", "i'll generate", "creating"],
        "read": ["will read", "going to read", "let me examine", "i'll review", "reading"],
        "calculate": ["will calculate", "going to compute", "let me figure", "computing"],
        "execute": ["will run", "going to execute", "let me run", "executing", "running"],
        "analyze": ["will analyze", "going to analyze", "let me analyze", "analyzing"],
        "fix": ["will fix", "going to fix", "let me fix", "fixing", "correcting"],
        "test": ["will test", "going to test", "let me test", "testing", "verifying"],
    }

    # Action indicators
    ACTION_INDICATORS = {
        "search": ["searched", "found", "results show", "search returned", "query results"],
        "write": ["wrote", "created", "generated", "here's the code", "here is the"],
        "read": ["read", "examined", "reviewed", "content shows", "file contains"],
        "calculate": ["calculated", "computed", "result is", "equals", "total"],
        "execute": ["ran", "executed", "output:", "returned", "result:"],
        "analyze": ["analyzed", "analysis shows", "found that", "discovered"],
        "fix": ["fixed", "corrected", "updated", "changed", "modified"],
        "test": ["tested", "test passed", "test failed", "verification", "confirmed"],
    }

    # Contradiction patterns
    CONTRADICTIONS = [
        ("will search", "without searching"),
        ("will read", "without reading"),
        ("will test", "skipping test"),
        ("will verify", "assuming correct"),
        ("need to check", "assuming"),
        ("should validate", "looks correct"),
    ]

    def __init__(self, min_turns: int = 3):
        self.min_turns = min_turns

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect reasoning-action mismatches."""
        agent_turns = [t for t in turns if t.participant_type == "agent"]

        if len(agent_turns) < self.min_turns:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few agent turns to analyze reasoning patterns",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # 1. Check for intent-action mismatches within turns
        within_turn = self._detect_within_turn_mismatch(agent_turns)
        issues.extend(within_turn)
        for issue in within_turn:
            affected_turns.extend(issue.get("turns", []))

        # 2. Check for cross-turn intent-action mismatches
        cross_turn = self._detect_cross_turn_mismatch(agent_turns)
        issues.extend(cross_turn)
        for issue in cross_turn:
            affected_turns.extend(issue.get("turns", []))

        # 3. Check for explicit contradictions
        contradictions = self._detect_contradictions(agent_turns)
        issues.extend(contradictions)
        for issue in contradictions:
            affected_turns.extend(issue.get("turns", []))

        # 4. Check for saying-doing gaps
        saying_doing = self._detect_saying_doing_gap(agent_turns)
        issues.extend(saying_doing)
        for issue in saying_doing:
            affected_turns.extend(issue.get("turns", []))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.82,
                failure_mode=None,
                explanation="No reasoning-action mismatches detected",
                detector_name=self.name,
            )

        if len(issues) >= 3:
            severity = TurnAwareSeverity.SEVERE
        elif len(issues) >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.88, 0.52 + len(issues) * 0.12)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F16",
            explanation=f"Reasoning-action mismatch: {len(issues)} issues found",
            affected_turns=list(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=(
                "Ensure reasoning aligns with actions: 1) Verify intended actions are executed, "
                "2) Add explicit action logging, 3) Implement thought-action consistency checks, "
                "4) Use ReAct-style structured reasoning with explicit action verification."
            ),
            detector_name=self.name,
        )

    def _detect_within_turn_mismatch(self, turns: List[TurnSnapshot]) -> list:
        """Detect mismatches between reasoning and action within same turn."""
        issues = []

        for turn in turns:
            content_lower = turn.content.lower()

            for action_type, intent_phrases in self.INTENT_MARKERS.items():
                # Check if intent is expressed
                has_intent = any(phrase in content_lower for phrase in intent_phrases)

                if has_intent:
                    # Check if corresponding action is taken
                    action_phrases = self.ACTION_INDICATORS.get(action_type, [])
                    has_action = any(phrase in content_lower for phrase in action_phrases)

                    # If intent without action, might be mismatch
                    # But only flag if turn is substantial (not just planning)
                    if not has_action and len(turn.content) > 200:
                        issues.append({
                            "type": "intent_without_action",
                            "turns": [turn.turn_number],
                            "intent": action_type,
                            "description": f"Expressed intent to '{action_type}' but no action taken",
                        })

        return issues[:2]

    def _detect_cross_turn_mismatch(self, turns: List[TurnSnapshot]) -> list:
        """Detect mismatches between turns (intent in one, no action in next)."""
        issues = []

        for i in range(len(turns) - 1):
            current = turns[i]
            next_turn = turns[i + 1]
            current_lower = current.content.lower()
            next_lower = next_turn.content.lower()

            for action_type, intent_phrases in self.INTENT_MARKERS.items():
                has_intent = any(phrase in current_lower for phrase in intent_phrases)

                if has_intent:
                    # Check if next turn has the action
                    action_phrases = self.ACTION_INDICATORS.get(action_type, [])
                    next_has_action = any(phrase in next_lower for phrase in action_phrases)

                    # Check if next turn abandons the intent
                    abandons = any(phrase in next_lower for phrase in [
                        "instead", "actually", "let me", "different approach",
                        "skip", "ignore", "without"
                    ])

                    if not next_has_action and abandons:
                        issues.append({
                            "type": "abandoned_intent",
                            "turns": [current.turn_number, next_turn.turn_number],
                            "intent": action_type,
                            "description": f"Intent to '{action_type}' abandoned in next turn",
                        })

        return issues[:2]

    def _detect_contradictions(self, turns: List[TurnSnapshot]) -> list:
        """Detect explicit contradictions in reasoning."""
        issues = []

        for turn in turns:
            content_lower = turn.content.lower()

            for intent_phrase, contradiction_phrase in self.CONTRADICTIONS:
                if intent_phrase in content_lower and contradiction_phrase in content_lower:
                    issues.append({
                        "type": "explicit_contradiction",
                        "turns": [turn.turn_number],
                        "intent": intent_phrase,
                        "contradiction": contradiction_phrase,
                        "description": f"Contradiction: '{intent_phrase}' but '{contradiction_phrase}'",
                    })

        return issues[:2]

    def _detect_saying_doing_gap(self, turns: List[TurnSnapshot]) -> list:
        """Detect gaps between what agent says and does."""
        issues = []

        for turn in turns:
            content_lower = turn.content.lower()

            # Check for "I did X" without evidence of X
            claim_action_pairs = [
                ("tested", "test"),
                ("verified", "verif"),
                ("checked", "check"),
                ("validated", "valid"),
                ("confirmed", "confirm"),
            ]

            for claim, evidence in claim_action_pairs:
                if f"i {claim}" in content_lower or f"i have {claim}" in content_lower:
                    # Look for evidence of actual testing/verification
                    evidence_markers = [
                        "output:", "result:", "returned", "shows",
                        "passed", "failed", "error:", "success"
                    ]
                    has_evidence = any(marker in content_lower for marker in evidence_markers)

                    if not has_evidence:
                        issues.append({
                            "type": "claim_without_evidence",
                            "turns": [turn.turn_number],
                            "claim": claim,
                            "description": f"Claims to have {claim} but no evidence shown",
                        })

        return issues[:2]
