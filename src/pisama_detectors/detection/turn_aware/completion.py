"""
F14: Completion Misjudgment Detector
====================================

Analyzes whether task completion is correctly assessed:
1. Premature completion - declaring done when not finished
2. Incomplete requirements - not all requirements addressed
3. Unfinished work - obvious gaps in deliverables
4. False success claims - claiming success despite failures
5. Missed acceptance criteria - not meeting stated criteria

Based on MAST research (NeurIPS 2025): FM-3.1 Completion Misjudgment (23%)
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


class TurnAwareCompletionMisjudgmentDetector(EmbeddingMixin, TurnAwareDetector):
    """Detects F14: Completion Misjudgment in conversations.

    Analyzes whether task completion is correctly assessed:
    1. Premature completion - declaring done when not finished
    2. Incomplete requirements - not all requirements addressed
    3. Unfinished work - obvious gaps in deliverables
    4. False success claims - claiming success despite failures
    5. Missed acceptance criteria - not meeting stated criteria

    Phase 2 Enhancement (v2.0): Uses semantic similarity to:
    - Detect confidence level in completion claims (high confidence vs uncertain)
    - Identify semantic inconsistencies between completion and status
    - Track requirement coverage using embedding similarity
    - Detect incomplete deliverables through semantic gap analysis

    Based on MAST research (NeurIPS 2025): FM-3.1 Completion Misjudgment (23%)
    """

    name = "TurnAwareCompletionMisjudgmentDetector"
    version = "2.0"  # Phase 2: Enhanced with semantic analysis
    supported_failure_modes = ["F14"]

    # Premature completion indicators - made more specific to reduce FPs
    PREMATURE_COMPLETION = [
        "task is complete", "task complete", "task finished",
        "all done", "completely done", "fully completed",
        "mission accomplished", "final version",
        "implementation complete", "development complete",
        "work is complete", "everything is done",
        # Added for improved recall (v2.1) - implicit completion claims
        "here's the solution", "here's the implementation",
        "i've implemented", "i have implemented",
        "solution below", "see attached",
        "here is the final", "here's the finished",
        "ready for review", "ready for deployment",
    ]

    # Incomplete indicators following completion - stricter patterns
    INCOMPLETE_INDICATORS = [
        "still need to", "remaining tasks", "left to do", "not yet finished",
        "missing parts", "incomplete implementation", "partial solution",
        "TODO:", "FIXME:", "TBD:", "WIP:",
        "placeholder code", "stub implementation", "mock data",
        "need to finish", "haven't completed", "not fully",
    ]

    # False success indicators - stricter patterns
    FALSE_SUCCESS = [
        "should probably work", "might not work", "probably won't work",
        "seems to work but", "appears to work but", "looks like it might",
        "assuming it works", "hopefully it works", "fingers crossed",
        "not sure if it works", "haven't tested", "untested code",
    ]

    # Continuation needed indicators - stricter patterns
    CONTINUATION_NEEDED = [
        "next step is to", "then we need to", "after that we need",
        "following that we must", "additionally we need",
        "don't forget to", "remember to also", "make sure to also",
        "still need to complete", "have to also do", "need to also finish",
    ]

    def __init__(self, min_turns: int = 2, min_issues_to_flag: int = 1):
        self.min_turns = min_turns
        self.min_issues_to_flag = min_issues_to_flag  # Lowered for recall: was 2, now 1

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect completion misjudgment issues."""
        if len(turns) < self.min_turns:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few turns to analyze",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # 1. Check for completion followed by incompleteness
        completion_issues = self._detect_premature_completion(turns)
        issues.extend(completion_issues)
        for issue in completion_issues:
            affected_turns.extend(issue.get("turns", []))

        # 2. Check for false success claims
        false_success_issues = self._detect_false_success(turns)
        issues.extend(false_success_issues)
        for issue in false_success_issues:
            affected_turns.extend(issue.get("turns", []))

        # 3. Check for continuation needed after completion claim
        continuation_issues = self._detect_continuation_needed(turns)
        issues.extend(continuation_issues)
        for issue in continuation_issues:
            affected_turns.extend(issue.get("turns", []))

        # 4. Check for TODO/FIXME in "completed" work
        todo_issues = self._detect_unfinished_markers(turns)
        issues.extend(todo_issues)
        for issue in todo_issues:
            affected_turns.extend(issue.get("turns", []))

        # Require multiple issues to reduce false positives
        if len(issues) < self.min_issues_to_flag:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.8,
                failure_mode=None,
                explanation=f"Insufficient evidence ({len(issues)} issue(s), need {self.min_issues_to_flag}+)",
                detector_name=self.name,
            )

        if len(issues) >= 3:
            severity = TurnAwareSeverity.SEVERE
        elif len(issues) >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.85, 0.5 + len(issues) * 0.12)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F14",
            explanation=f"Completion misjudgment: {len(issues)} issues found",
            affected_turns=list(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=(
                "Verify completion properly: 1) Check all requirements are met, "
                "2) Test the output before declaring done, 3) Review for TODOs/FIXMEs, "
                "4) Validate against acceptance criteria."
            ),
            detector_name=self.name,
        )

    def _detect_premature_completion(self, turns: List[TurnSnapshot]) -> list:
        """Detect completion claims followed by incompleteness."""
        issues = []
        for i, turn in enumerate(turns):
            content_lower = turn.content.lower()
            # Check for completion claim
            has_completion = any(ind in content_lower for ind in self.PREMATURE_COMPLETION)
            if has_completion:
                # Check same turn or next turns for incompleteness
                all_content = content_lower
                if i < len(turns) - 1:
                    all_content += " " + turns[i + 1].content.lower()

                for ind in self.INCOMPLETE_INDICATORS:
                    if ind.lower() in all_content:
                        issues.append({
                            "type": "premature_completion",
                            "turns": [turn.turn_number],
                            "indicator": ind,
                            "description": f"Completion claim with incompleteness: '{ind}'",
                        })
                        break
        return issues[:3]

    def _detect_false_success(self, turns: List[TurnSnapshot]) -> list:
        """Detect uncertain success claims.

        Phase 2 Enhancement: Uses semantic similarity to detect lack of confidence
        in completion claims even without explicit uncertainty keywords.
        """
        issues = []

        # Keyword detection
        for turn in turns:
            content_lower = turn.content.lower()
            for indicator in self.FALSE_SUCCESS:
                if indicator in content_lower:
                    issues.append({
                        "type": "false_success",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Uncertain success: '{indicator}'",
                        "method": "keyword",
                    })
                    break

        # Semantic detection - detect uncertain completion claims
        if self.embedder and len(issues) < 2:
            uncertain_completion_patterns = [
                "I think it works, but I'm not completely certain",
                "The solution might be correct, but needs verification",
                "This appears to solve the problem, though untested",
            ]

            confident_completion_patterns = [
                "The task is completed successfully and verified",
                "All requirements have been met and tested",
                "Solution is confirmed working as expected",
            ]

            for turn in turns:
                if len(issues) >= 3:
                    break

                if any(issue.get("turns") == [turn.turn_number] for issue in issues):
                    continue

                content = turn.content[:400]

                # Check for completion claim
                has_completion = any(ind in content.lower() for ind in self.PREMATURE_COMPLETION)
                if not has_completion:
                    continue

                # Compare confidence: uncertain vs confident completion
                uncertain_sims = self.batch_semantic_similarity(content, uncertain_completion_patterns)
                confident_sims = self.batch_semantic_similarity(content, confident_completion_patterns)

                if uncertain_sims and confident_sims:
                    max_uncertain = max(uncertain_sims)
                    max_confident = max(confident_sims)

                    # If more similar to uncertain than confident completion
                    if max_uncertain >= 0.62 and max_uncertain > max_confident + 0.10:
                        issues.append({
                            "type": "false_success",
                            "turns": [turn.turn_number],
                            "uncertainty_score": max_uncertain,
                            "confidence_score": max_confident,
                            "description": f"Semantically uncertain completion claim (uncertainty: {max_uncertain:.2f})",
                            "method": "semantic",
                        })

        return issues[:3]

    def _detect_continuation_needed(self, turns: List[TurnSnapshot]) -> list:
        """Detect continuation needed after completion."""
        issues = []
        completion_found = False

        for turn in turns:
            content_lower = turn.content.lower()

            # Track if completion was claimed
            if any(ind in content_lower for ind in self.PREMATURE_COMPLETION):
                completion_found = True

            # After completion, check for continuation needs
            if completion_found:
                for indicator in self.CONTINUATION_NEEDED:
                    if indicator in content_lower:
                        issues.append({
                            "type": "continuation_needed",
                            "turns": [turn.turn_number],
                            "indicator": indicator,
                            "description": f"More work needed after completion: '{indicator}'",
                        })
                        break
        return issues[:2]

    def _detect_unfinished_markers(self, turns: List[TurnSnapshot]) -> list:
        """Detect TODO/FIXME markers in supposedly complete work."""
        issues = []
        markers = ["TODO", "FIXME", "XXX", "HACK", "TBD", "WIP"]

        for turn in turns:
            content = turn.content
            for marker in markers:
                if marker in content:
                    issues.append({
                        "type": "unfinished_marker",
                        "turns": [turn.turn_number],
                        "marker": marker,
                        "description": f"Unfinished marker: {marker}",
                    })
                    break
        return issues[:3]
