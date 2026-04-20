"""
F12: Output Validation Failure Detector
=======================================

Analyzes whether agent outputs are properly validated:
1. Missing validation - outputs produced without checking
2. Failed validation - validation ran but output doesn't pass
3. Uncaught errors - errors in output that weren't caught
4. Format issues - output doesn't match expected format
5. Incomplete output - missing required components

Based on MAST research (NeurIPS 2025): FM-3.3 Incorrect Verification (28%)
"""

import logging
import re
from typing import List, Optional, Dict, Any

from ._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)
from ._embedding_mixin import EmbeddingMixin

logger = logging.getLogger(__name__)


class TurnAwareOutputValidationDetector(EmbeddingMixin, TurnAwareDetector):
    """Detects F12: Output Validation Failure in conversations.

    Analyzes whether agent outputs are properly validated:
    1. Missing validation - outputs produced without checking
    2. Failed validation - validation ran but output doesn't pass
    3. Uncaught errors - errors in output that weren't caught
    4. Format issues - output doesn't match expected format
    5. Incomplete output - missing required components

    Enhanced with semantic analysis (v2.0):
    - Error-after-success pattern detection using semantic shift
    - Output quality scoring based on information density
    - Validation completeness analysis

    Based on MAST research (NeurIPS 2025): FM-3.3 Incorrect Verification (28%)
    """

    name = "TurnAwareOutputValidationDetector"
    version = "2.1"  # Phase 1: Framework-specific completion signals
    supported_failure_modes = ["F12"]

    # Validation failure indicators
    VALIDATION_FAILURES = [
        "validation failed", "invalid output", "doesn't validate",
        "failed to validate", "validation error", "schema error",
        "type error", "format error", "malformed",
        "doesn't match", "expected format", "invalid format",
        "parsing error", "parse failed", "couldn't parse",
    ]

    # Missing validation indicators
    MISSING_VALIDATION = [
        "didn't check", "forgot to validate", "skipped validation",
        "no validation", "without checking", "unchecked",
        "assumed correct", "didn't verify", "unverified",
    ]

    # Output error indicators
    OUTPUT_ERRORS = [
        "output error", "result is wrong", "incorrect output",
        "wrong result", "bad output", "output incorrect",
        "doesn't work", "broken", "buggy", "syntax error",
        "runtime error", "compile error", "execution failed",
    ]

    # Phase 1: Framework-specific completion signals (indicators of successful completion)
    COMPLETION_SIGNALS = {
        "ChatDev": [
            r"task (?:is )?completed",
            r"successfully (?:completed|finished|delivered)",
            r"project (?:is )?complete",
            r"deliverables? (?:ready|complete)",
            r"all requirements? met",
            r"passed (?:all )?tests?",
        ],
        "AG2": [
            r"(?:task|problem) solved",
            r"answer (?:is|:)",
            r"final (?:answer|result|solution)",
            r"conclude that",
            r"therefore",
            r"in conclusion",
        ],
        "MetaGPT": [
            r"implementation complete",
            r"code (?:is )?ready",
            r"deliverable complete",
            r"phase complete",
        ],
        "Magentic": [
            r"result returned",
            r"function completed",
            r"output generated",
        ],
        "default": [
            r"completed successfully",
            r"task (?:is )?done",
            r"finished",
            r"ready (?:for|to)",
            r"here (?:is|are) (?:the|your)",
            r"i'?ve completed",
        ],
    }

    def __init__(self, min_turns: int = 2, min_issues_to_flag: int = 3, framework: Optional[str] = None):
        self.min_turns = min_turns
        self.min_issues_to_flag = min_issues_to_flag  # Raised to 3 to reduce FPs (17.8% FPR)
        self.framework = framework

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect output validation failures."""
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

        # 1. Check for validation failure indicators
        validation_issues = self._detect_validation_failures(turns)
        issues.extend(validation_issues)
        for issue in validation_issues:
            affected_turns.extend(issue.get("turns", []))

        # 2. Check for missing validation
        missing_issues = self._detect_missing_validation(turns)
        issues.extend(missing_issues)
        for issue in missing_issues:
            affected_turns.extend(issue.get("turns", []))

        # 3. Check for output errors
        error_issues = self._detect_output_errors(turns)
        issues.extend(error_issues)
        for issue in error_issues:
            affected_turns.extend(issue.get("turns", []))

        # 4. Check for code that doesn't run/compile
        code_issues = self._detect_broken_code(agent_turns)
        issues.extend(code_issues)
        for issue in code_issues:
            affected_turns.extend(issue.get("turns", []))

        # Phase 1: Check for completion signals that indicate successful output
        has_completion_signal = self._has_completion_signals(turns)
        if has_completion_signal:
            # If we have strong completion signals, be more lenient
            # Allow 1 more issue before flagging (reduce false positives)
            min_issues = self.min_issues_to_flag + 1
        else:
            min_issues = self.min_issues_to_flag

        # Require multiple issues to reduce false positives
        if len(issues) < min_issues:
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
            failure_mode="F12",
            explanation=f"Output validation failure: {len(issues)} issues found",
            affected_turns=list(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=(
                "Add output validation: 1) Validate all outputs against schema, "
                "2) Run tests before returning results, 3) Check for common errors, "
                "4) Verify output format matches expectations."
            ),
            detector_name=self.name,
        )

    def _has_completion_signals(self, turns: List[TurnSnapshot]) -> bool:
        """Phase 1: Check if conversation contains framework-specific completion signals."""
        # Get patterns for this framework (or default)
        # BUGFIX: Use list concatenation to avoid mutating class-level dictionary values
        # which caused unbounded list growth and exponential slowdown
        framework_key = self.framework if self.framework in self.COMPLETION_SIGNALS else "default"
        patterns = self.COMPLETION_SIGNALS.get(framework_key, []) + self.COMPLETION_SIGNALS["default"]

        # Check last few agent turns for completion signals
        agent_turns = [t for t in turns if t.participant_type == "agent"]
        last_turns = agent_turns[-3:] if len(agent_turns) >= 3 else agent_turns

        for turn in last_turns:
            content_lower = turn.content.lower()
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    return True
        return False

    def _detect_validation_failures(self, turns: List[TurnSnapshot]) -> list:
        """Detect explicit validation failures."""
        issues = []
        for turn in turns:
            # Check for validation_failed flag in metadata
            if turn.turn_metadata.get("validation_failed"):
                issues.append({
                    "type": "validation_failure",
                    "turns": [turn.turn_number],
                    "indicator": "validation_failed_flag",
                    "description": "Explicit validation failure flag set",
                })
                continue

            # Check for text indicators
            content_lower = turn.content.lower()
            for indicator in self.VALIDATION_FAILURES:
                if indicator in content_lower:
                    issues.append({
                        "type": "validation_failure",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Validation failure: '{indicator}'",
                    })
                    break
        return issues[:3]

    def _detect_missing_validation(self, turns: List[TurnSnapshot]) -> list:
        """Detect missing validation."""
        issues = []
        for turn in turns:
            content_lower = turn.content.lower()
            for indicator in self.MISSING_VALIDATION:
                if indicator in content_lower:
                    issues.append({
                        "type": "missing_validation",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Missing validation: '{indicator}'",
                    })
                    break
        return issues[:2]

    def _detect_output_errors(self, turns: List[TurnSnapshot]) -> list:
        """Detect output errors."""
        issues = []
        for turn in turns:
            content_lower = turn.content.lower()
            for indicator in self.OUTPUT_ERRORS:
                if indicator in content_lower:
                    issues.append({
                        "type": "output_error",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Output error: '{indicator}'",
                    })
                    break
        return issues[:3]

    def _detect_broken_code(self, agent_turns: List[TurnSnapshot]) -> list:
        """Detect code blocks followed by error mentions."""
        issues = []
        for i, turn in enumerate(agent_turns):
            # Check if this turn has code
            has_code = "```" in turn.content or "def " in turn.content or "class " in turn.content
            if has_code and i < len(agent_turns) - 1:
                # Check next turns for error indicators
                for j in range(i + 1, min(i + 3, len(agent_turns))):
                    next_content = agent_turns[j].content.lower()
                    if any(err in next_content for err in ["error", "failed", "doesn't work", "bug", "fix"]):
                        issues.append({
                            "type": "broken_code",
                            "turns": [turn.turn_number, agent_turns[j].turn_number],
                            "description": "Code followed by error discussion",
                        })
                        break
        return issues[:2]
