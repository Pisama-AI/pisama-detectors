"""
F13: Quality Gate Bypass Detector
=================================

Analyzes whether quality checks are properly followed:
1. Skipped reviews - code review or QA steps skipped
2. Ignored warnings - warnings present but ignored
3. Bypassed checks - explicitly skipping quality gates
4. Missing tests - no testing before deployment/completion
5. Rush to completion - moving forward despite issues

Based on MAST research (NeurIPS 2025): FM-3.2 No/Incomplete Verification (50%)
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


class TurnAwareQualityGateBypassDetector(EmbeddingMixin, TurnAwareDetector):
    """Detects F13: Quality Gate Bypass in conversations.

    Analyzes whether quality checks are properly followed:
    1. Skipped reviews - code review or QA steps skipped
    2. Ignored warnings - warnings present but ignored
    3. Bypassed checks - explicitly skipping quality gates
    4. Missing tests - no testing before deployment/completion
    5. Rush to completion - moving forward despite issues

    Enhanced with semantic analysis (v2.0):
    - Sentiment shift detection (positive → negative)
    - Quality discussion density scoring
    - Verification completeness tracking

    Based on MAST research (NeurIPS 2025): FM-3.2 No/Incomplete Verification (50%)
    """

    name = "TurnAwareQualityGateBypassDetector"
    version = "2.1"  # Added rubber-stamp verification detection
    supported_failure_modes = ["F13"]

    # Bypass indicators - made more specific to reduce FPs
    BYPASS_INDICATORS = [
        "skip the review", "let's skip testing", "skipping the test",
        "bypass the check", "bypass validation",
        "ignore the warning", "ignoring the error",
        "no need to test", "skip testing", "skip review",
        "good enough for now", "ship it anyway", "move on anyway",
        "we can fix it later", "TODO: fix later", "FIXME: later",
        # Deferral patterns - stricter
        "defer to next release", "post-release fix",
        "add this in v2", "phase 2 feature", "next sprint item",
        "out of scope for now", "future work item",
        # Added for improved recall (v2.1)
        "moving forward without", "accepting the risk",
        "deploying despite", "skipping for now",
        "we can test later", "test this later",
        "proceed without testing", "no time to test",
        "pushing without review", "merge without approval",
    ]

    # Warning ignore indicators - stricter
    WARNING_IGNORES = [
        "ignore this warning", "suppress this warning", "disable warning for",
        "warning ignored because", "warnings disabled for",
        "lint disable for", "noqa:", "pylint: disable=",
        "eslint-disable-next", "despite the warning", "@suppress(",
        # Error handling - stricter
        "ignoring this error", "error ignored because", "skip this error",
        "known issue in", "accepted risk for", "won't fix because",
        "proceed anyway because", "continue anyway despite",
    ]

    # Missing quality steps - stricter
    MISSING_QUALITY = [
        "no tests written", "without any testing", "untested code",
        "no review done", "without code review", "unreviewed code",
        "no qa performed", "skip qa step", "no quality check",
        "didn't test this", "haven't tested yet", "not tested yet",
        # Incomplete verification - stricter
        "no verification done", "unverified changes", "not verified yet",
        "no validation performed", "not validated yet",
        "assume it's correct", "trust me on this",
    ]

    # Rush indicators - stricter
    RUSH_INDICATORS = [
        "quick and dirty fix", "just ship it", "good enough for now",
        "will fix this later", "temporary workaround", "hack for now",
        "workaround for the", "shortcut to avoid", "quick fix for",
        "time constraint forces", "deadline pressure",
        # Rush patterns - stricter
        "crunch mode", "time pressure on", "minimal viable product",
        "bare minimum for", "cut corners on", "expedite at cost",
    ]

    # Verifier role indicators (v2.1 - rubber-stamp detection)
    VERIFIER_ROLE_INDICATORS = [
        "verifier", "validator", "reviewer", "checker", "qa",
        "agent_verifier", "verify_agent", "verification",
    ]

    # Rubber-stamp verification patterns (weak verification without rigor)
    RUBBER_STAMP_PATTERNS = [
        # Explicit rubber-stamping without checking
        "looks correct without", "seems correct but",
        "looks good to me", "lgtm",
        # Acceptance without evidence
        "approve without review", "skip verification",
        "no review needed", "verification not required",
        "skip the check", "bypass the review",
    ]

    # Weak verification phrases (verification claimed but no substance)
    WEAK_VERIFICATION_PHRASES = [
        "verified", "validation complete", "check complete",
        "review complete", "qa passed", "looks good",
        "all good", "approved", "accepted",
    ]

    def __init__(self, min_turns: int = 2, min_issues_to_flag: int = 1):
        self.min_turns = min_turns
        self.min_issues_to_flag = min_issues_to_flag  # Lowered for recall: was 2, now 1

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect quality gate bypass issues."""
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

        # 1. Check for bypass indicators
        bypass_issues = self._detect_bypass(turns)
        issues.extend(bypass_issues)
        for issue in bypass_issues:
            affected_turns.extend(issue.get("turns", []))

        # 2. Check for warning ignores
        warning_issues = self._detect_warning_ignores(turns)
        issues.extend(warning_issues)
        for issue in warning_issues:
            affected_turns.extend(issue.get("turns", []))

        # 3. Check for missing quality steps
        missing_issues = self._detect_missing_quality(turns)
        issues.extend(missing_issues)
        for issue in missing_issues:
            affected_turns.extend(issue.get("turns", []))

        # 4. Check for rush indicators
        rush_issues = self._detect_rush(turns)
        issues.extend(rush_issues)
        for issue in rush_issues:
            affected_turns.extend(issue.get("turns", []))

        # 5. Check for rubber-stamp verification (v2.1)
        rubber_stamp_issues = self._detect_rubber_stamp_verification(turns)
        issues.extend(rubber_stamp_issues)
        for issue in rubber_stamp_issues:
            affected_turns.extend(issue.get("turns", []))

        # 6. Check for absent verification (v2.1)
        absent_issues = self._detect_absent_verification(turns)
        issues.extend(absent_issues)
        for issue in absent_issues:
            affected_turns.extend(issue.get("turns", []))

        # 7. Check for test-then-deploy without results
        test_deploy_issues = self._detect_test_deploy_bypass(turns)
        issues.extend(test_deploy_issues)
        for issue in test_deploy_issues:
            affected_turns.extend(issue.get("turns", []))

        # Require at least one strong evidence issue for detection
        strong_types = {"bypass", "warning_ignore", "rush", "rubber_stamp", "absent_verification", "test_deploy_bypass"}
        has_strong_evidence = any(
            issue.get("type") in strong_types and issue.get("method") != "semantic"
            for issue in issues
        )

        # Require minimum issues AND strong evidence to reduce false positives
        if len(issues) < self.min_issues_to_flag or not has_strong_evidence:
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
            failure_mode="F13",
            explanation=f"Quality gate bypass: {len(issues)} issues found",
            affected_turns=list(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=(
                "Enforce quality gates: 1) Don't skip code reviews, "
                "2) Address warnings before proceeding, 3) Write tests before completion, "
                "4) Follow established quality processes."
            ),
            detector_name=self.name,
        )

    def _detect_bypass(self, turns: List[TurnSnapshot]) -> list:
        """Detect explicit bypass of quality gates.

        Phase 2 Enhancement: Uses semantic similarity to detect bypass intentions
        even when not explicitly stated with keywords.
        """
        issues = []

        # Keyword detection (primary method)
        for turn in turns:
            content_lower = turn.content.lower()
            for indicator in self.BYPASS_INDICATORS:
                if indicator in content_lower:
                    issues.append({
                        "type": "bypass",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Quality bypass: '{indicator}'",
                        "method": "keyword",
                    })
                    break

        # Semantic detection (if embeddings available)
        if self.embedder and len(issues) < 2:  # Only if not enough keyword matches
            bypass_patterns = [
                "Let's skip the testing phase to save time",
                "We can bypass code review this time",
                "Don't need to validate, just ship it",
            ]

            for turn in turns:
                if len(issues) >= 3:  # Limit total issues
                    break

                # Skip if already flagged by keywords
                if any(issue.get("turns") == [turn.turn_number] for issue in issues):
                    continue

                # Check semantic similarity to bypass patterns
                similarities = self.batch_semantic_similarity(turn.content[:300], bypass_patterns)
                if similarities:
                    max_sim = max(similarities)
                    if max_sim >= 0.80:  # High similarity = likely bypass (raised from 0.70)
                        issues.append({
                            "type": "bypass",
                            "turns": [turn.turn_number],
                            "similarity": max_sim,
                            "description": f"Semantic bypass detected (similarity: {max_sim:.2f})",
                            "method": "semantic",
                        })

        return issues[:3]

    def _detect_warning_ignores(self, turns: List[TurnSnapshot]) -> list:
        """Detect ignored warnings."""
        issues = []
        for turn in turns:
            content_lower = turn.content.lower()
            for indicator in self.WARNING_IGNORES:
                if indicator in content_lower:
                    issues.append({
                        "type": "warning_ignore",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Warning ignored: '{indicator}'",
                    })
                    break
        return issues[:2]

    def _detect_missing_quality(self, turns: List[TurnSnapshot]) -> list:
        """Detect missing quality steps.

        Phase 2 Enhancement: Uses semantic analysis to detect implicit
        admissions of skipped quality processes.
        """
        issues = []

        # Keyword detection
        for turn in turns:
            content_lower = turn.content.lower()
            for indicator in self.MISSING_QUALITY:
                if indicator in content_lower:
                    issues.append({
                        "type": "missing_quality",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Missing quality: '{indicator}'",
                        "method": "keyword",
                    })
                    break

        # Semantic detection (augment keyword findings)
        if self.embedder and len(issues) < 2:
            missing_patterns = [
                "We didn't write tests for this feature",
                "Code review wasn't performed on this change",
                "This needs to be verified but we haven't done that yet",
            ]

            for turn in turns:
                if len(issues) >= 2:
                    break

                if any(issue.get("turns") == [turn.turn_number] for issue in issues):
                    continue

                similarities = self.batch_semantic_similarity(turn.content[:300], missing_patterns)
                if similarities:
                    max_sim = max(similarities)
                    if max_sim >= 0.85:  # Raised from 0.78 to further reduce FPs
                        issues.append({
                            "type": "missing_quality",
                            "turns": [turn.turn_number],
                            "similarity": max_sim,
                            "description": f"Semantic missing quality detected (similarity: {max_sim:.2f})",
                            "method": "semantic",
                        })

        return issues[:2]

    def _detect_rush(self, turns: List[TurnSnapshot]) -> list:
        """Detect rush to completion."""
        issues = []
        for turn in turns:
            content_lower = turn.content.lower()
            for indicator in self.RUSH_INDICATORS:
                if indicator in content_lower:
                    issues.append({
                        "type": "rush",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Rush indicator: '{indicator}'",
                    })
                    break
        return issues[:2]

    def _detect_rubber_stamp_verification(self, turns: List[TurnSnapshot]) -> list:
        """Detect superficial/rubber-stamp verification (v2.1).

        This addresses the MAST F13 pattern where:
        - A Verifier role exists but does weak verification
        - Verification just confirms consensus without rigorous checking
        - No actual validation logic or error detection performed

        Returns:
            List of issues with rubber-stamp verification patterns
        """
        issues = []

        # First, check if there's a verifier role in the conversation
        verifier_turns = []

        for turn in turns:
            content_lower = turn.content.lower()
            participant_lower = (turn.participant_id or "").lower()

            # Check if this is a verifier role
            is_verifier = False
            for indicator in self.VERIFIER_ROLE_INDICATORS:
                if indicator in participant_lower or f"name': '{indicator}" in content_lower:
                    is_verifier = True
                    break

            if is_verifier:
                verifier_turns.append(turn)

        # If no explicit verifier, look for verification-like turns
        if not verifier_turns:
            for turn in turns:
                content_lower = turn.content.lower()
                if any(phrase in content_lower for phrase in self.WEAK_VERIFICATION_PHRASES):
                    verifier_turns.append(turn)

        # Check verifier turns for rubber-stamp patterns
        for turn in verifier_turns:
            content_lower = turn.content.lower()

            # Check for rubber-stamp patterns
            for pattern in self.RUBBER_STAMP_PATTERNS:
                if pattern in content_lower:
                    # Check if there's actual verification logic
                    has_substance = self._has_verification_substance(turn.content)

                    if not has_substance:
                        issues.append({
                            "type": "rubber_stamp",
                            "turns": [turn.turn_number],
                            "pattern": pattern,
                            "description": f"Rubber-stamp verification: '{pattern}' without rigorous checking",
                        })
                        break  # One issue per turn

        # Also detect when verification is claimed but no evidence provided
        for turn in turns:
            content_lower = turn.content.lower()

            # Check for weak verification without substance
            for phrase in self.WEAK_VERIFICATION_PHRASES:
                if phrase in content_lower:
                    # Skip if already flagged
                    if any(turn.turn_number in issue.get("turns", []) for issue in issues):
                        continue

                    has_substance = self._has_verification_substance(turn.content)
                    if not has_substance:
                        issues.append({
                            "type": "weak_verification",
                            "turns": [turn.turn_number],
                            "phrase": phrase,
                            "description": f"Weak verification claimed: '{phrase}' without evidence",
                        })
                        break

        return issues[:3]  # Limit to avoid false positives

    def _detect_absent_verification(self, turns: List[TurnSnapshot]) -> list:
        """Detect complete absence of verification in completed traces (v2.1).

        F13 is often about verification NOT happening at all, not explicit bypass.
        This detects when a task is completed without any verification step.
        """
        issues = []

        # Check if there's a completion claim
        completion_patterns = [
            "task complete", "done", "finished", "all done",
            "solution_found", "final answer", "implementation complete",
            "here's the solution", "problem solved", "mission accomplished",
        ]

        # Verification-related keywords that should appear somewhere
        verification_keywords = [
            "test", "verify", "validate", "check", "review",
            "confirm", "ensure", "assert", "evaluate", "examine",
        ]

        has_completion = False
        has_verification = False

        for turn in turns:
            content_lower = turn.content.lower()

            # Check for completion
            for pattern in completion_patterns:
                if pattern in content_lower:
                    has_completion = True
                    break

            # Check for verification keywords
            for keyword in verification_keywords:
                if keyword in content_lower:
                    has_verification = True
                    break

        # If completed without verification keywords, flag it
        if has_completion and not has_verification:
            issues.append({
                "type": "absent_verification",
                "turns": [turns[-1].turn_number] if turns else [],
                "description": "Task completed without any verification/testing mentioned",
            })

        return issues

    def _detect_test_deploy_bypass(self, turns: List[TurnSnapshot]) -> list:
        """Detect test step followed by deploy without showing test results."""
        issues = []

        # Look for test → deploy sequence
        test_keywords = ["test", "testing", "tester", "run_tests", "execute_tests"]
        deploy_keywords = ["deploy", "deployment", "deployer", "release", "production", "publish"]
        test_result_keywords = ["passed", "failed", "success", "error", "result", "output"]

        test_turn_idx = None
        deploy_turn_idx = None

        for i, turn in enumerate(turns):
            content_lower = turn.content.lower()
            turn_name_lower = turn.participant_id.lower()

            # Check if this is a test turn
            if any(kw in content_lower or kw in turn_name_lower for kw in test_keywords):
                test_turn_idx = i

            # Check if this is a deploy turn
            if any(kw in content_lower or kw in turn_name_lower for kw in deploy_keywords):
                deploy_turn_idx = i

        # If we have test → deploy sequence
        if test_turn_idx is not None and deploy_turn_idx is not None and test_turn_idx < deploy_turn_idx:
            # Check if there are test results between test and deploy
            has_test_results = False
            for i in range(test_turn_idx, deploy_turn_idx):
                content_lower = turns[i].content.lower()
                if any(kw in content_lower for kw in test_result_keywords):
                    has_test_results = True
                    break

            # If no test results shown, flag it
            if not has_test_results:
                issues.append({
                    "type": "test_deploy_bypass",
                    "turns": [test_turn_idx, deploy_turn_idx],
                    "description": "Tests run but results not shown before deployment",
                })

        return issues

    def _has_verification_substance(self, content: str) -> bool:
        """Check if verification content has substantive checking.

        Substantive verification includes:
        - Error detection/correction
        - Mathematical or logical validation
        - Test execution results
        - Specific issue identification

        Returns:
            True if verification has substance, False if rubber-stamp
        """
        content_lower = content.lower()

        # Indicators of substantive verification
        substance_indicators = [
            # Error detection
            "error found", "bug detected", "issue identified", "problem found",
            "incorrect", "wrong", "mistake", "flaw", "defect",
            # Logical validation
            "because", "since", "therefore", "thus", "hence",
            "the reason is", "this is because", "due to",
            # Mathematical validation
            "calculation shows", "computed as", "evaluates to",
            "=", "equals", "results in",
            # Test execution
            "test passed", "test failed", "execution result",
            "output shows", "returns", "produces",
            # Specific issue identification
            "specifically", "in particular", "notably",
            "line ", "function ", "variable ",
        ]

        substance_count = sum(1 for ind in substance_indicators if ind in content_lower)

        # Require at least 2 substance indicators for real verification
        return substance_count >= 2
