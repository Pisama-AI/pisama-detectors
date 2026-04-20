"""
F3: Resource Misallocation Detector
===================================

Analyzes whether agents have appropriate resources:
1. Missing tools/capabilities - agent needs something they don't have
2. Wrong agent for task - task assigned to agent without required skills
3. Resource overload - one agent given too much work
4. Underutilized agents - some agents doing nothing while others overloaded
5. Tool/API failures - resources not working as expected
"""

import logging
from collections import Counter
from typing import List, Optional, Dict, Any

from ._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)

logger = logging.getLogger(__name__)


class TurnAwareResourceMisallocationDetector(TurnAwareDetector):
    """Detects F3: Resource Misallocation in multi-agent conversations.

    Analyzes whether agents have appropriate resources:
    1. Missing tools/capabilities - agent needs something they don't have
    2. Wrong agent for task - task assigned to agent without required skills
    3. Resource overload - one agent given too much work
    4. Underutilized agents - some agents doing nothing while others overloaded
    5. Tool/API failures - resources not working as expected

    This is the 2nd most common failure mode in MAST (36% prevalence).
    """

    name = "TurnAwareResourceMisallocationDetector"
    version = "1.0"
    supported_failure_modes = ["F3"]

    # Resource complaint indicators - made more specific to reduce FPs
    RESOURCE_COMPLAINTS = [
        "don't have access to", "no access to the", "cannot access the",
        "missing required", "resource not available", "resource unavailable",
        "need permission to", "not authorized to", "access denied for",
        "tool not found:", "api error:", "api failure",
        "resource missing", "not installed on", "import error:",
    ]

    # Capability mismatch indicators - made more specific
    CAPABILITY_MISMATCH = [
        "not my area of expertise", "outside my designated scope",
        "i am not qualified to", "i don't know how to",
        "beyond my capabilities as", "not designed for this",
        "should be handled by another agent", "need a specialist for",
    ]

    # Overload indicators - made more specific
    OVERLOAD_INDICATORS = [
        "too many tasks assigned", "system is overloaded",
        "can't handle all these", "workload is too high",
        "queue is full", "rate limit exceeded", "being throttled",
        "operation timed out", "request timed out",
    ]

    def __init__(
        self,
        min_turns: int = 2,
        min_issues_to_flag: int = 1,  # Lowered for token explosion detection
        token_explosion_threshold: int = 2000,  # Flag if >2000 tokens for simple tasks
    ):
        self.min_turns = min_turns
        self.min_issues_to_flag = min_issues_to_flag
        self.token_explosion_threshold = token_explosion_threshold

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect resource misallocation issues."""
        agent_turns = [t for t in turns if t.participant_type == "agent"]
        tool_turns = [t for t in turns if t.participant_type == "tool"]

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

        # 1. Check for resource complaints
        resource_issues = self._detect_resource_complaints(agent_turns + tool_turns)
        issues.extend(resource_issues)
        for issue in resource_issues:
            affected_turns.extend(issue.get("turns", []))

        # 2. Check for capability mismatches
        capability_issues = self._detect_capability_mismatch(agent_turns)
        issues.extend(capability_issues)
        for issue in capability_issues:
            affected_turns.extend(issue.get("turns", []))

        # 3. Check for overload indicators
        overload_issues = self._detect_overload(agent_turns)
        issues.extend(overload_issues)
        for issue in overload_issues:
            affected_turns.extend(issue.get("turns", []))

        # 4. Check for tool/API failures in tool responses
        tool_issues = self._detect_tool_failures(tool_turns)
        issues.extend(tool_issues)
        for issue in tool_issues:
            affected_turns.extend(issue.get("turns", []))

        # 5. Check for uneven work distribution (multi-agent)
        distribution_issues = self._detect_uneven_distribution(agent_turns)
        issues.extend(distribution_issues)

        # 6. Check for token explosion (excessive token usage)
        token_issues = self._detect_token_explosion(agent_turns)
        issues.extend(token_issues)
        for issue in token_issues:
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

        # Severity based on issue count and types
        if len(issues) >= 4 or any(i["type"] == "tool_failure" for i in issues):
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
            failure_mode="F3",
            explanation=f"Resource misallocation: {len(issues)} issues found",
            affected_turns=list(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=(
                "Review resource allocation: 1) Ensure agents have required tools/access, "
                "2) Match agent capabilities to task requirements, "
                "3) Balance workload across agents, 4) Add fallback resources."
            ),
            detector_name=self.name,
        )

    def _detect_resource_complaints(self, turns: List[TurnSnapshot]) -> list:
        """Detect complaints about missing resources."""
        issues = []
        for turn in turns:
            content_lower = turn.content.lower()
            for indicator in self.RESOURCE_COMPLAINTS:
                if indicator in content_lower:
                    issues.append({
                        "type": "resource_complaint",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Resource issue: '{indicator}'",
                    })
                    break
        return issues[:4]

    def _detect_capability_mismatch(self, agent_turns: List[TurnSnapshot]) -> list:
        """Detect agents saying they can't do something."""
        issues = []
        for turn in agent_turns:
            content_lower = turn.content.lower()
            for indicator in self.CAPABILITY_MISMATCH:
                if indicator in content_lower:
                    issues.append({
                        "type": "capability_mismatch",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Capability mismatch: '{indicator}'",
                    })
                    break
        return issues[:3]

    def _detect_overload(self, agent_turns: List[TurnSnapshot]) -> list:
        """Detect overload complaints."""
        issues = []
        for turn in agent_turns:
            content_lower = turn.content.lower()
            for indicator in self.OVERLOAD_INDICATORS:
                if indicator in content_lower:
                    issues.append({
                        "type": "overload",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Overload indicator: '{indicator}'",
                    })
                    break
        return issues[:2]

    def _detect_tool_failures(self, tool_turns: List[TurnSnapshot]) -> list:
        """Detect tool/API failures in tool responses."""
        issues = []
        # Made patterns more specific to reduce FPs - require error context
        failure_indicators = [
            "error:", "error occurred", "error returned",
            "failed to", "operation failed", "request failed",
            "exception:", "traceback:", "stack trace",
            "http 404", "http 500", "status 401", "status 403",
            "connection refused", "connection failed", "timed out",
        ]
        for turn in tool_turns:
            content_lower = turn.content.lower()
            for indicator in failure_indicators:
                if indicator in content_lower:
                    issues.append({
                        "type": "tool_failure",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Tool failure: '{indicator}'",
                    })
                    break
        return issues[:3]

    def _detect_uneven_distribution(self, agent_turns: List[TurnSnapshot]) -> list:
        """Detect uneven work distribution across agents."""
        agent_counts = Counter(t.participant_id for t in agent_turns)

        if len(agent_counts) < 2:
            return []

        counts = list(agent_counts.values())
        max_count = max(counts)
        min_count = min(counts)

        # Only flag severe imbalance: 5x+ more turns and at least 8 turns for most active
        # This is a high bar to reduce FPs - true misallocation is extreme
        if max_count >= 5 * min_count and max_count >= 8:
            most_active = max(agent_counts, key=agent_counts.get)
            least_active = min(agent_counts, key=agent_counts.get)
            return [{
                "type": "uneven_distribution",
                "most_active": most_active,
                "least_active": least_active,
                "ratio": max_count / min_count if min_count > 0 else max_count,
                "description": f"Uneven workload: {most_active} has {max_count} turns vs {min_count} for {least_active}",
            }]
        return []

    def _detect_token_explosion(self, agent_turns: List[TurnSnapshot]) -> list:
        """Detect excessive token usage (token explosion)."""
        issues = []

        for turn in agent_turns:
            # Get token usage from turn_metadata
            tokens_in = turn.turn_metadata.get('tokens_input', 0)
            tokens_out = turn.turn_metadata.get('tokens_output', 0)
            total_tokens = tokens_in + tokens_out

            # Flag if total tokens exceed threshold
            if total_tokens > self.token_explosion_threshold:
                issues.append({
                    "type": "token_explosion",
                    "turn_number": turn.turn_number,
                    "participant": turn.participant_id,
                    "tokens_input": tokens_in,
                    "tokens_output": tokens_out,
                    "total_tokens": total_tokens,
                    "description": f"Excessive token usage: {total_tokens} tokens (threshold: {self.token_explosion_threshold})",
                    "turns": [turn.turn_number],
                })

        return issues
