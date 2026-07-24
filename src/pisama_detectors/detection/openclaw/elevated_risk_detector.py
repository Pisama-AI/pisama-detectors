"""
OpenClaw Elevated Risk Detector
================================

Detects risky operations relative to the session's privilege level:
- elevated_mode=True with risky tool calls (admin actions, data ops, code exec)
- elevated_mode=False but risky tools present = escalation attempt (higher severity)

Uses keyword-based matching to catch both OS-level and domain-specific risky tools.

Mapped to failure mode F5 (Privilege Escalation / Security Risk).
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

# Exact tool name matches (OS-level risky tools) — these are tool *names*, not
# verbs, so they stay local rather than living in the shared verb taxonomy.
RISKY_TOOLS_EXACT: Set[str] = {
    "read_file",
    "write_file",
    "delete_file",
    "list_dir",
    "exec",
    "eval",
    "run_code",
    "shell",
    "system",
    "os_command",
    "run_command",
    "subprocess",
}

# Keyword-based risky tool detection — sourced from the shared safety taxonomy.
from pisama_detectors.detection.safety import OPENCLAW_RISKY_KEYWORDS as RISKY_KEYWORDS

# Risky patterns in tool input data
RISKY_INPUT_PATTERNS = [
    re.compile(r"(?:rm|del|remove)\s+.*-(?:rf|r)", re.IGNORECASE),
    re.compile(r"/etc/(?:passwd|shadow|sudoers)", re.IGNORECASE),
    re.compile(r"(?:ssn|social.security|credit.card)", re.IGNORECASE),
    re.compile(r"(?:bulk|mass|all)\s+(?:delete|remove|ban|reset)", re.IGNORECASE),
]


class OpenClawElevatedRiskDetector(TurnAwareDetector):
    """Detects F5: Elevated privilege risks in OpenClaw sessions.

    Uses keyword-based matching to detect risky operations including:
    - OS-level tools (exec, write_file, shell)
    - Domain-specific admin actions (delete_user, modify_permissions, ban_user)
    - Risky data operations (bulk export, data dump, PII access)
    """

    name = "OpenClawElevatedRiskDetector"
    version = "1.1"
    supported_failure_modes = ["F5"]

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        session = (conversation_metadata or {}).get("session", {})
        return self.detect_session(session)

    def detect_session(self, session: dict) -> TurnAwareDetectionResult:
        events = session.get("events", [])
        elevated_mode = session.get("elevated_mode", False)

        if not events:
            return self._no_detection("No events in session")

        risky_calls: List[Dict[str, Any]] = []
        affected_turns: List[int] = []

        for i, evt in enumerate(events):
            if evt.get("type") != "tool.call":
                continue
            tool_name = (evt.get("tool_name") or "").lower()
            tool_input = evt.get("tool_input", {})

            # Check for inherently risky tools (keyword/pattern match)
            risk = self._assess_tool_risk(tool_name, tool_input)
            if risk:
                risky_calls.append(
                    {
                        "index": i,
                        "tool_name": evt.get("tool_name"),
                        "category": risk["category"],
                        "reason": risk["reason"],
                        "tool_input": tool_input,
                    }
                )
                affected_turns.append(i)
            elif elevated_mode:
                # In elevated mode, ANY tool call is potentially risky
                risky_calls.append(
                    {
                        "index": i,
                        "tool_name": evt.get("tool_name"),
                        "category": "elevated_operation",
                        "reason": f"Tool '{tool_name}' called in elevated mode session",
                        "tool_input": tool_input,
                    }
                )
                affected_turns.append(i)

        if not risky_calls:
            return self._no_detection("No risky tool calls found")

        is_escalation = not elevated_mode
        categories_hit = list({r["category"] for r in risky_calls})

        confidence = min(1.0, 0.7 + len(risky_calls) * 0.1)

        if is_escalation:
            severity = TurnAwareSeverity.SEVERE
            explanation = (
                f"Escalation attempt: {len(risky_calls)} risky tool call(s) "
                f"in non-elevated session (categories: {', '.join(categories_hit)})"
            )
        else:
            if len(risky_calls) >= 5 or len(categories_hit) >= 3:
                severity = TurnAwareSeverity.MODERATE
            else:
                severity = TurnAwareSeverity.MINOR
            explanation = (
                f"Elevated session with {len(risky_calls)} risky tool call(s) "
                f"(categories: {', '.join(categories_hit)})"
            )

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F5",
            explanation=explanation,
            affected_turns=sorted(set(affected_turns)),
            evidence={
                "elevated_mode": elevated_mode,
                "is_escalation_attempt": is_escalation,
                "risky_calls": risky_calls,
                "categories": categories_hit,
            },
            suggested_fix=(
                "Restrict risky tool access in non-elevated sessions. "
                "Use allowlists per privilege level and require explicit "
                "elevation before granting admin/data/code execution access."
            ),
            detector_name=self.name,
        )

    def _assess_tool_risk(self, tool_name: str, tool_input: Any) -> Optional[Dict[str, str]]:
        """Assess whether a tool call is risky using exact match + keyword matching."""
        # Exact match against known OS-level risky tools
        if tool_name in RISKY_TOOLS_EXACT:
            category = "system_commands"
            for cat, tools in {
                "file_system": {"read_file", "write_file", "delete_file", "list_dir"},
                "code_execution": {"exec", "eval", "run_code"},
                "system_commands": {"shell", "system", "os_command", "run_command", "subprocess"},
            }.items():
                if tool_name in tools:
                    category = cat
                    break
            return {"category": category, "reason": f"Known risky tool: {tool_name}"}

        # Keyword-based matching for domain-specific tools
        name_parts = set(re.split(r"[_\-.\s]", tool_name))
        for category, keywords in RISKY_KEYWORDS.items():
            matches = name_parts & keywords
            if matches:
                return {
                    "category": category,
                    "reason": f"Tool name contains risky keywords: {', '.join(matches)}",
                }

        # Check tool input for risky patterns
        if tool_input:
            input_str = str(tool_input)
            for pattern in RISKY_INPUT_PATTERNS:
                if pattern.search(input_str):
                    return {
                        "category": "risky_input",
                        "reason": f"Tool input matches risky pattern: {pattern.pattern}",
                    }

        return None

    def _no_detection(self, explanation: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=explanation,
            detector_name=self.name,
        )
