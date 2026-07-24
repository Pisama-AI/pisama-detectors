"""
OpenClaw Tool Abuse Detector
=============================

Detects tool misuse patterns in OpenClaw sessions:
- Excessive total tool calls (>4 in a single session)
- Redundant calls: same tool called multiple times in rapid succession
- High error rate (>50% failures with >3 calls)
- Use of sensitive / dangerous tool names (exact + keyword match)

Mapped to failure mode F14 (Resource Abuse / Tool Misuse).
"""

import hashlib
import json
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

SENSITIVE_TOOLS: Set[str] = {
    "exec",
    "eval",
    "shell",
    "run_command",
    "system",
    "os_command",
    "write_file",
    "delete_file",
}

# Keywords in tool names that signal potential abuse
ABUSE_TOOL_KEYWORDS: Set[str] = {
    "delete",
    "ban",
    "block",
    "suspend",
    "terminate",
    "bulk",
    "mass",
    "export",
    "dump",
}

EXCESSIVE_CALL_THRESHOLD = 4
REDUNDANCY_THRESHOLD = 3  # Same tool called 3+ times = redundant
ERROR_RATE_THRESHOLD = 0.5
MIN_CALLS_FOR_ERROR_RATE = 3


def _hash_input(tool_input: Any) -> str:
    """Produce a stable hash for a tool_input value."""
    raw = json.dumps(tool_input, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class OpenClawToolAbuseDetector(TurnAwareDetector):
    """Detects F14: Tool abuse in OpenClaw sessions.

    Checks:
    1. Total tool.call count exceeds threshold (>4).
    2. Redundant calls: same tool called 3+ times (with same or similar inputs).
    3. Error rate (tool.result with status='failed') exceeds 50% when >3 calls.
    4. Calls to sensitive/dangerous tools.
    """

    name = "OpenClawToolAbuseDetector"
    version = "1.1"
    supported_failure_modes = ["F14"]

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        session = (conversation_metadata or {}).get("session", {})
        return self.detect_session(session)

    def detect_session(self, session: dict) -> TurnAwareDetectionResult:
        events = session.get("events", [])
        if not events:
            return self._no_detection("No events in session")

        tool_calls = [e for e in events if e.get("type") == "tool.call"]
        tool_results = [e for e in events if e.get("type") == "tool.result"]
        total_calls = len(tool_calls)

        violations: List[Dict[str, Any]] = []
        affected_turns: List[int] = []

        # --- 1. Excessive tool calls ---
        if total_calls > EXCESSIVE_CALL_THRESHOLD:
            call_indices = [i for i, e in enumerate(events) if e.get("type") == "tool.call"]
            violations.append(
                {
                    "type": "excessive_calls",
                    "total_calls": total_calls,
                    "threshold": EXCESSIVE_CALL_THRESHOLD,
                    "description": (
                        f"{total_calls} tool calls exceed threshold of {EXCESSIVE_CALL_THRESHOLD}"
                    ),
                }
            )
            affected_turns.extend(call_indices)

        # --- 2. Redundant calls (same tool name repeated) ---
        tool_name_counts: Dict[str, List[int]] = {}
        for i, evt in enumerate(events):
            if evt.get("type") == "tool.call":
                name = (evt.get("tool_name") or "").lower()
                tool_name_counts.setdefault(name, []).append(i)

        for name, indices in tool_name_counts.items():
            if len(indices) >= REDUNDANCY_THRESHOLD:
                violations.append(
                    {
                        "type": "redundant_calls",
                        "tool_name": name,
                        "call_count": len(indices),
                        "description": (
                            f"Tool '{name}' called {len(indices)} times "
                            f"(redundancy threshold: {REDUNDANCY_THRESHOLD})"
                        ),
                    }
                )
                affected_turns.extend(indices)

        # --- 3. High error rate ---
        if total_calls >= MIN_CALLS_FOR_ERROR_RATE:
            failed = sum(
                1 for r in tool_results if (r.get("tool_result") or {}).get("status") == "failed"
            )
            error_rate = failed / total_calls if total_calls else 0.0

            if error_rate > ERROR_RATE_THRESHOLD:
                fail_indices = [
                    i
                    for i, e in enumerate(events)
                    if e.get("type") == "tool.result"
                    and (e.get("tool_result") or {}).get("status") == "failed"
                ]
                violations.append(
                    {
                        "type": "high_error_rate",
                        "error_rate": round(error_rate, 3),
                        "failed_calls": failed,
                        "total_calls": total_calls,
                        "description": (
                            f"Error rate {error_rate:.0%} ({failed}/{total_calls}) exceeds 50%"
                        ),
                    }
                )
                affected_turns.extend(fail_indices)

        # --- 4. Sensitive tool usage (exact + keyword match) ---
        sensitive_used: List[Dict[str, Any]] = []
        for i, evt in enumerate(events):
            if evt.get("type") == "tool.call":
                tool_name = (evt.get("tool_name") or "").lower()
                # Exact match
                if tool_name in SENSITIVE_TOOLS:
                    sensitive_used.append(
                        {
                            "index": i,
                            "tool_name": evt.get("tool_name"),
                            "match": "exact",
                        }
                    )
                    affected_turns.append(i)
                else:
                    # Keyword match
                    name_parts = set(re.split(r"[_\-.\s]", tool_name))
                    matches = name_parts & ABUSE_TOOL_KEYWORDS
                    if matches:
                        sensitive_used.append(
                            {
                                "index": i,
                                "tool_name": evt.get("tool_name"),
                                "match": "keyword",
                                "keywords": list(matches),
                            }
                        )
                        affected_turns.append(i)

        if sensitive_used:
            tool_names = list({s["tool_name"] for s in sensitive_used})
            violations.append(
                {
                    "type": "sensitive_tools",
                    "tools_used": tool_names,
                    "call_count": len(sensitive_used),
                    "description": f"Sensitive tools used: {', '.join(tool_names)}",
                }
            )

        if not violations:
            return self._no_detection("No tool abuse detected")

        # Confidence scales with number of violation types
        confidence = min(1.0, 0.4 + len(violations) * 0.25)

        # Severity: sensitive tools or multiple violations = higher severity
        has_sensitive = any(v["type"] == "sensitive_tools" for v in violations)
        has_redundant = any(v["type"] == "redundant_calls" for v in violations)
        if has_sensitive and len(violations) >= 2:
            severity = TurnAwareSeverity.SEVERE
        elif has_sensitive or len(violations) >= 2 or has_redundant:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F14",
            explanation=(
                f"Tool abuse detected: {len(violations)} violation(s) in {total_calls} tool calls"
            ),
            affected_turns=sorted(set(affected_turns)),
            evidence={
                "violations": violations,
                "total_tool_calls": total_calls,
                "total_events": len(events),
            },
            suggested_fix=(
                "Limit tool call frequency, add retry budgets, and restrict "
                "access to sensitive tools via sandbox or allowlists."
            ),
            detector_name=self.name,
        )

    def _no_detection(self, explanation: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=explanation,
            detector_name=self.name,
        )
