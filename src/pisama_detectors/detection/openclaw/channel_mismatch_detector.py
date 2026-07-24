"""
OpenClaw Channel Mismatch Detector
====================================

Detects channel inconsistencies in OpenClaw sessions:
1. Cross-channel routing: events with a different channel than the session-level channel
2. Content format violations: messages inappropriate for the delivery channel
   - WhatsApp: code blocks, markdown tables, messages >1000 chars
   - Telegram: messages >4096 chars (platform limit)
   - Slack: PII patterns (SSN, credit card numbers)
   - Discord: messages >2000 chars (platform limit)

Mapped to failure mode F6 (Communication Failure / Format Mismatch).
"""

import logging
import re
from typing import Any, Dict, List, Optional

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

# PII patterns for Slack channel checks
SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")

# Markdown / formatting patterns for WhatsApp
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```")
MARKDOWN_TABLE_PATTERN = re.compile(r"\|[-:]+\|")


class OpenClawChannelMismatchDetector(TurnAwareDetector):
    """Detects F6: Channel-inappropriate content in OpenClaw sessions.

    Primary check: cross-channel routing (event-level channel differs from
    session-level channel).
    Secondary check: message content formatting violations for the channel.
    """

    name = "OpenClawChannelMismatchDetector"
    version = "1.1"
    supported_failure_modes = ["F6"]

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        session = (conversation_metadata or {}).get("session", {})
        return self.detect_session(session)

    def detect_session(self, session: dict) -> TurnAwareDetectionResult:
        events = session.get("events", [])
        session_channel = (session.get("channel") or "web").lower()

        if not events:
            return self._no_detection("No events in session")

        violations: List[Dict[str, Any]] = []
        affected_turns: List[int] = []

        # --- Check 1: Cross-channel routing ---
        for i, evt in enumerate(events):
            evt_channel = (evt.get("channel") or "").lower()
            if evt_channel and evt_channel != session_channel:
                violations.append(
                    {
                        "type": "cross_channel_routing",
                        "event_index": i,
                        "event_type": evt.get("type", ""),
                        "session_channel": session_channel,
                        "event_channel": evt_channel,
                        "issue": (
                            f"Event routed to '{evt_channel}' but session channel "
                            f"is '{session_channel}'"
                        ),
                        "severity": "moderate",
                    }
                )
                affected_turns.append(i)

        # --- Check 2: Message content formatting ---
        for i, evt in enumerate(events):
            if evt.get("type") not in ("message.sent", "agent.message"):
                continue
            content = self._extract_content(evt)
            if not content:
                continue

            msg_violations = self._check_channel(session_channel, content, i)
            if msg_violations:
                violations.extend(msg_violations)
                affected_turns.append(i)

        if not violations:
            return self._no_detection(f"All events consistent with {session_channel} channel")

        # Confidence scales with violation count
        confidence = min(1.0, 0.5 + len(violations) * 0.15)

        # Severity based on violation types
        has_pii = any(v.get("type") == "pii_exposure" for v in violations)
        has_cross_channel = any(v.get("type") == "cross_channel_routing" for v in violations)

        if has_pii:
            severity = TurnAwareSeverity.SEVERE
        elif has_cross_channel and len(violations) >= 2:
            severity = TurnAwareSeverity.MODERATE
        elif has_cross_channel or len(violations) >= 3:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F6",
            explanation=(
                f"Channel mismatch: {len(violations)} violation(s) "
                f"for {session_channel} channel across {len(events)} events"
            ),
            affected_turns=sorted(set(affected_turns)),
            evidence={
                "channel": session_channel,
                "violations": violations,
                "total_events": len(events),
            },
            suggested_fix=(
                "Ensure all events route through the session's designated channel. "
                "Adapt message formatting per channel. Truncate long messages, "
                "convert markdown to plain text for WhatsApp, and never expose "
                "PII in any channel."
            ),
            detector_name=self.name,
        )

    # ------------------------------------------------------------------
    # Content extraction
    # ------------------------------------------------------------------

    def _extract_content(self, evt: dict) -> str:
        """Extract message content from event, checking multiple key locations."""
        data = evt.get("data") or {}
        # Try multiple content keys used by different data generators
        for key in ("content", "message", "text"):
            val = data.get(key, "")
            if val:
                return str(val)
        # Fall back to direct event keys
        for key in ("content", "message", "text"):
            val = evt.get(key, "")
            if val:
                return str(val)
        return ""

    # ------------------------------------------------------------------
    # Channel-specific checks
    # ------------------------------------------------------------------

    def _check_channel(self, channel: str, content: str, event_index: int) -> List[Dict[str, Any]]:
        violations: List[Dict[str, Any]] = []

        if channel == "whatsapp":
            violations.extend(self._check_whatsapp(content, event_index))
        elif channel == "telegram":
            violations.extend(self._check_telegram(content, event_index))
        elif channel == "slack":
            violations.extend(self._check_slack(content, event_index))
        elif channel == "discord":
            violations.extend(self._check_discord(content, event_index))

        return violations

    def _check_whatsapp(self, content: str, idx: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if CODE_BLOCK_PATTERN.search(content):
            results.append(
                {
                    "type": "formatting",
                    "event_index": idx,
                    "issue": "Code blocks not rendered on WhatsApp",
                    "severity": "minor",
                }
            )

        if MARKDOWN_TABLE_PATTERN.search(content):
            results.append(
                {
                    "type": "formatting",
                    "event_index": idx,
                    "issue": "Markdown tables not rendered on WhatsApp",
                    "severity": "minor",
                }
            )

        if len(content) > 1000:
            results.append(
                {
                    "type": "length",
                    "event_index": idx,
                    "issue": f"Message too long for WhatsApp ({len(content)} chars, max 1000)",
                    "severity": "moderate",
                }
            )

        return results

    def _check_telegram(self, content: str, idx: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if len(content) > 4096:
            results.append(
                {
                    "type": "length",
                    "event_index": idx,
                    "issue": f"Exceeds Telegram limit ({len(content)} chars, max 4096)",
                    "severity": "moderate",
                }
            )

        return results

    def _check_slack(self, content: str, idx: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if SSN_PATTERN.search(content):
            results.append(
                {
                    "type": "pii_exposure",
                    "event_index": idx,
                    "issue": "Potential SSN detected in Slack message",
                    "severity": "severe",
                }
            )

        if CREDIT_CARD_PATTERN.search(content):
            results.append(
                {
                    "type": "pii_exposure",
                    "event_index": idx,
                    "issue": "Potential credit card number in Slack message",
                    "severity": "severe",
                }
            )

        return results

    def _check_discord(self, content: str, idx: int) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        if len(content) > 2000:
            results.append(
                {
                    "type": "length",
                    "event_index": idx,
                    "issue": f"Exceeds Discord limit ({len(content)} chars, max 2000)",
                    "severity": "moderate",
                }
            )

        return results

    def _no_detection(self, explanation: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=explanation,
            detector_name=self.name,
        )
