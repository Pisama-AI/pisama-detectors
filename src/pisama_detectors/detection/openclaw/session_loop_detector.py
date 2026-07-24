"""
OpenClaw Session Loop Detector
==============================

Detects repetitive patterns in OpenClaw session event streams:
- Consecutive identical tool calls (same name + input hash)
- Fuzzy tool call loops (same tool, same keys, only minor value changes)
- Ping-pong patterns between session.spawn/send and a fixed target
- A-B-A-B alternating ping-pong between two targets
- Repeated message.sent events with identical content

Mapped to failure mode F8 (Infinite Loop / Repetitive Behavior).
"""

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_REPEATS = 3
MIN_FUZZY_REPEATS = 5  # Higher threshold for fuzzy loops (same structure, different values)


def _hash_input(tool_input: Any) -> str:
    """Produce a stable hash for a tool_input value."""
    raw = json.dumps(tool_input, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _structural_hash(tool_input: Any) -> str:
    """Hash that captures structure (keys) but replaces values with type placeholders."""
    if isinstance(tool_input, dict):
        normalized = {k: type(v).__name__ for k, v in sorted(tool_input.items())}
        raw = json.dumps(normalized, sort_keys=True)
    else:
        raw = type(tool_input).__name__
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class OpenClawSessionLoopDetector(TurnAwareDetector):
    """Detects F8: Session-level loops in OpenClaw event streams.

    Looks for:
    1. Consecutive identical tool.call events (same tool_name + tool_input hash).
       3+ identical consecutive calls triggers detection.
    2. Fuzzy tool call loops (same tool, same keys, only 1 value differs).
    3. Ping-pong patterns where session.spawn or session.send targets the
       same session/agent repeatedly, including A-B-A-B alternation.
    4. Repeated message.sent events with identical or near-identical content.
    """

    name = "OpenClawSessionLoopDetector"
    version = "1.1"
    supported_failure_modes = ["F8"]

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Delegate to detect_session when called via the base interface."""
        session = (conversation_metadata or {}).get("session", {})
        return self.detect_session(session)

    def detect_session(self, session: dict) -> TurnAwareDetectionResult:
        events = session.get("events", [])
        if not events:
            return self._no_detection("No events in session")

        issues: List[Dict[str, Any]] = []
        affected_turns: List[int] = []

        # --- 1. Consecutive identical tool calls ---
        tool_loop = self._detect_tool_call_loop(events)
        if tool_loop["detected"]:
            issues.append(tool_loop)
            affected_turns.extend(tool_loop.get("turns", []))

        # --- 2. Fuzzy tool call loop (same keys, minor value changes) ---
        if not tool_loop["detected"]:
            fuzzy_loop = self._detect_fuzzy_tool_loop(events)
            if fuzzy_loop["detected"]:
                issues.append(fuzzy_loop)
                affected_turns.extend(fuzzy_loop.get("turns", []))

        # --- 3. Spawn / send ping-pong ---
        ping_pong = self._detect_spawn_ping_pong(events)
        if ping_pong["detected"]:
            issues.append(ping_pong)
            affected_turns.extend(ping_pong.get("turns", []))

        # --- 4. Repeated message.sent events ---
        msg_loop = self._detect_message_sent_loop(events)
        if msg_loop["detected"]:
            issues.append(msg_loop)
            affected_turns.extend(msg_loop.get("turns", []))

        if not issues:
            return self._no_detection("No loop patterns detected")

        max_repeats = max(i.get("repeat_count", 0) for i in issues)
        confidence = min(1.0, max_repeats / 5)

        if max_repeats >= 6:
            severity = TurnAwareSeverity.SEVERE
        elif max_repeats >= 4:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F8",
            explanation=f"Session loop detected: {len(issues)} pattern(s), max {max_repeats} repeats",
            affected_turns=sorted(set(affected_turns)),
            evidence={"issues": issues, "total_events": len(events)},
            suggested_fix=(
                "Add loop-breaking conditions or retry limits. "
                "Ensure tool calls advance state rather than repeating identically."
            ),
            detector_name=self.name,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_tool_call_loop(self, events: List[dict]) -> Dict[str, Any]:
        """Find runs of consecutive tool.call events with identical name+input."""
        tool_calls = [
            (idx, evt) for idx, evt in enumerate(events) if evt.get("type") == "tool.call"
        ]

        if len(tool_calls) < MIN_CONSECUTIVE_REPEATS:
            return {"detected": False}

        best_run_start = 0
        best_run_len = 1
        run_start = 0
        run_len = 1

        for i in range(1, len(tool_calls)):
            prev_idx, prev = tool_calls[i - 1]
            cur_idx, cur = tool_calls[i]

            # Allow intermediate events (agent.turn, tool.result) between tool calls
            same_name = prev.get("tool_name") == cur.get("tool_name")
            same_input = _hash_input(prev.get("tool_input")) == _hash_input(cur.get("tool_input"))
            consecutive = cur_idx - prev_idx <= 4  # allow up to 3 events between tool calls

            if same_name and same_input and consecutive:
                run_len += 1
            else:
                if run_len > best_run_len:
                    best_run_len = run_len
                    best_run_start = run_start
                run_start = i
                run_len = 1

        if run_len > best_run_len:
            best_run_len = run_len
            best_run_start = run_start

        if best_run_len >= MIN_CONSECUTIVE_REPEATS:
            affected = [tool_calls[best_run_start + j][0] for j in range(best_run_len)]
            sample_evt = tool_calls[best_run_start][1]
            return {
                "detected": True,
                "type": "tool_call_loop",
                "repeat_count": best_run_len,
                "tool_name": sample_evt.get("tool_name"),
                "turns": affected,
                "description": (
                    f"Tool '{sample_evt.get('tool_name')}' called "
                    f"{best_run_len} times consecutively with identical input"
                ),
            }

        return {"detected": False}

    def _detect_fuzzy_tool_loop(self, events: List[dict]) -> Dict[str, Any]:
        """Detect tool call loops where inputs have same keys but minor value changes."""
        tool_calls = [
            (idx, evt) for idx, evt in enumerate(events) if evt.get("type") == "tool.call"
        ]

        if len(tool_calls) < MIN_FUZZY_REPEATS:
            return {"detected": False}

        best_run_start = 0
        best_run_len = 1
        run_start = 0
        run_len = 1

        for i in range(1, len(tool_calls)):
            prev_idx, prev = tool_calls[i - 1]
            cur_idx, cur = tool_calls[i]

            same_name = prev.get("tool_name") == cur.get("tool_name")
            consecutive = cur_idx - prev_idx <= 4

            # Same structure (keys match, types match) even if values differ
            same_structure = _structural_hash(prev.get("tool_input")) == _structural_hash(
                cur.get("tool_input")
            )

            if same_name and same_structure and consecutive:
                run_len += 1
            else:
                if run_len > best_run_len:
                    best_run_len = run_len
                    best_run_start = run_start
                run_start = i
                run_len = 1

        if run_len > best_run_len:
            best_run_len = run_len
            best_run_start = run_start

        if best_run_len >= MIN_FUZZY_REPEATS:
            affected = [tool_calls[best_run_start + j][0] for j in range(best_run_len)]
            sample_evt = tool_calls[best_run_start][1]
            return {
                "detected": True,
                "type": "fuzzy_tool_loop",
                "repeat_count": best_run_len,
                "tool_name": sample_evt.get("tool_name"),
                "turns": affected,
                "description": (
                    f"Tool '{sample_evt.get('tool_name')}' called "
                    f"{best_run_len} times with same structure but varying values"
                ),
            }

        return {"detected": False}

    def _detect_spawn_ping_pong(self, events: List[dict]) -> Dict[str, Any]:
        """Detect alternating spawn/send events targeting the same session.

        Also detects A-B-A-B alternation between two targets.
        """
        spawn_send = [
            (idx, evt)
            for idx, evt in enumerate(events)
            if evt.get("type") in ("session.spawn", "session.send")
        ]

        if len(spawn_send) < MIN_CONSECUTIVE_REPEATS:
            return {"detected": False}

        targets = [(idx, self._target(evt)) for idx, evt in spawn_send]
        targets = [(idx, t) for idx, t in targets if t]

        if len(targets) < MIN_CONSECUTIVE_REPEATS:
            return {"detected": False}

        # --- Check 1: Same target repeated ---
        best_count = 1
        best_target = ""
        best_indices: List[int] = []
        count = 1
        cur_target = targets[0][1]
        indices = [targets[0][0]]

        for i in range(1, len(targets)):
            t = targets[i][1]
            if t == cur_target:
                count += 1
                indices.append(targets[i][0])
            else:
                if count > best_count:
                    best_count = count
                    best_target = cur_target
                    best_indices = list(indices)
                cur_target = t
                count = 1
                indices = [targets[i][0]]

        if count > best_count:
            best_count = count
            best_target = cur_target
            best_indices = list(indices)

        if best_count >= MIN_CONSECUTIVE_REPEATS:
            return {
                "detected": True,
                "type": "spawn_ping_pong",
                "repeat_count": best_count,
                "target": best_target,
                "turns": best_indices,
                "description": (
                    f"Spawn/send ping-pong: {best_count} events targeting '{best_target}'"
                ),
            }

        # --- Check 2: A-B-A-B alternation between two targets ---
        abab = self._detect_abab_pattern(targets)
        if abab:
            return abab

        return {"detected": False}

    def _detect_abab_pattern(self, targets: List[Tuple[int, str]]) -> Optional[Dict[str, Any]]:
        """Detect A-B-A-B alternating pattern between two agents."""
        if len(targets) < 4:
            return None

        best_len = 0
        best_start = 0
        best_pair: Tuple[str, str] = ("", "")

        for start in range(len(targets) - 3):
            a = targets[start][1]
            b = targets[start + 1][1]
            if a == b:
                continue

            run_len = 2
            for j in range(start + 2, len(targets)):
                expected = a if (j - start) % 2 == 0 else b
                if targets[j][1] == expected:
                    run_len += 1
                else:
                    break

            if run_len > best_len:
                best_len = run_len
                best_start = start
                best_pair = (a, b)

        # 4+ alternations (A-B-A-B) = 2 full cycles
        if best_len >= 4:
            affected = [targets[best_start + j][0] for j in range(best_len)]
            return {
                "detected": True,
                "type": "abab_ping_pong",
                "repeat_count": best_len,
                "targets": list(best_pair),
                "turns": affected,
                "description": (
                    f"A-B-A-B ping-pong: {best_len} alternating events "
                    f"between '{best_pair[0]}' and '{best_pair[1]}'"
                ),
            }

        return None

    def _detect_message_sent_loop(self, events: List[dict]) -> Dict[str, Any]:
        """Detect repeated message.sent events with identical or near-identical content."""
        msg_events = [
            (idx, evt)
            for idx, evt in enumerate(events)
            if evt.get("type") in ("message.sent", "message.send")
        ]

        if len(msg_events) < MIN_CONSECUTIVE_REPEATS:
            return {"detected": False}

        def _msg_content(evt: dict) -> str:
            # Content can be top-level or in data dict
            content = evt.get("content", "") or evt.get("message", "") or evt.get("text", "")
            if not content:
                data = evt.get("data", {})
                if isinstance(data, dict):
                    content = (
                        data.get("content", "") or data.get("message", "") or data.get("text", "")
                    )
            return str(content).strip()

        # Find runs of identical message content
        best_run_len = 1
        best_run_start = 0
        run_len = 1
        run_start = 0

        for i in range(1, len(msg_events)):
            prev_content = _msg_content(msg_events[i - 1][1])
            curr_content = _msg_content(msg_events[i][1])

            if prev_content and prev_content == curr_content:
                run_len += 1
            else:
                if run_len > best_run_len:
                    best_run_len = run_len
                    best_run_start = run_start
                run_start = i
                run_len = 1

        if run_len > best_run_len:
            best_run_len = run_len
            best_run_start = run_start

        if best_run_len >= MIN_CONSECUTIVE_REPEATS:
            affected = [msg_events[best_run_start + j][0] for j in range(best_run_len)]
            sample_content = _msg_content(msg_events[best_run_start][1])
            return {
                "detected": True,
                "type": "message_sent_loop",
                "repeat_count": best_run_len,
                "turns": affected,
                "content_preview": sample_content[:100],
                "description": (f"Message sent {best_run_len} times with identical content"),
            }

        return {"detected": False}

    def _target(self, evt: dict) -> str:
        """Extract target agent/session from event, checking nested data dict."""
        # Check top-level fields first
        target = (
            evt.get("target_session", "")
            or evt.get("target_agent", "")
            or evt.get("spawned_session_id", "")
        )
        if target:
            return target

        # Check nested data dict
        data = evt.get("data", {})
        if isinstance(data, dict):
            target = (
                data.get("target_agent", "")
                or data.get("target_session", "")
                or data.get("recipient", "")
                or data.get("recipient_agent", "")
                or data.get("target", "")
                or data.get("source_agent", "")
                or data.get("spawned_session_id", "")
                or data.get("agent", "")
            )
        return target or ""

    def _no_detection(self, explanation: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=explanation,
            detector_name=self.name,
        )
