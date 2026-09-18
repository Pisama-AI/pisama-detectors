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

from pisama_detectors.detection.precision_guards import (
    driven_by_distinct_inputs,
    outcome_is_success,
)
from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

MIN_CONSECUTIVE_REPEATS = 3
MIN_FUZZY_REPEATS = 5  # Higher threshold for fuzzy loops (same structure, different values)


# --- Fuzzy message-loop support -------------------------------------------
#
# A message-loop check that tests `prev_content == curr_content` misses an
# agent repeating itself with a few words changed. What separates a genuine
# loop with decorative variation from legitimate repetitive work (e.g.
# per-item status messages) is WHERE the variation sits: decorative variation
# is APPENDED to a shared stem; informational variation is EMBEDDED, bracketed
# by common text on both sides. A long shared prefix with a negligible shared
# suffix means the message repeated and something was tacked on the end.
_APPENDED_PREFIX_MIN = 0.25
_APPENDED_SUFFIX_MAX = 0.15

# One above the exact bar, not two. The tool lane uses MIN_FUZZY_REPEATS (5)
# because a structural hash over tool arguments is a loose match; an
# identical-stem-plus-appended-filler message is a much tighter one, so a
# lower bar (4) is warranted. The exact tier already fires at 3 identical
# messages, so 4 near-identical is consistent.
MIN_FUZZY_MESSAGE_REPEATS = 4


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _common_suffix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(reversed(a), reversed(b)):
        if x != y:
            break
        n += 1
    return n


def _variation_is_appended(a: str, b: str) -> bool:
    """True if two messages are the same message with text tacked on the end."""
    longest = max(len(a), len(b))
    if not longest:
        return False
    prefix = _common_prefix_len(a, b) / longest
    suffix = _common_suffix_len(a, b) / longest
    return prefix >= _APPENDED_PREFIX_MIN and suffix < _APPENDED_SUFFIX_MAX


def _message_content(evt: dict) -> str:
    """Message text, top-level or nested under `data`. Shared by both message tiers."""
    content = evt.get("content", "") or evt.get("message", "") or evt.get("text", "")
    if not content:
        data = evt.get("data", {})
        if isinstance(data, dict):
            content = data.get("content", "") or data.get("message", "") or data.get("text", "")
    return str(content).strip()


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

        # --- 5. Fuzzy message loop (same message, decorative variation) ---
        if not msg_loop["detected"]:
            fuzzy_msg = self._detect_fuzzy_message_loop(events)
            if fuzzy_msg["detected"]:
                issues.append(fuzzy_msg)
                affected_turns.extend(fuzzy_msg.get("turns", []))

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
            same_name = self._tool_name(prev) == self._tool_name(cur)
            same_input = _hash_input(self._tool_input(prev)) == _hash_input(self._tool_input(cur))
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
            # Outcome awareness: identical tool calls are how POLLING works.
            # An agent that calls get_export_status(job_id) four times while
            # the job walks queued -> 41% -> 88% -> completed is making
            # progress, not spinning. Only treat the run as a loop when the
            # results coming back do NOT change.
            if self._results_show_progress(events, affected):
                return {"detected": False}
            sample_evt = tool_calls[best_run_start][1]
            _name = self._tool_name(sample_evt)
            return {
                "detected": True,
                "type": "tool_call_loop",
                "repeat_count": best_run_len,
                "tool_name": _name,
                "turns": affected,
                "description": (
                    f"Tool '{_name}' called "
                    f"{best_run_len} times consecutively with identical input"
                ),
            }

        return {"detected": False}

    @staticmethod
    def _results_show_progress(events: List[dict], call_indices: List[int]) -> bool:
        """Did the tool RESULTS between these repeated calls differ?

        Distinct results mean the repetition was productive (a poll that
        advanced). Identical or absent results mean the agent really is
        re-asking the same question and getting the same answer, so the
        detector fires exactly as before.
        """
        windows: List[List[str]] = []
        results: List[Any] = []
        bounds = list(call_indices) + [len(events)]
        for start, end in zip(bounds, bounds[1:]):
            window: List[str] = []
            for evt in events[start + 1:end]:
                if evt.get("type") == "tool.call":
                    break  # a later call's result is not evidence for this poll
                if evt.get("type") != "tool.result":
                    continue
                result = evt.get("tool_result")
                results.append(result if result is not None else evt)
                window.append(
                    json.dumps(result, sort_keys=True, default=str)
                    if result is not None
                    else str(evt.get("content", ""))
                )
                break  # pair each call with its nearest following result only
            windows.append(window)
        if not driven_by_distinct_inputs(windows):
            return False

        # Different error prose is not progress. Repeated attempts that all
        # fail with a changing timeout, request ID, timestamp, or retry count
        # remain a loop. A changing result suppresses the detector only when
        # there is positive evidence of completion or measurable advancement.
        outcomes = [outcome_is_success(result) for result in results]
        if results and all(outcome is False for outcome in outcomes):
            return False

        # A terminal success after queued/running/failed observations is real
        # forward movement. Repeated calls that each return "success" with a
        # different request ID are not: every attempt was already terminal.
        if (
            outcomes
            and outcomes[-1] is True
            and any(outcome is not True for outcome in outcomes[:-1])
        ):
            return True

        return OpenClawSessionLoopDetector._has_monotonic_progress(results)

    @staticmethod
    def _has_monotonic_progress(results: List[Any]) -> bool:
        """Recognize explicit monotonic progress without trusting volatile text."""
        progress_keys = {
            "progress", "percent", "percentage", "pct", "position",
            "completed", "processed", "current", "offset",
        }
        values: List[float] = []

        def find_progress(value: Any) -> Optional[float]:
            if not isinstance(value, dict):
                return None
            for key, item in value.items():
                if str(key).lower() in progress_keys and isinstance(item, (int, float)):
                    return float(item)
            for item in value.values():
                found = find_progress(item)
                if found is not None:
                    return found
            return None

        for result in results:
            value = find_progress(result)
            if value is None:
                return False
            values.append(value)
        return len(values) >= 2 and all(b >= a for a, b in zip(values, values[1:])) and any(
            b > a for a, b in zip(values, values[1:])
        )

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

            same_name = self._tool_name(prev) == self._tool_name(cur)
            consecutive = cur_idx - prev_idx <= 4

            # Same structure (keys match, types match) even if values differ
            same_structure = _structural_hash(self._tool_input(prev)) == _structural_hash(
                self._tool_input(cur)
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
            run_inputs = [
                _hash_input(self._tool_input(tool_calls[best_run_start + j][1]))
                for j in range(best_run_len)
            ]
            if len(set(run_inputs)) == best_run_len:
                # All inputs are distinct — this is diverse tool use, not a loop.
                # A real fuzzy loop would have repeated or near-identical inputs
                # (e.g. retrying with a slightly different query each time).
                return {"detected": False}

            affected = [tool_calls[best_run_start + j][0] for j in range(best_run_len)]
            sample_evt = tool_calls[best_run_start][1]
            return {
                "detected": True,
                "type": "fuzzy_tool_loop",
                "repeat_count": best_run_len,
                "tool_name": self._tool_name(sample_evt),
                "turns": affected,
                "description": (
                    f"Tool '{self._tool_name(sample_evt)}' called "
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

    def _detect_fuzzy_message_loop(self, events: List[dict]) -> Dict[str, Any]:
        """Detect repeated message.sent whose only variation is appended text.

        Mirrors _detect_fuzzy_tool_loop, including its HIGHER bar: a fuzzy match
        is weaker evidence than an exact one, so it needs MIN_FUZZY_REPEATS
        rather than MIN_CONSECUTIVE_REPEATS.
        """
        msg_events = [
            (idx, evt)
            for idx, evt in enumerate(events)
            if evt.get("type") in ("message.sent", "message.send")
        ]
        if len(msg_events) < MIN_FUZZY_MESSAGE_REPEATS:
            return {"detected": False}

        best_run_start = run_start = 0
        best_run_len = run_len = 1
        for i in range(1, len(msg_events)):
            prev = _message_content(msg_events[i - 1][1])
            curr = _message_content(msg_events[i][1])
            if prev and curr and _variation_is_appended(prev, curr):
                run_len += 1
            else:
                if run_len > best_run_len:
                    best_run_len, best_run_start = run_len, run_start
                run_start, run_len = i, 1
        if run_len > best_run_len:
            best_run_len, best_run_start = run_len, run_start

        if best_run_len < MIN_FUZZY_MESSAGE_REPEATS:
            return {"detected": False}

        affected = [msg_events[best_run_start + j][0] for j in range(best_run_len)]
        return {
            "detected": True,
            "type": "fuzzy_message_loop",
            "repeat_count": best_run_len,
            "turns": affected,
            "description": (
                f"Agent sent {best_run_len} near-identical messages "
                f"differing only by appended text"
            ),
        }

    @staticmethod
    def _tool_name(evt: dict) -> Any:
        """Tool name, checking top-level then the nested ``data`` dict.

        Golden fixtures put ``tool_name`` at the top level, but real ingested
        sessions (the sync/webhook payload) nest it as ``data.name``. Reading
        only the top level makes every ``tool.call`` resolve to None, which
        makes the loop scan treat ALL tool calls as identical and report a
        single run spanning the whole session (confidence 1.0 on every
        session with >=5 tool calls -- even when each call was a *different*
        tool). Mirrors ``_target`` / ``_message_content`` which already check
        ``data``.
        """
        name = evt.get("tool_name")
        if name is not None:
            return name
        data = evt.get("data")
        if isinstance(data, dict):
            return data.get("name") or data.get("tool_name")
        return None

    @staticmethod
    def _tool_input(evt: dict) -> Any:
        """Tool input, checking top-level then the nested ``data`` dict."""
        ti = evt.get("tool_input")
        if ti is not None:
            return ti
        data = evt.get("data")
        if isinstance(data, dict):
            return data.get("tool_input", data.get("input", data.get("args")))
        return None

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
