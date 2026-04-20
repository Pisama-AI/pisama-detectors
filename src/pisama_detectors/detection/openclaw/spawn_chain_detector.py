"""
OpenClaw Spawn Chain Detector
==============================

Detects suspicious session.spawn chains in OpenClaw:
- Spawn depth exceeding threshold (counts total child sessions, not just events)
- Circular references (spawned session ID equals current or prior IDs)
- Privilege escalation patterns across spawns
- Depth field exceeding safe threshold

Mapped to failure mode F11 (Coordination Failure / Delegation Storm).
"""

import logging
from typing import Any, Dict, List, Optional, Set

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

MAX_SAFE_SPAWN_DEPTH = 3

# Keywords suggesting higher-privilege agents
PRIVILEGE_KEYWORDS = {
    "admin", "root", "supervisor", "manager", "elevated", "privileged",
    "system", "internal", "master", "controller",
}


class OpenClawSpawnChainDetector(TurnAwareDetector):
    """Detects F11: Spawn chain issues in OpenClaw sessions.

    Checks:
    1. Total child sessions spawned (from data fields) exceeding threshold.
    2. Circular references: a spawned_session_id matching the current session
       or any previously spawned session.
    3. Privilege escalation: targeting higher-privilege agents.
    4. Depth field: explicit depth indicator exceeding safe threshold.
    """

    name = "OpenClawSpawnChainDetector"
    version = "1.1"
    supported_failure_modes = ["F11"]

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        session = (conversation_metadata or {}).get("session", {})
        return self.detect_session(session)

    def detect_session(self, session: dict) -> TurnAwareDetectionResult:
        events = session.get("events", [])
        session_id = session.get("session_id", "")

        if not events:
            return self._no_detection("No events in session")

        spawn_events = [
            (i, evt)
            for i, evt in enumerate(events)
            if evt.get("type") == "session.spawn"
        ]

        if not spawn_events:
            return self._no_detection("No spawn events in session")

        issues: List[Dict[str, Any]] = []
        affected_turns: List[int] = []

        # Count total child sessions from data fields (not just event count)
        total_children = self._count_total_children(spawn_events)
        spawn_count = max(len(spawn_events), total_children)

        # Check max depth from data fields
        max_depth = self._get_max_depth(spawn_events)

        # --- 1. Excessive spawn depth ---
        if spawn_count > MAX_SAFE_SPAWN_DEPTH:
            indices = [idx for idx, _ in spawn_events]
            issues.append({
                "type": "excessive_depth",
                "spawn_count": spawn_count,
                "threshold": MAX_SAFE_SPAWN_DEPTH,
                "description": f"{spawn_count} spawns exceed safe depth of {MAX_SAFE_SPAWN_DEPTH}",
            })
            affected_turns.extend(indices)

        # --- 1b. Depth field check ---
        if max_depth > MAX_SAFE_SPAWN_DEPTH and not issues:
            indices = [idx for idx, _ in spawn_events]
            issues.append({
                "type": "excessive_depth",
                "spawn_count": spawn_count,
                "depth": max_depth,
                "threshold": MAX_SAFE_SPAWN_DEPTH,
                "description": (
                    f"Spawn depth {max_depth} exceeds safe threshold "
                    f"of {MAX_SAFE_SPAWN_DEPTH}"
                ),
            })
            affected_turns.extend(indices)

        # --- 2. Circular references ---
        seen_ids: Set[str] = {session_id} if session_id else set()
        for idx, evt in spawn_events:
            spawned_id = self._get_data_field(evt, "spawned_session_id")
            if spawned_id and spawned_id in seen_ids:
                issues.append({
                    "type": "circular_reference",
                    "spawned_session_id": spawned_id,
                    "event_index": idx,
                    "description": f"Circular spawn: '{spawned_id}' already seen",
                })
                affected_turns.append(idx)
            if spawned_id:
                seen_ids.add(spawned_id)

            # Also check child_session_ids list for circular refs
            child_ids = self._get_child_session_ids(evt)
            for cid in child_ids:
                if cid in seen_ids:
                    issues.append({
                        "type": "circular_reference",
                        "spawned_session_id": cid,
                        "event_index": idx,
                        "description": f"Circular spawn: child '{cid}' already seen",
                    })
                    affected_turns.append(idx)
                seen_ids.add(cid)

        # --- 3. Privilege escalation pattern ---
        targets: List[tuple] = []
        for idx, evt in spawn_events:
            agent = self._get_target_agent(evt).lower()
            if agent:
                targets.append((idx, agent))

        escalation_chain: List[str] = []
        escalation_indices: List[int] = []
        for idx, agent in targets:
            if any(kw in agent for kw in PRIVILEGE_KEYWORDS):
                escalation_chain.append(agent)
                escalation_indices.append(idx)

        if len(escalation_chain) >= 2:
            issues.append({
                "type": "privilege_escalation",
                "chain": escalation_chain,
                "description": (
                    f"Privilege escalation pattern: spawning {len(escalation_chain)} "
                    f"privileged agents ({', '.join(escalation_chain)})"
                ),
            })
            affected_turns.extend(escalation_indices)

        if not issues:
            return self._no_detection("Spawn chain within safe limits")

        confidence = min(1.0, spawn_count / 4)

        has_circular = any(i["type"] == "circular_reference" for i in issues)
        has_escalation = any(i["type"] == "privilege_escalation" for i in issues)

        if has_circular:
            severity = TurnAwareSeverity.SEVERE
        elif has_escalation or spawn_count > MAX_SAFE_SPAWN_DEPTH * 2:
            severity = TurnAwareSeverity.SEVERE
        elif spawn_count > MAX_SAFE_SPAWN_DEPTH:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F11",
            explanation=f"Spawn chain issue: {len(issues)} problem(s) across {spawn_count} spawns",
            affected_turns=sorted(set(affected_turns)),
            evidence={
                "issues": issues,
                "spawn_count": spawn_count,
                "max_depth": max_depth,
                "session_id": session_id,
                "spawned_ids": list(seen_ids - {session_id}),
            },
            suggested_fix=(
                "Limit spawn chain depth with a max_depth parameter. "
                "Track visited session IDs to prevent circular spawning. "
                "Require explicit authorization for privileged agent spawns."
            ),
            detector_name=self.name,
        )

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------

    def _get_data_field(self, evt: dict, field: str) -> str:
        """Get a field from event, checking both top-level and nested data dict."""
        val = evt.get(field, "")
        if val:
            return str(val)
        data = evt.get("data", {})
        if isinstance(data, dict):
            val = data.get(field, "")
        return str(val) if val else ""

    def _get_target_agent(self, evt: dict) -> str:
        """Extract target agent name from event."""
        target = evt.get("target_agent", "")
        if target:
            return str(target)
        data = evt.get("data", {})
        if isinstance(data, dict):
            target = (
                data.get("target_agent", "")
                or data.get("agent", "")
                or data.get("target", "")
            )
        return str(target) if target else ""

    def _get_child_session_ids(self, evt: dict) -> List[str]:
        """Extract child session IDs from event data."""
        ids: List[str] = []
        data = evt.get("data", {})
        if not isinstance(data, dict):
            return ids

        # child_session_ids: list
        child_ids = data.get("child_session_ids", [])
        if isinstance(child_ids, list):
            ids.extend(str(cid) for cid in child_ids if cid)

        # child_session_id: single value
        single = data.get("child_session_id", "")
        if single:
            ids.append(str(single))

        return ids

    def _count_total_children(self, spawn_events: List[tuple]) -> int:
        """Count total number of child sessions from data fields."""
        total = 0
        for _, evt in spawn_events:
            data = evt.get("data", {})
            if not isinstance(data, dict):
                total += 1
                continue

            # Check count field
            count = data.get("count", 0)
            if isinstance(count, (int, float)) and count > 1:
                total += int(count)
                continue

            # Check child_session_ids list length
            child_ids = data.get("child_session_ids", [])
            if isinstance(child_ids, list) and len(child_ids) > 0:
                total += len(child_ids)
                continue

            # Check child_sessions list length
            child_sessions = data.get("child_sessions", [])
            if isinstance(child_sessions, list) and len(child_sessions) > 0:
                total += len(child_sessions)
                continue

            # Default: 1 child per spawn event
            total += 1

        return total

    def _get_max_depth(self, spawn_events: List[tuple]) -> int:
        """Get the maximum depth value from spawn event data."""
        max_depth = 0
        for _, evt in spawn_events:
            data = evt.get("data", {})
            if isinstance(data, dict):
                depth = data.get("depth", 0)
                if isinstance(depth, (int, float)):
                    max_depth = max(max_depth, int(depth))
        return max_depth

    def _no_detection(self, explanation: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=explanation,
            detector_name=self.name,
        )
