"""
LangGraph Checkpoint Corruption Detector
==========================================

Validates checkpoint integrity in LangGraph executions:
- Monotonic supersteps: each checkpoint's superstep >= previous
- No gaps: superstep N followed by N+1 (not N+3)
- State consistency: checkpoint state matches state_snapshot for same superstep
- Valid state: checkpoint state contains all keys from state_schema
"""

import logging
from typing import Any, Dict, List, Optional, Set

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)


class LangGraphCheckpointCorruptionDetector(TurnAwareDetector):
    """Detects checkpoint integrity violations in LangGraph executions.

    Validates checkpoints for:
    1. Monotonic ordering (supersteps never decrease)
    2. No gaps in superstep sequence
    3. State consistency with state_snapshots
    4. Schema completeness (all state_schema keys present)
    """

    name = "LangGraphCheckpointCorruptionDetector"
    version = "1.0"
    supported_failure_modes = ["F3"]

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Delegate to graph execution analysis."""
        graph_execution = (conversation_metadata or {}).get("graph_execution", {})
        if graph_execution:
            return self.detect_graph_execution(graph_execution)
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation="No graph_execution data provided",
            detector_name=self.name,
        )

    def detect_graph_execution(self, graph_execution: Dict[str, Any]) -> TurnAwareDetectionResult:
        """Validate checkpoint integrity."""
        checkpoints = graph_execution.get("checkpoints", [])

        if not checkpoints:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No checkpoints in graph execution",
                detector_name=self.name,
            )

        # Sort checkpoints by created_at or by superstep
        checkpoints = sorted(
            checkpoints,
            key=lambda c: (c.get("superstep", 0), c.get("created_at", "")),
        )

        state_snapshots = graph_execution.get("state_snapshots", [])
        state_schema = graph_execution.get("state_schema", {})
        schema_keys: Set[str] = set(state_schema.get("keys", []))

        # Build superstep -> state_snapshot lookup
        snapshot_by_step: Dict[int, Dict[str, Any]] = {}
        for snap in state_snapshots:
            step = snap.get("superstep", -1)
            snapshot_by_step[step] = snap.get("state", {}) or {}

        issues: List[Dict[str, Any]] = []
        affected_supersteps: List[int] = []

        # 1. Check monotonic ordering
        for i in range(1, len(checkpoints)):
            prev_step = checkpoints[i - 1].get("superstep", 0)
            curr_step = checkpoints[i].get("superstep", 0)
            if curr_step < prev_step:
                issues.append(
                    {
                        "type": "non_monotonic",
                        "checkpoint_index": i,
                        "previous_superstep": prev_step,
                        "current_superstep": curr_step,
                        "checkpoint_id": checkpoints[i].get("checkpoint_id", ""),
                        "description": (
                            f"Checkpoint {i} has superstep {curr_step} "
                            f"which is less than previous superstep {prev_step}"
                        ),
                    }
                )
                affected_supersteps.append(curr_step)

        # 2. Check for gaps in superstep sequence
        supersteps = [c.get("superstep", 0) for c in checkpoints]
        for i in range(1, len(supersteps)):
            expected = supersteps[i - 1] + 1
            actual = supersteps[i]
            # Allow same superstep (multiple checkpoints per step)
            if actual > expected:
                gap_size = actual - expected
                issues.append(
                    {
                        "type": "superstep_gap",
                        "expected_superstep": expected,
                        "actual_superstep": actual,
                        "gap_size": gap_size,
                        "checkpoint_index": i,
                        "checkpoint_id": checkpoints[i].get("checkpoint_id", ""),
                        "description": (
                            f"Gap in checkpoint sequence: expected superstep "
                            f"{expected} but found {actual} (gap of {gap_size})"
                        ),
                    }
                )
                affected_supersteps.append(actual)

        # 3. State consistency with state_snapshots
        for checkpoint in checkpoints:
            cp_step = checkpoint.get("superstep", -1)
            cp_state = checkpoint.get("state", {}) or {}
            cp_id = checkpoint.get("checkpoint_id", "")

            if cp_step in snapshot_by_step:
                snap_state = snapshot_by_step[cp_step]
                inconsistencies = self._compare_states(cp_state, snap_state)
                if inconsistencies:
                    issues.append(
                        {
                            "type": "state_inconsistency",
                            "superstep": cp_step,
                            "checkpoint_id": cp_id,
                            "inconsistencies": inconsistencies,
                            "description": (
                                f"Checkpoint at superstep {cp_step} state differs "
                                f"from state_snapshot: {len(inconsistencies)} mismatches"
                            ),
                        }
                    )
                    affected_supersteps.append(cp_step)

        # 4. Schema completeness
        if schema_keys:
            for checkpoint in checkpoints:
                cp_state = checkpoint.get("state", {}) or {}
                cp_step = checkpoint.get("superstep", -1)
                cp_id = checkpoint.get("checkpoint_id", "")
                cp_keys = set(cp_state.keys())
                missing = schema_keys - cp_keys
                if missing:
                    issues.append(
                        {
                            "type": "missing_schema_keys",
                            "superstep": cp_step,
                            "checkpoint_id": cp_id,
                            "missing_keys": sorted(missing),
                            "description": (
                                f"Checkpoint at superstep {cp_step} missing "
                                f"schema keys: {', '.join(sorted(missing))}"
                            ),
                        }
                    )
                    affected_supersteps.append(cp_step)

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation=(f"All {len(checkpoints)} checkpoints pass integrity checks"),
                detector_name=self.name,
            )

        # Determine confidence and severity
        issue_types = {i["type"] for i in issues}

        n = len(issues)
        if "state_inconsistency" in issue_types:
            confidence = min(0.95, 0.80 + 0.04 * n)
        elif "non_monotonic" in issue_types:
            confidence = min(0.92, 0.75 + 0.04 * n)
        elif "superstep_gap" in issue_types:
            confidence = min(0.88, 0.70 + 0.04 * n)
        else:
            confidence = min(0.82, 0.60 + 0.04 * n)

        if "state_inconsistency" in issue_types or "non_monotonic" in issue_types:
            severity = TurnAwareSeverity.SEVERE
        elif "superstep_gap" in issue_types:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        type_counts: Dict[str, int] = {}
        for issue in issues:
            type_counts[issue["type"]] = type_counts.get(issue["type"], 0) + 1
        summary = [f"{c} {t}" for t, c in type_counts.items()]

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F3",
            explanation=(
                f"Checkpoint integrity issues: {', '.join(summary)} "
                f"across {len(checkpoints)} checkpoints"
            ),
            affected_turns=sorted(set(affected_supersteps)),
            evidence={
                "issues": issues,
                "total_checkpoints": len(checkpoints),
                "issue_count": len(issues),
            },
            suggested_fix=(
                "Ensure checkpoints are created at every superstep without gaps. "
                "Validate checkpoint state against the state schema before persisting. "
                "Use a MemorySaver or SqliteSaver with integrity checks enabled."
            ),
            detector_name=self.name,
        )

    def _compare_states(
        self,
        checkpoint_state: Dict[str, Any],
        snapshot_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Compare checkpoint state with snapshot state, returning mismatches."""
        mismatches: List[Dict[str, Any]] = []

        all_keys = set(checkpoint_state.keys()) | set(snapshot_state.keys())

        for key in all_keys:
            in_cp = key in checkpoint_state
            in_snap = key in snapshot_state

            if in_cp and not in_snap:
                mismatches.append(
                    {
                        "key": key,
                        "issue": "extra_in_checkpoint",
                        "description": f"Key '{key}' in checkpoint but not in snapshot",
                    }
                )
            elif in_snap and not in_cp:
                mismatches.append(
                    {
                        "key": key,
                        "issue": "missing_in_checkpoint",
                        "description": f"Key '{key}' in snapshot but not in checkpoint",
                    }
                )
            elif checkpoint_state[key] != snapshot_state[key]:
                mismatches.append(
                    {
                        "key": key,
                        "issue": "value_mismatch",
                        "description": f"Key '{key}' has different values in checkpoint vs snapshot",
                    }
                )

        return mismatches
