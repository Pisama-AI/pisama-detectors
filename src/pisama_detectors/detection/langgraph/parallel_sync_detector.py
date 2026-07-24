"""
LangGraph Parallel Sync Detector
==================================

Detects synchronization issues in parallel node execution:
- State write conflicts: multiple nodes in the same superstep writing
  to the same state key WITHOUT a proper join node downstream
- Missing join/aggregation: no single node in the next superstep to
  merge parallel results
- Race conditions: parallel nodes reading and writing the same key
- Failed parallel nodes: one branch fails while siblings succeed
- Downstream failure after parallel: join/merge node fails or skipped
- State errors after parallel execution: error/failure indicators in state
"""

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)


class LangGraphParallelSyncDetector(TurnAwareDetector):
    """Detects parallel execution synchronization issues in LangGraph.

    Analyzes nodes executing in the same superstep for:
    1. Write conflicts (multiple nodes writing to the same state key
       without downstream join)
    2. Missing join node after parallel branches
    3. Race conditions (parallel read-then-write on same key)
    4. Failed parallel nodes (mixed success/failure in same superstep)
    5. Downstream failures after parallel execution
    6. State error indicators after parallel steps
    """

    name = "LangGraphParallelSyncDetector"
    version = "1.1"
    supported_failure_modes = ["F11"]

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
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
        nodes = graph_execution.get("nodes", [])

        if not nodes:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No nodes in graph execution",
                detector_name=self.name,
            )

        # Group nodes by superstep
        superstep_nodes: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            step = node.get("superstep", -1)
            superstep_nodes[step].append(node)

        # Find supersteps with parallel execution (>1 node)
        parallel_steps = {
            step: step_nodes for step, step_nodes in superstep_nodes.items() if len(step_nodes) > 1
        }

        issues: List[Dict[str, Any]] = []
        affected_supersteps: List[int] = []

        if parallel_steps:
            for step, step_nodes in sorted(parallel_steps.items()):
                # 1. Check for write conflicts (only flag if no proper join)
                write_conflicts = self._detect_write_conflicts(step, step_nodes, superstep_nodes)
                if write_conflicts:
                    issues.append(write_conflicts)
                    affected_supersteps.append(step)

                # 2. Check for race conditions
                race_conditions = self._detect_race_conditions(step, step_nodes)
                if race_conditions:
                    issues.append(race_conditions)
                    affected_supersteps.append(step)

                # 3. Check for missing join node
                missing_join = self._detect_missing_join(step, step_nodes, superstep_nodes)
                if missing_join:
                    issues.append(missing_join)
                    affected_supersteps.append(step)

                # 4. Check for failed parallel nodes
                failed_parallel = self._detect_failed_parallel(step, step_nodes)
                if failed_parallel:
                    issues.append(failed_parallel)
                    affected_supersteps.append(step)

                # 5. Check for downstream failure after parallel
                downstream = self._detect_downstream_failure(step, step_nodes, superstep_nodes)
                if downstream:
                    issues.append(downstream)
                    affected_supersteps.append(step)

        # 6. Check state snapshots for error indicators after parallel steps
        state_snapshots = graph_execution.get("state_snapshots", [])
        if state_snapshots and parallel_steps:
            state_issues = self._detect_state_errors(state_snapshots, parallel_steps)
            issues.extend(state_issues)

        if not issues:
            msg = (
                f"Parallel execution in {len(parallel_steps)} supersteps, no sync issues detected"
                if parallel_steps
                else "No parallel execution detected"
            )
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation=msg,
                detector_name=self.name,
            )

        # Determine confidence and severity
        issue_types = {i["type"] for i in issues}
        has_write_conflict = "write_conflict" in issue_types
        has_race = "race_condition" in issue_types
        has_failed = "failed_parallel" in issue_types
        has_downstream = "downstream_failure" in issue_types
        has_state_error = "state_error_after_parallel" in issue_types

        if has_failed or has_downstream:
            confidence = min(0.90, 0.7 + len(issues) * 0.05)
        elif has_write_conflict or has_race:
            confidence = min(0.85, 0.60 + len(issues) * 0.05)
        elif has_state_error:
            confidence = min(0.80, 0.55 + len(issues) * 0.05)
        else:
            confidence = min(0.75, 0.50 + len(issues) * 0.05)

        if has_failed and has_downstream:
            severity = TurnAwareSeverity.SEVERE
        elif has_write_conflict or has_failed:
            severity = TurnAwareSeverity.SEVERE
        elif has_race or has_downstream or has_state_error:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        type_counts: Dict[str, int] = {}
        for i in issues:
            type_counts[i["type"]] = type_counts.get(i["type"], 0) + 1
        summary = [f"{c} {t}" for t, c in type_counts.items()]

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F11",
            explanation=(
                f"Parallel sync issues: {', '.join(summary)} "
                f"across {len(parallel_steps)} parallel supersteps"
            ),
            affected_turns=sorted(set(affected_supersteps)),
            evidence={
                "issues": issues,
                "parallel_supersteps": len(parallel_steps),
                "total_supersteps": len(superstep_nodes),
            },
            suggested_fix=(
                "Use state reducers (e.g., operator.add for lists) to handle "
                "concurrent writes. Add an explicit join/aggregation node after "
                "parallel branches. Avoid having parallel nodes write to the "
                "same state key without a reducer."
            ),
            detector_name=self.name,
        )

    def _get_output_keys(self, node: Dict[str, Any]) -> Set[str]:
        outputs = node.get("outputs", {})
        if isinstance(outputs, dict):
            return set(outputs.keys())
        return set()

    def _get_input_keys(self, node: Dict[str, Any]) -> Set[str]:
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict):
            return set(inputs.keys())
        return set()

    def _detect_write_conflicts(
        self,
        superstep: int,
        step_nodes: List[Dict[str, Any]],
        all_superstep_nodes: Dict[int, List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Detect multiple parallel nodes writing to the same state key.

        Only flags as a conflict if there's NO proper join node in the next
        superstep (map-reduce pattern with join is expected).
        """
        key_writers: Dict[str, List[str]] = {}

        for node in step_nodes:
            node_id = node.get("node_id", "")
            for key in self._get_output_keys(node):
                key_writers.setdefault(key, []).append(node_id)

        conflicts = {key: writers for key, writers in key_writers.items() if len(writers) > 1}

        if not conflicts:
            return None

        # Check if there's a proper join node in the next superstep
        next_step = all_superstep_nodes.get(superstep + 1, [])
        has_join = len(next_step) == 1 and next_step[0].get("status") in ("succeeded", "completed")

        if has_join:
            # Map-reduce pattern with successful join — not a real conflict
            return None

        return {
            "type": "write_conflict",
            "superstep": superstep,
            "conflicts": dict(conflicts),
            "description": (
                f"Superstep {superstep}: {len(conflicts)} state key(s) written by "
                f"multiple parallel nodes without proper join: "
                + ", ".join(f"'{k}' by [{', '.join(w)}]" for k, w in conflicts.items())
            ),
        }

    def _detect_race_conditions(
        self, superstep: int, step_nodes: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect parallel nodes that read and write the same key."""
        races: List[Dict[str, Any]] = []

        for i, node_a in enumerate(step_nodes):
            a_id = node_a.get("node_id", "")
            a_reads = self._get_input_keys(node_a)
            a_writes = self._get_output_keys(node_a)

            for node_b in step_nodes[i + 1 :]:
                b_id = node_b.get("node_id", "")
                b_reads = self._get_input_keys(node_b)
                b_writes = self._get_output_keys(node_b)

                conflicting_keys = (a_reads & b_writes) | (b_reads & a_writes)
                if conflicting_keys:
                    races.append(
                        {
                            "nodes": [a_id, b_id],
                            "keys": list(conflicting_keys),
                        }
                    )

        if not races:
            return None

        return {
            "type": "race_condition",
            "superstep": superstep,
            "races": races,
            "description": (
                f"Superstep {superstep}: {len(races)} potential race condition(s) "
                f"between parallel nodes reading/writing same keys"
            ),
        }

    def _detect_missing_join(
        self,
        superstep: int,
        step_nodes: List[Dict[str, Any]],
        all_superstep_nodes: Dict[int, List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Detect missing join/aggregation node after parallel branches."""
        next_step_nodes = all_superstep_nodes.get(superstep + 1, [])

        if not next_step_nodes:
            return {
                "type": "missing_join",
                "superstep": superstep,
                "parallel_count": len(step_nodes),
                "description": (
                    f"Superstep {superstep} has {len(step_nodes)} parallel nodes "
                    f"but no nodes in superstep {superstep + 1} to join results"
                ),
            }

        if len(next_step_nodes) > 1 and len(next_step_nodes) >= len(step_nodes):
            return {
                "type": "missing_join",
                "superstep": superstep,
                "parallel_count": len(step_nodes),
                "next_count": len(next_step_nodes),
                "description": (
                    f"Superstep {superstep} has {len(step_nodes)} parallel nodes "
                    f"followed by {len(next_step_nodes)} nodes in superstep "
                    f"{superstep + 1} (expected a single join node)"
                ),
            }

        return None

    def _detect_failed_parallel(
        self, superstep: int, step_nodes: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect mixed success/failure among parallel nodes in same superstep."""
        [n.get("status", "") for n in step_nodes]
        succeeded = [n for n in step_nodes if n.get("status") in ("succeeded", "completed")]
        failed = [n for n in step_nodes if n.get("status") == "failed"]

        if failed and succeeded:
            return {
                "type": "failed_parallel",
                "superstep": superstep,
                "failed_nodes": [n.get("node_id", "") for n in failed],
                "succeeded_nodes": [n.get("node_id", "") for n in succeeded],
                "description": (
                    f"Superstep {superstep}: {len(failed)} node(s) failed while "
                    f"{len(succeeded)} succeeded in parallel execution"
                ),
            }

        return None

    def _detect_downstream_failure(
        self,
        superstep: int,
        step_nodes: List[Dict[str, Any]],
        all_superstep_nodes: Dict[int, List[Dict[str, Any]]],
    ) -> Optional[Dict[str, Any]]:
        """Detect downstream node failure/skip after parallel execution."""
        next_step_nodes = all_superstep_nodes.get(superstep + 1, [])
        if not next_step_nodes:
            return None

        failed_downstream = [n for n in next_step_nodes if n.get("status") in ("failed", "skipped")]

        if failed_downstream:
            return {
                "type": "downstream_failure",
                "parallel_superstep": superstep,
                "parallel_count": len(step_nodes),
                "failed_nodes": [
                    {
                        "node_id": n.get("node_id", ""),
                        "status": n.get("status", ""),
                        "node_type": n.get("node_type", ""),
                    }
                    for n in failed_downstream
                ],
                "description": (
                    f"After parallel superstep {superstep}, {len(failed_downstream)} "
                    f"downstream node(s) failed/skipped"
                ),
            }

        return None

    def _detect_state_errors(
        self,
        state_snapshots: List[Dict[str, Any]],
        parallel_steps: Dict[int, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Check state snapshots for error indicators after parallel steps."""
        issues = []
        error_keywords = {"error", "fail", "timeout", "partial_failure", "sync_failed"}

        for ss in state_snapshots:
            step = ss.get("superstep", -1)
            state = ss.get("state", {})

            # Only check states AFTER parallel execution
            is_after_parallel = any(ps < step for ps in parallel_steps)
            if not is_after_parallel:
                continue

            state_str = str(state).lower()
            found_errors = [kw for kw in error_keywords if kw in state_str]

            if found_errors:
                issues.append(
                    {
                        "type": "state_error_after_parallel",
                        "superstep": step,
                        "error_indicators": found_errors,
                        "description": (
                            f"State at superstep {step} (after parallel execution) "
                            f"contains error indicators: {', '.join(found_errors)}"
                        ),
                    }
                )

        return issues
