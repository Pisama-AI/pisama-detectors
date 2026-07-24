"""
Iteration Escape Detection for Dify Workflows
==============================================

Detects runaway iteration and loop nodes in Dify workflows.
Checks for excessive iteration counts, missing break conditions,
child nodes modifying parent scope, and failed/stopped iterations.

Dify-specific: targets iteration and loop node types with
parent_node_id / iteration_index child structure.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

# Thresholds
MAX_SAFE_ITERATIONS = 100
WARN_ITERATIONS = 50


class DifyIterationEscapeDetector(TurnAwareDetector):
    """Detects runaway or escaped iteration/loop nodes in Dify workflows.

    Checks for excessive iteration counts, missing exit conditions,
    parent-scope variable modification by child nodes, and
    failed/stopped status after many iterations.
    """

    name = "DifyIterationEscapeDetector"
    version = "1.0"
    supported_failure_modes = ["F11"]  # Coordination / loop failure

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Delegate to detect_workflow_run if metadata contains workflow_run."""
        workflow_run = (conversation_metadata or {}).get("workflow_run", {})
        if workflow_run:
            return self.detect_workflow_run(workflow_run)
        return self._no_detection("No workflow_run data provided")

    def detect_workflow_run(self, workflow_run: dict) -> TurnAwareDetectionResult:
        """Analyze Dify workflow run for iteration escape issues.

        Args:
            workflow_run: Dify workflow_run dict with nodes list.

        Returns:
            Detection result with iteration findings.
        """
        nodes = workflow_run.get("nodes", [])
        if not nodes:
            return self._no_detection("No nodes in workflow run")

        # Find iteration/loop parent nodes
        iter_nodes = [n for n in nodes if n.get("node_type") in ("iteration", "loop")]

        if not iter_nodes:
            return self._no_detection("No iteration/loop nodes found")

        issues: List[Dict[str, Any]] = []
        affected_node_ids: List[str] = []

        for iter_node in iter_nodes:
            iter_id = iter_node.get("node_id", "")
            iter_title = iter_node.get("title", "")
            iter_status = iter_node.get("status", "")

            # Count child nodes belonging to this iteration
            children = [n for n in nodes if n.get("parent_node_id") == iter_id]
            child_count = len(children)

            # Count distinct iteration indices
            indices = [
                n.get("iteration_index") for n in children if n.get("iteration_index") is not None
            ]
            max_index = max(indices) if indices else 0
            iteration_count = max_index + 1 if indices else child_count

            # Check 1: Excessive iteration count
            if iteration_count > MAX_SAFE_ITERATIONS:
                affected_node_ids.append(iter_id)
                issues.append(
                    {
                        "type": "excessive_iterations",
                        "node_id": iter_id,
                        "title": iter_title,
                        "iteration_count": iteration_count,
                        "threshold": MAX_SAFE_ITERATIONS,
                    }
                )

            # Check 2: Failed/stopped iteration (any count > 1)
            if iter_status in ("failed", "stopped") and iteration_count > 1:
                affected_node_ids.append(iter_id)
                issues.append(
                    {
                        "type": "iteration_failure",
                        "node_id": iter_id,
                        "title": iter_title,
                        "status": iter_status,
                        "iteration_count": iteration_count,
                    }
                )

            # Check 3: No visible break/exit condition in outputs
            iter_outputs = iter_node.get("outputs", {})
            outputs_str = str(iter_outputs).lower()
            has_break = any(
                kw in outputs_str
                for kw in ("break", "exit", "stop", "terminate", "max_iterations", "limit")
            )
            if not has_break and iteration_count > WARN_ITERATIONS:
                affected_node_ids.append(iter_id)
                issues.append(
                    {
                        "type": "no_break_condition",
                        "node_id": iter_id,
                        "title": iter_title,
                        "iteration_count": iteration_count,
                    }
                )

            # Check 4: Child nodes referencing parent-level variables
            parent_ref_children = self._find_parent_references(children)
            if parent_ref_children:
                affected_node_ids.append(iter_id)
                issues.append(
                    {
                        "type": "parent_scope_modification",
                        "node_id": iter_id,
                        "title": iter_title,
                        "children_with_parent_refs": parent_ref_children,
                    }
                )

            # Check 5: Non-contiguous iteration indices (index gaps)
            if indices:
                sorted_indices = sorted(indices)
                expected_indices = list(range(sorted_indices[0], sorted_indices[-1] + 1))
                if sorted_indices != expected_indices:
                    missing = sorted(set(expected_indices) - set(sorted_indices))
                    affected_node_ids.append(iter_id)
                    issues.append(
                        {
                            "type": "index_corruption",
                            "node_id": iter_id,
                            "title": iter_title,
                            "actual_indices": sorted_indices,
                            "missing_indices": missing,
                            "iteration_count": iteration_count,
                        }
                    )

            # Check 6: Duplicate outputs across iterations (scope leak)
            if len(children) > 1:
                child_output_strs = [
                    json.dumps(c.get("outputs", {}), sort_keys=True, default=str) for c in children
                ]
                if len(set(child_output_strs)) == 1:
                    affected_node_ids.append(iter_id)
                    issues.append(
                        {
                            "type": "scope_leak_duplicate_outputs",
                            "node_id": iter_id,
                            "title": iter_title,
                            "iteration_count": iteration_count,
                            "child_count": len(children),
                        }
                    )

            # Check 7: Iteration overrun (more iterations than input items)
            iter_inputs = iter_node.get("inputs", {})
            input_items = iter_inputs.get("items", [])
            max_configured = iter_inputs.get("max_iterations") or iter_inputs.get("max_attempts")
            if isinstance(input_items, list) and len(input_items) > 0:
                if iteration_count > len(input_items):
                    affected_node_ids.append(iter_id)
                    issues.append(
                        {
                            "type": "iteration_overrun",
                            "node_id": iter_id,
                            "title": iter_title,
                            "input_count": len(input_items),
                            "iteration_count": iteration_count,
                        }
                    )
            elif max_configured is not None and iteration_count > max_configured:
                affected_node_ids.append(iter_id)
                issues.append(
                    {
                        "type": "loop_overrun",
                        "node_id": iter_id,
                        "title": iter_title,
                        "max_configured": max_configured,
                        "iteration_count": iteration_count,
                    }
                )

        if not issues:
            return self._no_detection("No iteration escape issues found")

        # Compute confidence based on issue types and iteration counts
        max_iter = max(
            (i.get("iteration_count", 0) for i in issues),
            default=0,
        )
        has_structural = any(
            i["type"]
            in (
                "index_corruption",
                "scope_leak_duplicate_outputs",
                "iteration_overrun",
                "loop_overrun",
            )
            for i in issues
        )
        if max_iter > MAX_SAFE_ITERATIONS:
            confidence = min(0.95, 0.7 + (max_iter - MAX_SAFE_ITERATIONS) / 500)
        elif has_structural:
            confidence = min(0.90, 0.6 + len(issues) * 0.1)
        elif max_iter > WARN_ITERATIONS:
            confidence = (
                0.6 + (max_iter - WARN_ITERATIONS) / (MAX_SAFE_ITERATIONS - WARN_ITERATIONS) * 0.15
            )
        else:
            confidence = 0.5

        # Severity
        has_excessive = any(i["type"] == "excessive_iterations" for i in issues)
        has_failure = any(i["type"] == "iteration_failure" for i in issues)
        has_overrun = any(i["type"] in ("iteration_overrun", "loop_overrun") for i in issues)
        if has_excessive and has_failure:
            severity = TurnAwareSeverity.SEVERE
        elif has_excessive or has_failure or has_overrun:
            severity = TurnAwareSeverity.MODERATE
        elif has_structural:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F11",
            explanation=(
                f"Iteration escape: {len(issues)} issue(s) across "
                f"{len(iter_nodes)} iteration/loop node(s)"
            ),
            affected_turns=list(range(len(set(affected_node_ids)))),
            evidence={
                "issues": issues,
                "total_iteration_nodes": len(iter_nodes),
                "max_iteration_count": max_iter,
            },
            suggested_fix=(
                "Add explicit break/exit conditions to iteration nodes. "
                "Set max_iterations limits in loop configuration. "
                "Avoid modifying parent-scope variables from within iteration children."
            ),
            detector_name=self.name,
        )

    def _find_parent_references(self, children: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Find child nodes whose outputs reference parent-level variables."""
        refs = []
        parent_keywords = ("parent", "global", "workflow_var", "sys.")
        for child in children:
            outputs_str = str(child.get("outputs", {})).lower()
            inputs_str = str(child.get("inputs", {})).lower()
            combined = outputs_str + inputs_str
            for kw in parent_keywords:
                if kw in combined:
                    refs.append(
                        {
                            "child_node_id": child.get("node_id", ""),
                            "child_title": child.get("title", ""),
                            "keyword": kw,
                        }
                    )
                    break
        return refs

    def _no_detection(self, reason: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=reason,
            detector_name=self.name,
        )
