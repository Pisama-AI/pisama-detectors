"""
LangGraph Tool Failure Detector
=================================

Detects tool node failures and recovery patterns in LangGraph:
- Tool nodes with status=="failed" and error content
- Retry patterns: same tool node appearing in the next superstep
- Fallback patterns: different node handling the failure
- Uncaught failures: tool fails with no recovery, causing graph failure
"""

import logging
from typing import Any, Dict, List, Optional

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)


class LangGraphToolFailureDetector(TurnAwareDetector):
    """Detects tool execution failures in LangGraph graph executions.

    Analyzes tool nodes for:
    1. Failed tool nodes with error details
    2. Retry patterns (same tool in subsequent superstep)
    3. Fallback patterns (different node handles the failure)
    4. Uncaught failures (tool fails, no recovery, graph fails)
    """

    name = "LangGraphToolFailureDetector"
    version = "1.0"
    supported_failure_modes = ["F14"]

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
        """Analyze tool nodes for failure patterns."""
        nodes = graph_execution.get("nodes", [])
        graph_status = graph_execution.get("status", "")

        # Find tool nodes
        tool_nodes = [n for n in nodes if n.get("node_type") == "tool"]
        if not tool_nodes:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No tool nodes in graph execution",
                detector_name=self.name,
            )

        # Find failed tool nodes
        failed_tools = [t for t in tool_nodes if t.get("status") == "failed"]
        if not failed_tools:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="All tool nodes succeeded",
                detector_name=self.name,
            )

        # Build superstep -> nodes mapping for recovery analysis
        superstep_nodes: Dict[int, List[Dict[str, Any]]] = {}
        for node in nodes:
            step = node.get("superstep", -1)
            if step not in superstep_nodes:
                superstep_nodes[step] = []
            superstep_nodes[step].append(node)

        issues: List[Dict[str, Any]] = []
        affected_indices: List[int] = []

        for failed_tool in failed_tools:
            tool_id = failed_tool.get("node_id", "")
            tool_title = failed_tool.get("title", tool_id)
            tool_superstep = failed_tool.get("superstep", -1)
            error = failed_tool.get("error")
            node_idx = nodes.index(failed_tool) if failed_tool in nodes else -1

            if node_idx >= 0:
                affected_indices.append(node_idx)

            # Check for retry pattern: same node_id in next superstep
            next_step_nodes = superstep_nodes.get(tool_superstep + 1, [])
            is_retried = any(n.get("node_id") == tool_id for n in next_step_nodes)

            # Check for fallback pattern: different node handles the failure
            has_fallback = bool(next_step_nodes) and not is_retried

            # Check for uncaught failure: no recovery and graph failed
            is_uncaught = (
                not is_retried and not has_fallback and graph_status in ("failed", "error")
            )

            issue: Dict[str, Any] = {
                "tool_id": tool_id,
                "tool_title": tool_title,
                "superstep": tool_superstep,
                "has_error": error is not None and error != "",
                "error_preview": (str(error)[:200] if error else None),
                "is_retried": is_retried,
                "has_fallback": has_fallback,
                "is_uncaught": is_uncaught,
            }

            if is_uncaught:
                issue["type"] = "uncaught_failure"
                issue["description"] = (
                    f"Tool '{tool_title}' failed at superstep {tool_superstep} "
                    f"with no recovery, causing graph failure"
                )
            elif is_retried:
                # Check if retry succeeded
                retry_node = next(
                    (n for n in next_step_nodes if n.get("node_id") == tool_id),
                    None,
                )
                retry_succeeded = retry_node is not None and retry_node.get("status") == "succeeded"
                issue["type"] = "retried_failure"
                issue["retry_succeeded"] = retry_succeeded
                issue["description"] = (
                    f"Tool '{tool_title}' failed at superstep {tool_superstep}, "
                    f"retried at superstep {tool_superstep + 1} "
                    f"({'succeeded' if retry_succeeded else 'also failed'})"
                )
            elif has_fallback:
                fallback_nodes = [n.get("title", n.get("node_id", "")) for n in next_step_nodes]
                issue["type"] = "fallback_handled"
                issue["fallback_nodes"] = fallback_nodes
                issue["description"] = (
                    f"Tool '{tool_title}' failed at superstep {tool_superstep}, "
                    f"handled by fallback: {', '.join(fallback_nodes)}"
                )
            else:
                issue["type"] = "tool_failure"
                issue["description"] = f"Tool '{tool_title}' failed at superstep {tool_superstep}"

            issues.append(issue)

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="No tool failure issues detected",
                detector_name=self.name,
            )

        # Determine confidence and severity
        uncaught_count = sum(1 for i in issues if i["type"] == "uncaught_failure")
        retried_failed = sum(
            1
            for i in issues
            if i["type"] == "retried_failure" and not i.get("retry_succeeded", False)
        )

        # Confidence scales with count + nature of failures so repeated real
        # traces produce distinguishable scores instead of collapsing to 3
        # fixed buckets.
        fail_ratio = len(failed_tools) / max(len(tool_nodes), 1)
        if uncaught_count > 0:
            confidence = min(0.97, 0.80 + 0.05 * uncaught_count + 0.10 * fail_ratio)
            severity = TurnAwareSeverity.SEVERE
        elif retried_failed > 0:
            confidence = min(0.88, 0.65 + 0.06 * retried_failed + 0.10 * fail_ratio)
            severity = TurnAwareSeverity.MODERATE
        else:
            confidence = min(0.75, 0.50 + 0.05 * len(issues) + 0.10 * fail_ratio)
            severity = TurnAwareSeverity.MINOR

        type_counts: Dict[str, int] = {}
        for issue in issues:
            type_counts[issue["type"]] = type_counts.get(issue["type"], 0) + 1
        summary = [f"{c} {t}" for t, c in type_counts.items()]

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F14",
            explanation=(
                f"Tool failures detected: {', '.join(summary)} out of {len(tool_nodes)} tool nodes"
            ),
            affected_turns=sorted(set(affected_indices)),
            evidence={
                "issues": issues,
                "total_tool_nodes": len(tool_nodes),
                "failed_tool_count": len(failed_tools),
                "graph_status": graph_status,
            },
            suggested_fix=(
                "Add error handling for tool nodes: implement retry logic with "
                "backoff, add fallback tool alternatives, or use try/except "
                "patterns in tool node implementations. For uncaught failures, "
                "add a global error handler node."
            ),
            detector_name=self.name,
        )
