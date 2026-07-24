"""
LangGraph Recursion Detector
=============================

Detects recursion-related failures in LangGraph graph executions:
- Graph hit the recursion_limit (status == "recursion_limit")
- Superstep count approaching the recursion limit (>90%)
- Unbounded node repetition across supersteps indicating cycles
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


class LangGraphRecursionDetector(TurnAwareDetector):
    """Detects recursion and unbounded cycle failures in LangGraph executions.

    Analyzes graph_execution data for:
    1. Status == "recursion_limit" (definitive hit)
    2. total_supersteps / recursion_limit ratio > 0.9 (approaching limit)
    3. Node repetition patterns across supersteps (unbounded cycles)
    """

    name = "LangGraphRecursionDetector"
    version = "1.0"
    supported_failure_modes = ["F11"]

    def __init__(
        self,
        limit_ratio_threshold: float = 0.9,
        node_repetition_threshold: int = 3,
    ):
        self.limit_ratio_threshold = limit_ratio_threshold
        self.node_repetition_threshold = node_repetition_threshold

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Delegate to graph execution analysis when metadata contains graph_execution."""
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
        """Analyze a LangGraph execution for recursion failures."""
        status = graph_execution.get("status", "")
        total_supersteps = graph_execution.get("total_supersteps", 0)
        recursion_limit = graph_execution.get("recursion_limit", 256)
        nodes = graph_execution.get("nodes", [])

        issues: List[Dict[str, Any]] = []
        affected_turns: List[int] = []

        # 1. Definitive recursion limit hit
        if status == "recursion_limit":
            issues.append(
                {
                    "type": "recursion_limit_hit",
                    "total_supersteps": total_supersteps,
                    "recursion_limit": recursion_limit,
                    "description": (
                        f"Graph execution hit recursion limit "
                        f"({total_supersteps}/{recursion_limit} supersteps)"
                    ),
                }
            )

        # 2. Approaching recursion limit
        if recursion_limit > 0 and status != "recursion_limit":
            ratio = total_supersteps / recursion_limit
            if ratio > self.limit_ratio_threshold:
                issues.append(
                    {
                        "type": "approaching_limit",
                        "ratio": round(ratio, 3),
                        "total_supersteps": total_supersteps,
                        "recursion_limit": recursion_limit,
                        "description": (
                            f"Graph used {ratio:.1%} of recursion limit "
                            f"({total_supersteps}/{recursion_limit})"
                        ),
                    }
                )

        # 3. Node repetition patterns across supersteps
        node_repetition = self._detect_node_repetition(nodes)
        if node_repetition:
            issues.append(node_repetition)
            affected_turns.extend(node_repetition.get("affected_indices", []))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="No recursion issues detected in graph execution",
                detector_name=self.name,
            )

        # Confidence scales with the maximum limit-approach ratio so traces
        # at 70% and 95% of limit are distinguishable.
        max_ratio = max((i.get("ratio", 0.0) for i in issues), default=0.0)
        if status == "recursion_limit":
            confidence = min(0.98, 0.90 + 0.03 * len(issues))
        elif any(i["type"] == "approaching_limit" for i in issues):
            confidence = min(0.88, 0.55 + 0.35 * max_ratio)
        else:
            confidence = min(0.70, 0.45 + 0.05 * len(issues))

        # Determine severity
        if status == "recursion_limit":
            severity = TurnAwareSeverity.SEVERE
        elif any(i.get("ratio", 0) > 0.95 for i in issues):
            severity = TurnAwareSeverity.SEVERE
        elif any(i["type"] == "approaching_limit" for i in issues):
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        descriptions = [i["description"] for i in issues]

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F11",
            explanation="; ".join(descriptions),
            affected_turns=sorted(set(affected_turns)),
            evidence={
                "issues": issues,
                "total_supersteps": total_supersteps,
                "recursion_limit": recursion_limit,
                "status": status,
            },
            suggested_fix=(
                "Add explicit termination conditions to prevent unbounded recursion. "
                "Consider reducing the recursion_limit or adding cycle-breaking logic "
                "in conditional edges."
            ),
            detector_name=self.name,
        )

    def _detect_node_repetition(self, nodes: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Detect nodes that repeat across many supersteps (unbounded cycle signal)."""
        if not nodes:
            return None

        # Group nodes by node_id and count distinct supersteps each appears in
        node_supersteps: Dict[str, List[int]] = {}
        for idx, node in enumerate(nodes):
            node_id = node.get("node_id", "")
            superstep = node.get("superstep", -1)
            if node_id not in node_supersteps:
                node_supersteps[node_id] = []
            node_supersteps[node_id].append(superstep)

        unique_supersteps = len({n.get("superstep", -1) for n in nodes})
        if unique_supersteps < 3:
            return None

        # Find nodes appearing in many distinct supersteps
        repeated_nodes: List[Dict[str, Any]] = []
        for node_id, supersteps in node_supersteps.items():
            distinct = len(set(supersteps))
            if distinct >= self.node_repetition_threshold:
                repeated_nodes.append(
                    {
                        "node_id": node_id,
                        "superstep_count": distinct,
                        "supersteps": sorted(set(supersteps)),
                    }
                )

        if not repeated_nodes:
            return None

        # Find affected node indices
        repeated_ids = {rn["node_id"] for rn in repeated_nodes}
        affected_indices = [
            idx for idx, n in enumerate(nodes) if n.get("node_id", "") in repeated_ids
        ]

        worst = max(repeated_nodes, key=lambda r: r["superstep_count"])

        return {
            "type": "node_repetition",
            "repeated_nodes": repeated_nodes,
            "affected_indices": affected_indices,
            "description": (
                f"Node '{worst['node_id']}' appears in {worst['superstep_count']} "
                f"supersteps out of {unique_supersteps} total, suggesting an unbounded cycle"
            ),
        }
