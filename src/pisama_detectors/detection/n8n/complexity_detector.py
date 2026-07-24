"""
F15: Workflow Complexity Detection for n8n
==========================================

Detects overly complex n8n workflows that are hard to maintain:
- Node count > 50 (workflow too large)
- Branch depth > 10 (deeply nested conditions)
- Cyclomatic complexity > threshold (too many execution paths)
- Execution time consistently > 5 minutes
- Single workflow doing too many unrelated tasks

This is n8n-specific because it analyzes workflow graph structure complexity
rather than conversational interaction complexity.
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

# Complexity thresholds
DEFAULT_MAX_NODE_COUNT = 25
DEFAULT_MAX_BRANCH_DEPTH = 6
DEFAULT_MAX_CYCLOMATIC_COMPLEXITY = 10
DEFAULT_MAX_EXECUTION_TIME_MS = 300_000  # 5 minutes

# Node types that indicate branching (increase cyclomatic complexity)
BRANCHING_NODE_TYPES = {
    "n8n-nodes-base.if",
    "n8n-nodes-base.switch",
    "n8n-nodes-base.router",
    "n8n-nodes-base.merge",
    "n8n-nodes-base.split",
}


class N8NComplexityDetector(TurnAwareDetector):
    """Detects F15: Workflow Complexity issues in n8n workflows.

    Analyzes workflow structure for:
    1. Excessive node count (>50 nodes)
    2. Deep branching (>10 levels of conditions)
    3. High cyclomatic complexity (many execution paths)
    4. Long execution times (>5 minutes consistently)
    5. Multiple unrelated concerns in single workflow

    n8n-specific manifestation of F15 (Workflow Complexity):
    In conversational agents, this is about task decomposition.
    In n8n workflows, this is about graph structure and maintainability.
    """

    name = "N8NComplexityDetector"
    version = "1.0"
    supported_failure_modes = ["F15"]

    def __init__(
        self,
        max_node_count: int = DEFAULT_MAX_NODE_COUNT,
        max_branch_depth: int = DEFAULT_MAX_BRANCH_DEPTH,
        max_cyclomatic_complexity: int = DEFAULT_MAX_CYCLOMATIC_COMPLEXITY,
        max_execution_time_ms: int = DEFAULT_MAX_EXECUTION_TIME_MS,
    ):
        """Initialize complexity detector.

        Args:
            max_node_count: Maximum nodes before flagging as too complex
            max_branch_depth: Maximum nesting level of branches
            max_cyclomatic_complexity: Maximum cyclomatic complexity score
            max_execution_time_ms: Maximum execution time threshold
        """
        self.max_node_count = max_node_count
        self.max_branch_depth = max_branch_depth
        self.max_cyclomatic_complexity = max_cyclomatic_complexity
        self.max_execution_time_ms = max_execution_time_ms

    def detect_workflow(self, workflow_json: Dict[str, Any]) -> TurnAwareDetectionResult:
        """Detect complexity issues by analyzing raw n8n workflow JSON directly.

        Calculates:
        1. Node count from workflow_json["nodes"]
        2. Branching depth from the connection graph
        3. Cyclomatic complexity: E - N + 2P (edges, nodes, connected components)
        4. Branching node count using BRANCHING_NODE_TYPES
        5. Multiple unrelated concerns by categorizing node types

        Args:
            workflow_json: Raw n8n workflow JSON with "nodes" and "connections" keys.

        Returns:
            TurnAwareDetectionResult with detected complexity issues.
        """
        nodes = workflow_json.get("nodes", [])
        connections = workflow_json.get("connections", {})

        if len(nodes) < 2:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Need at least 2 nodes to detect complexity issues",
                detector_name=self.name,
            )

        # Build node lookup and adjacency graph
        node_lookup: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            node_name = node.get("name", "")
            if node_name:
                node_lookup[node_name] = node

        adjacency: Dict[str, List[str]] = {
            node.get("name", ""): [] for node in nodes if node.get("name")
        }
        total_edges = 0

        for source_name, outputs in connections.items():
            if not isinstance(outputs, dict):
                continue
            for output_key, output_branches in outputs.items():
                if not isinstance(output_branches, list):
                    continue
                for branch in output_branches:
                    if not isinstance(branch, list):
                        continue
                    for conn in branch:
                        if not isinstance(conn, dict):
                            continue
                        dest_name = conn.get("node", "")
                        if dest_name and source_name in adjacency:
                            adjacency[source_name].append(dest_name)
                            total_edges += 1

        issues: List[Dict[str, Any]] = []
        affected_node_indices: List[int] = []
        node_index_map = {node.get("name", ""): idx for idx, node in enumerate(nodes)}

        # Scale thresholds by workflow size: small workflows get more lenient
        # thresholds, large workflows get stricter ones.
        node_count = len(nodes)
        if node_count < 15:
            threshold_scale = 1.3  # More lenient for small workflows
        elif node_count > 40:
            threshold_scale = 0.8  # Stricter for large workflows
        else:
            threshold_scale = 1.0

        scaled_max_node_count = int(self.max_node_count * threshold_scale)
        scaled_max_branch_depth = int(self.max_branch_depth * threshold_scale)
        scaled_max_cyclomatic = int(self.max_cyclomatic_complexity * threshold_scale)

        # 1. Check node count
        if node_count > scaled_max_node_count:
            issues.append(
                {
                    "detected": True,
                    "type": "excessive_nodes",
                    "node_count": node_count,
                    "threshold": scaled_max_node_count,
                    "explanation": f"Workflow has {node_count} nodes (threshold: {scaled_max_node_count})",
                    "turns": list(range(node_count)),
                }
            )
            affected_node_indices.extend(range(node_count))

        # 2. Calculate branching depth from the connection graph
        branching_depth = self._calculate_workflow_branch_depth(adjacency, node_lookup)
        if branching_depth > scaled_max_branch_depth:
            branching_indices = [
                node_index_map[name]
                for name, data in node_lookup.items()
                if data.get("type", "") in BRANCHING_NODE_TYPES and name in node_index_map
            ]
            issues.append(
                {
                    "detected": True,
                    "type": "deep_branching",
                    "branch_depth": branching_depth,
                    "threshold": scaled_max_branch_depth,
                    "explanation": f"Workflow has {branching_depth} levels of nested branches (threshold: {scaled_max_branch_depth})",
                    "turns": branching_indices,
                }
            )
            affected_node_indices.extend(branching_indices)

        # 3. Calculate cyclomatic complexity: E - N + 2P
        num_connected_components = self._count_connected_components(
            adjacency, set(node_lookup.keys())
        )
        cyclomatic_complexity = total_edges - node_count + 2 * num_connected_components
        # Ensure minimum of 1
        cyclomatic_complexity = max(1, cyclomatic_complexity)

        if cyclomatic_complexity > scaled_max_cyclomatic:
            branching_indices = [
                node_index_map[name]
                for name, data in node_lookup.items()
                if data.get("type", "") in BRANCHING_NODE_TYPES and name in node_index_map
            ]
            issues.append(
                {
                    "detected": True,
                    "type": "high_cyclomatic_complexity",
                    "complexity": cyclomatic_complexity,
                    "edges": total_edges,
                    "nodes": node_count,
                    "connected_components": num_connected_components,
                    "threshold": scaled_max_cyclomatic,
                    "explanation": (
                        f"Workflow has cyclomatic complexity of {cyclomatic_complexity} "
                        f"(E={total_edges} - N={node_count} + 2*P={num_connected_components}, "
                        f"threshold: {scaled_max_cyclomatic})"
                    ),
                    "turns": branching_indices,
                }
            )
            affected_node_indices.extend(branching_indices)

        # 4. Count branching nodes directly from node types
        branching_nodes = []
        for node in nodes:
            node_type = node.get("type", "")
            if node_type in BRANCHING_NODE_TYPES:
                branching_nodes.append(
                    {
                        "name": node.get("name", ""),
                        "type": node_type,
                    }
                )

        # 5. Check for multiple unrelated concerns
        multiple_concerns = self._detect_workflow_multiple_concerns(nodes)
        if multiple_concerns:
            issues.append(multiple_concerns)
            affected_node_indices.extend(range(node_count))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No complexity issues detected in workflow JSON",
                detector_name=self.name,
                evidence={
                    "node_count": node_count,
                    "branching_depth": branching_depth,
                    "cyclomatic_complexity": cyclomatic_complexity,
                    "branching_nodes": len(branching_nodes),
                    "connected_components": num_connected_components,
                },
            )

        # Determine severity
        has_excessive = any(i.get("type") == "excessive_nodes" for i in issues)
        has_high_complexity = any(i.get("type") == "high_cyclomatic_complexity" for i in issues)

        severity = TurnAwareSeverity.MINOR
        if has_excessive or has_high_complexity:
            severity = TurnAwareSeverity.MODERATE
        if len(issues) >= 3:
            severity = TurnAwareSeverity.SEVERE

        confidence = 0.85 if len(issues) >= 2 else 0.75

        explanations = [issue["explanation"] for issue in issues]
        full_explanation = "; ".join(explanations)

        fixes = []
        if has_excessive or multiple_concerns:
            fixes.append("Split workflow into smaller sub-workflows using Execute Workflow nodes")
        if branching_depth > self.max_branch_depth or has_high_complexity:
            fixes.append(
                "Simplify branching logic - consider using Switch node instead of nested IFs"
            )

        suggested_fix = "; ".join(fixes) if fixes else None

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F15",
            explanation=full_explanation,
            affected_turns=sorted(set(affected_node_indices)),
            evidence={
                "issues": issues,
                "node_count": node_count,
                "branching_depth": branching_depth,
                "cyclomatic_complexity": cyclomatic_complexity,
                "branching_nodes": branching_nodes,
                "connected_components": num_connected_components,
                "total_edges": total_edges,
            },
            suggested_fix=suggested_fix,
            detector_name=self.name,
            detector_version=self.version,
        )

    def _calculate_workflow_branch_depth(
        self,
        adjacency: Dict[str, List[str]],
        node_lookup: Dict[str, Dict[str, Any]],
    ) -> int:
        """Calculate maximum branching depth from the workflow connection graph.

        Traverses the graph tracking depth changes: branching nodes (if, switch)
        increase depth, merge nodes decrease depth.

        Args:
            adjacency: Adjacency list of the workflow graph.
            node_lookup: Dict mapping node name to node definition.

        Returns:
            Maximum branch depth encountered.
        """
        if not adjacency:
            return 0

        # Find root nodes (nodes with no incoming edges)
        all_nodes = set(adjacency.keys())
        nodes_with_incoming: Set[str] = set()
        for neighbors in adjacency.values():
            for n in neighbors:
                nodes_with_incoming.add(n)
        root_nodes = all_nodes - nodes_with_incoming
        if not root_nodes:
            # No clear root, start from first node
            root_nodes = {next(iter(adjacency))} if adjacency else set()

        max_depth = 0
        visited: Set[str] = set()

        def dfs_depth(node: str, current_depth: int) -> None:
            nonlocal max_depth
            if node in visited:
                return
            visited.add(node)

            node_type = node_lookup.get(node, {}).get("type", "")

            if node_type in ("n8n-nodes-base.if", "n8n-nodes-base.switch", "n8n-nodes-base.router"):
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif node_type in ("n8n-nodes-base.merge",):
                current_depth = max(0, current_depth - 1)

            for neighbor in adjacency.get(node, []):
                dfs_depth(neighbor, current_depth)

            visited.discard(node)

        for root in root_nodes:
            dfs_depth(root, 0)

        return max_depth

    def _count_connected_components(
        self,
        adjacency: Dict[str, List[str]],
        all_nodes: Set[str],
    ) -> int:
        """Count connected components in the workflow graph (undirected).

        Args:
            adjacency: Directed adjacency list.
            all_nodes: Set of all node names.

        Returns:
            Number of connected components.
        """
        # Build undirected adjacency
        undirected: Dict[str, Set[str]] = {node: set() for node in all_nodes}
        for source, neighbors in adjacency.items():
            for dest in neighbors:
                if source in undirected:
                    undirected[source].add(dest)
                if dest in undirected:
                    undirected[dest].add(source)

        visited: Set[str] = set()
        components = 0

        for node in all_nodes:
            if node not in visited:
                components += 1
                # BFS to visit entire component
                queue = [node]
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    visited.add(current)
                    for neighbor in undirected.get(current, set()):
                        if neighbor not in visited:
                            queue.append(neighbor)

        return max(1, components)

    def _detect_workflow_multiple_concerns(
        self, nodes: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect if workflow handles multiple unrelated concerns by categorizing node types.

        Args:
            nodes: List of node definitions from workflow JSON.

        Returns:
            Dict describing the issue if detected, None otherwise.
        """
        categories: Dict[str, List[int]] = defaultdict(list)

        for i, node in enumerate(nodes):
            node_type = node.get("type", "").lower()

            if "http" in node_type or "api" in node_type:
                categories["data_fetch"].append(i)
            elif "function" in node_type or "code" in node_type:
                categories["transform"].append(i)
            elif "if" in node_type or "switch" in node_type:
                categories["validation"].append(i)
            elif "email" in node_type or "slack" in node_type or "webhook" in node_type:
                categories["notification"].append(i)
            elif (
                "database" in node_type
                or "sql" in node_type
                or "postgres" in node_type
                or "mysql" in node_type
            ):
                categories["storage"].append(i)
            elif "ai" in node_type or "openai" in node_type or "langchain" in node_type:
                categories["ai_processing"].append(i)
            elif "spreadsheet" in node_type or "csv" in node_type:
                categories["file_processing"].append(i)
            else:
                categories["other"].append(i)

        active_categories = {k: v for k, v in categories.items() if v}

        # "other" and categories that commonly co-occur (like "validation"
        # which is infrastructure, not a separate concern) are related to
        # whichever domain is present. Only count truly UNRELATED groups.
        # "transform", "validation", "other" are infrastructure — not
        # independent functional concerns.
        infrastructure_categories = {"transform", "validation", "other"}
        unrelated_categories = {k for k in active_categories if k not in infrastructure_categories}

        # Flag when a workflow touches 3+ UNRELATED functional areas,
        # which indicates it should be decomposed or split.
        if len(unrelated_categories) >= 3:
            return {
                "detected": True,
                "type": "multiple_concerns",
                "categories": list(active_categories.keys()),
                "unrelated_categories": list(unrelated_categories),
                "category_count": len(unrelated_categories),
                "explanation": f"Workflow handles {len(unrelated_categories)} unrelated concerns: {', '.join(sorted(unrelated_categories))}",
                "turns": list(range(len(nodes))),
            }

        return None

    def _detect_excessive_nodes(self, turns: List[TurnSnapshot]) -> Optional[Dict[str, Any]]:
        """Detect if workflow has too many nodes."""
        node_count = len(turns)

        if node_count > self.max_node_count:
            return {
                "detected": True,
                "type": "excessive_nodes",
                "node_count": node_count,
                "threshold": self.max_node_count,
                "explanation": f"Workflow has {node_count} nodes (threshold: {self.max_node_count})",
                "turns": list(range(len(turns))),
            }

        return None

    def _calculate_branch_depth(self, turns: List[TurnSnapshot]) -> int:
        """Calculate maximum branch depth in workflow."""
        max_depth = 0
        current_depth = 0
        depth_stack = []

        for turn in turns:
            node_type = turn.turn_metadata.get("node_type", "")

            # Branching nodes increase depth
            if node_type in {"n8n-nodes-base.if", "n8n-nodes-base.switch"}:
                depth_stack.append(current_depth)
                current_depth += 1
                max_depth = max(max_depth, current_depth)

            # Merge nodes decrease depth
            elif node_type in {"n8n-nodes-base.merge"}:
                if depth_stack:
                    current_depth = depth_stack.pop()

        return max_depth

    def _detect_deep_branching(self, turns: List[TurnSnapshot]) -> Optional[Dict[str, Any]]:
        """Detect if workflow has deeply nested branches."""
        branch_depth = self._calculate_branch_depth(turns)

        if branch_depth > self.max_branch_depth:
            return {
                "detected": True,
                "type": "deep_branching",
                "branch_depth": branch_depth,
                "threshold": self.max_branch_depth,
                "explanation": f"Workflow has {branch_depth} levels of nested branches (threshold: {self.max_branch_depth})",
                "turns": [
                    i
                    for i, turn in enumerate(turns)
                    if turn.turn_metadata.get("node_type") in BRANCHING_NODE_TYPES
                ],
            }

        return None

    def _calculate_cyclomatic_complexity(self, turns: List[TurnSnapshot]) -> int:
        """Calculate cyclomatic complexity of workflow.

        Cyclomatic complexity = E - N + 2P where:
        - E = number of edges (connections between nodes)
        - N = number of nodes
        - P = number of connected components (usually 1)

        Simplified: Start at 1, +1 for each branching node
        """
        complexity = 1  # Base complexity

        for turn in turns:
            node_type = turn.turn_metadata.get("node_type", "")
            if node_type in BRANCHING_NODE_TYPES:
                # Each branching node adds at least 1 to complexity
                complexity += 1
                # Switch nodes can add more based on number of cases
                if node_type == "n8n-nodes-base.switch":
                    cases = turn.turn_metadata.get("switch_cases", 2)
                    complexity += max(0, cases - 2)

        return complexity

    def _detect_high_cyclomatic_complexity(
        self, turns: List[TurnSnapshot]
    ) -> Optional[Dict[str, Any]]:
        """Detect if workflow has high cyclomatic complexity."""
        complexity = self._calculate_cyclomatic_complexity(turns)

        if complexity > self.max_cyclomatic_complexity:
            return {
                "detected": True,
                "type": "high_cyclomatic_complexity",
                "complexity": complexity,
                "threshold": self.max_cyclomatic_complexity,
                "explanation": f"Workflow has cyclomatic complexity of {complexity} (threshold: {self.max_cyclomatic_complexity})",
                "turns": [
                    i
                    for i, turn in enumerate(turns)
                    if turn.turn_metadata.get("node_type") in BRANCHING_NODE_TYPES
                ],
            }

        return None

    def _detect_long_execution(
        self, turns: List[TurnSnapshot], metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect if workflow consistently takes too long to execute."""
        # Get execution time from metadata
        execution_time_ms = None
        if metadata:
            execution_time_ms = metadata.get("workflow_duration_ms")

        if execution_time_ms is None and turns:
            # Try to calculate from turn timestamps
            if turns[0].turn_metadata.get("timestamp") and turns[-1].turn_metadata.get("timestamp"):
                try:
                    from datetime import datetime

                    start = turns[0].turn_metadata["timestamp"]
                    end = turns[-1].turn_metadata["timestamp"]

                    if isinstance(start, str):
                        start = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    if isinstance(end, str):
                        end = datetime.fromisoformat(end.replace("Z", "+00:00"))

                    execution_time_ms = int((end - start).total_seconds() * 1000)
                except Exception as e:
                    logger.warning(f"Failed to calculate execution time: {e}")

        if execution_time_ms is not None and execution_time_ms > self.max_execution_time_ms:
            return {
                "detected": True,
                "type": "long_execution",
                "execution_time_ms": execution_time_ms,
                "threshold": self.max_execution_time_ms,
                "explanation": f"Workflow took {execution_time_ms / 1000:.1f}s (threshold: {self.max_execution_time_ms / 1000:.1f}s)",
                "turns": list(range(len(turns))),
            }

        return None

    def _detect_multiple_concerns(self, turns: List[TurnSnapshot]) -> Optional[Dict[str, Any]]:
        """Detect if workflow handles multiple unrelated concerns.

        Heuristic: Group nodes by type and check if there are many distinct
        functional groups (data fetch, transform, validation, notification, etc.)
        """
        # Categorize nodes by function
        categories = defaultdict(list)

        for i, turn in enumerate(turns):
            node_type = turn.turn_metadata.get("node_type", "unknown")

            # Categorize by node type
            if "http" in node_type.lower() or "api" in node_type.lower():
                categories["data_fetch"].append(i)
            elif "function" in node_type.lower() or "code" in node_type.lower():
                categories["transform"].append(i)
            elif "if" in node_type.lower() or "switch" in node_type.lower():
                categories["validation"].append(i)
            elif (
                "email" in node_type.lower()
                or "slack" in node_type.lower()
                or "webhook" in node_type.lower()
            ):
                categories["notification"].append(i)
            elif "database" in node_type.lower() or "sql" in node_type.lower():
                categories["storage"].append(i)
            elif "ai" in node_type.lower() or "openai" in node_type.lower():
                categories["ai_processing"].append(i)
            else:
                categories["other"].append(i)

        # Filter out empty categories
        active_categories = {k: v for k, v in categories.items() if v}

        # If workflow has 4+ distinct functional categories, it might be too complex
        if len(active_categories) >= 4:
            return {
                "detected": True,
                "type": "multiple_concerns",
                "categories": list(active_categories.keys()),
                "category_count": len(active_categories),
                "explanation": f"Workflow handles {len(active_categories)} distinct concerns: {', '.join(active_categories.keys())}",
                "turns": list(range(len(turns))),
            }

        return None

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect complexity issues in n8n workflow."""
        if len(turns) < 2:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Need at least 2 nodes to detect complexity issues",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # 1. Detect excessive nodes
        excessive_nodes = self._detect_excessive_nodes(turns)
        if excessive_nodes:
            issues.append(excessive_nodes)
            affected_turns.extend(excessive_nodes.get("turns", []))

        # 2. Detect deep branching
        deep_branching = self._detect_deep_branching(turns)
        if deep_branching:
            issues.append(deep_branching)
            affected_turns.extend(deep_branching.get("turns", []))

        # 3. Detect high cyclomatic complexity
        high_complexity = self._detect_high_cyclomatic_complexity(turns)
        if high_complexity:
            issues.append(high_complexity)
            affected_turns.extend(high_complexity.get("turns", []))

        # 4. Detect long execution time
        long_execution = self._detect_long_execution(turns, conversation_metadata)
        if long_execution:
            issues.append(long_execution)
            affected_turns.extend(long_execution.get("turns", []))

        # 5. Detect multiple concerns
        multiple_concerns = self._detect_multiple_concerns(turns)
        if multiple_concerns:
            issues.append(multiple_concerns)
            affected_turns.extend(multiple_concerns.get("turns", []))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No complexity issues detected",
                detector_name=self.name,
            )

        # Determine severity
        severity = TurnAwareSeverity.MINOR
        if excessive_nodes or high_complexity:
            severity = TurnAwareSeverity.MODERATE
        if len(issues) >= 3:  # Multiple complexity indicators
            severity = TurnAwareSeverity.SEVERE

        # Calculate confidence
        confidence = 0.85 if len(issues) >= 2 else 0.75

        # Build explanation
        explanations = [issue["explanation"] for issue in issues]
        full_explanation = "; ".join(explanations)

        # Suggest fixes
        fixes = []
        if excessive_nodes or multiple_concerns:
            fixes.append("Split workflow into smaller sub-workflows using Execute Workflow nodes")
        if deep_branching or high_complexity:
            fixes.append(
                "Simplify branching logic - consider using Switch node instead of nested IFs"
            )
        if long_execution:
            fixes.append("Optimize slow operations or split into async sub-workflows")

        suggested_fix = "; ".join(fixes) if fixes else None

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F15",
            explanation=full_explanation,
            affected_turns=sorted(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=suggested_fix,
            detector_name=self.name,
            detector_version=self.version,
        )
