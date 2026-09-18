"""
F5: Flawed Workflow Design Detection (MAST Taxonomy)
====================================================

Detects structural problems in agent workflow design:
- Unreachable nodes (dead ends)
- Missing error handling paths
- Infinite loop potential
- Bottleneck nodes
- Missing termination conditions
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class WorkflowIssue(str, Enum):
    UNREACHABLE_NODE = "unreachable_node"
    DEAD_END = "dead_end"
    MISSING_ERROR_HANDLING = "missing_error_handling"
    INFINITE_LOOP_RISK = "infinite_loop_risk"
    BOTTLENECK = "bottleneck"
    MISSING_TERMINATION = "missing_termination"
    ORPHAN_NODE = "orphan_node"
    EXCESSIVE_DEPTH = "excessive_depth"  # v1.1: Long sequential chains


class WorkflowSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class WorkflowNode:
    id: str
    name: str
    node_type: str
    incoming: list[str]
    outgoing: list[str]
    has_error_handler: bool = False
    is_terminal: bool = False


@dataclass
class WorkflowAnalysisResult:
    detected: bool
    issues: list[WorkflowIssue]
    severity: WorkflowSeverity
    confidence: float
    node_count: int
    edge_count: int
    problematic_nodes: list[str]
    explanation: str
    suggested_fix: Optional[str] = None
    # True when every issue is a baseline artifact of linearizing a
    # conversation/tool stream into nodes (MISSING_TERMINATION + DEAD_END +
    # EXCESSIVE_DEPTH) with no real design issue present. Callers that know
    # their channel produces linear streams (claude_code) gate on this.
    artifact_only: bool = False


class FlawedWorkflowDetector:
    """
    Detects F5: Flawed Workflow Design - structural problems in agent graphs.

    Analyzes workflow topology for unreachable nodes, dead ends,
    missing error handling, and other structural issues.
    """

    def __init__(
        self,
        require_error_handling: bool = True,
        max_bottleneck_ratio: float = 0.5,
    ):
        self.require_error_handling = require_error_handling
        self.max_bottleneck_ratio = max_bottleneck_ratio

    def _build_graph(self, nodes: list[WorkflowNode]) -> tuple[dict, dict]:
        forward = defaultdict(list)
        backward = defaultdict(list)

        for node in nodes:
            for target in node.outgoing:
                forward[node.id].append(target)
                backward[target].append(node.id)

        return dict(forward), dict(backward)

    def _find_reachable(
        self,
        start: str,
        graph: dict[str, list[str]],
    ) -> set[str]:
        visited = set()
        stack = [start]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    stack.append(neighbor)

        return visited

    def _detect_unreachable(
        self,
        nodes: list[WorkflowNode],
        forward: dict,
        backward: dict,
    ) -> list[str]:
        entry_points = [n.id for n in nodes if not backward.get(n.id)]
        if not entry_points:
            entry_points = [nodes[0].id] if nodes else []

        reachable = set()
        for entry in entry_points:
            reachable |= self._find_reachable(entry, forward)

        all_nodes = {n.id for n in nodes}
        return list(all_nodes - reachable)

    def _detect_dead_ends(
        self,
        nodes: list[WorkflowNode],
        forward: dict,
    ) -> list[str]:
        terminal_nodes = {n.id for n in nodes if n.is_terminal}

        dead_ends = []
        for node in nodes:
            if not forward.get(node.id) and node.id not in terminal_nodes:
                dead_ends.append(node.id)

        return dead_ends

    def _detect_infinite_loop_risk(
        self,
        nodes: list[WorkflowNode],
        forward: dict,
    ) -> list[str]:
        """Nodes that can reach a cycle (the loop-risk set), in O(V + E).

        A node is risky iff a forward walk from it can revisit a node already
        on its path — i.e. it can reach a cycle — or it has a self-loop.

        The previous implementation enumerated every simple path from every
        node (a per-path ``visited`` set with no global "fully explored" memo).
        On a cycle-free DAG it never short-circuits, so it walks all paths,
        which is exponential in node count when fan-out > 1. This computes the
        same result by finding cycle members via an iterative three-colour DFS,
        then taking reverse-reachability to every node that can reach one.
        """
        node_ids = [n.id for n in nodes]
        # Explicit self-loops are trivially on a cycle.
        on_cycle: set[str] = {nid for nid in node_ids if nid in forward.get(nid, [])}

        white, gray, black = 0, 1, 2
        color: dict[str, int] = {nid: white for nid in node_ids}

        for root in node_ids:
            if color[root] != white:
                continue
            # Iterative DFS. ``path`` mirrors the GRAY stack (the simple path
            # from root); ``path_pos`` maps a node to its index so a back-edge
            # can mark the whole cycle segment with one slice. Each edge is
            # traversed once and each node coloured BLACK once → O(V + E).
            color[root] = gray
            path: list[str] = [root]
            path_pos: dict[str, int] = {root: 0}
            stack = [(root, iter(forward.get(root, [])))]
            while stack:
                node, it = stack[-1]
                descended = False
                for nb in it:
                    c = color.get(nb)
                    if c is None:
                        continue  # edge to an unknown node id; ignore
                    if c == white:
                        color[nb] = gray
                        path_pos[nb] = len(path)
                        path.append(nb)
                        stack.append((nb, iter(forward.get(nb, []))))
                        descended = True
                        break
                    if c == gray:
                        # Back-edge: nb .. current top are all on a cycle.
                        for member in path[path_pos[nb] :]:
                            on_cycle.add(member)
                    # BLACK neighbour: fully explored, no new cycle via it.
                if not descended:
                    color[node] = black
                    path.pop()
                    path_pos.pop(node, None)
                    stack.pop()

        if not on_cycle:
            return []

        # Reverse-reachability: any node that can reach a cycle member is risky.
        reverse: dict[str, list[str]] = defaultdict(list)
        for src, targets in forward.items():
            for tgt in targets:
                reverse[tgt].append(src)

        risky: set[str] = set(on_cycle)
        work = list(on_cycle)
        while work:
            cur = work.pop()
            for pred in reverse.get(cur, []):
                if pred not in risky:
                    risky.add(pred)
                    work.append(pred)

        return list(risky)

    def _detect_bottlenecks(
        self,
        nodes: list[WorkflowNode],
        forward: dict,
        backward: dict,
    ) -> list[str]:
        bottlenecks = []
        total_edges = sum(len(edges) for edges in forward.values())
        num_nodes = len(nodes)

        for node in nodes:
            incoming = len(backward.get(node.id, []))
            outgoing = len(forward.get(node.id, []))

            if total_edges > 0:
                throughput = (incoming + outgoing) / total_edges
                # v1.1: Improved bottleneck formula — also detect nodes
                # with incoming > half the total nodes (convergence hub)
                is_throughput_bottleneck = throughput > self.max_bottleneck_ratio and incoming > 2
                is_convergence_hub = num_nodes > 3 and incoming > num_nodes / 2
                if is_throughput_bottleneck or is_convergence_hub:
                    bottlenecks.append(node.id)

        return bottlenecks

    def _detect_missing_error_handling(
        self,
        nodes: list[WorkflowNode],
    ) -> list[str]:
        missing = []
        for node in nodes:
            if not node.has_error_handler and node.node_type not in ["start", "end", "condition"]:
                missing.append(node.id)
        return missing

    def _detect_orphan_nodes(
        self,
        nodes: list[WorkflowNode],
        forward: dict,
        backward: dict,
    ) -> list[str]:
        """Detect orphan nodes — disconnected from the main flow.

        An orphan is a non-start node with no incoming connections, indicating
        it can never be reached during normal workflow execution.
        """
        if len(nodes) <= 1:
            return []
        # Find the start node(s)
        {n.id for n in nodes if n.node_type == "start" or not backward.get(n.id)}
        # A node is orphan if it has no incoming but is not the primary start
        primary_start = nodes[0].id if nodes else None
        orphans = []
        for node in nodes:
            if node.id == primary_start:
                continue
            has_incoming = bool(backward.get(node.id))
            has_outgoing = bool(forward.get(node.id))
            if not has_incoming and not has_outgoing:
                orphans.append(node.id)
            elif not has_incoming and node.node_type not in ("start",):
                # Non-start node with no incoming = unreachable
                orphans.append(node.id)
        return orphans

    def _detect_excessive_depth(
        self,
        nodes: list[WorkflowNode],
        forward: dict,
        backward: dict,
    ) -> int:
        """v1.1: Detect excessive sequential chain depth.

        Returns the longest path length.  Workflows with depth > 8 are
        likely over-sequential and should be parallelised or decomposed.
        """
        if not nodes:
            return 0

        # Find entry points (no incoming edges)
        entry_points = [n.id for n in nodes if not backward.get(n.id)]
        if not entry_points:
            entry_points = [nodes[0].id]

        max_depth = 0
        for start in entry_points:
            visited: set[str] = set()
            stack = [(start, 0)]
            while stack:
                node_id, depth = stack.pop()
                if node_id in visited:
                    continue
                visited.add(node_id)
                max_depth = max(max_depth, depth)
                for neighbor in forward.get(node_id, []):
                    if neighbor not in visited:
                        stack.append((neighbor, depth + 1))

        return max_depth

    def _detect_missing_termination(
        self,
        nodes: list[WorkflowNode],
        forward: dict,
    ) -> bool:
        terminal_nodes = [n for n in nodes if n.is_terminal]
        if not terminal_nodes:
            return True

        for node in nodes:
            if not forward.get(node.id):
                if not any(n.id == node.id for n in terminal_nodes):
                    pass

        return False

    def detect(
        self,
        nodes: list[WorkflowNode],
    ) -> WorkflowAnalysisResult:
        if not nodes:
            return WorkflowAnalysisResult(
                detected=False,
                issues=[],
                severity=WorkflowSeverity.NONE,
                confidence=0.0,
                node_count=0,
                edge_count=0,
                problematic_nodes=[],
                explanation="No workflow nodes provided",
            )

        forward, backward = self._build_graph(nodes)
        edge_count = sum(len(edges) for edges in forward.values())

        issues = []
        problematic = []

        unreachable = self._detect_unreachable(nodes, forward, backward)
        if unreachable:
            issues.append(WorkflowIssue.UNREACHABLE_NODE)
            problematic.extend(unreachable)

        dead_ends = self._detect_dead_ends(nodes, forward)
        if dead_ends:
            issues.append(WorkflowIssue.DEAD_END)
            problematic.extend(dead_ends)

        loop_risk = self._detect_infinite_loop_risk(nodes, forward)
        if loop_risk:
            issues.append(WorkflowIssue.INFINITE_LOOP_RISK)
            problematic.extend(loop_risk)

        bottlenecks = self._detect_bottlenecks(nodes, forward, backward)
        if bottlenecks:
            issues.append(WorkflowIssue.BOTTLENECK)
            problematic.extend(bottlenecks)

        orphans = self._detect_orphan_nodes(nodes, forward, backward)
        if orphans:
            issues.append(WorkflowIssue.ORPHAN_NODE)
            problematic.extend(orphans)

        # v1.1: Check for excessive sequential depth
        max_depth = self._detect_excessive_depth(nodes, forward, backward)
        if max_depth > 5:
            issues.append(WorkflowIssue.EXCESSIVE_DEPTH)

        if self.require_error_handling:
            missing_handlers = self._detect_missing_error_handling(nodes)
            non_trivial = [n for n in nodes if n.node_type not in ("start", "end", "condition")]
            has_any_handler = any(n.has_error_handler for n in nodes)
            # Only flag when the workflow shows intent to use error handlers
            # (at least one node has one) but significant nodes are missing them.
            # If NO nodes have handlers, it's a design choice — not a defect.
            if (
                missing_handlers
                and has_any_handler
                and len(missing_handlers) >= len(non_trivial) * 0.5
            ):
                issues.append(WorkflowIssue.MISSING_ERROR_HANDLING)
                problematic.extend(missing_handlers[:5])

        if self._detect_missing_termination(nodes, forward):
            issues.append(WorkflowIssue.MISSING_TERMINATION)

        if not issues:
            return WorkflowAnalysisResult(
                detected=False,
                issues=[],
                severity=WorkflowSeverity.NONE,
                confidence=0.9,
                node_count=len(nodes),
                edge_count=edge_count,
                problematic_nodes=[],
                explanation="Workflow structure appears valid",
            )

        # Separate structural artifacts (present in any linear sequential trace)
        # from real workflow design issues that indicate intentional flaws.
        # MISSING_TERMINATION, DEAD_END, and EXCESSIVE_DEPTH arise from the
        # node-building logic for conversation traces, not from design choices.
        baseline_artifacts = frozenset(
            {
                WorkflowIssue.MISSING_TERMINATION,
                WorkflowIssue.DEAD_END,
                WorkflowIssue.EXCESSIVE_DEPTH,
            }
        )
        real_issues = [i for i in issues if i not in baseline_artifacts]
        artifacts_only = len(real_issues) == 0

        if WorkflowIssue.INFINITE_LOOP_RISK in issues or (
            WorkflowIssue.MISSING_TERMINATION in issues and real_issues
        ):
            severity = WorkflowSeverity.SEVERE
        elif len(issues) >= 3:
            severity = WorkflowSeverity.MODERATE
        else:
            severity = WorkflowSeverity.MINOR

        # v1.1: Confidence based on severity and issue count.
        # v1.2: Artifact-only traces (linear conversations that always fire
        # MISSING_TERMINATION + DEAD_END + EXCESSIVE_DEPTH) get a lower,
        # depth-scaled confidence — they're not genuine design flaws.
        # v1.3: Floor raised so deeper traces stay at HIGH severity (conf≥0.80)
        # while still being graded below the previous constant 0.90.
        if artifacts_only:
            confidence = min(0.90, 0.65 + max_depth * 0.020)
        elif severity == WorkflowSeverity.SEVERE:
            confidence = min(0.75 + len(issues) * 0.05, 0.95)
        elif len(issues) >= 3:
            confidence = min(0.6 + len(issues) * 0.05, 0.90)
        elif len(issues) >= 2:
            confidence = 0.55
        else:
            # Single issue: discriminate by issue type.
            # Structural issues (dead ends, unreachable) are more concerning
            # than advisory issues (bottleneck, orphan, missing error handling).
            single_issue = issues[0]
            if single_issue in (WorkflowIssue.DEAD_END, WorkflowIssue.UNREACHABLE_NODE):
                confidence = 0.70
            elif single_issue == WorkflowIssue.EXCESSIVE_DEPTH:
                # Scale with actual depth: deeper sequential chains are more
                # concerning than marginally-over-threshold ones.
                confidence = min(0.75, 0.40 + max_depth * 0.025)
            elif single_issue in (
                WorkflowIssue.ORPHAN_NODE,
                WorkflowIssue.BOTTLENECK,
                WorkflowIssue.MISSING_ERROR_HANDLING,
            ):
                confidence = 0.50
            else:
                confidence = 0.55

        issue_names = [i.value for i in issues]
        unique_problematic = sorted(set(problematic))[:5]
        explanation = (
            f"Workflow has {len(issues)} structural issues: {', '.join(issue_names)}. "
            f"Affected nodes: {', '.join(unique_problematic)}"
        )

        fixes = []
        if WorkflowIssue.UNREACHABLE_NODE in issues:
            fixes.append("Add edges to unreachable nodes or remove them")
        if WorkflowIssue.DEAD_END in issues:
            fixes.append("Add termination or continuation from dead-end nodes")
        if WorkflowIssue.INFINITE_LOOP_RISK in issues:
            fixes.append("Add loop counters or termination conditions")
        if WorkflowIssue.BOTTLENECK in issues:
            fixes.append("Parallelize or split bottleneck nodes")
        if WorkflowIssue.MISSING_ERROR_HANDLING in issues:
            fixes.append("Add error handlers to critical nodes")

        return WorkflowAnalysisResult(
            detected=True,
            issues=issues,
            severity=severity,
            confidence=confidence,
            node_count=len(nodes),
            edge_count=edge_count,
            problematic_nodes=sorted(set(problematic)),
            explanation=explanation,
            suggested_fix="; ".join(fixes) if fixes else None,
            artifact_only=artifacts_only,
        )

    def detect_from_trace(
        self,
        trace: dict,
    ) -> WorkflowAnalysisResult:
        spans = trace.get("spans", [])
        if not spans:
            return WorkflowAnalysisResult(
                detected=False,
                issues=[],
                severity=WorkflowSeverity.NONE,
                confidence=0.0,
                node_count=0,
                edge_count=0,
                problematic_nodes=[],
                explanation="No spans in trace",
            )

        nodes = []
        for i, span in enumerate(spans):
            incoming = [spans[i - 1].get("name", f"span_{i - 1}")] if i > 0 else []
            outgoing = [spans[i + 1].get("name", f"span_{i + 1}")] if i < len(spans) - 1 else []

            nodes.append(
                WorkflowNode(
                    id=span.get("span_id", f"span_{i}"),
                    name=span.get("name", f"span_{i}"),
                    node_type=span.get("type", "agent"),
                    incoming=incoming,
                    outgoing=outgoing,
                    has_error_handler="error" in span.get("attributes", {}),
                    is_terminal=(i == len(spans) - 1),
                )
            )

        return self.detect(nodes)
