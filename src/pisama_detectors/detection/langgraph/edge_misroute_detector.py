"""
LangGraph Edge Misroute Detector
==================================

Detects routing issues in LangGraph conditional edges:
- Target node referenced by edge does not exist in nodes list
- Dead-end routes: non-terminal nodes with no outgoing edges
- Unreachable nodes: nodes not targeted by any edge and not the entry point
- Condition name mismatches: condition text suggesting wrong routing
- Condition-title semantic mismatch: condition references target type/name
  that contradicts the actual target node
- State value vs edge condition contradiction
- Node output vs edge condition contradiction
- Condition value vs target name semantic mismatch
- Skipped conditional edge targets (wrong branch taken)
"""

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

from pisama_detectors.detection.turn_aware._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)

logger = logging.getLogger(__name__)

# Node types that are expected to be terminal (no outgoing edges needed)
TERMINAL_NODE_TYPES = {"human", "end", "__end__"}

# Condition name patterns that suggest routing to an end/terminal
END_CONDITION_PATTERNS = {"end", "finish", "stop", "terminate", "complete", "done"}

# Condition name patterns that suggest routing to processing/continuation
PROCESS_CONDITION_PATTERNS = {"process", "continue", "next", "run", "execute", "start"}

# Node type keywords for semantic matching
NODE_TYPE_KEYWORDS = {
    "tool": {"tool", "execute", "action", "function", "invoke"},
    "llm": {"llm", "model", "generate", "respond", "chat", "answer"},
    "human": {"human", "user", "manual", "approve", "review"},
    "router": {"router", "route", "switch", "branch", "decide"},
}


class LangGraphEdgeMisrouteDetector(TurnAwareDetector):
    """Detects edge misrouting in LangGraph graph executions.

    Analyzes edges and nodes for:
    1. Missing target nodes (edge points to non-existent node)
    2. Dead-end routes (non-terminal nodes without outgoing edges)
    3. Unreachable nodes (no incoming edges, not the first node)
    4. Condition name mismatches (condition text contradicts target)
    5. Condition-title semantic mismatch
    6. State value contradicting edge condition
    7. Node output contradicting edge condition
    8. Condition value vs target name mismatch
    9. Skipped conditional edge targets
    """

    name = "LangGraphEdgeMisrouteDetector"
    version = "1.2"
    supported_failure_modes = ["F12"]

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

    def detect_graph_execution(
        self, graph_execution: Dict[str, Any]
    ) -> TurnAwareDetectionResult:
        nodes = graph_execution.get("nodes", [])
        edges = graph_execution.get("edges", [])

        if not nodes:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No nodes in graph execution",
                detector_name=self.name,
            )

        # Build lookup structures
        node_ids: Set[str] = {n.get("node_id", "") for n in nodes}
        node_map: Dict[str, Dict[str, Any]] = {
            n.get("node_id", ""): n for n in nodes
        }
        node_types: Dict[str, str] = {
            n.get("node_id", ""): n.get("node_type", "") for n in nodes
        }
        node_titles: Dict[str, str] = {
            n.get("node_id", ""): n.get("title", "") for n in nodes
        }

        nodes_with_outgoing: Set[str] = set()
        nodes_with_incoming: Set[str] = set()

        issues: List[Dict[str, Any]] = []

        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            edge_type = edge.get("edge_type", "")
            condition = edge.get("condition", "")

            nodes_with_outgoing.add(source)
            nodes_with_incoming.add(target)

            # 1. Missing target node (skip known terminals and unexecuted sources)
            if target and target not in node_ids and target not in TERMINAL_NODE_TYPES:
                source_node_info = node_map.get(source, {})
                source_status = source_node_info.get("status", "")
                if source_status in ("succeeded", "completed"):
                    # For conditional edges, only flag if no sibling target
                    # was executed (otherwise this is just the not-taken path)
                    skip = False
                    if edge_type == "conditional":
                        for other_edge in edges:
                            if (other_edge.get("source") == source
                                    and other_edge.get("target") != target
                                    and other_edge.get("edge_type") == "conditional"):
                                other_tgt = node_map.get(
                                    other_edge.get("target", ""), {}
                                )
                                if other_tgt.get("status") in (
                                    "succeeded", "completed"
                                ):
                                    skip = True
                                    break
                    if not skip:
                        issues.append({
                            "type": "missing_target",
                            "source": source,
                            "target": target,
                            "edge_type": edge_type,
                            "description": (
                                f"Edge from '{source}' targets non-existent "
                                f"node '{target}'"
                            ),
                        })

            # 4. Condition name mismatch (end/process patterns)
            if edge_type == "conditional" and condition and target in node_ids:
                mismatch = self._check_condition_mismatch(
                    condition, target, node_types.get(target, ""),
                    node_titles.get(target, ""),
                )
                if mismatch:
                    issues.append(mismatch)

            # 5. Condition-title semantic mismatch (route_to_X type check)
            if edge_type == "conditional" and condition and target in node_ids:
                semantic = self._check_condition_title_mismatch(
                    condition, target,
                    node_types.get(target, ""),
                    node_titles.get(target, ""),
                )
                if semantic:
                    issues.append(semantic)

        # 2. Dead-end routes
        for node in nodes:
            nid = node.get("node_id", "")
            ntype = node.get("node_type", "")
            title = node.get("title", "").lower()

            is_terminal = (
                ntype in TERMINAL_NODE_TYPES
                or title in TERMINAL_NODE_TYPES
                or nid in TERMINAL_NODE_TYPES
            )

            if nid not in nodes_with_outgoing and not is_terminal:
                status = graph_execution.get("status", "")
                if status == "completed":
                    node_superstep = node.get("superstep", -1)
                    max_superstep = max(
                        (n.get("superstep", 0) for n in nodes), default=0
                    )
                    if node_superstep >= max_superstep:
                        continue

                issues.append({
                    "type": "dead_end",
                    "node_id": nid,
                    "node_type": ntype,
                    "title": node.get("title", ""),
                    "description": (
                        f"Node '{nid}' ({ntype}) has no outgoing edges "
                        f"but is not a terminal node"
                    ),
                })

        # 3. Unreachable nodes
        if nodes:
            entry_node = min(nodes, key=lambda n: n.get("superstep", 0))
            entry_id = entry_node.get("node_id", "")

            for node in nodes:
                nid = node.get("node_id", "")
                if nid != entry_id and nid not in nodes_with_incoming:
                    issues.append({
                        "type": "unreachable",
                        "node_id": nid,
                        "node_type": node.get("node_type", ""),
                        "title": node.get("title", ""),
                        "description": (
                            f"Node '{nid}' has no incoming edges and "
                            f"is not the entry point"
                        ),
                    })

        # 6. State value vs edge condition contradiction
        state_snapshots = graph_execution.get("state_snapshots", [])
        if state_snapshots and edges:
            state_issues = self._check_state_condition_contradiction(
                edges, state_snapshots, node_map
            )
            issues.extend(state_issues)

        # 7. Node output vs edge condition contradiction
        output_issues = self._check_output_condition_contradiction(
            edges, node_map
        )
        issues.extend(output_issues)

        # 8. Condition value vs target name semantic mismatch
        cond_target_issues = self._check_condition_value_target_mismatch(
            edges, node_map
        )
        issues.extend(cond_target_issues)

        # 9. Skipped conditional edge targets
        skipped_issues = self._check_skipped_conditional_targets(
            edges, node_map, state_snapshots
        )
        issues.extend(skipped_issues)

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="No edge routing issues detected",
                detector_name=self.name,
            )

        issue_types = {i["type"] for i in issues}
        has_output_contradiction = "output_condition_contradiction" in issue_types
        has_state_contradiction = "state_condition_contradiction" in issue_types
        has_missing = "missing_target" in issue_types
        has_cond_target = "condition_value_target_mismatch" in issue_types

        if has_missing or has_state_contradiction or has_output_contradiction:
            confidence = min(0.95, 0.8 + len(issues) * 0.05)
        elif has_cond_target:
            confidence = min(0.92, 0.70 + len(issues) * 0.08)
        elif "condition_title_mismatch" in issue_types or "skipped_conditional" in issue_types:
            confidence = min(0.90, 0.65 + len(issues) * 0.08)
        else:
            confidence = min(0.90, 0.6 + len(issues) * 0.1)

        if has_missing or has_state_contradiction or has_output_contradiction:
            severity = TurnAwareSeverity.SEVERE
        elif "dead_end" in issue_types or "condition_title_mismatch" in issue_types or has_cond_target:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        type_counts: Dict[str, int] = {}
        for i in issues:
            type_counts[i["type"]] = type_counts.get(i["type"], 0) + 1
        summary_parts = [f"{count} {itype}" for itype, count in type_counts.items()]

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F12",
            explanation=f"Edge routing issues: {', '.join(summary_parts)}",
            affected_turns=[],
            evidence={
                "issues": issues,
                "total_nodes": len(nodes),
                "total_edges": len(edges),
            },
            suggested_fix=(
                "Verify all conditional edge targets exist in the graph. "
                "Ensure non-terminal nodes have outgoing edges. "
                "Check that condition functions route to the intended nodes."
            ),
            detector_name=self.name,
        )

    def _check_condition_mismatch(
        self,
        condition: str,
        target_id: str,
        target_type: str,
        target_title: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if a condition name contradicts the target node (end vs process)."""
        condition_lower = condition.lower().replace("_", " ")
        target_lower = (target_title or target_id).lower()

        condition_suggests_end = any(
            p in condition_lower for p in END_CONDITION_PATTERNS
        )
        target_is_processing = target_type in ("llm", "tool", "subgraph", "map_reduce")

        if condition_suggests_end and target_is_processing:
            return {
                "type": "condition_mismatch",
                "condition": condition,
                "target_id": target_id,
                "target_type": target_type,
                "description": (
                    f"Condition '{condition}' suggests termination but "
                    f"routes to processing node '{target_id}' ({target_type})"
                ),
            }

        condition_suggests_process = any(
            p in condition_lower for p in PROCESS_CONDITION_PATTERNS
        )
        target_is_terminal = (
            target_type in TERMINAL_NODE_TYPES
            or any(p in target_lower for p in END_CONDITION_PATTERNS)
        )

        if condition_suggests_process and target_is_terminal:
            return {
                "type": "condition_mismatch",
                "condition": condition,
                "target_id": target_id,
                "target_type": target_type,
                "description": (
                    f"Condition '{condition}' suggests continuation but "
                    f"routes to terminal node '{target_id}'"
                ),
            }

        return None

    def _check_condition_title_mismatch(
        self,
        condition: str,
        target_id: str,
        target_type: str,
        target_title: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if condition references a node type that contradicts the target.

        For example: condition says "route_to_tool" but target is an LLM node.
        """
        cond_lower = condition.lower().replace("_", " ").replace("==", " ")

        # Only check "route_to_X" type patterns in the condition text
        for ntype, keywords in NODE_TYPE_KEYWORDS.items():
            for kw in keywords:
                pattern = rf"(?:route.to|go.to|send.to|path.to)\s*{kw}"
                if re.search(pattern, cond_lower):
                    if target_type and target_type != ntype:
                        return {
                            "type": "condition_title_mismatch",
                            "condition": condition,
                            "target_id": target_id,
                            "target_type": target_type,
                            "target_title": target_title,
                            "description": (
                                f"Condition '{condition}' references '{kw}' "
                                f"but routes to {target_type} node "
                                f"'{target_title or target_id}'"
                            ),
                        }

        return None

    def _check_state_condition_contradiction(
        self,
        edges: List[Dict[str, Any]],
        state_snapshots: List[Dict[str, Any]],
        node_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check if state values contradict the edge conditions that were taken."""
        issues = []

        # Build state at each superstep
        state_by_step: Dict[int, Dict[str, Any]] = {}
        for ss in state_snapshots:
            step = ss.get("superstep", -1)
            state_by_step[step] = ss.get("state", {})

        # Group edges by source to detect parallel execution
        edges_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            if edge.get("edge_type") == "conditional":
                edges_by_source[edge.get("source", "")].append(edge)

        for edge in edges:
            if edge.get("edge_type") != "conditional":
                continue
            condition = edge.get("condition", "")
            target = edge.get("target", "")
            source = edge.get("source", "")
            if not condition:
                continue

            target_node = node_map.get(target, {})
            target_status = target_node.get("status", "")

            # Only check edges where the target was actually executed
            if target_status not in ("succeeded", "completed"):
                continue

            # Parse condition: "key == 'value'" pattern
            match = re.match(
                r"(\w+)\s*==\s*['\"]([^'\"]+)['\"]", condition
            )
            if not match:
                continue

            cond_key = match.group(1)
            cond_value = match.group(2)

            # Check state at the router's superstep
            source_node = node_map.get(source, {})
            source_step = source_node.get("superstep", -1)

            state = state_by_step.get(source_step, {})
            if not state:
                state = state_by_step.get(source_step + 1, {})

            actual_value = state.get(cond_key)
            if actual_value is not None and str(actual_value) != cond_value:
                # Skip if all siblings succeeded AND state matches a sibling
                sibling_edges = edges_by_source.get(source, [])
                if len(sibling_edges) > 1:
                    all_succeeded = all(
                        node_map.get(e.get("target", ""), {}).get("status", "")
                        in ("succeeded", "completed")
                        for e in sibling_edges
                    )
                    if all_succeeded:
                        output_matches_sibling = False
                        for sib in sibling_edges:
                            sib_match = re.match(
                                r"(\w+)\s*==\s*['\"]([^'\"]+)['\"]",
                                sib.get("condition", ""),
                            )
                            if sib_match and sib_match.group(1) == cond_key:
                                if str(actual_value) == sib_match.group(2):
                                    output_matches_sibling = True
                                    break
                        if output_matches_sibling:
                            continue

                issues.append({
                    "type": "state_condition_contradiction",
                    "condition": condition,
                    "source": source,
                    "target": target,
                    "expected_value": cond_value,
                    "actual_value": str(actual_value),
                    "state_key": cond_key,
                    "description": (
                        f"State has {cond_key}='{actual_value}' but edge "
                        f"condition '{condition}' was taken (routing to '{target}')"
                    ),
                })

        return issues

    def _check_output_condition_contradiction(
        self,
        edges: List[Dict[str, Any]],
        node_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check if node outputs contradict the edge condition that was taken.

        For each conditional edge where the target was executed, check if
        the source node (or any upstream node) produced an output value for
        the condition key that contradicts the condition value.
        """
        issues = []

        # Group edges by source to detect parallel execution
        edges_by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            if edge.get("edge_type") == "conditional":
                edges_by_source[edge.get("source", "")].append(edge)

        for edge in edges:
            if edge.get("edge_type") != "conditional":
                continue
            condition = edge.get("condition", "")
            target = edge.get("target", "")
            source = edge.get("source", "")
            if not condition:
                continue

            target_node = node_map.get(target, {})
            target_status = target_node.get("status", "")

            # Only check executed edges
            if target_status not in ("succeeded", "completed"):
                continue

            # Parse "key == 'value'" pattern
            match = re.match(
                r"(\w+)\s*==\s*['\"]([^'\"]+)['\"]", condition
            )
            if not match:
                continue

            cond_key = match.group(1)
            cond_value = match.group(2)

            # Check source node outputs first
            source_node = node_map.get(source, {})
            source_outputs = source_node.get("outputs", {})
            source_step = source_node.get("superstep", -1)

            actual_value = source_outputs.get(cond_key)

            # If not in source, check upstream nodes (at or before source superstep)
            if actual_value is None:
                for nid, node in node_map.items():
                    if node.get("superstep", 99) <= source_step:
                        out = node.get("outputs", {})
                        if cond_key in out:
                            actual_value = out[cond_key]
                            break

            if actual_value is not None and str(actual_value) != cond_value:
                # Skip if all sibling branches succeeded AND the output
                # matches one of the sibling conditions (true parallel exec)
                sibling_edges = edges_by_source.get(source, [])
                if len(sibling_edges) > 1:
                    all_succeeded = all(
                        node_map.get(e.get("target", ""), {}).get("status", "")
                        in ("succeeded", "completed")
                        for e in sibling_edges
                    )
                    if all_succeeded:
                        # Check if output matches any sibling condition
                        output_matches_sibling = False
                        for sib in sibling_edges:
                            sib_match = re.match(
                                r"(\w+)\s*==\s*['\"]([^'\"]+)['\"]",
                                sib.get("condition", ""),
                            )
                            if sib_match and sib_match.group(1) == cond_key:
                                if str(actual_value) == sib_match.group(2):
                                    output_matches_sibling = True
                                    break
                        if output_matches_sibling:
                            continue

                issues.append({
                    "type": "output_condition_contradiction",
                    "condition": condition,
                    "source": source,
                    "target": target,
                    "expected_value": cond_value,
                    "actual_value": str(actual_value),
                    "key": cond_key,
                    "description": (
                        f"Node output has {cond_key}='{actual_value}' but "
                        f"edge condition '{condition}' was taken "
                        f"(routing to '{target}')"
                    ),
                })

        return issues

    def _check_condition_value_target_mismatch(
        self,
        edges: List[Dict[str, Any]],
        node_map: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check if condition value doesn't match the target node name.

        For example: condition says route == 'weather_service' but the target
        node is named 'NewsService' or 'DatabaseQuery'. The condition value
        should semantically relate to the target.
        """
        issues = []

        # Generic suffixes that don't count as meaningful overlap
        generic_suffixes = {
            "tool", "service", "handler", "agent", "processor",
            "node", "queue", "manager", "worker", "module",
            "search", "path", "process", "step",
        }

        # Generic condition values not expected to match target names
        generic_values = {
            "true", "false", "yes", "no", "default", "none",
            "high", "low", "medium", "valid", "invalid",
            "premium", "standard", "basic", "normal",
            "available", "unavailable", "found", "not_found",
            "success", "error", "failed", "pending",
            "positive", "negative", "neutral",
            "user_found",
        }

        for edge in edges:
            if edge.get("edge_type") != "conditional":
                continue
            condition = edge.get("condition", "")
            target = edge.get("target", "")
            if not condition:
                continue

            target_node = node_map.get(target, {})
            target_status = target_node.get("status", "")

            # Only check executed edges
            if target_status not in ("succeeded", "completed"):
                continue

            target_title = target_node.get("title", "")
            if not target_title:
                continue

            # Extract condition value
            value_match = re.search(r"['\"]([^'\"]+)['\"]", condition)
            if not value_match:
                continue

            cond_value = value_match.group(1)

            # Skip generic values
            if cond_value.lower() in generic_values:
                continue
            if len(cond_value) < 3:
                continue

            # Normalize for comparison
            cond_val_lower = cond_value.lower().replace("_", "").replace("-", "")
            title_lower = target_title.lower().replace("_", "").replace("-", "")

            # Skip if condition value is in the target title or vice versa
            if cond_val_lower in title_lower or title_lower in cond_val_lower:
                continue

            # Split into meaningful words
            cond_words = set(re.split(r"[\s_\-]", cond_value.lower()))
            cond_words -= {"", "to", "the", "a", "is", "and", "or"}

            title_words = set(
                re.split(r"(?<=[a-z])(?=[A-Z])|[\s_\-]", target_title)
            )
            title_words = {w.lower() for w in title_words}
            title_words -= {"", "to", "the", "a", "is", "and", "or"}

            # Remove generic suffixes — they don't count as meaningful overlap
            meaningful_cond = cond_words - generic_suffixes
            meaningful_title = title_words - generic_suffixes

            # Check for meaningful word overlap (including singular/plural)
            has_meaningful_overlap = False
            for cw in meaningful_cond:
                for tw in meaningful_title:
                    # Exact match or singular/plural match
                    if cw == tw or cw.rstrip("s") == tw.rstrip("s"):
                        has_meaningful_overlap = True
                        break
                if has_meaningful_overlap:
                    break

            if has_meaningful_overlap:
                continue

            # No meaningful overlap — flag as mismatch
            if not meaningful_cond:
                # All condition words are generic, skip
                continue

            issues.append({
                "type": "condition_value_target_mismatch",
                "condition": condition,
                "target": target,
                "target_title": target_title,
                "condition_value": cond_value,
                "description": (
                    f"Condition value '{cond_value}' has no semantic match "
                    f"with target node '{target_title}'"
                ),
            })

        return issues

    def _check_skipped_conditional_targets(
        self,
        edges: List[Dict[str, Any]],
        node_map: Dict[str, Dict[str, Any]],
        state_snapshots: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Check for skipped conditional targets that may indicate misrouting.

        Flags skipped branches UNLESS state or node outputs confirm the
        executed branch was the correct choice.
        """
        issues = []

        # Build state by superstep
        state_by_step: Dict[int, Dict[str, Any]] = {}
        for ss in (state_snapshots or []):
            step = ss.get("superstep", -1)
            state_by_step[step] = ss.get("state", {})

        # Group conditional edges by source (router)
        router_edges: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            if edge.get("edge_type") == "conditional":
                router_edges[edge.get("source", "")].append(edge)

        for source, cond_edges in router_edges.items():
            if len(cond_edges) < 2:
                continue

            executed_targets = []
            skipped_targets = []
            for edge in cond_edges:
                target = edge.get("target", "")
                target_node = node_map.get(target, {})
                status = target_node.get("status", "")
                if status in ("succeeded", "completed"):
                    executed_targets.append(edge)
                elif status == "skipped":
                    skipped_targets.append(edge)

            if not (executed_targets and skipped_targets):
                continue

            # Get state at the router's superstep
            source_node = node_map.get(source, {})
            source_step = source_node.get("superstep", -1)
            state = state_by_step.get(source_step, {})
            if not state:
                state = state_by_step.get(source_step + 1, {})

            # Check if state or source outputs confirm the executed branch
            executed_edge = executed_targets[0]
            executed_cond = executed_edge.get("condition", "")
            executed_match = re.match(
                r"(\w+)\s*==\s*['\"]([^'\"]+)['\"]", executed_cond
            )

            routing_confirmed = False
            if executed_match:
                key = executed_match.group(1)
                expected_val = executed_match.group(2)

                # Check state snapshots
                if state:
                    actual_val = str(state.get(key, ""))
                    if actual_val == expected_val:
                        routing_confirmed = True

                # Also check source node outputs
                if not routing_confirmed:
                    source_outputs = source_node.get("outputs", {})
                    if key in source_outputs:
                        if str(source_outputs[key]) == expected_val:
                            routing_confirmed = True

                # Check upstream node outputs
                if not routing_confirmed:
                    for nid, node in node_map.items():
                        if node.get("superstep", 99) <= source_step:
                            out = node.get("outputs", {})
                            if key in out and str(out[key]) == expected_val:
                                routing_confirmed = True
                                break

            if routing_confirmed:
                continue  # Correct routing confirmed

            # State/outputs don't confirm — flag as suspicious
            for skipped_edge in skipped_targets:
                issues.append({
                    "type": "skipped_conditional",
                    "source": source,
                    "skipped_target": skipped_edge.get("target", ""),
                    "skipped_condition": skipped_edge.get("condition", ""),
                    "executed_target": executed_edge.get("target", ""),
                    "executed_condition": executed_cond,
                    "description": (
                        f"Router '{source}' skipped target "
                        f"'{skipped_edge.get('target', '')}' "
                        f"(condition: {skipped_edge.get('condition', '')}) "
                        f"while executing '{executed_edge.get('target', '')}'"
                    ),
                })

        return issues
