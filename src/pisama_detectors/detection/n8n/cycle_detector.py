"""
F11: Graph Cycle Detection for n8n Workflows
=============================================

Detects circular execution patterns in n8n workflows:
- Same node executing multiple times in sequence
- Circular delegation patterns (A->B->C->A)
- Retry storms that indicate stuck workflows
- Ping-pong patterns between nodes
- Semantic loops where normalized node names repeat

This is n8n-specific because workflow execution order is deterministic
based on the graph structure, unlike conversational agent turn-taking.
"""

import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set, Tuple

from pisama_detectors.detection.turn_aware._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)

logger = logging.getLogger(__name__)

# Benign patterns that indicate normal retry/pagination behavior (not a bug)
BENIGN_LOOP_PATTERNS: Set[str] = {
    "retry", "retrying", "attempt", "page", "pagination", "offset",
    "cursor", "next", "previous", "batch", "chunk", "poll", "polling",
    "heartbeat", "ping", "healthcheck", "status", "progress",
    "paginate", "iterate", "load items", "next page", "fetch page",
    "scroll", "enumerate", "for each", "foreach", "loop over",
    "split in batches", "splitinbatches", "wait", "delay", "schedule",
    "cron", "interval", "webhook response", "respond to webhook",
}


def content_similarity(a: str, b: str) -> float:
    """Calculate similarity ratio between two strings (0.0 to 1.0)."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def normalize_node_name(name: str) -> str:
    """Normalize node name by stripping trailing numbers and action suffixes.

    This allows detecting semantic loops where nodes like:
        "Agent Answer 1", "Agent Answer 2", "Agent Answer 3"
    are recognized as the same logical node "Agent".

    Examples:
        "Agent Answer 1" -> "Agent"
        "User Repeat 2" -> "User"
        "L1 Support Round 2" -> "L1 Support"
        "Alex Proposes" -> "Alex"
        "Seller Offer" -> "Seller"
    """
    normalized = name.strip()

    # Remove "Round X" suffix
    normalized = re.sub(r'\s+Round\s+\d+$', '', normalized, flags=re.IGNORECASE)

    # Remove trailing numbers with optional separator
    normalized = re.sub(r'[\s_\-]*\(?[0-9]+\)?$', '', normalized)

    # Remove trailing version indicators
    normalized = re.sub(r'[\s_\-]*v[0-9]+$', '', normalized, flags=re.IGNORECASE)

    # Remove common action suffixes (order matters - longer patterns first)
    action_patterns = [
        r'\s+Returns\s+to\s+\w+$',  # "Returns to React"
        r'\s+Wavers$', r'\s+Proposes$', r'\s+Reconsiders$', r'\s+Agrees$',
        r'\s+Confirms$', r'\s+Finalizes$', r'\s+Decides$', r'\s+Concludes$',
        r'\s+Offer$', r'\s+Counter$', r'\s+Response$', r'\s+Answer$',
        r'\s+Reply$', r'\s+Repeat$', r'\s+Escalates$',
        r'\s+Round$', r'\s+Turn$', r'\s+Step$', r'\s+Phase$', r'\s+Iteration$',
        r'\s+Initial$', r'\s+Contaminated$', r'\s+Updated$', r'\s+Modified$',
    ]
    for pattern in action_patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)

    return normalized.strip()


class N8NCycleDetector(TurnAwareDetector):
    """Detects F11: Coordination Failure / Workflow Cycles in n8n workflows.

    Analyzes workflow execution for:
    1. Repeated node sequences indicating loops
    2. Circular delegation patterns
    3. Retry/error cycles
    4. Parallel branch conflicts
    5. Semantic loops (normalized node names repeat)

    n8n-specific manifestation of F11 (Coordination Failure):
    In conversational agents, this is about delegation loops between agents.
    In n8n workflows, this is about graph cycles and execution loops.
    """

    name = "N8NCycleDetector"
    version = "1.1"
    supported_failure_modes = ["F11"]

    def __init__(
        self,
        min_cycle_repetitions: int = 4,  # Increased from 3 to reduce FP
        max_healthy_retries: int = 4,
        content_similarity_threshold: float = 0.75,  # Raised from 0.6 to reduce FPs — require stronger evidence
    ):
        """Initialize cycle detector.

        Args:
            min_cycle_repetitions: Minimum times a pattern must repeat to be a cycle
            max_healthy_retries: Maximum retries before flagging as problematic
            content_similarity_threshold: Min content similarity to confirm real loop
        """
        self.min_cycle_repetitions = min_cycle_repetitions
        self.max_healthy_retries = max_healthy_retries
        self.content_similarity_threshold = content_similarity_threshold

    def _is_benign_loop(self, turns: List[TurnSnapshot]) -> bool:
        """Check if the detected loop is actually benign behavior.

        Benign loops include:
        - Pagination (fetching pages of data)
        - Polling (checking status repeatedly)
        - Retry logic (automatic retries on failure)
        - Pipeline stages (same operation on different data)
        """
        for turn in turns:
            content_lower = turn.content.lower()
            node_lower = turn.participant_id.lower()
            node_type = turn.turn_metadata.get("node_type", "").lower()

            # Check if content, node name, or node type suggests benign pattern
            for pattern in BENIGN_LOOP_PATTERNS:
                if pattern in content_lower or pattern in node_lower or pattern in node_type:
                    return True

            # n8n built-in loop nodes are always benign
            if "splitinbatches" in node_type or "loopoveritems" in node_type:
                return True

        # Check for pipeline pattern: monotonically increasing execution times
        # indicate sequential processing, not a stuck loop
        timestamps = []
        for turn in turns:
            ts = turn.turn_metadata.get("execution_time") or turn.turn_metadata.get("startedAt")
            if ts is not None:
                try:
                    timestamps.append(float(ts) if isinstance(ts, (int, float)) else 0)
                except (ValueError, TypeError):
                    pass

        if len(timestamps) >= 4:
            increasing = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
            if increasing and timestamps[-1] > timestamps[0]:
                return True

        return False

    def _verify_content_similarity(
        self, turns: List[TurnSnapshot], cycle_length: int
    ) -> Tuple[bool, float]:
        """Verify that loop iterations have similar content (not just same node names).

        Returns:
            Tuple of (is_similar, avg_similarity)
        """
        if len(turns) < cycle_length * 2:
            return False, 0.0

        similarities = []
        for i in range(cycle_length, len(turns)):
            prev_content = turns[i - cycle_length].content
            curr_content = turns[i].content
            sim = content_similarity(prev_content, curr_content)
            similarities.append(sim)

        if not similarities:
            return False, 0.0

        avg_sim = sum(similarities) / len(similarities)
        return avg_sim >= self.content_similarity_threshold, avg_sim

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect graph cycles and coordination failures in n8n workflow."""
        if len(turns) < 4:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Need at least 4 nodes to detect cycles",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # 1. Detect semantic loops (normalized node names)
        semantic = self._detect_semantic_loop(turns)
        if semantic["detected"]:
            issues.append(semantic)
            affected_turns.extend(semantic.get("turns", []))

        # 2. Detect exact node sequence repetition
        sequence_cycle = self._detect_sequence_cycle(turns)
        if sequence_cycle["detected"]:
            issues.append(sequence_cycle)
            affected_turns.extend(sequence_cycle.get("turns", []))

        # 3. Detect circular delegation (A->B->C->A)
        circular = self._detect_circular_delegation(turns)
        if circular["detected"]:
            issues.append(circular)
            affected_turns.extend(circular.get("turns", []))

        # 4. Detect ping-pong pattern (A->B->A->B)
        pingpong = self._detect_pingpong(turns)
        if pingpong["detected"]:
            issues.append(pingpong)
            affected_turns.extend(pingpong.get("turns", []))

        # 5. Detect retry storms
        retry = self._detect_retry_storm(turns)
        if retry["detected"]:
            issues.append(retry)
            affected_turns.extend(retry.get("turns", []))

        # 6. Detect node over-execution
        over_exec = self._detect_node_overexecution(turns)
        if over_exec["detected"]:
            issues.append(over_exec)
            affected_turns.extend(over_exec.get("turns", []))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="No cycle patterns detected in workflow execution",
                detector_name=self.name,
            )

        # Filter out benign loops (pagination, polling, retries)
        if self._is_benign_loop(turns):
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.80,
                failure_mode=None,
                explanation="Loop pattern detected but appears to be benign (retry/pagination/polling)",
                detector_name=self.name,
            )

        # Verify content similarity for the strongest issue
        strongest_issue = max(issues, key=lambda i: i.get("repetitions", 0))
        cycle_length = strongest_issue.get("cycle_length", 2)
        is_similar, avg_similarity = self._verify_content_similarity(turns, cycle_length)

        # If content isn't similar, this might be a false positive
        if not is_similar and avg_similarity < 0.2:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.75,
                failure_mode=None,
                explanation=f"Loop pattern detected but content differs between iterations (similarity: {avg_similarity:.2f})",
                detector_name=self.name,
            )

        # Determine severity based on cycle characteristics
        has_infinite = any(i.get("potentially_infinite", False) for i in issues)
        cycle_lengths = [i.get("cycle_length", 0) for i in issues]
        max_repetitions = max((i.get("repetitions", 0) for i in issues), default=0)

        if has_infinite or max_repetitions >= 5:
            severity = TurnAwareSeverity.SEVERE
        elif max_repetitions >= 3 or any(l >= 4 for l in cycle_lengths):
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        # Adjust confidence based on content similarity
        base_confidence = min(0.95, 0.6 + len(issues) * 0.15)
        confidence = base_confidence * (0.5 + 0.5 * avg_similarity)  # Scale by similarity

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F11",
            explanation=f"Workflow cycle detected: {len(issues)} patterns found (content similarity: {avg_similarity:.2f})",
            affected_turns=list(set(affected_turns)),
            evidence={
                "issues": issues,
                "total_nodes": len(turns),
                "unique_nodes": len(set(t.participant_id for t in turns)),
                "content_similarity": avg_similarity,
            },
            suggested_fix=(
                "Add cycle-breaking conditions to prevent infinite loops. "
                "Use n8n's IF node to check exit conditions. "
                "Add maximum iteration limits or use the Loop Over Items node with bounds."
            ),
            detector_name=self.name,
        )

    def detect_workflow(self, workflow_json: Dict[str, Any]) -> TurnAwareDetectionResult:
        """Detect graph cycles by analyzing raw n8n workflow JSON directly.

        Builds an adjacency graph from workflow connections and runs DFS-based
        cycle detection. Checks for explicit loop-break conditions (IF nodes with
        exit paths) and flags cycles without break conditions as potential infinite
        loops. Also detects self-loops.

        Args:
            workflow_json: Raw n8n workflow JSON with "nodes" and "connections" keys.

        Returns:
            TurnAwareDetectionResult with detected cycle issues.
        """
        nodes = workflow_json.get("nodes", [])
        connections = workflow_json.get("connections", {})

        if not nodes:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No nodes found in workflow JSON",
                detector_name=self.name,
            )

        # Build node lookup
        node_lookup: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            node_name = node.get("name", "")
            if node_name:
                node_lookup[node_name] = node

        # Build adjacency graph from connections
        adjacency: Dict[str, List[str]] = {
            node.get("name", ""): [] for node in nodes if node.get("name")
        }

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

        issues: List[Dict[str, Any]] = []
        affected_node_names: List[str] = []

        # 1. Detect self-loops (node connected to itself)
        for node_name, neighbors in adjacency.items():
            if node_name in neighbors:
                node_type = node_lookup.get(node_name, {}).get("type", "")
                issues.append({
                    "type": "self_loop",
                    "node": node_name,
                    "node_type": node_type,
                    "has_break_condition": False,
                    "potentially_infinite": True,
                    "description": f"Self-loop: node '{node_name}' is connected to itself",
                })
                affected_node_names.append(node_name)

        # 2. DFS-based cycle detection
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        cycles_found: List[List[str]] = []

        def dfs_find_cycles(node: str, path: List[str]) -> None:
            """DFS traversal to find all cycles in the graph."""
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in adjacency.get(node, []):
                if neighbor == node:
                    # Self-loop already handled above
                    continue
                if neighbor not in visited:
                    dfs_find_cycles(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle: extract the cycle from the path
                    cycle_start_idx = path.index(neighbor)
                    cycle = path[cycle_start_idx:] + [neighbor]
                    cycles_found.append(cycle)

            path.pop()
            rec_stack.discard(node)

        for node_name in adjacency:
            if node_name not in visited:
                dfs_find_cycles(node_name, [])

        # 3. Analyze each detected cycle
        for cycle in cycles_found:
            cycle_nodes = cycle[:-1]  # Remove the repeated closing node
            has_break_condition = self._cycle_has_break_condition(
                cycle_nodes, node_lookup, adjacency
            )

            potentially_infinite = not has_break_condition

            issues.append({
                "type": "graph_cycle",
                "cycle": cycle,
                "cycle_length": len(cycle_nodes),
                "has_break_condition": has_break_condition,
                "potentially_infinite": potentially_infinite,
                "description": (
                    f"Cycle detected: {' -> '.join(cycle)}"
                    + (" (no break condition found)" if potentially_infinite else " (has break condition)")
                ),
            })
            affected_node_names.extend(cycle_nodes)

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="No cycle patterns detected in workflow graph",
                detector_name=self.name,
            )

        # Determine severity
        infinite_count = sum(1 for i in issues if i.get("potentially_infinite", False))
        self_loop_count = sum(1 for i in issues if i.get("type") == "self_loop")
        cycle_count = sum(1 for i in issues if i.get("type") == "graph_cycle")

        if infinite_count >= 2 or self_loop_count >= 1:
            severity = TurnAwareSeverity.SEVERE
        elif infinite_count >= 1 or cycle_count >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.95, 0.7 + len(issues) * 0.1)

        # Map affected node names to indices
        node_index_map = {node.get("name", ""): idx for idx, node in enumerate(nodes)}
        affected_turns = sorted({
            node_index_map[name]
            for name in set(affected_node_names)
            if name in node_index_map
        })

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F11",
            explanation=f"Workflow cycle detected: {len(issues)} cycle patterns found in graph structure",
            affected_turns=affected_turns,
            evidence={
                "issues": issues,
                "total_nodes": len(nodes),
                "total_cycles": cycle_count,
                "total_self_loops": self_loop_count,
                "cycles_without_break": infinite_count,
            },
            suggested_fix=(
                "Add cycle-breaking conditions to prevent infinite loops. "
                "Use IF nodes with explicit exit conditions inside cycles. "
                "Add maximum iteration limits or use the Loop Over Items node with bounds."
            ),
            detector_name=self.name,
        )

    def _cycle_has_break_condition(
        self,
        cycle_nodes: List[str],
        node_lookup: Dict[str, Dict[str, Any]],
        adjacency: Dict[str, List[str]],
    ) -> bool:
        """Check if a detected cycle has an explicit loop-break condition.

        A break condition exists if there is an IF node in the cycle that has
        at least one output branch leading outside the cycle (an exit path).

        Args:
            cycle_nodes: List of node names forming the cycle.
            node_lookup: Dict mapping node name to node definition.
            adjacency: Adjacency list for the graph.

        Returns:
            True if the cycle has a break condition, False otherwise.
        """
        cycle_set = set(cycle_nodes)

        for node_name in cycle_nodes:
            node_data = node_lookup.get(node_name, {})
            node_type = node_data.get("type", "")

            # IF and Switch nodes can serve as break conditions
            if node_type in ("n8n-nodes-base.if", "n8n-nodes-base.switch"):
                # Check if any output branch leads outside the cycle
                neighbors = adjacency.get(node_name, [])
                for neighbor in neighbors:
                    if neighbor not in cycle_set:
                        # This IF/Switch has a branch that exits the cycle
                        return True

        return False

    def _detect_semantic_loop(self, turns: List[TurnSnapshot]) -> Dict[str, Any]:
        """Detect semantic loops where normalized node names repeat in a pattern.

        This catches patterns like:
            "Agent Answer 1" -> "User Repeat 1" -> "Agent Answer 2" -> "User Repeat 2"

        Where the normalized pattern is:
            "Agent" -> "User" -> "Agent" -> "User"
        """
        original_sequence = [t.participant_id for t in turns]
        normalized_sequence = [normalize_node_name(t.participant_id) for t in turns]
        n = len(normalized_sequence)

        if n < 4:
            return {"detected": False}

        # If all node executions have different output content, it's NOT a loop —
        # same node names but different work being done each time.
        contents = [t.content.strip() for t in turns if t.content.strip()]
        if contents and len(set(contents)) == len(contents):
            return {"detected": False}

        # Skip initial trigger/setup nodes
        start_idx = 0
        for i, name in enumerate(normalized_sequence):
            if 'trigger' in name.lower() or 'webhook' in name.lower() or 'initialize' in name.lower():
                start_idx = i + 1
            else:
                break

        working_norm = normalized_sequence[start_idx:]
        working_orig = original_sequence[start_idx:]
        working_turns = turns[start_idx:]
        n = len(working_norm)

        if n < 4:
            return {"detected": False}

        # Check for cycle lengths from 2 to n/2 using normalized names
        for cycle_len in range(2, min(n // 2 + 1, 6)):
            repetitions = 0
            start_pattern = working_norm[:cycle_len]

            # Skip if pattern contains all same nodes
            unique_nodes = set(start_pattern)
            if len(unique_nodes) < 2:
                continue

            for i in range(0, n - cycle_len + 1, cycle_len):
                if working_norm[i:i + cycle_len] == start_pattern:
                    repetitions += 1
                else:
                    break

            if repetitions >= self.min_cycle_repetitions:
                affected_indices = list(range(repetitions * cycle_len))
                return {
                    "detected": True,
                    "type": "semantic_loop",
                    "cycle_length": cycle_len,
                    "repetitions": repetitions,
                    "normalized_pattern": start_pattern,
                    "original_nodes": [working_orig[i] for i in affected_indices],
                    "turns": [working_turns[i].turn_number for i in affected_indices],
                    "potentially_infinite": repetitions >= 3,
                    "description": f"Semantic loop: {start_pattern} repeated {repetitions} times",
                }

        # Check for alternating pattern (A->B->A->B) with normalized names
        for i in range(n - 3):
            a, b = working_norm[i], working_norm[i + 1]
            if a != b:  # Must be different nodes, allow LM patterns
                count = 1
                j = i + 2
                while j + 1 < n:
                    if working_norm[j] == a and working_norm[j + 1] == b:
                        count += 1
                        j += 2
                    else:
                        break

                if count >= self.min_cycle_repetitions:
                    affected_indices = list(range(i, i + count * 2))
                    return {
                        "detected": True,
                        "type": "semantic_loop",
                        "cycle_length": 2,
                        "repetitions": count,
                        "normalized_pattern": [a, b],
                        "original_nodes": [working_orig[idx] for idx in affected_indices],
                        "turns": [working_turns[idx].turn_number for idx in affected_indices],
                        "potentially_infinite": count >= 3,
                        "description": f"Semantic alternating: {a} <-> {b} ({count} times)",
                    }

        return {"detected": False}

    def _detect_sequence_cycle(self, turns: List[TurnSnapshot]) -> Dict[str, Any]:
        """Detect exact sequence repetition in node execution.

        Example: [A, B, C, A, B, C, A, B, C] -> cycle of [A, B, C]
        """
        node_sequence = [t.participant_id for t in turns]
        n = len(node_sequence)

        # Check for cycle lengths from 2 to n/2
        for cycle_len in range(2, n // 2 + 1):
            repetitions = 0
            start_pattern = node_sequence[:cycle_len]

            for i in range(0, n - cycle_len + 1, cycle_len):
                if node_sequence[i:i + cycle_len] == start_pattern:
                    repetitions += 1
                else:
                    break

            if repetitions >= self.min_cycle_repetitions:
                return {
                    "detected": True,
                    "type": "sequence_cycle",
                    "cycle_length": cycle_len,
                    "repetitions": repetitions,
                    "pattern": start_pattern,
                    "turns": [turns[i].turn_number for i in range(repetitions * cycle_len)],
                    "potentially_infinite": repetitions >= 4,
                    "description": f"Sequence cycle: {start_pattern} repeated {repetitions} times",
                }

        return {"detected": False}

    def _detect_circular_delegation(self, turns: List[TurnSnapshot]) -> Dict[str, Any]:
        """Detect circular delegation patterns using normalized names.

        Catches patterns like L1 -> L2 -> L3 -> L1 even when full node names
        are "Get L1 Support", "LM L1", "L1 Support", etc.
        """
        # Use normalized names for detection
        normalized_sequence = [normalize_node_name(t.participant_id) for t in turns]

        # Skip utility nodes
        skip_patterns = ['trigger', 'webhook', 'initialize', 'send to pisama', 'get']
        working_idx = []
        working_names = []
        for i, name in enumerate(normalized_sequence):
            if not any(p in name.lower() for p in skip_patterns) and 'lm' not in name.lower():
                working_idx.append(i)
                working_names.append(name)

        if len(working_names) < 3:
            return {"detected": False}

        # Find nodes that appear multiple times
        node_positions: Dict[str, List[int]] = {}
        for i, name in enumerate(working_names):
            if name not in node_positions:
                node_positions[name] = []
            node_positions[name].append(i)

        # Check if any node appears at regular intervals
        for node, positions in node_positions.items():
            if len(positions) >= 2:  # Allow 2+ appearances
                intervals = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]

                # Accept if intervals are somewhat consistent (allow up to 2 unique values)
                # and average interval is at least 2
                if intervals:
                    avg_interval = sum(intervals) / len(intervals)
                    interval_variance = len(set(intervals))

                    # Accept if: avg >= 2 AND (all same OR up to 2 different values with >=3 appearances)
                    if avg_interval >= 2 and (interval_variance <= 1 or (interval_variance <= 2 and len(positions) >= 3)):
                        cycle_length = int(round(avg_interval))
                        original_turns = [working_idx[p] for p in positions]

                        return {
                            "detected": True,
                            "type": "circular_delegation",
                            "cycle_length": cycle_length,
                            "repetitions": len(positions),
                            "anchor_node": node,
                            "turns": [turns[t].turn_number for t in original_turns],
                            "potentially_infinite": len(positions) >= 3,
                            "description": f"Circular delegation: {node} appears {len(positions)} times (avg interval {avg_interval:.1f})",
                        }

        return {"detected": False}

    def _detect_pingpong(self, turns: List[TurnSnapshot]) -> Dict[str, Any]:
        """Detect ping-pong pattern between two nodes (A->B->A->B)."""
        if len(turns) < 4:
            return {"detected": False}

        node_sequence = [t.participant_id for t in turns]

        for i in range(len(node_sequence) - 3):
            a, b = node_sequence[i], node_sequence[i + 1]
            if a != b:
                count = 1
                j = i + 2
                while j + 1 < len(node_sequence):
                    if node_sequence[j] == a and node_sequence[j + 1] == b:
                        count += 1
                        j += 2
                    else:
                        break

                if count >= self.min_cycle_repetitions:
                    affected = [turns[k].turn_number for k in range(i, min(i + count * 2, len(turns)))]
                    return {
                        "detected": True,
                        "type": "pingpong",
                        "cycle_length": 2,
                        "repetitions": count,
                        "nodes": [a, b],
                        "turns": affected,
                        "potentially_infinite": count >= 4,
                        "description": f"Ping-pong: {a} <-> {b} ({count} times)",
                    }

        return {"detected": False}

    def _detect_retry_storm(self, turns: List[TurnSnapshot]) -> Dict[str, Any]:
        """Detect retry storms where a node executes repeatedly."""
        consecutive_counts: List[Tuple[str, int, int]] = []
        current_node = None
        current_count = 0
        current_start = 0

        for i, turn in enumerate(turns):
            if turn.participant_id == current_node:
                current_count += 1
            else:
                if current_count > 1:
                    consecutive_counts.append((current_node, current_count, current_start))
                current_node = turn.participant_id
                current_count = 1
                current_start = i

        if current_count > 1:
            consecutive_counts.append((current_node, current_count, current_start))

        for node, count, start in consecutive_counts:
            if count > self.max_healthy_retries:
                has_errors = any(
                    any(err in turns[j].content.lower() for err in ['error', 'fail', 'retry', 'exception'])
                    for j in range(start, min(start + count, len(turns)))
                )

                affected = [turns[j].turn_number for j in range(start, min(start + count, len(turns)))]
                return {
                    "detected": True,
                    "type": "retry_storm",
                    "node": node,
                    "repetitions": count,
                    "has_error_indicators": has_errors,
                    "turns": affected,
                    "potentially_infinite": count >= 6,
                    "description": f"Retry storm: {node} executed {count} consecutive times",
                }

        return {"detected": False}

    def _detect_node_overexecution(self, turns: List[TurnSnapshot]) -> Dict[str, Any]:
        """Detect nodes that execute far more times than expected."""
        node_counts = Counter(t.participant_id for t in turns)
        total_nodes = len(turns)

        threshold = max(5, int(total_nodes * 0.4))

        overexecuted = [(node, count) for node, count in node_counts.items() if count > threshold]

        if overexecuted:
            worst_node, worst_count = max(overexecuted, key=lambda x: x[1])
            node_turns = [t.turn_number for t in turns if t.participant_id == worst_node]

            return {
                "detected": True,
                "type": "node_overexecution",
                "node": worst_node,
                "repetitions": worst_count,
                "total_executions": total_nodes,
                "percentage": worst_count / total_nodes * 100,
                "turns": node_turns,
                "all_overexecuted": overexecuted,
                "potentially_infinite": worst_count >= total_nodes * 0.5,
                "description": f"Overexecution: {worst_node} ran {worst_count}/{total_nodes} times",
            }

        return {"detected": False}
