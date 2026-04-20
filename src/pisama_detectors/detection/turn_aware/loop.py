"""
F5: Loop / Repetitive Behavior Detection
========================================

Analyzes conversation for:
1. Repetitive agent responses (exact + semantic)
2. Cyclic conversation patterns
3. Coordination loops (A→B→A→B in multi-agent)
4. Stuck loops where same content repeats
"""

import logging
from typing import List, Optional, Dict, Any

from ._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)
from ._embedding_mixin import EmbeddingMixin

logger = logging.getLogger(__name__)


class TurnAwareLoopDetector(EmbeddingMixin, TurnAwareDetector):
    """Detects F5: Infinite Loop / Repetitive Behavior across conversation turns.

    Analyzes conversation for:
    1. Repetitive agent responses (exact + semantic)
    2. Cyclic conversation patterns
    3. Coordination loops (A→B→A→B in multi-agent)
    4. Stuck loops where same content repeats

    Phase 2 Enhancement: Uses semantic similarity to detect loops where content
    is paraphrased but semantically identical (e.g., "let me try again" vs "I'll attempt once more").
    """

    name = "TurnAwareLoopDetector"
    version = "2.0"  # Phase 2: Enhanced with semantic loop detection
    supported_failure_modes = ["F5"]

    def __init__(
        self,
        repetition_threshold: float = 0.9,
        min_repetitions: int = 3,  # Raised from 2 to reduce FPs (18.3% FPR)
    ):
        self.repetition_threshold = repetition_threshold
        self.min_repetitions = min_repetitions

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect loops and repetitive behavior in conversation."""
        if len(turns) < 4:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few turns to detect loops",
                detector_name=self.name,
            )

        # Detect framework from metadata or turn metadata
        framework = ""
        if conversation_metadata:
            framework = conversation_metadata.get("framework", "").lower()
        if not framework and turns and turns[0].turn_metadata:
            framework = turns[0].turn_metadata.get("framework", "").lower()

        # For n8n, all turns are effectively "agent" turns (nodes)
        if "n8n" in framework:
            agent_turns = turns  # All n8n nodes are treated as agents
        else:
            agent_turns = [t for t in turns if t.participant_type == "agent"]

        if len(agent_turns) < 3:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few agent turns to detect loops",
                detector_name=self.name,
            )

        loop_issues = []
        affected_turns = []

        # Check for exact duplicate responses (by hash)
        hash_counts = {}
        for turn in agent_turns:
            h = turn.content_hash
            if h not in hash_counts:
                hash_counts[h] = []
            hash_counts[h].append(turn.turn_number)

        for h, turn_numbers in hash_counts.items():
            if len(turn_numbers) >= self.min_repetitions:
                loop_issues.append({
                    "type": "exact_repetition",
                    "turns": turn_numbers,
                    "repetition_count": len(turn_numbers),
                    "description": f"Exact same response repeated {len(turn_numbers)} times",
                })
                affected_turns.extend(turn_numbers)

        # Check for cyclic patterns (A -> B -> A -> B)
        cyclic_pattern = self._detect_cyclic_pattern(agent_turns)
        if cyclic_pattern["detected"]:
            loop_issues.append({
                "type": "cyclic_pattern",
                "cycle_length": cyclic_pattern["cycle_length"],
                "turns": cyclic_pattern["turns"],
                "description": f"Cyclic pattern detected with length {cyclic_pattern['cycle_length']}",
            })
            affected_turns.extend(cyclic_pattern["turns"])

        # Phase 2: Check for coordination loops (A→B→A→B in multi-agent)
        coord_loop = self._detect_coordination_loop(agent_turns)
        if coord_loop["detected"]:
            loop_issues.append({
                "type": "coordination_loop",
                "agent_sequence": coord_loop["sequence"],
                "turns": coord_loop["turns"],
                "description": f"Coordination loop: agents alternating without progress",
            })
            affected_turns.extend(coord_loop["turns"])

        # Phase 2: Check for semantic loops (paraphrased repetition)
        semantic_loop = self._detect_semantic_loop(agent_turns)
        if semantic_loop["detected"]:
            loop_issues.append({
                "type": "semantic_loop",
                "turns": semantic_loop["turns"],
                "description": f"Semantic loop: {semantic_loop['count']} semantically similar responses",
            })
            affected_turns.extend(semantic_loop["turns"])

        # n8n-specific: Detect repeated node executions (workflow loops)
        if "n8n" in framework:
            n8n_loop = self._detect_n8n_workflow_loop(agent_turns)
            if n8n_loop["detected"]:
                loop_issues.append({
                    "type": "n8n_workflow_loop",
                    "node": n8n_loop["node"],
                    "turns": n8n_loop["turns"],
                    "repetition_count": n8n_loop["count"],
                    "description": f"n8n workflow loop: node '{n8n_loop['node']}' executed {n8n_loop['count']} times",
                })
                affected_turns.extend(n8n_loop["turns"])

        if not loop_issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.9,
                failure_mode=None,
                explanation="No repetitive patterns detected",
                detector_name=self.name,
            )

        # Determine severity
        max_repetitions = max(
            (i.get("repetition_count", 0) for i in loop_issues),
            default=0
        )
        if max_repetitions >= 4 or any(i["type"] == "cyclic_pattern" for i in loop_issues):
            severity = TurnAwareSeverity.SEVERE
        elif max_repetitions >= 3:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.95, 0.6 + max_repetitions * 0.1)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F5",
            explanation=f"Loop/repetition detected: {len(loop_issues)} patterns found",
            affected_turns=list(set(affected_turns)),
            evidence={
                "issues": loop_issues,
                "total_agent_turns": len(agent_turns),
            },
            suggested_fix=(
                "Implement loop breaking logic. Add variation detection to prevent "
                "repeated identical responses. Consider maximum retry limits."
            ),
            detector_name=self.name,
        )

    def _detect_cyclic_pattern(
        self,
        agent_turns: List[TurnSnapshot],
    ) -> Dict[str, Any]:
        """Detect cyclic patterns in agent responses."""
        if len(agent_turns) < 4:
            return {"detected": False}

        hashes = [t.content_hash for t in agent_turns]

        # Check for cycle lengths 2 and 3
        for cycle_len in [2, 3]:
            if len(hashes) >= cycle_len * 2:
                for start in range(len(hashes) - cycle_len * 2 + 1):
                    pattern = hashes[start:start + cycle_len]
                    next_seq = hashes[start + cycle_len:start + cycle_len * 2]
                    if pattern == next_seq:
                        affected_turns = [
                            agent_turns[i].turn_number
                            for i in range(start, min(start + cycle_len * 2, len(agent_turns)))
                        ]
                        return {
                            "detected": True,
                            "cycle_length": cycle_len,
                            "turns": affected_turns,
                        }

        return {"detected": False}

    def _detect_coordination_loop(
        self,
        agent_turns: List[TurnSnapshot],
    ) -> Dict[str, Any]:
        """Detect coordination loops (A→B→A→B pattern) in multi-agent systems.

        Phase 2: Identifies when two or more agents alternate without making progress.
        Common in multi-agent coordination failures.

        Args:
            agent_turns: List of agent turns

        Returns:
            Dict with detection result, agent sequence, and affected turns
        """
        if len(agent_turns) < 4:
            return {"detected": False}

        # Get agent ID sequence
        agent_sequence = [t.participant_id for t in agent_turns]

        # Check for A→B→A→B pattern (2-agent loop)
        for i in range(len(agent_sequence) - 3):
            if (agent_sequence[i] == agent_sequence[i + 2] and
                agent_sequence[i + 1] == agent_sequence[i + 3] and
                agent_sequence[i] != agent_sequence[i + 1]):
                # Found A→B→A→B loop
                return {
                    "detected": True,
                    "sequence": agent_sequence[i:i + 4],
                    "turns": [agent_turns[i + j].turn_number for j in range(4)],
                }

        # Check for A→B→C→A→B→C pattern (3-agent loop)
        for i in range(len(agent_sequence) - 5):
            if (agent_sequence[i] == agent_sequence[i + 3] and
                agent_sequence[i + 1] == agent_sequence[i + 4] and
                agent_sequence[i + 2] == agent_sequence[i + 5] and
                len(set(agent_sequence[i:i + 3])) == 3):
                # Found A→B→C→A→B→C loop
                return {
                    "detected": True,
                    "sequence": agent_sequence[i:i + 6],
                    "turns": [agent_turns[i + j].turn_number for j in range(6)],
                }

        return {"detected": False}

    def _detect_semantic_loop(
        self,
        agent_turns: List[TurnSnapshot],
        similarity_threshold: float = 0.92,
    ) -> Dict[str, Any]:
        """Detect semantic loops using embeddings.

        Phase 2: Identifies when agents repeat semantically similar content even if
        worded differently (e.g., "let me try again" vs "I'll attempt once more").

        Args:
            agent_turns: List of agent turns
            similarity_threshold: Minimum similarity to consider a semantic loop (0.92)

        Returns:
            Dict with detection result, turn numbers, and count
        """
        if not self.embedder or len(agent_turns) < 3:
            return {"detected": False}

        try:
            # Get content from all agent turns (limited to 200 chars for embedding)
            contents = [t.content[:200] for t in agent_turns]

            # Find semantically similar consecutive turns
            similar_pairs = []
            for i in range(len(contents) - 1):
                similarity = self.semantic_similarity(contents[i], contents[i + 1])
                if similarity >= similarity_threshold:
                    similar_pairs.append((i, i + 1, similarity))

            # Check if we have at least 2 consecutive similar pairs (3+ turns)
            if len(similar_pairs) >= 2:
                affected_turn_indices = set()
                for i, j, _ in similar_pairs:
                    affected_turn_indices.add(i)
                    affected_turn_indices.add(j)

                affected_turn_numbers = [agent_turns[i].turn_number for i in sorted(affected_turn_indices)]

                return {
                    "detected": True,
                    "turns": affected_turn_numbers,
                    "count": len(affected_turn_indices),
                }

        except Exception as e:
            logger.debug(f"Semantic loop detection failed: {e}")

        return {"detected": False}

    def _detect_n8n_workflow_loop(
        self,
        turns: List[TurnSnapshot],
    ) -> Dict[str, Any]:
        """Detect n8n workflow loops based on repeated node executions.

        In n8n workflows, loops manifest as:
        1. Same node executing multiple times with similar outputs
        2. Retry patterns where nodes re-execute on failure
        3. Workflow loops where execution cycles back to earlier nodes

        Args:
            turns: List of turns (n8n nodes)

        Returns:
            Dict with detection result, node name, turn numbers, and count
        """
        if len(turns) < 3:
            return {"detected": False}

        # Count node executions by participant_id (node name)
        node_counts: Dict[str, List[int]] = {}
        for turn in turns:
            node_name = turn.participant_id
            if node_name not in node_counts:
                node_counts[node_name] = []
            node_counts[node_name].append(turn.turn_number)

        # Find nodes that execute too many times (potential loop)
        for node_name, turn_numbers in node_counts.items():
            if len(turn_numbers) >= self.min_repetitions:
                # Check if outputs are similar (indicating stuck loop)
                node_turns = [t for t in turns if t.participant_id == node_name]

                # Check for identical content hashes
                hashes = [t.content_hash for t in node_turns]
                unique_hashes = set(hashes)

                # If all executions produce same output, definitely a loop
                if len(unique_hashes) == 1:
                    return {
                        "detected": True,
                        "node": node_name,
                        "turns": turn_numbers,
                        "count": len(turn_numbers),
                    }

                # If >50% of outputs are identical, likely a loop
                max_hash_count = max(hashes.count(h) for h in unique_hashes)
                if max_hash_count >= len(hashes) * 0.5 and max_hash_count >= self.min_repetitions:
                    return {
                        "detected": True,
                        "node": node_name,
                        "turns": turn_numbers,
                        "count": len(turn_numbers),
                    }

        return {"detected": False}
