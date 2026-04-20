"""
F11: Coordination Failure Detector
==================================

Analyzes multi-agent conversations for:
1. Conflicting actions - agents doing contradictory things
2. Redundant work - multiple agents doing the same task
3. Missed handoffs - expected agent input that never comes
4. Role confusion - agents taking on wrong responsibilities
5. Inconsistent state - agents having different views of progress
"""

import logging
from typing import List, Optional, Dict, Any

from ._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)

logger = logging.getLogger(__name__)


class TurnAwareCoordinationFailureDetector(TurnAwareDetector):
    """Detects F11: Coordination Failure across conversation turns.

    Analyzes multi-agent conversations for:
    1. Conflicting actions - agents doing contradictory things
    2. Redundant work - multiple agents doing the same task
    3. Missed handoffs - expected agent input that never comes
    4. Role confusion - agents taking on wrong responsibilities
    5. Inconsistent state - agents having different views of progress

    This is the most common failure mode in MAST (40% prevalence).
    """

    name = "TurnAwareCoordinationFailureDetector"
    version = "1.0"
    supported_failure_modes = ["F11"]

    # Conflict indicators - stronger signals of inter-agent disagreement
    # Tuned to reduce FP from common words like "however", "error"
    CONFLICT_INDICATORS = [
        "i disagree", "that's wrong", "you made a mistake",
        "let me correct", "that's not correct", "should not have",
        "you shouldn't", "incorrect approach", "wrong approach",
        "redo this", "start over", "conflicting with", "contradicts",
        "not what i asked", "misunderstood", "that's incorrect",
    ]

    # Redundancy indicators - signs of duplicate work
    REDUNDANCY_INDICATORS = [
        "already done", "already completed", "duplicate", "same as",
        "just did that", "already implemented", "implemented earlier",
        "was done by", "redundant", "again?", "repeated",
    ]

    # Handoff phrases - expecting input from others
    HANDOFF_PHRASES = [
        "waiting for", "need input from", "once you", "after you",
        "please provide", "send me", "pass to", "hand off",
        "your turn", "over to you", "expecting", "depends on",
    ]

    def __init__(
        self,
        min_agents: int = 2,
        conflict_threshold: float = 0.1,
        redundancy_threshold: float = 0.6,  # Raised from 0.3 - avoid FPs on similar discussions
        min_issues_to_flag: int = 2,  # Balanced: was 5 (too strict), now 2
    ):
        self.min_agents = min_agents
        self.conflict_threshold = conflict_threshold
        self.redundancy_threshold = redundancy_threshold
        self.min_issues_to_flag = min_issues_to_flag

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect coordination failures in multi-agent conversation."""
        # Extract unique agents
        agents = set()
        for turn in turns:
            if turn.participant_type == "agent":
                agents.add(turn.participant_id)

        # Need multiple agents for coordination failure
        if len(agents) < self.min_agents:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Single agent conversation - no coordination needed",
                detector_name=self.name,
            )

        agent_turns = [t for t in turns if t.participant_type == "agent"]
        if len(agent_turns) < 3:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few agent turns to detect coordination issues",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # 1. Check for conflicting statements
        conflict_issues = self._detect_conflicts(agent_turns)
        issues.extend(conflict_issues)
        for issue in conflict_issues:
            affected_turns.extend(issue.get("turns", []))

        # 2. Check for redundant work
        redundancy_issues = self._detect_redundancy(agent_turns)
        issues.extend(redundancy_issues)
        for issue in redundancy_issues:
            affected_turns.extend(issue.get("turns", []))

        # 3. Check for missed handoffs
        handoff_issues = self._detect_missed_handoffs(agent_turns, agents)
        issues.extend(handoff_issues)
        for issue in handoff_issues:
            affected_turns.extend(issue.get("turns", []))

        # 4. Check for role confusion (agent doing work outside its role)
        role_issues = self._detect_role_confusion(agent_turns, agents)
        issues.extend(role_issues)
        for issue in role_issues:
            affected_turns.extend(issue.get("turns", []))

        # Require multiple issues to flag coordination failure (avoid FPs)
        if len(issues) < self.min_issues_to_flag:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation=f"Insufficient evidence ({len(issues)} issue(s), need {self.min_issues_to_flag}+)",
                detector_name=self.name,
            )

        # Determine severity based on number and type of issues
        if len(issues) >= 4 or any(i["type"] == "conflict" for i in issues):
            severity = TurnAwareSeverity.SEVERE
        elif len(issues) >= 3:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.9, 0.5 + len(issues) * 0.15)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F11",
            explanation=f"Coordination failure detected: {len(issues)} issues found across {len(agents)} agents",
            affected_turns=list(set(affected_turns)),
            evidence={
                "issues": issues,
                "num_agents": len(agents),
                "agents": list(agents),
            },
            suggested_fix=(
                "Improve agent coordination by: 1) Adding explicit handoff protocols, "
                "2) Implementing shared state management, 3) Adding coordination checkpoints, "
                "4) Defining clear role boundaries."
            ),
            detector_name=self.name,
        )

    def _detect_conflicts(self, agent_turns: List[TurnSnapshot]) -> List[Dict]:
        """Detect conflicting statements between agents."""
        issues = []
        for i, turn in enumerate(agent_turns):
            content_lower = turn.content.lower()
            for indicator in self.CONFLICT_INDICATORS:
                if indicator in content_lower:
                    # Check if this conflicts with a previous turn from different agent
                    for j in range(max(0, i - 3), i):
                        prev_turn = agent_turns[j]
                        if prev_turn.participant_id != turn.participant_id:
                            issues.append({
                                "type": "conflict",
                                "turns": [prev_turn.turn_number, turn.turn_number],
                                "indicator": indicator,
                                "agents": [prev_turn.participant_id, turn.participant_id],
                                "description": f"Potential conflict: agent uses '{indicator}'",
                            })
                            break
                    break  # Only one issue per turn
        return issues[:3]  # Limit to first 3

    def _detect_redundancy(self, agent_turns: List[TurnSnapshot]) -> List[Dict]:
        """Detect redundant work between agents."""
        issues = []

        # Check for explicit redundancy indicators
        for turn in agent_turns:
            content_lower = turn.content.lower()
            for indicator in self.REDUNDANCY_INDICATORS:
                if indicator in content_lower:
                    issues.append({
                        "type": "redundancy",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Redundancy indicator: '{indicator}'",
                    })
                    break

        # Check for similar content from different agents
        for i, turn1 in enumerate(agent_turns):
            for j, turn2 in enumerate(agent_turns[i + 1:], i + 1):
                if turn1.participant_id != turn2.participant_id:
                    similarity = self._content_similarity(turn1.content, turn2.content)
                    if similarity > self.redundancy_threshold:
                        issues.append({
                            "type": "redundant_work",
                            "turns": [turn1.turn_number, turn2.turn_number],
                            "similarity": similarity,
                            "agents": [turn1.participant_id, turn2.participant_id],
                            "description": f"Similar work by different agents ({similarity:.0%} overlap)",
                        })
                        if len(issues) >= 3:
                            return issues

        return issues[:3]

    def _detect_missed_handoffs(
        self, agent_turns: List[TurnSnapshot], agents: set
    ) -> List[Dict]:
        """Detect missed handoffs where expected input never comes."""
        issues = []

        for i, turn in enumerate(agent_turns):
            content_lower = turn.content.lower()
            for phrase in self.HANDOFF_PHRASES:
                if phrase in content_lower:
                    # Check if subsequent turns address the handoff
                    handoff_addressed = False
                    for j in range(i + 1, min(i + 5, len(agent_turns))):
                        next_turn = agent_turns[j]
                        if next_turn.participant_id != turn.participant_id:
                            handoff_addressed = True
                            break

                    if not handoff_addressed:
                        issues.append({
                            "type": "missed_handoff",
                            "turns": [turn.turn_number],
                            "phrase": phrase,
                            "description": f"Handoff expected ('{phrase}') but not addressed",
                        })
                        break

        return issues[:2]

    def _detect_role_confusion(
        self, agent_turns: List[TurnSnapshot], agents: set
    ) -> List[Dict]:
        """Detect role confusion where agents step outside their roles."""
        # Extract role names from agent IDs
        role_keywords = {}
        for agent in agents:
            # Parse role from IDs like "chatdev:CEO" or "metagpt:Architect"
            if ":" in agent:
                role = agent.split(":")[-1].lower()
                role_keywords[agent] = role

        if len(role_keywords) < 2:
            return []

        issues = []

        # Check for role-inconsistent actions
        for turn in agent_turns:
            agent_role = role_keywords.get(turn.participant_id, "")
            content_lower = turn.content.lower()

            # CEO/Manager shouldn't write code
            if agent_role in ["ceo", "manager", "productmanager", "pm"]:
                if "def " in content_lower or "class " in content_lower or "```python" in content_lower:
                    issues.append({
                        "type": "role_confusion",
                        "turns": [turn.turn_number],
                        "agent": turn.participant_id,
                        "expected_role": agent_role,
                        "description": f"Manager/CEO agent writing code (role confusion)",
                    })

            # Coder/Programmer shouldn't be making product decisions
            if agent_role in ["programmer", "coder", "developer", "engineer"]:
                if any(phrase in content_lower for phrase in [
                    "product decision", "user story", "requirement is",
                    "we should pivot", "market analysis"
                ]):
                    issues.append({
                        "type": "role_confusion",
                        "turns": [turn.turn_number],
                        "agent": turn.participant_id,
                        "expected_role": agent_role,
                        "description": f"Developer making product decisions (role confusion)",
                    })

        return issues[:2]

    def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate simple word overlap similarity."""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0
