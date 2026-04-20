"""
F2: Task Decomposition Detector
================================

Detects F2: Poor Task Decomposition in conversations.

Analyzes whether complex tasks are properly broken down into subtasks:
1. Missing decomposition - complex task handled as single step
2. Vague subtasks - non-actionable or unclear steps
3. Missing dependencies - steps out of order or disconnected
4. Over-decomposition - simple task broken into too many steps
5. Circular dependencies - steps that depend on each other

Based on MAST taxonomy for orchestration failures.
"""

import logging
import re
from typing import List, Optional, Dict, Any

from ._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)

logger = logging.getLogger(__name__)


class TurnAwareTaskDecompositionDetector(TurnAwareDetector):
    """Detects F2: Poor Task Decomposition in conversations.

    Analyzes whether complex tasks are properly broken down into subtasks:
    1. Missing decomposition - complex task handled as single step
    2. Vague subtasks - non-actionable or unclear steps
    3. Missing dependencies - steps out of order or disconnected
    4. Over-decomposition - simple task broken into too many steps
    5. Circular dependencies - steps that depend on each other

    Based on MAST taxonomy for orchestration failures.
    """

    name = "TurnAwareTaskDecompositionDetector"
    version = "1.0"
    supported_failure_modes = ["F2"]

    # Indicators of task complexity requiring decomposition
    COMPLEX_TASK_INDICATORS = [
        "system", "platform", "application", "service", "architecture",
        "authentication", "authorization", "database", "migration",
        "refactor", "integration", "infrastructure", "deployment",
        "pipeline", "workflow", "multi-step", "end-to-end",
    ]

    # Indicators that decomposition is happening
    DECOMPOSITION_PATTERNS = [
        r"(?:step|phase|stage)\s*\d+",
        r"\d+[.)]\s+\w+",
        r"(?:first|then|next|finally|lastly)[,:]",
        r"[-•*]\s+\w+",
        r"(?:task|subtask|sub-task)\s*\d*[:.]\s*\w+",
    ]

    # Indicators of vague/non-actionable steps
    VAGUE_INDICATORS = [
        "etc", "various", "miscellaneous", "general", "overall",
        "appropriate", "as needed", "if necessary", "possibly",
        "might", "maybe", "could potentially", "consider",
        "high-level", "broadly", "generally speaking",
        "explore options", "look into", "think about",
    ]

    # Action verbs that make steps actionable
    ACTION_VERBS = [
        "create", "build", "implement", "write", "configure",
        "set up", "install", "deploy", "test", "validate",
        "define", "design", "develop", "add", "remove",
        "update", "modify", "fix", "integrate", "connect",
        "display", "show", "render", "format", "parse",
        "fetch", "load", "save", "store", "delete",
        "call", "invoke", "execute", "run", "process",
    ]

    def __init__(
        self,
        min_steps_for_complex: int = 3,
        max_vague_ratio: float = 0.3,
    ):
        self.min_steps_for_complex = min_steps_for_complex
        self.max_vague_ratio = max_vague_ratio

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect poor task decomposition in conversation."""
        user_turns = [t for t in turns if t.participant_type == "user"]
        agent_turns = [t for t in turns if t.participant_type == "agent"]
        system_turns = [t for t in turns if t.participant_type == "system"]

        # Find the task source (user, system prompt, or first agent)
        task_turns = []
        if user_turns:
            task_turns = user_turns
        elif system_turns and any(len(t.content) > 50 for t in system_turns):
            task_turns = [t for t in system_turns if len(t.content) > 50]
        elif agent_turns:
            task_turns = [agent_turns[0]]

        if not task_turns or not agent_turns:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Need task and agent response for decomposition analysis",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # Check if task is complex
        task_content = " ".join([t.content.lower() for t in task_turns])
        is_complex_task = any(
            ind in task_content for ind in self.COMPLEX_TASK_INDICATORS
        )

        # Analyze agent responses for decomposition
        agent_content = " ".join([t.content for t in agent_turns])
        agent_content_lower = agent_content.lower()

        # Check for decomposition patterns
        has_decomposition = any(
            re.search(pattern, agent_content, re.IGNORECASE | re.MULTILINE)
            for pattern in self.DECOMPOSITION_PATTERNS
        )

        # Extract steps if decomposition exists
        steps = self._extract_steps(agent_content)

        # Issue 1: Complex task without decomposition
        if is_complex_task and not has_decomposition:
            issues.append({
                "type": "missing_decomposition",
                "description": "Complex task handled without proper step breakdown",
            })
            for t in task_turns:
                affected_turns.append(t.turn_number)

        # Issue 2: Check for vague steps
        if steps:
            vague_steps = self._find_vague_steps(steps)
            vague_ratio = len(vague_steps) / len(steps) if steps else 0

            if vague_ratio > self.max_vague_ratio:
                issues.append({
                    "type": "vague_subtasks",
                    "vague_count": len(vague_steps),
                    "total_steps": len(steps),
                    "vague_ratio": vague_ratio,
                    "description": f"{len(vague_steps)}/{len(steps)} steps are vague or non-actionable",
                })

        # Issue 3: Check for missing action verbs
        if steps:
            non_actionable = [
                s for s in steps
                if not any(verb in s.lower() for verb in self.ACTION_VERBS)
            ]
            if len(non_actionable) > len(steps) // 2:
                issues.append({
                    "type": "non_actionable_steps",
                    "count": len(non_actionable),
                    "description": f"{len(non_actionable)}/{len(steps)} steps lack clear action verbs",
                })

        # Issue 4: Complex task with too few steps
        if is_complex_task and has_decomposition and len(steps) < self.min_steps_for_complex:
            issues.append({
                "type": "insufficient_decomposition",
                "steps_found": len(steps),
                "min_required": self.min_steps_for_complex,
                "description": f"Complex task has only {len(steps)} steps (minimum {self.min_steps_for_complex} recommended)",
            })

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.8,
                failure_mode=None,
                explanation=f"Task decomposition appears adequate ({len(steps)} steps found)",
                detector_name=self.name,
            )

        # Determine severity
        if any(i["type"] == "missing_decomposition" for i in issues):
            severity = TurnAwareSeverity.SEVERE
        elif len(issues) >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.85, 0.5 + len(issues) * 0.15)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F2",
            explanation=f"Task decomposition issues: {len(issues)} problems found",
            affected_turns=list(set(affected_turns)),
            evidence={
                "issues": issues,
                "is_complex_task": is_complex_task,
                "steps_found": len(steps),
                "has_decomposition": has_decomposition,
            },
            suggested_fix=(
                "Break down complex tasks into clear, actionable steps. "
                "Each step should have a specific action verb and measurable outcome."
            ),
            detector_name=self.name,
        )

    def _extract_steps(self, content: str) -> List[str]:
        """Extract steps/subtasks from agent content."""
        steps = []

        # Try numbered list first
        numbered = re.findall(r'\d+[.)]\s*([^\n]+)', content)
        if numbered:
            return numbered

        # Try bullet points
        bullets = re.findall(r'[-•*]\s+([^\n]+)', content)
        if bullets:
            return bullets

        # Try step/phase patterns
        step_matches = re.findall(
            r'(?:step|phase|stage)\s*\d*[:.]\s*([^\n]+)',
            content,
            re.IGNORECASE
        )
        if step_matches:
            return step_matches

        return steps

    def _find_vague_steps(self, steps: List[str]) -> List[str]:
        """Find steps that are vague or non-actionable."""
        vague = []
        for step in steps:
            step_lower = step.lower()
            vague_count = sum(1 for ind in self.VAGUE_INDICATORS if ind in step_lower)
            if vague_count >= 1:
                vague.append(step)
            elif len(step.split()) < 3:
                # Very short steps are often vague
                vague.append(step)
        return vague
