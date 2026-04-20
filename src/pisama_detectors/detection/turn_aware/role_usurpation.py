"""
F9: Role Usurpation Detector
============================

Analyzes whether agents act outside their designated roles:
1. Role boundary violation - agent performs tasks outside role scope
2. Role ambiguity - unclear or conflicting role assignments
3. Unauthorized actions - agent exceeds permission boundaries
4. Role drift - gradual deviation from assigned responsibilities

Based on MAST research (NeurIPS 2025): FM-2.5 Role Usurpation (3%)
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
from ._embedding_mixin import EmbeddingMixin

logger = logging.getLogger(__name__)


class TurnAwareRoleUsurpationDetector(EmbeddingMixin, TurnAwareDetector):
    """Detects F9: Role Usurpation (FM-2.4) in conversations.

    Analyzes whether agents act outside their designated roles:
    1. Role boundary violation - agent performs tasks outside role scope
    2. Role ambiguity - unclear or conflicting role assignments
    3. Unauthorized actions - agent exceeds permission boundaries
    4. Role drift - gradual deviation from assigned responsibilities

    == IMPORTANT FINDINGS (Jan 2026) ==

    Analysis of MAST benchmark F9 cases revealed:

    1. F9 IS PART OF CASCADE FAILURES
       - 100% of F9+ cases have multiple other failures (F1, F3, F5, F7, F8, F11, F12, F14)
       - F9 rarely appears in isolation; it's part of systemic failures
       - Error rate 2.5x higher in F9+ cases (40% vs 16%)

    2. RULE-BASED DETECTION HAS FUNDAMENTAL LIMITS
       - Current patterns detect 1/40 F9+ cases (2.5% recall)
       - Cascade detection was tested but adds FPs without improving TPs
       - The trajectories don't show obvious role usurpation patterns
       - Most F9+ cases show agents performing their roles, but with errors

    3. BENCHMARK F9 DEFINITION
       - FM-2.4 in MAST paper = Role Usurpation
       - Occurs when agents exceed role boundaries or claim others' responsibilities
       - Often manifests subtly (not explicit "I'm taking over" statements)

    4. LLM JUDGE MAPPING BUG
       - LLM judge (_enums.py) incorrectly maps F9 to "Information Withholding"
       - Should be "Role Usurpation" to match benchmark
       - Fix required before using hybrid LLM escalation for F9

    CURRENT STATUS: F1=0.182 (1 TP, 0 FP, 39 FN) - baseline is maintained
    RECOMMENDATION: Hybrid LLM escalation required for meaningful F9 detection
    """

    name = "TurnAwareRoleUsurpationDetector"
    version = "2.0"  # Phase 2: Enhanced with semantic analysis
    supported_failure_modes = ["F9"]

    # Role definitions with semantic descriptions
    ROLE_DEFINITIONS = {
        "coordinator": {
            "description": "Orchestrates workflow, delegates tasks, manages communication between agents",
            "indicators": ["i will assign", "let me delegate", "i'll coordinate", "as coordinator",
                         "orchestrating", "managing the team", "distributing work"],
            "actions": ["assign", "delegate", "coordinate", "organize", "manage team", "distribute"],
        },
        "executor": {
            "description": "Implements solutions, executes code, performs concrete actions",
            "indicators": ["executing now", "i will implement", "implementing the", "as executor",
                         "running the code", "performing the task", "making changes"],
            "actions": ["execute", "implement", "run", "perform", "build", "create", "modify"],
        },
        "reviewer": {
            "description": "Evaluates work quality, provides feedback, validates outputs",
            "indicators": ["reviewing your", "let me review", "my review shows", "as reviewer",
                         "upon review", "checking the quality", "validating output"],
            "actions": ["review", "evaluate", "validate", "check quality", "assess", "critique"],
        },
        "researcher": {
            "description": "Gathers information, investigates problems, analyzes data",
            "indicators": ["researching this", "my research shows", "investigating the", "as researcher",
                         "analyzing the data", "gathering information", "exploring options"],
            "actions": ["research", "investigate", "analyze", "gather info", "explore", "study"],
        },
        "planner": {
            "description": "Designs strategies, creates plans, outlines approaches",
            "indicators": ["planning the approach", "let me plan", "my plan is", "as planner",
                         "designing the strategy", "outlining steps", "creating roadmap"],
            "actions": ["plan", "design", "strategize", "outline", "architect", "blueprint"],
        },
        "tester": {
            "description": "Tests functionality, verifies correctness, finds bugs",
            "indicators": ["testing this", "let me test", "my tests show", "as tester",
                         "verifying functionality", "checking for bugs", "running tests"],
            "actions": ["test", "verify", "debug", "check", "validate functionality", "qa"],
        },
    }

    # ChatDev/MetaGPT specific role mappings
    # Maps agent name patterns to role categories
    # NOTE: CTO excluded - they legitimately review/handle code
    EXECUTIVE_PATTERNS = [
        "chief executive", "ceo", "chief product", "cpo",
        "chief human resources", "human resources officer", "chief creative", "cco",
    ]
    IMPLEMENTER_PATTERNS = [
        "programmer", "developer", "engineer", "coder", "simplecoder",
    ]
    REVIEWER_PATTERNS = [
        "code reviewer", "reviewer", "tester", "qa ",
    ]

    # Code patterns that indicate implementation work
    CODE_PATTERNS = [
        "def ", "class ", "import ", "from ", "```python", "```java", "```js",
        "function ", "const ", "let ", "var ", "public class", "private ",
        "if __name__", "async def", "await ", "return ", "self.", "this.",
    ]

    # AG2-specific execution markers (code was run)
    AG2_EXECUTION_MARKERS = [
        "exitcode:", "code output:", ">>> ", "executed successfully",
        "output:", "result:", "execution result",
    ]

    # Explicit verbal patterns indicating role usurpation
    USURPATION_PATTERNS = [
        (r"(?:tak|assum)(?:ing|e)\s+(?:over|control|charge)", "task_hijacking"),
        (r"(?:skip|bypass|ignore)(?:ping|ed)?\s+(?:the\s+)?(?:review|approval)", "authority_bypass"),
        (r"I(?:'ll|\s+will)\s+(?:decide|determine)\s+(?:to\s+|the\s+|which\s+)", "decision_overreach"),
    ]

    def __init__(self, min_turns: int = 2, strict_mode: bool = False, min_violations: int = 1):
        self.min_turns = min_turns
        self.strict_mode = strict_mode  # If True, be more aggressive in detection
        self.min_violations = min_violations  # Minimum violations to trigger detection

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect role usurpation issues."""
        logger.debug(f"F9 detect: {len(turns)} turns, embedder={self.embedder is not None}")
        if len(turns) < self.min_turns:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few turns to analyze",
                detector_name=self.name,
            )

        # Track agent roles across conversation
        agent_roles = self._infer_agent_roles(turns)
        logger.debug(f"F9: Inferred roles: {agent_roles}")

        # Detect various types of role violations
        violations = []
        affected_turns = []

        # 0. Executive writing code (ChatDev-specific pattern)
        # NOTE: Disabled - causes high FP rate without improving TP
        # In ChatDev, executives review code (which appears in their turns)
        # but this is normal behavior, not role usurpation
        # exec_code_violations = self._detect_executive_coding(turns)

        # 1. Boundary violations (agent acting outside assigned role)
        # NOTE: Disabled - semantic similarity approach has high FP rate
        # The similarity between roles (e.g., reviewer/tester) causes false triggers
        # boundary_violations = self._detect_boundary_violations(turns, agent_roles)
        boundary_violations = []

        # 2. Role conflicts (multiple agents claiming same role)
        role_conflicts = self._detect_role_conflicts(turns, agent_roles)
        violations.extend(role_conflicts)
        for v in role_conflicts:
            affected_turns.extend(v.get("turns", []))

        # 3. Unauthorized actions (agent performing actions outside permission scope)
        unauthorized = self._detect_unauthorized_actions(turns, agent_roles)
        violations.extend(unauthorized)
        for v in unauthorized:
            affected_turns.extend(v.get("turns", []))

        # 4. AG2-specific patterns (self-response, unapproved execution)
        ag2_violations = self._detect_ag2_violations(turns)
        violations.extend(ag2_violations)
        for v in ag2_violations:
            affected_turns.extend(v.get("turns", []))

        # 5. Explicit verbal usurpation patterns (taking over, bypassing approval)
        explicit_violations = self._detect_explicit_usurpation(turns)
        violations.extend(explicit_violations)
        for v in explicit_violations:
            affected_turns.extend(v.get("turns", []))

        # 5b. Multi-task usurpation (agent doing multiple distinct tasks)
        multi_task_violations = self._detect_multi_task_usurpation(turns)
        violations.extend(multi_task_violations)
        for v in multi_task_violations:
            affected_turns.extend(v.get("turns", []))

        # 6. Cascade indicators - DISABLED
        # F9 cascade detection was tested but does not improve F9 detection:
        # - Loose thresholds: 7 TP, 164 FP (F1=0.077)
        # - Tight thresholds: 1 TP, 8 FP (F1=0.105)
        # The cascade pattern (errors + recovery loops) correlates with F9
        # but is not discriminative enough - many F9- cases also have errors.
        # cascade_violations = self._detect_cascade_indicators(turns)
        # violations.extend(cascade_violations)

        if len(violations) < self.min_violations:
            # AMBIGUOUS confidence for F9 when no violations found
            # This triggers LLM escalation in hybrid mode (0.40-0.85 range)
            # since F9 rule-based detection only achieves 2.5% recall on MAST
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.50,  # Ambiguous → LLM escalation in hybrid mode
                failure_mode=None,
                explanation=f"Insufficient evidence ({len(violations)} violations < {self.min_violations} required)",
                detector_name=self.name,
            )

        # Severity based on violation count and types
        critical_violations = [v for v in violations if v.get("severity") == "critical"]
        if len(critical_violations) >= 2 or len(violations) >= 5:
            severity = TurnAwareSeverity.SEVERE
        elif len(violations) >= 3:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.88, 0.45 + len(violations) * 0.12)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F9",
            explanation=f"Role usurpation: {len(violations)} violations found across {len(set(affected_turns))} turns",
            affected_turns=list(set(affected_turns)),
            evidence={
                "violations": violations[:5],  # Top 5 violations
                "agent_roles": agent_roles,
                "critical_violations": len(critical_violations),
            },
            suggested_fix=(
                "Enforce role boundaries: 1) Clearly define agent roles upfront, "
                "2) Add permission checks before actions, 3) Use role-based access control, "
                "4) Monitor for role drift and conflicts."
            ),
            detector_name=self.name,
        )

    def _infer_agent_roles(self, turns: List[TurnSnapshot]) -> Dict[str, Dict[str, Any]]:
        """Infer agent roles from conversation using semantic analysis.

        Phase 2 Enhancement: Uses embeddings to match agent behavior to role descriptions.
        """
        agent_roles = {}
        agent_turns = {}

        # Collect turns by agent
        for turn in turns:
            agent_id = turn.participant_id or "unknown"
            if agent_id not in agent_turns:
                agent_turns[agent_id] = []
            agent_turns[agent_id].append(turn)

        # Infer role for each agent
        for agent_id, agent_turn_list in agent_turns.items():
            # Combine first few turns for role inference
            combined_content = " ".join([t.content[:200] for t in agent_turn_list[:3]])

            # Method 1: Semantic similarity to role descriptions (if embeddings available)
            if self.embedder:
                best_role = self._semantic_role_matching(combined_content)
                if best_role:
                    agent_roles[agent_id] = best_role
                    continue

            # Method 2: Keyword-based fallback
            keyword_role = self._keyword_role_matching(agent_id, combined_content)
            if keyword_role:
                agent_roles[agent_id] = keyword_role
            else:
                # No clear role detected
                agent_roles[agent_id] = {
                    "role": "unclear",
                    "confidence": 0.2,
                    "method": "none",
                }

        return agent_roles

    def _semantic_role_matching(self, content: str) -> Optional[Dict[str, Any]]:
        """Use semantic similarity to match agent content to role descriptions.

        Phase 2: Core semantic enhancement for better role inference.
        """
        if not self.embedder:
            return None

        try:
            # Compare content to each role description
            role_similarities = {}
            for role_name, role_def in self.ROLE_DEFINITIONS.items():
                # Use role description for semantic matching
                similarity = self.semantic_similarity(content, role_def["description"])
                if similarity >= 0:  # Valid similarity score
                    role_similarities[role_name] = similarity

            if not role_similarities:
                return None

            # Get best matching role
            best_role = max(role_similarities, key=role_similarities.get)
            best_score = role_similarities[best_role]
            logger.debug(f"F9 semantic role: best={best_role} score={best_score:.3f}, all={role_similarities}")

            # Require reasonable confidence threshold
            if best_score >= 0.50:  # Semantic similarity threshold
                return {
                    "role": best_role,
                    "confidence": best_score,
                    "method": "semantic",
                    "description": self.ROLE_DEFINITIONS[best_role]["description"],
                }

        except Exception as e:
            logger.debug(f"Semantic role matching failed: {e}")

        return None

    def _keyword_role_matching(self, agent_id: str, content: str) -> Optional[Dict[str, Any]]:
        """Fallback keyword-based role matching."""
        agent_lower = agent_id.lower()
        content_lower = content.lower()

        # Check agent ID
        for role_name, role_def in self.ROLE_DEFINITIONS.items():
            if role_name in agent_lower:
                return {
                    "role": role_name,
                    "confidence": 0.75,
                    "method": "agent_id",
                    "description": role_def["description"],
                }

        # Check content indicators
        role_scores = {}
        for role_name, role_def in self.ROLE_DEFINITIONS.items():
            score = sum(1 for indicator in role_def["indicators"] if indicator in content_lower)
            if score > 0:
                role_scores[role_name] = score

        if role_scores:
            best_role = max(role_scores, key=role_scores.get)
            score = role_scores[best_role]
            return {
                "role": best_role,
                "confidence": min(0.70, 0.4 + score * 0.15),
                "method": "keyword",
                "description": self.ROLE_DEFINITIONS[best_role]["description"],
            }

        return None

    def _is_executive_role(self, participant_id: str) -> bool:
        """Check if participant is an executive (CEO, CPO, CTO, etc.)."""
        id_lower = participant_id.lower()
        return any(pattern in id_lower for pattern in self.EXECUTIVE_PATTERNS)

    def _is_implementer_role(self, participant_id: str) -> bool:
        """Check if participant is an implementer (Programmer, Developer, etc.)."""
        id_lower = participant_id.lower()
        return any(pattern in id_lower for pattern in self.IMPLEMENTER_PATTERNS)

    def _contains_significant_code(self, content: str) -> int:
        """Count code pattern matches in content."""
        return sum(1 for pattern in self.CODE_PATTERNS if pattern in content)

    def _detect_executive_coding(self, turns: List[TurnSnapshot]) -> List[Dict[str, Any]]:
        """Detect executives writing code (primary ChatDev role usurpation pattern).

        In ChatDev, role usurpation often manifests as CEO/CPO/CTO agents
        writing implementation code instead of delegating to programmers.
        """
        violations = []

        for turn in turns:
            agent_id = turn.participant_id or "unknown"

            # Check if this is an executive role
            if not self._is_executive_role(agent_id):
                continue

            # Check if the content contains significant code
            code_matches = self._contains_significant_code(turn.content)

            # Executives writing code is a role usurpation
            # Threshold: 3+ code patterns indicates actual implementation
            if code_matches >= 3:
                violations.append({
                    "type": "executive_coding",
                    "turns": [turn.turn_number],
                    "agent": agent_id,
                    "code_patterns": code_matches,
                    "severity": "critical",
                    "description": f"Executive {agent_id} writing implementation code ({code_matches} code patterns)",
                })

        return violations[:5]  # Limit to top 5

    def _detect_boundary_violations(
        self,
        turns: List[TurnSnapshot],
        agent_roles: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect agents acting outside their role boundaries.

        Phase 2: Uses semantic similarity to detect boundary violations beyond keywords.
        """
        violations = []

        for turn in turns:
            agent_id = turn.participant_id or "unknown"

            # Skip if no clear role assigned
            if agent_id not in agent_roles or agent_roles[agent_id]["role"] == "unclear":
                continue

            assigned_role = agent_roles[agent_id]["role"]
            content = turn.content

            # Check if agent is performing actions from other roles
            for other_role, role_def in self.ROLE_DEFINITIONS.items():
                if other_role == assigned_role:
                    continue

                # Semantic boundary check (if embeddings available)
                if self.embedder:
                    # Check similarity to other role's description
                    other_similarity = self.semantic_similarity(content[:300], role_def["description"])

                    # Also check similarity to OWN role for comparison
                    own_role_def = self.ROLE_DEFINITIONS.get(assigned_role, {})
                    own_similarity = self.semantic_similarity(
                        content[:300], own_role_def.get("description", "")
                    ) if own_role_def else 0.0

                    logger.debug(f"F9 boundary: {agent_id}({assigned_role}) vs {other_role}: other={other_similarity:.3f}, own={own_similarity:.3f}")

                    # Role boundary violation detection:
                    # Balanced thresholds - reduce FPs while maintaining recall
                    # 1. Moderate similarity to other role (>= 0.48)
                    # 2. Clear margin over own role (0.12)
                    # 3. Not strong fit to own role (< 0.55)
                    if (other_similarity >= 0.48 and
                        other_similarity > own_similarity + 0.12 and
                        own_similarity < 0.55):
                        violations.append({
                            "type": "boundary_violation",
                            "turns": [turn.turn_number],
                            "agent": agent_id,
                            "assigned_role": assigned_role,
                            "violated_role": other_role,
                            "similarity": other_similarity,
                            "own_similarity": own_similarity,
                            "severity": "moderate",
                            "description": f"{agent_id} ({assigned_role}) acting as {other_role}",
                        })
                        break  # Only report first violation per turn

                # Keyword fallback
                else:
                    content_lower = content.lower()
                    matches = sum(1 for indicator in role_def["indicators"] if indicator in content_lower)
                    if matches >= 2:  # Multiple indicators = likely violation
                        violations.append({
                            "type": "boundary_violation",
                            "turns": [turn.turn_number],
                            "agent": agent_id,
                            "assigned_role": assigned_role,
                            "violated_role": other_role,
                            "matches": matches,
                            "severity": "moderate",
                            "description": f"{agent_id} ({assigned_role}) using {other_role} patterns",
                        })
                        break

        return violations[:5]  # Limit to top 5

    def _detect_role_conflicts(
        self,
        turns: List[TurnSnapshot],
        agent_roles: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect multiple agents claiming or performing the same role."""
        conflicts = []

        # Group agents by role
        role_to_agents = {}
        for agent_id, role_info in agent_roles.items():
            role = role_info["role"]
            if role == "unclear":
                continue
            if role not in role_to_agents:
                role_to_agents[role] = []
            role_to_agents[role].append(agent_id)

        # Check for conflicts (multiple agents with same role)
        for role, agents in role_to_agents.items():
            if len(agents) > 1:
                # Find turns where both agents act in that role
                conflict_turns = []
                for turn in turns:
                    agent_id = turn.participant_id or "unknown"
                    if agent_id in agents:
                        conflict_turns.append(turn.turn_number)

                if len(conflict_turns) >= 2:  # Actual conflict (both agents active)
                    conflicts.append({
                        "type": "role_conflict",
                        "turns": conflict_turns[:3],  # Sample turns
                        "role": role,
                        "agents": agents,
                        "severity": "critical",  # Role conflicts are serious
                        "description": f"Multiple agents ({', '.join(agents)}) assigned {role} role",
                    })

        return conflicts

    def _detect_unauthorized_actions(
        self,
        turns: List[TurnSnapshot],
        agent_roles: Dict[str, Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect agents performing actions outside their permission scope.

        NOTE: This method is currently disabled due to high false positive rate.
        Simple keyword matching like 'delete' triggers on discussion of deletion,
        not actual deletion actions.
        """
        # Temporarily disabled to reduce FPs
        return []

        unauthorized = []

        # High-privilege actions that require specific roles
        sensitive_actions = {
            "delete": ["executor", "coordinator"],  # Only these roles can delete
            "deploy": ["executor", "coordinator"],
            "approve": ["reviewer", "coordinator"],
            "reject": ["reviewer", "coordinator"],
            "assign": ["coordinator"],  # Only coordinator can assign
            "delegate": ["coordinator"],
        }

        for turn in turns:
            agent_id = turn.participant_id or "unknown"

            # Skip if no role assigned
            if agent_id not in agent_roles or agent_roles[agent_id]["role"] == "unclear":
                continue

            assigned_role = agent_roles[agent_id]["role"]
            content_lower = turn.content.lower()

            # Check for sensitive actions
            for action, allowed_roles in sensitive_actions.items():
                if action in content_lower and assigned_role not in allowed_roles:
                    unauthorized.append({
                        "type": "unauthorized_action",
                        "turns": [turn.turn_number],
                        "agent": agent_id,
                        "assigned_role": assigned_role,
                        "action": action,
                        "allowed_roles": allowed_roles,
                        "severity": "critical",
                        "description": f"{agent_id} ({assigned_role}) attempted '{action}' (requires {allowed_roles})",
                    })

        return unauthorized[:3]  # Limit to top 3

    def _get_framework(self, turns: List[TurnSnapshot]) -> str:
        """Get framework from turn metadata."""
        for turn in turns:
            if turn.turn_metadata:
                fw = turn.turn_metadata.get("framework", "")
                if fw:
                    return fw.lower()
        return "unknown"

    def _detect_ag2_violations(self, turns: List[TurnSnapshot]) -> List[Dict[str, Any]]:
        """Detect AG2-specific role usurpation patterns.

        AG2/AutoGen patterns:
        - Many consecutive turns from same agent (3+) without user checkpoint
          indicates the agent is proceeding without approval

        NOTE: The role='user' field in AG2 metadata is a messaging convention,
        not an indicator of usurpation. Removed that pattern to avoid FPs.
        """
        violations = []

        # Check if this is AG2 framework
        framework = self._get_framework(turns)
        if "ag2" not in framework and "autogen" not in framework:
            return []

        # Known human/user identifiers
        USER_IDENTIFIERS = ["user", "human", "person", "customer"]

        def is_likely_user(participant_id: str) -> bool:
            """Check if participant_id looks like a real user."""
            pid_lower = (participant_id or "").lower()
            return any(uid in pid_lower for uid in USER_IDENTIFIERS)

        # Pattern: Many consecutive turns from same participant without user
        # Threshold of 3+ consecutive turns indicates agent proceeding autonomously
        prev_pid = None
        consecutive_count = 0
        for turn in turns:
            pid = turn.participant_id or ""
            if is_likely_user(pid):
                # Real user turn resets the count
                prev_pid = None
                consecutive_count = 0
            elif prev_pid == pid:
                consecutive_count += 1
                # Require 3+ consecutive turns (stricter threshold)
                if consecutive_count >= 3:
                    violations.append({
                        "type": "autonomous_agent",
                        "turns": [turn.turn_number],
                        "agent": pid,
                        "severity": "critical",
                        "description": f"Agent '{pid}' proceeding autonomously ({consecutive_count+1} turns without user)",
                    })
            else:
                consecutive_count = 1
                prev_pid = pid

        return violations[:3]  # Limit to top 3

    def _detect_explicit_usurpation(self, turns: List[TurnSnapshot]) -> List[Dict[str, Any]]:
        """Detect explicit verbal patterns indicating role usurpation.

        High-precision patterns that clearly indicate an agent is:
        - Taking over another's role
        - Bypassing required approvals
        - Making decisions outside their scope
        """
        violations = []

        for turn in turns:
            if turn.participant_type != "agent":
                continue

            content_lower = turn.content.lower()

            for pattern, vtype in self.USURPATION_PATTERNS:
                if re.search(pattern, content_lower):
                    violations.append({
                        "type": vtype,
                        "turns": [turn.turn_number],
                        "agent": turn.participant_id,
                        "severity": "critical",
                        "description": f"Agent {turn.participant_id}: {vtype.replace('_', ' ')}",
                    })
                    break  # Only report first pattern match per turn

        return violations[:5]  # Limit to top 5

    def _detect_multi_task_usurpation(self, turns: List[TurnSnapshot]) -> List[Dict[str, Any]]:
        """Detect agents performing multiple distinct tasks that should be separate roles.

        Pattern: "I've also [done X], [done Y], [done Z]" indicates role boundary violation.
        """
        violations = []

        # Task verbs that typically belong to different roles
        task_verbs = [
            "written", "wrote", "created", "generated", "composed",
            "edited", "revised", "modified", "updated", "changed",
            "formatted", "styled", "designed", "structured",
            "reviewed", "checked", "validated", "verified", "tested",
            "scheduled", "planned", "organized", "arranged",
            "published", "deployed", "released", "launched",
            "approved", "authorized", "signed off",
        ]

        # Conjunction patterns indicating multiple tasks
        multi_task_patterns = [
            r"(?:i(?:'ve| have| also)?\s+(?:also|additionally)?\s+(?:written|wrote|created|edited|formatted|reviewed|scheduled|published|approved|tested|designed).*?(?:and|,)\s+(?:also\s+)?(?:written|wrote|created|edited|formatted|reviewed|scheduled|published|approved|tested|designed))",
            r"by the way,?\s+i(?:'ve| have)?\s+(?:also|additionally)?\s+(?:written|wrote|created|edited|formatted|reviewed|scheduled|published)",
        ]

        for turn in turns:
            if turn.participant_type != "agent":
                continue

            content_lower = turn.content.lower()

            # Check for multi-task patterns
            for pattern in multi_task_patterns:
                if re.search(pattern, content_lower):
                    # Count distinct task verbs mentioned
                    task_count = sum(1 for verb in task_verbs if verb in content_lower)

                    if task_count >= 3:  # At least 3 different tasks
                        violations.append({
                            "type": "multi_task_usurpation",
                            "turns": [turn.turn_number],
                            "agent": turn.participant_id,
                            "task_count": task_count,
                            "severity": "moderate",
                            "description": f"Agent {turn.participant_id} performing {task_count} distinct tasks (role boundary violation)",
                        })
                        break  # Only report once per turn

        return violations[:3]  # Limit to top 3

    def _detect_cascade_indicators(self, turns: List[TurnSnapshot]) -> List[Dict[str, Any]]:
        """Detect F9 cascade indicators: errors, complexity, recovery loops.

        Key discovery: F9 is a systemic failure pattern with these signals:
        - Error rate 2.5x higher in F9+ cases (40% vs 16%)
        - Longer conversations (6.4 vs 4.8 messages)
        - Error recovery loops (error → fix → error pattern)

        CRITICAL: Only returns violations when ALL conditions are met.
        Partial matches do NOT trigger F9 detection.
        """
        # Signal 1: Error/traceback presence (strong indicators only)
        strong_error_patterns = ["traceback", "exception:", "exitcode: 1", "exitcode:1"]

        strong_error_count = 0
        for turn in turns:
            content_lower = turn.content.lower()
            if any(p in content_lower for p in strong_error_patterns):
                strong_error_count += 1

        # Signal 2: Conversation complexity
        message_count = len(turns)

        # Signal 3: Error recovery loop (REQUIRED)
        has_recovery_loop = self._detect_recovery_loop(turns)

        # F9 Cascade Pattern requires ALL THREE signals:
        # 1. Recovery loop with strong error
        # 2. At least one strong error (traceback/exception/exitcode)
        # 3. Extended conversation (7+ messages)
        if not has_recovery_loop:
            return []

        if strong_error_count < 1:
            return []

        if message_count <= 6:
            return []

        # All conditions met - return cascade violation
        return [{
            "type": "cascade_pattern",
            "turns": [],
            "severity": "critical",
            "description": f"Systemic cascade: {strong_error_count} tracebacks, {message_count} messages, recovery_loop=True",
        }]

    def _detect_recovery_loop(self, turns: List[TurnSnapshot]) -> bool:
        """Detect error → revised code → error pattern.

        This pattern indicates the system had to step in and provide revised code,
        which is a form of role boundary violation.

        Requires at least one STRONG error indicator (traceback, exception, exitcode).
        """
        # Strong error patterns that indicate actual code execution failure
        strong_patterns = ["traceback", "exception:", "exitcode: 1", "exitcode:1"]

        strong_error_indices = []
        any_error_indices = []

        for i, turn in enumerate(turns):
            content_lower = turn.content.lower()
            if any(p in content_lower for p in strong_patterns):
                strong_error_indices.append(i)
                any_error_indices.append(i)
            elif "error:" in content_lower or "error in" in content_lower:
                any_error_indices.append(i)

        # Recovery loop requires:
        # 1. At least one strong error (traceback/exception/exitcode)
        # 2. At least 2 error-related turns with gap between them
        if len(strong_error_indices) == 0:
            return False

        if len(any_error_indices) < 2:
            return False

        # Must have turns between errors (indicating retry)
        return (any_error_indices[-1] - any_error_indices[0]) >= 2
