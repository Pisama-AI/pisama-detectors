"""
F1: Specification Mismatch Detector
===================================

Analyzes whether agent outputs match the user's requirements:
1. Missing required features - user asked for X but agent didn't provide
2. Extra unrequested features - agent added things user didn't ask for
3. Misinterpreted requirements - agent did something different than asked
4. Incomplete implementation - partial fulfillment of requirements
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


class TurnAwareSpecificationMismatchDetector(EmbeddingMixin, TurnAwareDetector):
    """Detects F1: Specification Mismatch in conversations.

    Analyzes whether agent outputs match the user's requirements:
    1. Missing required features - user asked for X but agent didn't provide
    2. Extra unrequested features - agent added things user didn't ask for
    3. Misinterpreted requirements - agent did something different than asked
    4. Incomplete implementation - partial fulfillment of requirements

    Phase 2 Enhancement: Uses semantic similarity (EmbeddingMixin) for requirement
    matching instead of simple keyword matching. This improves detection accuracy
    by understanding semantic equivalence (e.g., "authentication" matches "login system").

    This is the 3rd most common failure mode in MAST (30% prevalence).
    """

    name = "TurnAwareSpecificationMismatchDetector"
    version = "2.0"  # Phase 2: Enhanced with semantic matching
    supported_failure_modes = ["F1"]

    # Requirement indicators in user messages
    REQUIREMENT_KEYWORDS = [
        "must", "should", "need", "require", "want", "please",
        "make sure", "ensure", "implement", "create", "build",
        "add", "include", "support", "feature", "functionality",
    ]

    # Mismatch indicators - signs of spec violation
    MISMATCH_INDICATORS = [
        "instead of", "rather than", "different from", "not what",
        "missing", "forgot", "didn't include", "left out",
        "extra", "unnecessary", "not requested", "not needed",
        "incomplete", "partial", "only part", "some of",
    ]

    def __init__(
        self,
        min_requirement_terms: int = 2,
        coverage_threshold: float = 0.55,  # Raised from 0.40 to reduce FPs (41% FPR)
    ):
        self.min_requirement_terms = min_requirement_terms
        self.coverage_threshold = coverage_threshold

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect specification mismatches between requirements and outputs."""
        user_turns = [t for t in turns if t.participant_type == "user"]
        agent_turns = [t for t in turns if t.participant_type == "agent"]
        system_turns = [t for t in turns if t.participant_type == "system"]

        # Handle multi-agent systems where all participants are "agent" role
        # (e.g., ChatDev where CEO gives task to Programmer)
        # The task can come from: user turns, system prompt, or first agent
        synthetic_user_turns = []
        agents_to_check = agent_turns
        if user_turns:
            synthetic_user_turns = user_turns
        elif system_turns and any(len(t.content) > 50 for t in system_turns):
            # Multi-agent: system prompt contains the task
            synthetic_user_turns = [t for t in system_turns if len(t.content) > 50]
        elif agent_turns:
            # Multi-agent: first agent's message contains the task
            synthetic_user_turns = [agent_turns[0]]
            agents_to_check = agent_turns[1:]

        if not synthetic_user_turns or not agents_to_check:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Need both user requirements and agent output",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # Extract requirements from user turns (or first agent in multi-agent)
        requirements = self._extract_requirements(synthetic_user_turns)

        if len(requirements) < self.min_requirement_terms:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.5,
                failure_mode=None,
                explanation="Could not extract clear requirements from user input",
                detector_name=self.name,
            )

        # Check agent responses against requirements
        agent_content = " ".join([t.content for t in agents_to_check])

        # 1. Check requirement coverage - only flag if below threshold
        coverage_result = self._check_coverage(requirements, agent_content)
        if coverage_result["coverage"] < self.coverage_threshold:
            issues.append({
                "type": "missing_requirements",
                "uncovered": coverage_result["uncovered"][:5],
                "coverage_ratio": coverage_result["coverage"],
                "description": f"Missing requirements: {', '.join(coverage_result['uncovered'][:3])} ({coverage_result['coverage']:.0%} coverage < {self.coverage_threshold:.0%} threshold)",
            })
            for ut in synthetic_user_turns:
                affected_turns.append(ut.turn_number)

        # 2. Check for explicit mismatch indicators
        mismatch_issues = self._check_mismatch_indicators(agents_to_check)
        issues.extend(mismatch_issues)
        for issue in mismatch_issues:
            affected_turns.extend(issue.get("turns", []))

        # 3. Check for scope creep (unrequested additions)
        scope_issues = self._check_scope_creep(synthetic_user_turns, agents_to_check)
        issues.extend(scope_issues)
        for issue in scope_issues:
            affected_turns.extend(issue.get("turns", []))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.8,
                failure_mode=None,
                explanation=f"Requirements appear to be addressed ({coverage_result['coverage']:.0%} coverage)",
                detector_name=self.name,
            )

        # Determine severity
        if any(i["type"] == "missing_requirements" and i.get("coverage_ratio", 1) < 0.5 for i in issues):
            severity = TurnAwareSeverity.SEVERE
        elif len(issues) >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.85, 0.5 + len(issues) * 0.12)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F1",
            explanation=f"Specification mismatch: {len(issues)} issues found",
            affected_turns=list(set(affected_turns)),
            evidence={
                "issues": issues,
                "extracted_requirements": list(requirements)[:10],
            },
            suggested_fix=(
                "Review requirements more carefully. Consider: 1) Extracting explicit "
                "requirements before implementing, 2) Validating output against each "
                "requirement, 3) Asking clarifying questions for ambiguous specs."
            ),
            detector_name=self.name,
        )

    def _extract_requirements(self, user_turns: List[TurnSnapshot]) -> set:
        """Extract requirement keywords from user messages."""
        requirements = set()

        for turn in user_turns:
            content_lower = turn.content.lower()
            words = content_lower.split()

            # Extract nouns/key terms after requirement keywords
            for i, word in enumerate(words):
                if word in self.REQUIREMENT_KEYWORDS:
                    # Get next 1-3 words as potential requirement
                    for j in range(i + 1, min(i + 4, len(words))):
                        candidate = words[j].strip(",.;:!?")
                        if len(candidate) > 3 and candidate.isalpha():
                            requirements.add(candidate)

            # Also extract capitalized terms (likely proper nouns/features)
            for word in turn.content.split():
                clean = word.strip(",.;:!?()")
                if clean and clean[0].isupper() and len(clean) > 2:
                    requirements.add(clean.lower())

        return requirements

    def _chunk_output(self, agent_content: str, max_len: int = 500) -> List[str]:
        """Chunk agent output into smaller pieces for semantic matching.

        Args:
            agent_content: Full agent output text
            max_len: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        # Split on paragraphs first (double newlines or sentences)
        paragraphs = [p.strip() for p in agent_content.split('\n\n') if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk) + len(para) <= max_len:
                current_chunk += para + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para + " "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [agent_content[:max_len]]

    def _semantic_requirement_matching(
        self,
        requirements: set,
        agent_content: str,
        similarity_threshold: float = 0.82  # Raised from 0.75 to reduce FPs
    ) -> dict:
        """Use embeddings to check if requirements are semantically met.

        Phase 2 Enhancement: Semantic matching understands synonyms and paraphrasing.
        For example, "authentication system" matches "login functionality".

        Args:
            requirements: Set of requirement keywords/phrases
            agent_content: Agent output text to check
            similarity_threshold: Minimum similarity score to consider a match (0.82, raised from 0.75)

        Returns:
            Dict with covered/uncovered requirements and coverage ratio
        """
        if not self.embedder or not requirements:
            # Fallback to keyword matching if embeddings unavailable
            return self._keyword_requirement_matching(requirements, agent_content)

        try:
            # Chunk agent output for better semantic matching
            output_chunks = self._chunk_output(agent_content, max_len=500)

            # OPTIMIZATION: Pre-encode all passages ONCE instead of per-requirement
            # This reduces O(n*m) to O(n+m) embedding calls
            passage_embs = self.embedder.encode_passages(output_chunks) if output_chunks else None
            if passage_embs is None or len(passage_embs) == 0:
                return self._keyword_requirement_matching(requirements, agent_content)

            covered = set()
            uncovered = set()

            # For each requirement, compute similarity with pre-encoded passages
            for req in requirements:
                try:
                    query_emb = self.embedder.encode_query(req)
                    similarities = self.embedder.batch_similarity(query_emb, passage_embs)

                    if similarities is not None and len(similarities) > 0:
                        best_match = float(max(similarities))
                        if best_match >= similarity_threshold:
                            covered.add(req)
                        else:
                            uncovered.add(req)
                    else:
                        # Fallback to keyword
                        if req in agent_content.lower():
                            covered.add(req)
                        else:
                            uncovered.add(req)
                except Exception:
                    # Fallback to keyword for this requirement
                    if req in agent_content.lower():
                        covered.add(req)
                    else:
                        uncovered.add(req)

            coverage = len(covered) / len(requirements) if requirements else 1.0

            return {
                "covered": list(covered),
                "uncovered": list(uncovered),
                "coverage": coverage,
                "method": "semantic",
            }

        except Exception as e:
            logger.debug(f"Semantic requirement matching failed: {e}, falling back to keyword")
            return self._keyword_requirement_matching(requirements, agent_content)

    def _keyword_requirement_matching(self, requirements: set, agent_content: str) -> dict:
        """Keyword-based requirement matching (fallback for when embeddings unavailable).

        Args:
            requirements: Set of requirement keywords
            agent_content: Agent output text

        Returns:
            Dict with covered/uncovered requirements and coverage ratio
        """
        agent_lower = agent_content.lower()
        covered = set()
        uncovered = set()

        for req in requirements:
            if req in agent_lower:
                covered.add(req)
            else:
                uncovered.add(req)

        coverage = len(covered) / len(requirements) if requirements else 1.0

        return {
            "covered": list(covered),
            "uncovered": list(uncovered),
            "coverage": coverage,
            "method": "keyword",
        }

    def _check_coverage(self, requirements: set, agent_content: str) -> dict:
        """Check how many requirements are addressed in agent output.

        Phase 2: Prefers semantic matching, falls back to keyword matching.
        """
        return self._semantic_requirement_matching(requirements, agent_content)

    def _check_mismatch_indicators(self, agent_turns: List[TurnSnapshot]) -> list:
        """Check for explicit mismatch indicators in agent output."""
        issues = []

        for turn in agent_turns:
            content_lower = turn.content.lower()
            for indicator in self.MISMATCH_INDICATORS:
                if indicator in content_lower:
                    issues.append({
                        "type": "explicit_mismatch",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Mismatch indicator found: '{indicator}'",
                    })
                    break  # One per turn

        return issues[:3]

    def _check_scope_creep(
        self, user_turns: List[TurnSnapshot], agent_turns: List[TurnSnapshot]
    ) -> list:
        """Check for scope creep - agent adding unrequested features."""
        issues = []

        # Look for phrases indicating unrequested additions
        scope_indicators = [
            "i also added", "i've included extra", "bonus feature",
            "additionally", "as a bonus", "extra functionality",
            "i went ahead and", "while i was at it",
        ]

        for turn in agent_turns:
            content_lower = turn.content.lower()
            for indicator in scope_indicators:
                if indicator in content_lower:
                    issues.append({
                        "type": "scope_creep",
                        "turns": [turn.turn_number],
                        "indicator": indicator,
                        "description": f"Potential scope creep: '{indicator}'",
                    })
                    break

        return issues[:2]
