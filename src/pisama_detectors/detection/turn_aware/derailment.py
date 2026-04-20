"""
F6: Task Derailment Detection
=============================

Tracks topic consistency and task focus across the conversation,
detecting when:
1. Agent responses drift from the original task
2. Conversation topic changes unexpectedly
3. Agent addresses wrong task or substitutes tasks
"""

import re
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


class TurnAwareDerailmentDetector(EmbeddingMixin, TurnAwareDetector):
    """Detects F6: Task Derailment across conversation turns.

    Tracks topic consistency and task focus across the conversation,
    detecting when:
    1. Agent responses drift from the original task
    2. Conversation topic changes unexpectedly
    3. Agent addresses wrong task or substitutes tasks

    Uses sliding window analysis to detect gradual drift.

    Enhanced with semantic embeddings (v2.0):
    - Embedding-based topic drift detection when available
    - Falls back to keyword-based for speed or when embeddings unavailable
    - Progressive drift detection using similarity trends

    Based on MAST research (NeurIPS 2025): FM-2.3 Task Derailment (20% prevalence)
    """

    name = "TurnAwareDerailmentDetector"
    version = "2.1"  # Phase 1: Benign pattern whitelisting
    supported_failure_modes = ["F6"]

    # Code patterns (shared with F7 detector logic)
    CODE_PATTERNS = [
        "def ", "class ", "import ", "from ", "function ",
        "const ", "let ", "var ", "return ", "if (", "if(",
        "for (", "for(", "while ", "=>", "```", "{", "}",
        "self.", "this.", "async ", "await ",
    ]

    # Phase 1: Framework-specific benign patterns (legitimate role transitions, not derailment)
    BENIGN_PATTERNS = {
        "ChatDev": [
            r"now switching to\b",
            r"role transition",
            r"moving to (?:the )?next (?:phase|stage)",
            r"switching (?:to|from) .+ (?:role|phase)",
            r"transitioning to",
            r"handing off to",
            r"passing (?:control|task) to",
        ],
        "AG2": [
            r"let me think",
            r"reasoning step",
            r"analyzing (?:the )?problem",
            r"breaking down (?:the )?task",
            r"let me (?:check|verify|validate)",
        ],
        "MetaGPT": [
            r"moving to implementation phase",
            r"switching context",
            r"proceeding to",
            r"transitioning from .+ to",
        ],
        "AutoGen": [
            r"delegating to",
            r"coordinating with",
            r"checking with",
        ],
        "n8n": [
            r"node executed",
            r"workflow triggered",
            r"execution completed",
            r"processing input",
            r"returning output",
            r"fetching data",
            r"sending request",
            r"received response",
            r"transforming data",
            r"filtering results",
            r"merging data",
            r"splitting data",
        ],
        "default": [
            r"as (?:you )?(?:requested|asked)",
            r"to address your (?:question|request)",
            r"focusing on (?:the|your)",
        ],
    }

    def __init__(
        self,
        drift_threshold: float = 0.55,  # Lowered for better recall (was 0.70)
        min_turns_for_analysis: int = 3,
        window_size: int = 5,
        require_strong_evidence: bool = False,  # Disabled for MAST recall (was True)
        framework: Optional[str] = None,
    ):
        self.drift_threshold = drift_threshold
        self.min_turns_for_analysis = min_turns_for_analysis
        self.window_size = window_size
        self.require_strong_evidence = require_strong_evidence
        self.framework = framework
        self._embedder = None

    @property
    def embedder(self):
        if self._embedder is None:
            try:
                from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder
                self._embedder = get_embedder()
            except ImportError:
                logger.warning("EmbeddingService not available, using fallback")
                self._embedder = "fallback"
        return self._embedder

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect task derailment across conversation turns."""
        if len(turns) < self.min_turns_for_analysis:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few turns to analyze for derailment",
                detector_name=self.name,
            )

        # Extract initial task/topic from first user turn (or first turn if all agents)
        user_turns = [t for t in turns if t.participant_type == "user"]
        agent_turns = [t for t in turns if t.participant_type == "agent"]
        system_turns = [t for t in turns if t.participant_type == "system"]

        # Handle multi-agent systems where all participants are "agent" role
        # (e.g., ChatDev where CEO gives task to Programmer)
        # The task can come from: user turns, system prompt, or first agent
        initial_task = None
        if user_turns:
            initial_task = user_turns[0].content
        elif system_turns and any(len(t.content) > 50 for t in system_turns):
            # Multi-agent: system prompt contains the task
            task_turns = [t for t in system_turns if len(t.content) > 50]
            initial_task = task_turns[0].content
        elif agent_turns:
            # Treat first agent as task-giver, rest as executors
            initial_task = agent_turns[0].content
            agent_turns = agent_turns[1:]  # Remaining agents to check for derailment

        if not initial_task or not agent_turns:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No user-agent interaction found",
                detector_name=self.name,
            )
        initial_terms = self._extract_key_terms(initial_task)

        # Check if user is asking for code
        is_code_task = self._is_code_task(initial_task)

        derailment_issues = []
        affected_turns = []
        high_drift_count = 0

        # OPTIMIZATION: Pre-filter turns and use batch similarity computation
        turns_to_check = []
        for agent_turn in agent_turns:
            # Skip drift check for code responses to code tasks
            if is_code_task and self._is_code_response(agent_turn.content):
                continue  # Code response to code task - don't flag as drift

            # Phase 1: Skip drift check for benign patterns (legitimate role transitions)
            if self._is_benign_pattern(agent_turn.content):
                continue  # Benign framework pattern - not derailment

            turns_to_check.append(agent_turn)

        # OPTIMIZATION: Compute drift scores in batch using batch_semantic_similarity
        similarities = []  # Initialize for progressive drift check
        if turns_to_check and self.embedder:
            turn_contents = [t.content for t in turns_to_check]
            similarities = self.batch_semantic_similarity(initial_task, turn_contents)

            if similarities:
                # Process batch results
                for i, (agent_turn, sim) in enumerate(zip(turns_to_check, similarities)):
                    drift_score = 1.0 - sim
                    coverage = sim

                    if drift_score > self.drift_threshold:
                        high_drift_count += 1
                        if not self.require_strong_evidence:
                            derailment_issues.append({
                                "type": "topic_drift",
                                "turn": agent_turn.turn_number,
                                "drift_score": drift_score,
                                "coverage": coverage,
                                "description": f"Agent turn {agent_turn.turn_number} drifted from task (drift={drift_score:.2f})",
                            })
                            affected_turns.append(agent_turn.turn_number)
            else:
                # Fallback to per-turn if batch fails
                for agent_turn in turns_to_check:
                    drift_score, coverage = self._compute_topic_drift(
                        initial_task, agent_turn.content
                    )
                    if drift_score > self.drift_threshold:
                        high_drift_count += 1
                        if not self.require_strong_evidence:
                            derailment_issues.append({
                                "type": "topic_drift",
                                "turn": agent_turn.turn_number,
                                "drift_score": drift_score,
                                "coverage": coverage,
                                "description": f"Agent turn {agent_turn.turn_number} drifted from task (drift={drift_score:.2f})",
                            })
                            affected_turns.append(agent_turn.turn_number)
        else:
            # No turns to check or no embedder - use keyword fallback per turn
            for agent_turn in turns_to_check:
                drift_score, coverage = self._compute_topic_drift(
                    initial_task, agent_turn.content, use_embeddings=False
                )
                if drift_score > self.drift_threshold:
                    high_drift_count += 1
                    if not self.require_strong_evidence:
                        derailment_issues.append({
                            "type": "topic_drift",
                            "turn": agent_turn.turn_number,
                            "drift_score": drift_score,
                            "coverage": coverage,
                            "description": f"Agent turn {agent_turn.turn_number} drifted from task (drift={drift_score:.2f})",
                        })
                        affected_turns.append(agent_turn.turn_number)

        # Check for progressive drift (getting worse over time)
        # OPTIMIZATION: Use pre-computed similarities to avoid re-encoding
        progressive_drift = {"detected": False}
        if len(turns_to_check) >= 3 and similarities:
            # Use pre-computed similarities from batch_semantic_similarity above
            progressive_drift = self._detect_progressive_drift_from_similarities(
                turns_to_check, similarities
            )
        elif len(agent_turns) >= 3 and not similarities:
            # Fallback only if no batch similarities were computed
            progressive_drift = self._detect_progressive_drift(
                initial_task, agent_turns, is_code_task
            )
        # Handle progressive drift detection result (from either path)
        if progressive_drift["detected"]:
            derailment_issues.append({
                "type": "progressive_drift",
                "turn": progressive_drift["worst_turn"],
                "description": "Agent responses progressively drifting from task",
                "drift_progression": progressive_drift.get("drift_scores", []),
            })
            affected_turns.append(progressive_drift["worst_turn"])

        # Check for task substitution (strongest signal)
        task_substitution = self._detect_task_substitution(
            initial_task, agent_turns
        )
        if task_substitution["detected"]:
            derailment_issues.append({
                "type": "task_substitution",
                "turn": task_substitution["turn"],
                "description": task_substitution["description"],
            })
            affected_turns.append(task_substitution["turn"])

        # If requiring strong evidence, only detect if we have:
        # 1. Task substitution, OR
        # 2. Progressive drift, OR
        # 3. Multiple (3+) high-drift turns
        if self.require_strong_evidence:
            has_strong_evidence = (
                task_substitution["detected"] or
                progressive_drift["detected"] or
                high_drift_count >= 3
            )
            if not has_strong_evidence:
                return TurnAwareDetectionResult(
                    detected=False,
                    severity=TurnAwareSeverity.NONE,
                    confidence=0.9,
                    failure_mode=None,
                    explanation="Agent maintained task focus across all turns",
                    detector_name=self.name,
                )
            # If we have strong evidence from high_drift_count, add drift issues
            elif high_drift_count >= 3 and not derailment_issues:
                derailment_issues.append({
                    "type": "high_drift",
                    "turn": "multiple",
                    "description": f"Multiple turns ({high_drift_count}) showed high topic drift from initial task",
                    "drift_count": high_drift_count,
                })
                # Add affected turn numbers
                for agent_turn in agent_turns:
                    if not (is_code_task and self._is_code_response(agent_turn.content)):
                        drift_score, _ = self._compute_topic_drift(initial_task, agent_turn.content)
                        if drift_score > self.drift_threshold:
                            affected_turns.append(agent_turn.turn_number)

        if not derailment_issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.9,
                failure_mode=None,
                explanation="Agent maintained task focus across all turns",
                detector_name=self.name,
            )

        # Determine severity
        if any(i["type"] == "progressive_drift" for i in derailment_issues):
            severity = TurnAwareSeverity.SEVERE
        elif any(i["type"] == "task_substitution" for i in derailment_issues):
            severity = TurnAwareSeverity.MODERATE
        elif len(derailment_issues) >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.95, 0.5 + len(derailment_issues) * 0.15)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F6",
            explanation=f"Task derailment detected: {len(derailment_issues)} issues",
            affected_turns=list(set(affected_turns)),
            evidence={
                "issues": derailment_issues,
                "initial_task_terms": list(initial_terms)[:10],
                "total_turns": len(turns),
            },
            suggested_fix=(
                "Add task reminders in agent prompts. Consider using: "
                "'Stay focused on: [ORIGINAL_TASK]. Do not address unrelated topics.'"
            ),
            detector_name=self.name,
        )

    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms from text, excluding stopwords."""
        words = text.lower().split()
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "and", "but", "or", "nor", "so", "yet", "not", "it",
            "this", "that", "these", "those", "i", "you", "we", "they",
            "me", "him", "her", "us", "them", "my", "your", "our", "their",
        }
        return {w for w in words if len(w) > 2 and w not in stopwords}

    def _is_benign_pattern(self, content: str) -> bool:
        """Phase 1: Check if content contains benign framework-specific patterns."""
        content_lower = content.lower()

        # Get patterns for this framework (or default)
        # BUGFIX: Use list concatenation to avoid mutating class-level dictionary values
        # which caused unbounded list growth and exponential slowdown
        framework_key = self.framework if self.framework in self.BENIGN_PATTERNS else "default"
        patterns = self.BENIGN_PATTERNS.get(framework_key, []) + self.BENIGN_PATTERNS["default"]

        # Check if any benign pattern matches
        for pattern in patterns:
            if re.search(pattern, content_lower):
                return True
        return False

    def _is_code_task(self, task: str) -> bool:
        """Check if the task is asking for code."""
        code_keywords = [
            "write", "code", "function", "implement", "create",
            "program", "script", "fix", "debug", "add", "python",
            "javascript", "java", "def", "class", "method",
            "build", "develop", "generate", "make",
        ]
        task_lower = task.lower()
        return any(kw in task_lower for kw in code_keywords)

    def _is_code_response(self, content: str) -> bool:
        """Check if response contains code."""
        content_lower = content.lower()
        code_pattern_count = sum(1 for p in self.CODE_PATTERNS if p.lower() in content_lower)
        return code_pattern_count >= 2

    def _compute_topic_drift(
        self,
        task: str,
        output: str,
        use_embeddings: bool = True,
    ) -> tuple:
        """Compute topic drift and task coverage.

        Enhanced with embedding-based semantic similarity (v2.0).
        Falls back to keyword-based when embeddings unavailable.

        Returns:
            (drift_score, coverage) tuple where drift_score in [0,1]
            Higher drift_score = more drift from task
        """
        # Try embedding-based similarity first (more accurate)
        if use_embeddings:
            similarity = self.semantic_similarity(task, output)
            if similarity >= 0:  # Embeddings available
                # Convert similarity to drift score (inverted)
                drift_score = 1.0 - similarity
                coverage = similarity  # Similarity approximates coverage
                return drift_score, coverage

        # Fallback to keyword-based
        task_terms = self._extract_key_terms(task)
        output_terms = self._extract_key_terms(output)

        if not task_terms:
            return 0.0, 1.0

        overlap = task_terms & output_terms
        coverage = len(overlap) / len(task_terms)

        # New terms not in task
        new_terms = output_terms - task_terms
        novelty_ratio = len(new_terms) / max(len(output_terms), 1)

        drift_score = (1 - coverage) * 0.6 + novelty_ratio * 0.4
        return min(drift_score, 1.0), coverage

    def _detect_progressive_drift(
        self,
        initial_task: str,
        agent_turns: List[TurnSnapshot],
        is_code_task: bool = False,
    ) -> Dict[str, Any]:
        """Detect if agent responses are progressively drifting.

        Enhanced with batch embedding analysis (v2.0).
        Phase 1: Skip benign patterns.
        """
        # Filter turns for analysis
        turns_to_analyze = []
        for turn in agent_turns:
            if is_code_task and self._is_code_response(turn.content):
                continue
            # Phase 1: Skip benign patterns
            if self._is_benign_pattern(turn.content):
                continue
            turns_to_analyze.append(turn)

        if len(turns_to_analyze) < 3:
            return {"detected": False}

        # Try batch embedding analysis first (more efficient)
        responses = [t.content for t in turns_to_analyze]
        semantic_result = self.detect_semantic_drift(initial_task, responses)

        if semantic_result.get("available"):
            # Use embedding-based progressive drift detection
            if semantic_result.get("progressive_drift"):
                similarities = semantic_result["similarities"]
                # Find worst turn (lowest similarity)
                min_idx = similarities.index(min(similarities))
                worst_turn = turns_to_analyze[min_idx].turn_number

                drift_scores = [
                    {"turn": t.turn_number, "drift": 1.0 - sim, "similarity": sim}
                    for t, sim in zip(turns_to_analyze, similarities)
                ]

                return {
                    "detected": True,
                    "worst_turn": worst_turn,
                    "drift_scores": drift_scores,
                    "avg_similarity": semantic_result["avg_similarity"],
                    "method": "embedding",
                }

            return {"detected": False, "method": "embedding"}

        # Fallback to keyword-based progressive drift
        drift_scores = []
        for turn in turns_to_analyze:
            drift, _ = self._compute_topic_drift(initial_task, turn.content, use_embeddings=False)
            drift_scores.append({
                "turn": turn.turn_number,
                "drift": drift,
            })

        # Check if drift is increasing
        scores = [d["drift"] for d in drift_scores]
        mid = len(scores) // 2
        first_half_avg = sum(scores[:mid]) / mid if mid > 0 else 0
        second_half_avg = sum(scores[mid:]) / (len(scores) - mid) if len(scores) > mid else 0

        # Raised threshold from 0.2 to 0.3 to reduce false positives
        if second_half_avg > first_half_avg + 0.3:  # Significant increase
            worst_turn = max(drift_scores, key=lambda x: x["drift"])["turn"]
            return {
                "detected": True,
                "worst_turn": worst_turn,
                "drift_scores": drift_scores,
                "first_half_avg": first_half_avg,
                "second_half_avg": second_half_avg,
                "method": "keyword",
            }

        return {"detected": False, "method": "keyword"}

    def _detect_progressive_drift_from_similarities(
        self,
        turns: List[TurnSnapshot],
        similarities: List[float],
    ) -> Dict[str, Any]:
        """Detect progressive drift using pre-computed similarities.

        OPTIMIZATION: This avoids re-encoding texts that were already encoded
        in the main detect() method's batch_semantic_similarity call.
        """
        if len(similarities) < 3 or len(turns) != len(similarities):
            return {"detected": False}

        # Build drift scores from pre-computed similarities
        drift_scores = [
            {"turn": t.turn_number, "drift": 1.0 - sim, "similarity": sim}
            for t, sim in zip(turns, similarities)
        ]

        # Check for progressive drift (similarity decreasing over time)
        mid = len(similarities) // 2
        first_half = similarities[:mid]
        second_half = similarities[mid:]
        first_avg = sum(first_half) / len(first_half) if first_half else 0
        second_avg = sum(second_half) / len(second_half) if second_half else 0

        # 10% degradation threshold for progressive drift
        if second_avg < first_avg - 0.1:
            min_idx = similarities.index(min(similarities))
            worst_turn = turns[min_idx].turn_number

            return {
                "detected": True,
                "worst_turn": worst_turn,
                "drift_scores": drift_scores,
                "avg_similarity": sum(similarities) / len(similarities),
                "method": "embedding_precomputed",
            }

        return {"detected": False, "method": "embedding_precomputed"}

    def _detect_task_substitution(
        self,
        initial_task: str,
        agent_turns: List[TurnSnapshot],
    ) -> Dict[str, Any]:
        """Detect if agent is doing a different but related task."""
        # Common substitution pairs
        substitution_pairs = [
            ("authentication", "authorization"),
            ("encrypt", "decrypt"),
            ("upload", "download"),
            ("create", "delete"),
            ("read", "write"),
            ("frontend", "backend"),
        ]

        task_lower = initial_task.lower()

        for correct, wrong in substitution_pairs:
            if correct in task_lower:
                # Check if any agent turn focuses on wrong task
                for turn in agent_turns:
                    output_lower = turn.content.lower()
                    correct_count = output_lower.count(correct)
                    wrong_count = output_lower.count(wrong)

                    if wrong_count > correct_count + 2:
                        return {
                            "detected": True,
                            "turn": turn.turn_number,
                            "description": f"Agent focused on '{wrong}' instead of '{correct}'",
                        }

        return {"detected": False}
