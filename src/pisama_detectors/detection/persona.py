from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from pisama_detectors._config import get_settings
from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder

settings = get_settings()


class RoleType(Enum):
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    ASSISTANT = "assistant"
    SPECIALIST = "specialist"
    CONVERSATIONAL = "conversational"
    EVALUATOR = "evaluator"


# Thresholds lowered by ~0.07 to improve hard-case detection (borderline drift)
ROLE_THRESHOLDS: Dict[RoleType, Dict[str, float]] = {
    RoleType.CREATIVE: {
        "consistency_threshold": 0.48,
        "drift_threshold": 0.25,
        "flexibility_bonus": 0.15,
    },
    RoleType.ANALYTICAL: {
        "consistency_threshold": 0.68,
        "drift_threshold": 0.12,
        "flexibility_bonus": 0.0,
    },
    RoleType.ASSISTANT: {
        "consistency_threshold": 0.58,
        "drift_threshold": 0.18,
        "flexibility_bonus": 0.08,
    },
    RoleType.SPECIALIST: {
        "consistency_threshold": 0.65,
        "drift_threshold": 0.14,
        "flexibility_bonus": 0.05,
    },
    RoleType.CONVERSATIONAL: {
        "consistency_threshold": 0.51,
        "drift_threshold": 0.22,
        "flexibility_bonus": 0.12,
    },
    RoleType.EVALUATOR: {
        "consistency_threshold": 0.75,  # Stricter — evaluators must maintain judgment standards
        "drift_threshold": 0.10,
        "flexibility_bonus": 0.0,
    },
}

ROLE_KEYWORDS: Dict[RoleType, List[str]] = {
    RoleType.CREATIVE: [
        "writer",
        "artist",
        "creative",
        "storyteller",
        "poet",
        "designer",
        "imaginative",
    ],
    RoleType.ANALYTICAL: ["analyst", "researcher", "data", "scientific", "logical", "statistical"],
    RoleType.ASSISTANT: ["assistant", "helper", "support", "general", "helpful"],
    RoleType.SPECIALIST: ["expert", "specialist", "professional", "domain", "technical"],
    RoleType.CONVERSATIONAL: ["chat", "conversational", "friendly", "casual", "companion"],
    RoleType.EVALUATOR: [
        "evaluator",
        "reviewer",
        "qa",
        "tester",
        "judge",
        "auditor",
        "assessor",
        "critic",
    ],
}


@dataclass
class Agent:
    id: str
    persona_description: str
    allowed_actions: List[str]
    role_type: Optional[RoleType] = None
    custom_thresholds: Optional[Dict[str, float]] = None


@dataclass
class PersonaConsistencyResult:
    consistent: bool
    score: float
    method: str
    drift_detected: bool
    drift_magnitude: Optional[float] = None
    issues: Optional[List[str]] = None
    role_type: Optional[RoleType] = None
    confidence: float = 0.0
    factors: Dict[str, float] = field(default_factory=dict)
    raw_score: Optional[float] = None
    evidence: Optional[Dict] = None


class PersonaConsistencyScorer:
    def __init__(
        self,
        consistency_threshold: Optional[float] = None,
        drift_threshold: Optional[float] = None,
        confidence_scaling: float = 1.0,
    ):
        self._embedder = None
        self.default_consistency_threshold = consistency_threshold or 0.7
        self.default_drift_threshold = drift_threshold or 0.15
        self.confidence_scaling = confidence_scaling
        self._role_embedding_cache: Dict[str, np.ndarray] = {}

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    # Role-domain keyword map: maps persona keywords to expected domain vocabulary
    _ROLE_DOMAIN_KEYWORDS = {
        "legal": [
            "law",
            "statute",
            "section",
            "court",
            "legal",
            "act",
            "regulation",
            "case",
            "ruling",
            "compliance",
        ],
        "code": [
            "code",
            "bug",
            "function",
            "variable",
            "error",
            "sql",
            "api",
            "vulnerability",
            "line",
            "class",
        ],
        "review": [
            "review",
            "found",
            "issue",
            "suggest",
            "improvement",
            "quality",
            "coverage",
            "analysis",
        ],
        "data": [
            "data",
            "statistic",
            "trend",
            "percent",
            "increase",
            "decrease",
            "p-value",
            "revenue",
            "metric",
        ],
        "support": [
            "ticket",
            "issue",
            "resolve",
            "escalate",
            "dns",
            "configuration",
            "troubleshoot",
        ],
        "medical": [
            "symptom",
            "condition",
            "diagnosis",
            "consult",
            "healthcare",
            "professional",
            "specialist",
        ],
        "writer": ["documentation", "api", "endpoint", "parameter", "response", "example", "guide"],
        "schedul": ["calendar", "meeting", "available", "book", "invite", "conference", "slot"],
        "translat": ["translate", "text", "language", "register", "formal", "meaning", "original"],
        "test": ["test", "pass", "fail", "bug", "coverage", "reproduction", "suite", "assertion"],
        "research": ["study", "paper", "found", "model", "accuracy", "benchmark", "contribution"],
        "security": [
            "security",
            "vulnerability",
            "exploit",
            "auth",
            "permission",
            "injection",
            "xss",
        ],
        # Fiction/narrative writing: match persona descriptions with "stor..."
        # (story/storyteller/stories) or "narrat..." (narrative/narrator).
        # Looks for prose markers in output — past-tense verbs, sensory nouns,
        # character references — that characterize narrative writing.
        "stor": [
            "character",
            "scene",
            "story",
            "narrative",
            "plot",
            "dialogue",
            "setting",
            "imagery",
            "description",
            "moment",
            "emotion",
            "memory",
            "night",
            "morning",
            "evening",
            "years",
            "stood",
            "gazed",
            "watched",
            "felt",
            "saw",
            "heard",
            "looked",
            "hands",
            "eyes",
            "face",
            "heart",
            "mind",
        ],
        "narrat": [
            "character",
            "scene",
            "story",
            "narrative",
            "plot",
            "dialogue",
            "setting",
            "imagery",
            "description",
            "moment",
            "emotion",
            "stood",
            "gazed",
            "watched",
            "felt",
            "saw",
            "heard",
            "looked",
            "hands",
            "eyes",
            "face",
            "heart",
            "mind",
        ],
    }

    def _compute_role_action_relevance(self, persona_desc: str, output: str) -> float:
        """Check if output demonstrates domain-appropriate actions for the persona.

        Returns 0.0-1.0 where higher means the output is relevant to the role.
        """
        persona_lower = persona_desc.lower()
        output_lower = output.lower()

        best_match = 0.0
        for role_keyword, domain_terms in self._ROLE_DOMAIN_KEYWORDS.items():
            if role_keyword in persona_lower:
                matches = sum(1 for t in domain_terms if t in output_lower)
                relevance = min(1.0, matches / 3.0)  # 3+ domain terms = full match
                best_match = max(best_match, relevance)

        return best_match

    def _detect_evaluator_leniency(self, output: str) -> float:
        """Detect evaluator leniency: approving work despite identified issues.

        Returns a score 0.0-1.0 where higher = more lenient.
        """
        import re

        lower = output.lower()

        # Count problem identifications
        problem_words = [
            "bug",
            "error",
            "issue",
            "problem",
            "fail",
            "broken",
            "missing",
            "incorrect",
            "wrong",
            "defect",
            "flaw",
        ]
        problem_count = sum(1 for w in problem_words if w in lower)

        # Count approval signals
        approval_patterns = [
            r"\bpass(?:ed|ing|es)?\b",
            r"\bapprov(?:ed|ing|e)\b",
            r"\blooks?\s+good\b",
            r"\boverall\s+(?:good|great|acceptable|satisfactory)\b",
            r"\bship\s+it\b",
            r"\bready\s+(?:for|to)\b",
            r"\bno\s+(?:major|critical|blocking)\s+issues?\b",
            r"\bmeets?\s+(?:the\s+)?requirements?\b",
        ]
        approval_count = sum(1 for p in approval_patterns if re.search(p, lower))

        # v2.3: Raised thresholds — 3 problems + 1 approval is normal
        # balanced review, not leniency. Require stronger signal.
        if problem_count >= 5 and approval_count >= 2:
            return min((problem_count * 0.10 + approval_count * 0.20), 1.0)
        elif problem_count >= 4 and approval_count >= 3:
            return min((problem_count * 0.08 + approval_count * 0.22), 1.0)
        return 0.0

    def _calibrate_confidence(
        self,
        semantic_sim: float,
        lexical_overlap: float,
        tone_score: float,
        output_length: int,
        drift_detected: bool = False,
        role_action_boost: float = 0.0,
    ) -> float:
        """Calibrate confidence based on evidence quality.

        Returns high confidence for consistent personas (so drift confidence
        is low when inverted), and lower confidence when drift is detected
        (so drift confidence is higher when inverted).

        v2.3: Include role_action_boost in calibration — high action
        relevance should reduce drift confidence even if semantic sim is low.
        """
        if drift_detected:
            # Drift detected → confidence of drift = 1 - weighted_score
            base_confidence = 1.0 - (
                semantic_sim * 0.4
                + lexical_overlap * 0.2
                + tone_score * 0.15
                + role_action_boost * 0.25
            )
        else:
            # Consistent → confidence of drift is low (= score itself)
            base_confidence = (
                semantic_sim * 0.4
                + lexical_overlap * 0.2
                + tone_score * 0.15
                + role_action_boost * 0.25
            )

        calibrated = min(0.99, base_confidence * self.confidence_scaling)
        return round(calibrated, 4)

    def _detect_role_type(self, persona_description: str) -> RoleType:
        """Auto-detect role type from persona description."""
        desc_lower = persona_description.lower()

        scores = {}
        for role_type, keywords in ROLE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            scores[role_type] = score

        if max(scores.values()) == 0:
            return RoleType.ASSISTANT

        return max(scores, key=scores.get)

    def _get_thresholds(self, agent: Agent) -> Dict[str, float]:
        """Get role-specific thresholds for the agent."""
        if agent.custom_thresholds:
            return agent.custom_thresholds

        role_type = agent.role_type or self._detect_role_type(agent.persona_description)
        return ROLE_THRESHOLDS.get(role_type, ROLE_THRESHOLDS[RoleType.ASSISTANT])

    def _compute_semantic_similarity(self, agent: Agent, output: str) -> float:
        """Compute semantic similarity with caching."""
        if self.embedder is None:
            return self._compute_lexical_overlap(agent.persona_description, output)

        cache_key = f"{agent.id}:{agent.persona_description[:50]}"

        if cache_key not in self._role_embedding_cache:
            self._role_embedding_cache[cache_key] = self.embedder.encode(agent.persona_description)

        persona_embedding = self._role_embedding_cache[cache_key]
        output_embedding = self.embedder.encode(output)

        return float(self.embedder.similarity(persona_embedding, output_embedding))

    def _compute_lexical_overlap(self, persona: str, output: str) -> float:
        """Compute lexical overlap as secondary signal."""
        persona_words = set(persona.lower().split())
        output_words = set(output.lower().split())

        if not persona_words:
            return 0.0

        overlap = len(persona_words & output_words)
        return min(1.0, overlap / (len(persona_words) * 0.3))

    def _compute_tone_consistency(self, output: str, expected_tone: str = "neutral") -> float:
        """Check if output matches expected tone."""
        formal_markers = ["therefore", "consequently", "furthermore", "regarding"]
        casual_markers = ["hey", "yeah", "cool", "awesome", "lol"]

        output_lower = output.lower()
        formal_score = sum(1 for m in formal_markers if m in output_lower)
        casual_score = sum(1 for m in casual_markers if m in output_lower)

        if expected_tone == "formal":
            return min(1.0, 0.5 + formal_score * 0.1 - casual_score * 0.15)
        elif expected_tone == "casual":
            return min(1.0, 0.5 + casual_score * 0.1 - formal_score * 0.15)

        return 0.7

    def score_consistency(
        self,
        agent: Agent,
        output: str,
        recent_outputs: Optional[List[str]] = None,
    ) -> PersonaConsistencyResult:
        thresholds = self._get_thresholds(agent)
        consistency_threshold = thresholds["consistency_threshold"]
        drift_threshold = thresholds["drift_threshold"]
        flexibility_bonus = thresholds.get("flexibility_bonus", 0.0)

        role_type = agent.role_type or self._detect_role_type(agent.persona_description)

        semantic_sim = self._compute_semantic_similarity(agent, output)
        lexical_overlap = self._compute_lexical_overlap(agent.persona_description, output)
        tone_score = self._compute_tone_consistency(output)

        raw_score = semantic_sim

        # v2.2: Role-action relevance — if the output demonstrates
        # domain-appropriate actions, the agent IS on-persona even when
        # embedding similarity is low (short persona descriptions have
        # unreliable semantic similarity to domain-specific outputs).
        role_action_boost = self._compute_role_action_relevance(agent.persona_description, output)

        # v2.3: Weight role-action relevance higher than semantic similarity.
        # Short persona descriptions (< 15 words) have unreliable embedding
        # similarity — domain-action signals are more reliable in that case.
        persona_word_count = len(agent.persona_description.split())
        if persona_word_count < 15 and role_action_boost > 0.3:
            # Short persona + strong action match: trust actions over embeddings
            weighted_score = (
                semantic_sim * 0.20
                + lexical_overlap * 0.20
                + tone_score * 0.15
                + role_action_boost * 0.45
                + flexibility_bonus
            )
        else:
            weighted_score = (
                semantic_sim * 0.30
                + lexical_overlap * 0.20
                + tone_score * 0.15
                + role_action_boost * 0.35
                + flexibility_bonus
            )
        weighted_score = min(1.0, weighted_score)

        factors = {
            "semantic_similarity": round(semantic_sim, 4),
            "lexical_overlap": round(lexical_overlap, 4),
            "tone_consistency": round(tone_score, 4),
            "role_action_boost": round(role_action_boost, 4),
            "flexibility_bonus": flexibility_bonus,
        }

        drift_detected = False
        drift_magnitude = None

        if recent_outputs and len(recent_outputs) >= 3:
            recent_embeddings = self.embedder.encode(recent_outputs)
            avg_recent = np.mean(recent_embeddings, axis=0)
            output_embedding = self.embedder.encode(output)
            drift_magnitude = float(1 - self.embedder.similarity(avg_recent, output_embedding))

            adjusted_drift_threshold = drift_threshold
            if role_type == RoleType.CREATIVE:
                adjusted_drift_threshold *= 1.3

            drift_detected = drift_magnitude > adjusted_drift_threshold
            factors["drift_magnitude"] = round(drift_magnitude, 4)
        else:
            # Without recent_outputs, fall back to score-based drift detection
            # so that drift_detected is consistent with the `consistent` flag.
            drift_detected = weighted_score <= consistency_threshold

        # v2.1: Evaluator leniency detection — if agent has evaluator role
        # and output approves/passes despite identifying problems, flag as drift.
        # From Anthropic: evaluators "identify legitimate issues, then talk
        # themselves into deciding they weren't a big deal."
        evaluator_leniency = 0.0
        if role_type == RoleType.EVALUATOR:
            evaluator_leniency = self._detect_evaluator_leniency(output)
            factors["evaluator_leniency"] = round(evaluator_leniency, 4)
            if evaluator_leniency >= 0.5:
                drift_detected = True
                # Reduce weighted score to reflect the leniency-as-drift
                weighted_score = min(weighted_score, 1.0 - evaluator_leniency * 0.3)

        confidence = self._calibrate_confidence(
            semantic_sim=semantic_sim,
            lexical_overlap=lexical_overlap,
            tone_score=tone_score,
            output_length=len(output),
            drift_detected=drift_detected,
            role_action_boost=role_action_boost,
        )

        evidence = {
            "output_length": len(output),
            "consistency_threshold": consistency_threshold,
            "role_type": role_type.value if role_type else None,
        }
        if evaluator_leniency > 0:
            evidence["evaluator_leniency"] = evaluator_leniency

        if weighted_score > consistency_threshold and not drift_detected:
            return PersonaConsistencyResult(
                consistent=True,
                score=float(weighted_score),
                method="multi_factor_scoring",
                drift_detected=False,
                role_type=role_type,
                confidence=confidence,
                factors=factors,
                raw_score=raw_score,
                evidence=evidence,
            )

        issues = []
        if weighted_score < consistency_threshold:
            issues.append(
                f"Output deviates from persona (score: {weighted_score:.2f}, threshold: {consistency_threshold:.2f})"
            )
        if drift_detected:
            if drift_magnitude is not None:
                issues.append(f"Persona drift detected (magnitude: {drift_magnitude:.2f})")
            else:
                issues.append(
                    f"Persona drift detected (score below threshold: {weighted_score:.2f})"
                )

        return PersonaConsistencyResult(
            consistent=weighted_score > consistency_threshold and not drift_detected,
            score=float(weighted_score),
            method="multi_factor_scoring_with_drift",
            drift_detected=drift_detected,
            drift_magnitude=drift_magnitude,
            issues=issues if issues else None,
            role_type=role_type,
            confidence=confidence,
            factors=factors,
            raw_score=raw_score,
            evidence=evidence,
        )

    def detect_role_usurpation(
        self,
        agent: Agent,
        output: str,
        all_agents: List[Agent],
    ) -> Optional[str]:
        agent_embedding = self.embedder.encode(agent.persona_description)
        output_embedding = self.embedder.encode(output)

        own_similarity = self.embedder.similarity(agent_embedding, output_embedding)

        for other_agent in all_agents:
            if other_agent.id == agent.id:
                continue

            other_embedding = self.embedder.encode(other_agent.persona_description)
            other_similarity = self.embedder.similarity(other_embedding, output_embedding)

            if other_similarity > own_similarity + 0.1:
                return other_agent.id

        return None


persona_scorer = PersonaConsistencyScorer()
