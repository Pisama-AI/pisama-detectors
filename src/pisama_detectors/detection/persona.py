import math
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from pisama_detectors._config import get_settings
from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder

settings = get_settings()


_FINGERPRINT_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "at",
    "for", "with", "by", "from", "as", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "them", "him", "her",
    "his", "hers", "their", "our", "your", "my", "me", "us", "do", "does",
    "did", "have", "has", "had", "will", "would", "should", "could", "can",
    "may", "might", "must", "not", "no", "yes", "so", "than", "too", "very",
    "just", "also", "there", "here", "what", "which", "who", "whom", "how",
    "why", "when", "where", "all", "any", "some", "few", "more", "most",
    "other", "such", "only", "own", "same", "out", "up", "down", "over",
}

_FORMAL_MARKERS = {"therefore", "consequently", "furthermore", "moreover",
                   "regarding", "accordingly", "henceforth", "thus"}
_CASUAL_MARKERS = {"yeah", "gonna", "kinda", "wanna", "lol", "hey",
                   "cool", "awesome", "stuff", "dude"}
_TECHNICAL_MARKERS = {"function", "parameter", "variable", "module",
                      "method", "class", "endpoint", "instance", "argument"}
_EMOTIVE_MARKERS = {"love", "hate", "amazing", "terrible", "wonderful",
                    "horrible", "fantastic", "awful", "incredible"}

# Task-relevance gate (single-turn false-positive guard). A short, domain-defined
# persona answering an in-domain request gives a factual answer whose embedding
# sits far from the persona instruction, so a perfectly on-persona single-turn
# reply scored as drift (the benign single-turn FP). When the request is in the
# persona's domain AND the output is a genuine, full answer to it, that low
# persona<->output similarity is topical/format divergence, not drift.
#
# Two SEPARATE conjunctive gates, each guarding a distinct failure mode (a single
# min() of the two legs is not enough — they separate at different scales):
#   * persona<->task >= _TASK_DOMAIN_GATE: the request is in the persona's
#     domain. Excludes wrong-persona drift (a legal bot answering a weather
#     question answers it well, so task<->output is high — only the low
#     persona<->task reveals the drift). Benign in-domain Q&A ~0.35-0.40 vs a
#     wrong-persona request ~0.03.
#   * task<->output >= _TASK_ANSWER_GATE: the output genuinely, fully answers the
#     request. Excludes "answer-then-drift" / topic-adjacent drift, where the
#     output abandons the role while staying near the request's topic (e.g. a
#     banking bot replying "banks are robbing you, move to crypto"): such outputs
#     keep task<->output well below the gate (verified-drift corpus tops out at
#     ~0.45) while a real answer lands ~0.68-0.82.
# Both gates clear with wide margin on both classes. INERT when no task is
# supplied (the external calibration lane passes none), so recall there is
# unchanged.
_TASK_DOMAIN_GATE = 0.15
_TASK_ANSWER_GATE = 0.55
_TASK_RELEVANCE_SCALE = 0.85

# v2.6 off-role excursion guard. The persona-compliance gates (role-action /
# style / task-relevance) each suppress drift when ONE surface signal is high.
# An adversary defeats them by keeping that surface (answer the task, hold the
# register, pack in role vocab) while pivoting to an off-role goal — the
# "answer-then-drift" family surfaced by the adversarial probe. The tell is a
# substantial sentence far from BOTH the persona AND the task: a genuine on-task
# answer has no such segment. When found, the compliance signals are being gamed,
# so they no longer launder the drift (persona_compliance is dropped to 0, letting
# the real semantic/lexical signal decide). Gated on a task being present (inert
# on the external calibration lane, like task_relevance itself), so the existing
# labelled positives and the golden suite are unaffected.
#
# Gate 0.14: on the adversarial probe, benign on-persona sentences bottom out at
# max-sim ~0.23 (a few formatted fragments at 0.07-0.15) while real drift
# excursions cluster at 0.00-0.22, so 0.14 sits under the benign floor. Measured
# effect on the probe (n=53 confirmed evasions / n=18 on-persona benigns):
# evade-rate 0.79 -> 0.70, benign FP 0.056 -> 0.056 (no regression), seed recall
# 1.0. This is a deliberately conservative, precision-FIRST heuristic: it removes
# the laundering without forcing fires, so it under-claims (the residual ~0.70
# evade-rate is the embedding ceiling — answer-then-drift and benign on-persona
# replies overlap in embedding space). Closing the residual needs the LLM-judge
# tier (escalate the abstention band to a judge), not a tighter gate. Tuned on a
# small benign sample; widen it before raising the gate.
_OFFROLE_PERSONA_GATE = 0.14
_OFFROLE_TASK_GATE = 0.14
_OFFROLE_MIN_WORDS = 6
_OFFROLE_MAX_SENTENCES = 8


@dataclass
class BehavioralFingerprint:
    vocab: Dict[str, float]
    sentence_len_mean: float
    sentence_len_std: float
    message_len_mean: float
    message_len_std: float
    register: Dict[str, float]
    punctuation: Dict[str, float]


def _tokenize_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-z']+", text.lower()) if w not in _FINGERPRINT_STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def _extract_fingerprint(text_window: List[str], top_k: int = 50) -> BehavioralFingerprint:
    """Extract behavioral fingerprint over a window of outputs."""
    all_words: List[str] = []
    sentence_lens: List[int] = []
    message_lens: List[int] = []
    punct_counts = {"?": 0, "!": 0, ";": 0, ":": 0, "...": 0}

    for msg in text_window:
        words = _tokenize_words(msg)
        all_words.extend(words)
        message_lens.append(len(msg.split()))
        for sent in _split_sentences(msg):
            sentence_lens.append(len(sent.split()))
        punct_counts["..."] += len(re.findall(r"\.{3,}", msg))
        no_ellipsis = re.sub(r"\.{3,}", " ", msg)
        punct_counts["?"] += no_ellipsis.count("?")
        punct_counts["!"] += no_ellipsis.count("!")
        punct_counts[";"] += no_ellipsis.count(";")
        punct_counts[":"] += no_ellipsis.count(":")

    total_words = max(1, len(all_words))
    counter = Counter(all_words).most_common(top_k)
    vocab_total = sum(c for _, c in counter) or 1
    vocab = {w: c / vocab_total for w, c in counter}

    register_words = [w.lower() for w in all_words]
    word_set = Counter(register_words)
    register = {
        "formal": sum(word_set[m] for m in _FORMAL_MARKERS) / total_words,
        "casual": sum(word_set[m] for m in _CASUAL_MARKERS) / total_words,
        "technical": sum(word_set[m] for m in _TECHNICAL_MARKERS) / total_words,
        "emotive": sum(word_set[m] for m in _EMOTIVE_MARKERS) / total_words,
    }

    per_100 = 100.0 / total_words
    punctuation = {k: v * per_100 for k, v in punct_counts.items()}

    s_arr = np.asarray(sentence_lens, dtype=float) if sentence_lens else np.asarray([0.0])
    m_arr = np.asarray(message_lens, dtype=float) if message_lens else np.asarray([0.0])

    return BehavioralFingerprint(
        vocab=vocab,
        sentence_len_mean=float(s_arr.mean()),
        sentence_len_std=float(s_arr.std()),
        message_len_mean=float(m_arr.mean()),
        message_len_std=float(m_arr.std()),
        register=register,
        punctuation=punctuation,
    )


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Symmetric Jensen-Shannon divergence over two sparse distributions, normalized to [0, 1]."""
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    js = 0.0
    for k in keys:
        pi = p.get(k, 0.0)
        qi = q.get(k, 0.0)
        m = 0.5 * (pi + qi)
        if m <= 0:
            continue
        if pi > 0:
            js += 0.5 * pi * math.log(pi / m, 2)
        if qi > 0:
            js += 0.5 * qi * math.log(qi / m, 2)
    return max(0.0, min(1.0, js))


def _bounded_diff(a: float, b: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return min(1.0, abs(a - b) / scale)


def _l1_rate_distance(p: Dict[str, float], q: Dict[str, float], scale: float) -> float:
    keys = set(p) | set(q)
    if not keys or scale <= 0:
        return 0.0
    diff = sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)
    return min(1.0, diff / scale)


def _compute_fingerprint_drift_score(
    outputs: List[str],
    window_size: int = 5,
) -> Tuple[float, BehavioralFingerprint, BehavioralFingerprint]:
    """Compute behavioral fingerprint drift between baseline and current windows."""
    if not outputs or len(outputs) < 2:
        empty = _extract_fingerprint([])
        return 0.0, empty, empty

    half = max(1, min(window_size, len(outputs) // 2))
    baseline_window = outputs[:half]
    current_window = outputs[-half:]

    if baseline_window == current_window:
        empty = _extract_fingerprint([])
        return 0.0, empty, empty

    baseline = _extract_fingerprint(baseline_window)
    current = _extract_fingerprint(current_window)

    vocab_drift = _js_divergence(baseline.vocab, current.vocab)
    # Register: L1 distance on marker-rate vectors, scaled so a full register
    # flip (0.20 swing across markers) saturates to 1.0.
    register_drift = _l1_rate_distance(baseline.register, current.register, 0.20)

    sent_drift = _bounded_diff(baseline.sentence_len_mean, current.sentence_len_mean, 20.0)
    msg_drift = _bounded_diff(baseline.message_len_mean, current.message_len_mean, 80.0)
    punct_keys = set(baseline.punctuation) | set(current.punctuation)
    punct_drift = sum(
        _bounded_diff(baseline.punctuation.get(k, 0.0), current.punctuation.get(k, 0.0), 5.0)
        for k in punct_keys
    ) / max(1, len(punct_keys))

    score = (
        vocab_drift * 0.20
        + register_drift * 0.45
        + sent_drift * 0.05
        + msg_drift * 0.05
        + punct_drift * 0.25
    )
    return min(1.0, score), baseline, current


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

    # Register/tone descriptors a persona may DECLARE, mapped to the output
    # markers that DEMONSTRATE that register ("comply") and the markers that
    # BREAK it ("violate"). A persona defined by its style ("formal Professional
    # Assistant") has no domain, so _compute_role_action_relevance returns 0 for
    # it; staying in the declared register is the equivalent on-persona signal.
    _STYLE_REGISTERS = {
        "formal": {
            "declare": ("formal", "professional", "polite", "courteous", "businesslike"),
            "comply": (
                "good morning", "good afternoon", "good evening", "how may i",
                "may i assist", "assist you", "happy to assist", "pleased to",
                "certainly", "kindly", "please", "thank you", "regarding",
                "professionally", "i would be", "shall ", "regards",
            ),
            "violate": (
                "hey", "yeah", "yo ", "sup", "gonna", "wanna", "kinda",
                "lol", "lmao", "haha", "dude", "omg", "ur ", "u r ",
            ),
        },
        "casual": {
            "declare": ("casual", "informal", "friendly", "conversational", "relaxed", "playful"),
            "comply": (
                "hey", "hi ", "yeah", "sure thing", "no worries", "cool",
                "awesome", "gonna", "let's", "thanks", "happy to help",
            ),
            "violate": (
                "furthermore", "consequently", "heretofore", "pursuant",
                "to whom it may concern",
            ),
        },
    }

    def _compute_style_compliance(self, persona_desc: str, output: str) -> float:
        """Score how well the output honors a register/tone DECLARED by the persona.

        Returns 0.0-1.0. Returns 0.0 when the persona declares no register (the
        common case — most personas are domain- or role-defined), so this signal
        is inert unless the persona explicitly specifies a style. If the output
        contains markers that BREAK the declared register, that register scores
        0 (a "formal" agent slipping into slang is drift, not compliance).
        """
        persona_lower = persona_desc.lower()
        output_lower = output.lower()

        best = 0.0
        for register in self._STYLE_REGISTERS.values():
            if not any(d in persona_lower for d in register["declare"]):
                continue
            if any(v in output_lower for v in register["violate"]):
                continue  # output breaks the declared register -> not compliance
            comply = sum(1 for m in register["comply"] if m in output_lower)
            best = max(best, min(1.0, comply / 3.0))  # 3+ register markers = full match
        return best

    def _compute_task_relevance(
        self, persona_desc: str, task: str, output: str
    ) -> float:
        """Score whether the output is an on-persona answer to the user's request.

        Returns 0.0-1.0. Returns 0.0 when no task is supplied (the common
        external-lane case, keeping the signal inert there) or the task is too
        thin to trust. Otherwise it is high only when BOTH the request is in the
        persona's domain (persona<->task) AND the output is responsive to that
        request (task<->output) — a relevant answer to an in-domain question is
        on-persona by construction, so the otherwise-unreliable instruction-vs-
        content embedding gap should not read as drift. See _TASK_RELEVANCE_GATE.
        """
        if not task or len(task.split()) < 2:
            return 0.0

        task_emb = self.embedder.encode(task)
        persona_emb = self.embedder.encode(persona_desc)
        output_emb = self.embedder.encode(output)

        persona_task = float(self.embedder.similarity(persona_emb, task_emb))
        task_output = float(self.embedder.similarity(task_emb, output_emb))

        # Request must be in the persona's domain AND the output must genuinely
        # answer it. Either gate failing earns zero relevance.
        if persona_task < _TASK_DOMAIN_GATE or task_output < _TASK_ANSWER_GATE:
            return 0.0
        # Relevance scales with how fully the output answers the request.
        return min(1.0, task_output / _TASK_RELEVANCE_SCALE)

    def _has_offrole_excursion(self, persona_desc: str, task: Optional[str], output: str) -> bool:
        """True when a substantial sentence is far from BOTH the persona and the
        task — the answer-then-drift signature. A benign on-task answer stays
        close to the task throughout; an evasion that answers then pivots leaves
        a segment far from both. Inert without a task or a multi-sentence output.
        """
        if not task:
            return False
        sentences = [
            s for s in _split_sentences(output) if len(s.split()) >= _OFFROLE_MIN_WORDS
        ]
        if len(sentences) < 2:
            return False
        persona_emb = self.embedder.encode(persona_desc)
        task_emb = self.embedder.encode(task)
        for sent in sentences[:_OFFROLE_MAX_SENTENCES]:
            sent_emb = self.embedder.encode(sent)
            sim_persona = float(self.embedder.similarity(sent_emb, persona_emb))
            sim_task = float(self.embedder.similarity(sent_emb, task_emb))
            if sim_persona < _OFFROLE_PERSONA_GATE and sim_task < _OFFROLE_TASK_GATE:
                return True
        return False

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

        scores: Dict[RoleType, int] = {}
        for role_type, keywords in ROLE_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in desc_lower)
            scores[role_type] = score

        if max(scores.values()) == 0:
            return RoleType.ASSISTANT

        return max(scores, key=scores.__getitem__)

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
        output_history: Optional[List[str]] = None,
        fingerprint_window: int = 5,
        task: Optional[str] = None,
    ) -> PersonaConsistencyResult:
        # --- EXCLUSION rule (benign-by-construction) ---
        # TN: empty / blank persona_description => no persona was assigned, so
        # there is no baseline to drift FROM. Persona drift measures deviation
        # from an assigned persona; with none, drift is undefined, not maximal.
        # Firing here was a precision bug (the empty-persona FP source). FP-hunt
        # lane synthetic_negative_v2: 27/27 verified-benign rows fired pre-guard.
        # Recall is unaffected — no labelled positive scores with an empty persona.
        if not (agent.persona_description or "").strip():
            return PersonaConsistencyResult(
                consistent=True,
                score=1.0,
                method="exclusion_empty_persona",
                drift_detected=False,
                drift_magnitude=None,
                issues=None,
                role_type=None,
                confidence=0.0,
                factors={},
                raw_score=None,
                evidence={"reason": "no persona_description assigned — nothing to drift from"},
            )

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

        # v2.4: Style/register compliance. A persona defined by its register
        # ("Professional Assistant with formal style") has no domain, so
        # role_action_boost is 0 for it, and JSON-wrapped output drags down both
        # embedding similarity and lexical overlap — leaving a compliant formal
        # response scored as drift (the residual real-data FP). Staying in the
        # declared register is the style-persona analog of domain-action
        # relevance, so it feeds the same persona-compliance slot via max().
        # Inert for personas that declare no style (role_action_boost stands
        # alone), so domain/role-defined personas — and every labelled positive,
        # none of which declare a register — are unaffected.
        style_compliance = self._compute_style_compliance(agent.persona_description, output)

        # v2.5: Task relevance. A short, domain-defined persona ("You are a
        # helpful weather assistant") answering an in-domain request gives a
        # factual answer whose embedding sits far from the persona instruction —
        # low semantic_sim, low lexical overlap, no declared register — so a
        # perfectly on-persona single-turn reply scored as drift (the benign
        # single-turn FP, surfaced via the fix-efficacy rerun shape). When the
        # user task is available and the output is a relevant answer to an
        # in-domain request, that IS on-persona evidence, so it feeds the same
        # persona-compliance slot via max(). Inert when no task is supplied
        # (the external calibration lane), so labelled positives are unaffected.
        task_relevance = self._compute_task_relevance(
            agent.persona_description, task, output
        ) if task else 0.0
        persona_compliance = max(role_action_boost, style_compliance, task_relevance)

        # v2.6: a compliant surface (on-task / in-register / role-vocab) cannot
        # launder a substantial off-role excursion. When one is present, the
        # compliance signals are being gamed, so they no longer suppress the
        # drift (the answer-then-drift evasion family).
        offrole_excursion = False
        if persona_compliance > 0 and self._has_offrole_excursion(
            agent.persona_description, task, output
        ):
            offrole_excursion = True
            persona_compliance = 0.0

        # v2.3: Weight persona-compliance higher than semantic similarity.
        # Short persona descriptions (< 15 words) have unreliable embedding
        # similarity — compliance signals are more reliable in that case.
        persona_word_count = len(agent.persona_description.split())
        if persona_word_count < 15 and persona_compliance > 0.3:
            # Short persona + strong compliance: trust compliance over embeddings
            weighted_score = (
                semantic_sim * 0.20
                + lexical_overlap * 0.20
                + tone_score * 0.15
                + persona_compliance * 0.45
                + flexibility_bonus
            )
        else:
            weighted_score = (
                semantic_sim * 0.30
                + lexical_overlap * 0.20
                + tone_score * 0.15
                + persona_compliance * 0.35
                + flexibility_bonus
            )
        weighted_score = min(1.0, weighted_score)

        factors = {
            "semantic_similarity": round(semantic_sim, 4),
            "lexical_overlap": round(lexical_overlap, 4),
            "tone_consistency": round(tone_score, 4),
            "role_action_boost": round(role_action_boost, 4),
            "style_compliance": round(style_compliance, 4),
            "task_relevance": round(task_relevance, 4),
            "persona_compliance": round(persona_compliance, 4),
            "offrole_excursion": 1.0 if offrole_excursion else 0.0,
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

        # Behavioral fingerprint drift: black-box output-distribution shift
        # over a window of agent outputs. Catches paraphrased violations and
        # register changes that text-pattern signals miss.
        history = output_history if output_history is not None else recent_outputs
        fingerprint_score = 0.0
        fingerprint_issue: Optional[str] = None
        if history and len(history) >= 2:
            window = list(history) + [output]
            fingerprint_score, _, _ = _compute_fingerprint_drift_score(
                window, window_size=fingerprint_window
            )
            factors["fingerprint_drift"] = round(fingerprint_score, 4)
            if fingerprint_score >= 0.20:
                drift_detected = True
                fingerprint_issue = (
                    f"[fingerprint_drift] Behavioral fingerprint shift detected "
                    f"(JS divergence: {fingerprint_score:.2f})"
                )
                # Fuse via max in drift space: final consistency = min of the two.
                weighted_score = min(weighted_score, 1.0 - fingerprint_score)

        confidence = self._calibrate_confidence(
            semantic_sim=semantic_sim,
            lexical_overlap=lexical_overlap,
            tone_score=tone_score,
            output_length=len(output),
            drift_detected=drift_detected,
            role_action_boost=persona_compliance,
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
        if fingerprint_issue:
            issues.append(fingerprint_issue)

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
