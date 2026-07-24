"""
Loop Detection for Multi-Agent Systems (MAST F8)
=================================================

Detects infinite loops and repetitive patterns in agent behavior.

Version History:
- v1.0: Initial implementation with structural, hash, and semantic detection
- v1.1: Added semantic clustering for paraphrased loops
- v1.2: Added summary/recap whitelisting to reduce false positives
  - Agents summarizing their work shouldn't be flagged as loops
  - Recap/review patterns are benign repetition
"""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import List, Optional

from pisama_detectors._config import get_settings, get_tenant_thresholds
from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder

# Detector version
DETECTOR_VERSION = "1.6"

# v1.6: Monotonic-counter keys that are bookkeeping, not work progress.
# When these are the ONLY strictly-distinct keys, the agent is still looping
# on the same work (same tool call, same error) — don't short-circuit.
_BOOKKEEPING_KEYS = frozenset(
    {
        "iteration_count",
        "iteration_index",
        "iteration",
        "turn_count",
        "turn",
        "step",
        "superstep",
        "retry",
        "retry_count",
        "retries",
        "attempt",
        "attempts",
        "format_attempts",
        "execution_time_ms",
        "timestamp_ms",
        "timestamp",
        "sequence_num",
        "sequence",
        "seq",
    }
)

settings = get_settings()

# v1.2: Summary/recap patterns that indicate benign repetition, not loops
SUMMARY_WHITELIST_PATTERNS = [
    r"\b(?:to summarize|in summary|summarizing|to recap|recapping)\b",
    r"\b(?:so far|thus far|up to this point|at this point)\b",
    r"\b(?:what we've done|what i've done|what has been done)\b",
    r"\b(?:reviewing|let me review|to review)\b",
    r"\b(?:accomplishments|completed so far|progress report)\b",
    r"\b(?:status update|current status|where we are)\b",
    r"\b(?:here's what|here is what)\s+(?:we've|i've|has been)\b",
    r"\b(?:quick recap|brief summary|overview of)\b",
    r"\b(?:wrapping up|to wrap up|in conclusion)\b",
    r"\b(?:let me go over|going over what)\b",
]

# v1.2: Progress reporting patterns
PROGRESS_WHITELIST_PATTERNS = [
    r"\b(?:step \d+ of \d+|task \d+ of \d+)\b",
    r"\b(?:phase \d+|iteration \d+|round \d+)\b",
    r"\b(?:checkpoint|milestone|progress)\b",
    r"\b(?:moving on to|proceeding to|next up)\b",
    r"\b(?:completed step \d+|finished step \d+)\b",
]


@dataclass
class LoopDetectionResult:
    detected: bool
    confidence: float
    method: Optional[str]
    cost: float
    loop_start_index: Optional[int] = None
    loop_length: Optional[int] = None
    raw_score: Optional[float] = None
    evidence: Optional[dict] = None
    framework: Optional[str] = None  # Framework used for detection thresholds


@dataclass
class StateSnapshot:
    agent_id: str
    state_delta: dict
    content: str
    sequence_num: int


class MultiLevelLoopDetector:
    def __init__(
        self,
        structural_threshold: Optional[float] = None,
        semantic_threshold: Optional[float] = None,
        window_size: Optional[int] = None,
        min_matches_for_loop: Optional[int] = None,
        confidence_scaling: Optional[float] = None,
        framework: Optional[str] = None,
        tenant_settings: Optional[dict] = None,
    ):
        self._embedder = None
        self.framework = framework
        self.tenant_settings = tenant_settings

        # Get thresholds with tenant overrides if available
        if tenant_settings or framework:
            thresholds = get_tenant_thresholds(tenant_settings, framework)
            self.structural_threshold = structural_threshold or thresholds.structural_threshold
            self.semantic_threshold = semantic_threshold or thresholds.semantic_threshold
            self.window_size = window_size or thresholds.loop_detection_window
            self.min_matches_for_loop = min_matches_for_loop or thresholds.min_matches_for_loop
            self.confidence_scaling = confidence_scaling or thresholds.confidence_scaling
        else:
            # Fall back to global settings
            self.structural_threshold = structural_threshold or settings.structural_threshold
            self.semantic_threshold = semantic_threshold or settings.semantic_threshold
            self.window_size = window_size or settings.loop_detection_window
            self.min_matches_for_loop = min_matches_for_loop or 2
            self.confidence_scaling = confidence_scaling or 1.0

    @classmethod
    def for_framework(cls, framework: str) -> "MultiLevelLoopDetector":
        """Create a detector configured for a specific framework.

        Args:
            framework: Framework name (langgraph, autogen, crewai, etc.)

        Returns:
            MultiLevelLoopDetector with framework-specific thresholds
        """
        return cls(framework=framework)

    @classmethod
    def for_tenant(
        cls, tenant_settings: Optional[dict], framework: Optional[str] = None
    ) -> "MultiLevelLoopDetector":
        """Create a detector configured for a specific tenant.

        Uses tenant-specific threshold overrides merged with framework defaults.

        Args:
            tenant_settings: Tenant's settings dict (from tenant.settings)
            framework: Framework name for framework-specific defaults

        Returns:
            MultiLevelLoopDetector with tenant-specific thresholds
        """
        return cls(framework=framework, tenant_settings=tenant_settings)

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def _is_summary_or_progress(self, text: str) -> bool:
        """
        v1.2: Check if text is a summary/recap or progress report.

        These patterns indicate benign repetition (agent reviewing work)
        rather than an actual loop.
        """
        text_lower = text.lower()

        for pattern in SUMMARY_WHITELIST_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        for pattern in PROGRESS_WHITELIST_PATTERNS:
            if re.search(pattern, text_lower):
                return True

        return False

    def _has_summary_pattern_in_window(self, states: List["StateSnapshot"]) -> bool:
        """
        v1.2: Check if any state in the window contains summary patterns.

        If the recent window contains summary/recap language, the repetition
        might be benign rather than a loop.
        """
        for state in states[-5:]:  # Check last 5 states
            if self._is_summary_or_progress(state.content):
                return True
        return False

    def _calibrate_confidence(
        self,
        raw_score: float,
        method: str,
        evidence_strength: float,
        loop_length: int,
    ) -> float:
        """Calibrate confidence based on evidence strength and detection method."""
        base_confidence = {
            "structural": 0.96,
            "hash": 0.80,
            "semantic": 0.70,
            "semantic_clustering": 0.75,  # Slightly higher than basic semantic
            "lexical": 0.68,
        }.get(method, 0.5)

        length_factor = min(1.0, loop_length / 5)
        evidence_factor = evidence_strength

        calibrated = (
            base_confidence * 0.5 + raw_score * 0.25 + length_factor * 0.15 + evidence_factor * 0.10
        )
        calibrated = min(0.99, calibrated * self.confidence_scaling)

        return round(calibrated, 4)

    def _no_loop(self, **kwargs) -> LoopDetectionResult:
        return LoopDetectionResult(
            detected=False,
            confidence=0.0,
            method=None,
            cost=0.0,
            framework=self.framework,
            **kwargs,
        )

    def detect_loop(self, states: List[StateSnapshot]) -> LoopDetectionResult:
        if len(states) < 3:
            return self._no_loop()

        # v1.3: If the current state is a summary/progress report, short-circuit
        # to avoid flagging legitimate recap as a loop.
        # v1.6: Only short-circuit if the summary language is isolated (agent
        # wrapping up) AND earlier states don't already look like a loop. A
        # single word match on the last state while earlier states clearly
        # repeat the same work should still flag as a loop.
        current = states[-1]
        if self._is_summary_or_progress(current.content):
            if not self._earlier_states_look_like_loop(states[:-1]):
                return self._no_loop(evidence={"summary_short_circuit": True})

        # v1.5: If state_delta has multiple keys and at least one key has
        # strictly distinct values across every state, this is iteration
        # (fibonacci progress, batch processing, retry attempts), not cycling.
        # Gated on >=2 keys so alternating A/B or stuck-clarification loops —
        # which carry only a single surface-level key — still fall through to
        # lexical/semantic detection.
        if self._has_iterating_signal(states):
            return self._no_loop(evidence={"iterating_signal": True})

        window = states[-self.window_size : -1] if len(states) > self.window_size else states[:-1]

        return (
            self._detect_structural_loop(current, window, states)
            or self._detect_hash_loop(current, window, states)
            or self._detect_semantic_loop(current, window, states)
            or self._detect_lexical_loop(current, window, states)
            or self.detect_semantic_loop_with_clustering(states)
            or self._no_loop()
        )

    def _detect_structural_loop(
        self, current: StateSnapshot, window: List[StateSnapshot], states: List[StateSnapshot]
    ) -> Optional[LoopDetectionResult]:
        """Tier 1: Structural matching — same agent, same state keys, no progress."""
        matches = [
            i
            for i, prev in enumerate(window)
            if self._structural_match(current, prev)
            and not self._has_meaningful_progress(prev, current)
        ]
        if not matches:
            return None

        # v1.4: Batch-iteration guard. A genuine loop cycles through repeating
        # state values (e.g. thermostat 72→68→72→68); batch iteration produces
        # strictly distinct values per key (doc: 1,2,3,4…). If every key in the
        # window has fully distinct values, this is iteration, not a loop.
        if not self._has_value_repetition([*window, current]):
            return None

        first_match = matches[0]
        loop_length = len(window) - first_match
        window_start = max(0, len(states) - self.window_size - 1)
        raw_score = len(matches) / len(window)

        return LoopDetectionResult(
            detected=True,
            confidence=self._calibrate_confidence(
                raw_score, "structural", min(1.0, len(matches) / 2), loop_length
            ),
            method="structural",
            cost=0.0,
            loop_start_index=window_start + first_match,
            loop_length=loop_length,
            raw_score=raw_score,
            evidence={
                "structural_matches": len(matches),
                "window_size": len(window),
                "structural_threshold": self.structural_threshold,
            },
            framework=self.framework,
        )

    def _detect_hash_loop(
        self, current: StateSnapshot, window: List[StateSnapshot], states: List[StateSnapshot]
    ) -> Optional[LoopDetectionResult]:
        """Tier 2: Hash collision — identical state_delta content."""
        current_hash = self._compute_state_hash(current)
        matches = [
            i for i, prev in enumerate(window) if self._compute_state_hash(prev) == current_hash
        ]
        if not matches:
            return None

        # v1.6: Hash of empty state_delta ({}) is trivially constant. If all
        # matched states have empty deltas, require that the content itself
        # also repeats — otherwise the "hash match" is spurious (different
        # agents/content happen to share an empty delta).
        if not current.state_delta:
            matched_contents = [window[i].content for i in matches if not window[i].state_delta]
            if matched_contents and not any(
                c == current.content
                or (
                    len(c) > 20
                    and len(current.content) > 20
                    and (c.startswith(current.content[:40]) or current.content.startswith(c[:40]))
                )
                for c in matched_contents
            ):
                return None

        first_match = matches[0]
        loop_length = len(window) - first_match
        raw_score = len(matches) / len(window)

        return LoopDetectionResult(
            detected=True,
            confidence=self._calibrate_confidence(
                raw_score, "hash", min(1.0, len(matches) / 2), loop_length
            ),
            method="hash",
            cost=0.0,
            loop_start_index=len(states) - 1 - loop_length,
            loop_length=loop_length,
            raw_score=raw_score,
            evidence={"hash_matches": len(matches), "window_size": len(window)},
            framework=self.framework,
        )

    def _detect_semantic_loop(
        self, current: StateSnapshot, window: List[StateSnapshot], states: List[StateSnapshot]
    ) -> Optional[LoopDetectionResult]:
        """Tier 3: Semantic similarity — embedding-based with progress filter."""
        try:
            contents = [s.content for s in window] + [current.content]
            if len(contents) < 4:
                return None

            embeddings = self.embedder.encode(contents)
            current_emb = embeddings[-1]

            high_sim_matches = [
                (i, self.embedder.similarity(current_emb, emb))
                for i, emb in enumerate(embeddings[:-1])
            ]
            high_sim_matches = [
                (i, sim)
                for i, sim in high_sim_matches
                if sim > self.semantic_threshold
                and not self._has_meaningful_progress(window[i], current)
            ]

            # v1.2: If current state is a summary/recap, don't flag as loop
            if self._is_summary_or_progress(current.content):
                return (
                    self._no_loop(evidence={"summary_pattern_detected": True})
                    if high_sim_matches
                    else None
                )

            if len(high_sim_matches) < self.min_matches_for_loop:
                return None

            first_match_idx = high_sim_matches[0][0]
            avg_similarity = sum(s for _, s in high_sim_matches) / len(high_sim_matches)
            max_similarity = max(s for _, s in high_sim_matches)
            loop_length = len(window) - first_match_idx
            window_start = max(0, len(states) - self.window_size - 1)

            return LoopDetectionResult(
                detected=True,
                confidence=self._calibrate_confidence(
                    avg_similarity, "semantic", min(1.0, len(high_sim_matches) / 4), loop_length
                ),
                method="semantic",
                cost=0.0,
                loop_start_index=window_start + first_match_idx,
                loop_length=loop_length,
                raw_score=avg_similarity,
                evidence={
                    "semantic_matches": len(high_sim_matches),
                    "avg_similarity": round(avg_similarity, 4),
                    "max_similarity": round(max_similarity, 4),
                    "threshold": self.semantic_threshold,
                },
                framework=self.framework,
            )
        except Exception:
            return None

    _LEXICAL_STOPWORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "shall",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "this",
            "that",
            "these",
            "those",
            "and",
            "or",
            "but",
            "if",
            "then",
            "else",
            "when",
            "where",
            "why",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "whose",
            "to",
            "of",
            "in",
            "on",
            "at",
            "by",
            "for",
            "with",
            "about",
            "from",
            "as",
            "into",
            "than",
            "so",
            "just",
            "more",
            "most",
            "some",
            "any",
            "all",
            "not",
            "no",
            "nor",
            "only",
            "own",
            "same",
            "very",
            "too",
            "also",
            "here",
            "there",
            "please",
            "like",
            "still",
        }
    )

    def _tokenize_content(self, text: str) -> set:
        """Extract content word tokens as 4-char prefixes so morphological
        variants (clarify/clarification/clarifying, update/updating/updates)
        collapse to the same stem. Trade-off: over-collapses some unrelated
        short words, but maximizes recall for stuck-topic loop detection."""
        import re

        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        return {t[:4] for t in tokens if t not in self._LEXICAL_STOPWORDS and len(t) >= 3}

    def _detect_lexical_loop(
        self, current: StateSnapshot, window: List[StateSnapshot], states: List[StateSnapshot]
    ) -> Optional[LoopDetectionResult]:
        """Tier 2.5: Lexical overlap — catches loops where content shares topic/keywords
        but embedding similarity is below the semantic threshold (e.g. repeated
        clarification requests phrased differently each time)."""
        current_tokens = self._tokenize_content(current.content)
        if len(current_tokens) < 2:
            return None

        lexical_threshold = 0.30
        matches = []
        for i, prev in enumerate(window):
            if prev.agent_id != current.agent_id:
                continue
            prev_tokens = self._tokenize_content(prev.content)
            if not prev_tokens:
                continue
            union = current_tokens | prev_tokens
            intersection = current_tokens & prev_tokens
            jaccard = len(intersection) / len(union) if union else 0.0
            if jaccard >= lexical_threshold and not self._has_meaningful_progress(prev, current):
                matches.append((i, jaccard))

        if len(matches) < self.min_matches_for_loop:
            return None

        first_match_idx = matches[0][0]
        avg_overlap = sum(j for _, j in matches) / len(matches)
        loop_length = len(window) - first_match_idx
        window_start = max(0, len(states) - self.window_size - 1)

        return LoopDetectionResult(
            detected=True,
            confidence=self._calibrate_confidence(
                avg_overlap, "lexical", min(1.0, len(matches) / 4), loop_length
            ),
            method="lexical",
            cost=0.0,
            loop_start_index=window_start + first_match_idx,
            loop_length=loop_length,
            raw_score=avg_overlap,
            evidence={
                "lexical_matches": len(matches),
                "avg_jaccard": round(avg_overlap, 4),
                "threshold": lexical_threshold,
            },
            framework=self.framework,
        )

    def _earlier_states_look_like_loop(self, earlier: List["StateSnapshot"]) -> bool:
        """v1.6: Cheap check — do earlier states in the sequence already
        exhibit loop-like repetition (same content, same delta ignoring
        bookkeeping counters, or the same ~40-char prefix)?"""
        if len(earlier) < 2:
            return False
        # Identical content or content prefix match
        for i in range(len(earlier)):
            for j in range(i + 1, len(earlier)):
                a, b = earlier[i].content, earlier[j].content
                if a and a == b:
                    return True
                if a and b and len(a) > 30 and len(b) > 30 and a[:30] == b[:30]:
                    return True

        # Identical state_delta with bookkeeping keys stripped
        def _stripped(sd: dict) -> str:
            cleaned = {k: v for k, v in sd.items() if k not in _BOOKKEEPING_KEYS}
            return json.dumps(cleaned, sort_keys=True, default=str)

        for i in range(len(earlier)):
            if not earlier[i].state_delta:
                continue
            si = _stripped(earlier[i].state_delta)
            for j in range(i + 1, len(earlier)):
                if not earlier[j].state_delta:
                    continue
                if si == _stripped(earlier[j].state_delta):
                    return True
        return False

    def _has_iterating_signal(self, states: List[StateSnapshot]) -> bool:
        """Heuristic: multi-key state_delta where at least one non-bookkeeping
        key has strictly distinct values across every state indicates iteration
        (each step does new work on a different item), not a loop.

        v1.6: Bookkeeping counters (iteration_count, turn_count, retry_count,
        attempts, timestamps…) are excluded — they increment on every state
        regardless of whether real work is progressing. A stuck retry loop
        has strictly increasing retry_count but is still a loop."""
        all_keys: set = set()
        for s in states:
            all_keys |= set(s.state_delta.keys())
        # Only consider work-identifier keys (non-bookkeeping)
        work_keys = {k for k in all_keys if k not in _BOOKKEEPING_KEYS}
        if len(work_keys) < 2:
            return False
        for k in work_keys:
            values = []
            present = 0
            for s in states:
                if k not in s.state_delta:
                    continue
                present += 1
                v = s.state_delta[k]
                try:
                    hash(v)
                    values.append(v)
                except TypeError:
                    values.append(json.dumps(v, sort_keys=True, default=str))
            if present == len(states) and len(set(values)) == len(values):
                return True
        return False

    def _has_value_repetition(self, states: List[StateSnapshot]) -> bool:
        """Return True if any state_delta key has a repeating value across
        the given states. An all-distinct value sequence for every key means
        the agent is iterating (new work each step), not cycling."""
        all_keys: set = set()
        for s in states:
            all_keys |= set(s.state_delta.keys())
        if not all_keys:
            return True  # no structural signal; keep existing behavior
        for k in all_keys:
            seen: set = set()
            for s in states:
                if k not in s.state_delta:
                    continue
                v = s.state_delta[k]
                try:
                    hash(v)
                    key = v
                except TypeError:
                    key = json.dumps(v, sort_keys=True, default=str)
                if key in seen:
                    return True
                seen.add(key)
        return False

    def _structural_match(self, a: StateSnapshot, b: StateSnapshot) -> bool:
        return a.agent_id == b.agent_id and set(a.state_delta.keys()) == set(b.state_delta.keys())

    def _has_meaningful_progress(self, prev: StateSnapshot, current: StateSnapshot) -> bool:
        # v1.6: Bookkeeping keys (iteration counters, timestamps) change on
        # every step regardless of work progress — exclude from progress check.
        prev_keys = set(prev.state_delta.keys()) - _BOOKKEEPING_KEYS
        curr_keys = set(current.state_delta.keys()) - _BOOKKEEPING_KEYS
        delta_keys = curr_keys - prev_keys
        value_changes = sum(
            1 for k in curr_keys if k in prev_keys and current.state_delta[k] != prev.state_delta[k]
        )
        if len(delta_keys) > 0 or value_changes >= 2:
            return True
        if value_changes >= 1 and current.content:
            progress_markers = ["completed", "finished", "done", "next", "step", "moving on"]
            if any(m in current.content.lower() for m in progress_markers):
                return True
        return False

    def _compute_state_hash(self, state: StateSnapshot) -> str:
        normalized = json.dumps(state.state_delta, sort_keys=True)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def detect_semantic_loop_with_clustering(
        self, states: List[StateSnapshot]
    ) -> Optional[LoopDetectionResult]:
        """Advanced semantic loop detection using embedding clustering.

        Uses KMeans clustering to find groups of semantically similar states,
        then checks if recent states are repeatedly falling into the same cluster
        (indicating a semantic loop where the agent keeps doing similar things).
        """
        if len(states) < 6:
            return None

        try:
            from sklearn.cluster import KMeans

            embeddings = self.embedder.encode([s.content for s in states])
            n_samples = len(embeddings)
            n_clusters = min(max(2, n_samples // 4), 5)

            if n_samples < n_clusters * 2:
                return None

            cluster_labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(
                embeddings
            )
            recent_window = min(self.window_size, len(cluster_labels))
            recent_labels = cluster_labels[-recent_window:]

            evidence = self._find_cluster_pattern(recent_labels, cluster_labels)
            if evidence is None:
                return None

            # v1.6: Batch-iteration guard for clustering path. If within the
            # dominant/cycling cluster the states carry a non-bookkeeping
            # state_delta key whose values are strictly distinct, the agent is
            # processing a batch of different items — not looping.
            if self._cluster_is_batch_iteration(states, cluster_labels, evidence):
                return None

            return self._build_clustering_result(
                embeddings, cluster_labels, recent_window, n_clusters, evidence
            )
        except Exception:
            return None

    def _cluster_is_batch_iteration(self, states, cluster_labels, evidence) -> bool:
        """v1.6: Return True if the clusters look like batch iteration rather
        than a loop. Batch iteration: within a cluster (same kind of work),
        there is a non-bookkeeping state_delta key whose values are all
        strictly distinct — the agent is processing item 1, item 2, item 3…"""
        target_clusters = set()
        if "dominant_cluster" in evidence:
            target_clusters.add(evidence["dominant_cluster"])
        # For cycle patterns, consider all clusters present
        if evidence.get("type", "").endswith("cycle") or evidence.get("cycle_length"):
            target_clusters.update(int(c) for c in evidence.get("cluster_distribution", {}).keys())
        if not target_clusters:
            return False

        for cluster_id in target_clusters:
            cluster_states = [s for s, lbl in zip(states, cluster_labels) if lbl == cluster_id]
            if len(cluster_states) < 2:
                continue
            all_keys: set = set()
            for s in cluster_states:
                all_keys |= set(s.state_delta.keys())
            work_keys = {k for k in all_keys if k not in _BOOKKEEPING_KEYS}
            for k in work_keys:
                values = []
                for s in cluster_states:
                    if k not in s.state_delta:
                        continue
                    v = s.state_delta[k]
                    try:
                        hash(v)
                        values.append(v)
                    except TypeError:
                        values.append(json.dumps(v, sort_keys=True, default=str))
                if len(values) == len(cluster_states) and len(set(values)) == len(values):
                    return True
        return False

    def _find_cluster_pattern(self, recent_labels, cluster_labels) -> Optional[dict]:
        """Analyze cluster assignments for dominance or cyclic patterns."""
        cluster_counts = {}
        for label in recent_labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1

        max_cluster_count = max(cluster_counts.values())
        dominant_cluster = max(cluster_counts, key=cluster_counts.get)
        dominance_ratio = max_cluster_count / len(recent_labels)

        # Check for cyclic patterns in cluster sequence
        cycle_length = 0
        for potential_cycle in range(2, len(recent_labels) // 2 + 1):
            pattern = tuple(recent_labels[-potential_cycle:])
            check_against = tuple(recent_labels[-2 * potential_cycle : -potential_cycle])
            if pattern == check_against:
                cycle_length = potential_cycle
                break

        evidence = {}
        is_loop = False

        if dominance_ratio >= 0.6 and max_cluster_count >= self.min_matches_for_loop:
            is_loop = True
            evidence["type"] = "cluster_dominance"
            evidence["dominant_cluster"] = int(dominant_cluster)
            evidence["dominance_ratio"] = round(dominance_ratio, 3)
            evidence["cluster_count"] = max_cluster_count

        if cycle_length >= 2:
            is_loop = True
            evidence["type"] = (
                evidence.get("type", "") + "_cycle" if evidence.get("type") else "cluster_cycle"
            )
            evidence["cycle_length"] = cycle_length

        if not is_loop:
            return None

        evidence["cluster_distribution"] = {int(k): v for k, v in cluster_counts.items()}
        return evidence

    def _build_clustering_result(
        self, embeddings, cluster_labels, recent_window: int, n_clusters: int, evidence: dict
    ) -> LoopDetectionResult:
        """Validate cluster loop with intra-cluster similarity and build result."""
        dominant_cluster = evidence.get("dominant_cluster", 0)
        max_cluster_count = evidence.get("cluster_count", 0)
        dominance_ratio = evidence.get("dominance_ratio", 0.0)

        # Calculate within-cluster similarity to validate
        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == dominant_cluster]
        cluster_embs = [embeddings[i] for i in cluster_indices[-5:]]

        avg_intra_sim = 0.0
        if len(cluster_embs) >= 2:
            sims = [
                self.embedder.similarity(cluster_embs[i], cluster_embs[j])
                for i in range(len(cluster_embs))
                for j in range(i + 1, len(cluster_embs))
            ]
            avg_intra_sim = sum(sims) / len(sims) if sims else 0.0

        evidence["avg_intra_cluster_similarity"] = round(avg_intra_sim, 4)
        evidence["n_clusters"] = n_clusters

        raw_score = dominance_ratio * 0.5 + avg_intra_sim * 0.5

        # Find loop start (first state in dominant cluster in recent window)
        loop_start_index = None
        for i in range(len(cluster_labels) - recent_window, len(cluster_labels)):
            if cluster_labels[i] == dominant_cluster:
                loop_start_index = i
                break

        return LoopDetectionResult(
            detected=True,
            confidence=self._calibrate_confidence(
                raw_score, "semantic_clustering", min(1.0, max_cluster_count / 5), max_cluster_count
            ),
            method="semantic_clustering",
            cost=0.0,
            loop_start_index=loop_start_index,
            loop_length=max_cluster_count,
            raw_score=raw_score,
            evidence=evidence,
            framework=self.framework,
        )

    def detect_loop_enhanced(self, states: List[StateSnapshot]) -> LoopDetectionResult:
        """Enhanced loop detection with all methods including clustering.

        Order of detection (cheapest to most expensive):
        1. Structural matching (O(n), no API calls)
        2. Hash collision (O(n), no API calls)
        3. Basic semantic similarity (embedding generation + pairwise comparison)
        4. Clustering-based semantic (embedding generation + KMeans)
        """
        # First try the standard methods
        result = self.detect_loop(states)
        if result.detected:
            return result

        # If standard methods didn't detect, try clustering-based semantic
        clustering_result = self.detect_semantic_loop_with_clustering(states)
        if clustering_result:
            return clustering_result

        return LoopDetectionResult(
            detected=False,
            confidence=0.0,
            method=None,
            cost=0.0,
            framework=self.framework,
        )


loop_detector = MultiLevelLoopDetector()
