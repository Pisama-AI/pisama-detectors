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
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pisama_detectors._config import get_settings, get_tenant_thresholds
from pisama_detectors.detection.precision_guards import (
    outcome_is_success,
    reports_explicit_failure,
)
from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder

# Detector version
# v1.9: content fingerprint tier — catches loops with empty state_deltas where
# the same content prefix repeats 3+ times (MAST ChatDev/MetaGPT pattern);
# structural matching with empty deltas now requires content similarity to
# avoid false positives on healthy tool-call iteration.
# v2.0: extend structural-tier guard for empty-delta traces from agent_id=="unknown"
# to ALL agents. Use embedding cosine similarity (threshold 0.65) instead of
# Jaccard: cosine separates real loops (same-stuck-problem, median 0.83) from
# AG2-style topic-switching FPs (different sub-tasks, median 0.65) much better
# than word overlap. Grid search on cleaned corpus confirmed 0.65 as best F1 point.
# v2.1: structural-tier recurrence requirement — a key-set match alone is not a
# loop. Each structural match must be a genuine return to the same state
# (bookkeeping-stripped work payload recurs, or — for a trivial payload — the
# content also repeats). Kills systemic over-firing on all-distinct-state traces
# (e.g. a planner→…→planner workflow whose key-set is identical every step but
# whose work differs) while preserving every genuine recurrence, including stuck
# retry loops whose only-changing field is a counter. The content tiers
# (semantic/lexical/content_fingerprint) keep their own guards and are untouched,
# so paraphrased loops with no shared state_delta still fire.
# v2.2: a state_delta only counts as loop evidence if it carries WORK. v2.1 got
# this right for the structural tier but expressed it in two incomplete places:
# (a) the hash tier keyed on raw state_delta identity, guarded only against the
# empty delta, and (b) "substantive" was judged on key names alone. So a producer
# that stamps a fixed channel label on every snapshot — {"step": "work"},
# {"event_type": ..., "category": ...}, or a trace_only delta whose only varying
# key was the content that got nulled to None — collided with itself by
# construction and fired at 0.90-0.98 on traces that obviously progressed.
# Channel/type keys and None-valued keys are now non-work, and the hash tier
# demands the same content confirmation the empty delta always got. Deltas that
# name actual work ({"query": "inventory SKU-1001"}, a static last_error under a
# ticking retry counter) are untouched and still fire on paraphrased content.
# v2.3: no-signal guard. A snapshot whose state_delta carries no work AND whose
# content is unreadable is not evidence of anything, but such snapshots are all
# identical to each other, so a trace made entirely of them read as a perfect
# loop. Detection now requires at least one snapshot in the trace to carry some
# signal. Any trace with real content or a real payload is unaffected.
# v2.4: exact-output fan-out suppression requires structured, distinct external
# work input (not changing peer prose/counters), or explicit terminal progress.
DETECTOR_VERSION = "2.4"

# v1.6: Monotonic-counter keys that are bookkeeping, not work progress.
# When these are the ONLY strictly-distinct keys, the agent is still looping
# on the same work (same tool call, same error) — don't short-circuit.
_BOOKKEEPING_KEYS = frozenset(
    {
        "iteration_count",
        "iteration_index",
        "iteration",
        "iteration_num",
        "iter",
        "iter_num",
        "turn_count",
        "turn",
        "step",
        "step_num",
        "superstep",
        "retry",
        "retry_count",
        "retries",
        "attempt",
        "attempts",
        "attempt_num",
        "attempt_n",
        "num_attempts",
        "format_attempts",
        "cycle",
        "cycle_num",
        "cycle_count",
        "loop_count",
        "loop_index",
        "loop_num",
        "epoch",
        "epoch_num",
        "round",
        "round_num",
        "round_count",
        "tick",
        "execution_time_ms",
        "timestamp_ms",
        "timestamp",
        "sequence_num",
        "sequence",
        "seq",
    }
)

# v1.7: Status-like state_delta keys and the terminal/success values that mean
# the run *resolved*. A sequence that reaches one of these made progress
# (e.g. pending→retry→retry→success) — it terminated, so it must not be flagged
# as a stuck loop even though its structure repeats. Keyed on the structured
# status field (not free-text content) and gated on an actual transition so a
# constant "status: ok" on every step is not mistaken for progress.
_STATUS_KEYS = frozenset(
    {
        "status",
        "state",
        "outcome",
        "result",
        "phase",
        "stage",
        "task_status",
        "job_status",
        "step_status",
        "run_status",
    }
)
_TERMINAL_SUCCESS_VALUES = frozenset(
    {
        "success",
        "succeeded",
        "successful",
        "complete",
        "completed",
        "done",
        "finished",
        "resolved",
        "passed",
        "pass",
        "ok",
        "fulfilled",
    }
)

# v2.3: content values that are placeholders for absent data rather than
# something the agent said. Compared case-insensitively after stripping.
_UNINFORMATIVE_CONTENT = frozenset(
    {"", "none", "null", "nil", "n/a", "na", "-", "{}", "[]", "()"}
)

# v2.4: Fields that can change during a peer-to-peer bounce without supplying
# new work. They must not turn changing counters or rephrased hand-off chatter
# into the "distinct input" needed to excuse an otherwise exact repetition.
_NON_INPUT_DELTA_KEYS = frozenset(
    {
        *_BOOKKEEPING_KEYS,
        *_STATUS_KEYS,
        "agent", "agent_id", "agent_name", "agent_role", "role", "speaker",
        "author", "sender", "actor", "participant", "participant_id", "from", "to",
        "event_type", "event", "event_name", "type", "kind", "category", "channel",
        "node", "node_name", "span_kind", "message_type",
        "bounce", "bounces", "bounce_count", "peer_bounce", "handoff", "handoffs",
        "handoff_count", "handoff_target", "delegate_to", "delegation_target",
        "next_agent", "recipient",
        "message", "content", "text", "prose", "reply", "response_text", "note",
        "notes", "summary", "description", "explanation", "thought", "reasoning",
        # Per-attempt telemetry IDs are not workload identities. A retry gets a new
        # call/request/span ID even when it is doing exactly the same work.
        "call_id", "request_id", "response_id", "event_id", "execution_id", "run_id",
        "span_id", "trace_id",
    }
)
_COUNTERISH_INPUT_KEY_RE = re.compile(
    r"(?:^|_)(?:attempt|bounce|cycle|handoff|iteration|retry|round|step|turn)"
    r"(?:s|_count|_index|_num|_number|$)",
    re.IGNORECASE,
)
_TERMINAL_PROGRESS_RE = re.compile(
    r"\b(?:all\s+)?(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(?:[\w-]+\s+){0,2}"
    r"(?:checked|completed|delivered|finished|indexed|processed|reviewed|screened|"
    r"submitted|validated)\b",
    re.IGNORECASE,
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


def _strip_non_input_fields(value):
    """Remove coordination/telemetry-only fields from an input payload."""
    if isinstance(value, dict):
        cleaned = {}
        for key, child in value.items():
            key_text = str(key).strip().lower()
            if (
                key_text in _NON_INPUT_DELTA_KEYS
                or _COUNTERISH_INPUT_KEY_RE.search(key_text)
            ):
                continue
            child_clean = _strip_non_input_fields(child)
            if child_clean not in (None, "", {}, []):
                cleaned[str(key)] = child_clean
        return cleaned
    if isinstance(value, (list, tuple)):
        cleaned_items = [_strip_non_input_fields(child) for child in value]
        return [child for child in cleaned_items if child not in (None, "", {}, [])]
    return value


def _structured_external_input_fingerprint(state: "StateSnapshot") -> Optional[str]:
    """Fingerprint real structured work supplied by an intervening participant."""
    if not isinstance(state.state_delta, dict) or not state.state_delta:
        return None
    if outcome_is_success(state.state_delta) is False:
        return None
    if reports_explicit_failure(state.content or ""):
        return None
    cleaned = _strip_non_input_fields(state.state_delta)
    if not cleaned:
        return None
    if reports_explicit_failure(json.dumps(cleaned, sort_keys=True, default=str)):
        return None
    return json.dumps(cleaned, sort_keys=True, default=str)


def _has_terminal_progress_after(
    states: List["StateSnapshot"], index: int, repeated_agent: Optional[str]
) -> bool:
    """Require explicit terminal status or quantified completion after a repeat."""
    tail = states[index + 1:]
    for state in reversed(tail):
        # A successful telemetry write or another participant's unrelated work
        # cannot retroactively turn this agent's repeated output into progress.
        if state.agent_id != repeated_agent:
            continue
        verdict = outcome_is_success(state.state_delta)
        if verdict is not None:
            return verdict

    # Some adapters retain only text. Keep this fallback narrow: a quantified
    # completed-work statement from the repeating agent is useful terminal
    # evidence; generic "progress" or peer hand-off prose is not.
    return any(
        state.agent_id == repeated_agent
        and not reports_explicit_failure(state.content or "")
        and _TERMINAL_PROGRESS_RE.search(state.content or "")
        for state in tail
    )


def _repeats_driven_by_distinct_inputs(
    states: List["StateSnapshot"], indices: List[int]
) -> bool:
    """Return whether repeats have trustworthy evidence of distinct work.

    A different participant plus different prose is not enough: two agents can
    bounce the same task forever while changing round numbers and wording. The
    exemption requires either structured, non-bookkeeping external/tool input
    in every interval, or explicit terminal/quantified progress after the last
    repetition.
    """
    if len(indices) < 2:
        return False
    repeated_agent = states[indices[0]].agent_id if indices else None
    if _has_terminal_progress_after(states, indices[-1], repeated_agent):
        return True

    intervening = [states[start + 1:end] for start, end in zip(indices, indices[1:])]
    fingerprints = []
    for window in intervening:
        external_inputs = sorted({
            fingerprint
            for state in window
            if state.agent_id != repeated_agent
            for fingerprint in [_structured_external_input_fingerprint(state)]
            if fingerprint is not None
        })
        if not external_inputs:
            return False
        fingerprints.append(json.dumps(external_inputs, sort_keys=True))
    return len(set(fingerprints)) == len(fingerprints)


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
            self.structural_threshold = (
                structural_threshold
                if structural_threshold is not None
                else thresholds.structural_threshold
            )
            self.semantic_threshold = (
                semantic_threshold
                if semantic_threshold is not None
                else thresholds.semantic_threshold
            )
            self.window_size = (
                window_size if window_size is not None else thresholds.loop_detection_window
            )
            self.min_matches_for_loop = (
                min_matches_for_loop
                if min_matches_for_loop is not None
                else thresholds.min_matches_for_loop
            )
            self.confidence_scaling = (
                confidence_scaling
                if confidence_scaling is not None
                else thresholds.confidence_scaling
            )
        else:
            # Fall back to global settings
            self.structural_threshold = (
                structural_threshold
                if structural_threshold is not None
                else settings.structural_threshold
            )
            self.semantic_threshold = (
                semantic_threshold
                if semantic_threshold is not None
                else settings.semantic_threshold
            )
            self.window_size = (
                window_size if window_size is not None else settings.loop_detection_window
            )
            self.min_matches_for_loop = (
                min_matches_for_loop if min_matches_for_loop is not None else 2
            )
            self.confidence_scaling = (
                confidence_scaling if confidence_scaling is not None else 1.0
            )

        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Reject invalid detector settings before they reach matching code."""
        for name, value in (
            ("structural_threshold", self.structural_threshold),
            ("semantic_threshold", self.semantic_threshold),
        ):
            try:
                is_valid = (
                    not isinstance(value, bool)
                    and math.isfinite(value)
                    and 0.0 <= value <= 1.0
                )
            except (TypeError, ValueError):
                is_valid = False
            if not is_valid:
                raise ValueError(f"{name} must be between 0 and 1")

        for name, value in (
            ("window_size", self.window_size),
            ("min_matches_for_loop", self.min_matches_for_loop),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1")

        try:
            confidence_scaling_is_valid = (
                not isinstance(self.confidence_scaling, bool)
                and math.isfinite(self.confidence_scaling)
                and self.confidence_scaling > 0.0
            )
        except (TypeError, ValueError):
            confidence_scaling_is_valid = False
        if not confidence_scaling_is_valid:
            raise ValueError("confidence_scaling must be greater than 0")

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
        # v2.3: A loop is a claim about what the agent DID. If no snapshot in the
        # trace carries either a work payload or readable content, the trace says
        # nothing about the agent's behaviour and the uniformity of its snapshots
        # is an artifact of the data being absent — not a repetition. Deliberately
        # whole-trace and not per-snapshot: a single blank step inside a real
        # trace is still legitimate loop evidence, so this only suppresses traces
        # that are entirely signal-free.
        if not any(
            self._delta_is_substantive(s.state_delta) or self._content_is_informative(s.content)
            for s in states
        ):
            return self._no_loop(evidence={"no_signal": True})

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
            or self._detect_content_fingerprint_loop(states)
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

        # v2.0: When state_deltas are empty, structural matching is vacuous regardless
        # of agent name. Use embedding cosine similarity to confirm real content
        # repetition. Threshold 0.65 was chosen by grid search on the cleaned corpus:
        # separates real loops (same-agent content repeating, median cosine 0.83) from
        # AG2-style topic-switching FPs (different sub-tasks each turn, median 0.65).
        # Falls back to Jaccard if embedder is unavailable.
        if not current.state_delta:
            try:
                cur_content = current.content or ""
                window_contents = [window[i].content or "" for i in matches]
                all_contents = [cur_content] + window_contents
                embs = self.embedder.encode(all_contents)
                cur_emb = embs[0]
                matches = [
                    matches[j] for j, emb in enumerate(embs[1:])
                    if self.embedder.similarity(cur_emb, emb) >= 0.65
                ]
            except Exception:
                # Embedder unavailable: fall back to Jaccard
                matches = [
                    i for i in matches
                    if self._content_similar(current.content, window[i].content, threshold=0.5)
                ]
            if not matches:
                return None

        # v2.1: Structural recurrence requirement — key-set similarity alone is
        # not a loop. Require each remaining match to be a genuine return to the
        # same state (``_is_loop_repeat``): the bookkeeping-stripped work payload
        # recurs (a stuck retry, only a counter ticking) OR, for a trivial
        # payload, the content also repeats. This drops the systemic over-fire on
        # all-distinct traces — e.g. a planner→…→planner workflow whose key-set is
        # identical every step but whose work (response) differs — while keeping
        # genuine recurrences. The empty-state_delta path above already applies
        # its own cosine guard, so only non-empty deltas need this filter.
        if current.state_delta:
            matches = [i for i in matches if self._is_loop_repeat(current, window[i])]
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
        window_start = max(0, len(states) - self.window_size)
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

        # v1.6: Hash of empty state_delta ({}) is trivially constant. Require
        # content similarity to confirm the hash match is real.
        # v1.8: Also require same agent_id — different agents sharing an empty
        # state_delta and a common prompt prefix (e.g. a shared task description)
        # produce spurious cross-agent matches that don't indicate a loop.
        # v2.2: an EMPTY delta is only the loudest case of a delta with no work in
        # it. One that carries nothing but a channel label ({"step": "work"},
        # {"event_type": ...}) or nulled-out content collides with itself on every
        # snapshot the producer emits, so the hash says nothing about the agent —
        # it fired at 0.90 on six snapshots describing six different actions.
        # Both cases now take the same content confirmation. The structural tier
        # already applies this rule via _is_loop_repeat; this is the hash tier's
        # equivalent, which had been keying on the raw delta.
        if not self._delta_is_substantive(current.state_delta):
            matches = [
                i
                for i in matches
                if (
                    window[i].agent_id == current.agent_id
                    and (
                        window[i].content == current.content
                        or (
                            len(window[i].content) > 20
                            and len(current.content) > 20
                            and window[i].content[:40] == current.content[:40]
                        )
                    )
                )
            ]
            if not matches:
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

    def _detect_content_fingerprint_loop(
        self, states: List[StateSnapshot]
    ) -> Optional[LoopDetectionResult]:
        """v1.9: Content fingerprint detection — catches loops where the same
        content prefix appears 3+ times, even across different agents.

        MAST ChatDev/MetaGPT traces have empty state_deltas but clear content
        repetition (Code_Reviewer asking for same fix, Programmer submitting same
        code). Structural/hash tiers miss these because they require state_delta
        signals. This tier uses content prefix fingerprints to detect repetition.

        Progress guards:
        1. If the RESPONSE states (between repeating requests) differ, it's healthy
           iteration (skip→different song→skip→different song), not a loop.
        2. Fingerprints must match on more than just a short command prefix.
        """
        if len(states) < 4:
            return None

        # Build fingerprints: first 80 chars, normalized
        def fingerprint(s: StateSnapshot) -> str:
            c = s.content.strip()[:80].lower()
            c = re.sub(r"\s+", " ", c)
            return c

        fps = [fingerprint(s) for s in states]
        from collections import Counter

        fp_counts = Counter(fps)

        # Find fingerprints appearing 3+ times (loop signal)
        repeated = [(fp, cnt) for fp, cnt in fp_counts.items() if cnt >= 3 and len(fp) > 20]
        if not repeated:
            return None

        # v1.9: Same-agent + full-content guard. A stuck loop has the SAME
        # agent repeating IDENTICAL full content. Just matching prefixes is not
        # enough — healthy code generation often shares the same prefix/structure
        # but produces different full output each time.
        #
        # Strategy: find states where ONE agent produces the same FULL content
        # 2+ times. That's the loop signal — an agent truly stuck.
        loop_agent = None
        loop_indices = []

        # Group states by agent
        from collections import defaultdict

        agent_states = defaultdict(list)
        for i, s in enumerate(states):
            agent_states[s.agent_id].append((i, s.content))

        # Find an agent with repeated full content
        for agent, indexed_contents in agent_states.items():
            if len(indexed_contents) < 2:
                continue
            content_to_indices = defaultdict(list)
            for idx, content in indexed_contents:
                content_to_indices[content].append(idx)
            # Find content that appears 2+ times
            for content, indices in content_to_indices.items():
                if len(indices) >= 2 and len(content) > 50:
                    loop_agent = agent
                    loop_indices = indices
                    break
            if loop_agent:
                break

        if not loop_agent or len(loop_indices) < 2:
            return None

        # Identical output can be healthy fan-out when each repeat answers a
        # different intervening input. Empty or identical windows remain loop
        # evidence, so this suppresses only explicit progress.
        if _repeats_driven_by_distinct_inputs(states, loop_indices):
            return None

        indices = loop_indices
        match_count = len(indices)

        loop_start = indices[0]
        loop_length = indices[-1] - indices[0]
        # v1.9: Exact content duplication is a strong signal. Score based on
        # match count, not ratio to total states (3 exact dups in 25 states is
        # just as bad as 3 in 6). Min 0.5 for 2 matches, scaling up to 1.0.
        raw_score = min(1.0, 0.3 + 0.2 * match_count)

        # Get a preview of the repeated content
        repeated_content = states[indices[0]].content
        preview = repeated_content[:40] if repeated_content else ""

        return LoopDetectionResult(
            detected=True,
            # Use structural base confidence — exact content match is strong evidence
            confidence=self._calibrate_confidence(
                raw_score, "structural", min(1.0, match_count / 3), loop_length
            ),
            method="content_fingerprint",
            cost=0.0,
            loop_start_index=loop_start,
            loop_length=loop_length,
            raw_score=raw_score,
            evidence={
                "exact_content_matches": match_count,
                "content_preview": preview,
                "loop_agent": loop_agent,
                "total_states": len(states),
            },
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

            # v1.8: Same-agent guard — when all spans lack state_delta, cross-agent
            # semantic similarity is driven by shared prompt context, not repetition.
            if not current.state_delta:
                high_sim_matches = [
                    (i, sim)
                    for i, sim in high_sim_matches
                    if not window[i].state_delta and window[i].agent_id == current.agent_id
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
            window_start = len(states) - 1 - len(window)

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
        window_start = len(states) - 1 - len(window)

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

    def _content_similar(self, a: str, b: str, threshold: float = 0.6) -> bool:
        """v1.9: Check if two content strings are similar enough to indicate
        repetition. Uses prefix match + Jaccard on word tokens. Threshold 0.6
        balances catching real loops (same error message rephrased) vs. healthy
        iteration (distinct results each step)."""
        if not a or not b:
            return False
        if a == b:
            return True
        # Prefix match (first 50 chars) catches exact-same-message loops
        if len(a) > 40 and len(b) > 40 and a[:50] == b[:50]:
            return True
        # Jaccard on word tokens for paraphrased loops
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return False
        jaccard = len(words_a & words_b) / len(words_a | words_b)
        return jaccard >= threshold

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
        if a.agent_id != b.agent_id:
            return False
        keys_a = set(a.state_delta.keys())
        keys_b = set(b.state_delta.keys())
        # Exact key-set equality is the structural_threshold == 1.0 case and is
        # the fast path / prior behavior. The per-framework structural_threshold
        # (0.88-0.98, set in _config.py) relaxes this to a Jaccard key-overlap
        # floor so the knob is actually live: n8n (0.98) stays near-exact for its
        # deterministic DAG state, looser frameworks (crewai 0.88) tolerate minor
        # key drift between iterations. Tenant overrides flow through the same
        # self.structural_threshold. At threshold 1.0 only exact matches pass, so
        # callers that want the old strict behavior keep it.
        if keys_a == keys_b:
            return True
        union = keys_a | keys_b
        if not union:
            return True
        if self.structural_threshold >= 1.0:
            return False
        return (len(keys_a & keys_b) / len(union)) >= self.structural_threshold

    def _has_meaningful_progress(self, prev: StateSnapshot, current: StateSnapshot) -> bool:
        # v1.6: Bookkeeping keys (iteration counters, timestamps) change on
        # every step regardless of work progress — exclude from progress check.
        prev_keys = set(prev.state_delta.keys()) - _BOOKKEEPING_KEYS
        curr_keys = set(current.state_delta.keys()) - _BOOKKEEPING_KEYS

        # v1.7: A status-like field transitioning into a terminal/success value
        # is unambiguous progress — the run resolved. A healthy retry sequence
        # (status: pending→retry→retry→success) repeats structurally but is not
        # a stuck loop. Gated on a real transition (prev value differs) so a
        # constant success status across every step is not counted as progress.
        for k in curr_keys:
            if k.lower() not in _STATUS_KEYS:
                continue
            cur_val = str(current.state_delta.get(k, "")).strip().lower()
            if cur_val in _TERMINAL_SUCCESS_VALUES:
                prev_val = str(prev.state_delta.get(k, "")).strip().lower()
                if prev_val != cur_val:
                    return True

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

    # v2.1: state_delta keys that only restate WHO is acting, not WHAT work is
    # being done. Within a per-agent group these are constant (= the group key)
    # or status chatter, so a delta consisting solely of them is not a work
    # payload: its recurrence alone never makes a loop.
    _IDENTITY_DELTA_KEYS = frozenset(
        {
            "agent", "agent_id", "agent_name", "agent_role", "role",
            "speaker", "author", "sender", "actor",
            "participant", "participant_id", "from", "to",
        }
    )

    # v2.2: keys that name the KIND of event or the channel it came down, not the
    # work in it. Ingestion stamps these on every snapshot it emits, so they are
    # constant by construction within a run and their recurrence is a property of
    # the adapter rather than of the agent. Note the contrast with real work keys
    # that happen to hold a verb: {"action": "check_inventory"} says WHAT is being
    # done and stays substantive; {"event_type": "agent.message"} only says which
    # pipe the snapshot arrived on.
    _CHANNEL_DELTA_KEYS = frozenset(
        {
            "event_type", "event", "event_name", "type", "kind", "category",
            "channel", "node", "node_name", "span_kind", "message_type",
        }
    )
    # Deliberately NOT here: "source". In RAG/grounding traces it names the
    # retrieved document, which is real work — an agent pinned on one source is
    # the loop, not the framing.

    def _meaningful_delta(self, state_delta: dict) -> dict:
        """state_delta with bookkeeping keys (iteration/retry/attempt counters,
        timestamps, step counters) stripped. A stuck retry loop ticks only its
        counter while the work payload (tool_args, error_code, http_status…)
        stays identical, so two such states share a meaningful delta even though
        their raw state_hash differs."""
        return {k: v for k, v in state_delta.items() if k not in _BOOKKEEPING_KEYS}

    def _meaningful_delta_hash(self, state: StateSnapshot) -> str:
        """Canonical hash of the bookkeeping-stripped state_delta. Returns ""
        when nothing remains (all-bookkeeping delta)."""
        meaningful = self._meaningful_delta(state.state_delta)
        if not meaningful:
            return ""
        return json.dumps(self._deep_canonicalize(meaningful), sort_keys=True, default=str)

    def _content_is_informative(self, content: str) -> bool:
        """v2.3: whether a snapshot's content says anything at all.

        Some adapters emit a placeholder string rather than an empty one when a
        span carries no prompt/response/tool fields. Treating that as text would
        make every such snapshot identical to every other, which is a loop by
        every measure the detector has."""
        return bool(content) and content.strip().lower() not in _UNINFORMATIVE_CONTENT

    def _delta_is_substantive(self, state_delta: dict) -> bool:
        """True when the bookkeeping-stripped delta carries an actual work key —
        something beyond agent identity, event channel and status.
        ``{query: 'inventory'}`` is substantive (a stuck re-query);
        ``{agent: planner, status: ok}`` is not (just identity + a generic OK),
        and neither is ``{event_type: agent.message, category: chat}`` (v2.2:
        which pipe the snapshot came down, stamped identically on every one).

        v2.2: a key whose value carries no information — ``None``, or an empty
        string/dict/list — does not make a delta substantive either. Ingestion
        that nulls content fields while keeping the keys produces a delta with
        nothing in it, not a work payload recurring."""
        meaningful = self._meaningful_delta(state_delta)
        return any(
            k.lower() not in self._IDENTITY_DELTA_KEYS
            and k.lower() not in self._CHANNEL_DELTA_KEYS
            and k.lower() not in _STATUS_KEYS
            and v is not None
            and v != ""
            and v != {}
            and v != []
            for k, v in meaningful.items()
        )

    def _is_loop_repeat(self, a: StateSnapshot, b: StateSnapshot) -> bool:
        """Whether two same-agent snapshots are a genuine return to the same
        state (the loop precondition):

        1. Their bookkeeping-stripped work payloads match (same tool/error/args,
           only a counter ticking). If that payload is SUBSTANTIVE (a real work
           key, not just agent-identity + status), it is a genuine recurrence
           regardless of content paraphrase — an agent re-issuing the same query.
           If the payload is TRIVIAL (``{agent, status: ok}``), require the
           content to also be similar: a planner→…→planner workflow whose key-set
           is identical but whose response differs each step is progress, not a
           loop.
        2. No matching work payload (differing deltas): fall back to content
           repetition — the signal the semantic/content/lexical tiers loop on."""
        ha, hb = self._meaningful_delta_hash(a), self._meaningful_delta_hash(b)
        if ha and ha == hb:
            if self._delta_is_substantive(a.state_delta):
                return True
            if a.content and b.content and not self._content_similar(a.content, b.content, threshold=0.5):
                return False
            return True
        return self._content_similar(a.content, b.content, threshold=0.5)

    @staticmethod
    def _deep_canonicalize(value):
        """Recursively canonicalize a value so that semantically-equal
        structures hash identically.

        json.dumps(sort_keys=True) sorts dict keys but NOT list elements,
        so [{"id":"a"},{"id":"b"}] and [{"id":"b"},{"id":"a"}] hash
        differently even though the agent state is the same. That made
        the cheap O(1) loop detector silently miss many loops, pushing
        traffic to the expensive O(N²) semantic clustering tier.
        """
        if isinstance(value, dict):
            return {k: MultiLevelLoopDetector._deep_canonicalize(value[k]) for k in sorted(value)}
        if isinstance(value, (list, tuple)):
            canonical = [MultiLevelLoopDetector._deep_canonicalize(v) for v in value]
            try:
                return sorted(canonical, key=lambda v: json.dumps(v, sort_keys=True, default=str))
            except TypeError:
                return canonical
        return value

    def _compute_state_hash(self, state: StateSnapshot) -> str:
        canon = self._deep_canonicalize(state.state_delta)
        normalized = json.dumps(canon, sort_keys=True, default=str)
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
            # v1.7: A cluster *cycle* (A→B→A→B) returns to a prior semantic
            # state — lack of progress, the defining trait of a loop — even when
            # each step's surface text (and therefore its state_delta
            # fingerprint) differs. But the same 2-cycle shape is also produced
            # by ordinary tool use: call→result→call→result, where each cycle
            # advances through a *distinct work item* (record_id 5001/5002/5003,
            # app-001/002/003). That is batch progress, not a loop.
            #
            # Discriminator (the task's "lack of progress" framing): batch
            # iteration carries a strictly-distinct *identifier-like* work key
            # (numbers or single-token IDs); a genuine oscillation's only
            # distinct values are free-text restatements of the same decision.
            # Dominance keeps the original (looser) veto unchanged.
            is_cycle = bool(evidence.get("cycle_length"))
            veto = self._cluster_is_batch_iteration(
                states, cluster_labels, evidence, require_identifier_like=is_cycle
            )
            if veto:
                return None

            return self._build_clustering_result(
                embeddings, cluster_labels, recent_window, n_clusters, evidence
            )
        except Exception:
            return None

    def _cluster_is_batch_iteration(
        self, states, cluster_labels, evidence, *, require_identifier_like: bool = False
    ) -> bool:
        """v1.6: Return True if the clusters look like batch iteration rather
        than a loop. Batch iteration: within a cluster (same kind of work),
        there is a non-bookkeeping state_delta key whose values are all
        strictly distinct — the agent is processing item 1, item 2, item 3…

        v1.7: When ``require_identifier_like`` is set, the strictly-distinct key
        must also be *identifier-like* (numbers, single-token IDs, or a
        templated counter) to count. Used for cluster cycles, where a genuine
        oscillation loop restates the same decision in free text each turn
        (distinct but not identifier-like) while real call→result batch
        iteration advances a work-item id/counter.
        """
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
                raw_values = []
                hashable = []
                for s in cluster_states:
                    if k not in s.state_delta:
                        continue
                    v = s.state_delta[k]
                    raw_values.append(v)
                    try:
                        hash(v)
                        hashable.append(v)
                    except TypeError:
                        hashable.append(json.dumps(v, sort_keys=True, default=str))
                if len(hashable) == len(cluster_states) and len(set(hashable)) == len(hashable):
                    if require_identifier_like and not self._values_are_identifier_like(raw_values):
                        continue
                    return True
        return False

    @staticmethod
    def _values_are_identifier_like(values) -> bool:
        """v1.7: True when every value is a work-item identifier — a number or a
        single-token string (record ids like 'tick-5001', slugs, counters). A
        free-text value (multi-word, e.g. an agent restating "approach A is
        more suitable") is NOT identifier-like: distinct free-text across a
        cluster is paraphrase, not batch progress. Booleans are rejected
        (True/False flip-flop is not a work-item sequence).

        v1.8: a templated counter is also identifier-like — distinct values that
        share one skeleton once digit runs are blanked ("step 0".."step 5",
        "page 2"/"page 3") are a prefix+counter work sequence, not paraphrase.
        Without this the cluster-cycle veto flagged monotonically advancing
        steps (which only differ by an incrementing number) as an oscillation."""
        if not values:
            return False

        def _atomic_id(v) -> bool:
            if isinstance(v, bool):
                return False
            if isinstance(v, (int, float)):
                return True
            return isinstance(v, str) and len(v.split()) == 1

        if all(_atomic_id(v) for v in values):
            return True

        # Templated counter: every value is a string reducing to the same
        # digit-blanked skeleton, the values are distinct, and at least one
        # carries a digit (a real counter, not constant free text).
        if all(isinstance(v, str) for v in values):
            skeletons = {re.sub(r"\d+", "#", v).strip() for v in values}
            has_digit = any(ch.isdigit() for v in values for ch in v)
            if len(skeletons) == 1 and has_digit and len(set(values)) == len(values):
                return True
        return False

    def _find_cluster_pattern(self, recent_labels, cluster_labels) -> Optional[dict]:
        """Analyze cluster assignments for dominance or cyclic patterns."""
        cluster_counts: Dict[Any, int] = {}
        for label in recent_labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1

        max_cluster_count = max(cluster_counts.values())
        dominant_cluster = max(cluster_counts, key=cluster_counts.__getitem__)
        dominance_ratio = max_cluster_count / len(recent_labels)

        # Check for cyclic patterns in cluster sequence
        cycle_length = 0
        for potential_cycle in range(2, len(recent_labels) // 2 + 1):
            pattern = tuple(recent_labels[-potential_cycle:])
            check_against = tuple(recent_labels[-2 * potential_cycle : -potential_cycle])
            if pattern == check_against:
                cycle_length = potential_cycle
                break

        evidence: Dict[str, Any] = {}
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

    def _cluster_semantic_similarity(
        self,
        embeddings,
        cluster_labels,
        recent_window: int,
        evidence: dict,
    ) -> float:
        """Measure semantic recurrence for dominance and cycle-only patterns."""
        dominant_cluster = evidence.get("dominant_cluster")
        if dominant_cluster is not None:
            cluster_indices = [
                i for i, label in enumerate(cluster_labels) if label == dominant_cluster
            ]
            cluster_embs = [embeddings[i] for i in cluster_indices[-5:]]
            similarities = [
                self.embedder.similarity(cluster_embs[i], cluster_embs[j])
                for i in range(len(cluster_embs))
                for j in range(i + 1, len(cluster_embs))
            ]
            return sum(similarities) / len(similarities) if similarities else 0.0

        cycle_length = int(evidence.get("cycle_length", 0))
        if cycle_length < 2 or recent_window < cycle_length * 2:
            return 0.0

        cycle_start = len(cluster_labels) - cycle_length * 2
        similarities = [
            self.embedder.similarity(
                embeddings[index],
                embeddings[index + cycle_length],
            )
            for index in range(cycle_start, cycle_start + cycle_length)
        ]
        return sum(similarities) / len(similarities) if similarities else 0.0

    def _build_clustering_result(
        self, embeddings, cluster_labels, recent_window: int, n_clusters: int, evidence: dict
    ) -> Optional[LoopDetectionResult]:
        """Validate cluster semantics against the configured threshold."""
        semantic_similarity = self._cluster_semantic_similarity(
            embeddings,
            cluster_labels,
            recent_window,
            evidence,
        )
        evidence["avg_intra_cluster_similarity"] = round(semantic_similarity, 4)
        evidence["semantic_threshold"] = self.semantic_threshold
        if semantic_similarity <= self.semantic_threshold:
            return None

        cluster_distribution = evidence.get("cluster_distribution", {})
        max_cluster_count = evidence.get(
            "cluster_count",
            max(cluster_distribution.values(), default=0),
        )
        dominance_ratio = evidence.get(
            "dominance_ratio",
            max_cluster_count / recent_window if recent_window else 0.0,
        )
        evidence["n_clusters"] = n_clusters

        raw_score = dominance_ratio * 0.5 + semantic_similarity * 0.5

        dominant_cluster = evidence.get("dominant_cluster")
        loop_start_index: Optional[int]
        if dominant_cluster is None:
            loop_start_index = len(cluster_labels) - 2 * int(evidence["cycle_length"])
        else:
            loop_start_index = next(
                (
                    i
                    for i in range(len(cluster_labels) - recent_window, len(cluster_labels))
                    if cluster_labels[i] == dominant_cluster
                ),
                None,
            )

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
        """Alias for detect_loop — kept for backward compatibility.

        detect_loop() already runs structural → hash → content_fingerprint →
        semantic → lexical → clustering in that order; this used to re-call
        clustering after, which paid the embedding+KMeans cost twice on traces
        that didn't loop.
        """
        return self.detect_loop(states)


loop_detector = MultiLevelLoopDetector()
