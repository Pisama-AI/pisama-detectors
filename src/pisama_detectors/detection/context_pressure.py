"""
F20: Context Pressure Detection
================================

Detects when an agent's output quality degrades due to context window
saturation — the "context anxiety" failure mode identified in Anthropic's
harness design research.

Unlike the overflow detector (which checks raw token count vs limit),
this detector looks for *behavioral* signals that context pressure is
affecting output quality:

- Declining output verbosity in later states
- Premature wrap-up language ("I'll leave that for now")
- Scope narrowing (addressing fewer subtasks over time)
- Quality cliff (abrupt drop in detail in the final states)
- Token trajectory approaching model context limit

Reference: Anthropic Engineering Blog, "Harness Design for Long-Running
Application Development" (2026)
"""

DETECTOR_VERSION = "1.0"
DETECTOR_NAME = "ContextPressureDetector"

import logging
import re
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PressureSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PressureSignal:
    signal_type: str
    description: str
    strength: float  # 0.0-1.0
    evidence: Optional[str] = None


@dataclass
class ContextPressureResult:
    detected: bool
    confidence: float
    severity: PressureSeverity
    context_utilization: float  # 0.0-1.0, cumulative tokens / limit
    output_decline_ratio: float  # later_avg / earlier_avg (< 1.0 = decline)
    premature_signals: List[str] = field(default_factory=list)
    cliff_detected: bool = False
    signals: List[PressureSignal] = field(default_factory=list)
    raw_score: Optional[float] = None
    calibration_info: Optional[Dict[str, Any]] = None


# Premature wrap-up language patterns (from Anthropic's observations)
WRAPUP_PATTERNS = [
    (r"\bI'?ll leave (?:that|this|the rest)\b", "premature_leave"),
    (r"\bthis should be sufficient\b", "sufficiency_claim"),
    (r"\bwrapping up\b", "explicit_wrapup"),
    (r"\bfor brevity\b", "brevity_excuse"),
    (r"\bI'?ll skip\b", "explicit_skip"),
    (r"\bleaving (?:that|this|the rest) for\b", "deferred_work"),
    (r"\bfor now\b.*\b(?:move on|proceed|continue)\b", "deferred_continuation"),
    (r"\blet me (?:quickly|briefly) (?:summarize|wrap|finish)\b", "rushed_conclusion"),
    (r"\bI'?ll (?:just )?note that\b.*\binstead of\b", "shortcut_acknowledgment"),
    (r"\bdue to (?:space|length|context) (?:constraints|limitations)\b", "explicit_constraint"),
    (r"\bI'?ve covered the (?:key|main|essential) (?:points|parts)\b", "selective_coverage"),
    (r"\bthe remaining (?:items|tasks|points) (?:are|can be) (?:similar|straightforward)\b", "assumed_trivial"),
]

# Compile patterns once
_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), label) for p, label in WRAPUP_PATTERNS]

# Default context limits by model family
DEFAULT_CONTEXT_LIMITS = {
    "claude-3": 200_000,
    "claude-3.5": 200_000,
    "claude-4": 200_000,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5": 16_384,
}


class ContextPressureDetector:
    """Detects context-pressure-induced quality degradation."""

    def __init__(
        self,
        decline_threshold: float = 0.50,
        cliff_sigma: float = 2.0,
        min_states: int = 4,
        wrapup_weight: float = 0.20,
        decline_weight: float = 0.30,
        utilization_weight: float = 0.25,
        cliff_weight: float = 0.25,
    ):
        self.decline_threshold = decline_threshold
        self.cliff_sigma = cliff_sigma
        self.min_states = min_states
        self.wrapup_weight = wrapup_weight
        self.decline_weight = decline_weight
        self.utilization_weight = utilization_weight
        self.cliff_weight = cliff_weight

    def detect(
        self,
        states: List[Dict[str, Any]],
        context_limit: Optional[int] = None,
        task_complexity: Optional[str] = None,
    ) -> ContextPressureResult:
        """Detect context pressure signals in a sequence of agent states.

        Args:
            states: List of state dicts with keys:
                - token_count (int): tokens used in this state
                - state_delta (dict/str): output content
                - sequence_num (int): position in trace
            context_limit: Model context window size (auto-detected if None)
            task_complexity: Optional task description for scope analysis
        """
        if len(states) < self.min_states:
            return ContextPressureResult(
                detected=False, confidence=0.0, severity=PressureSeverity.NONE,
                context_utilization=0.0, output_decline_ratio=1.0,
            )

        signals: List[PressureSignal] = []

        # Extract output lengths per state
        output_lengths = []
        cumulative_tokens = 0
        token_counts = []
        outputs = []

        for s in states:
            delta = s.get("state_delta", {})
            if isinstance(delta, dict):
                text = delta.get("output", "") or delta.get("response", "") or str(delta)
            else:
                text = str(delta) if delta else ""
            output_lengths.append(len(text))
            outputs.append(text)
            tc = s.get("token_count", 0) or 0
            cumulative_tokens += tc
            token_counts.append(cumulative_tokens)

        # --- Signal 1: Token trajectory / context utilization ---
        if context_limit is None:
            context_limit = self._infer_context_limit(states)
        utilization = cumulative_tokens / context_limit if context_limit > 0 else 0.0
        util_score = 0.0
        if utilization > 0.85:
            util_score = min((utilization - 0.70) / 0.30, 1.0)
            signals.append(PressureSignal(
                signal_type="high_utilization",
                description=f"Context {utilization:.0%} utilized ({cumulative_tokens:,}/{context_limit:,} tokens)",
                strength=util_score,
            ))

        # --- Signal 2: Output length decline ---
        n = len(output_lengths)
        split = max(n // 3, 1)
        early_avg = statistics.mean(output_lengths[:split]) if output_lengths[:split] else 1.0
        late_avg = statistics.mean(output_lengths[-split:]) if output_lengths[-split:] else 1.0
        decline_ratio = late_avg / max(early_avg, 1.0)
        decline_score = 0.0
        if decline_ratio < self.decline_threshold:
            decline_score = min((1.0 - decline_ratio) / 0.50, 1.0)
            signals.append(PressureSignal(
                signal_type="output_decline",
                description=f"Output length declined {1 - decline_ratio:.0%} (early avg {early_avg:.0f} → late avg {late_avg:.0f} chars)",
                strength=decline_score,
            ))

        # --- Signal 3: Premature wrap-up language ---
        wrapup_matches = []
        # Only scan the last 40% of states (where pressure manifests)
        scan_start = int(n * 0.6)
        for text in outputs[scan_start:]:
            lower_text = text.lower()
            for pattern, label in _COMPILED_PATTERNS:
                if pattern.search(lower_text):
                    wrapup_matches.append(label)
        wrapup_score = min(len(set(wrapup_matches)) / 3.0, 1.0)
        if wrapup_matches:
            signals.append(PressureSignal(
                signal_type="premature_wrapup",
                description=f"Found {len(set(wrapup_matches))} wrap-up language patterns in final states",
                strength=wrapup_score,
                evidence=", ".join(sorted(set(wrapup_matches))[:5]),
            ))

        # --- Signal 4: Quality cliff ---
        cliff_detected = False
        cliff_score = 0.0
        if n >= 5 and statistics.stdev(output_lengths) > 0:
            mean_len = statistics.mean(output_lengths)
            std_len = statistics.stdev(output_lengths)
            tail_start = int(n * 0.8)
            tail_lengths = output_lengths[tail_start:]
            for tl in tail_lengths:
                if tl < mean_len - self.cliff_sigma * std_len:
                    cliff_detected = True
                    break
            if cliff_detected:
                cliff_score = 0.8
                signals.append(PressureSignal(
                    signal_type="quality_cliff",
                    description=f"Abrupt output length drop (>{self.cliff_sigma}σ) in final 20% of states",
                    strength=cliff_score,
                ))

        # --- Aggregate confidence ---
        confidence = (
            self.utilization_weight * util_score
            + self.decline_weight * decline_score
            + self.wrapup_weight * wrapup_score
            + self.cliff_weight * cliff_score
        )
        confidence = min(confidence, 1.0)

        # Require at least 2 signals to detect (ensemble gate)
        active_signals = sum(1 for s in signals if s.strength > 0)
        detected = active_signals >= 2 and confidence >= 0.30

        # Severity
        if confidence >= 0.80:
            severity = PressureSeverity.CRITICAL
        elif confidence >= 0.60:
            severity = PressureSeverity.HIGH
        elif confidence >= 0.40:
            severity = PressureSeverity.MEDIUM
        elif detected:
            severity = PressureSeverity.LOW
        else:
            severity = PressureSeverity.NONE

        return ContextPressureResult(
            detected=detected,
            confidence=confidence,
            severity=severity,
            context_utilization=utilization,
            output_decline_ratio=decline_ratio,
            premature_signals=sorted(set(wrapup_matches)),
            cliff_detected=cliff_detected,
            signals=signals,
        )

    def _infer_context_limit(self, states: List[Dict[str, Any]]) -> int:
        """Infer context limit from state metadata or default to 128k."""
        for s in states:
            delta = s.get("state_delta", {})
            if isinstance(delta, dict):
                model = delta.get("model", "")
                if model:
                    model_lower = model.lower()
                    for prefix, limit in DEFAULT_CONTEXT_LIMITS.items():
                        if prefix in model_lower:
                            return limit
        return 128_000  # Safe default


# Singleton for convenience
context_pressure_detector = ContextPressureDetector()
