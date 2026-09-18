"""
Convergence Issue Detection for Iterative Agent Systems
========================================================

Detects when an iterative agent system exhibits convergence problems:
- Plateau: Metric improvement stalls for extended periods
- Regression: Metric worsens past a previously achieved best
- Thrashing: Metric oscillates without a clear trend
- Divergence: Metric consistently trends in the wrong direction

Designed for autonomous research agents (e.g., Karpathy's autoresearch),
iterative code generation, and any agent system that produces numerical
performance signals over multiple iterations.

Version History:
- v1.0: Initial implementation with plateau, regression, thrashing, divergence
"""

DETECTOR_VERSION = "1.1"
DETECTOR_NAME = "ConvergenceDetector"

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Per-metric-type plateau / regression thresholds. Loss-family metrics
# (val_bpb, train_loss, perplexity) are intrinsically noisier and tolerate
# larger regressions; ranking metrics (f1, accuracy) demand tighter detection
# because a 1pp drop is a real signal. Pulled out of __init__ so callers can
# override per-call without mutating shared detector state.
_DEFAULT_TYPE_THRESHOLDS: Dict[str, Dict[str, float]] = {
    # Noisy reward landscapes — let larger jitter slide before flagging.
    "loss": {"plateau_threshold": 0.01, "regression_tolerance": 0.03},
    "reward": {"plateau_threshold": 0.02, "regression_tolerance": 0.02},
    # Bounded ranking metrics — tighter is better.
    "f1": {"plateau_threshold": 0.005, "regression_tolerance": 0.02},
    "accuracy": {"plateau_threshold": 0.005, "regression_tolerance": 0.02},
}


# Map raw metric_name (as it appears in caller payload or golden input_data)
# onto a normalized metric_type key used by _DEFAULT_TYPE_THRESHOLDS. Keep
# this permissive — unknown names return None, falling back to instance
# thresholds.
_METRIC_NAME_TO_TYPE: Dict[str, str] = {
    # loss / perplexity family
    "loss": "loss",
    "train_loss": "loss",
    "val_loss": "loss",
    "validation_loss": "loss",
    "ce_loss": "loss",
    "mse": "loss",
    "mae": "loss",
    "perplexity": "loss",
    "ppl": "loss",
    "val_bpb": "loss",
    "bpb": "loss",
    "bits_per_byte": "loss",
    # reward family
    "reward": "reward",
    "return": "reward",
    "score": "reward",
    "value": "reward",
    # ranking metrics
    "f1": "f1",
    "f1_score": "f1",
    "macro_f1": "f1",
    "micro_f1": "f1",
    "accuracy": "accuracy",
    "acc": "accuracy",
    "top1": "accuracy",
    "top5": "accuracy",
}


def _normalize_metric_type(metric_name: Optional[str]) -> Optional[str]:
    """Coerce a raw metric_name into a normalized type key, or None.

    Case-insensitive. Unknown names return None — the caller falls back to
    its configured instance thresholds.
    """
    if not metric_name:
        return None
    return _METRIC_NAME_TO_TYPE.get(metric_name.strip().lower())


class ConvergenceFailureType(str, Enum):
    PLATEAU = "plateau"
    REGRESSION = "regression"
    THRASHING = "thrashing"
    DIVERGENCE = "divergence"


class ConvergenceSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class ConvergenceIssue:
    failure_type: ConvergenceFailureType
    description: str
    severity: ConvergenceSeverity
    evidence: Optional[Dict[str, Any]] = None


@dataclass
class ConvergenceResult:
    detected: bool
    confidence: float
    failure_type: Optional[str] = None
    severity: ConvergenceSeverity = ConvergenceSeverity.NONE
    best_value: Optional[float] = None
    current_value: Optional[float] = None
    improvement_rate: Optional[float] = None
    steps_since_best: Optional[int] = None
    issues: List[ConvergenceIssue] = field(default_factory=list)
    evidence: Optional[Dict[str, Any]] = None
    raw_score: Optional[float] = None


class ConvergenceDetector:
    """Detects convergence issues in iterative agent metric sequences."""

    def __init__(
        self,
        plateau_threshold: float = 0.02,
        plateau_window: int = 10,
        regression_tolerance: float = 0.02,
        thrashing_min_reversals: int = 3,
        min_steps: int = 3,
        metric_type: Optional[str] = None,
        type_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Args:
            plateau_threshold: Min improvement rate per step to not be a plateau.
                Raised to 0.02 (from 0.01) to allow 10-15% variance before flagging.
            plateau_window: Number of recent steps to evaluate for plateau.
                Raised to 10 (from 8) to require more steps before declaring plateau.
            regression_tolerance: Max acceptable regression from best (as fraction of best).
            thrashing_min_reversals: Min direction changes in window to flag thrashing.
            min_steps: Minimum number of data points required for detection.
            metric_type: Default normalized metric_type ("loss" / "reward" /
                "f1" / "accuracy"). When set, lookups against ``type_thresholds``
                (or the built-in defaults) replace ``plateau_threshold`` and
                ``regression_tolerance`` per call — without mutating instance
                state. Can be overridden by ``detect_convergence_issues``.
            type_thresholds: Override the built-in per-type thresholds. Keys
                are normalized metric_type strings; values are dicts with
                "plateau_threshold" / "regression_tolerance" entries. Missing
                keys fall through to the built-in defaults.
        """
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.regression_tolerance = regression_tolerance
        self.thrashing_min_reversals = thrashing_min_reversals
        self.min_steps = min_steps
        self.metric_type = metric_type
        # Build the effective lookup table once. Caller overrides win over
        # the built-in defaults on per-key basis.
        merged: Dict[str, Dict[str, float]] = {
            k: dict(v) for k, v in _DEFAULT_TYPE_THRESHOLDS.items()
        }
        for k, v in (type_thresholds or {}).items():
            merged.setdefault(k, {}).update(v)
        self.type_thresholds = merged

    def _resolve_thresholds(
        self, metric_type: Optional[str]
    ) -> Tuple[float, float]:
        """Return (plateau_threshold, regression_tolerance) for this call.

        Resolution order:
            1. Per-call ``metric_type`` if recognized in ``type_thresholds``.
            2. Instance ``self.metric_type`` if recognized.
            3. Instance ``self.plateau_threshold`` / ``self.regression_tolerance``.

        Unknown ``metric_type`` values fall through to instance defaults —
        never raise — so the detector stays usable with novel metric names.
        """
        chosen = metric_type or self.metric_type
        if chosen and chosen in self.type_thresholds:
            entry = self.type_thresholds[chosen]
            return (
                entry.get("plateau_threshold", self.plateau_threshold),
                entry.get("regression_tolerance", self.regression_tolerance),
            )
        return self.plateau_threshold, self.regression_tolerance

    def detect_convergence_issues(
        self,
        metrics: List[Dict[str, Any]],
        direction: str = "minimize",
        window_size: Optional[int] = None,
        metric_type: Optional[str] = None,
        metric_name: Optional[str] = None,
    ) -> ConvergenceResult:
        """Detect convergence issues in a metric time series.

        Args:
            metrics: List of dicts with at least 'value' key.
                     Optional: 'step', 'label'.
            direction: 'minimize' (lower is better) or 'maximize' (higher is better).
            window_size: Override plateau_window for this call.
            metric_type: Optional normalized type ("loss"/"reward"/"f1"/
                "accuracy") routing to per-type thresholds. Wins over
                ``metric_name`` and the instance default. Unknown values
                fall back to instance thresholds.
            metric_name: Optional raw metric name (e.g. "val_bpb", "f1_score").
                Normalized via ``_normalize_metric_type``; only used when
                ``metric_type`` is None.

        Returns:
            ConvergenceResult with detection outcome.
        """
        if not metrics or len(metrics) < self.min_steps:
            return ConvergenceResult(
                detected=False,
                confidence=0.0,
                evidence={
                    "reason": "insufficient_data",
                    "num_steps": len(metrics) if metrics else 0,
                },
            )

        values = [m["value"] for m in metrics]
        window = window_size or self.plateau_window
        issues: List[ConvergenceIssue] = []

        # Resolve effective per-call thresholds without mutating instance
        # state — preserves thread/async safety. metric_type explicit beats
        # metric_name; both override the instance default.
        resolved_type = metric_type or _normalize_metric_type(metric_name)
        plateau_threshold, regression_tolerance = self._resolve_thresholds(resolved_type)

        # Track best value
        if direction == "minimize":
            best_value = min(values)
            best_idx = values.index(best_value)

            def is_better(a, b):
                return a < b
        else:
            best_value = max(values)
            best_idx = values.index(best_value)

        current_value = values[-1]
        steps_since_best = len(values) - 1 - best_idx

        # --- Check 1: Regression ---
        regression_issue = self._check_regression(
            values,
            best_value,
            current_value,
            direction,
            steps_since_best,
            regression_tolerance=regression_tolerance,
        )
        if regression_issue:
            issues.append(regression_issue)

        # --- Check 2: Plateau ---
        plateau_issue = self._check_plateau(
            values, direction, window, plateau_threshold=plateau_threshold,
        )
        if plateau_issue:
            issues.append(plateau_issue)

        # --- Check 3: Thrashing ---
        thrashing_issue = self._check_thrashing(values, window)
        if thrashing_issue:
            issues.append(thrashing_issue)

        # --- Check 4: Divergence ---
        divergence_issue = self._check_divergence(values, direction, window)
        if divergence_issue:
            issues.append(divergence_issue)

        # Compute improvement rate over recent window
        recent = values[-min(window, len(values)) :]
        if len(recent) >= 2:
            if direction == "minimize":
                improvement_rate = (
                    (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-10) / len(recent)
                )
            else:
                improvement_rate = (
                    (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-10) / len(recent)
                )
        else:
            improvement_rate = 0.0

        # --- Overall trend guard ---
        # If the metric improved substantially from start to end (>30% in
        # the right direction), suppress MINOR regression/thrashing issues
        # PROVIDED the current value is still at or near the best — those
        # are normal noisy-SGD fluctuations. When the current value has
        # drifted far from best, the regression is real even if first-to-
        # last still nets positive (e.g., 1.0 → best 0.5 → current 0.65
        # nets 0.35 overall, but a 30% regression off best is a real
        # signal, not a converged series).
        first_val, last_val = values[0], values[-1]
        scale = max(abs(first_val), abs(last_val), 1e-10)
        if direction == "minimize":
            overall_improvement = (first_val - last_val) / scale
            near_best = last_val <= best_value * (1 + regression_tolerance)
        else:
            overall_improvement = (last_val - first_val) / scale
            near_best = last_val >= best_value * (1 - regression_tolerance)
        # Check smoothness: count direction reversals over the full series.
        # If more than 50% of steps reverse direction, the series oscillates
        # and the "improvement" is just lucky start/end alignment.
        reversals = sum(
            1
            for i in range(2, len(values))
            if (values[i] - values[i - 1]) * (values[i - 1] - values[i - 2]) < 0
        )
        max_reversals = max(1, len(values) - 2)
        is_smooth = (reversals / max_reversals) < 0.60
        if overall_improvement > 0.30 and is_smooth and near_best:
            issues = [
                i
                for i in issues
                if i.failure_type
                not in (ConvergenceFailureType.REGRESSION, ConvergenceFailureType.THRASHING)
            ]
        # When improvement is very high, tail-end plateau is expected
        # (metric has converged, further improvement diminishes)
        if overall_improvement > 0.50:
            issues = [
                i
                for i in issues
                if i.failure_type != ConvergenceFailureType.PLATEAU
                or i.severity not in (ConvergenceSeverity.MINOR, ConvergenceSeverity.MODERATE)
            ]

        if not issues:
            return ConvergenceResult(
                detected=False,
                confidence=0.0,
                best_value=best_value,
                current_value=current_value,
                improvement_rate=improvement_rate,
                steps_since_best=steps_since_best,
            )

        # Pick the primary issue. Severity dominates: a CRITICAL regression
        # off-best is the right primary even if a minor plateau co-fires
        # on the same window. type_priority is a tiebreaker among issues
        # of equal severity — divergence > thrashing > plateau > regression,
        # because divergence/thrashing are more specific than plain
        # regression (regression mechanically co-occurs whenever we're
        # past best).
        type_priority = {
            ConvergenceFailureType.DIVERGENCE: 4,
            ConvergenceFailureType.THRASHING: 3,
            ConvergenceFailureType.PLATEAU: 2,
            ConvergenceFailureType.REGRESSION: 1,
        }
        severity_rank = {
            ConvergenceSeverity.CRITICAL: 4,
            ConvergenceSeverity.SEVERE: 3,
            ConvergenceSeverity.MODERATE: 2,
            ConvergenceSeverity.MINOR: 1,
            ConvergenceSeverity.NONE: 0,
        }
        issues.sort(
            key=lambda i: (severity_rank[i.severity], type_priority.get(i.failure_type, 0)),
            reverse=True,
        )
        primary = issues[0]

        # Aggregate confidence from the primary issue
        confidence = primary.evidence.get("confidence", 0.5) if primary.evidence else 0.5

        return ConvergenceResult(
            detected=True,
            confidence=confidence,
            failure_type=primary.failure_type.value,
            severity=primary.severity,
            best_value=best_value,
            current_value=current_value,
            improvement_rate=improvement_rate,
            steps_since_best=steps_since_best,
            issues=issues,
            evidence={
                "primary_failure": primary.failure_type.value,
                "issue_count": len(issues),
                "direction": direction,
                "num_steps": len(values),
                "window_size": window,
            },
            raw_score=confidence,
        )

    def _check_regression(
        self,
        values: List[float],
        best_value: float,
        current_value: float,
        direction: str,
        steps_since_best: int,
        regression_tolerance: Optional[float] = None,
    ) -> Optional[ConvergenceIssue]:
        """Check if current value has regressed past the best.

        ``regression_tolerance`` overrides ``self.regression_tolerance`` when
        provided — set per-call by metric_type routing.
        """
        if steps_since_best == 0:
            return None

        tol = regression_tolerance if regression_tolerance is not None else self.regression_tolerance

        # A *relative* regression is only meaningful when the best value is
        # meaningfully non-zero. When best_value is ~0 — e.g. a per-step token
        # series that contains one empty/initial step, so min(series) == 0 —
        # the old fallback `regression_frac = abs(current - best)` collapsed to
        # the raw current magnitude. `confidence = regression_frac / tol` then
        # saturated to 1.0 and severity to CRITICAL for ANY non-zero later
        # step, making a single zero-valued step a guaranteed critical fire.
        # Anchor "meaningfully non-zero" on the series scale so a tiny-but-
        # nonzero best can't anchor a near-infinite relative regression either;
        # skip the regression check when it cannot be computed. Plateau /
        # thrashing / divergence still run on the same series.
        series_scale = max((abs(v) for v in values), default=0.0)
        if abs(best_value) <= 1e-9 * max(series_scale, 1.0):
            return None

        regression_frac = abs(current_value - best_value) / abs(best_value)

        # Check direction
        if direction == "minimize":
            regressed = current_value > best_value * (1 + tol)
        else:
            regressed = current_value < best_value * (1 - tol)

        if not regressed:
            return None

        confidence = min(1.0, regression_frac / max(tol, 1e-10))

        if regression_frac > 0.1:
            severity = ConvergenceSeverity.CRITICAL
        elif regression_frac > 0.05:
            severity = ConvergenceSeverity.SEVERE
        elif regression_frac > tol:
            severity = ConvergenceSeverity.MODERATE
        else:
            severity = ConvergenceSeverity.MINOR

        return ConvergenceIssue(
            failure_type=ConvergenceFailureType.REGRESSION,
            description=(
                f"Metric regressed by {regression_frac:.1%} from best value "
                f"({best_value:.4f}) to current ({current_value:.4f}), "
                f"{steps_since_best} steps after best."
            ),
            severity=severity,
            evidence={
                "confidence": confidence,
                "regression_frac": regression_frac,
                "steps_since_best": steps_since_best,
            },
        )

    def _check_plateau(
        self,
        values: List[float],
        direction: str,
        window: int,
        plateau_threshold: Optional[float] = None,
    ) -> Optional[ConvergenceIssue]:
        """Check if metric has plateaued (no meaningful improvement).

        ``plateau_threshold`` overrides ``self.plateau_threshold`` when
        provided — set per-call by metric_type routing.
        """
        thr = plateau_threshold if plateau_threshold is not None else self.plateau_threshold

        recent = values[-min(window, len(values)) :]
        if len(recent) < 2:
            return None

        # Require the series to fill its plateau window. A plateau is "no
        # meaningful improvement sustained over `window` steps", so a series
        # shorter than the window has too few points to establish a stall.
        # (dify's 3-node Start->Classify->End workflows fired plateau on a
        # 3-point series against a window of 10 — 1,764 dogfood FPs.) The guard
        # is self-scaling: a genuine detection using a smaller window still
        # fires once its series fills that window; 10+-step trajectories are
        # unaffected.
        if len(values) < window:
            return None

        # Compute per-step improvement
        improvements = []
        for i in range(1, len(recent)):
            if direction == "minimize":
                imp = recent[i - 1] - recent[i]
            else:
                imp = recent[i] - recent[i - 1]
            improvements.append(imp)

        # Normalize by scale
        scale = max(abs(v) for v in recent) if any(v != 0 for v in recent) else 1.0
        normalized_improvements = [imp / scale for imp in improvements]
        avg_improvement = sum(normalized_improvements) / len(normalized_improvements)

        if avg_improvement >= thr:
            return None

        # Slow-but-steady exception: a series that improves on EVERY step,
        # even if each step is below the plateau threshold, is not a plateau —
        # it's a monotonic slow trajectory. Plateau requires actual stalled
        # steps, not just slow ones. The cumulative improvement floor (1%
        # of scale) keeps near-asymptotic oscillation from sneaking through.
        all_improving = all(ni > 0 for ni in normalized_improvements)
        cumulative = sum(normalized_improvements)
        if all_improving and cumulative >= 0.01:
            return None

        # Count how many steps have negligible improvement
        stalled_steps = sum(1 for ni in normalized_improvements if ni < thr)
        stall_ratio = stalled_steps / len(normalized_improvements)

        confidence = min(1.0, stall_ratio)

        if stall_ratio >= 0.9:
            severity = ConvergenceSeverity.SEVERE
        elif stall_ratio >= 0.7:
            severity = ConvergenceSeverity.MODERATE
        else:
            severity = ConvergenceSeverity.MINOR

        return ConvergenceIssue(
            failure_type=ConvergenceFailureType.PLATEAU,
            description=(
                f"Metric plateaued: avg improvement {avg_improvement:.6f} per step "
                f"over last {len(recent)} steps (threshold: {thr}). "
                f"{stalled_steps}/{len(normalized_improvements)} steps showed no meaningful progress."
            ),
            severity=severity,
            evidence={
                "confidence": confidence,
                "avg_improvement": avg_improvement,
                "stalled_steps": stalled_steps,
                "stall_ratio": stall_ratio,
            },
        )

    def _check_thrashing(
        self,
        values: List[float],
        window: int,
    ) -> Optional[ConvergenceIssue]:
        """Check if metric is oscillating (frequent direction changes)."""
        recent = values[-min(window, len(values)) :]
        if len(recent) < 3:
            return None

        # Count direction reversals
        reversals = 0
        for i in range(2, len(recent)):
            prev_dir = recent[i - 1] - recent[i - 2]
            curr_dir = recent[i] - recent[i - 1]
            if prev_dir * curr_dir < 0:  # Sign change
                reversals += 1

        max_possible_reversals = len(recent) - 2
        if max_possible_reversals == 0:
            return None

        if reversals < self.thrashing_min_reversals:
            return None

        reversal_ratio = reversals / max_possible_reversals
        confidence = min(1.0, reversal_ratio)

        if reversal_ratio >= 0.8:
            severity = ConvergenceSeverity.SEVERE
        elif reversal_ratio >= 0.6:
            severity = ConvergenceSeverity.MODERATE
        else:
            severity = ConvergenceSeverity.MINOR

        return ConvergenceIssue(
            failure_type=ConvergenceFailureType.THRASHING,
            description=(
                f"Metric thrashing: {reversals} direction reversals in {len(recent)} steps "
                f"(reversal ratio: {reversal_ratio:.0%}). "
                f"No consistent trend detected."
            ),
            severity=severity,
            evidence={
                "confidence": confidence,
                "reversals": reversals,
                "reversal_ratio": reversal_ratio,
                "max_possible_reversals": max_possible_reversals,
            },
        )

    def _check_divergence(
        self,
        values: List[float],
        direction: str,
        window: int,
    ) -> Optional[ConvergenceIssue]:
        """Check if metric is consistently moving in the wrong direction."""
        recent = values[-min(window, len(values)) :]
        if len(recent) < 3:
            return None

        # Count steps moving in wrong direction
        wrong_direction_steps = 0
        for i in range(1, len(recent)):
            if direction == "minimize":
                if recent[i] > recent[i - 1]:
                    wrong_direction_steps += 1
            else:
                if recent[i] < recent[i - 1]:
                    wrong_direction_steps += 1

        total_steps = len(recent) - 1
        wrong_ratio = wrong_direction_steps / total_steps

        if wrong_ratio < 0.7:
            return None

        # Compute magnitude of divergence
        if direction == "minimize":
            total_change = recent[-1] - recent[0]
        else:
            total_change = recent[0] - recent[-1]

        # Positive total_change means divergence
        if total_change <= 0:
            return None

        scale = max(abs(v) for v in recent) if any(v != 0 for v in recent) else 1.0
        normalized_change = total_change / scale

        confidence = min(1.0, wrong_ratio * min(1.0, normalized_change / 0.05))

        if wrong_ratio >= 0.9 and normalized_change > 0.1:
            severity = ConvergenceSeverity.CRITICAL
        elif wrong_ratio >= 0.8:
            severity = ConvergenceSeverity.SEVERE
        elif wrong_ratio >= 0.7:
            severity = ConvergenceSeverity.MODERATE
        else:
            severity = ConvergenceSeverity.MINOR

        return ConvergenceIssue(
            failure_type=ConvergenceFailureType.DIVERGENCE,
            description=(
                f"Metric diverging: {wrong_direction_steps}/{total_steps} steps moving "
                f"in wrong direction over last {len(recent)} steps. "
                f"Total change: {total_change:.4f} ({normalized_change:.1%} of scale)."
            ),
            severity=severity,
            evidence={
                "confidence": confidence,
                "wrong_ratio": wrong_ratio,
                "total_change": total_change,
                "normalized_change": normalized_change,
            },
        )


# Singleton instance
convergence_detector = ConvergenceDetector()
