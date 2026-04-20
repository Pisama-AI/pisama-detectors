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

DETECTOR_VERSION = "1.0"
DETECTOR_NAME = "ConvergenceDetector"

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


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
        """
        self.plateau_threshold = plateau_threshold
        self.plateau_window = plateau_window
        self.regression_tolerance = regression_tolerance
        self.thrashing_min_reversals = thrashing_min_reversals
        self.min_steps = min_steps

    def detect_convergence_issues(
        self,
        metrics: List[Dict[str, Any]],
        direction: str = "minimize",
        window_size: Optional[int] = None,
    ) -> ConvergenceResult:
        """Detect convergence issues in a metric time series.

        Args:
            metrics: List of dicts with at least 'value' key.
                     Optional: 'step', 'label'.
            direction: 'minimize' (lower is better) or 'maximize' (higher is better).
            window_size: Override plateau_window for this call.

        Returns:
            ConvergenceResult with detection outcome.
        """
        if not metrics or len(metrics) < self.min_steps:
            return ConvergenceResult(
                detected=False,
                confidence=0.0,
                evidence={"reason": "insufficient_data", "num_steps": len(metrics) if metrics else 0},
            )

        values = [m["value"] for m in metrics]
        window = window_size or self.plateau_window
        issues: List[ConvergenceIssue] = []

        # Track best value
        if direction == "minimize":
            best_value = min(values)
            best_idx = values.index(best_value)
            is_better = lambda a, b: a < b
        else:
            best_value = max(values)
            best_idx = values.index(best_value)
            is_better = lambda a, b: a > b

        current_value = values[-1]
        steps_since_best = len(values) - 1 - best_idx

        # --- Check 1: Regression ---
        regression_issue = self._check_regression(
            values, best_value, current_value, direction, steps_since_best,
        )
        if regression_issue:
            issues.append(regression_issue)

        # --- Check 2: Plateau ---
        plateau_issue = self._check_plateau(values, direction, window)
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
        recent = values[-min(window, len(values)):]
        if len(recent) >= 2:
            if direction == "minimize":
                improvement_rate = (recent[0] - recent[-1]) / max(abs(recent[0]), 1e-10) / len(recent)
            else:
                improvement_rate = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-10) / len(recent)
        else:
            improvement_rate = 0.0

        # --- Overall trend guard ---
        # If the metric improved substantially from start to end (>30% in
        # the right direction), suppress MINOR regression/thrashing issues.
        # These are normal fluctuations in a clearly converging series
        # (e.g., noisy SGD, cosine annealing, stepwise learning rate).
        first_val, last_val = values[0], values[-1]
        scale = max(abs(first_val), abs(last_val), 1e-10)
        if direction == "minimize":
            overall_improvement = (first_val - last_val) / scale
        else:
            overall_improvement = (last_val - first_val) / scale
        # Check smoothness: count direction reversals over the full series.
        # If more than 50% of steps reverse direction, the series oscillates
        # and the "improvement" is just lucky start/end alignment.
        reversals = sum(
            1 for i in range(2, len(values))
            if (values[i] - values[i-1]) * (values[i-1] - values[i-2]) < 0
        )
        max_reversals = max(1, len(values) - 2)
        is_smooth = (reversals / max_reversals) < 0.60
        if overall_improvement > 0.30 and is_smooth:
            issues = [
                i for i in issues
                if i.failure_type not in (ConvergenceFailureType.REGRESSION, ConvergenceFailureType.THRASHING)
            ]
        # When improvement is very high, tail-end plateau is expected
        # (metric has converged, further improvement diminishes)
        if overall_improvement > 0.50:
            issues = [
                i for i in issues
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

        # Pick the primary issue. Divergence and thrashing are more specific
        # than regression (regression fires whenever we're past best, which
        # always co-occurs with divergence/thrashing). Prefer the more
        # explanatory failure type.
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
            key=lambda i: (type_priority.get(i.failure_type, 0), severity_rank[i.severity]),
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
    ) -> Optional[ConvergenceIssue]:
        """Check if current value has regressed past the best."""
        if steps_since_best == 0:
            return None

        if best_value == 0:
            regression_frac = abs(current_value - best_value)
        else:
            regression_frac = abs(current_value - best_value) / abs(best_value)

        # Check direction
        if direction == "minimize":
            regressed = current_value > best_value * (1 + self.regression_tolerance)
        else:
            regressed = current_value < best_value * (1 - self.regression_tolerance)

        if not regressed:
            return None

        confidence = min(1.0, regression_frac / max(self.regression_tolerance, 1e-10))

        if regression_frac > 0.1:
            severity = ConvergenceSeverity.CRITICAL
        elif regression_frac > 0.05:
            severity = ConvergenceSeverity.SEVERE
        elif regression_frac > self.regression_tolerance:
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
            evidence={"confidence": confidence, "regression_frac": regression_frac,
                      "steps_since_best": steps_since_best},
        )

    def _check_plateau(
        self,
        values: List[float],
        direction: str,
        window: int,
    ) -> Optional[ConvergenceIssue]:
        """Check if metric has plateaued (no meaningful improvement)."""
        recent = values[-min(window, len(values)):]
        if len(recent) < 2:
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

        if avg_improvement >= self.plateau_threshold:
            return None

        # Count how many steps have negligible improvement
        stalled_steps = sum(1 for ni in normalized_improvements if ni < self.plateau_threshold)
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
                f"over last {len(recent)} steps (threshold: {self.plateau_threshold}). "
                f"{stalled_steps}/{len(normalized_improvements)} steps showed no meaningful progress."
            ),
            severity=severity,
            evidence={"confidence": confidence, "avg_improvement": avg_improvement,
                      "stalled_steps": stalled_steps, "stall_ratio": stall_ratio},
        )

    def _check_thrashing(
        self,
        values: List[float],
        window: int,
    ) -> Optional[ConvergenceIssue]:
        """Check if metric is oscillating (frequent direction changes)."""
        recent = values[-min(window, len(values)):]
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
            evidence={"confidence": confidence, "reversals": reversals,
                      "reversal_ratio": reversal_ratio,
                      "max_possible_reversals": max_possible_reversals},
        )

    def _check_divergence(
        self,
        values: List[float],
        direction: str,
        window: int,
    ) -> Optional[ConvergenceIssue]:
        """Check if metric is consistently moving in the wrong direction."""
        recent = values[-min(window, len(values)):]
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
            evidence={"confidence": confidence, "wrong_ratio": wrong_ratio,
                      "total_change": total_change, "normalized_change": normalized_change},
        )


# Singleton instance
convergence_detector = ConvergenceDetector()
