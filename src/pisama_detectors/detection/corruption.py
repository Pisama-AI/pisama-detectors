"""
F7: Memory/State Corruption Detection (MAST Taxonomy)
======================================================

Detects when agent memory or state becomes corrupted, including:
- Structured state corruption (type drift, schema violations)
- Text-based context corruption (ignoring related context)
- Cross-field inconsistencies
- Hallucinated references

Version History:
- v1.0: Initial implementation (structured state only)
- v1.1: Improved for adversarial accuracy:
  - Text-based context analysis for ignored context
  - Related topic extraction and verification
  - Narrow focus detection patterns
  - Task dependency awareness
"""

# Detector version for tracking
DETECTOR_VERSION = "1.1"
DETECTOR_NAME = "SemanticCorruptionDetector"

from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import re


@dataclass
class CorruptionIssue:
    issue_type: str
    field: Optional[str]
    message: str
    severity: str


@dataclass
class CorruptionResult:
    detected: bool
    confidence: float
    issues: List[CorruptionIssue]
    issue_count: int
    max_severity: str
    raw_score: Optional[float] = None
    calibration_info: Optional[Dict[str, Any]] = None


@dataclass
class StateSnapshot:
    state_delta: dict
    agent_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class VelocityConfig:
    """Configuration for state change velocity thresholds."""
    window_seconds: float = 5.0
    max_changes_per_window: int = 10
    high_velocity_fields: List[str] = field(default_factory=lambda: [
        "counter", "count", "iteration", "step", "progress",
        "timestamp", "updated_at", "last_seen", "version",
    ])
    ignore_velocity_for_types: List[type] = field(default_factory=lambda: [bool])


@dataclass
class Schema:
    fields: Dict[str, type]
    required_fields: List[str]


class SemanticCorruptionDetector:
    """
    Detects F7: Memory/State Corruption in agent systems.

    Handles both structured state corruption and text-based context corruption.
    """

    # v1.1: Related topic mappings for context verification
    # When a task mentions key topic, related topics should typically be addressed
    RELATED_TOPICS = {
        'authentication': ['session', 'token', 'login', 'logout', 'credential', 'password', 'security'],
        'auth': ['session', 'token', 'login', 'logout', 'credential', 'password', 'security'],
        'login': ['session', 'authentication', 'token', 'logout', 'security'],
        'database': ['transaction', 'connection', 'query', 'migration', 'schema', 'index'],
        'api': ['endpoint', 'request', 'response', 'error handling', 'validation', 'rate limit'],
        'payment': ['refund', 'transaction', 'billing', 'subscription', 'invoice'],
        'user': ['profile', 'permission', 'role', 'account', 'preference'],
        'cache': ['invalidation', 'expiration', 'refresh', 'consistency'],
        'file': ['permission', 'path', 'encoding', 'backup', 'cleanup'],
        'error': ['logging', 'handling', 'recovery', 'notification', 'retry'],
        'test': ['coverage', 'assertion', 'mock', 'fixture', 'edge case'],
        'deployment': ['rollback', 'release', 'staging', 'production', 'ci/cd', 'pipeline', 'version'],
        'networking': ['dns', 'firewall', 'load balancer', 'proxy', 'ssl', 'certificate', 'port'],
        'monitoring': ['alert', 'metric', 'logging', 'dashboard', 'threshold', 'uptime'],
        'security': ['encryption', 'vulnerability', 'audit', 'compliance', 'access control', 'firewall'],
        'config': ['environment variable', 'secret', 'setting', 'parameter', 'flag', 'option'],
        'configuration': ['environment variable', 'secret', 'setting', 'parameter', 'flag', 'option'],
        'migration': ['schema', 'rollback', 'version', 'data transfer', 'compatibility', 'backup'],
        'notification': ['email', 'webhook', 'alert', 'push', 'sms', 'template'],
        'search': ['index', 'query', 'relevance', 'filter', 'ranking', 'pagination'],
        'queue': ['consumer', 'producer', 'dead letter', 'retry', 'ordering', 'backpressure'],
        'messaging': ['consumer', 'producer', 'dead letter', 'retry', 'ordering', 'backpressure'],
        'storage': ['bucket', 'upload', 'download', 'permission', 'lifecycle', 'backup'],
        'logging': ['level', 'format', 'rotation', 'aggregation', 'filter', 'structured'],
        'session': ['cookie', 'token', 'expiration', 'renewal', 'invalidation', 'storage'],
        'scheduling': ['cron', 'interval', 'retry', 'timeout', 'concurrency', 'queue'],
        'backup': ['restore', 'snapshot', 'retention', 'encryption', 'verification', 'schedule'],
        'encryption': ['key', 'certificate', 'hash', 'salt', 'algorithm', 'rotation'],
        'permission': ['role', 'access', 'grant', 'deny', 'scope', 'policy'],
        'integration': ['webhook', 'api', 'sync', 'mapping', 'transform', 'retry'],
        'webhook': ['endpoint', 'payload', 'signature', 'retry', 'timeout', 'validation'],
        'email': ['template', 'delivery', 'bounce', 'spam', 'attachment', 'queue'],
    }

    # v1.1: Patterns indicating narrow focus (potential context ignorance)
    NARROW_FOCUS_PATTERNS = [
        (r'\bonly\s+(?:addressed|fixed|updated|changed|modified)\b', 'explicit_narrow_focus'),
        (r'\bjust\s+(?:the|this|that)\s+\w+\b', 'narrow_scope'),
        (r'\bfocused\s+(?:specifically|only|solely)\s+on\b', 'explicit_narrow_focus'),
        (r'\bwithout\s+(?:touching|changing|modifying|affecting)\b', 'explicit_exclusion'),
        (r'\bdidn\'t\s+(?:touch|change|modify|address)\b', 'explicit_exclusion'),
        (r'\bignored?\s+(?:the|any)\b', 'explicit_ignorance'),
        (r'\bspecific\s+(?:fix|change|update)\b', 'narrow_scope'),
    ]

    # v1.1: Patterns indicating comprehensive handling (reduces false positives)
    COMPREHENSIVE_PATTERNS = [
        r'\balso\s+(?:updated|checked|verified|ensured)\b',
        r'\badditionally\b',
        r'\brelated\s+(?:to|changes|updates)\b',
        r'\bensured?\s+(?:that|consistency)\b',
        r'\bconsidered?\s+(?:the|all|related)\b',
        r'\bacross\s+(?:the|all|related)\b',
    ]

    def __init__(
        self,
        velocity_config: Optional[VelocityConfig] = None,
        confidence_scaling: float = 1.0,
    ):
        email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        url_pattern = re.compile(r'^https?://[a-zA-Z0-9][-a-zA-Z0-9]*(\.[a-zA-Z0-9][-a-zA-Z0-9]*)+(/.*)?$')
        
        self.domain_validators: Dict[str, Callable] = {
            "age": lambda v: isinstance(v, (int, float)) and 0 <= v <= 150,
            "price": lambda v: isinstance(v, (int, float)) and v >= 0,
            "percentage": lambda v: isinstance(v, (int, float)) and 0 <= v <= 100,
            "email": lambda v: isinstance(v, str) and bool(email_pattern.match(v)),
            "url": lambda v: isinstance(v, str) and bool(url_pattern.match(v)),
            "phone": lambda v: isinstance(v, str) and len(re.sub(r'\D', '', v)) >= 10,
            "uuid": lambda v: isinstance(v, str) and len(v) == 36 and v.count('-') == 4,
        }
        self.known_ids: set = set()
        self.velocity_config = velocity_config or VelocityConfig()
        self._change_history: Dict[str, deque] = {}
        self._field_velocities: Dict[str, float] = {}
        self.confidence_scaling = confidence_scaling
    
    def _is_high_velocity_field(self, field_name: str) -> bool:
        """Check if a field is expected to change rapidly."""
        field_lower = field_name.lower()
        return any(
            kw in field_lower 
            for kw in self.velocity_config.high_velocity_fields
        )
    
    def _update_velocity_tracking(self, field: str, timestamp: datetime) -> float:
        """Update velocity tracking for a field and return current velocity."""
        if field not in self._change_history:
            self._change_history[field] = deque(maxlen=100)
        
        self._change_history[field].append(timestamp)
        
        window_start = timestamp - timedelta(seconds=self.velocity_config.window_seconds)
        recent_changes = [
            t for t in self._change_history[field]
            if t >= window_start
        ]
        
        velocity = len(recent_changes) / self.velocity_config.window_seconds
        self._field_velocities[field] = velocity
        return velocity
    
    def _should_suppress_for_velocity(
        self,
        field: str,
        value: Any,
        timestamp: datetime,
    ) -> bool:
        """Check if a change should be suppressed due to expected high velocity."""
        if self._is_high_velocity_field(field):
            return True

        if type(value) in self.velocity_config.ignore_velocity_for_types:
            return False

        velocity = self._update_velocity_tracking(field, timestamp)

        if velocity > self.velocity_config.max_changes_per_window / self.velocity_config.window_seconds:
            return True

        return False

    def _detect_type_drift(
        self,
        prev_state: StateSnapshot,
        current_state: StateSnapshot,
    ) -> List[CorruptionIssue]:
        """Detect type changes in fields between states."""
        issues = []
        prev_data = prev_state.state_delta
        curr_data = current_state.state_delta

        # Check for type changes in common fields
        common_fields = set(prev_data.keys()) & set(curr_data.keys())
        for field in common_fields:
            prev_val = prev_data[field]
            curr_val = curr_data[field]

            prev_type = type(prev_val)
            curr_type = type(curr_val)

            # Detect type drift (e.g., int -> str)
            if prev_type != curr_type:
                # Allow None transitions
                if prev_val is None or curr_val is None:
                    continue

                # Allow numeric conversions (int <-> float)
                if {prev_type, curr_type} <= {int, float}:
                    continue

                issues.append(CorruptionIssue(
                    issue_type="type_drift",
                    field=field,
                    message=f"Type changed from {prev_type.__name__} to {curr_type.__name__}",
                    severity="high",
                ))

        return issues

    @staticmethod
    def _flatten_nested_dicts(state_delta: dict, prefix: str = "") -> dict:
        """Flatten nested dicts so inner fields are exposed for corruption checks.

        E.g. ``{'json': {'salary': 125000}}`` → ``{'json.salary': 125000}``.
        Only recurses one level deep to avoid excessive expansion.
        """
        flat: dict = {}
        for key, value in state_delta.items():
            full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            if isinstance(value, dict) and value:
                # Recurse one level into nested dicts
                flat.update(SemanticCorruptionDetector._flatten_nested_dicts(value, full_key))
            else:
                flat[full_key] = value
        return flat

    def detect_corruption(
        self,
        prev_state: StateSnapshot,
        current_state: StateSnapshot,
        schema: Optional[Schema] = None,
    ) -> List[CorruptionIssue]:
        issues = []

        if schema:
            issues.extend(self._validate_schema(current_state, schema))

        # Detect type drift between states (even without schema)
        issues.extend(self._detect_type_drift(prev_state, current_state))

        issues.extend(self._validate_cross_field_consistency(current_state))
        issues.extend(self._validate_domain_constraints(current_state))
        issues.extend(self._detect_hallucinated_references(prev_state, current_state))
        issues.extend(self._detect_value_copying(current_state))
        issues.extend(self._detect_suspicious_rapid_changes(prev_state, current_state))
        issues.extend(self._detect_anomalous_value_changes(prev_state, current_state))

        # Also check flattened nested dicts (e.g. n8n entries with 'json' wrapper).
        # Only if there are nested dicts that differ between states.
        prev_flat = self._flatten_nested_dicts(prev_state.state_delta)
        curr_flat = self._flatten_nested_dicts(current_state.state_delta)
        if prev_flat != prev_state.state_delta or curr_flat != current_state.state_delta:
            flat_prev = StateSnapshot(state_delta=prev_flat, agent_id=prev_state.agent_id, timestamp=prev_state.timestamp)
            flat_curr = StateSnapshot(state_delta=curr_flat, agent_id=current_state.agent_id, timestamp=current_state.timestamp)
            # Only run value-change detection on flattened data.
            # Skip _detect_type_drift (field_disappeared too noisy for nested keys)
            # and schema checks (dotted keys don't match domain validators).
            issues.extend(self._detect_anomalous_value_changes(flat_prev, flat_curr))

        filtered_issues = self._apply_velocity_filtering(issues, current_state)

        return filtered_issues
    
    def detect_corruption_with_confidence(
        self,
        prev_state: StateSnapshot,
        current_state: StateSnapshot,
        schema: Optional[Schema] = None,
    ) -> CorruptionResult:
        issues = self.detect_corruption(prev_state, current_state, schema)
        
        max_severity = "low"
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for issue in issues:
            sev = issue.severity
            if sev in severity_counts:
                severity_counts[sev] += 1
            if self._severity_rank(sev) > self._severity_rank(max_severity):
                max_severity = sev
        
        raw_score = self._calculate_raw_score(issues, severity_counts)
        confidence, calibration_info = self._calibrate_confidence(
            issues=issues,
            severity_counts=severity_counts,
            max_severity=max_severity,
            raw_score=raw_score,
        )
        
        return CorruptionResult(
            detected=len(issues) > 0,
            confidence=confidence,
            issues=issues,
            issue_count=len(issues),
            max_severity=max_severity if issues else "none",
            raw_score=raw_score,
            calibration_info=calibration_info,
        )
    
    def _severity_rank(self, severity: str) -> int:
        ranks = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return ranks.get(severity, 0)
    
    def _calculate_raw_score(
        self,
        issues: List[CorruptionIssue],
        severity_counts: Dict[str, int],
    ) -> float:
        if not issues:
            return 0.0
        
        score = (
            severity_counts.get("low", 0) * 0.1 +
            severity_counts.get("medium", 0) * 0.25 +
            severity_counts.get("high", 0) * 0.4 +
            severity_counts.get("critical", 0) * 0.6
        )
        
        return min(1.0, score)
    
    def _calibrate_confidence(
        self,
        issues: List[CorruptionIssue],
        severity_counts: Dict[str, int],
        max_severity: str,
        raw_score: float,
    ) -> Tuple[float, Dict[str, Any]]:
        if not issues:
            return 0.0, {
                "issue_count": 0,
                "severity_counts": severity_counts,
                "max_severity": "none",
                "raw_score": 0.0,
                "confidence_scaling": self.confidence_scaling,
            }
        
        severity_weight = {
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95,
        }.get(max_severity, 0.5)
        
        issue_types = set(i.issue_type for i in issues)
        diversity_factor = min(1.0, len(issue_types) / 4)
        
        issue_factor = min(0.3, len(issues) * 0.05)
        
        base_confidence = (
            severity_weight * 0.40 +
            raw_score * 0.30 +
            diversity_factor * 0.15 +
            issue_factor
        )
        
        calibrated = min(0.99, base_confidence * self.confidence_scaling)
        
        calibration_info = {
            "issue_count": len(issues),
            "severity_counts": severity_counts,
            "max_severity": max_severity,
            "severity_weight": severity_weight,
            "diversity_factor": round(diversity_factor, 4),
            "issue_types": list(issue_types),
            "raw_score": round(raw_score, 4),
            "confidence_scaling": self.confidence_scaling,
        }
        
        return round(calibrated, 4), calibration_info
    
    # Issue types that should NEVER be suppressed by velocity filtering,
    # even on high-velocity fields like "version" or "timestamp".
    _VELOCITY_IMMUNE_ISSUES = {
        "type_drift", "monotonic_regression", "sign_flip",
        "status_regression", "data_loss", "content_replacement",
        "identity_mutation", "regression_nullification",
    }

    def _apply_velocity_filtering(
        self,
        issues: List[CorruptionIssue],
        state: StateSnapshot,
    ) -> List[CorruptionIssue]:
        """Filter out issues for fields with expected high velocity."""
        filtered = []

        for issue in issues:
            # Never suppress high-severity corruption signals
            if issue.issue_type in self._VELOCITY_IMMUNE_ISSUES:
                filtered.append(issue)
                continue

            if issue.field is None:
                filtered.append(issue)
                continue

            fields = issue.field.split(",")
            should_suppress = False

            for field in fields:
                field = field.strip()
                value = state.state_delta.get(field)

                if self._should_suppress_for_velocity(field, value, state.timestamp):
                    should_suppress = True
                    break

            if not should_suppress:
                filtered.append(issue)

        return filtered
    
    def _detect_suspicious_rapid_changes(
        self,
        prev: StateSnapshot,
        current: StateSnapshot,
    ) -> List[CorruptionIssue]:
        """Detect suspiciously rapid changes that don't fit expected patterns."""
        issues = []
        
        time_delta = (current.timestamp - prev.timestamp).total_seconds()
        if time_delta <= 0:
            time_delta = 0.001
        
        changed_fields = []
        for key in set(prev.state_delta.keys()) | set(current.state_delta.keys()):
            prev_val = prev.state_delta.get(key)
            curr_val = current.state_delta.get(key)
            
            if prev_val != curr_val and not self._is_high_velocity_field(key):
                changed_fields.append(key)
        
        change_rate = len(changed_fields) / time_delta
        
        if change_rate > 20 and len(changed_fields) > 5:
            issues.append(CorruptionIssue(
                issue_type="suspicious_rapid_changes",
                field=",".join(changed_fields[:5]) + ("..." if len(changed_fields) > 5 else ""),
                message=f"Unusually high change rate: {len(changed_fields)} fields in {time_delta:.2f}s",
                severity="medium",
            ))
        
        return issues
    
    # Status fields and their expected progression order (forward-only)
    STATUS_PROGRESSIONS = {
        "status": ["pending", "processing", "shipped", "delivered", "completed"],
        "order_status": ["pending", "processing", "shipped", "delivered", "completed"],
        "state": ["created", "active", "suspended", "closed", "archived"],
        "workflow_status": ["draft", "review", "approved", "published"],
        "ticket_status": ["open", "in_progress", "resolved", "closed"],
    }

    # Fields that should only increase (monotonic)
    MONOTONIC_INCREASING_FIELDS = {
        "version", "revision", "build_number", "sequence",
        "last_login", "last_modified", "last_updated", "updated_at",
        "last_seen", "last_activity",
    }

    # Fields representing core identity (should not all change at once)
    IDENTITY_FIELDS = {"first_name", "last_name", "name", "username", "display_name"}

    # Known score-grade mappings
    GRADE_SCORE_RANGES = {
        "A": (90, 100), "A+": (97, 100), "A-": (90, 93),
        "B": (80, 89), "B+": (87, 89), "B-": (80, 83),
        "C": (70, 79), "D": (60, 69), "F": (0, 59),
    }

    def _detect_anomalous_value_changes(
        self,
        prev: StateSnapshot,
        current: StateSnapshot,
    ) -> List[CorruptionIssue]:
        """Detect anomalous value changes: sign flips, extreme magnitude shifts,
        status regressions, monotonic violations, identity mutations, duplicates."""
        issues: List[CorruptionIssue] = []

        for key in set(prev.state_delta.keys()) & set(current.state_delta.keys()):
            prev_val = prev.state_delta[key]
            curr_val = current.state_delta[key]

            if prev_val == curr_val:
                continue

            # Nullification: non-None value replaced with None
            if prev_val is not None and curr_val is None:
                issues.append(CorruptionIssue(
                    issue_type="data_loss",
                    field=key,
                    message=f"Field nullified: {str(prev_val)[:50]} → None",
                    severity="high",
                ))
                continue

            # Type drift inline: different types for same key (not involving None)
            if type(prev_val) is not type(curr_val) and prev_val is not None and curr_val is not None:
                # Skip int↔float which is benign
                if not (isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float))):
                    issues.append(CorruptionIssue(
                        issue_type="type_drift",
                        field=key,
                        message=f"Type changed: {type(prev_val).__name__} → {type(curr_val).__name__}",
                        severity="high",
                    ))

            # Numeric anomalies (exclude booleans — True/False transitions are normal)
            if (isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float))
                    and not isinstance(prev_val, bool) and not isinstance(curr_val, bool)):
                # Sign flip
                if prev_val > 0 and curr_val < 0:
                    issues.append(CorruptionIssue(
                        issue_type="sign_flip",
                        field=key,
                        message=f"Value flipped sign: {prev_val} → {curr_val}",
                        severity="high",
                    ))
                # Extreme magnitude change (>5x or >90% drop)
                elif prev_val != 0:
                    ratio = abs(curr_val / prev_val)
                    if ratio > 5.0 or ratio < 0.1:
                        issues.append(CorruptionIssue(
                            issue_type="extreme_magnitude_change",
                            field=key,
                            message=f"Value changed by {ratio:.1f}x: {prev_val} → {curr_val}",
                            severity="medium",
                        ))

                # Monotonic field regression (version, build_number, etc.)
                key_lower = key.lower()
                if any(mf in key_lower for mf in self.MONOTONIC_INCREASING_FIELDS):
                    if curr_val < prev_val:
                        issues.append(CorruptionIssue(
                            issue_type="monotonic_regression",
                            field=key,
                            message=f"Monotonic field decreased: {prev_val} → {curr_val}",
                            severity="high",
                        ))

            # Boolean state change detection (security-relevant fields)
            elif isinstance(prev_val, bool) and isinstance(curr_val, bool):
                if prev_val != curr_val:
                    security_booleans = {
                        "authenticated", "enabled", "active", "verified", "locked",
                        "authorized", "valid", "approved", "is_admin", "is_active",
                        "is_verified", "is_enabled", "is_locked", "is_authorized",
                        "has_access", "is_valid", "confirmed", "is_confirmed",
                    }
                    key_lower = key.lower()
                    is_security = key_lower in security_booleans or any(
                        sb in key_lower for sb in security_booleans
                    )
                    if is_security:
                        issues.append(CorruptionIssue(
                            issue_type="security_boolean_flip",
                            field=key,
                            message=f"Security-relevant boolean flipped: {prev_val} → {curr_val}",
                            severity="high",
                        ))

            # String anomalies
            elif isinstance(prev_val, str) and isinstance(curr_val, str):
                # String cleared (non-empty → empty)
                if len(prev_val) > 5 and len(curr_val) == 0:
                    issues.append(CorruptionIssue(
                        issue_type="string_cleared",
                        field=key,
                        message=f"Non-empty string cleared: '{prev_val[:50]}...' → ''",
                        severity="high",
                    ))
                # String drastically truncated (>80% shorter)
                elif len(prev_val) > 20 and len(curr_val) > 0 and len(curr_val) < len(prev_val) * 0.2:
                    issues.append(CorruptionIssue(
                        issue_type="string_truncated",
                        field=key,
                        message=f"String drastically truncated: {len(prev_val)} → {len(curr_val)} chars",
                        severity="medium",
                    ))

                if len(prev_val) > 5 and len(curr_val) > 5:
                    prev_words = set(prev_val.lower().split())
                    curr_words = set(curr_val.lower().split())
                    if prev_words and curr_words:
                        overlap = len(prev_words & curr_words)
                        total = len(prev_words | curr_words)
                        if total > 3 and overlap / total < 0.1:
                            issues.append(CorruptionIssue(
                                issue_type="content_replacement",
                                field=key,
                                message=f"Content completely replaced (<10% word overlap)",
                                severity="medium",
                            ))

                # Status/enum regression detection
                key_lower = key.lower()
                for status_field, progression in self.STATUS_PROGRESSIONS.items():
                    if status_field in key_lower:
                        prev_lower = prev_val.lower().strip()
                        curr_lower = curr_val.lower().strip()
                        if prev_lower in progression and curr_lower in progression:
                            prev_idx = progression.index(prev_lower)
                            curr_idx = progression.index(curr_lower)
                            if curr_idx < prev_idx:
                                issues.append(CorruptionIssue(
                                    issue_type="status_regression",
                                    field=key,
                                    message=f"Status regressed: '{prev_val}' → '{curr_val}'",
                                    severity="high",
                                ))
                        break

                # Monotonic string fields (timestamps as strings)
                if any(mf in key_lower for mf in self.MONOTONIC_INCREASING_FIELDS):
                    if curr_val < prev_val:  # String comparison works for ISO dates
                        issues.append(CorruptionIssue(
                            issue_type="monotonic_regression",
                            field=key,
                            message=f"Monotonic field decreased: '{prev_val}' → '{curr_val}'",
                            severity="high",
                        ))

            # List anomalies: items lost or duplicated
            elif isinstance(prev_val, list) and isinstance(curr_val, list):
                if len(prev_val) > 0 and len(curr_val) == 0:
                    issues.append(CorruptionIssue(
                        issue_type="data_loss",
                        field=key,
                        message=f"List emptied: {len(prev_val)} items → 0",
                        severity="high",
                    ))
                elif len(prev_val) > 0 and len(curr_val) < len(prev_val) * 0.5:
                    issues.append(CorruptionIssue(
                        issue_type="data_loss",
                        field=key,
                        message=f"List shrunk significantly: {len(prev_val)} → {len(curr_val)}",
                        severity="medium",
                    ))

                # Duplicate item detection in lists of dicts
                if curr_val and isinstance(curr_val[0], dict):
                    id_key = None
                    for candidate in ("id", "item_id", "key", "name"):
                        if all(candidate in item for item in curr_val if isinstance(item, dict)):
                            id_key = candidate
                            break
                    if id_key:
                        ids = [item[id_key] for item in curr_val if isinstance(item, dict)]
                        if len(ids) != len(set(ids)):
                            from collections import Counter
                            dupes = [v for v, c in Counter(ids).items() if c > 1]
                            issues.append(CorruptionIssue(
                                issue_type="duplicate_items",
                                field=key,
                                message=f"Duplicate items by '{id_key}': {dupes[:3]}",
                                severity="high",
                            ))

            # Dict anomalies: keys lost
            elif isinstance(prev_val, dict) and isinstance(curr_val, dict):
                lost_keys = set(prev_val.keys()) - set(curr_val.keys())
                if lost_keys and len(lost_keys) > len(prev_val) * 0.5:
                    issues.append(CorruptionIssue(
                        issue_type="data_loss",
                        field=key,
                        message=f"Dict lost {len(lost_keys)} keys: {list(lost_keys)[:3]}",
                        severity="medium",
                    ))

        # Check for fields that disappeared entirely.
        # Only flag when multiple fields vanish (single field disappearance
        # is often an optional/nullable field in n8n payloads).
        lost_fields = set(prev.state_delta.keys()) - set(current.state_delta.keys())
        non_velocity_lost = [f for f in lost_fields if not self._is_high_velocity_field(f)]
        if len(non_velocity_lost) >= 3:
            issues.append(CorruptionIssue(
                issue_type="field_disappeared",
                field=",".join(list(non_velocity_lost)[:3]),
                message=f"{len(non_velocity_lost)} fields disappeared: {list(non_velocity_lost)[:3]}",
                severity="high",
            ))

        # Cross-field semantic inconsistency: score vs grade
        curr_data = current.state_delta
        if "score" in curr_data and "grade" in curr_data:
            score = curr_data["score"]
            grade = curr_data["grade"]
            if isinstance(score, (int, float)) and isinstance(grade, str):
                grade_upper = grade.upper().strip()
                if grade_upper in self.GRADE_SCORE_RANGES:
                    lo, hi = self.GRADE_SCORE_RANGES[grade_upper]
                    if not (lo <= score <= hi):
                        issues.append(CorruptionIssue(
                            issue_type="cross_field_inconsistency",
                            field="score,grade",
                            message=f"Score {score} inconsistent with grade '{grade}' (expected {lo}-{hi})",
                            severity="high",
                        ))

        # Identity field mutation detection: multiple identity fields changed at once
        # while a stable identifier (email, user_id) stayed the same
        common_keys = set(prev.state_delta.keys()) & set(current.state_delta.keys())
        changed_identity = [
            k for k in common_keys
            if k.lower() in self.IDENTITY_FIELDS
            and prev.state_delta[k] != current.state_delta[k]
        ]
        stable_identifiers = [
            k for k in common_keys
            if k.lower() in {"email", "user_id", "account_id", "id"}
            and prev.state_delta[k] == current.state_delta[k]
        ]
        if len(changed_identity) >= 2 and stable_identifiers:
            issues.append(CorruptionIssue(
                issue_type="identity_mutation",
                field=",".join(changed_identity),
                message=f"Multiple identity fields changed ({changed_identity}) while {stable_identifiers} stayed the same",
                severity="high",
            ))

        # Nullification alongside regression: dependent field set to null
        # when a status-type field regresses
        status_regressed = any(i.issue_type == "status_regression" for i in issues)
        if status_regressed:
            for key in common_keys:
                if current.state_delta[key] is None and prev.state_delta[key] is not None:
                    issues.append(CorruptionIssue(
                        issue_type="regression_nullification",
                        field=key,
                        message=f"Field '{key}' nullified alongside status regression",
                        severity="medium",
                    ))

        return issues

    def _validate_schema(self, state: StateSnapshot, schema: Schema) -> List[CorruptionIssue]:
        issues = []
        data = state.state_delta
        
        for key, value in data.items():
            if key not in schema.fields:
                issues.append(CorruptionIssue(
                    issue_type="hallucinated_key",
                    field=key,
                    message=f"Key '{key}' not in schema",
                    severity="medium",
                ))
            elif not isinstance(value, schema.fields[key]):
                issues.append(CorruptionIssue(
                    issue_type="type_drift",
                    field=key,
                    message=f"Expected {schema.fields[key].__name__}, got {type(value).__name__}",
                    severity="high",
                ))
        
        for required in schema.required_fields:
            if required not in data:
                issues.append(CorruptionIssue(
                    issue_type="missing_field",
                    field=required,
                    message=f"Required field '{required}' missing",
                    severity="high",
                ))
        
        return issues
    
    def _validate_cross_field_consistency(self, state: StateSnapshot) -> List[CorruptionIssue]:
        issues = []
        data = state.state_delta
        
        if "start_date" in data and "end_date" in data:
            try:
                if data["start_date"] > data["end_date"]:
                    issues.append(CorruptionIssue(
                        issue_type="cross_field_inconsistency",
                        field="start_date,end_date",
                        message="Start date after end date",
                        severity="high",
                    ))
            except (TypeError, ValueError):
                pass
        
        if "min_value" in data and "max_value" in data:
            try:
                if data["min_value"] > data["max_value"]:
                    issues.append(CorruptionIssue(
                        issue_type="cross_field_inconsistency",
                        field="min_value,max_value",
                        message="Min value greater than max value",
                        severity="high",
                    ))
            except (TypeError, ValueError):
                pass
        
        return issues
    
    def _validate_domain_constraints(self, state: StateSnapshot) -> List[CorruptionIssue]:
        issues = []
        data = state.state_delta
        
        for key, validator in self.domain_validators.items():
            if key in data:
                try:
                    if not validator(data[key]):
                        issues.append(CorruptionIssue(
                            issue_type="domain_violation",
                            field=key,
                            message=f"Value {data[key]} violates domain constraint for {key}",
                            severity="medium",
                        ))
                except Exception:
                    issues.append(CorruptionIssue(
                        issue_type="domain_violation",
                        field=key,
                        message=f"Could not validate {key}",
                        severity="low",
                    ))
        
        return issues
    
    def _detect_hallucinated_references(
        self,
        prev: StateSnapshot,
        current: StateSnapshot,
    ) -> List[CorruptionIssue]:
        issues = []
        data = current.state_delta
        
        id_fields = [k for k in data if k.endswith("_id")]
        for field in id_fields:
            ref_id = data[field]
            if isinstance(ref_id, str) and ref_id not in self.known_ids:
                if prev.state_delta.get(field) != ref_id:
                    issues.append(CorruptionIssue(
                        issue_type="hallucinated_reference",
                        field=field,
                        message=f"Reference '{ref_id}' may not exist",
                        severity="high",
                    ))
        
        return issues
    
    def _detect_value_copying(self, state: StateSnapshot) -> List[CorruptionIssue]:
        issues = []
        data = state.state_delta
        values = list(data.values())
        keys = list(data.keys())
        
        for i, v1 in enumerate(values):
            if not isinstance(v1, str) or len(v1) <= 10:
                continue
            for j, v2 in enumerate(values[i + 1:], i + 1):
                if v1 == v2:
                    issues.append(CorruptionIssue(
                        issue_type="suspicious_value_copy",
                        field=f"{keys[i]},{keys[j]}",
                        message=f"Identical values in different fields",
                        severity="low",
                    ))
        
        return issues
    
    def register_known_id(self, id_value: str):
        self.known_ids.add(id_value)

    # =========================================================================
    # v1.1: Text-based context corruption detection
    # =========================================================================

    def _extract_task_topics(self, task: str) -> List[str]:
        """v1.1: Extract key topics from task description."""
        task_lower = task.lower()
        found_topics = []

        for topic in self.RELATED_TOPICS.keys():
            if topic in task_lower:
                found_topics.append(topic)

        return found_topics

    def _get_expected_related_topics(self, task_topics: List[str]) -> List[str]:
        """v1.1: Get related topics that should be addressed given task topics."""
        related = set()
        for topic in task_topics:
            if topic in self.RELATED_TOPICS:
                related.update(self.RELATED_TOPICS[topic])
        return list(related)

    def _check_narrow_focus_patterns(self, output: str) -> List[tuple]:
        """v1.1: Check for patterns indicating narrow/incomplete focus."""
        found_patterns = []
        output_lower = output.lower()

        for pattern, pattern_type in self.NARROW_FOCUS_PATTERNS:
            if re.search(pattern, output_lower):
                found_patterns.append((pattern_type, pattern))

        return found_patterns

    def _check_comprehensive_patterns(self, output: str) -> bool:
        """v1.1: Check if output shows signs of comprehensive handling."""
        output_lower = output.lower()

        for pattern in self.COMPREHENSIVE_PATTERNS:
            if re.search(pattern, output_lower):
                return True

        return False

    def _stem(self, word: str) -> str:
        """Simple suffix stripping for topic matching."""
        for suffix in ('ment', 'tion', 'sion', 'ing', 'ed', 'er', 'ly', 'es', 's'):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    def _check_related_topics_addressed(
        self,
        output: str,
        expected_related: List[str],
    ) -> tuple[List[str], List[str]]:
        """v1.1: Check which related topics are addressed vs missing."""
        output_lower = output.lower()
        addressed = []
        missing = []

        for topic in expected_related:
            # Check if topic or its stemmed variants are mentioned
            if topic in output_lower:
                addressed.append(topic)
            else:
                # Also try stemmed form of the topic
                stemmed = self._stem(topic)
                if stemmed != topic and stemmed in output_lower:
                    addressed.append(topic)
                else:
                    missing.append(topic)

        return addressed, missing

    def detect_from_text(
        self,
        task: str,
        output: str,
        context: Optional[str] = None,
    ) -> CorruptionResult:
        """
        v1.1: Detect context/memory corruption in text-based outputs.

        Checks if the output ignores related context that should be addressed
        given the task description.

        Args:
            task: The task description/instruction
            output: The agent's output text
            context: Optional additional context that was provided

        Returns:
            CorruptionResult with detection status and issues
        """
        issues = []

        # Extract topics from task
        task_topics = self._extract_task_topics(task)

        if not task_topics:
            # No recognized topics, can't verify context handling
            return CorruptionResult(
                detected=False,
                confidence=0.0,
                issues=[],
                issue_count=0,
                max_severity="none",
            )

        # Get expected related topics
        expected_related = self._get_expected_related_topics(task_topics)

        # Check for narrow focus patterns
        narrow_patterns = self._check_narrow_focus_patterns(output)

        # Check for comprehensive handling
        is_comprehensive = self._check_comprehensive_patterns(output)

        # Check which related topics are addressed
        addressed, missing = self._check_related_topics_addressed(output, expected_related)

        # Determine if there's a context corruption issue
        # Issue detected if: narrow focus patterns found + missing related topics + not comprehensive
        if narrow_patterns and missing and not is_comprehensive:
            issues.append(CorruptionIssue(
                issue_type="context_ignored",
                field=",".join(task_topics),
                message=f"Task mentions {task_topics} but output shows narrow focus, missing: {missing[:3]}",
                severity="medium",
            ))

        # Also flag if significant related topics missing without explicit narrow focus
        # (but only if many related topics are expected and most are missing)
        if len(missing) >= 2 and len(missing) > len(addressed) and not is_comprehensive:
            issues.append(CorruptionIssue(
                issue_type="incomplete_context",
                field=",".join(missing[:3]),
                message=f"Related topics not addressed: {missing[:3]}",
                severity="low",
            ))

        # NEW: If zero related topics are addressed and 2+ are missing,
        # the output completely ignored the context domain
        if not addressed and len(missing) >= 2 and not is_comprehensive:
            issues.append(CorruptionIssue(
                issue_type="context_completely_ignored",
                field=",".join(task_topics),
                message=f"Output addresses none of the related topics for {task_topics}: missing {missing[:3]}",
                severity="medium",
            ))

        # Check context parameter if provided
        if context:
            context_keywords = self._extract_context_keywords(context)
            missing_context = [kw for kw in context_keywords if kw.lower() not in output.lower()]
            if len(missing_context) > len(context_keywords) * 0.7:  # More than 70% missing
                issues.append(CorruptionIssue(
                    issue_type="context_not_used",
                    field="context",
                    message=f"Provided context largely ignored: {missing_context[:3]}",
                    severity="high",
                ))

        # Calculate result
        if not issues:
            return CorruptionResult(
                detected=False,
                confidence=0.0,
                issues=[],
                issue_count=0,
                max_severity="none",
            )

        max_severity = max((self._severity_rank(i.severity) for i in issues), default=0)
        severity_map = {0: "low", 1: "medium", 2: "high", 3: "critical"}

        # Confidence based on number of indicators
        confidence = min(0.9, 0.3 + len(issues) * 0.2 + len(narrow_patterns) * 0.1)

        return CorruptionResult(
            detected=True,
            confidence=round(confidence, 2),
            issues=issues,
            issue_count=len(issues),
            max_severity=severity_map.get(max_severity, "low"),
        )

    def _extract_context_keywords(self, context: str) -> List[str]:
        """v1.1: Extract key terms from context string."""
        # Extract significant words (nouns, technical terms)
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]{3,}\b', context)
        # Filter common words
        stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'will', 'would', 'should', 'could', 'when', 'where', 'what', 'which', 'there', 'their', 'them', 'then', 'than', 'also', 'just', 'only', 'some', 'more', 'other', 'into', 'your', 'about'}
        return [w for w in words if w.lower() not in stopwords][:10]

    def detect(
        self,
        task: str,
        output: str,
        context: Optional[str] = None,
    ) -> CorruptionResult:
        """
        v1.1: Main detection entry point for text-based detection.

        Convenience method that wraps detect_from_text.
        """
        return self.detect_from_text(task, output, context)


corruption_detector = SemanticCorruptionDetector()
