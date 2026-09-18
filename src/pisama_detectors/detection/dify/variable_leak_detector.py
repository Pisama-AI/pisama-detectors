"""
Variable Leak Detection for Dify Workflows
===========================================

Scans all node outputs for sensitive data patterns including API keys,
passwords, PII, and environment variable references. Also detects
iteration variable scope leaks where child node outputs appear in
unrelated parent-scope nodes.

Dify-specific: recursively scans node inputs/outputs dicts and checks
iteration scope boundaries via parent_node_id.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Set

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

# Sensitive data patterns with confidence levels
SENSITIVE_PATTERNS: Dict[str, List[Dict[str, Any]]] = {
    "api_key": [
        {"pattern": re.compile(r"sk-[a-zA-Z0-9]{20,}"), "label": "OpenAI/generic API key"},
        {"pattern": re.compile(r"AKIA[A-Z0-9]{16}"), "label": "AWS access key"},
        {"pattern": re.compile(r"xox[bp]-[a-zA-Z0-9\-]+"), "label": "Slack token"},
        {"pattern": re.compile(r"ghp_[a-zA-Z0-9]{36}"), "label": "GitHub PAT"},
        {"pattern": re.compile(r"Bearer\s+[a-zA-Z0-9\-_.]{20,}"), "label": "Bearer token"},
        {"pattern": re.compile(r"token_[a-z0-9]{32}"), "label": "Generic token"},
    ],
    "password": [
        {
            "pattern": re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),
            "label": "Password assignment",
        },
        {"pattern": re.compile(r"passwd\s*[:=]\s*\S+", re.IGNORECASE), "label": "Passwd assignment"},
        {"pattern": re.compile(r"secret\s*[:=]\s*\S+", re.IGNORECASE), "label": "Secret assignment"},
        {
            "pattern": re.compile(r"credentials?\s*[:=]\s*\S+", re.IGNORECASE),
            "label": "Credentials assignment",
        },
    ],
    "pii": [
        {"pattern": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "label": "SSN pattern"},
        {"pattern": re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "label": "Credit card number"},
        {
            "pattern": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            "label": "Email address",
        },
    ],
    "env_var": [
        {"pattern": re.compile(r"\$\{ENV_[A-Z_]+\}"), "label": "ENV_ variable"},
        {"pattern": re.compile(r"\$SECRET_[A-Z_]+"), "label": "$SECRET_ variable"},
        {"pattern": re.compile(r"process\.env\.[A-Z_]+"), "label": "process.env reference"},
    ],
}

# Confidence per category. Phase 19b: bumped password/pii 0.7 → 0.8 so
# single-signal detections clear thr=0.75. A detected leak is a confident
# positive verdict; 0.7 sat just below the calibrated threshold and was
# the root of the v1-lite recall collapse on dify_variable_leak.
CATEGORY_CONFIDENCE: Dict[str, float] = {
    "api_key": 0.85,
    "password": 0.8,
    "pii": 0.8,
    "env_var": 0.65,
}


class DifyVariableLeakDetector(TurnAwareDetector):
    """Detects sensitive variable leakage in Dify workflow node outputs.

    Recursively scans all node outputs for API keys, passwords, PII,
    and environment variable references. Also checks for iteration
    scope leaks.
    """

    name = "DifyVariableLeakDetector"
    version = "1.0"
    supported_failure_modes = ["F14"]  # Information leakage

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Delegate to detect_workflow_run if metadata contains workflow_run."""
        workflow_run = (conversation_metadata or {}).get("workflow_run", {})
        if workflow_run:
            return self.detect_workflow_run(workflow_run)
        return self._no_detection("No workflow_run data provided")

    def detect_workflow_run(self, workflow_run: dict) -> TurnAwareDetectionResult:
        """Analyze Dify workflow run for variable leakage.

        Args:
            workflow_run: Dify workflow_run dict with nodes list.

        Returns:
            Detection result with leak findings.
        """
        nodes = workflow_run.get("nodes", [])
        if not nodes:
            return self._no_detection("No nodes in workflow run")

        issues: List[Dict[str, Any]] = []
        affected_node_ids: List[str] = []
        max_confidence = 0.0

        # Scan each node's outputs for sensitive patterns
        for node in nodes:
            node_id = node.get("node_id", "")
            node_title = node.get("title", "")
            outputs = node.get("outputs", {})

            # Recursively extract all string values from outputs
            strings = self._extract_strings(outputs)

            for text in strings:
                for category, patterns in SENSITIVE_PATTERNS.items():
                    for pat_info in patterns:
                        for match in pat_info["pattern"].finditer(text):
                            cat_conf = CATEGORY_CONFIDENCE.get(category, 0.5)
                            max_confidence = max(max_confidence, cat_conf)
                            affected_node_ids.append(node_id)
                            issues.append(
                                {
                                    "type": "sensitive_data",
                                    "category": category,
                                    "label": pat_info["label"],
                                    "node_id": node_id,
                                    "title": node_title,
                                    "matched_preview": self._redact(match.group()),
                                    # The redacted preview is deliberately lossy;
                                    # use a non-reversible fingerprint when deciding
                                    # whether several distinct addresses leaked.
                                    "_match_fingerprint": hashlib.sha256(
                                        match.group().strip().casefold().encode("utf-8")
                                    ).hexdigest(),
                                    "confidence": cat_conf,
                                }
                            )

        # Bulk-exposure gate for contact PII. The email pattern is a bare
        # address regex applied to every string in every node output, so it
        # cannot tell an exfiltrated address list from the ONE address a
        # workflow exists to handle. A newsletter double-opt-in confirming a
        # subscriber's own address back to that subscriber is the entire
        # business payload, not a leak — and as written, every CRM, helpdesk
        # and notification workflow is structurally unable to pass.
        #
        # A single distinct address is therefore not treated as a leak on its
        # own. It still counts when the run exposes SEVERAL distinct addresses
        # (bulk exposure) or when a credential/secret category also fired,
        # which is what an actual leak looks like.
        issues = self._gate_contact_pii(issues)
        for issue in issues:
            issue.pop("_match_fingerprint", None)

        # Check for iteration scope leaks
        scope_leaks = self._check_scope_leaks(nodes)
        for leak in scope_leaks:
            issues.append(leak)

        if not issues:
            return self._no_detection("No sensitive data or scope leaks detected")

        # Derive attribution from the surviving evidence. Contact-PII gating
        # can remove every issue associated with a node, so retaining the
        # pre-gate list would report nodes that have no emitted finding.
        affected_node_ids = [
            str(issue.get("node_id") or issue.get("target_node_id") or "")
            for issue in issues
            if issue.get("node_id") or issue.get("target_node_id")
        ]

        # max_confidence was accumulated before the PII gate ran; recompute it
        # from the surviving issues so a gated-away category cannot keep
        # inflating the score.
        surviving: List[float] = [
            float(i["confidence"]) for i in issues if isinstance(i.get("confidence"), (int, float))
        ]
        max_confidence = max(surviving) if surviving else 0.0

        confidence = max_confidence if max_confidence > 0 else 0.6

        # Severity based on category
        categories_found: Set[str] = set()
        for issue in issues:
            category_value = issue.get("category")
            if isinstance(category_value, str):
                categories_found.add(category_value)
        has_api_key = "api_key" in categories_found
        has_pii = "pii" in categories_found
        has_scope_leak = any(i.get("type") == "scope_leak" for i in issues)

        if has_api_key or (has_pii and len(issues) >= 2):
            severity = TurnAwareSeverity.SEVERE
        elif has_pii or has_scope_leak or len(issues) >= 3:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F14",
            explanation=(
                f"Variable leak: {len(issues)} sensitive pattern(s) found "
                f"across workflow nodes (categories: {', '.join(categories_found)})"
            ),
            # affected_turns=[] — was a count masquerading as indices; real
            # node IDs go in evidence below.
            affected_turns=[],
            evidence={
                "affected_node_ids": sorted(set(affected_node_ids)),
                "issues": issues,
                "categories_found": list(categories_found),
                "total_nodes_scanned": len(nodes),
            },
            suggested_fix=(
                "Remove sensitive data from node outputs. Use Dify's "
                "environment variable feature for secrets instead of inline values. "
                "Add output sanitization nodes before returning results to users."
            ),
            detector_name=self.name,
        )

    def _extract_strings(self, obj: Any, depth: int = 0) -> List[str]:
        """Recursively extract all string values from nested dicts/lists."""
        if depth > 10:
            return []
        strings = []
        if isinstance(obj, str):
            strings.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                strings.extend(self._extract_strings(v, depth + 1))
        elif isinstance(obj, list):
            for item in obj:
                strings.extend(self._extract_strings(item, depth + 1))
        return strings

    def _redact(self, value: str) -> str:
        """Redact sensitive value, keeping first 4 and last 2 chars."""
        if len(value) <= 8:
            return value[:2] + "***"
        return value[:4] + "***" + value[-2:]

    # Contact-PII labels that describe ONE person's routing address rather than
    # a secret. Bulk exposure of these matters; a single instance does not.
    _CONTACT_PII_LABELS = frozenset({"Email address", "Phone number"})
    _BULK_CONTACT_PII_MIN_DISTINCT = 2

    @classmethod
    def _gate_contact_pii(cls, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop contact-PII issues unless they show bulk or accompany a secret."""
        contact = [
            i for i in issues
            if i.get("category") == "pii" and i.get("label") in cls._CONTACT_PII_LABELS
        ]
        if not contact:
            return issues

        other_secret = any(
            i.get("category") in {"api_key", "password", "token", "secret"}
            for i in issues
        )
        distinct = {
            str(i.get("_match_fingerprint") or i.get("matched_preview", ""))
            .strip()
            .lower()
            for i in contact
        }
        if other_secret or len(distinct) >= cls._BULK_CONTACT_PII_MIN_DISTINCT:
            return issues

        keep = [i for i in issues if i not in contact]
        return keep

    def _check_scope_leaks(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for iteration child outputs leaking into non-iteration nodes.

        If a node inside an iteration (has parent_node_id) produces output
        values that appear verbatim in the inputs of a node outside the
        iteration, that is a scope leak.
        """
        leaks: List[Dict[str, Any]] = []

        # Collect iteration children and their output strings
        child_outputs: Dict[str, Set[str]] = {}  # parent_id -> set of output strings
        for node in nodes:
            parent_id = node.get("parent_node_id")
            if parent_id:
                outputs = node.get("outputs", {})
                strings = self._extract_strings(outputs)
                # Only track non-trivial strings
                significant = {s for s in strings if len(s) > 20}
                if parent_id not in child_outputs:
                    child_outputs[parent_id] = set()
                child_outputs[parent_id].update(significant)

        if not child_outputs:
            return leaks

        # Check non-iteration nodes for leaked values
        for node in nodes:
            if node.get("parent_node_id"):
                continue  # Skip iteration children
            node_id = node.get("node_id", "")
            # Skip the iteration parent nodes themselves
            if node.get("node_type") in ("iteration", "loop"):
                continue

            input_strings = self._extract_strings(node.get("inputs", {}))
            for text in input_strings:
                for parent_id, outputs in child_outputs.items():
                    for out_val in outputs:
                        if out_val in text:
                            leaks.append(
                                {
                                    "type": "scope_leak",
                                    "source_iteration_id": parent_id,
                                    "target_node_id": node_id,
                                    "target_title": node.get("title", ""),
                                    "leaked_value_preview": out_val[:100],
                                }
                            )
                            break
        return leaks

    def _no_detection(self, reason: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=reason,
            detector_name=self.name,
        )
