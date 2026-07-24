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
            "pattern": re.compile(r"password\s*=\s*\S+", re.IGNORECASE),
            "label": "Password assignment",
        },
        {"pattern": re.compile(r"passwd\s*=\s*\S+", re.IGNORECASE), "label": "Passwd assignment"},
        {"pattern": re.compile(r"secret\s*=\s*\S+", re.IGNORECASE), "label": "Secret assignment"},
        {"pattern": re.compile(r"credentials", re.IGNORECASE), "label": "Credentials reference"},
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

# Confidence per category
CATEGORY_CONFIDENCE: Dict[str, float] = {
    "api_key": 0.8,
    "password": 0.7,
    "pii": 0.7,
    "env_var": 0.6,
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
                        match = pat_info["pattern"].search(text)
                        if match:
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
                                    "confidence": cat_conf,
                                }
                            )

        # Check for iteration scope leaks
        scope_leaks = self._check_scope_leaks(nodes)
        for leak in scope_leaks:
            issues.append(leak)
            affected_node_ids.append(leak.get("target_node_id", ""))

        if not issues:
            return self._no_detection("No sensitive data or scope leaks detected")

        confidence = max_confidence if max_confidence > 0 else 0.6

        # Severity based on category
        categories_found = {i.get("category") for i in issues if "category" in i}
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
            affected_turns=list(range(len(set(affected_node_ids)))),
            evidence={
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

    def _check_scope_leaks(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check for iteration child outputs leaking into non-iteration nodes.

        If a node inside an iteration (has parent_node_id) produces output
        values that appear verbatim in the inputs of a node outside the
        iteration, that is a scope leak.
        """
        leaks = []

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
