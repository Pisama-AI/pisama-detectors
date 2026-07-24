"""
Classifier Drift Detection for Dify Workflows
==============================================

Detects question_classifier node drift and miscategorization in
Dify workflows. Checks for low confidence scores, fallback categories,
inter-classifier disagreement, and category mismatches against
configured category lists.

Dify-specific: targets question_classifier node type and its
outputs.category / inputs.categories structure.
"""

import logging
from typing import Any, Dict, List, Optional

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

# Labels treated as fallback / unclassified
FALLBACK_LABELS = {"other", "unknown", "default", "fallback", "none", "unclassified", "misc"}

# Minimum confidence threshold
LOW_CONFIDENCE_THRESHOLD = 0.5


class DifyClassifierDriftDetector(TurnAwareDetector):
    """Detects question classifier drift in Dify workflows.

    Checks for low classification confidence, fallback category selection,
    inter-classifier disagreement, and output categories that do not match
    any configured input category.
    """

    name = "DifyClassifierDriftDetector"
    version = "1.0"
    supported_failure_modes = ["F9"]  # Task derailment / misrouting

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
        """Analyze Dify workflow run for classifier drift.

        Args:
            workflow_run: Dify workflow_run dict with nodes list.

        Returns:
            Detection result with classifier drift findings.
        """
        nodes = workflow_run.get("nodes", [])
        if not nodes:
            return self._no_detection("No nodes in workflow run")

        classifier_nodes = [n for n in nodes if n.get("node_type") == "question_classifier"]
        if not classifier_nodes:
            return self._no_detection("No question_classifier nodes found")

        issues: List[Dict[str, Any]] = []
        affected_node_ids: List[str] = []
        drift_signals = 0

        for node in classifier_nodes:
            node_id = node.get("node_id", "")
            node_title = node.get("title", "")
            inputs = node.get("inputs", {})
            outputs = node.get("outputs", {})

            # Signal 1: Low confidence
            confidence_val = self._get_confidence(outputs)
            if confidence_val is not None and confidence_val < LOW_CONFIDENCE_THRESHOLD:
                drift_signals += 1
                affected_node_ids.append(node_id)
                issues.append(
                    {
                        "type": "low_confidence",
                        "node_id": node_id,
                        "title": node_title,
                        "confidence": confidence_val,
                        "threshold": LOW_CONFIDENCE_THRESHOLD,
                    }
                )

            # Signal 2: Fallback category
            category = self._get_category(outputs)
            if category and category.lower().strip() in FALLBACK_LABELS:
                drift_signals += 1
                affected_node_ids.append(node_id)
                issues.append(
                    {
                        "type": "fallback_category",
                        "node_id": node_id,
                        "title": node_title,
                        "category": category,
                    }
                )

            # Signal 3: Category not in configured list
            configured_categories = self._get_configured_categories(inputs)
            if category and configured_categories:
                normalized_category = category.lower().strip()
                normalized_configured = {c.lower().strip() for c in configured_categories}
                if normalized_category not in normalized_configured:
                    drift_signals += 1
                    affected_node_ids.append(node_id)
                    issues.append(
                        {
                            "type": "category_mismatch",
                            "node_id": node_id,
                            "title": node_title,
                            "output_category": category,
                            "configured_categories": configured_categories,
                        }
                    )

        # Signal 4: Inter-classifier disagreement
        if len(classifier_nodes) >= 2:
            disagreements = self._check_disagreements(classifier_nodes)
            for d in disagreements:
                drift_signals += 1
                issues.append(d)
                affected_node_ids.extend(d.get("node_ids", []))

        if not issues:
            return self._no_detection("No classifier drift signals found")

        # Confidence based on number of drift signals
        confidence = min(0.95, 0.5 + drift_signals * 0.15)

        # Severity
        if drift_signals >= 3:
            severity = TurnAwareSeverity.SEVERE
        elif drift_signals >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F9",
            explanation=(
                f"Classifier drift: {drift_signals} drift signal(s) across "
                f"{len(classifier_nodes)} classifier node(s)"
            ),
            affected_turns=list(range(len(set(affected_node_ids)))),
            evidence={
                "issues": issues,
                "total_classifier_nodes": len(classifier_nodes),
                "drift_signals": drift_signals,
            },
            suggested_fix=(
                "Review and retrain question classifier categories. "
                "Add a catch-all handler for low-confidence classifications. "
                "Ensure classifier categories cover the full input distribution."
            ),
            detector_name=self.name,
        )

    def _get_confidence(self, outputs: Dict[str, Any]) -> Optional[float]:
        """Extract confidence/score from classifier outputs."""
        for key in ("confidence", "score", "probability"):
            val = outputs.get(key)
            if val is not None:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    pass
        return None

    def _get_category(self, outputs: Dict[str, Any]) -> Optional[str]:
        """Extract category/class from classifier outputs."""
        for key in ("category", "class", "class_name", "label", "result"):
            val = outputs.get(key)
            if isinstance(val, str) and val:
                return val
        return None

    def _get_configured_categories(self, inputs: Dict[str, Any]) -> List[str]:
        """Extract configured categories from classifier inputs."""
        for key in ("categories", "classes", "labels", "options"):
            val = inputs.get(key)
            if isinstance(val, list):
                return [str(c) for c in val if c]
        return []

    def _check_disagreements(self, classifier_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check if multiple classifiers disagree on the same input."""
        disagreements = []

        # Group classifiers that share the same input text
        input_groups: Dict[str, List[Dict[str, Any]]] = {}
        for node in classifier_nodes:
            inputs = node.get("inputs", {})
            input_text = (
                inputs.get("query", "") or inputs.get("text", "") or inputs.get("input", "")
            )
            if isinstance(input_text, str) and input_text:
                key = input_text[:200]  # Truncate for grouping
                if key not in input_groups:
                    input_groups[key] = []
                input_groups[key].append(node)

        for input_key, group in input_groups.items():
            if len(group) < 2:
                continue

            categories = []
            for node in group:
                cat = self._get_category(node.get("outputs", {}))
                if cat:
                    categories.append((node.get("node_id", ""), cat))

            if len(categories) >= 2:
                unique_cats = {c for _, c in categories}
                if len(unique_cats) > 1:
                    disagreements.append(
                        {
                            "type": "classifier_disagreement",
                            "input_preview": input_key[:100],
                            "classifications": [
                                {"node_id": nid, "category": cat} for nid, cat in categories
                            ],
                            "node_ids": [nid for nid, _ in categories],
                        }
                    )

        return disagreements

    def _no_detection(self, reason: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=reason,
            detector_name=self.name,
        )
