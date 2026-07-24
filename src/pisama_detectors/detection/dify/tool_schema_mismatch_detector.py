"""
Tool Schema Mismatch Detection for Dify Workflows
==================================================

Detects schema validation issues in Dify tool nodes: missing required
inputs, type mismatches, extra unexpected fields, null required fields,
and explicit schema-related errors in failure messages.

Dify-specific: targets tool node type and its inputs.schema /
status / error structure.
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

# Keywords indicating schema-related errors in failure messages
SCHEMA_ERROR_KEYWORDS = [
    "schema",
    "validation",
    "type",
    "required",
    "missing",
    "invalid",
    "expected",
    "unexpected",
    "mismatch",
    "format",
]

# Python/JSON type names for matching
TYPE_NAMES = {
    "str": "string",
    "string": "string",
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
    "list": "array",
    "array": "array",
    "dict": "object",
    "object": "object",
}


class DifyToolSchemaMismatchDetector(TurnAwareDetector):
    """Detects tool node schema validation mismatches in Dify workflows.

    Checks for missing required inputs, type mismatches against declared
    schemas, null values for required fields, extra unexpected inputs,
    and schema-related error messages in failed tool nodes.
    """

    name = "DifyToolSchemaMismatchDetector"
    version = "1.0"
    supported_failure_modes = ["F12"]  # Output validation failure

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
        """Analyze Dify workflow run for tool schema mismatches.

        Args:
            workflow_run: Dify workflow_run dict with nodes list.

        Returns:
            Detection result with schema mismatch findings.
        """
        nodes = workflow_run.get("nodes", [])
        if not nodes:
            return self._no_detection("No nodes in workflow run")

        _tool_like_types = {"tool", "http_request", "api_request", "webhook"}
        tool_nodes = [n for n in nodes if n.get("node_type") in _tool_like_types]
        if not tool_nodes:
            return self._no_detection("No tool nodes found")

        issues: List[Dict[str, Any]] = []
        affected_node_ids: List[str] = []
        max_confidence = 0.0

        for node in tool_nodes:
            node_id = node.get("node_id", "")
            node_title = node.get("title", "")
            inputs = node.get("inputs", {})
            outputs = node.get("outputs", {})
            status = node.get("status", "")

            # Check 1: Schema-based validation (if schema is defined)
            schema = inputs.get("schema", {})
            if schema:
                schema_issues = self._validate_against_schema(
                    node_id, node_title, inputs, outputs, schema
                )
                for si in schema_issues:
                    issues.append(si)
                    affected_node_ids.append(node_id)
                    max_confidence = max(max_confidence, si.get("confidence", 0.7))

            # Check 2: Null values for required fields
            null_issues = self._check_null_required(node_id, node_title, inputs, schema)
            for ni in null_issues:
                issues.append(ni)
                affected_node_ids.append(node_id)
                max_confidence = max(max_confidence, 0.7)

            # Check 3: Error message with schema-related keywords
            error_msg = node.get("error", "") or node.get("error_message", "")
            if status == "failed" or error_msg:
                error_issue = self._check_error_message(node_id, node_title, node)
                if error_issue:
                    issues.append(error_issue)
                    affected_node_ids.append(node_id)
                    max_confidence = max(max_confidence, 0.85)

        if not issues:
            return self._no_detection("No tool schema mismatches detected")

        confidence = max_confidence if max_confidence > 0 else 0.6

        # Severity
        has_schema_error = any(i["type"] == "schema_error_in_failure" for i in issues)
        has_missing = any(i["type"] == "missing_required_input" for i in issues)
        if has_schema_error and has_missing:
            severity = TurnAwareSeverity.SEVERE
        elif has_schema_error or len(issues) >= 3:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F12",
            explanation=(
                f"Tool schema mismatch: {len(issues)} issue(s) across "
                f"{len(tool_nodes)} tool node(s)"
            ),
            affected_turns=list(range(len(set(affected_node_ids)))),
            evidence={
                "issues": issues,
                "total_tool_nodes": len(tool_nodes),
            },
            suggested_fix=(
                "Verify tool input schemas match upstream node outputs. "
                "Add type conversion nodes (Code or Template Transform) before "
                "tool nodes. Ensure all required fields are populated and non-null."
            ),
            detector_name=self.name,
        )

    def _validate_against_schema(
        self,
        node_id: str,
        node_title: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Validate inputs/outputs against the declared schema."""
        issues = []

        # Schema can be {"properties": {...}, "required": [...]}
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])

        # Check required inputs are present
        for field in required_fields:
            if field not in inputs or field == "schema":
                issues.append(
                    {
                        "type": "missing_required_input",
                        "node_id": node_id,
                        "title": node_title,
                        "field": field,
                        "confidence": 0.7,
                    }
                )

        # Check type mismatches for provided inputs
        for field, field_schema in properties.items():
            if field in inputs and field != "schema":
                expected_type = field_schema.get("type", "")
                actual_value = inputs[field]
                if not self._type_matches(actual_value, expected_type):
                    issues.append(
                        {
                            "type": "type_mismatch",
                            "node_id": node_id,
                            "title": node_title,
                            "field": field,
                            "expected_type": expected_type,
                            "actual_type": type(actual_value).__name__,
                            "confidence": 0.75,
                        }
                    )

        # Check for extra unexpected inputs
        schema_fields = set(properties.keys())
        input_fields = {k for k in inputs.keys() if k != "schema"}
        extra = input_fields - schema_fields
        if extra and schema_fields:  # Only flag if schema defines fields
            issues.append(
                {
                    "type": "extra_inputs",
                    "node_id": node_id,
                    "title": node_title,
                    "extra_fields": list(extra),
                    "confidence": 0.5,
                }
            )

        return issues

    def _check_null_required(
        self,
        node_id: str,
        node_title: str,
        inputs: Dict[str, Any],
        schema: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check for null values in required fields."""
        issues = []
        required_fields = schema.get("required", []) if schema else []

        for field in required_fields:
            val = inputs.get(field)
            if val is None:
                issues.append(
                    {
                        "type": "null_required_field",
                        "node_id": node_id,
                        "title": node_title,
                        "field": field,
                    }
                )
        return issues

    def _check_error_message(
        self,
        node_id: str,
        node_title: str,
        node: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Check failed node for schema-related error messages."""
        # Look for error in outputs or top-level error field
        error_text = ""
        outputs = node.get("outputs", {})
        if isinstance(outputs, dict):
            error_text = str(outputs.get("error", "")) + str(outputs.get("message", ""))
        error_text += str(node.get("error", ""))
        error_lower = error_text.lower()

        matched_keywords = [kw for kw in SCHEMA_ERROR_KEYWORDS if kw in error_lower]
        if matched_keywords:
            return {
                "type": "schema_error_in_failure",
                "node_id": node_id,
                "title": node_title,
                "error_preview": error_text[:300],
                "matched_keywords": matched_keywords,
                "confidence": 0.85,
            }
        return None

    def _type_matches(self, value: Any, expected_type: str) -> bool:
        """Check if a value matches the expected JSON schema type."""
        if value is None:
            return False  # Null handled separately
        normalized = TYPE_NAMES.get(expected_type.lower(), expected_type.lower())
        actual_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        actual_type = actual_map.get(type(value), "unknown")
        # Allow int where number is expected
        if normalized == "number" and actual_type == "integer":
            return True
        return actual_type == normalized

    def _no_detection(self, reason: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=reason,
            detector_name=self.name,
        )
