"""
F12: Schema Mismatch Detection for n8n Workflows
=================================================

Detects schema mismatches between consecutive n8n nodes where:
- Node A outputs JSON with certain structure
- Node B expects JSON with different structure
- Leading to data loss, errors, or undefined behavior

This is n8n-specific because schema flow is explicit in workflow automation,
unlike conversational agents where data flow is implicit.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from pisama_detectors.detection.turn_aware._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)

logger = logging.getLogger(__name__)

# Known n8n node output types for schema inference
NODE_OUTPUT_TYPES: Dict[str, str] = {
    "n8n-nodes-base.httpRequest": "json",
    "@n8n/n8n-nodes-langchain.agent": "text",
    "@n8n/n8n-nodes-langchain.chainLlm": "text",
    "n8n-nodes-base.code": "any",
    "n8n-nodes-base.set": "items",
    "n8n-nodes-base.if": "items",
    "n8n-nodes-base.switch": "items",
    "n8n-nodes-base.merge": "items",
    "n8n-nodes-base.spreadsheetFile": "items",
    "n8n-nodes-base.postgres": "items",
    "n8n-nodes-base.mysql": "items",
}

# Known input type expectations for n8n node types
NODE_INPUT_TYPES: Dict[str, str] = {
    "n8n-nodes-base.httpRequest": "any",
    "@n8n/n8n-nodes-langchain.agent": "text",
    "@n8n/n8n-nodes-langchain.chainLlm": "text",
    "n8n-nodes-base.code": "any",
    "n8n-nodes-base.set": "items",
    "n8n-nodes-base.if": "items",
    "n8n-nodes-base.switch": "items",
    "n8n-nodes-base.merge": "items",
    "n8n-nodes-base.spreadsheetFile": "items",
    "n8n-nodes-base.postgres": "items",
    "n8n-nodes-base.mysql": "items",
}

# Type compatibility matrix: (source_output, dest_input) -> compatible
TYPE_COMPATIBILITY: Dict[tuple, bool] = {
    ("json", "text"): False,
    ("json", "items"): False,
    ("text", "items"): False,
    ("text", "json"): False,
    ("items", "text"): False,
    ("items", "json"): False,
    ("json", "json"): True,
    ("text", "text"): True,
    ("items", "items"): True,
    ("any", "any"): True,
}


# Benign type-pair whitelist: these mismatches are commonly valid in n8n
# because n8n automatically coerces between these types.
BENIGN_TYPE_PAIRS: Set[tuple] = {
    ("json", "text"),
    ("text", "json"),
    ("items", "any"),
    # ("any", *) is handled separately — any source is always compatible
}


class N8NSchemaDetector(TurnAwareDetector):
    """Detects F12: Output Validation Failure / Schema Mismatch in n8n workflows.

    Analyzes workflow execution for:
    1. Missing required fields in node outputs
    2. Type mismatches between producer and consumer nodes
    3. Schema drift over workflow execution
    4. Undefined field access in downstream nodes

    n8n-specific manifestation of F12 (Output Validation Failure):
    In conversational agents, this is about validating LLM outputs.
    In n8n workflows, this is about JSON schema compatibility between nodes.
    """

    name = "N8NSchemaDetector"
    version = "1.1"
    supported_failure_modes = ["F12"]

    def __init__(
        self,
        strict_mode: bool = False,
        required_field_threshold: float = 0.3,
    ):
        """Initialize schema detector.

        Args:
            strict_mode: If True, flag any schema differences (not just breaking ones)
            required_field_threshold: Fraction of fields that must match to avoid flagging (0.3 = 30%)
        """
        self.strict_mode = strict_mode
        self.required_field_threshold = required_field_threshold

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect schema mismatches between consecutive n8n nodes."""
        if len(turns) < 2:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Need at least 2 nodes to check schema compatibility",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        for i in range(len(turns) - 1):
            producer = turns[i]
            consumer = turns[i + 1]

            # Extract schemas from content
            producer_schema = self._extract_schema(producer.content)
            consumer_expected = self._infer_expected_schema(consumer)

            if producer_schema and consumer_expected:
                mismatch = self._check_schema_mismatch(
                    producer_schema,
                    consumer_expected,
                    producer.participant_id,
                    consumer.participant_id,
                )
                if mismatch:
                    issues.append(mismatch)
                    affected_turns.extend([producer.turn_number, consumer.turn_number])

            # Check for error indicators in content
            error_indicators = self._detect_schema_errors(consumer.content)
            if error_indicators:
                issues.append({
                    "type": "schema_error",
                    "producer": producer.participant_id,
                    "consumer": consumer.participant_id,
                    "errors": error_indicators,
                    "description": f"Schema-related errors detected in {consumer.participant_id}",
                })
                affected_turns.extend([producer.turn_number, consumer.turn_number])

        # Check for progressive schema drift (only if we have error indicators)
        # Drift detection alone causes too many false positives
        has_error_indicators = any(i.get("type") == "schema_error" for i in issues)
        if has_error_indicators:
            drift = self._detect_schema_drift(turns)
            if drift["detected"]:
                issues.append(drift)
                affected_turns.extend(drift.get("turns", []))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="No schema mismatches detected between nodes",
                detector_name=self.name,
            )

        # Determine severity
        error_count = sum(1 for i in issues if i.get("type") == "schema_error")
        mismatch_count = sum(1 for i in issues if i.get("type") in ("field_mismatch", "type_mismatch"))

        if error_count >= 2 or mismatch_count >= 3:
            severity = TurnAwareSeverity.SEVERE
        elif error_count >= 1 or mismatch_count >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.95, 0.6 + len(issues) * 0.1)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F12",
            explanation=f"Schema mismatch: {len(issues)} incompatibilities found between nodes",
            affected_turns=list(set(affected_turns)),
            evidence={
                "issues": issues,
                "total_transitions": len(turns) - 1,
            },
            suggested_fix=(
                "Ensure consistent JSON schemas between connected nodes. "
                "Add schema validation nodes or transform nodes to handle type mismatches. "
                "Use n8n's Expression Editor to properly map fields."
            ),
            detector_name=self.name,
        )

    def detect_workflow(self, workflow_json: Dict[str, Any]) -> TurnAwareDetectionResult:
        """Detect schema mismatches by analyzing raw n8n workflow JSON directly.

        Builds a node lookup from workflow_json["nodes"], iterates over connections
        to find connected pairs, infers output/input types using known n8n node types,
        and flags type mismatches and invalid expression references.

        Args:
            workflow_json: Raw n8n workflow JSON with "nodes" and "connections" keys.

        Returns:
            TurnAwareDetectionResult with detected schema issues.
        """
        nodes = workflow_json.get("nodes", [])
        connections = workflow_json.get("connections", {})

        if not nodes:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No nodes found in workflow JSON",
                detector_name=self.name,
            )

        # Build node lookup dict keyed by node name
        node_lookup: Dict[str, Dict[str, Any]] = {}
        for node in nodes:
            node_name = node.get("name", "")
            if node_name:
                node_lookup[node_name] = node

        issues: List[Dict[str, Any]] = []
        affected_node_names: List[str] = []

        # Iterate over connections to find connected node pairs
        # n8n connections format: { "SourceNode": { "main": [[{"node": "DestNode", "type": "main", "index": 0}]] } }
        for source_name, outputs in connections.items():
            source_node = node_lookup.get(source_name)
            if source_node is None:
                continue

            source_type = source_node.get("type", "")
            source_output_type = NODE_OUTPUT_TYPES.get(source_type, "unknown")

            # outputs is typically {"main": [[{...}], [{...}]]}
            if not isinstance(outputs, dict):
                continue

            for output_key, output_branches in outputs.items():
                if not isinstance(output_branches, list):
                    continue
                for branch in output_branches:
                    if not isinstance(branch, list):
                        continue
                    for conn in branch:
                        if not isinstance(conn, dict):
                            continue
                        dest_name = conn.get("node", "")
                        dest_node = node_lookup.get(dest_name)
                        if dest_node is None:
                            continue

                        dest_type = dest_node.get("type", "")
                        dest_input_type = NODE_INPUT_TYPES.get(dest_type, "unknown")

                        # Skip checks for unknown/community node types —
                        # we cannot reliably infer their schemas.
                        if source_output_type == "unknown" or dest_input_type == "unknown":
                            pass  # Skip — don't flag unknown/community nodes
                        elif source_output_type == "any" or dest_input_type == "any":
                            pass  # compatible
                        else:
                            compat_key = (source_output_type, dest_input_type)
                            # Check benign whitelist — commonly valid in n8n
                            if compat_key in BENIGN_TYPE_PAIRS:
                                # Benign mismatch — skip or set very low confidence
                                issues.append({
                                    "type": "benign_type_mismatch",
                                    "severity": "info",
                                    "confidence": 0.2,
                                    "source_node": source_name,
                                    "dest_node": dest_name,
                                    "description": (
                                        f"Benign type pair: {source_name} outputs "
                                        f"'{source_output_type}' -> {dest_name} "
                                        f"expects '{dest_input_type}' (auto-coerced)"
                                    ),
                                })
                            else:
                                is_compatible = TYPE_COMPATIBILITY.get(compat_key, True)
                                if not is_compatible:
                                    issues.append({
                                        "type": "type_mismatch",
                                        "source_node": source_name,
                                        "source_node_type": source_type,
                                        "source_output_type": source_output_type,
                                        "dest_node": dest_name,
                                        "dest_node_type": dest_type,
                                        "dest_input_type": dest_input_type,
                                        "description": (
                                            f"Type mismatch: {source_name} ({source_type}) outputs "
                                            f"'{source_output_type}' but {dest_name} ({dest_type}) "
                                            f"expects '{dest_input_type}'"
                                        ),
                                    })
                                    affected_node_names.extend([source_name, dest_name])

        # Check for $json.fieldName expression references in node parameters
        # that reference fields not produced by known upstream nodes
        expression_issues = self._check_expression_references(
            node_lookup, connections
        )
        issues.extend(expression_issues)
        for issue in expression_issues:
            affected_node_names.extend([
                issue.get("node", ""),
                issue.get("upstream_node", ""),
            ])

        # Filter: only count real issues (not benign mismatches) for detection
        real_issues = [
            i for i in issues
            if i.get("type") not in ("benign_type_mismatch", "unknown_node_type")
        ]

        if not real_issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="No schema mismatches detected in workflow JSON",
                detector_name=self.name,
            )

        # Determine severity
        type_mismatch_count = sum(1 for i in real_issues if i.get("type") == "type_mismatch")
        expr_ref_count = sum(1 for i in real_issues if i.get("type") == "expression_reference")

        if type_mismatch_count >= 3 or (type_mismatch_count >= 1 and expr_ref_count >= 2):
            severity = TurnAwareSeverity.SEVERE
        elif type_mismatch_count >= 2 or expr_ref_count >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.95, 0.6 + len(real_issues) * 0.1)

        # Map affected node names to turn indices (use node list order)
        node_index_map = {node.get("name", ""): idx for idx, node in enumerate(nodes)}
        affected_turns = sorted({
            node_index_map[name]
            for name in set(affected_node_names)
            if name in node_index_map
        })

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F12",
            explanation=f"Schema mismatch: {len(real_issues)} incompatibilities found in workflow JSON",
            affected_turns=affected_turns,
            evidence={
                "issues": issues,
                "total_nodes": len(nodes),
                "total_connections": sum(
                    sum(
                        len(conn_list)
                        for branch in output_branches
                        if isinstance(branch, list)
                        for conn_list in [branch]
                    )
                    for outputs in connections.values()
                    if isinstance(outputs, dict)
                    for output_branches in outputs.values()
                    if isinstance(output_branches, list)
                ),
            },
            suggested_fix=(
                "Ensure consistent data types between connected nodes. "
                "Add Set or Code nodes to transform data between incompatible types. "
                "Verify $json expression references match fields produced by upstream nodes."
            ),
            detector_name=self.name,
        )

    def _check_expression_references(
        self,
        node_lookup: Dict[str, Dict[str, Any]],
        connections: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Check for $json.fieldName expression references that may be invalid.

        Scans node parameters for $json.fieldName expressions and checks
        whether the referenced fields are plausibly produced by upstream nodes.

        Returns list of issues for invalid references.
        """
        issues: List[Dict[str, Any]] = []

        # Build reverse lookup: dest_name -> list of source_names
        upstream_map: Dict[str, List[str]] = {}
        for source_name, outputs in connections.items():
            if not isinstance(outputs, dict):
                continue
            for output_key, output_branches in outputs.items():
                if not isinstance(output_branches, list):
                    continue
                for branch in output_branches:
                    if not isinstance(branch, list):
                        continue
                    for conn in branch:
                        if not isinstance(conn, dict):
                            continue
                        dest_name = conn.get("node", "")
                        if dest_name:
                            if dest_name not in upstream_map:
                                upstream_map[dest_name] = []
                            upstream_map[dest_name].append(source_name)

        # For each node, find $json references in parameters
        json_field_pattern = re.compile(r'\$json\.([a-zA-Z_][a-zA-Z0-9_]*)')

        for node_name, node_data in node_lookup.items():
            parameters = node_data.get("parameters", {})
            params_str = json.dumps(parameters)
            referenced_fields = set(json_field_pattern.findall(params_str))

            if not referenced_fields:
                continue

            # Get upstream node types to check what fields they might produce
            upstream_sources = upstream_map.get(node_name, [])
            if not upstream_sources:
                # Node has $json references but no upstream connection
                for field_name in referenced_fields:
                    issues.append({
                        "type": "expression_reference",
                        "node": node_name,
                        "upstream_node": "",
                        "field": field_name,
                        "description": (
                            f"Node '{node_name}' references $json.{field_name} "
                            f"but has no upstream connections"
                        ),
                    })
                continue

            # Check if upstream nodes are of types that produce specific known output types
            # Text-outputting nodes (LLM agents/chains) don't produce structured JSON fields
            for source_name in upstream_sources:
                source_node = node_lookup.get(source_name)
                if source_node is None:
                    continue
                source_type = source_node.get("type", "")
                source_output = NODE_OUTPUT_TYPES.get(source_type, "unknown")

                if source_output == "text":
                    # Text nodes produce plain text, not structured JSON fields
                    for field_name in referenced_fields:
                        issues.append({
                            "type": "expression_reference",
                            "node": node_name,
                            "upstream_node": source_name,
                            "field": field_name,
                            "description": (
                                f"Node '{node_name}' references $json.{field_name} "
                                f"but upstream '{source_name}' ({source_type}) "
                                f"outputs plain text, not structured JSON"
                            ),
                        })

        return issues

    def _extract_schema(self, content: str) -> Optional[Dict[str, str]]:
        """Extract JSON schema (field names and types) from node output.

        Returns dict mapping field name to type string.
        """
        content = content.strip()

        # Try to parse as JSON
        if content.startswith('{') or content.startswith('['):
            try:
                data = json.loads(content)
                return self._schema_from_data(data)
            except json.JSONDecodeError:
                pass

        # Try to extract from key: value format (from _clean_n8n_content)
        schema = {}
        for line in content.split('\n'):
            if ':' in line:
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    schema[key] = self._infer_type(value)

        return schema if schema else None

    def _schema_from_data(self, data: Any, prefix: str = "") -> Dict[str, str]:
        """Convert JSON data to schema (field -> type mapping)."""
        schema = {}

        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                schema[full_key] = self._type_name(value)

                # Recurse into nested objects (one level only)
                if isinstance(value, dict) and not prefix:
                    nested = self._schema_from_data(value, full_key)
                    schema.update(nested)

        elif isinstance(data, list) and data:
            # Get schema from first element
            schema[prefix or "[]"] = f"array<{self._type_name(data[0])}>"
            if isinstance(data[0], dict) and not prefix:
                nested = self._schema_from_data(data[0], prefix or "[]")
                schema.update(nested)

        return schema

    def _type_name(self, value: Any) -> str:
        """Get type name for a value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "unknown"

    def _infer_type(self, value_str: str) -> str:
        """Infer type from string representation."""
        value_str = value_str.strip()

        if value_str.lower() in ('null', 'none', ''):
            return "null"
        elif value_str.lower() in ('true', 'false'):
            return "boolean"
        elif value_str.startswith('{'):
            return "object"
        elif value_str.startswith('['):
            return "array"
        else:
            try:
                int(value_str)
                return "integer"
            except ValueError:
                pass
            try:
                float(value_str)
                return "number"
            except ValueError:
                pass
        return "string"

    def _infer_expected_schema(self, consumer: TurnSnapshot) -> Optional[Dict[str, str]]:
        """Infer expected input schema for a consumer node based on its behavior.

        This is heuristic-based since we don't have the actual n8n workflow definition.
        """
        node_name = consumer.participant_id.lower()
        content = consumer.content.lower()

        # Common node types and their expected inputs
        if any(x in node_name for x in ['email', 'mail', 'send']):
            return {"to": "string", "subject": "string", "body": "string"}
        elif any(x in node_name for x in ['http', 'request', 'api']):
            return {"url": "string"}
        elif any(x in node_name for x in ['slack', 'discord', 'teams']):
            return {"channel": "string", "message": "string"}
        elif any(x in node_name for x in ['database', 'db', 'postgres', 'mysql']):
            return {"query": "string"}

        # Check content for field access patterns like $json.fieldName
        accessed_fields = set()
        import re
        for match in re.finditer(r'\$json\.([a-zA-Z_][a-zA-Z0-9_]*)', content):
            accessed_fields.add(match.group(1))

        if accessed_fields:
            return {field: "any" for field in accessed_fields}

        return None

    def _check_schema_mismatch(
        self,
        producer_schema: Dict[str, str],
        expected_schema: Dict[str, str],
        producer_name: str,
        consumer_name: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if producer output matches consumer expectations."""
        missing_fields = []
        type_mismatches = []

        for field, expected_type in expected_schema.items():
            if field not in producer_schema:
                missing_fields.append(field)
            elif expected_type != "any" and producer_schema[field] != expected_type:
                type_mismatches.append({
                    "field": field,
                    "expected": expected_type,
                    "actual": producer_schema[field],
                })

        # Calculate compatibility score
        total_expected = len(expected_schema)
        matched = total_expected - len(missing_fields) - len(type_mismatches)
        compatibility = matched / total_expected if total_expected > 0 else 1.0

        if missing_fields or type_mismatches:
            if compatibility < self.required_field_threshold or self.strict_mode:
                return {
                    "type": "field_mismatch" if missing_fields else "type_mismatch",
                    "producer": producer_name,
                    "consumer": consumer_name,
                    "missing_fields": missing_fields,
                    "type_mismatches": type_mismatches,
                    "compatibility": compatibility,
                    "description": (
                        f"Schema incompatibility: {producer_name} -> {consumer_name}. "
                        f"Missing: {missing_fields}, Type mismatches: {len(type_mismatches)}"
                    ),
                }

        return None

    def _detect_schema_errors(self, content: str) -> List[str]:
        """Detect error patterns indicating schema issues."""
        errors = []
        content_lower = content.lower()

        error_patterns = [
            ("undefined", "Accessing undefined field"),
            ("cannot read property", "Property access on null/undefined"),
            ("is not defined", "Reference error on undefined field"),
            ("expected string", "Type mismatch - expected string"),
            ("expected number", "Type mismatch - expected number"),
            ("expected array", "Type mismatch - expected array"),
            ("expected object", "Type mismatch - expected object"),
            ("invalid json", "JSON parsing error"),
            ("null reference", "Null pointer/reference"),
            ("field not found", "Missing field in input"),
            ("required field", "Required field missing"),
        ]

        for pattern, description in error_patterns:
            if pattern in content_lower:
                errors.append(description)

        return errors

    def _detect_schema_drift(self, turns: List[TurnSnapshot]) -> Dict[str, Any]:
        """Detect progressive schema drift over workflow execution.

        Schema drift occurs when the JSON structure changes incrementally
        through the workflow, losing or transforming fields.
        """
        if len(turns) < 3:
            return {"detected": False}

        # Track field presence across turns
        all_fields: Set[str] = set()
        field_presence: List[Set[str]] = []

        for turn in turns:
            schema = self._extract_schema(turn.content)
            if schema:
                fields = set(schema.keys())
                all_fields.update(fields)
                field_presence.append(fields)

        if len(field_presence) < 3:
            return {"detected": False}

        # Check for monotonically decreasing field count (data loss)
        lost_fields = []
        for i in range(1, len(field_presence)):
            lost = field_presence[i - 1] - field_presence[i]
            if lost:
                lost_fields.extend(lost)

        # Significant drift if we lost >60% of original fields
        # (stricter threshold to reduce false positives)
        if len(lost_fields) >= len(all_fields) * 0.6:
            return {
                "detected": True,
                "type": "schema_drift",
                "lost_fields": list(set(lost_fields)),
                "original_field_count": len(all_fields),
                "final_field_count": len(field_presence[-1]) if field_presence else 0,
                "turns": [t.turn_number for t in turns],
                "description": f"Schema drift: lost {len(set(lost_fields))} fields over workflow execution",
            }

        return {"detected": False}
