"""
F14: Error Handling Detection for n8n
=====================================

Detects hidden errors in n8n workflows where nodes fail but workflow continues:
- Node has error but continueOnFail=true hides it
- Downstream nodes receive null/invalid data from failed nodes
- Workflow marked "success" but contains node failures
- Error rate exceeds acceptable threshold

This is n8n-specific because it analyzes workflow error propagation patterns
unique to the n8n node execution model.
"""

import logging
from typing import Any, Dict, List, Optional, Set

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetectionResult,
    TurnAwareDetector,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

# Indicators of null/invalid data — require structural context to avoid FPs
# These are checked with word boundaries or structural patterns
NULL_DATA_INDICATORS: Set[str] = {
    '"null"',  # JSON null value
    ": null",  # YAML/JSON null
    "undefined",
    '"none"',  # JSON "none" string
    ": none",
    "cannot read property",
    "typeerror:",
    "referenceerror:",
}

# Maximum acceptable error rate (% of nodes that can fail)
DEFAULT_MAX_ERROR_RATE = 0.15  # 15% of nodes (raised from 10% — transactional workflows have ~5-8%)


class N8NErrorDetector(TurnAwareDetector):
    """Detects F14: Error Handling failures in n8n workflows.

    Analyzes workflow execution for:
    1. Nodes with errors that continued execution (continueOnFail=true)
    2. Data flow issues from failed nodes to downstream nodes
    3. Workflow success despite node failures
    4. High error rates indicating systemic issues

    n8n-specific manifestation of F14 (Error Handling):
    In conversational agents, this is about graceful error handling.
    In n8n workflows, this is about hidden failures and data propagation.
    """

    name = "N8NErrorDetector"
    version = "1.0"
    supported_failure_modes = ["F14"]

    def __init__(
        self,
        max_error_rate: float = DEFAULT_MAX_ERROR_RATE,
        check_downstream_data: bool = True,
    ):
        """Initialize error detector.

        Args:
            max_error_rate: Maximum acceptable error rate (0.0-1.0)
            check_downstream_data: Whether to check downstream nodes for invalid data
        """
        self.max_error_rate = max_error_rate
        self.check_downstream_data = check_downstream_data

    def _has_error(self, turn: TurnSnapshot) -> bool:
        """Check if a turn represents a node with an error.

        Requires structural evidence of error, not just substring matches.
        This prevents FPs from content like 'the failed login attempt' or
        'error message handling code'.
        """
        # Check turn metadata for error flag (most reliable signal)
        if turn.turn_metadata.get("has_error"):
            return True

        # Check for node execution status in metadata
        status = turn.turn_metadata.get("status", "").lower()
        if status in ("error", "failed", "crashed"):
            return True

        # Only check content for unambiguous error patterns
        content_lower = turn.content.lower().strip()

        # Node output that starts with error indicator (not just contains it)
        if content_lower.startswith(("error:", "exception:", "fatal:")):
            return True

        # JSON error objects
        if '"error"' in content_lower and (
            '"message"' in content_lower or '"stack"' in content_lower
        ):
            return True

        return False

    def _is_workflow_successful(self, metadata: Optional[Dict[str, Any]]) -> bool:
        """Check if workflow was marked as successful."""
        if not metadata:
            return True  # Assume success if no metadata

        status = metadata.get("workflow_status", "success").lower()
        return status in ["success", "completed", "finished"]

    def _continued_on_fail(self, turn: TurnSnapshot) -> bool:
        """Check if node had continueOnFail enabled."""
        return turn.turn_metadata.get("continue_on_fail", False)

    def _has_invalid_data(self, turn: TurnSnapshot) -> bool:
        """Check if turn contains indicators of null/invalid data."""
        content_lower = turn.content.lower()
        return any(indicator in content_lower for indicator in NULL_DATA_INDICATORS)

    def _detect_hidden_failures(
        self, turns: List[TurnSnapshot], metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect nodes with errors that continued execution."""
        hidden_failures = []

        for turn in turns:
            if self._has_error(turn) and self._continued_on_fail(turn):
                hidden_failures.append(
                    {
                        "turn": turn.turn_number,
                        "node": turn.participant_id,
                        "node_type": turn.turn_metadata.get("node_type", "unknown"),
                    }
                )

        if hidden_failures:
            # Check if workflow was still marked successful
            workflow_successful = self._is_workflow_successful(metadata)

            return {
                "detected": True,
                "type": "hidden_failure",
                "failures": hidden_failures,
                "workflow_status": ("success" if workflow_successful else "failed"),
                "explanation": f"{len(hidden_failures)} node(s) failed but continued execution (continueOnFail=true)",
                "turns": [f["turn"] for f in hidden_failures],
            }

        return None

    def _detect_invalid_data_propagation(
        self, turns: List[TurnSnapshot]
    ) -> Optional[Dict[str, Any]]:
        """Detect downstream nodes receiving invalid data from failed nodes."""
        if not self.check_downstream_data:
            return None

        invalid_data_issues = []

        # Build a map of node execution order
        for i, turn in enumerate(turns):
            if self._has_error(turn):
                # Check subsequent turns for invalid data
                for j in range(i + 1, min(i + 5, len(turns))):  # Check next 4 nodes
                    downstream_turn = turns[j]
                    if self._has_invalid_data(downstream_turn):
                        invalid_data_issues.append(
                            {
                                "failed_node_turn": i,
                                "failed_node": turn.participant_id,
                                "affected_node_turn": j,
                                "affected_node": downstream_turn.participant_id,
                            }
                        )

        if invalid_data_issues:
            return {
                "detected": True,
                "type": "invalid_data_propagation",
                "issues": invalid_data_issues,
                "explanation": f"{len(invalid_data_issues)} downstream node(s) received invalid data from failed nodes",
                "turns": [issue["affected_node_turn"] for issue in invalid_data_issues],
            }

        return None

    def _detect_high_error_rate(self, turns: List[TurnSnapshot]) -> Optional[Dict[str, Any]]:
        """Detect if error rate exceeds acceptable threshold."""
        if not turns:
            return None

        error_count = sum(1 for turn in turns if self._has_error(turn))
        error_rate = error_count / len(turns)

        if error_rate > self.max_error_rate:
            return {
                "detected": True,
                "type": "high_error_rate",
                "error_count": error_count,
                "total_nodes": len(turns),
                "error_rate": error_rate,
                "threshold": self.max_error_rate,
                "explanation": f"Error rate {error_rate:.1%} exceeds threshold {self.max_error_rate:.1%} ({error_count}/{len(turns)} nodes failed)",
                "turns": [i for i, turn in enumerate(turns) if self._has_error(turn)],
            }

        return None

    def _detect_success_despite_failures(
        self, turns: List[TurnSnapshot], metadata: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Detect workflow marked successful despite node failures."""
        if not self._is_workflow_successful(metadata):
            return None  # Workflow already marked as failed

        failed_nodes = [(i, turn) for i, turn in enumerate(turns) if self._has_error(turn)]

        if failed_nodes:
            return {
                "detected": True,
                "type": "success_despite_failures",
                "failed_nodes": [
                    {
                        "turn": i,
                        "node": turn.participant_id,
                        "node_type": turn.turn_metadata.get("node_type", "unknown"),
                    }
                    for i, turn in failed_nodes
                ],
                "explanation": f"Workflow marked as successful but {len(failed_nodes)} node(s) failed",
                "turns": [i for i, _ in failed_nodes],
            }

        return None

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect error handling failures in n8n workflow."""
        if len(turns) < 1:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Need at least 1 node to detect error handling issues",
                detector_name=self.name,
            )

        issues = []
        affected_turns = []

        # 1. Detect hidden failures (continueOnFail=true)
        hidden = self._detect_hidden_failures(turns, conversation_metadata)
        if hidden:
            issues.append(hidden)
            affected_turns.extend(hidden.get("turns", []))

        # 2. Detect invalid data propagation
        invalid_data = self._detect_invalid_data_propagation(turns)
        if invalid_data:
            issues.append(invalid_data)
            affected_turns.extend(invalid_data.get("turns", []))

        # 3. Detect high error rate
        high_error_rate = self._detect_high_error_rate(turns)
        if high_error_rate:
            issues.append(high_error_rate)
            affected_turns.extend(high_error_rate.get("turns", []))

        # 4. Detect success despite failures
        success_despite_failures = self._detect_success_despite_failures(
            turns, conversation_metadata
        )
        if success_despite_failures:
            issues.append(success_despite_failures)
            affected_turns.extend(success_despite_failures.get("turns", []))

        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="No error handling issues detected",
                detector_name=self.name,
            )

        # Determine severity
        severity = TurnAwareSeverity.MODERATE
        if hidden or success_despite_failures:
            severity = TurnAwareSeverity.SEVERE
        elif high_error_rate:
            severity = TurnAwareSeverity.MODERATE

        # Calculate confidence
        confidence = 0.90 if hidden or success_despite_failures else 0.80

        # Build explanation
        explanations = [issue["explanation"] for issue in issues]
        full_explanation = "; ".join(explanations)

        # Suggest fixes
        fixes = []
        if hidden:
            fixes.append("Set continueOnFail=false to stop workflow on errors")
        if invalid_data:
            fixes.append("Add data validation nodes after potentially failing operations")
        if high_error_rate:
            fixes.append("Review workflow logic - high error rate indicates systemic issues")
        if success_despite_failures:
            fixes.append("Add error handler nodes to properly handle failures")

        suggested_fix = "; ".join(fixes) if fixes else None

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F14",
            explanation=full_explanation,
            affected_turns=sorted(set(affected_turns)),
            evidence={"issues": issues},
            suggested_fix=suggested_fix,
            detector_name=self.name,
            detector_version=self.version,
        )

    # ---- Workflow JSON analysis (static / pre-execution) ----

    # Node types considered AI / high-risk
    _AI_NODE_KEYWORDS = {"langchain", "openai", "anthropic", "llm", "agent", "chain", "chat"}

    # Node types that are error-handling infrastructure (don't need their own
    # error handling and should not be counted as "unprotected").
    _ERROR_INFRA_TYPES = {
        "n8n-nodes-base.errorTrigger",
        "n8n-nodes-base.errorHandler",
        "n8n-nodes-base.noOp",
    }

    # Utility / terminal nodes that are typically part of error-handling
    # chains (notification after error trigger, etc.) and don't need their
    # own error handling.
    _TERMINAL_UTILITY_TYPES = {
        "n8n-nodes-base.set",
        "n8n-nodes-base.noOp",
    }

    def _is_ai_node(self, node: Dict[str, Any]) -> bool:
        """Return True if the node type indicates an AI / LLM node."""
        node_type = node.get("type", "").lower()
        return any(kw in node_type for kw in self._AI_NODE_KEYWORDS)

    def _node_has_error_handling(self, node: Dict[str, Any]) -> bool:
        """Return True if the node has any form of error handling configured.

        Checks both the ``settings.continueOnFail`` flag and the newer
        ``onError`` property.
        """
        settings = node.get("settings", {}) or {}
        if settings.get("continueOnFail") is True:
            return True
        if node.get("onError") not in (None, "", "stopWorkflow"):
            # "stopWorkflow" is the default (no explicit handling)
            return True
        return False

    def _build_connection_map(self, workflow_json: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build a mapping from source-node name to list of target-node names.

        n8n connections are stored as::

            {
                "connections": {
                    "<source_node_name>": {
                        "main": [
                            [  // output index 0
                                {"node": "<target_node_name>", "type": "main", "index": 0},
                                ...
                            ]
                        ]
                    }
                }
            }
        """
        connections: Dict[str, List[str]] = {}
        raw = workflow_json.get("connections", {})
        for source_name, outputs in raw.items():
            targets: List[str] = []
            for _output_type, output_groups in outputs.items():
                for group in output_groups:
                    if isinstance(group, list):
                        for link in group:
                            if isinstance(link, dict) and "node" in link:
                                targets.append(link["node"])
            connections[source_name] = targets
        return connections

    def detect_workflow(self, workflow_json: Dict[str, Any]) -> TurnAwareDetectionResult:
        """Analyze raw n8n workflow JSON for error-handling risks.

        Checks performed:
        1. Nodes without any error handling (no ``continueOnFail`` and no error
           trigger connected).
        2. AI nodes (langchain / OpenAI / Anthropic) without error handling --
           flagged at highest severity.
        3. Missing ``n8n-nodes-base.errorTrigger`` node in the workflow.
        4. Nodes with ``continueOnFail: true`` that feed into critical
           downstream nodes, creating data-integrity risk.
        """
        nodes: List[Dict[str, Any]] = workflow_json.get("nodes", [])
        if not nodes:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Workflow contains no nodes to analyze",
                detector_name=self.name,
            )

        connection_map = self._build_connection_map(workflow_json)
        node_by_name: Dict[str, Dict[str, Any]] = {n.get("name", ""): n for n in nodes}

        issues: List[Dict[str, Any]] = []

        # --- Build set of nodes reachable from errorTrigger (error handler chain) ---
        error_chain_nodes: Set[str] = set()
        error_trigger_names = [
            n.get("name", "") for n in nodes if n.get("type", "") in self._ERROR_INFRA_TYPES
        ]
        # BFS from error trigger nodes
        queue = list(error_trigger_names)
        while queue:
            current = queue.pop(0)
            if current in error_chain_nodes:
                continue
            error_chain_nodes.add(current)
            for downstream in connection_map.get(current, []):
                if downstream not in error_chain_nodes:
                    queue.append(downstream)

        # --- 1 & 2: Nodes without error handling ---
        unprotected_nodes: List[Dict[str, Any]] = []
        unprotected_ai_nodes: List[Dict[str, Any]] = []

        for node in nodes:
            node_type = node.get("type", "")
            node_name = node.get("name", "")
            # Skip error-handling infrastructure and terminal utility nodes —
            # they don't need their own error handling.
            if node_type in self._ERROR_INFRA_TYPES or node_type in self._TERMINAL_UTILITY_TYPES:
                continue
            # Skip nodes that are part of error handler chains
            if node_name in error_chain_nodes:
                continue
            if not self._node_has_error_handling(node):
                entry = {
                    "node_name": node.get("name", "unknown"),
                    "node_type": node.get("type", "unknown"),
                }
                unprotected_nodes.append(entry)
                if self._is_ai_node(node):
                    unprotected_ai_nodes.append(entry)

        if unprotected_ai_nodes:
            issues.append(
                {
                    "detected": True,
                    "type": "unprotected_ai_nodes",
                    "nodes": unprotected_ai_nodes,
                    "explanation": (
                        f"{len(unprotected_ai_nodes)} AI node(s) have no error handling -- "
                        "these are highest risk because LLM calls are inherently unreliable"
                    ),
                }
            )

        # Only flag unprotected_nodes for workflows with >= 8 nodes.
        # Small workflows don't need error handling on every node.
        if unprotected_nodes and len(nodes) >= 8:
            issues.append(
                {
                    "detected": True,
                    "type": "unprotected_nodes",
                    "count": len(unprotected_nodes),
                    "total_nodes": len(nodes),
                    "nodes": unprotected_nodes,
                    "explanation": (
                        f"{len(unprotected_nodes)}/{len(nodes)} node(s) lack error handling "
                        "(no continueOnFail and no onError handler)"
                    ),
                }
            )

        # --- 3: Missing error trigger node ---
        # v1.1: Only flag if workflow has NO error handling at all.
        # If most nodes use continueOnFail or onError, that's sufficient.
        has_error_trigger = any(
            node.get("type", "") == "n8n-nodes-base.errorTrigger" for node in nodes
        )
        # Count nodes with explicit error handling
        nodes_with_error_handling = sum(
            1
            for n in nodes
            if n.get("continueOnFail")
            or n.get("onError")
            or n.get("settings", {}).get("continueOnFail")
        )
        # Only flag missing trigger if < 30% of nodes have error handling
        # (a workflow with extensive node-level handling doesn't need a global trigger)
        if not has_error_trigger and nodes_with_error_handling < max(1, len(nodes) * 0.30):
            issues.append(
                {
                    "detected": True,
                    "type": "missing_error_trigger",
                    "explanation": (
                        "Workflow has no errorTrigger node -- workflow-level errors "
                        "will go unhandled and unnotified"
                    ),
                }
            )

        # --- 4: continueOnFail nodes feeding critical downstream nodes ---
        # Only flag when the source is itself an AI/critical node (high risk
        # of producing garbage output) AND there is no global error trigger
        # to catch workflow-level failures.
        continue_on_fail_risk: List[Dict[str, Any]] = []

        for node in nodes:
            settings = node.get("settings", {}) or {}
            if not settings.get("continueOnFail"):
                continue

            # When the workflow has a global error trigger, only flag if the
            # *source* node is an AI/critical node whose failed output can
            # silently corrupt downstream data.
            if has_error_trigger and not self._is_ai_node(node):
                continue

            # If the source node has an explicit onError handler (not just
            # continueOnFail), the error is already handled gracefully — skip.
            node_settings = node.get("settings", {}) or {}
            node_on_error = node_settings.get("onError") or node.get("onError", "")
            if node_on_error and node_on_error != "stopWorkflow":
                continue

            node_name = node.get("name", "")
            downstream_names = connection_map.get(node_name, [])

            for ds_name in downstream_names:
                ds_node = node_by_name.get(ds_name)
                if ds_node is None:
                    continue
                ds_type = ds_node.get("type", "").lower()
                # Consider AI nodes, database nodes, and HTTP nodes as critical
                is_critical = (
                    self._is_ai_node(ds_node)
                    or "database" in ds_type
                    or "postgres" in ds_type
                    or "mysql" in ds_type
                    or "mongo" in ds_type
                    or "httprequest" in ds_type
                    or "webhook" in ds_type
                )
                if is_critical:
                    continue_on_fail_risk.append(
                        {
                            "source_node": node_name,
                            "source_type": node.get("type", "unknown"),
                            "downstream_node": ds_name,
                            "downstream_type": ds_node.get("type", "unknown"),
                        }
                    )

        if continue_on_fail_risk:
            issues.append(
                {
                    "detected": True,
                    "type": "continue_on_fail_data_integrity_risk",
                    "pairs": continue_on_fail_risk,
                    "explanation": (
                        f"{len(continue_on_fail_risk)} node(s) with continueOnFail=true feed into "
                        "critical downstream nodes, risking null/invalid data propagation"
                    ),
                }
            )

        # --- Build result ---
        if not issues:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.9,
                failure_mode=None,
                explanation="No error handling issues detected in workflow JSON",
                detector_name=self.name,
            )

        # Determine severity
        if unprotected_ai_nodes or continue_on_fail_risk:
            severity = TurnAwareSeverity.SEVERE
        elif not has_error_trigger:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        # Confidence scales with number of distinct issue types found.
        # For unprotected_nodes alone (without AI nodes), require 3+ distinct
        # issue types to reach the default detection threshold.
        issue_types = {i["type"] for i in issues}
        if issue_types == {"unprotected_nodes"}:
            confidence = 0.45  # Below default threshold — not flagged alone
        elif len(issue_types) < 3 and "unprotected_ai_nodes" not in issue_types:
            confidence = min(0.65, 0.50 + len(issues) * 0.08)
        else:
            confidence = min(0.95, 0.70 + len(issues) * 0.08)

        explanations = [issue["explanation"] for issue in issues]
        full_explanation = "; ".join(explanations)

        fixes: List[str] = []
        if unprotected_ai_nodes:
            fixes.append("Add onError or continueOnFail with downstream validation to all AI nodes")
        if not has_error_trigger:
            fixes.append("Add an n8n-nodes-base.errorTrigger node to capture workflow-level errors")
        if continue_on_fail_risk:
            fixes.append(
                "Add IF/Switch nodes after continueOnFail nodes to validate data before "
                "it reaches critical downstream nodes (databases, APIs, AI)"
            )
        if unprotected_nodes:
            fixes.append("Review remaining unprotected nodes and add appropriate error handling")

        suggested_fix = "; ".join(fixes) if fixes else None

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F14",
            explanation=full_explanation,
            evidence={"issues": issues, "total_nodes": len(nodes)},
            suggested_fix=suggested_fix,
            detector_name=self.name,
            detector_version=self.version,
        )
