"""
Model Fallback Detection for Dify Workflows
============================================

Detects silent model fallback in Dify LLM nodes where the requested
model differs from the model that actually served the response.
Catches both single-node model mismatches and multi-node fallback chains.

Dify-specific: targets llm node type. The actual serving model is found
in node.metadata.model (top-level node metadata), while the configured
model is in node.inputs.model.
"""

import logging
from typing import Any, Dict, List, Optional

from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)


class DifyModelFallbackDetector(TurnAwareDetector):
    """Detects silent model fallback in Dify LLM nodes.

    Checks for:
    - Input model config differing from actual serving model (in node.metadata)
    - Fallback reason indicators in node metadata
    - Fallback references in outputs or metadata text
    """

    name = "DifyModelFallbackDetector"
    version = "1.1"
    supported_failure_modes = ["F15"]  # Silent degradation

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        workflow_run = (conversation_metadata or {}).get("workflow_run", {})
        if workflow_run:
            return self.detect_workflow_run(workflow_run)
        return self._no_detection("No workflow_run data provided")

    def detect_workflow_run(self, workflow_run: dict) -> TurnAwareDetectionResult:
        nodes = workflow_run.get("nodes", [])
        if not nodes:
            return self._no_detection("No nodes in workflow run")

        llm_nodes = [n for n in nodes if n.get("node_type") == "llm"]
        if not llm_nodes:
            return self._no_detection("No LLM nodes found")

        issues: List[Dict[str, Any]] = []
        affected_node_ids: List[str] = []

        # Check 1: Per-node model mismatch (configured vs actual)
        for node in llm_nodes:
            mismatch = self._check_model_mismatch(node)
            if mismatch:
                issues.append(mismatch)
                affected_node_ids.append(node.get("node_id", ""))

        # Check 2: Fallback chain (failed LLM -> succeeding LLM with different model)
        chain_issues = self._check_fallback_chain(llm_nodes)
        for ci in chain_issues:
            issues.append(ci)
            affected_node_ids.extend(ci.get("node_ids", []))

        # Check 3: Fallback references in outputs or metadata
        for node in llm_nodes:
            ref_issue = self._check_fallback_references(node)
            if ref_issue:
                issues.append(ref_issue)
                affected_node_ids.append(node.get("node_id", ""))

        if not issues:
            return self._no_detection("No model fallback detected in LLM nodes")

        # Confidence
        has_mismatch = any(i["type"] == "model_mismatch" for i in issues)
        has_chain = any(i["type"] == "fallback_chain" for i in issues)
        has_metadata_reason = any(
            i.get("fallback_reason") for i in issues
            if i["type"] == "model_mismatch"
        )
        if has_chain:
            confidence = 0.85
        elif has_mismatch and has_metadata_reason:
            confidence = 0.90
        elif has_mismatch:
            confidence = 0.75
        else:
            confidence = 0.65

        # Severity
        if has_chain and has_mismatch:
            severity = TurnAwareSeverity.SEVERE
        elif has_chain or len(issues) >= 2:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F15",
            explanation=(
                f"Model fallback: {len(issues)} issue(s) in "
                f"{len(llm_nodes)} LLM node(s)"
            ),
            affected_turns=list(range(len(set(affected_node_ids)))),
            evidence={
                "issues": issues,
                "total_llm_nodes": len(llm_nodes),
            },
            suggested_fix=(
                "Pin model versions explicitly in LLM node configuration. "
                "Add error handling that alerts on model fallback instead of "
                "silently switching. Log the actual model used in workflow outputs."
            ),
            detector_name=self.name,
        )

    def _check_model_mismatch(self, node: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if a single LLM node's configured model differs from actual model."""
        inputs = node.get("inputs", {})
        outputs = node.get("outputs", {})
        node_metadata = node.get("metadata", {})

        input_model = self._extract_model_name(inputs, key="model")

        # Primary: check node.metadata.model (where Dify puts the actual serving model)
        output_model = self._extract_model_name(node_metadata, key="model")

        # Fallback: check outputs.model and outputs.metadata.model
        if not output_model:
            output_model = self._extract_model_name(outputs, key="model")
        if not output_model:
            output_metadata = outputs.get("metadata", {})
            if output_metadata:
                output_model = self._extract_model_name(output_metadata, key="model")

        if input_model and output_model and input_model != output_model:
            result = {
                "type": "model_mismatch",
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "requested_model": input_model,
                "actual_model": output_model,
            }
            # Check for explicit fallback reason in metadata
            fallback_reason = node_metadata.get("model_fallback_reason", "")
            if fallback_reason:
                result["fallback_reason"] = fallback_reason
            return result
        return None

    def _check_fallback_chain(
        self, llm_nodes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Detect fallback chains: failed or degraded LLM followed by success with different model."""
        issues = []
        for i in range(len(llm_nodes) - 1):
            curr = llm_nodes[i]
            nxt = llm_nodes[i + 1]

            curr_status = curr.get("status", "")
            nxt_status = nxt.get("status", "")

            # Check for explicit failure or for metadata indicating fallback
            curr_failed = curr_status == "failed"
            curr_has_fallback = bool(
                curr.get("metadata", {}).get("model_fallback_reason")
            )

            if (curr_failed or curr_has_fallback) and nxt_status in ("succeeded", "completed"):
                curr_model = self._extract_model_name(curr.get("inputs", {}), "model")
                nxt_model = self._extract_model_name(nxt.get("inputs", {}), "model")

                if curr_model and nxt_model and curr_model != nxt_model:
                    issues.append({
                        "type": "fallback_chain",
                        "failed_node_id": curr.get("node_id", ""),
                        "failed_title": curr.get("title", ""),
                        "failed_model": curr_model,
                        "fallback_node_id": nxt.get("node_id", ""),
                        "fallback_title": nxt.get("title", ""),
                        "fallback_model": nxt_model,
                        "node_ids": [
                            curr.get("node_id", ""),
                            nxt.get("node_id", ""),
                        ],
                    })
        return issues

    def _check_fallback_references(
        self, node: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check if outputs or metadata contain fallback/alternative model references."""
        # Check both outputs AND metadata for fallback indicators
        check_str = (
            str(node.get("outputs", {})).lower()
            + " "
            + str(node.get("metadata", {})).lower()
        )
        fallback_keywords = [
            "fallback", "alternative model", "backup model",
            "degraded", "model_fallback_reason",
        ]
        found = [kw for kw in fallback_keywords if kw in check_str]
        if found:
            return {
                "type": "fallback_reference",
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "keywords_found": found,
            }
        return None

    def _extract_model_name(
        self, data: Dict[str, Any], key: str = "model"
    ) -> Optional[str]:
        """Extract model name from a dict, handling nested structures."""
        if not isinstance(data, dict):
            return None
        val = data.get(key)
        if isinstance(val, str) and val:
            return val
        for alt_key in ("model_config", "model_name", "model_id"):
            val = data.get(alt_key)
            if isinstance(val, str) and val:
                return val
            if isinstance(val, dict):
                inner = val.get("model") or val.get("name") or val.get("id")
                if isinstance(inner, str) and inner:
                    return inner
        return None

    def _no_detection(self, reason: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=reason,
            detector_name=self.name,
        )
