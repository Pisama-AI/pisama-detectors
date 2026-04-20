"""
RAG Poisoning Detection for Dify Workflows
===========================================

Thin Dify-schema adapter on top of the framework-agnostic detector at
``backend/app/detection/rag_poisoning.py``. The injection-pattern table and
core scan/echo helpers live in the general module so LangGraph, OpenClaw, and
any future framework can reuse them.

Dify-specific: targets ``knowledge_retrieval`` node type and its
``outputs.documents[].content`` structure within ``workflow_run.nodes``.
"""

import logging
from typing import Any, Dict, List, Optional

from pisama_detectors.detection.rag_poisoning import (
    INJECTION_PATTERNS,
    check_llm_echo,
    scan_for_injections,
)
from pisama_detectors.detection.turn_aware._base import (
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
    TurnSnapshot,
)

logger = logging.getLogger(__name__)

class DifyRagPoisoningDetector(TurnAwareDetector):
    """Detects prompt injection payloads in Dify knowledge retrieval outputs.

    Scans knowledge_retrieval node documents for injection patterns and
    checks if downstream LLM nodes reproduce injected content, indicating
    successful poisoning.
    """

    name = "DifyRagPoisoningDetector"
    version = "1.1"
    supported_failure_modes = ["F6"]  # Prompt injection

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Delegate to detect_workflow_run if metadata contains workflow_run."""
        workflow_run = (conversation_metadata or {}).get("workflow_run", {})
        if workflow_run:
            return self.detect_workflow_run(workflow_run)
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation="No workflow_run data provided",
            detector_name=self.name,
        )

    def detect_workflow_run(self, workflow_run: dict) -> TurnAwareDetectionResult:
        """Analyze Dify workflow run for RAG poisoning.

        Args:
            workflow_run: Dify workflow_run dict with nodes list.

        Returns:
            Detection result with injection findings.
        """
        nodes = workflow_run.get("nodes", [])
        if not nodes:
            return self._no_detection("No nodes in workflow run")

        # Find knowledge_retrieval nodes and LLM nodes
        retrieval_nodes = [n for n in nodes if n.get("node_type") == "knowledge_retrieval"]
        llm_nodes = [n for n in nodes if n.get("node_type") == "llm"]

        if not retrieval_nodes:
            return self._no_detection("No knowledge_retrieval nodes found")

        issues: List[Dict[str, Any]] = []
        affected_node_ids: List[str] = []
        pattern_count = 0

        for node in retrieval_nodes:
            node_id = node.get("node_id", "unknown")
            outputs = node.get("outputs", {})
            documents = outputs.get("documents", [])

            for doc_idx, doc in enumerate(documents):
                content = doc.get("content", "") if isinstance(doc, dict) else str(doc)
                found_patterns = scan_for_injections(content)

                if found_patterns:
                    pattern_count += len(found_patterns)
                    affected_node_ids.append(node_id)
                    issues.append({
                        "type": "rag_injection",
                        "node_id": node_id,
                        "node_title": node.get("title", ""),
                        "document_index": doc_idx,
                        "patterns_found": found_patterns,
                        "content_preview": content[:200],
                    })

        # Check if downstream LLM echoes injected content
        if issues:
            echo_issues = self._check_llm_echo(issues, llm_nodes)
            issues.extend(echo_issues)
            for ei in echo_issues:
                affected_node_ids.append(ei.get("llm_node_id", ""))

        if not issues:
            return self._no_detection("No injection patterns in retrieved documents")

        # Confidence: 0.6 base + 0.1 per pattern, max 0.95
        confidence = min(0.95, 0.6 + pattern_count * 0.1)

        has_echo = any(i.get("type") == "llm_echo" for i in issues)
        if has_echo:
            severity = TurnAwareSeverity.SEVERE
        elif pattern_count >= 3:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F6",
            explanation=(
                f"RAG poisoning: {pattern_count} injection pattern(s) found in "
                f"{len(retrieval_nodes)} retrieval node(s)"
                + ("; LLM echoed injected content" if has_echo else "")
            ),
            affected_turns=list(range(len(affected_node_ids))),
            evidence={
                "issues": issues,
                "total_retrieval_nodes": len(retrieval_nodes),
                "pattern_count": pattern_count,
                "llm_echo_detected": has_echo,
            },
            suggested_fix=(
                "Sanitize retrieved documents before passing to LLM nodes. "
                "Add content filtering between knowledge_retrieval and llm nodes. "
                "Consider using Dify's sensitive word filter or a guardrail node."
            ),
            detector_name=self.name,
        )

    def _check_llm_echo(
        self,
        injection_issues: List[Dict[str, Any]],
        llm_nodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Check if LLM nodes echo injected patterns in their outputs.

        Wraps the shared ``check_llm_echo`` helper with Dify-specific result
        shaping (one issue per LLM node, with node_id/title metadata).
        """
        echo_issues: List[Dict[str, Any]] = []
        injected_strings = sorted({
            pat["matched"]
            for issue in injection_issues
            for pat in issue.get("patterns_found", [])
        })
        if not injected_strings:
            return echo_issues

        for llm_node in llm_nodes:
            llm_output = str(llm_node.get("outputs", {}))
            echoed = check_llm_echo(injected_strings, llm_output)
            if echoed:
                echo_issues.append({
                    "type": "llm_echo",
                    "llm_node_id": llm_node.get("node_id", ""),
                    "llm_node_title": llm_node.get("title", ""),
                    "echoed_patterns": echoed,
                    "output_preview": llm_output[:200],
                })
        return echo_issues

    def _no_detection(self, reason: str) -> TurnAwareDetectionResult:
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation=reason,
            detector_name=self.name,
        )
