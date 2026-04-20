"""
Dify Workflow Failure Detection
===============================

Framework-specific detectors for Dify workflow platform.
These analyze Dify workflow_run execution data for:
- RAG document poisoning via knowledge retrieval nodes
- Iteration/loop escape and runaway patterns
- Silent model fallback without user awareness
- Sensitive variable leakage across node boundaries
- Question classifier drift and miscategorization
- Tool node schema validation mismatches

These detectors are framework-specific for Dify workflows and complement
the generic turn-aware detectors with Dify-specific node type awareness.
"""

from .rag_poisoning_detector import DifyRagPoisoningDetector
from .iteration_escape_detector import DifyIterationEscapeDetector
from .model_fallback_detector import DifyModelFallbackDetector
from .variable_leak_detector import DifyVariableLeakDetector
from .classifier_drift_detector import DifyClassifierDriftDetector
from .tool_schema_mismatch_detector import DifyToolSchemaMismatchDetector

__all__ = [
    "DifyRagPoisoningDetector",
    "DifyIterationEscapeDetector",
    "DifyModelFallbackDetector",
    "DifyVariableLeakDetector",
    "DifyClassifierDriftDetector",
    "DifyToolSchemaMismatchDetector",
]
