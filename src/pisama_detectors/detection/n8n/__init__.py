"""
n8n Workflow Failure Detection
==============================

Structural detectors for n8n workflow execution failures.
Unlike conversational detectors, these analyze workflow structure:
- Schema mismatches between nodes
- Graph cycles and execution loops
- Resource/token explosion
- Workflow timeouts and stalls
- Hidden error handling issues
- Workflow complexity problems

These detectors are framework-specific for n8n workflows and don't apply
to conversational multi-agent systems.
"""

from .schema_detector import N8NSchemaDetector
from .cycle_detector import N8NCycleDetector
from .resource_detector import N8NResourceDetector
from .timeout_detector import N8NTimeoutDetector
from .error_detector import N8NErrorDetector
from .complexity_detector import N8NComplexityDetector

__all__ = [
    "N8NSchemaDetector",
    "N8NCycleDetector",
    "N8NResourceDetector",
    "N8NTimeoutDetector",
    "N8NErrorDetector",
    "N8NComplexityDetector",
]
