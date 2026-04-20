"""
LangGraph Failure Detection
============================

Structural detectors for LangGraph graph-based agent execution failures.
Unlike conversational detectors, these analyze graph execution structure:
- Recursion limit and unbounded cycles
- State corruption between supersteps
- Conditional edge misrouting
- Tool node failures and recovery
- Parallel execution sync issues
- Checkpoint integrity violations

These detectors are framework-specific for LangGraph and consume
the ``graph_execution`` data format.
"""

from .recursion_detector import LangGraphRecursionDetector
from .state_corruption_detector import LangGraphStateCorruptionDetector
from .edge_misroute_detector import LangGraphEdgeMisrouteDetector
from .tool_failure_detector import LangGraphToolFailureDetector
from .parallel_sync_detector import LangGraphParallelSyncDetector
from .checkpoint_corruption_detector import LangGraphCheckpointCorruptionDetector

__all__ = [
    "LangGraphRecursionDetector",
    "LangGraphStateCorruptionDetector",
    "LangGraphEdgeMisrouteDetector",
    "LangGraphToolFailureDetector",
    "LangGraphParallelSyncDetector",
    "LangGraphCheckpointCorruptionDetector",
]
