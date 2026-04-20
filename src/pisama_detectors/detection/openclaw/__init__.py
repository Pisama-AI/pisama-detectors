"""
OpenClaw Multi-Agent Platform Detection
========================================

Framework-specific detectors for the OpenClaw multi-agent platform.
These analyze OpenClaw session event streams for:
- Session loops (repeated tool calls and ping-pong spawns)
- Tool abuse (excessive calls, high error rates, sensitive tools)
- Elevated privilege risks (risky ops in elevated mode or escalation attempts)
- Spawn chain depth and circular references
- Channel-specific content mismatches
- Sandbox escape attempts

All detectors consume the OpenClaw session format with typed events.
"""

from .session_loop_detector import OpenClawSessionLoopDetector
from .tool_abuse_detector import OpenClawToolAbuseDetector
from .elevated_risk_detector import OpenClawElevatedRiskDetector
from .spawn_chain_detector import OpenClawSpawnChainDetector
from .channel_mismatch_detector import OpenClawChannelMismatchDetector
from .sandbox_escape_detector import OpenClawSandboxEscapeDetector

__all__ = [
    "OpenClawSessionLoopDetector",
    "OpenClawToolAbuseDetector",
    "OpenClawElevatedRiskDetector",
    "OpenClawSpawnChainDetector",
    "OpenClawChannelMismatchDetector",
    "OpenClawSandboxEscapeDetector",
]
