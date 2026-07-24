"""
LangGraph State Corruption Detector
=====================================

Detects state corruption between consecutive state_snapshots:
- Type changes: a key's value type changes (str->int, list->dict)
- Null injections: a key becomes None when previously set
- Unexpected field deletions: key present in snapshot N, missing in N+1
- Value explosion: list/dict size grows >10x between snapshots
- List shrinkage: append-only list getting shorter
- Identity field mutation: user_id, session_id, etc. changing value
- Counter decrease: monotonic counter going backwards
- Numeric value jumps: unreasonable scalar changes
- New field injection: unexpected keys appearing in state
- Node error signals: failed nodes with corruption-related errors
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

# Fields that should never change once set (identity fields)
IDENTITY_FIELDS = {
    "user_id",
    "session_id",
    "thread_id",
    "graph_id",
    "run_id",
    "conversation_id",
    "agent_id",
    "workflow_id",
    "trace_id",
}

# Fields expected to be monotonically increasing
COUNTER_FIELDS = {
    "step_count",
    "iteration_count",
    "turn_count",
    "message_count",
    "retry_count",
    "attempt_count",
    "total_steps",
    "superstep",
}

# Keywords in error messages that indicate state corruption
CORRUPTION_ERROR_KEYWORDS = {
    "corrupt",
    "invalid state",
    "state mismatch",
    "inconsistent",
    "integrity",
    "schema violation",
    "type error",
    "key error",
    "missing key",
    "unexpected type",
    "deserialization",
    "serialization",
    "state_error",
    "checkpoint",
    "invalid value",
    "malformed",
}


def _value_size(value: Any) -> int:
    """Return the size of a container value, or 0 for scalars."""
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, dict):
        return len(value)
    if isinstance(value, str):
        return len(value)
    return 0


def _type_label(value: Any) -> str:
    """Return a readable type label for a value."""
    if value is None:
        return "null"
    return type(value).__name__


class LangGraphStateCorruptionDetector(TurnAwareDetector):
    """Detects state corruption across LangGraph state snapshots.

    Compares consecutive state_snapshots for:
    1. Type changes on the same key
    2. Null injections (non-None -> None)
    3. Unexpected field deletions
    4. Value explosion (container size grows >10x)
    5. List shrinkage (append-only violation)
    6. Identity field mutation
    7. Counter decrease
    8. Numeric value jumps (>100x)
    9. New field injection
    10. Node error signals with corruption keywords
    """

    name = "LangGraphStateCorruptionDetector"
    version = "1.1"
    supported_failure_modes = ["F3"]

    def __init__(self, explosion_factor: float = 10.0):
        self.explosion_factor = explosion_factor

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Delegate to graph execution analysis."""
        graph_execution = (conversation_metadata or {}).get("graph_execution", {})
        if graph_execution:
            return self.detect_graph_execution(graph_execution)
        return TurnAwareDetectionResult(
            detected=False,
            severity=TurnAwareSeverity.NONE,
            confidence=0.0,
            failure_mode=None,
            explanation="No graph_execution data provided",
            detector_name=self.name,
        )

    def detect_graph_execution(self, graph_execution: Dict[str, Any]) -> TurnAwareDetectionResult:
        """Analyze state_snapshots and nodes for corruption signals."""
        snapshots = graph_execution.get("state_snapshots", [])
        nodes = graph_execution.get("nodes", [])

        corruption_signals: List[Dict[str, Any]] = []
        affected_supersteps: List[int] = []

        # --- Check state snapshots for corruption ---
        if len(snapshots) >= 2:
            snapshots = sorted(snapshots, key=lambda s: s.get("superstep", 0))

            for i in range(len(snapshots) - 1):
                prev_state = snapshots[i].get("state", {}) or {}
                curr_state = snapshots[i + 1].get("state", {}) or {}
                prev_step = snapshots[i].get("superstep", i)
                curr_step = snapshots[i + 1].get("superstep", i + 1)

                signals = self._compare_states(prev_state, curr_state, prev_step, curr_step)
                corruption_signals.extend(signals)
                for s in signals:
                    affected_supersteps.append(s.get("superstep_to", curr_step))

        # --- Check nodes for error signals ---
        if nodes:
            node_signals = self._detect_node_errors(nodes)
            corruption_signals.extend(node_signals)
            for s in node_signals:
                step = s.get("superstep", -1)
                if step >= 0:
                    affected_supersteps.append(step)

        if not corruption_signals:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.85,
                failure_mode=None,
                explanation="No state corruption detected across snapshots",
                detector_name=self.name,
            )

        # Confidence: 0.6 base + 0.1 per signal, capped at 0.95
        confidence = min(0.95, 0.6 + len(corruption_signals) * 0.1)

        # Severity based on signal types
        signal_types = {s["type"] for s in corruption_signals}
        severe_types = {
            "type_change",
            "field_deletion",
            "identity_mutation",
            "node_error",
        }
        moderate_types = {
            "null_injection",
            "counter_decrease",
            "counter_stall",
            "list_shrinkage",
        }

        if signal_types & severe_types:
            severity = TurnAwareSeverity.SEVERE
        elif signal_types & moderate_types:
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        type_counts: Dict[str, int] = {}
        for s in corruption_signals:
            type_counts[s["type"]] = type_counts.get(s["type"], 0) + 1
        summary_parts = [f"{count} {stype}" for stype, count in type_counts.items()]

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F3",
            explanation=(
                f"State corruption detected: {', '.join(summary_parts)} "
                f"across {max(len(snapshots), 1)} snapshots"
            ),
            affected_turns=sorted(set(affected_supersteps)),
            evidence={
                "signals": corruption_signals,
                "total_snapshots": len(snapshots),
                "signal_count": len(corruption_signals),
            },
            suggested_fix=(
                "Validate state schema between supersteps. Use TypedDict or "
                "Pydantic models for state to enforce type safety. Add state "
                "reducers that handle None values gracefully."
            ),
            detector_name=self.name,
        )

    def _compare_states(
        self,
        prev_state: Dict[str, Any],
        curr_state: Dict[str, Any],
        prev_step: int,
        curr_step: int,
    ) -> List[Dict[str, Any]]:
        """Compare two consecutive states and return corruption signals."""
        signals: List[Dict[str, Any]] = []

        # Check all keys in previous state
        for key in prev_state:
            prev_val = prev_state[key]

            if key not in curr_state:
                # Field deletion
                signals.append(
                    {
                        "type": "field_deletion",
                        "key": key,
                        "superstep_from": prev_step,
                        "superstep_to": curr_step,
                        "description": (
                            f"Key '{key}' present at superstep {prev_step} "
                            f"but missing at superstep {curr_step}"
                        ),
                    }
                )
                continue

            curr_val = curr_state[key]

            # Null injection
            if prev_val is not None and curr_val is None:
                signals.append(
                    {
                        "type": "null_injection",
                        "key": key,
                        "previous_type": _type_label(prev_val),
                        "superstep_from": prev_step,
                        "superstep_to": curr_step,
                        "description": (
                            f"Key '{key}' changed from {_type_label(prev_val)} "
                            f"to null at superstep {curr_step}"
                        ),
                    }
                )
                continue

            # Type change (including None → concrete type, which indicates
            # unexpected state initialization that may be corruption)
            if (
                prev_val is not None
                and curr_val is not None
                and type(prev_val) is not type(curr_val)
            ) or (
                prev_val is None
                and curr_val is not None
                and key not in IDENTITY_FIELDS  # New fields appearing is normal for identities
            ):
                signals.append(
                    {
                        "type": "type_change",
                        "key": key,
                        "previous_type": _type_label(prev_val),
                        "current_type": _type_label(curr_val),
                        "superstep_from": prev_step,
                        "superstep_to": curr_step,
                        "description": (
                            f"Key '{key}' type changed from "
                            f"{_type_label(prev_val)} to {_type_label(curr_val)} "
                            f"at superstep {curr_step}"
                        ),
                    }
                )
                continue

            # Value explosion
            prev_size = _value_size(prev_val)
            curr_size = _value_size(curr_val)
            if prev_size > 0 and curr_size > prev_size * self.explosion_factor:
                signals.append(
                    {
                        "type": "value_explosion",
                        "key": key,
                        "previous_size": prev_size,
                        "current_size": curr_size,
                        "growth_factor": round(curr_size / prev_size, 1),
                        "superstep_from": prev_step,
                        "superstep_to": curr_step,
                        "description": (
                            f"Key '{key}' size grew from {prev_size} to "
                            f"{curr_size} ({curr_size / prev_size:.0f}x) "
                            f"at superstep {curr_step}"
                        ),
                    }
                )
                continue

            # List shrinkage (append-only violation)
            if isinstance(prev_val, list) and isinstance(curr_val, list):
                if len(curr_val) < len(prev_val) and len(prev_val) >= 2:
                    signals.append(
                        {
                            "type": "list_shrinkage",
                            "key": key,
                            "previous_length": len(prev_val),
                            "current_length": len(curr_val),
                            "superstep_from": prev_step,
                            "superstep_to": curr_step,
                            "description": (
                                f"Key '{key}' list shrank from {len(prev_val)} "
                                f"to {len(curr_val)} items at superstep {curr_step}"
                            ),
                        }
                    )
                    continue

            # Identity field mutation
            key_lower = key.lower()
            if any(idf in key_lower for idf in IDENTITY_FIELDS):
                if prev_val != curr_val and prev_val is not None and curr_val is not None:
                    signals.append(
                        {
                            "type": "identity_mutation",
                            "key": key,
                            "previous_value": str(prev_val)[:100],
                            "current_value": str(curr_val)[:100],
                            "superstep_from": prev_step,
                            "superstep_to": curr_step,
                            "description": (
                                f"Identity field '{key}' changed from "
                                f"'{str(prev_val)[:50]}' to '{str(curr_val)[:50]}' "
                                f"at superstep {curr_step}"
                            ),
                        }
                    )
                    continue

            # Counter decrease or stall
            if any(cf in key_lower for cf in COUNTER_FIELDS):
                if (
                    isinstance(prev_val, (int, float))
                    and isinstance(curr_val, (int, float))
                    and curr_val < prev_val
                ):
                    signals.append(
                        {
                            "type": "counter_decrease",
                            "key": key,
                            "previous_value": prev_val,
                            "current_value": curr_val,
                            "superstep_from": prev_step,
                            "superstep_to": curr_step,
                            "description": (
                                f"Counter '{key}' decreased from {prev_val} "
                                f"to {curr_val} at superstep {curr_step}"
                            ),
                        }
                    )
                    continue
                # v2.1: Stalled counter — value should increase but doesn't
                if (
                    isinstance(prev_val, (int, float))
                    and isinstance(curr_val, (int, float))
                    and curr_val == prev_val
                    and prev_val > 0  # Only flag if counter started (not 0→0)
                    and curr_step - prev_step >= 1
                ):
                    signals.append(
                        {
                            "type": "counter_stall",
                            "key": key,
                            "value": curr_val,
                            "superstep_from": prev_step,
                            "superstep_to": curr_step,
                            "description": (
                                f"Counter '{key}' stalled at {curr_val} "
                                f"from superstep {prev_step} to {curr_step}"
                            ),
                        }
                    )
                    continue

            # Numeric value jump (>100x for non-zero values)
            if (
                isinstance(prev_val, (int, float))
                and isinstance(curr_val, (int, float))
                and prev_val != 0
                and abs(curr_val) > abs(prev_val) * 100
            ):
                signals.append(
                    {
                        "type": "value_jump",
                        "key": key,
                        "previous_value": prev_val,
                        "current_value": curr_val,
                        "jump_factor": round(abs(curr_val / prev_val), 1),
                        "superstep_from": prev_step,
                        "superstep_to": curr_step,
                        "description": (
                            f"Key '{key}' value jumped from {prev_val} to "
                            f"{curr_val} ({abs(curr_val / prev_val):.0f}x) "
                            f"at superstep {curr_step}"
                        ),
                    }
                )

        # New field injection: keys in curr but not in prev
        # Only flag if prev_state had reasonable number of keys (established schema)
        if len(prev_state) >= 3:
            new_keys = set(curr_state.keys()) - set(prev_state.keys())
            # Filter out common benign additions
            benign_prefixes = {"_", "debug_", "log_", "timestamp"}
            suspicious_new = [
                k for k in new_keys if not any(k.startswith(bp) for bp in benign_prefixes)
            ]
            if len(suspicious_new) >= 2:
                signals.append(
                    {
                        "type": "field_injection",
                        "new_keys": sorted(suspicious_new),
                        "superstep_from": prev_step,
                        "superstep_to": curr_step,
                        "description": (
                            f"{len(suspicious_new)} new field(s) injected at "
                            f"superstep {curr_step}: {', '.join(sorted(suspicious_new)[:5])}"
                        ),
                    }
                )

        return signals

    def _detect_node_errors(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Check failed nodes for corruption-related error messages."""
        signals: List[Dict[str, Any]] = []

        for node in nodes:
            status = node.get("status", "")
            if status != "failed":
                continue

            # Check error field
            error = node.get("error", "")
            error_str = str(error).lower()

            # Check outputs for error info too
            outputs = node.get("outputs", {})
            outputs_str = str(outputs).lower()

            combined = error_str + " " + outputs_str

            matched_keywords = [kw for kw in CORRUPTION_ERROR_KEYWORDS if kw in combined]

            if matched_keywords:
                signals.append(
                    {
                        "type": "node_error",
                        "node_id": node.get("node_id", ""),
                        "node_type": node.get("node_type", ""),
                        "superstep": node.get("superstep", -1),
                        "error_keywords": matched_keywords,
                        "error_preview": str(error)[:200],
                        "description": (
                            f"Node '{node.get('node_id', '')}' failed with "
                            f"corruption indicators: {', '.join(matched_keywords)}"
                        ),
                    }
                )

        return signals
