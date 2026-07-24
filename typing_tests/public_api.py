"""Compile-time contracts for the typed public API."""

import pisama_detectors as pd
from pisama_detectors.detection.cost import CostResult
from pisama_detectors.detection.loop import LoopDetectionResult
from pisama_detectors.detection.persona import PersonaConsistencyResult
from pisama_detectors.detection.turn_aware._base import TurnAwareDetectionResult

loop_result: LoopDetectionResult = pd.detect_loop([])
cost_result: CostResult = pd.calculate_cost("claude-sonnet-4-6", 10, 5)
persona_result: PersonaConsistencyResult = pd.detect_persona_drift(
    "reviewer",
    "Evidence-focused reviewer",
    "I checked the evidence.",
)
framework_result: TurnAwareDetectionResult = pd.detect_n8n_cycle({})
