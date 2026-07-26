"""Compile-time contracts for the typed public API."""

import pisama_detectors as pd
from pisama_detectors.detection.communication import CommunicationBreakdownResult
from pisama_detectors.detection.completion import CompletionResult
from pisama_detectors.detection.context import ContextNeglectResult
from pisama_detectors.detection.context_pressure import ContextPressureResult
from pisama_detectors.detection.convergence import ConvergenceResult
from pisama_detectors.detection.coordination import CoordinationAnalysisResult
from pisama_detectors.detection.corruption import CorruptionResult
from pisama_detectors.detection.cost import CostResult
from pisama_detectors.detection.decomposition import DecompositionResult
from pisama_detectors.detection.derailment import DerailmentResult
from pisama_detectors.detection.hallucination import HallucinationResult
from pisama_detectors.detection.injection import InjectionResult
from pisama_detectors.detection.loop import LoopDetectionResult
from pisama_detectors.detection.overflow import OverflowResult
from pisama_detectors.detection.persona import PersonaConsistencyResult
from pisama_detectors.detection.specification import SpecificationMismatchResult
from pisama_detectors.detection.turn_aware._base import TurnAwareDetectionResult
from pisama_detectors.detection.withholding import WithholdingResult
from pisama_detectors.detection.workflow import WorkflowAnalysisResult

loop_result: LoopDetectionResult = pd.detect_loop([])
corruption_result: CorruptionResult = pd.detect_corruption({}, {})
injection_result: InjectionResult = pd.detect_injection("")
hallucination_result: HallucinationResult = pd.detect_hallucination("", [])
hallucination_source: pd.HallucinationSource = {
    "content": "The API requires TLS.",
    "title": "Official Guide",
}
structured_hallucination_result: HallucinationResult = pd.detect_hallucination(
    "The API requires TLS (source: Official Guide).",
    [hallucination_source],
)
persona_result: PersonaConsistencyResult = pd.detect_persona_drift("", "", "")
coordination_result: CoordinationAnalysisResult = pd.detect_coordination([])
overflow_result: OverflowResult = pd.detect_overflow("", "")
provider_overflow_result: OverflowResult = pd.detect_overflow(
    "",
    "",
    provider_token_count=0,
)
derailment_result: DerailmentResult = pd.detect_derailment("", "")
context_result: ContextNeglectResult = pd.detect_context_neglect("", "")
communication_result: CommunicationBreakdownResult = pd.detect_communication("", "")
specification_result: SpecificationMismatchResult = pd.detect_specification("", "")
decomposition_result: DecompositionResult = pd.detect_decomposition("", [])
workflow_result: WorkflowAnalysisResult = pd.detect_workflow([])
withholding_result: WithholdingResult = pd.detect_withholding("", "")
completion_result: CompletionResult = pd.detect_completion("", [], "")
convergence_result: ConvergenceResult = pd.detect_convergence([])
cost_result: CostResult = pd.calculate_cost("claude-sonnet-4-6", 10, 5)
context_pressure_result: ContextPressureResult = pd.detect_context_pressure([])

langgraph_recursion: TurnAwareDetectionResult = pd.detect_langgraph_recursion({})
langgraph_state: TurnAwareDetectionResult = pd.detect_langgraph_state_corruption({})
langgraph_edge: TurnAwareDetectionResult = pd.detect_langgraph_edge_misroute({})
langgraph_checkpoint: TurnAwareDetectionResult = pd.detect_langgraph_checkpoint_corruption({})
langgraph_parallel: TurnAwareDetectionResult = pd.detect_langgraph_parallel_sync({})
langgraph_tool: TurnAwareDetectionResult = pd.detect_langgraph_tool_failure({})
dify_classifier: TurnAwareDetectionResult = pd.detect_dify_classifier_drift({})
dify_iteration: TurnAwareDetectionResult = pd.detect_dify_iteration_escape({})
dify_rag: TurnAwareDetectionResult = pd.detect_dify_rag_poisoning({})
dify_schema: TurnAwareDetectionResult = pd.detect_dify_tool_schema_mismatch({})
dify_variable: TurnAwareDetectionResult = pd.detect_dify_variable_leak({})
dify_fallback: TurnAwareDetectionResult = pd.detect_dify_model_fallback({})
n8n_cycle: TurnAwareDetectionResult = pd.detect_n8n_cycle({})
n8n_error: TurnAwareDetectionResult = pd.detect_n8n_error({})
n8n_timeout: TurnAwareDetectionResult = pd.detect_n8n_timeout({})
n8n_complexity: TurnAwareDetectionResult = pd.detect_n8n_complexity({})
n8n_schema: TurnAwareDetectionResult = pd.detect_n8n_schema({})
n8n_resource: TurnAwareDetectionResult = pd.detect_n8n_resource({})
openclaw_loop: TurnAwareDetectionResult = pd.detect_openclaw_session_loop({})
openclaw_sandbox: TurnAwareDetectionResult = pd.detect_openclaw_sandbox_escape({})
openclaw_tool: TurnAwareDetectionResult = pd.detect_openclaw_tool_abuse({})
openclaw_spawn: TurnAwareDetectionResult = pd.detect_openclaw_spawn_chain({})
openclaw_channel: TurnAwareDetectionResult = pd.detect_openclaw_channel_mismatch({})
openclaw_risk: TurnAwareDetectionResult = pd.detect_openclaw_elevated_risk({})
