"""Public API for Pisama detectors.

Provides simplified functions that wrap the core detection algorithms.
Each function takes plain Python dicts/lists and returns a result dataclass.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, ParamSpec, TypedDict, TypeVar

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

_P = ParamSpec("_P")
_R = TypeVar("_R")


class _HallucinationSourceContent(TypedDict):
    """Required fields for a public structured grounding source."""

    content: str


class HallucinationSource(_HallucinationSourceContent, total=False):
    """Structured source input with optional named-citation metadata."""

    metadata: Mapping[str, Any]
    id: Any
    label: Any
    name: Any
    source: Any
    title: Any
    url: Any


@dataclass
class DetectorInfo:
    """Registry entry for a detector."""

    name: str
    description: str
    function: Callable[..., Any]
    tier: str  # production, beta, experimental


# Will be populated below
DETECTOR_REGISTRY: Dict[str, DetectorInfo] = {}


def _register(
    name: str,
    description: str,
    tier: str = "production",
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorator to register a detector function."""

    def decorator(fn: Callable[_P, _R]) -> Callable[_P, _R]:
        DETECTOR_REGISTRY[name] = DetectorInfo(
            name=name,
            description=description,
            function=fn,
            tier=tier,
        )
        return fn

    return decorator


@_register("loop", "Detect infinite loops and repetitive patterns", "production")
def detect_loop(
    states: List[Dict[str, Any]],
    window_size: int = 5,
    similarity_threshold: float = 0.85,
) -> LoopDetectionResult:
    """Detect infinite loops in agent state sequences.

    Args:
        states: List of agent state dicts (each representing a step)
        window_size: Total recent-state window, including the current state
        similarity_threshold: Similarity threshold for semantic loop detection

    Returns:
        LoopDetectionResult with detected, confidence, loop_type, etc.

    Raises:
        ValueError: If window_size is less than 1 or similarity_threshold is
            outside the inclusive range from 0 to 1.
    """
    import json

    from pisama_detectors.detection.loop import MultiLevelLoopDetector, StateSnapshot

    if window_size < 1:
        raise ValueError("window_size must be at least 1")
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1")

    snapshots = []
    for i, state in enumerate(states):
        snapshots.append(
            StateSnapshot(
                agent_id=str(state.get("agent_id", "agent")),
                state_delta=state,
                content=json.dumps(state, sort_keys=True, default=str),
                sequence_num=i,
            )
        )

    detector = MultiLevelLoopDetector(
        window_size=window_size,
        semantic_threshold=similarity_threshold,
    )
    return detector.detect_loop(snapshots)


@_register("corruption", "Detect state corruption and invalid transitions", "production")
def detect_corruption(
    prev_state: Dict[str, Any],
    current_state: Dict[str, Any],
) -> CorruptionResult:
    """Detect state corruption between consecutive states.

    Args:
        prev_state: Previous agent state dict
        current_state: Current agent state dict

    Returns:
        CorruptionResult with detected, confidence, issues
    """
    from pisama_detectors.detection.corruption import SemanticCorruptionDetector, StateSnapshot

    detector = SemanticCorruptionDetector()
    prev = StateSnapshot(state_delta=prev_state, agent_id=str(prev_state.get("agent_id", "agent")))
    curr = StateSnapshot(
        state_delta=current_state, agent_id=str(current_state.get("agent_id", "agent"))
    )
    return detector.detect_corruption_with_confidence(prev_state=prev, current_state=curr)


@_register("injection", "Detect prompt injection and jailbreak attempts", "production")
def detect_injection(
    text: str,
) -> InjectionResult:
    """Detect prompt injection or jailbreak attempts in text.

    Args:
        text: Input text to check for injection

    Returns:
        InjectionResult with detected, confidence, attack_type
    """
    from pisama_detectors.detection.injection import InjectionDetector

    detector = InjectionDetector()
    return detector.detect_injection(text=text)


@_register("hallucination", "Detect factual inaccuracies and fabrications", "production")
def detect_hallucination(
    output: str,
    sources: Optional[Sequence[str | HallucinationSource]] = None,
) -> HallucinationResult:
    """Detect hallucinations in agent output.

    Args:
        output: Agent output text
        sources: Optional list of source documents for grounding

    Returns:
        HallucinationResult with detected, confidence, issues
    """
    from pisama_detectors.detection.hallucination import (
        HallucinationDetector,
        SourceDocument,
    )

    normalized_sources = []
    for source in sources or ():
        if isinstance(source, str):
            normalized_sources.append(SourceDocument(content=source))
            continue
        if not isinstance(source, Mapping):
            raise TypeError("Each hallucination source must be a string or mapping")
        content = source.get("content")
        if not isinstance(content, str):
            raise TypeError("Structured hallucination sources require string 'content'")
        raw_metadata = source.get("metadata", {})
        if not isinstance(raw_metadata, Mapping):
            raise TypeError("Structured source 'metadata' must be a mapping")
        metadata = dict(raw_metadata)
        for key in ("id", "label", "name", "source", "title", "url"):
            if key in source:
                metadata.setdefault(key, source[key])
        normalized_sources.append(SourceDocument(content=content, metadata=metadata))

    detector = HallucinationDetector()
    return detector.detect_hallucination(
        output=output,
        sources=normalized_sources or None,
    )


@_register("persona_drift", "Detect persona drift and role confusion", "production")
def detect_persona_drift(
    agent_id: str,
    persona_description: str,
    output: str,
    allowed_actions: Optional[List[str]] = None,
) -> PersonaConsistencyResult:
    """Detect persona drift in agent behavior.

    Args:
        agent_id: Agent identifier
        persona_description: Expected persona/role description
        output: Agent output to check
        allowed_actions: Optional list of allowed actions

    Returns:
        PersonaConsistencyResult with score, deviations
    """
    from pisama_detectors.detection.persona import Agent, PersonaConsistencyScorer

    agent = Agent(
        id=agent_id,
        persona_description=persona_description,
        allowed_actions=allowed_actions or [],
    )
    scorer = PersonaConsistencyScorer()
    return scorer.score_consistency(agent=agent, output=output)


@_register("coordination", "Detect agent handoff and communication failures", "production")
def detect_coordination(
    messages: List[Dict[str, Any]],
    agent_ids: Optional[List[str]] = None,
) -> CoordinationAnalysisResult:
    """Detect coordination failures between agents.

    Args:
        messages: List of inter-agent messages with sender, receiver, content
        agent_ids: Optional list of agent IDs in the system

    Returns:
        CoordinationAnalysisResult with issues, severity
    """
    from pisama_detectors.detection.coordination import CoordinationAnalyzer, Message

    parsed_messages = []
    for idx, msg in enumerate(messages):
        parsed_messages.append(
            Message(
                from_agent=msg.get("sender", msg.get("from_agent", "unknown")),
                to_agent=msg.get("receiver", msg.get("to_agent", "unknown")),
                content=msg.get("content", ""),
                timestamp=float(msg.get("timestamp", idx)),
                acknowledged=msg.get("acknowledged", False),
            )
        )

    analyzer = CoordinationAnalyzer()
    return analyzer.analyze_coordination_with_confidence(
        messages=parsed_messages,
        agent_ids=agent_ids or [],
    )


@_register("overflow", "Detect context window exhaustion", "production")
def detect_overflow(
    context: str,
    output: str,
    model: str = "claude-sonnet-4-6",
    *,
    provider_token_count: Optional[int] = None,
) -> OverflowResult:
    """Detect context overflow issues.

    Args:
        context: Context/conversation to inspect. If it already includes the
            latest agent output, pass an empty ``output`` to avoid double counting.
        output: Separately supplied latest agent output. Every non-empty value
            is counted in addition to ``context``.
        model: LLM model name (for token limit lookup)
        provider_token_count: Optional provider-reported count for the
            complete request represented by ``context`` and ``output``. When
            omitted, the detector uses a bounded offline estimate. Claude uses
            ``cl100k_base`` as a proxy, not Anthropic's proprietary tokenizer.

    Returns:
        OverflowResult with detected, severity, token counts

    Raises:
        ValueError: If provider_token_count is not a non-negative integer.
    """
    from pisama_detectors.detection.overflow import ContextOverflowDetector

    detector = ContextOverflowDetector()
    if provider_token_count is not None:
        if (
            isinstance(provider_token_count, bool)
            or not isinstance(provider_token_count, int)
            or provider_token_count < 0
        ):
            raise ValueError("provider_token_count must be a non-negative integer")
        current_tokens = provider_token_count
        token_count_source = "provider"
    else:
        current_tokens = detector.count_tokens(context, model)
        current_tokens += detector.count_tokens(output, model)
        token_count_source = "offline_estimate"

    result = detector.detect_overflow(current_tokens=current_tokens, model=model)
    result.details["token_count_source"] = token_count_source
    return result


@_register("derailment", "Detect task focus deviation", "beta")
def detect_derailment(
    task: str,
    output: str,
) -> DerailmentResult:
    """Detect task derailment.

    Args:
        task: Original task description
        output: Agent output

    Returns:
        DerailmentResult with detected, severity, explanation
    """
    from pisama_detectors.detection.derailment import TaskDerailmentDetector

    detector = TaskDerailmentDetector()
    return detector.detect(task=task, output=output)


@_register("context_neglect", "Detect context neglect in responses", "production")
def detect_context_neglect(
    context: str,
    output: str,
) -> ContextNeglectResult:
    """Detect context neglect.

    Args:
        context: Provided context
        output: Agent output

    Returns:
        ContextNeglectResult with detected, severity
    """
    from pisama_detectors.detection.context import ContextNeglectDetector

    detector = ContextNeglectDetector()
    return detector.detect(context=context, output=output)


@_register("communication", "Detect inter-agent communication breakdowns", "beta")
def detect_communication(
    sender_message: str,
    receiver_response: str,
) -> CommunicationBreakdownResult:
    """Detect communication breakdown between agents.

    Args:
        sender_message: Message from sender agent
        receiver_response: Response from receiver agent

    Returns:
        CommunicationBreakdownResult with detected, breakdown_type
    """
    from pisama_detectors.detection.communication import CommunicationBreakdownDetector

    detector = CommunicationBreakdownDetector()
    return detector.detect(
        sender_message=sender_message,
        receiver_response=receiver_response,
    )


@_register("specification", "Detect output vs spec mismatch", "production")
def detect_specification(
    user_intent: str,
    task_specification: str,
) -> SpecificationMismatchResult:
    """Detect specification mismatch.

    Args:
        user_intent: What the user asked for
        task_specification: What was specified/implemented

    Returns:
        SpecificationMismatchResult with detected, mismatch_type
    """
    from pisama_detectors.detection.specification import SpecificationMismatchDetector

    detector = SpecificationMismatchDetector()
    return detector.detect(
        user_intent=user_intent,
        task_specification=task_specification,
    )


@_register("decomposition", "Detect task breakdown failures", "production")
def detect_decomposition(
    task_description: str,
    decomposition: List[Dict[str, Any] | str],
) -> DecompositionResult:
    """Detect task decomposition failures.

    Args:
        task_description: Original task description
        decomposition: List of subtask dicts

    Returns:
        DecompositionResult with detected, issues
    """
    from pisama_detectors.detection.decomposition import TaskDecompositionDetector

    detector = TaskDecompositionDetector()
    return detector.detect(
        task_description=task_description,
        decomposition=decomposition,
    )


@_register("workflow", "Detect workflow execution issues", "beta")
def detect_workflow(
    nodes: List[Dict[str, Any]],
) -> WorkflowAnalysisResult:
    """Detect workflow design and execution issues.

    Args:
        nodes: List of workflow node dicts with keys id, name, node_type,
            incoming, outgoing, has_error_handler (optional), is_terminal (optional).

    Returns:
        WorkflowAnalysisResult with issues
    """
    from pisama_detectors.detection.workflow import FlawedWorkflowDetector, WorkflowNode

    workflow_nodes = [
        WorkflowNode(
            id=n.get("id", f"node_{i}"),
            name=n.get("name", n.get("id", f"node_{i}")),
            node_type=n.get("node_type", "agent"),
            incoming=n.get("incoming", []),
            outgoing=n.get("outgoing", []),
            has_error_handler=n.get("has_error_handler", False),
            is_terminal=n.get("is_terminal", False),
        )
        for i, n in enumerate(nodes)
    ]
    detector = FlawedWorkflowDetector()
    return detector.detect(workflow_nodes)


@_register("withholding", "Detect information withholding", "beta")
def detect_withholding(
    agent_output: str,
    internal_state: Any,
) -> WithholdingResult:
    """Detect information withholding.

    Args:
        agent_output: What the agent said
        internal_state: Agent's internal state (str or dict; dicts are serialized)

    Returns:
        WithholdingResult with detected, issues
    """
    import json

    from pisama_detectors.detection.withholding import InformationWithholdingDetector

    if not isinstance(internal_state, str):
        internal_state = json.dumps(internal_state, default=str)

    detector = InformationWithholdingDetector()
    return detector.detect(
        agent_output=agent_output,
        internal_state=internal_state,
    )


@_register("completion", "Detect premature/delayed task completion", "beta")
def detect_completion(
    task: str,
    subtasks: List[str],
    agent_output: str,
    success_criteria: Optional[List[str]] = None,
) -> CompletionResult:
    """Detect completion misjudgment.

    Args:
        task: Original task
        subtasks: List of subtasks
        agent_output: Agent's output
        success_criteria: Optional success criteria

    Returns:
        CompletionResult with detected, issues
    """
    from pisama_detectors.detection.completion import CompletionMisjudgmentDetector

    detector = CompletionMisjudgmentDetector()
    return detector.detect(
        task=task,
        subtasks=subtasks,
        agent_output=agent_output,
        success_criteria=success_criteria or [],
    )


@_register("convergence", "Detect metric plateau, regression, thrashing", "production")
def detect_convergence(
    metrics: List[float],
    direction: str = "minimize",
    window_size: int = 5,
) -> ConvergenceResult:
    """Detect convergence failures in optimization metrics.

    Args:
        metrics: List of metric values over time
        direction: 'minimize' or 'maximize'
        window_size: Window size for analysis

    Returns:
        ConvergenceResult with detected, failure_type, severity
    """
    from pisama_detectors.detection.convergence import ConvergenceDetector

    # Normalize metrics to dicts if plain floats
    normalized = []
    for i, m in enumerate(metrics):
        if isinstance(m, (int, float)):
            normalized.append({"step": i, "value": m})
        else:
            normalized.append(m)

    detector = ConvergenceDetector()
    return detector.detect_convergence_issues(
        metrics=normalized,
        direction=direction,
        window_size=window_size,
    )


@_register("cost", "Track token/cost budget", "production")
def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> CostResult:
    """Calculate LLM cost.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        CostResult with cost_usd, tokens
    """
    from pisama_detectors.detection.cost import CostCalculator

    calculator = CostCalculator()
    return calculator.calculate_cost(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ============================================================
# LangGraph-specific detectors
# ============================================================


@_register("langgraph_recursion", "Detect LangGraph recursion limit issues", "production")
def detect_langgraph_recursion(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect recursion issues in LangGraph executions."""
    from pisama_detectors.detection.langgraph import LangGraphRecursionDetector

    detector = LangGraphRecursionDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_state_corruption", "Detect LangGraph state corruption", "production")
def detect_langgraph_state_corruption(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect state corruption in LangGraph graph state."""
    from pisama_detectors.detection.langgraph import LangGraphStateCorruptionDetector

    detector = LangGraphStateCorruptionDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_edge_misroute", "Detect LangGraph edge misrouting", "beta")
def detect_langgraph_edge_misroute(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect edge misrouting in LangGraph conditional edges."""
    from pisama_detectors.detection.langgraph import LangGraphEdgeMisrouteDetector

    detector = LangGraphEdgeMisrouteDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_checkpoint_corruption", "Detect LangGraph checkpoint corruption", "beta")
def detect_langgraph_checkpoint_corruption(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect checkpoint corruption in LangGraph persistence."""
    from pisama_detectors.detection.langgraph import LangGraphCheckpointCorruptionDetector

    detector = LangGraphCheckpointCorruptionDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_parallel_sync", "Detect LangGraph parallel branch sync failures", "beta")
def detect_langgraph_parallel_sync(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect parallel branch synchronization issues in LangGraph."""
    from pisama_detectors.detection.langgraph import LangGraphParallelSyncDetector

    detector = LangGraphParallelSyncDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_tool_failure", "Detect LangGraph tool execution failures", "production")
def detect_langgraph_tool_failure(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect tool execution failures in LangGraph."""
    from pisama_detectors.detection.langgraph import LangGraphToolFailureDetector

    detector = LangGraphToolFailureDetector()
    return detector.detect_graph_execution(trace)


# ============================================================
# Dify-specific detectors
# ============================================================


@_register("dify_classifier_drift", "Detect Dify classifier drift", "beta")
def detect_dify_classifier_drift(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect classifier drift in Dify intent routing."""
    from pisama_detectors.detection.dify import DifyClassifierDriftDetector

    detector = DifyClassifierDriftDetector()
    return detector.detect_workflow_run(trace)


@_register("dify_iteration_escape", "Detect Dify iteration escape", "beta")
def detect_dify_iteration_escape(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect iteration escape in Dify loop nodes."""
    from pisama_detectors.detection.dify import DifyIterationEscapeDetector

    detector = DifyIterationEscapeDetector()
    return detector.detect_workflow_run(trace)


@_register("dify_rag_poisoning", "Detect Dify RAG poisoning", "production")
def detect_dify_rag_poisoning(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect RAG knowledge base poisoning in Dify."""
    from pisama_detectors.detection.dify import DifyRagPoisoningDetector

    detector = DifyRagPoisoningDetector()
    return detector.detect_workflow_run(trace)


@_register("dify_tool_schema_mismatch", "Detect Dify tool schema mismatch", "beta")
def detect_dify_tool_schema_mismatch(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect tool schema mismatches in Dify."""
    from pisama_detectors.detection.dify import DifyToolSchemaMismatchDetector

    detector = DifyToolSchemaMismatchDetector()
    return detector.detect_workflow_run(trace)


@_register("dify_variable_leak", "Detect Dify variable leak", "production")
def detect_dify_variable_leak(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect variable leaks between Dify workflow branches."""
    from pisama_detectors.detection.dify import DifyVariableLeakDetector

    detector = DifyVariableLeakDetector()
    return detector.detect_workflow_run(trace)


@_register("dify_model_fallback", "Detect Dify model fallback issues", "beta")
def detect_dify_model_fallback(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect silent model fallback in Dify."""
    from pisama_detectors.detection.dify import DifyModelFallbackDetector

    detector = DifyModelFallbackDetector()
    return detector.detect_workflow_run(trace)


# ============================================================
# n8n-specific detectors
# ============================================================


@_register("n8n_cycle", "Detect n8n workflow cycles", "production")
def detect_n8n_cycle(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect cycles in n8n workflow execution."""
    from pisama_detectors.detection.n8n import N8NCycleDetector

    detector = N8NCycleDetector()
    return detector.detect_workflow(trace)


@_register("n8n_error", "Detect n8n execution errors", "production")
def detect_n8n_error(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect error patterns in n8n workflows."""
    from pisama_detectors.detection.n8n import N8NErrorDetector

    detector = N8NErrorDetector()
    return detector.detect_workflow(trace)


@_register("n8n_timeout", "Detect n8n timeout issues", "production")
def detect_n8n_timeout(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect timeout issues in n8n executions."""
    from pisama_detectors.detection.n8n import N8NTimeoutDetector

    detector = N8NTimeoutDetector()
    return detector.detect_workflow(trace)


@_register("n8n_complexity", "Detect n8n workflow complexity issues", "beta")
def detect_n8n_complexity(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect excessive complexity in n8n workflows."""
    from pisama_detectors.detection.n8n import N8NComplexityDetector

    detector = N8NComplexityDetector()
    return detector.detect_workflow(trace)


@_register("n8n_schema", "Detect n8n schema mismatches", "beta")
def detect_n8n_schema(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect schema mismatches between n8n nodes."""
    from pisama_detectors.detection.n8n import N8NSchemaDetector

    detector = N8NSchemaDetector()
    return detector.detect_workflow(trace)


@_register("n8n_resource", "Detect n8n resource issues", "beta")
def detect_n8n_resource(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect resource issues in n8n workflows."""
    from pisama_detectors.detection.n8n import N8NResourceDetector

    detector = N8NResourceDetector()
    return detector.detect_workflow(trace)


# ============================================================
# OpenClaw-specific detectors
# ============================================================


@_register("openclaw_session_loop", "Detect OpenClaw session loops", "beta")
def detect_openclaw_session_loop(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect session loops in OpenClaw."""
    from pisama_detectors.detection.openclaw import OpenClawSessionLoopDetector

    detector = OpenClawSessionLoopDetector()
    return detector.detect_session(trace)


@_register("openclaw_sandbox_escape", "Detect OpenClaw sandbox escape", "production")
def detect_openclaw_sandbox_escape(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect sandbox escape attempts in OpenClaw."""
    from pisama_detectors.detection.openclaw import OpenClawSandboxEscapeDetector

    detector = OpenClawSandboxEscapeDetector()
    return detector.detect_session(trace)


@_register("openclaw_tool_abuse", "Detect OpenClaw tool abuse", "production")
def detect_openclaw_tool_abuse(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect tool abuse patterns in OpenClaw."""
    from pisama_detectors.detection.openclaw import OpenClawToolAbuseDetector

    detector = OpenClawToolAbuseDetector()
    return detector.detect_session(trace)


@_register("openclaw_spawn_chain", "Detect OpenClaw spawn chain issues", "beta")
def detect_openclaw_spawn_chain(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect excessive spawn chains in OpenClaw."""
    from pisama_detectors.detection.openclaw import OpenClawSpawnChainDetector

    detector = OpenClawSpawnChainDetector()
    return detector.detect_session(trace)


@_register("openclaw_channel_mismatch", "Detect OpenClaw channel mismatch", "beta")
def detect_openclaw_channel_mismatch(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect channel mismatches in OpenClaw communication."""
    from pisama_detectors.detection.openclaw import OpenClawChannelMismatchDetector

    detector = OpenClawChannelMismatchDetector()
    return detector.detect_session(trace)


@_register("openclaw_elevated_risk", "Detect OpenClaw elevated risk actions", "production")
def detect_openclaw_elevated_risk(trace: Dict[str, Any]) -> TurnAwareDetectionResult:
    """Detect elevated risk actions in OpenClaw."""
    from pisama_detectors.detection.openclaw import OpenClawElevatedRiskDetector

    detector = OpenClawElevatedRiskDetector()
    return detector.detect_session(trace)


@_register("context_pressure", "Detect context-pressure-induced quality degradation", "beta")
def detect_context_pressure(
    states: List[Dict[str, Any]],
    context_limit: Optional[int] = None,
    task_complexity: Optional[str] = None,
) -> ContextPressureResult:
    """Detect when agent output quality degrades due to context window saturation.

    Signals: token trajectory, output length decline, premature wrap-up language,
    quality cliff, scope narrowing.

    Args:
        states: List of state dicts with token_count, state_delta, sequence_num.
        context_limit: Model context window size (auto-detected if None).
        task_complexity: Optional task description for scope analysis.
    """
    from pisama_detectors.detection.context_pressure import context_pressure_detector

    return context_pressure_detector.detect(
        states=states,
        context_limit=context_limit,
        task_complexity=task_complexity,
    )


# ============================================================


def run_all_detectors(trace_data: Dict[str, Any]) -> Dict[str, Any]:
    """Run all applicable detectors on trace data.

    Args:
        trace_data: Dict with keys matching detector input fields

    Returns:
        Dict mapping detector_name -> result
    """
    results = {}

    trace = trace_data.get("trace")
    framework_candidates = [
        trace_data.get("framework"),
        trace.get("framework") if isinstance(trace, Mapping) else None,
    ]
    framework = next(
        (
            candidate.strip().lower()
            for candidate in framework_candidates
            if isinstance(candidate, str)
            and candidate.strip().lower() in {"langgraph", "dify", "n8n", "openclaw"}
        ),
        None,
    )
    framework_prefix = f"{framework}_" if framework is not None else None

    for name, info in DETECTOR_REGISTRY.items():
        if framework_prefix and _is_other_framework_detector(name, framework_prefix):
            continue
        try:
            # Only run detectors whose inputs are available
            result = _try_run_detector(name, info.function, trace_data)
            if result is not None:
                results[name] = result
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def _is_other_framework_detector(name: str, selected_prefix: str) -> bool:
    """Return whether a registered framework detector belongs to another framework."""
    framework_prefixes = ("langgraph_", "dify_", "n8n_", "openclaw_")
    return name.startswith(framework_prefixes) and not name.startswith(selected_prefix)


def _try_run_detector(name: str, fn: Callable[..., Any], data: Dict[str, Any]) -> Any:
    """Try to run a detector if its required inputs are available."""
    import inspect

    sig = inspect.signature(fn)
    required_params = [
        p.name for p in sig.parameters.values() if p.default is inspect.Parameter.empty
    ]

    # Check if all required params are available in data
    if not all(p in data for p in required_params):
        return None

    # Build kwargs from data
    kwargs = {}
    for p in sig.parameters:
        if p in data:
            kwargs[p] = data[p]

    return fn(**kwargs)
