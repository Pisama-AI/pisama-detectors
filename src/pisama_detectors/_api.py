"""Public API for Pisama detectors.

Provides simplified functions that wrap the core detection algorithms.
Each function takes plain Python dicts/lists and returns a result dataclass.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class DetectorInfo:
    """Registry entry for a detector."""
    name: str
    description: str
    function: Callable
    tier: str  # production, beta, experimental


# Will be populated below
DETECTOR_REGISTRY: Dict[str, DetectorInfo] = {}


def _register(name: str, description: str, tier: str = "production"):
    """Decorator to register a detector function."""
    def decorator(fn):
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
) -> Any:
    """Detect infinite loops in agent state sequences.

    Args:
        states: List of agent state dicts (each representing a step)
        window_size: Number of states to compare for pattern detection
        similarity_threshold: Similarity threshold for semantic loop detection

    Returns:
        LoopDetectionResult with detected, confidence, loop_type, etc.
    """
    from app.detection.loop import MultiLevelLoopDetector, StateSnapshot

    snapshots = []
    for i, state in enumerate(states):
        snapshots.append(StateSnapshot(
            step=i,
            state=state,
            timestamp=state.get("timestamp"),
        ))

    detector = MultiLevelLoopDetector()
    return detector.detect(snapshots)


@_register("corruption", "Detect state corruption and invalid transitions", "production")
def detect_corruption(
    prev_state: Dict[str, Any],
    current_state: Dict[str, Any],
) -> Any:
    """Detect state corruption between consecutive states.

    Args:
        prev_state: Previous agent state dict
        current_state: Current agent state dict

    Returns:
        CorruptionResult with detected, confidence, issues
    """
    from app.detection.corruption import SemanticCorruptionDetector, StateSnapshot

    detector = SemanticCorruptionDetector()
    prev = StateSnapshot(data=prev_state)
    curr = StateSnapshot(data=current_state)
    return detector.detect_corruption_with_confidence(prev_state=prev, current_state=curr)


@_register("injection", "Detect prompt injection and jailbreak attempts", "production")
def detect_injection(
    text: str,
) -> Any:
    """Detect prompt injection or jailbreak attempts in text.

    Args:
        text: Input text to check for injection

    Returns:
        InjectionResult with detected, confidence, attack_type
    """
    from app.detection.injection import InjectionDetector

    detector = InjectionDetector()
    return detector.detect(text=text)


@_register("hallucination", "Detect factual inaccuracies and fabrications", "production")
def detect_hallucination(
    output: str,
    sources: Optional[List[str]] = None,
) -> Any:
    """Detect hallucinations in agent output.

    Args:
        output: Agent output text
        sources: Optional list of source documents for grounding

    Returns:
        HallucinationResult with detected, confidence, issues
    """
    from app.detection.hallucination import HallucinationDetector

    detector = HallucinationDetector()
    return detector.detect_hallucination(output=output, context="\n".join(sources) if sources else None)


@_register("persona_drift", "Detect persona drift and role confusion", "production")
def detect_persona_drift(
    agent_id: str,
    persona_description: str,
    output: str,
    allowed_actions: Optional[List[str]] = None,
) -> Any:
    """Detect persona drift in agent behavior.

    Args:
        agent_id: Agent identifier
        persona_description: Expected persona/role description
        output: Agent output to check
        allowed_actions: Optional list of allowed actions

    Returns:
        PersonaConsistencyResult with score, deviations
    """
    from app.detection.persona import PersonaConsistencyScorer, Agent

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
) -> Any:
    """Detect coordination failures between agents.

    Args:
        messages: List of inter-agent messages with sender, receiver, content
        agent_ids: Optional list of agent IDs in the system

    Returns:
        CoordinationAnalysisResult with issues, severity
    """
    from app.detection.coordination import CoordinationAnalyzer, Message

    parsed_messages = []
    for msg in messages:
        parsed_messages.append(Message(
            sender=msg.get("sender", "unknown"),
            receiver=msg.get("receiver", "unknown"),
            content=msg.get("content", ""),
            timestamp=msg.get("timestamp"),
        ))

    analyzer = CoordinationAnalyzer()
    return analyzer.analyze(
        messages=parsed_messages,
        agent_ids=agent_ids or [],
    )


@_register("overflow", "Detect context window exhaustion", "production")
def detect_overflow(
    context: str,
    output: str,
    model: str = "claude-sonnet-4-6",
) -> Any:
    """Detect context overflow issues.

    Args:
        context: Full context/conversation
        output: Agent output
        model: LLM model name (for token limit lookup)

    Returns:
        OverflowResult with detected, severity, token counts
    """
    from app.detection.overflow import ContextOverflowDetector

    detector = ContextOverflowDetector()
    # Estimate token count from context length (rough: 1 token ≈ 4 chars)
    current_tokens = len(context) // 4
    return detector.detect_overflow(current_tokens=current_tokens, model=model)


@_register("derailment", "Detect task focus deviation", "beta")
def detect_derailment(
    task: str,
    output: str,
) -> Any:
    """Detect task derailment.

    Args:
        task: Original task description
        output: Agent output

    Returns:
        DerailmentResult with detected, severity, explanation
    """
    from app.detection.derailment import TaskDerailmentDetector

    detector = TaskDerailmentDetector()
    return detector.detect(task=task, output=output)


@_register("context_neglect", "Detect context neglect in responses", "production")
def detect_context_neglect(
    context: str,
    output: str,
) -> Any:
    """Detect context neglect.

    Args:
        context: Provided context
        output: Agent output

    Returns:
        ContextNeglectResult with detected, severity
    """
    from app.detection.context import ContextNeglectDetector

    detector = ContextNeglectDetector()
    return detector.detect(context=context, output=output)


@_register("communication", "Detect inter-agent communication breakdowns", "beta")
def detect_communication(
    sender_message: str,
    receiver_response: str,
) -> Any:
    """Detect communication breakdown between agents.

    Args:
        sender_message: Message from sender agent
        receiver_response: Response from receiver agent

    Returns:
        CommunicationBreakdownResult with detected, breakdown_type
    """
    from app.detection.communication import CommunicationBreakdownDetector

    detector = CommunicationBreakdownDetector()
    return detector.detect(
        sender_message=sender_message,
        receiver_response=receiver_response,
    )


@_register("specification", "Detect output vs spec mismatch", "production")
def detect_specification(
    user_intent: str,
    task_specification: str,
) -> Any:
    """Detect specification mismatch.

    Args:
        user_intent: What the user asked for
        task_specification: What was specified/implemented

    Returns:
        SpecificationMismatchResult with detected, mismatch_type
    """
    from app.detection.specification import SpecificationMismatchDetector

    detector = SpecificationMismatchDetector()
    return detector.detect(
        user_intent=user_intent,
        task_specification=task_specification,
    )


@_register("decomposition", "Detect task breakdown failures", "production")
def detect_decomposition(
    task_description: str,
    decomposition: List[Dict[str, Any]],
) -> Any:
    """Detect task decomposition failures.

    Args:
        task_description: Original task description
        decomposition: List of subtask dicts

    Returns:
        DecompositionResult with detected, issues
    """
    from app.detection.decomposition import TaskDecompositionDetector

    detector = TaskDecompositionDetector()
    return detector.detect(
        task_description=task_description,
        decomposition=decomposition,
    )


@_register("workflow", "Detect workflow execution issues", "beta")
def detect_workflow(
    workflow_definition: Dict[str, Any],
    execution_result: Dict[str, Any],
) -> Any:
    """Detect workflow design and execution issues.

    Args:
        workflow_definition: Workflow definition/spec
        execution_result: Execution trace/result

    Returns:
        WorkflowAnalysisResult with issues
    """
    from app.detection.workflow import FlawedWorkflowDetector

    detector = FlawedWorkflowDetector()
    return detector.detect(
        workflow_definition=workflow_definition,
        execution_result=execution_result,
    )


@_register("withholding", "Detect information withholding", "beta")
def detect_withholding(
    agent_output: str,
    internal_state: Dict[str, Any],
) -> Any:
    """Detect information withholding.

    Args:
        agent_output: What the agent said
        internal_state: Agent's internal state

    Returns:
        WithholdingResult with detected, issues
    """
    from app.detection.withholding import InformationWithholdingDetector

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
) -> Any:
    """Detect completion misjudgment.

    Args:
        task: Original task
        subtasks: List of subtasks
        agent_output: Agent's output
        success_criteria: Optional success criteria

    Returns:
        CompletionResult with detected, issues
    """
    from app.detection.completion import CompletionMisjudgmentDetector

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
) -> Any:
    """Detect convergence failures in optimization metrics.

    Args:
        metrics: List of metric values over time
        direction: 'minimize' or 'maximize'
        window_size: Window size for analysis

    Returns:
        ConvergenceResult with detected, failure_type, severity
    """
    from app.detection.convergence import ConvergenceDetector

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
) -> Any:
    """Calculate LLM cost.

    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        CostResult with cost_usd, tokens
    """
    from app.detection.cost import CostCalculator

    calculator = CostCalculator()
    return calculator.calculate(
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


# ============================================================
# Enterprise detectors
# ============================================================

@_register("grounding", "Detect source misattribution and ungrounded claims", "production")
def detect_grounding(
    agent_output: str,
    source_documents: List[str],
    task: Optional[str] = None,
) -> Any:
    """Detect grounding failures (claims not supported by sources)."""
    from app.detection_enterprise.grounding import GroundingDetector
    detector = GroundingDetector()
    return detector.detect(agent_output=agent_output, source_documents=source_documents, task=task)


@_register("retrieval_quality", "Detect retrieval quality degradation", "beta")
def detect_retrieval_quality(
    query: str,
    retrieved_documents: List[str],
    agent_output: str,
) -> Any:
    """Detect poor retrieval quality in RAG systems."""
    from app.detection_enterprise.retrieval_quality import RetrievalQualityDetector
    detector = RetrievalQualityDetector()
    return detector.detect(query=query, retrieved_documents=retrieved_documents, agent_output=agent_output)


@_register("quality_gate", "Detect quality gate bypass", "enterprise")
def detect_quality_gate(
    task: str,
    agent_output: str,
    required_gates: Optional[List[str]] = None,
) -> Any:
    """Detect quality gate bypass in agent workflows."""
    from app.detection_enterprise.quality_gate import QualityGateDetector
    detector = QualityGateDetector()
    return detector.detect(task=task, agent_output=agent_output, required_gates=required_gates)


@_register("tool_provision", "Detect tool provision failures", "enterprise")
def detect_tool_provision(
    task: str,
    agent_output: str,
    available_tools: Optional[List[str]] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Detect tool provision failures (wrong tools, missing tools)."""
    from app.detection_enterprise.tool_provision import ToolProvisionDetector
    detector = ToolProvisionDetector()
    return detector.detect(task=task, agent_output=agent_output, available_tools=available_tools, tool_calls=tool_calls)


# ============================================================
# LangGraph-specific detectors
# ============================================================

@_register("langgraph_recursion", "Detect LangGraph recursion limit issues", "production")
def detect_langgraph_recursion(trace: Dict[str, Any]) -> Any:
    """Detect recursion issues in LangGraph executions."""
    from app.detection_enterprise.orchestrator import LangGraphRecursionDetector
    detector = LangGraphRecursionDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_state_corruption", "Detect LangGraph state corruption", "production")
def detect_langgraph_state_corruption(trace: Dict[str, Any]) -> Any:
    """Detect state corruption in LangGraph graph state."""
    from app.detection_enterprise.orchestrator import LangGraphStateCorruptionDetector
    detector = LangGraphStateCorruptionDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_edge_misroute", "Detect LangGraph edge misrouting", "beta")
def detect_langgraph_edge_misroute(trace: Dict[str, Any]) -> Any:
    """Detect edge misrouting in LangGraph conditional edges."""
    from app.detection_enterprise.orchestrator import LangGraphEdgeMisrouteDetector
    detector = LangGraphEdgeMisrouteDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_checkpoint_corruption", "Detect LangGraph checkpoint corruption", "beta")
def detect_langgraph_checkpoint_corruption(trace: Dict[str, Any]) -> Any:
    """Detect checkpoint corruption in LangGraph persistence."""
    from app.detection_enterprise.orchestrator import LangGraphCheckpointCorruptionDetector
    detector = LangGraphCheckpointCorruptionDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_parallel_sync", "Detect LangGraph parallel branch sync failures", "beta")
def detect_langgraph_parallel_sync(trace: Dict[str, Any]) -> Any:
    """Detect parallel branch synchronization issues in LangGraph."""
    from app.detection_enterprise.orchestrator import LangGraphParallelSyncDetector
    detector = LangGraphParallelSyncDetector()
    return detector.detect_graph_execution(trace)


@_register("langgraph_tool_failure", "Detect LangGraph tool execution failures", "production")
def detect_langgraph_tool_failure(trace: Dict[str, Any]) -> Any:
    """Detect tool execution failures in LangGraph."""
    from app.detection_enterprise.orchestrator import LangGraphToolFailureDetector
    detector = LangGraphToolFailureDetector()
    return detector.detect_graph_execution(trace)


# ============================================================
# Dify-specific detectors
# ============================================================

@_register("dify_classifier_drift", "Detect Dify classifier drift", "beta")
def detect_dify_classifier_drift(trace: Dict[str, Any]) -> Any:
    """Detect classifier drift in Dify intent routing."""
    from app.detection_enterprise.orchestrator import DifyClassifierDriftDetector
    detector = DifyClassifierDriftDetector()
    return detector.detect(trace)


@_register("dify_iteration_escape", "Detect Dify iteration escape", "beta")
def detect_dify_iteration_escape(trace: Dict[str, Any]) -> Any:
    """Detect iteration escape in Dify loop nodes."""
    from app.detection_enterprise.orchestrator import DifyIterationEscapeDetector
    detector = DifyIterationEscapeDetector()
    return detector.detect(trace)


@_register("dify_rag_poisoning", "Detect Dify RAG poisoning", "production")
def detect_dify_rag_poisoning(trace: Dict[str, Any]) -> Any:
    """Detect RAG knowledge base poisoning in Dify."""
    from app.detection_enterprise.orchestrator import DifyRagPoisoningDetector
    detector = DifyRagPoisoningDetector()
    return detector.detect(trace)


@_register("dify_tool_schema_mismatch", "Detect Dify tool schema mismatch", "beta")
def detect_dify_tool_schema_mismatch(trace: Dict[str, Any]) -> Any:
    """Detect tool schema mismatches in Dify."""
    from app.detection_enterprise.orchestrator import DifyToolSchemaMismatchDetector
    detector = DifyToolSchemaMismatchDetector()
    return detector.detect(trace)


@_register("dify_variable_leak", "Detect Dify variable leak", "production")
def detect_dify_variable_leak(trace: Dict[str, Any]) -> Any:
    """Detect variable leaks between Dify workflow branches."""
    from app.detection_enterprise.orchestrator import DifyVariableLeakDetector
    detector = DifyVariableLeakDetector()
    return detector.detect(trace)


@_register("dify_model_fallback", "Detect Dify model fallback issues", "beta")
def detect_dify_model_fallback(trace: Dict[str, Any]) -> Any:
    """Detect silent model fallback in Dify."""
    from app.detection_enterprise.orchestrator import DifyModelFallbackDetector
    detector = DifyModelFallbackDetector()
    return detector.detect(trace)


# ============================================================
# n8n-specific detectors
# ============================================================

@_register("n8n_cycle", "Detect n8n workflow cycles", "production")
def detect_n8n_cycle(trace: Dict[str, Any]) -> Any:
    """Detect cycles in n8n workflow execution."""
    from app.detection_enterprise.orchestrator import N8NCycleDetector
    detector = N8NCycleDetector()
    return detector.detect(trace)


@_register("n8n_error", "Detect n8n execution errors", "production")
def detect_n8n_error(trace: Dict[str, Any]) -> Any:
    """Detect error patterns in n8n workflows."""
    from app.detection_enterprise.orchestrator import N8NErrorDetector
    detector = N8NErrorDetector()
    return detector.detect(trace)


@_register("n8n_timeout", "Detect n8n timeout issues", "production")
def detect_n8n_timeout(trace: Dict[str, Any]) -> Any:
    """Detect timeout issues in n8n executions."""
    from app.detection_enterprise.orchestrator import N8NTimeoutDetector
    detector = N8NTimeoutDetector()
    return detector.detect(trace)


@_register("n8n_complexity", "Detect n8n workflow complexity issues", "beta")
def detect_n8n_complexity(trace: Dict[str, Any]) -> Any:
    """Detect excessive complexity in n8n workflows."""
    from app.detection_enterprise.orchestrator import N8NComplexityDetector
    detector = N8NComplexityDetector()
    return detector.detect(trace)


@_register("n8n_schema", "Detect n8n schema mismatches", "beta")
def detect_n8n_schema(trace: Dict[str, Any]) -> Any:
    """Detect schema mismatches between n8n nodes."""
    from app.detection_enterprise.orchestrator import N8NSchemaDetector
    detector = N8NSchemaDetector()
    return detector.detect(trace)


@_register("n8n_resource", "Detect n8n resource issues", "beta")
def detect_n8n_resource(trace: Dict[str, Any]) -> Any:
    """Detect resource issues in n8n workflows."""
    from app.detection_enterprise.orchestrator import N8NResourceDetector
    detector = N8NResourceDetector()
    return detector.detect(trace)


# ============================================================
# OpenClaw-specific detectors
# ============================================================

@_register("openclaw_session_loop", "Detect OpenClaw session loops", "beta")
def detect_openclaw_session_loop(trace: Dict[str, Any]) -> Any:
    """Detect session loops in OpenClaw."""
    from app.detection_enterprise.orchestrator import OpenClawSessionLoopDetector
    detector = OpenClawSessionLoopDetector()
    return detector.detect(trace)


@_register("openclaw_sandbox_escape", "Detect OpenClaw sandbox escape", "production")
def detect_openclaw_sandbox_escape(trace: Dict[str, Any]) -> Any:
    """Detect sandbox escape attempts in OpenClaw."""
    from app.detection_enterprise.orchestrator import OpenClawSandboxEscapeDetector
    detector = OpenClawSandboxEscapeDetector()
    return detector.detect(trace)


@_register("openclaw_tool_abuse", "Detect OpenClaw tool abuse", "production")
def detect_openclaw_tool_abuse(trace: Dict[str, Any]) -> Any:
    """Detect tool abuse patterns in OpenClaw."""
    from app.detection_enterprise.orchestrator import OpenClawToolAbuseDetector
    detector = OpenClawToolAbuseDetector()
    return detector.detect(trace)


@_register("openclaw_spawn_chain", "Detect OpenClaw spawn chain issues", "beta")
def detect_openclaw_spawn_chain(trace: Dict[str, Any]) -> Any:
    """Detect excessive spawn chains in OpenClaw."""
    from app.detection_enterprise.orchestrator import OpenClawSpawnChainDetector
    detector = OpenClawSpawnChainDetector()
    return detector.detect(trace)


@_register("openclaw_channel_mismatch", "Detect OpenClaw channel mismatch", "beta")
def detect_openclaw_channel_mismatch(trace: Dict[str, Any]) -> Any:
    """Detect channel mismatches in OpenClaw communication."""
    from app.detection_enterprise.orchestrator import OpenClawChannelMismatchDetector
    detector = OpenClawChannelMismatchDetector()
    return detector.detect(trace)


@_register("openclaw_elevated_risk", "Detect OpenClaw elevated risk actions", "production")
def detect_openclaw_elevated_risk(trace: Dict[str, Any]) -> Any:
    """Detect elevated risk actions in OpenClaw."""
    from app.detection_enterprise.orchestrator import OpenClawElevatedRiskDetector
    detector = OpenClawElevatedRiskDetector()
    return detector.detect(trace)


@_register("context_pressure", "Detect context-pressure-induced quality degradation", "beta")
def detect_context_pressure(
    states: List[Dict[str, Any]],
    context_limit: Optional[int] = None,
    task_complexity: Optional[str] = None,
) -> Any:
    """Detect when agent output quality degrades due to context window saturation.

    Signals: token trajectory, output length decline, premature wrap-up language,
    quality cliff, scope narrowing.

    Args:
        states: List of state dicts with token_count, state_delta, sequence_num.
        context_limit: Model context window size (auto-detected if None).
        task_complexity: Optional task description for scope analysis.
    """
    from app.detection.context_pressure import context_pressure_detector
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

    for name, info in DETECTOR_REGISTRY.items():
        try:
            # Only run detectors whose inputs are available
            result = _try_run_detector(name, info.function, trace_data)
            if result is not None:
                results[name] = result
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def _try_run_detector(name: str, fn: Callable, data: Dict[str, Any]) -> Any:
    """Try to run a detector if its required inputs are available."""
    import inspect
    sig = inspect.signature(fn)
    required_params = [
        p.name for p in sig.parameters.values()
        if p.default is inspect.Parameter.empty
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
