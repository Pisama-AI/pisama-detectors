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
