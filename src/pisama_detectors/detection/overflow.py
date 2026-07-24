"""Context window overflow detection."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from .cost import cost_calculator


class OverflowSeverity(str, Enum):
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERFLOW = "overflow"


@dataclass
class OverflowResult:
    severity: OverflowSeverity
    current_tokens: int
    context_window: int
    usage_percent: float
    remaining_tokens: int
    estimated_overflow_in: Optional[int]
    warnings: List[str]
    details: Dict[str, Any] = field(default_factory=dict)
    detected: bool = False
    confidence: float = 0.0
    raw_score: Optional[float] = None
    calibration_info: Optional[Dict[str, Any]] = None


@dataclass
class TokenBreakdown:
    system_tokens: int
    message_tokens: int
    tool_tokens: int
    total_tokens: int
    per_turn_average: float


class ContextOverflowDetector:
    # Sprint 9 PP: Models with <= this context window are "small" and use a
    # lower warning threshold — conversations approach their limit faster.
    SMALL_CONTEXT_WINDOW = 16385

    def __init__(
        self,
        warning_threshold: float = 0.65,
        critical_threshold: float = 0.85,
        overflow_threshold: float = 0.95,
        confidence_scaling: float = 1.0,
        small_model_warning_threshold: float = 0.18,
    ):
        # Sprint 9 PP: warning_threshold raised 0.50 -> 0.65 (precision win
        # on 128k models at 55-60% usage) + separate small-model threshold
        # at 0.25 (gpt-3.5-turbo at ~30% correctly flagged).
        self._tokenizers: Dict[str, tiktoken.Encoding] = {}
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.overflow_threshold = overflow_threshold
        self.confidence_scaling = confidence_scaling
        self.small_model_warning_threshold = small_model_warning_threshold

    def _get_tokenizer(self, model: str) -> tiktoken.Encoding:
        try:
            if "gpt-4" in model or "gpt-3.5" in model:
                return tiktoken.encoding_for_model(model)
            elif "claude" in model:
                return tiktoken.get_encoding("cl100k_base")
            else:
                return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str, model: str = "gpt-4") -> int:
        if not text:
            return 0
        tokenizer = self._get_tokenizer(model)
        return len(tokenizer.encode(text))

    def count_messages_tokens(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4",
    ) -> TokenBreakdown:
        tokenizer = self._get_tokenizer(model)

        system_tokens = 0
        message_tokens = 0
        tool_tokens = 0

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                tokens = len(tokenizer.encode(content))
            elif isinstance(content, list):
                tokens = sum(
                    len(tokenizer.encode(str(c.get("text", ""))))
                    for c in content
                    if isinstance(c, dict)
                )
            else:
                tokens = 0

            tokens += 4

            if role == "system":
                system_tokens += tokens
            elif role == "tool" or msg.get("tool_calls"):
                tool_tokens += tokens
                if msg.get("tool_calls"):
                    for tc in msg.get("tool_calls", []):
                        tool_tokens += len(
                            tokenizer.encode(str(tc.get("function", {}).get("arguments", "")))
                        )
            else:
                message_tokens += tokens

        total = system_tokens + message_tokens + tool_tokens
        per_turn = message_tokens / max(
            1, len([m for m in messages if m.get("role") in ["user", "assistant"]])
        )

        return TokenBreakdown(
            system_tokens=system_tokens,
            message_tokens=message_tokens,
            tool_tokens=tool_tokens,
            total_tokens=total,
            per_turn_average=per_turn,
        )

    def detect_overflow(
        self,
        current_tokens: int,
        model: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        expected_output_tokens: int = 4096,
    ) -> OverflowResult:
        context_window = cost_calculator.get_context_window(model)

        # Sprint 9 PP: scale reserved output by context size so the effective
        # limit never drops below 40% of the window. A fixed 4096-token reserve
        # used to crush 8k/16k-window models (usage_percent > 1.0 on normal
        # traffic).
        scaled_reserve = min(expected_output_tokens, max(512, int(context_window * 0.05)))
        effective_limit = max(int(context_window * 0.4), context_window - scaled_reserve)

        usage_percent = current_tokens / effective_limit
        remaining = effective_limit - current_tokens

        # Sprint 9 PP: small-context models hit their limit faster. A
        # gpt-3.5-turbo conversation at 30% usage is legitimately approaching
        # overflow, whereas a gpt-4o conversation at 30% has 90k+ headroom.
        warning_threshold = (
            self.small_model_warning_threshold
            if context_window <= self.SMALL_CONTEXT_WINDOW
            else self.warning_threshold
        )

        warnings = []
        details = {}

        if usage_percent >= self.overflow_threshold:
            severity = OverflowSeverity.OVERFLOW
            warnings.append(f"Context window overflow imminent: {usage_percent:.1%} used")
        elif usage_percent >= self.critical_threshold:
            severity = OverflowSeverity.CRITICAL
            warnings.append(f"Context window critically full: {usage_percent:.1%} used")
        elif usage_percent >= warning_threshold:
            severity = OverflowSeverity.WARNING
            warnings.append(f"Context window filling up: {usage_percent:.1%} used")
        else:
            severity = OverflowSeverity.SAFE

        estimated_overflow_in = None
        if messages:
            breakdown = self.count_messages_tokens(messages, model)
            details["token_breakdown"] = {
                "system": breakdown.system_tokens,
                "messages": breakdown.message_tokens,
                "tools": breakdown.tool_tokens,
                "per_turn_avg": breakdown.per_turn_average,
            }

            if breakdown.per_turn_average > 0:
                estimated_overflow_in = int(remaining / breakdown.per_turn_average)
                details["estimated_turns_remaining"] = estimated_overflow_in

                if estimated_overflow_in < 5:
                    warnings.append(f"Estimated overflow in {estimated_overflow_in} turns")
                    if severity == OverflowSeverity.SAFE:
                        severity = OverflowSeverity.WARNING

            if breakdown.system_tokens > context_window * 0.3:
                warnings.append(
                    f"System prompt uses {breakdown.system_tokens / context_window:.1%} of context"
                )

            if breakdown.tool_tokens > context_window * 0.2:
                warnings.append(
                    f"Tool results using significant context: {breakdown.tool_tokens} tokens"
                )

        details["context_window"] = context_window
        details["effective_limit"] = effective_limit
        details["expected_output"] = expected_output_tokens
        details["model"] = model

        detected = severity in (
            OverflowSeverity.WARNING,
            OverflowSeverity.CRITICAL,
            OverflowSeverity.OVERFLOW,
        )
        raw_score = usage_percent
        confidence, calibration_info = self._calibrate_confidence(
            usage_percent=usage_percent,
            severity=severity,
            warning_count=len(warnings),
            estimated_overflow_in=estimated_overflow_in,
        )

        return OverflowResult(
            severity=severity,
            current_tokens=current_tokens,
            context_window=context_window,
            usage_percent=round(usage_percent, 4),
            remaining_tokens=max(0, remaining),
            estimated_overflow_in=estimated_overflow_in,
            warnings=warnings,
            details=details,
            detected=detected,
            confidence=confidence,
            raw_score=raw_score,
            calibration_info=calibration_info,
        )

    def _calibrate_confidence(
        self,
        usage_percent: float,
        severity: OverflowSeverity,
        warning_count: int,
        estimated_overflow_in: Optional[int],
    ) -> Tuple[float, Dict[str, Any]]:
        """Calibrate confidence based on usage metrics and severity."""
        severity_weight = {
            OverflowSeverity.SAFE: 0.0,
            OverflowSeverity.WARNING: 0.6,
            OverflowSeverity.CRITICAL: 0.8,
            OverflowSeverity.OVERFLOW: 0.95,
        }.get(severity, 0.5)

        usage_factor = min(1.0, usage_percent)

        warning_factor = min(0.15, warning_count * 0.05)

        overflow_urgency = 0.0
        if estimated_overflow_in is not None:
            if estimated_overflow_in <= 2:
                overflow_urgency = 0.2
            elif estimated_overflow_in <= 5:
                overflow_urgency = 0.1
            elif estimated_overflow_in <= 10:
                overflow_urgency = 0.05

        base_confidence = (
            severity_weight * 0.40 + usage_factor * 0.30 + warning_factor + overflow_urgency
        )

        calibrated = min(0.99, base_confidence * self.confidence_scaling)

        calibration_info = {
            "usage_percent": round(usage_percent, 4),
            "severity_weight": severity_weight,
            "warning_count": warning_count,
            "warning_factor": round(warning_factor, 4),
            "overflow_urgency": overflow_urgency,
            "estimated_overflow_in": estimated_overflow_in,
            "confidence_scaling": self.confidence_scaling,
        }

        return round(calibrated, 4), calibration_info

    def detect_memory_leak(
        self,
        token_history: List[int],
        model: str,
    ) -> Optional[Dict[str, Any]]:
        if len(token_history) < 5:
            return None

        context_window = cost_calculator.get_context_window(model)

        growth_rates = []
        for i in range(1, len(token_history)):
            if token_history[i - 1] > 0:
                rate = (token_history[i] - token_history[i - 1]) / token_history[i - 1]
                growth_rates.append(rate)

        if not growth_rates:
            return None

        avg_growth = sum(growth_rates) / len(growth_rates)

        expected_shrink = any(r < -0.1 for r in growth_rates)

        if avg_growth > 0.05 and not expected_shrink:
            current = token_history[-1]
            turns_to_overflow = 0
            projected = current

            while projected < context_window and turns_to_overflow < 100:
                projected = projected * (1 + avg_growth)
                turns_to_overflow += 1

            return {
                "leak_detected": True,
                "avg_growth_rate": avg_growth,
                "current_tokens": current,
                "projected_overflow_turns": turns_to_overflow,
                "recommendation": "Implement conversation summarization or context pruning",
            }

        return None

    def suggest_remediation(self, result: OverflowResult) -> List[str]:
        suggestions = []

        if result.severity == OverflowSeverity.OVERFLOW:
            suggestions.append("IMMEDIATE: Truncate or summarize conversation history")
            suggestions.append("Remove older messages, keeping only recent context")

        if result.severity in [OverflowSeverity.CRITICAL, OverflowSeverity.OVERFLOW]:
            suggestions.append("Summarize tool results instead of including full output")
            suggestions.append("Consider using a model with larger context window")

        breakdown = result.details.get("token_breakdown", {})

        if breakdown.get("system", 0) > result.context_window * 0.2:
            suggestions.append(
                "Reduce system prompt size - consider dynamic loading of instructions"
            )

        if breakdown.get("tools", 0) > result.context_window * 0.15:
            suggestions.append("Compress tool results - return summaries instead of full data")

        if result.usage_percent > 0.5:
            suggestions.append("Implement sliding window for message history")
            suggestions.append("Add periodic conversation summarization")

        return suggestions


overflow_detector = ContextOverflowDetector()
