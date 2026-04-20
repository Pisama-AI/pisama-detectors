"""LLM cost tracking and pricing database."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from datetime import datetime


@dataclass
class ModelPricing:
    input_per_1m: float
    output_per_1m: float
    context_window: int
    provider: str


LLM_PRICING_2025: Dict[str, ModelPricing] = {
    "gpt-4o": ModelPricing(2.50, 10.00, 128000, "openai"),
    "gpt-4o-mini": ModelPricing(0.15, 0.60, 128000, "openai"),
    "gpt-4-turbo": ModelPricing(10.00, 30.00, 128000, "openai"),
    "gpt-4": ModelPricing(30.00, 60.00, 8192, "openai"),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50, 16385, "openai"),
    "o1": ModelPricing(15.00, 60.00, 200000, "openai"),
    "o1-mini": ModelPricing(3.00, 12.00, 128000, "openai"),
    "o1-pro": ModelPricing(150.00, 600.00, 200000, "openai"),
    # Claude 3.5 models
    "claude-3-5-sonnet-20241022": ModelPricing(3.00, 15.00, 200000, "anthropic"),
    "claude-3-5-haiku-20241022": ModelPricing(0.80, 4.00, 200000, "anthropic"),
    "claude-3-opus-20240229": ModelPricing(15.00, 75.00, 200000, "anthropic"),
    # Claude 4 models
    "claude-sonnet-4-20250514": ModelPricing(3.00, 15.00, 200000, "anthropic"),
    "claude-opus-4-20250514": ModelPricing(15.00, 75.00, 200000, "anthropic"),
    # Claude 4.5 models (latest)
    "claude-opus-4-5-20251101": ModelPricing(15.00, 75.00, 200000, "anthropic"),
    "claude-sonnet-4-5-20251101": ModelPricing(3.00, 15.00, 200000, "anthropic"),
    "gemini-1.5-pro": ModelPricing(1.25, 5.00, 2000000, "google"),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30, 1000000, "google"),
    "gemini-2.0-flash": ModelPricing(0.10, 0.40, 1000000, "google"),
    "gemini-2.5-pro": ModelPricing(1.25, 10.00, 1000000, "google"),
    "mistral-large": ModelPricing(2.00, 6.00, 128000, "mistral"),
    "mistral-small": ModelPricing(0.20, 0.60, 32000, "mistral"),
    "codestral": ModelPricing(0.30, 0.90, 32000, "mistral"),
    "llama-3.1-405b": ModelPricing(3.00, 3.00, 128000, "meta"),
    "llama-3.1-70b": ModelPricing(0.70, 0.80, 128000, "meta"),
    "llama-3.1-8b": ModelPricing(0.10, 0.10, 128000, "meta"),
    "llama-3.3-70b": ModelPricing(0.60, 0.60, 128000, "meta"),
    "deepseek-v3": ModelPricing(0.27, 1.10, 64000, "deepseek"),
    "deepseek-r1": ModelPricing(0.55, 2.19, 64000, "deepseek"),
    "qwen-2.5-72b": ModelPricing(0.40, 0.40, 128000, "alibaba"),
    "command-r-plus": ModelPricing(2.50, 10.00, 128000, "cohere"),
    "command-r": ModelPricing(0.15, 0.60, 128000, "cohere"),
}

MODEL_ALIASES: Dict[str, str] = {
    "gpt-4o-2024-11-20": "gpt-4o",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-2024-05-13": "gpt-4o",
    "gpt-4-turbo-2024-04-09": "gpt-4-turbo",
    "gpt-4-0125-preview": "gpt-4-turbo",
    "gpt-4-1106-preview": "gpt-4-turbo",
    "gpt-3.5-turbo-0125": "gpt-3.5-turbo",
    "claude-3-5-sonnet-latest": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-latest": "claude-3-5-haiku-20241022",
    "claude-3-opus-latest": "claude-3-opus-20240229",
    "claude-sonnet-4-latest": "claude-sonnet-4-20250514",
    "claude-opus-4-latest": "claude-opus-4-20250514",
    # Claude 4.5 aliases
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-opus-4-5-latest": "claude-opus-4-5-20251101",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20251101",
    "claude-sonnet-4-5-latest": "claude-sonnet-4-5-20251101",
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",
}


@dataclass
class CostResult:
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    total_cost_cents: int
    model: str
    provider: str


class CostCalculator:
    def __init__(self):
        self.pricing = LLM_PRICING_2025
        self.aliases = MODEL_ALIASES
        self._custom_pricing: Dict[str, ModelPricing] = {}
    
    def add_custom_pricing(self, model: str, pricing: ModelPricing) -> None:
        self._custom_pricing[model] = pricing
    
    def _resolve_model(self, model: str) -> str:
        model_lower = model.lower()
        if model_lower in self.aliases:
            return self.aliases[model_lower]
        if model_lower in self.pricing:
            return model_lower
        for key in self.pricing:
            if key in model_lower or model_lower in key:
                return key
        return model_lower
    
    def get_pricing(self, model: str) -> Optional[ModelPricing]:
        resolved = self._resolve_model(model)
        if resolved in self._custom_pricing:
            return self._custom_pricing[resolved]
        return self.pricing.get(resolved)
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostResult:
        pricing = self.get_pricing(model)
        
        if pricing is None:
            return CostResult(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
                input_cost_usd=0.0,
                output_cost_usd=0.0,
                total_cost_usd=0.0,
                total_cost_cents=0,
                model=model,
                provider="unknown",
            )
        
        input_cost = (input_tokens / 1_000_000) * pricing.input_per_1m
        output_cost = (output_tokens / 1_000_000) * pricing.output_per_1m
        total_cost = input_cost + output_cost
        
        return CostResult(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            input_cost_usd=round(input_cost, 6),
            output_cost_usd=round(output_cost, 6),
            total_cost_usd=round(total_cost, 6),
            total_cost_cents=round(total_cost * 100, 2),
            model=self._resolve_model(model),
            provider=pricing.provider,
        )
    
    def calculate_trace_cost(
        self,
        spans: list,
    ) -> CostResult:
        total_input = 0
        total_output = 0
        total_input_cost = 0.0
        total_output_cost = 0.0
        providers = set()
        models = set()
        
        for span in spans:
            model = span.get("model", "unknown")
            input_tokens = span.get("input_tokens", 0) or span.get("prompt_tokens", 0)
            output_tokens = span.get("output_tokens", 0) or span.get("completion_tokens", 0)
            
            result = self.calculate_cost(model, input_tokens, output_tokens)
            
            total_input += input_tokens
            total_output += output_tokens
            total_input_cost += result.input_cost_usd
            total_output_cost += result.output_cost_usd
            providers.add(result.provider)
            models.add(result.model)
        
        total_cost = total_input_cost + total_output_cost
        
        return CostResult(
            input_tokens=total_input,
            output_tokens=total_output,
            total_tokens=total_input + total_output,
            input_cost_usd=round(total_input_cost, 6),
            output_cost_usd=round(total_output_cost, 6),
            total_cost_usd=round(total_cost, 6),
            total_cost_cents=round(total_cost * 100, 2),
            model=",".join(models) if models else "unknown",
            provider=",".join(providers) if providers else "unknown",
        )
    
    def get_context_window(self, model: str) -> int:
        pricing = self.get_pricing(model)
        return pricing.context_window if pricing else 128000
    
    def list_models(self) -> Dict[str, ModelPricing]:
        return {**self.pricing, **self._custom_pricing}


cost_calculator = CostCalculator()
