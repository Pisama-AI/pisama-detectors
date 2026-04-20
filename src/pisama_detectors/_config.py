"""Minimal config shim for the OSS pisama-detectors package.

The backend uses pydantic_settings.BaseSettings to load configuration
from env vars, but the OSS package should run without any app-specific
environment plumbing. This module provides:

- FrameworkThresholds dataclass with the framework-tuned defaults
- get_framework_thresholds(framework) — framework-specific thresholds
- get_tenant_thresholds(tenant_settings, framework) — apply tenant overrides
- Settings dataclass with embedding_model + threshold defaults
- get_settings() — cached singleton

For production deployments that want to override thresholds, callers
should pass explicit thresholds to detector constructors; this shim
intentionally does not read env vars.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, Optional


@dataclass
class FrameworkThresholds:
    structural_threshold: float
    semantic_threshold: float
    loop_detection_window: int
    min_matches_for_loop: int
    confidence_scaling: float


FRAMEWORK_THRESHOLDS: Dict[str, FrameworkThresholds] = {
    "langgraph": FrameworkThresholds(0.92, 0.88, 5, 2, 1.0),
    "autogen": FrameworkThresholds(0.90, 0.80, 10, 3, 0.95),
    "crewai": FrameworkThresholds(0.88, 0.82, 8, 2, 1.0),
    "langchain": FrameworkThresholds(0.95, 0.85, 7, 2, 1.0),
    "openai": FrameworkThresholds(0.93, 0.86, 6, 2, 1.0),
    "anthropic": FrameworkThresholds(0.93, 0.86, 6, 2, 1.0),
    "n8n": FrameworkThresholds(0.98, 0.90, 5, 2, 1.1),
    "dify": FrameworkThresholds(0.95, 0.85, 6, 2, 1.0),
    "openclaw": FrameworkThresholds(0.90, 0.82, 8, 3, 0.95),
    "managed_agents": FrameworkThresholds(0.93, 0.86, 6, 2, 1.0),
    "unknown": FrameworkThresholds(0.95, 0.85, 7, 2, 1.0),
}


def get_framework_thresholds(framework: Optional[str] = None) -> FrameworkThresholds:
    key = (framework or "unknown").lower().strip()
    return FRAMEWORK_THRESHOLDS.get(key, FRAMEWORK_THRESHOLDS["unknown"])


def get_tenant_thresholds(
    tenant_settings: Optional[Dict] = None,
    framework: Optional[str] = None,
) -> FrameworkThresholds:
    base = get_framework_thresholds(framework)
    if not tenant_settings:
        return base

    detection_config = tenant_settings.get("detection_thresholds", {})
    global_overrides = detection_config.get("global", {})
    framework_key = (framework or "unknown").lower().strip()
    framework_overrides = detection_config.get("frameworks", {}).get(framework_key, {})

    merged = {
        "structural_threshold": base.structural_threshold,
        "semantic_threshold": base.semantic_threshold,
        "loop_detection_window": base.loop_detection_window,
        "min_matches_for_loop": base.min_matches_for_loop,
        "confidence_scaling": base.confidence_scaling,
    }
    for overrides in (global_overrides, framework_overrides):
        for k, v in overrides.items():
            if k in merged:
                merged[k] = v
    return FrameworkThresholds(**merged)


@dataclass
class Settings:
    structural_threshold: float = 0.95
    semantic_threshold: float = 0.85
    loop_detection_window: int = 7
    embedding_model: str = "all-MiniLM-L6-v2"


@lru_cache
def get_settings() -> Settings:
    return Settings()
