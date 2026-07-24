"""Shared safety primitives for keyword/pattern matching across detectors.

Detectors that previously defined their own destructive-verb tables now import
from `destructive_keywords` to keep the canonical list in one place.
"""

from pisama_detectors.detection.safety.destructive_keywords import (
    ADMIN_VERBS,
    APPROVAL_HIGH_RISK_VERBS,
    BULK_DATA_VERBS,
    COWORK_DESTRUCTIVE_VERBS,
    DELETE_VERBS,
    DEPLOY_VERBS,
    EXECUTE_VERBS,
    EXPLORATION_DANGEROUS_VERBS,
    FINANCIAL_VERBS,
    OPENCLAW_RISKY_KEYWORDS,
    PERMISSION_VERBS,
    SEND_VERBS,
    WRITE_VERBS,
    make_verb_pattern,
)

__all__ = [
    "ADMIN_VERBS",
    "BULK_DATA_VERBS",
    "DELETE_VERBS",
    "DEPLOY_VERBS",
    "EXECUTE_VERBS",
    "FINANCIAL_VERBS",
    "PERMISSION_VERBS",
    "SEND_VERBS",
    "WRITE_VERBS",
    "APPROVAL_HIGH_RISK_VERBS",
    "COWORK_DESTRUCTIVE_VERBS",
    "EXPLORATION_DANGEROUS_VERBS",
    "OPENCLAW_RISKY_KEYWORDS",
    "make_verb_pattern",
]
