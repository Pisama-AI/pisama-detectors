"""RAG Poisoning Detection — framework-agnostic.

Detects prompt-injection payloads embedded in retrieved documents and checks
whether the agent output reproduced (echoed) the injected content. This is the
generalized core; framework adapters (Dify, LangGraph, OpenClaw) extract their
schema-specific document/output fields and call ``detect()`` here.

Before this module: the only RAG poisoning protection lived inside
``backend/app/detection/dify/rag_poisoning_detector.py``, hard-coded to Dify's
``workflow_run`` schema (knowledge_retrieval node type, ``outputs.documents[]``).
LangGraph and OpenClaw RAG flows had no protection because the logic was
trapped inside the Dify wrapper.

The injection-pattern table and the echo-detection helper now live here, with
the Dify detector reduced to a thin schema adapter that pulls retrieved
documents out of ``workflow_run.nodes`` and delegates.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class RagPoisoningResult:
    """Result of a RAG poisoning scan."""

    detected: bool
    confidence: float
    pattern_count: int = 0
    affected_documents: List[int] = field(default_factory=list)
    echo_detected: bool = False
    echoed_patterns: List[str] = field(default_factory=list)
    issues: List[Dict[str, Any]] = field(default_factory=list)


# ── Injection pattern table ──────────────────────────────────────────────────
# Patterns are grouped by category so callers can decide which to enable.
# This table is the authoritative source — Dify and any future framework
# detector imports from here.

INJECTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    "system_override": [
        re.compile(r"SYSTEM\s*:", re.IGNORECASE),
        re.compile(r"ignore\s+(all\s+)?previous", re.IGNORECASE),
        re.compile(r"new\s+instructions", re.IGNORECASE),
        re.compile(r"disregard", re.IGNORECASE),
        re.compile(r"forget\s+everything", re.IGNORECASE),
    ],
    "role_hijack": [
        re.compile(r"you\s+are\s+now", re.IGNORECASE),
        re.compile(r"act\s+as", re.IGNORECASE),
        re.compile(r"pretend\s+to\s+be", re.IGNORECASE),
    ],
    "model_specific": [
        re.compile(r"\[INST\]", re.IGNORECASE),
        re.compile(r"<<SYS>>", re.IGNORECASE),
        re.compile(r"</s>"),
    ],
    "zero_width_injection": [
        re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]"),
    ],
    "hidden_override": [
        re.compile(r"SYSTEM_OVERRIDE", re.IGNORECASE),
        re.compile(r"ADMIN_OVERRIDE", re.IGNORECASE),
        re.compile(r"REVEAL_SYSTEM_PROMPT", re.IGNORECASE),
        re.compile(r"skip\s+all\s+validation", re.IGNORECASE),
        re.compile(r"bypass\s+(all\s+)?access\s+control", re.IGNORECASE),
        re.compile(r"bypass\s+auth", re.IGNORECASE),
    ],
    "credential_exposure": [
        re.compile(r"API_KEY\s*=\s*\S+", re.IGNORECASE),
        re.compile(r"DB_PASSWORD\s*=\s*\S+", re.IGNORECASE),
        re.compile(r"ADMIN_TOKEN\s*=\s*\S+", re.IGNORECASE),
        re.compile(r"sk-prod-[a-zA-Z0-9]+"),
        re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}"),
        re.compile(r"(?:password|passwd|pwd)\s*[=:]\s*\S+", re.IGNORECASE),
        re.compile(r"SECRET_KEY\s*=\s*\S+", re.IGNORECASE),
        re.compile(r"(?:access_token|auth_token)\s*=\s*\S+", re.IGNORECASE),
    ],
    "malicious_redirect": [
        re.compile(r"https?://[^\s]*(?:exfil|malicious|attacker|steal)[^\s]*", re.IGNORECASE),
        re.compile(r"\?ref=malicious", re.IGNORECASE),
    ],
    "fabricated_authority": [
        re.compile(
            r"(?:dr\.?|prof\.?|professor)\s+\w+\s+\w+"
            r"(?:\s*,\s*[\w\s]+)?"
            r"\s+(?:from|at|of)\s+"
            r"(?:MIT|Stanford|Harvard|Johns\s+Hopkins|Mayo\s+Clinic|Goldman\s+Sachs|"
            r"Gartner|McKinsey|Forrester|NIST|Oxford|Cambridge|Yale|Princeton|"
            r"WHO|FDA|CDC|NIH|IEEE|ACM|Deloitte|BCG|Bain|JPMorgan|Morgan\s+Stanley)",
            re.IGNORECASE,
        ),
        re.compile(
            r"according\s+to\s+(?:dr\.?|prof\.?|professor)\s+\w+\s+\w+",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:prof\.?|professor|dr\.?)\s+\w+\s+\w+\s*\(\s*"
            r"(?:MIT|Stanford|Harvard|Johns\s+Hopkins|Mayo\s+Clinic|Oxford|Cambridge|Yale|Princeton)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:Gartner|McKinsey|Goldman\s+Sachs|Forrester|Deloitte|BCG|Bain|JPMorgan|Morgan\s+Stanley)"
            r"(?:'s)?\s+(?:analyst|expert|researcher|director|partner|strategist)\s+\w+\s+\w+",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:analyst|expert|researcher|officer|director|strategist)\s+\w+\s+\w+"
            r"\s+(?:from|at|of)\s+"
            r"(?:Gartner|McKinsey|Goldman\s+Sachs|Forrester|MIT|Stanford|Harvard|NIST|Deloitte)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:according\s+to|stated?\s+by|says)\s+"
            r"(?:Warren\s+Buffett|Elon\s+Musk|Jeff\s+Bezos|Mark\s+Zuckerberg|Tim\s+Cook|Bill\s+Gates|"
            r"Jamie\s+Dimon|Larry\s+Fink|Ray\s+Dalio|Janet\s+Yellen)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:99\.9{1,3}|100)\s*%\s*(?:uptime|accuracy|success|reliability|"
            r"satisfaction|effective|proven|guaranteed|certified|compliance)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:study|research|report|analysis|paper)\s+(?:by|from|at|published\s+by)\s+"
            r"(?:MIT|Stanford|Harvard|Johns\s+Hopkins|Mayo\s+Clinic|Goldman\s+Sachs|"
            r"Gartner|McKinsey|Forrester|NIST|Oxford|Cambridge|Yale|Princeton|"
            r"WHO|FDA|CDC|NIH|IEEE|ACM)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:scientifically|clinically|medically)\s+(?:proven|verified|validated|confirmed)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:endorsed|recommended|approved|certified)\s+by\s+"
            r"(?:the\s+)?(?:board|committee|council|association|institute|foundation)",
            re.IGNORECASE,
        ),
        re.compile(
            r"contact\s+(?:dr\.?|prof\.?|professor)\s+\w+\s+\w+\s+at\s+(?:extension|ext\.?|phone|\d)",
            re.IGNORECASE,
        ),
    ],
    "hidden_content": [
        re.compile(r"\[HIDDEN:?\s*[^\]]*\]", re.IGNORECASE),
        re.compile(r"INSTRUCTION_OVERRIDE", re.IGNORECASE),
        re.compile(r"<!--\s*(?:inject|override|system|admin)", re.IGNORECASE),
        re.compile(r"\[CONFIDENTIAL\s*(?:OVERRIDE|INSTRUCTION)\]", re.IGNORECASE),
        re.compile(r"<\s*(?:hidden|invisible|secret)\s*>", re.IGNORECASE),
    ],
    "fabricated_citation": [
        re.compile(
            r"(?:v\.\s+\w+.*?\d{4}|Case\s+No\.\s*\d+[-/]\d+)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:SEC|FTC|EPA|OSHA)\s+(?:ruling|regulation|directive|order)\s+(?:No\.\s*)?\d+",
            re.IGNORECASE,
        ),
        re.compile(
            r"doi:\s*10\.\d{4,}/[^\s]+",
            re.IGNORECASE,
        ),
    ],
    "dangerous_advice": [
        re.compile(
            r"(?:cure|treat|heal|remedy)\s+(?:for\s+)?(?:cancer|diabetes|HIV|AIDS|autism|"
            r"alzheimer|depression|anxiety|ADHD)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:guaranteed|risk[- ]?free)\s+(?:return|profit|income|investment)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:mix|combine|ingest|inject|consume)\s+(?:\w+\s+){0,3}"
            r"(?:bleach|chlorine|ammonia|mercury|cyanide)",
            re.IGNORECASE,
        ),
    ],
}


# ── Helpers ──────────────────────────────────────────────────────────────────


def scan_for_injections(content: str) -> List[Dict[str, Any]]:
    """Scan a single document for injection patterns.

    Returns a list of match dicts. Each dict has ``category``, ``matched``
    (the matched substring), and ``position`` (start offset).
    """
    found: List[Dict[str, Any]] = []
    for category, patterns in INJECTION_PATTERNS.items():
        for pattern in patterns:
            match = pattern.search(content)
            if match:
                found.append(
                    {
                        "category": category,
                        "matched": match.group(),
                        "position": match.start(),
                    }
                )
    return found


def check_llm_echo(injected_strings: List[str], llm_output: str) -> List[str]:
    """Check whether *llm_output* echoes any of *injected_strings*.

    Returns the subset of *injected_strings* that appear (case-insensitively)
    in *llm_output*. Empty list means no echo.
    """
    if not injected_strings or not llm_output:
        return []
    output_lower = llm_output.lower()
    return [s for s in injected_strings if s.lower() in output_lower]


# ── Top-level detector ───────────────────────────────────────────────────────


def detect(
    retrieved_documents: List[str],
    agent_output: str = "",
) -> RagPoisoningResult:
    """Detect RAG poisoning across retrieved documents and downstream output.

    The detector flags two situations:

    1. **Injection in retrieved content** — any of the documents contains a
       known injection pattern (system override, role hijack, fabricated
       authority, etc.).
    2. **Successful echo** — the agent output reproduces an injected substring,
       indicating the LLM was actually steered by the payload (highest-severity
       case).

    Args:
        retrieved_documents: List of document content strings retrieved for the
            agent to ground its output. Non-string entries are str()-cast.
        agent_output: The agent's output text. Optional — if empty, the echo
            check is skipped (poisoning is still flagged on pattern presence).

    Returns:
        RagPoisoningResult with detected/confidence/pattern_count/issues.
        Confidence ranges 0.6 (one pattern, no echo) to 0.99 (multiple patterns
        with confirmed echo).
    """
    if not retrieved_documents:
        return RagPoisoningResult(detected=False, confidence=0.0)

    issues: List[Dict[str, Any]] = []
    affected: List[int] = []
    pattern_count = 0
    all_matched: List[str] = []

    for idx, doc in enumerate(retrieved_documents):
        if not isinstance(doc, str):
            doc = str(doc) if doc is not None else ""
        if not doc:
            continue
        found = scan_for_injections(doc)
        if found:
            pattern_count += len(found)
            affected.append(idx)
            issues.append(
                {
                    "type": "rag_injection",
                    "document_index": idx,
                    "patterns_found": found,
                    "content_preview": doc[:200],
                }
            )
            for f in found:
                all_matched.append(f["matched"])

    if not issues:
        return RagPoisoningResult(detected=False, confidence=0.0)

    echoed = check_llm_echo(all_matched, agent_output)
    has_echo = bool(echoed)

    # Confidence: 0.6 base + 0.1 per pattern, max 0.95; +0.05 if echo confirmed.
    confidence = min(0.95, 0.6 + pattern_count * 0.1)
    if has_echo:
        confidence = min(0.99, confidence + 0.05)

    return RagPoisoningResult(
        detected=True,
        confidence=round(confidence, 4),
        pattern_count=pattern_count,
        affected_documents=affected,
        echo_detected=has_echo,
        echoed_patterns=echoed,
        issues=issues,
    )
