"""
F10: Communication Breakdown Detection (MAST Taxonomy)
======================================================

Detects when a message between agents is misunderstood or
misinterpreted, leading to incorrect behavior downstream.

This includes:
- Intent misalignment (sender meant X, receiver understood Y)
- Format mismatches (expected JSON, got prose)
- Semantic misinterpretation (ambiguous language)
"""

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Raw execution trace/log patterns — these are NOT structured messages
_TRACE_LOG_PATTERNS = [
    r"\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}",
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",
    r"\b(?:INFO|DEBUG|WARNING|ERROR)\b\]?\s+",
    r"RUN\.SH STARTING",
    r"AUTOGEN_TESTBED_SETTING",
    r"\*\*\[Preprocessing\]\*\*",
    r"=== (?:Test write|MetaGPT|Communication Log)",
]


class BreakdownType(str, Enum):
    INTENT_MISMATCH = "intent_mismatch"
    FORMAT_MISMATCH = "format_mismatch"
    SEMANTIC_AMBIGUITY = "semantic_ambiguity"
    INCOMPLETE_INFORMATION = "incomplete_information"
    CONFLICTING_INSTRUCTIONS = "conflicting_instructions"


class BreakdownSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class CommunicationBreakdownResult:
    detected: bool
    breakdown_type: Optional[BreakdownType]
    severity: BreakdownSeverity
    confidence: float
    intent_alignment: float
    format_match: bool
    explanation: str
    suggested_fix: Optional[str] = None


class CommunicationBreakdownDetector:
    """
    Detects F10: Communication Breakdown - message misunderstanding between agents.

    Analyzes message intent, format compliance, and semantic clarity
    to detect communication failures.
    """

    def __init__(
        self,
        intent_threshold: float = 0.35,  # v1.3: Lowered from 0.45 — subtle misalignments were passing
        check_format: bool = True,
        check_ambiguity: bool = True,
    ):
        self.intent_threshold = intent_threshold
        self.check_format = check_format
        self.check_ambiguity = check_ambiguity

    def _detect_expected_format(self, message: str) -> Optional[str]:
        format_hints = {
            "json": [r"\bjson\b", r"\{.*\}", r"format.*json", r"return.*json"],
            "list": [r"\blist\b", r"enumerate", r"bullet.*point", r"\d+\.\s"],
            "code": [r"```", r"\bcode\b", r"implement", r"function.*def", r"class\s+\w+"],
            "markdown": [r"#\s+", r"\*\*.*\*\*", r"##\s+"],
            "csv": [r"\bcsv\b", r"comma.*separated", r",.*,.*,"],
        }

        message_lower = message.lower()
        for fmt, patterns in format_hints.items():
            for pattern in patterns:
                if re.search(pattern, message_lower):
                    return fmt
        return None

    def _check_format_compliance(
        self,
        expected_format: Optional[str],
        response: str,
    ) -> tuple[bool, str]:
        if not expected_format:
            return True, "No specific format expected"

        if expected_format == "json":
            try:
                json.loads(response)
                return True, "Valid JSON"
            except json.JSONDecodeError:
                json_match = re.search(r"\{[^{}]*\}|\[[^\[\]]*\]", response)
                if json_match:
                    try:
                        json.loads(json_match.group())
                        return True, "JSON found in response"
                    except (json.JSONDecodeError, ValueError):
                        pass
                return False, "Expected JSON but response is not valid JSON"

        if expected_format == "list":
            list_patterns = [r"^\s*[-•*]\s+", r"^\s*\d+[.)]\s+"]
            for pattern in list_patterns:
                if re.search(pattern, response, re.MULTILINE):
                    return True, "List format detected"
            return False, "Expected list format but none detected"

        if expected_format == "code":
            if "```" in response or re.search(r"\bdef\s+\w+|class\s+\w+|function\s+\w+", response):
                return True, "Code format detected"
            return False, "Expected code but none detected"

        return True, f"Format check passed for {expected_format}"

    def _detect_ambiguous_language(self, message: str) -> list[str]:
        ambiguous_patterns = [
            (r"\b(it|this|that|these|those)\b(?!\s+is|\s+are|\s+was)", "ambiguous pronoun"),
            (r"\bsome\s+\w+", "vague quantifier"),
            (r"\bmaybe|perhaps|possibly|probably\b", "uncertain language"),
            (r"\betc\.?|and\s+so\s+on|and\s+more\b", "incomplete enumeration"),
            (r"\bsoon|later|eventually\b", "vague timeline"),
            (r"\b(good|bad|nice|fine|okay)\b", "subjective descriptor"),
        ]

        issues = []
        for pattern, issue_type in ambiguous_patterns:
            if re.search(pattern, message.lower()):
                issues.append(issue_type)

        return issues

    @staticmethod
    def _stem(word: str) -> str:
        """Strip common English suffixes so past-tense/plural forms align.

        e.g. 'schedule' / 'scheduled' / 'scheduling' all collapse to 'schedul'.
        """
        for suf in ("ing", "ed", "es", "s"):
            if word.endswith(suf) and len(word) > len(suf) + 2:
                word = word[: -len(suf)]
                break
        # Normalize base forms ending in silent 'e' to match their -ed/-ing stems
        # ("schedule" -> "schedul" matches "scheduled" -> "schedul").
        # Avoid truncating short words or digraph endings where 'e' is semantic.
        if (
            len(word) > 4
            and word.endswith("e")
            and not word.endswith(("ee", "oe", "ie", "ae", "ue"))
        ):
            word = word[:-1]
        return word

    _ACTION_VERBS = frozenset(
        {
            "create",
            "update",
            "delete",
            "get",
            "fetch",
            "send",
            "process",
            "analyze",
            "generate",
            "search",
            "find",
            "calculate",
            "compare",
            "summarize",
            "extract",
            "transform",
            "validate",
            "verify",
            "confirm",
            "acknowledge",
            "respond",
            "reply",
            "escalate",
            "delegate",
            "forward",
            "submit",
            "report",
            "transfer",
            "approve",
            "reject",
            "notify",
            "announce",
            "broadcast",
            "check",
            "monitor",
            "review",
            "implement",
            "deploy",
            "configure",
            "install",
            "migrate",
            "test",
            "debug",
            "fix",
            "resolve",
            "handle",
            "execute",
            "run",
            "backup",
            "restart",
            "restore",
            "compress",
            "resize",
            "increase",
            "decrease",
            "schedule",
            "export",
            "import",
            "renew",
            "trigger",
            "start",
            "stop",
        }
    )

    _STOP_WORDS = frozenset(
        {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "and",
            "or",
            "but",
            "not",
            "no",
            "if",
            "it",
            "i",
            "you",
            "we",
            "they",
            "he",
            "she",
            "this",
            "that",
            "will",
            "can",
            "do",
            "does",
            "did",
            "has",
            "have",
            "had",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "must",
            "need",
            "please",
            "me",
            "my",
            "your",
            "our",
            "their",
            "its",
            "been",
            "being",
            "i've",
            "i'll",
            "i'm",
            "there",
        }
    )

    # Sprint 9 PP compiled regexes for recall/precision fixes.
    _SMALL_ENTITY_RE = re.compile(r"\b[a-z][\w.-]*-\d+\b", re.IGNORECASE)
    _ALL_SCOPE_RE = re.compile(
        r"\b(?:all|entire|every|whole)\s+"
        r"(?:node|server|cluster|host|service|machine|instance)s?\b",
        re.IGNORECASE,
    )
    _SQL_RE = re.compile(
        r"\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE\s+TABLE|WITH\s+\w+\s+AS|"
        r"FROM\s+\w+|WHERE\s+\w+)\b",
        re.IGNORECASE,
    )
    _QUESTION_RE = re.compile(
        r"\b(?:how many|how much|what(?:\s+is|\s+was|'s)|give me|tell me|"
        r"what are|what were|which )\b",
        re.IGNORECASE,
    )
    _NUMERIC_ANS_RE = re.compile(
        r"\b\d[\d,]*\.?\d*\s*"
        r"(?:%|users|tokens|nodes|seconds|hours|days|months|years|"
        r"mb|gb|kb|tb|requests)?\b",
        re.IGNORECASE,
    )
    _REFUSAL_RE = re.compile(
        r"(?:don't have access|i don't have|cannot access|i am unable|"
        r"i'm unable|could you provide|please provide|i need "
        r"(?:the|to)|which\s+\w+\?|i do not have access)",
        re.IGNORECASE,
    )
    _CSV_ROW_RE = re.compile(r"^(?:[^,\n]+,){2,}[^,\n]+$", re.MULTILINE)
    _DEF_FN_RE = re.compile(r"\b(?:def|function)\s+\w+\s*\(")

    # Mutually-exclusive token pairs: if sender has X and receiver has Y,
    # the agent substituted the wrong resource / unit / scope.
    _SUBSTITUTION_PAIRS = [
        ({"fahrenheit"}, {"celsius"}),
        ({"celsius"}, {"fahrenheit"}),
        ({"cpu"}, {"memory", "ram"}),
        ({"memory", "ram"}, {"cpu"}),
        ({"disk"}, {"memory", "ram"}),
        ({"weekly"}, {"monthly", "daily", "yearly"}),
        ({"monthly"}, {"weekly", "daily", "yearly"}),
        ({"daily"}, {"weekly", "monthly", "yearly"}),
        ({"local"}, {"remote"}),
        ({"remote"}, {"local"}),
        ({"web"}, {"mail", "email", "database", "db"}),
        ({"csv"}, {"xml", "json"}),
        ({"xml"}, {"csv", "json"}),
        ({"immediately", "now", "immediate"}, {"scheduled", "next", "later"}),
        ({"compress"}, {"resize", "resized"}),
        ({"pool"}, {"timeout"}),
        ({"excluding"}, {"including"}),
        ({"including"}, {"excluding"}),
        ({"increase", "increased"}, {"decrease", "decreased"}),
        ({"production"}, {"staging", "development", "dev", "test"}),
        ({"admin", "administrator"}, {"standard", "regular", "basic", "user"}),
        ({"auto-renewal", "auto-renew", "automatic", "automated"}, {"manually", "manual"}),
        ({"failed", "failure", "failing"}, {"passing", "passed", "successful"}),
        # Filter contradiction: task specifies a scope filter ("older than N"),
        # response ignores the filter ("regardless of age/size/date").
        ({"older", "newer", "before", "after", "since"}, {"regardless"}),
    ]

    def _has_substitution_mismatch(
        self, request_words: set[str], response_words: set[str]
    ) -> Optional[tuple[str, str]]:
        """Return (sender_term, receiver_term) if a mutually-exclusive
        substitution is detected, else None."""
        for sender_set, receiver_set in self._SUBSTITUTION_PAIRS:
            sender_hit = request_words & sender_set
            receiver_hit = response_words & receiver_set
            # Must be present in sender AND absent in response, with receiver
            # using the alternate term (and sender NOT also using it).
            if sender_hit and receiver_hit and not (request_words & receiver_set):
                return (next(iter(sender_hit)), next(iter(receiver_hit)))
        return None

    def _has_contradictory_pair_mention(
        self, request_words: set[str], response_words: set[str]
    ) -> Optional[tuple[str, str]]:
        """Relaxed variant of substitution detection that fires even when
        the sender's pair-member also appears in the response.

        Example: task asks for "admin privileges", response says "created
        with standard user privileges. Admin requires approval." The
        strict check skips this because the response mentions "admin"; the
        contradiction is still real.

        Guardrails (to avoid conversion-task FPs like "convert CSV to XML"):
          - At least one receiver-set term must NOT be in the sender's
            vocabulary (i.e. the response introduced a term the task did
            not). This filters legit transformation tasks where the
            sender names both source and target format.
        """
        for sender_set, receiver_set in self._SUBSTITUTION_PAIRS:
            sender_hit = request_words & sender_set
            receiver_hit = response_words & receiver_set
            novel_receiver = receiver_hit - request_words
            if sender_hit and novel_receiver:
                return (next(iter(sender_hit)), next(iter(novel_receiver)))
        return None

    @staticmethod
    def _extract_numeric_tokens(text: str) -> list[tuple[str, str]]:
        """Extract (number, trailing_unit_word) pairs from text."""
        return re.findall(r"(\d+(?:\.\d+)?)\s*([A-Za-z%]+)?", text.lower())

    def _compute_intent_alignment(
        self,
        request: str,
        response: str,
        action_taken: Optional[str] = None,
    ) -> float:
        request_words = set(request.lower().split())
        response_words = set(response.lower().split())

        # Stem-normalize so past-tense/plural forms align with base forms.
        request_stems = {self._stem(w) for w in request_words}
        response_stems = {self._stem(w) for w in response_words}
        action_verb_stems = {self._stem(v) for v in self._ACTION_VERBS}

        request_actions = request_stems & action_verb_stems
        response_actions = response_stems & action_verb_stems

        # Content-word overlap (stop words filtered, stemmed).
        stop_stems = {self._stem(w) for w in self._STOP_WORDS}
        content_req = {w for w in request_stems if w not in stop_stems and len(w) > 2}
        content_resp = {w for w in response_stems if w not in stop_stems and len(w) > 2}
        content_overlap = len(content_req & content_resp) / len(content_req) if content_req else 0.0

        if not request_actions:
            # Pure-query request: rely on content overlap as proxy.
            return min(content_overlap * 1.5, 1.0)

        action_match = len(request_actions & response_actions) / len(request_actions)

        negative_indicators = {
            "error",
            "fail",
            "cannot",
            "unabl",
            "refus",
            "sorri",
            "don't",
            "can't",
            "doesn't",
            "couldn't",
        }
        if any(w in response_words for w in negative_indicators) or any(
            self._stem(w) in negative_indicators for w in response_words
        ):
            action_match *= 0.5

        # If content overlaps well despite action-verb mismatch, the response
        # likely addresses the topic in paraphrase — don't false-positive.
        return max(action_match, content_overlap * 0.85)

    def detect(
        self,
        sender_message: str,
        receiver_response: str,
        receiver_action: Optional[str] = None,
        sender_name: Optional[str] = None,
        receiver_name: Optional[str] = None,
    ) -> CommunicationBreakdownResult:
        # Skip format/intent checks on raw execution traces — these are log
        # data, not structured inter-agent messages.
        is_raw_trace = (
            sum(1 for p in _TRACE_LOG_PATTERNS if re.search(p, sender_message[:500])) >= 2
        )

        if self.check_format and not is_raw_trace:
            expected_format = self._detect_expected_format(sender_message)
            format_ok, format_msg = self._check_format_compliance(
                expected_format, receiver_response
            )
        else:
            expected_format = None
            format_ok, format_msg = True, "Format check disabled"

        intent_alignment = self._compute_intent_alignment(
            sender_message, receiver_response, receiver_action
        )
        # Raw traces have near-zero keyword overlap — don't flag as intent mismatch
        if is_raw_trace and intent_alignment < self.intent_threshold:
            intent_alignment = self.intent_threshold  # Neutralize

        ambiguities = self._detect_ambiguous_language(sender_message)

        # Token-substitution check: receiver replaced a critical sender term
        # with a mutually-exclusive alternative (e.g. Fahrenheit -> Celsius,
        # CPU -> memory, weekly -> monthly). Only applies to structured
        # messages, not raw traces.
        # Use regex tokenization (not whitespace split) so trailing punctuation
        # doesn't block matches ("Celsius." -> "celsius").
        request_words = set(re.findall(r"[A-Za-z0-9_-]+", sender_message.lower()))
        response_words = set(re.findall(r"[A-Za-z0-9_-]+", receiver_response.lower()))
        substitution = (
            None if is_raw_trace else self._has_substitution_mismatch(request_words, response_words)
        )

        # Numeric-unit mismatch: sender has "(N, unit_a)" and receiver has
        # "(N, unit_b)" for the same N — wrong unit was applied.
        unit_mismatch = None
        if not is_raw_trace:
            sender_nums = self._extract_numeric_tokens(sender_message)
            receiver_nums = self._extract_numeric_tokens(receiver_response)
            for n_s, u_s in sender_nums:
                if not u_s:
                    continue
                for n_r, u_r in receiver_nums:
                    if n_s == n_r and u_r and u_s != u_r and len(u_s) > 2 and len(u_r) > 2:
                        # Same number, different unit (and neither is empty).
                        unit_mismatch = (f"{n_s} {u_s}", f"{n_r} {u_r}")
                        break
                if unit_mismatch:
                    break

        breakdown_type = None
        detected = False

        is_refusal = any(
            marker in receiver_response.lower()
            for marker in (
                "don't have",
                "cannot",
                "i am unable",
                "i'm unable",
                "i need",
                "could you provide",
                "which ",
                "please provide",
            )
        )

        # Artifact delivered: request asked for a format-shaped output AND
        # response matches that format (JSON/list/code).
        artifact_delivered = (
            expected_format is not None
            and format_ok
            and len(receiver_response.strip()) >= 10
            and not is_refusal
        )

        # v1.3: Completion-marker exemption retained but tightened.
        # Legit confirmations ("Migration completed. Applied 3 migrations...")
        # have a completion marker AND strong topic overlap. Off-topic or
        # wrong-scope confirmations ("Cache cleared completely" for "delete
        # files older than N") are now caught separately via substitution
        # pairs and the lower intent threshold.
        completion_markers = (
            "completed",
            "complete.",
            "done.",
            "finished",
            "initiated",
            "triggered",
            "executed",
            "processed",
            "configured",
            "applied",
            "deployed",
            "installed",
            "launched",
            "delivered",
            "submitted",
            "scheduled",
            "sent to",
            "sent.",
            "has been",
            "have been",
            "will be",
            "is now",
            "are now",
            "up to date",
            "ready",
        )
        response_has_completion = any(m in receiver_response.lower() for m in completion_markers)
        request_stems = {self._stem(w) for w in request_words}
        response_stems = {self._stem(w) for w in response_words}
        stop_stems = {self._stem(w) for w in self._STOP_WORDS}
        content_req = {w for w in request_stems if w not in stop_stems and len(w) > 2}
        content_resp = {w for w in response_stems if w not in stop_stems and len(w) > 2}
        topic_overlap = len(content_req & content_resp) / len(content_req) if content_req else 0.0
        # v1.3: Kept overlap threshold at 0.30 to preserve legit confirmations
        # like "Migration completed. Applied 3 migrations..." where the response
        # vocabulary diverges from the sender's action verbs. Wrong-scope and
        # off-topic confirmations are now caught separately via expanded
        # substitution pairs (production/staging, admin/standard, etc.).
        action_confirmed = response_has_completion and topic_overlap >= 0.30 and not is_refusal

        # Anti-exemption: if the task mentions a substitution-pair term and
        # the response uses the paired alternate (e.g. task says "admin",
        # response confirms "standard user privileges. Admin requires..."),
        # the completion is of the wrong thing. The strict substitution
        # check at line 323 misses these because the response also mentions
        # the sender's term; but the contradiction is still real. Promote
        # to a substitution detection so the intent-mismatch fires even
        # when topic overlap and intent alignment are high.
        if not substitution and not is_raw_trace:
            contradictory_pair = self._has_contradictory_pair_mention(request_words, response_words)
            if contradictory_pair is not None:
                substitution = contradictory_pair

        # Sprint 9 PP: scope-expansion detection — sender names a specific
        # entity like "cluster-node-03", receiver claims action "on all nodes".
        scope_expansion = False
        if not is_raw_trace:
            sender_entity = self._SMALL_ENTITY_RE.search(sender_message)
            if sender_entity and self._ALL_SCOPE_RE.search(receiver_response):
                scope_expansion = True

        # Sprint 9 PP: artifact_delivered over-trusts CSV/markdown format
        # because _check_format_compliance returns True by default for those.
        # Require actual evidence (CSV row pattern in receiver) before
        # suppressing the intent-mismatch path.
        if artifact_delivered and expected_format == "csv":
            if not self._CSV_ROW_RE.search(receiver_response):
                artifact_delivered = False

        if substitution:
            detected = True
            breakdown_type = BreakdownType.INTENT_MISMATCH
        elif unit_mismatch:
            detected = True
            breakdown_type = BreakdownType.INTENT_MISMATCH
        elif scope_expansion:
            detected = True
            breakdown_type = BreakdownType.INTENT_MISMATCH
        elif not format_ok:
            detected = True
            breakdown_type = BreakdownType.FORMAT_MISMATCH
        elif (
            intent_alignment < self.intent_threshold
            and not artifact_delivered
            and not action_confirmed
        ):
            detected = True
            breakdown_type = BreakdownType.INTENT_MISMATCH
        elif len(ambiguities) >= 4:
            detected = True
            breakdown_type = BreakdownType.SEMANTIC_AMBIGUITY

        # Sprint 9 PP: precision exemptions for low-alignment FPs. Only apply
        # when the detection was based on the weak intent-alignment rule
        # (not substitution/unit/format/scope-expansion, which are
        # deterministic contradictions).
        if (
            detected
            and breakdown_type == BreakdownType.INTENT_MISMATCH
            and not substitution
            and not unit_mismatch
            and not scope_expansion
        ):
            # Refusal / clarification requests are not breakdowns.
            if self._REFUSAL_RE.search(receiver_response) and len(receiver_response) < 400:
                detected = False
                breakdown_type = None
            # SQL artifact (SELECT/INSERT/...) delivered — code artifact.
            elif self._SQL_RE.search(receiver_response):
                detected = False
                breakdown_type = None
            # Question with a numeric answer — direct factual response.
            elif (
                self._QUESTION_RE.search(sender_message)
                and self._NUMERIC_ANS_RE.search(receiver_response)
                and len(receiver_response) < 300
            ):
                detected = False
                breakdown_type = None
            # Code function definition delivered — Python / JS artifact.
            elif self._DEF_FN_RE.search(receiver_response):
                detected = False
                breakdown_type = None
            else:
                # Substantive response with partial topic overlap + numeric
                # data suggests a legitimate answer, not a breakdown.
                overlap_content_req = {
                    w for w in request_stems if w not in stop_stems and len(w) > 2
                }
                overlap_content_resp = {
                    w for w in response_stems if w not in stop_stems and len(w) > 2
                }
                overlap_ratio = (
                    len(overlap_content_req & overlap_content_resp) / len(overlap_content_req)
                    if overlap_content_req
                    else 0.0
                )
                if (
                    len(receiver_response) >= 120
                    and overlap_ratio >= 0.15
                    and self._NUMERIC_ANS_RE.search(receiver_response)
                ):
                    detected = False
                    breakdown_type = None

        if not detected:
            return CommunicationBreakdownResult(
                detected=False,
                breakdown_type=None,
                severity=BreakdownSeverity.NONE,
                confidence=intent_alignment,
                intent_alignment=intent_alignment,
                format_match=format_ok,
                explanation="Communication appears clear",
            )

        if breakdown_type == BreakdownType.FORMAT_MISMATCH:
            severity = BreakdownSeverity.MODERATE
            confidence = 0.9
            explanation = format_msg
            fix = f"Ensure response follows {expected_format} format. Add explicit format instructions."
        elif breakdown_type == BreakdownType.INTENT_MISMATCH:
            if intent_alignment < 0.2:
                severity = BreakdownSeverity.SEVERE
            else:
                severity = BreakdownSeverity.MODERATE
            # Substitution and unit mismatches are deterministic contradictions
            # (wrong unit, wrong env, wrong format) — confidence must reflect
            # the strong signal, not be dampened by accidental topic overlap.
            if substitution is not None or unit_mismatch is not None:
                confidence = 0.9
            else:
                confidence = 1 - intent_alignment
                # Moderate confidence when keywords overlap despite verb mismatch.
                # High keyword overlap means topics match — likely NOT a real breakdown.
                request_words = set(sender_message.lower().split())
                response_words = set(receiver_response.lower().split())
                # Filter stop words for more meaningful overlap
                _stop = {
                    "the",
                    "a",
                    "an",
                    "is",
                    "are",
                    "was",
                    "were",
                    "be",
                    "been",
                    "to",
                    "of",
                    "in",
                    "for",
                    "on",
                    "with",
                    "at",
                    "by",
                    "from",
                    "and",
                    "or",
                    "but",
                    "not",
                    "no",
                    "if",
                    "it",
                    "i",
                    "you",
                    "we",
                    "they",
                    "he",
                    "she",
                    "this",
                    "that",
                    "will",
                    "can",
                    "do",
                    "does",
                    "did",
                    "has",
                    "have",
                    "had",
                    "would",
                    "could",
                    "should",
                    "may",
                    "might",
                    "shall",
                    "must",
                    "need",
                }
                req_content = request_words - _stop
                resp_content = response_words - _stop
                if req_content:
                    content_overlap = len(req_content & resp_content) / len(req_content)
                    if content_overlap > 0.3:
                        # Topics overlap despite verb mismatch → reduce confidence
                        confidence *= max(0.4, 1.0 - content_overlap)
            explanation = (
                f"Response does not align with request intent. "
                f"Alignment score: {intent_alignment:.1%}"
            )
            fix = "Clarify request with specific action verbs and expected outcomes."
        else:
            severity = BreakdownSeverity.MINOR
            confidence = 0.6
            explanation = f"Ambiguous language detected: {', '.join(ambiguities)}"
            fix = "Replace ambiguous language with specific references."

        sender_label = f"'{sender_name}'" if sender_name else "sender"
        receiver_label = f"'{receiver_name}'" if receiver_name else "receiver"
        explanation = f"Communication from {sender_label} to {receiver_label}: {explanation}"

        return CommunicationBreakdownResult(
            detected=True,
            breakdown_type=breakdown_type,
            severity=severity,
            confidence=confidence,
            intent_alignment=intent_alignment,
            format_match=format_ok,
            explanation=explanation,
            suggested_fix=fix,
        )

    def detect_from_trace(
        self,
        trace: dict,
    ) -> list[CommunicationBreakdownResult]:
        results = []

        spans = trace.get("spans", [])
        for i in range(len(spans) - 1):
            sender = spans[i]
            receiver = spans[i + 1]

            sender_output = sender.get("output", {}).get("content", "")
            receiver.get("input", {}).get("message", "")
            receiver_output = receiver.get("output", {}).get("content", "")

            if sender_output and receiver_output:
                result = self.detect(
                    sender_message=sender_output,
                    receiver_response=receiver_output,
                    sender_name=sender.get("name"),
                    receiver_name=receiver.get("name"),
                )
                if result.detected:
                    results.append(result)

        return results
