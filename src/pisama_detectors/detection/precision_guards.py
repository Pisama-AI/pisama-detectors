"""Shared precision guards: polarity and outcome awareness for rule tiers.

Background
----------
An adversarial near-miss sweep (2026-08-18) found that 16 of 16 production
detectors probed fired on genuinely healthy agent traces. The finding was not
16 unrelated bugs but one repeated habit: **a rule matches the presence of a
surface token and fires, without asking whether the token asserts a failure or
its absence, and without asking whether the underlying operation actually
failed.**

Concretely, all of these were counted as evidence of a problem:

    "errors: 0"                       (a clean run)
    "critical alerts: none"           (a clean run)
    "timeout: not reached"            (a clean run)
    "0 failed, 0 upstream_failed"     (a clean run)
    "Validation passed for all items" (a node that PASSED)
    "no deprecated sources"           (a clean lineage)

This module supplies the two missing questions as reusable primitives so
detectors stop re-deriving them (badly) one regex at a time:

    asserts_absence(text, start, end)  -> polarity: is this token NEGATED?
    outcome_is_success(blob)           -> outcome: did the operation SUCCEED?

Design notes
------------
* Both helpers are deliberately CONSERVATIVE. They only return True on an
  explicit, local, unambiguous signal. When in doubt they return False /
  ``None``, which preserves the detector's existing behaviour. That keeps the
  blast radius of adopting them one-directional: they can only ever SUPPRESS a
  fire on evidence that is explicitly negated or explicitly successful, never
  create a new one.
* They are pure string/dict inspection: no I/O, no model calls, no config.
  Cheap enough to run inside the innermost regex loop of a rule tier.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "asserts_absence",
    "filter_absent_matches",
    "negates_problem",
    "outcome_is_success",
    "all_operations_permitted",
    "policy_decision",
    "workflow_reached_success",
    "conforms_to_declared_type",
    "is_lossless_coercion",
    "driven_by_distinct_inputs",
    "reports_clean_outcome",
    "reports_explicit_failure",
]


# ── Polarity ────────────────────────────────────────────────────────────────

# Words that, appearing just BEFORE a problem token, negate it.
#   "no errors", "zero failures", "without warnings", "free of vulnerabilities"
_LEADING_NEGATORS = (
    r"no|not|non|none|never|neither|nor|zero|without|sans|absent|"
    r"free\s+of|clear\s+of|devoid\s+of|lack(?:ing)?\s+of|nothing|"
    r"0+|n/?a"
)

# A leading negator, optionally with a few filler words between it and the
# token: "no known errors", "zero unexpected failures", "without any warnings".
_FILLER = (
    r"(?:\s+(?:any|known|other|further|new|remaining|outstanding|unexpected|"
    r"apparent|obvious|reported|detected|observed|open|active|critical|major|"
    r"minor|real)){0,3}"
)

_LEADING_RE = re.compile(
    rf"(?:^|[\s\(\[\{{,;:\"'|>*-])(?:{_LEADING_NEGATORS}){_FILLER}\s+$",
    re.IGNORECASE,
)

# Zero-ish values on the right-hand side of a key: "errors: 0", "warnings = none".
_ZERO_VALUE = r"(?:0+(?:\.0+)?|none|nil|null|n/?a|no|false|clean|ok|empty|-{1,2}|\[\])"

# "<token>[ up to a few more words] : 0 | none | false"
_TRAILING_KV_RE = re.compile(
    rf"^[\w./_-]*(?:[ \t]+[\w./_-]+){{0,3}}[ \t]*[:=][ \t]*{_ZERO_VALUE}\b",
    re.IGNORECASE,
)

# Verbs that, when negated right after the token, mean the bad thing did NOT
# happen: "timeout: not reached", "limit never exceeded", "alert not triggered".
_NEGATED_OUTCOME_RE = re.compile(
    r"^[\w./_-]*(?:[ \t]+[\w./_-]+){0,3}[ \t]*[:=]?[ \t]*"
    r"(?:was|were|is|are|has|have|had)?[ \t]*"
    r"(?:not|never|n't|no)[ \t]+"
    r"(?:been[ \t]+)?"
    r"(?:reached|exceeded|hit|triggered|breached|detected|found|present|"
    r"encountered|needed|required|used|raised|thrown|observed|reported|"
    r"seen|met|violated|crossed|surpassed)\b",
    re.IGNORECASE,
)

# Success verbs immediately after the token: "validation passed", "checks ok".
_TRAILING_SUCCESS_RE = re.compile(
    r"^[\w./_-]*(?:[ \t]+[\w./_-]+){0,3}[ \t]*[:=]?[ \t]*"
    r"(?:all[ \t]+)?"
    r"(?:passed|passing|succeeded|success(?:ful)?|ok|green|healthy|clean|"
    r"resolved|cleared|satisfied)\b",
    re.IGNORECASE,
)

# "88 of 88 tests passed", "412 of 412 daily partitions present"
_ALL_PASSED_RE = re.compile(
    r"\b(\d+)\s+of\s+\1\b[^.\n]{0,40}?\b(?:passed|succeeded|present|ok)\b",
    re.IGNORECASE,
)


def _clause_left(text: str, start: int, window: int) -> str:
    """Text immediately left of the match, clipped at a clause boundary."""
    left = text[max(0, start - window):start]
    # Do not let a negation leak across a sentence / list-item boundary.
    for sep in (". ", "\n", ";", " but ", " however ", " except "):
        idx = left.rfind(sep)
        if idx != -1:
            left = left[idx + len(sep):]
    return left


def asserts_absence(text: str, start: int, end: int, *, window: int = 48) -> bool:
    """True when the token at ``text[start:end]`` asserts the ABSENCE of a problem.

    This is the polarity question a rule tier must ask before counting a
    matched token as evidence. Returns True for constructions like::

        no errors            zero failures        without warnings
        errors: 0            risk flags: none     critical alerts: none
        timeout: not reached validation passed    88 of 88 tests passed

    and False for a genuine assertion ("3 errors", "the deploy failed").

    Conservative by design: an unrecognised construction returns False, so the
    caller keeps its existing behaviour.
    """
    if start < 0 or end > len(text) or start >= end:
        return False

    left = _clause_left(text, start, window)
    if _LEADING_RE.search(left):
        return True

    right = text[end:end + window]
    if _TRAILING_KV_RE.match(right):
        return True
    if _NEGATED_OUTCOME_RE.match(right):
        return True
    if _TRAILING_SUCCESS_RE.match(right):
        return True

    # "88 of 88 tests passed" is a clause-level assertion, so scan the whole
    # containing line rather than a fixed character window (a window clips
    # mid-phrase and silently misses the trailing verb).
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start:line_end if line_end != -1 else len(text)]
    if _ALL_PASSED_RE.search(line):
        return True

    return False


# Nouns that name a problem. "no <problem>" is GOOD news, not a finding to
# report — the distinction a negative-finding rule must make before treating
# "no errors" as a suppressed negative.
_PROBLEM_NOUNS = frozenset({
    "error", "errors", "failure", "failures", "fault", "faults", "exception",
    "exceptions", "bug", "bugs", "issue", "issues", "problem", "problems",
    "defect", "defects", "warning", "warnings", "alert", "alerts", "risk",
    "risks", "danger", "dangers", "vulnerability", "vulnerabilities",
    "breach", "breaches", "exploit", "exploits", "blocker", "blockers",
    "impediment", "impediments", "obstacle", "obstacles", "regression",
    "regressions", "incident", "incidents", "outage", "outages", "conflict",
    "conflicts", "violation", "violations", "gap", "gaps", "mismatch",
    "mismatches", "discrepancy", "discrepancies", "anomaly", "anomalies",
    "timeout", "timeouts", "crash", "crashes", "leak", "leaks", "retry",
    "retries", "misses", "miss", "stale", "deprecated", "downtime",
    "escalation", "escalations", "complaint", "complaints",
    # participles / adjectives that name the problem STATE, so "never failed"
    # and "nothing missing" read as the good news they are
    "failed", "failing", "missing", "broken", "blocked", "denied", "rejected",
    "degraded", "corrupted", "corrupt", "lost", "dropped", "delayed",
    "exceeded", "breached", "unavailable", "unreachable", "unhealthy",
    "invalid", "incorrect", "wrong", "skipped", "aborted", "cancelled",
})

_NEGATOR_PHRASE_RE = re.compile(
    r"^\W*(?:no|not|none|never|neither|nor|zero|without|nothing|0+)\b"
    r"(?P<rest>.*)$",
    re.IGNORECASE | re.DOTALL,
)


def negates_problem(phrase: str) -> bool:
    """True when ``phrase`` claims the ABSENCE of a problem.

    Distinguishes "no errors" / "zero regressions" / "nothing missing" (good
    news, and NOT a negative finding a summariser is suppressing) from
    "no access" / "no matching records" (a genuine negative finding).
    """
    if not phrase:
        return False
    match = _NEGATOR_PHRASE_RE.match(phrase.strip())
    if not match:
        return False
    words = re.findall(r"[a-zA-Z]+", match.group("rest").lower())
    return any(word in _PROBLEM_NOUNS for word in words[:3])


def filter_absent_matches(
    text: str, matches: Iterable[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Drop (start, end) spans whose token merely asserts an absence."""
    return [(s, e) for (s, e) in matches if not asserts_absence(text, s, e)]


# ── Outcome ─────────────────────────────────────────────────────────────────

_SUCCESS_TOKENS = {
    "success", "succeeded", "successful", "ok", "okay", "pass", "passed",
    "passing", "complete", "completed", "completed_successfully", "done",
    "finished", "healthy", "green", "allowed", "permitted", "granted",
    "accepted", "applied", "committed", "resolved",
}
_FAILURE_TOKENS = {
    "error", "errored", "failure", "failed", "failing", "denied", "blocked",
    "rejected", "refused", "forbidden", "timeout", "timed_out", "cancelled",
    "canceled", "aborted", "crashed", "exception", "unauthorized", "red",
}

_STATUS_KEYS = (
    "status", "state", "result", "outcome", "verdict", "disposition",
    "exit_status", "run_state", "phase",
)
_OK_FLAG_KEYS = ("ok", "success", "succeeded", "passed", "allowed", "permitted")
_CODE_KEYS = ("exit_code", "returncode", "return_code", "status_code", "code")


def _normalise(value: Any) -> str:
    return str(value).strip().strip(".").lower().replace(" ", "_")


def outcome_is_success(blob: Any) -> Optional[bool]:
    """Did the operation described by ``blob`` succeed?

    ``blob`` may be a dict (a tool-call record, a node result, an event) or a
    string (raw tool output). Returns:

        True   — an explicit success signal was found
        False  — an explicit failure signal was found
        None   — no explicit signal; the caller must not infer either way

    The tri-state matters: ``None`` is not ``False``. A detector that treats
    "unknown" as "failed" is how a clean run gets flagged in the first place.
    """
    if blob is None:
        return None

    if isinstance(blob, dict):
        for key in _OK_FLAG_KEYS:
            if key in blob and isinstance(blob[key], bool):
                return blob[key]
        for key in _CODE_KEYS:
            if key in blob and isinstance(blob[key], (int, float)):
                code = int(blob[key])
                if key in ("status_code", "code") and code >= 100:
                    return 200 <= code < 400
                return code == 0
        for key in _STATUS_KEYS:
            if key in blob:
                token = _normalise(blob[key])
                if token in _SUCCESS_TOKENS:
                    return True
                if token in _FAILURE_TOKENS:
                    return False
        for key in ("error", "errors", "exception", "failure"):
            if key in blob:
                val = blob[key]
                if val in (None, "", [], {}, 0, False):
                    return True
                return False
        return None

    if isinstance(blob, str):
        token = _normalise(blob)
        if token in _SUCCESS_TOKENS:
            return True
        if token in _FAILURE_TOKENS:
            return False
        return None

    return None


_POLICY_KEYS = (
    "sandbox_policy", "policy", "policy_decision", "permission", "decision",
    "authorization", "authz", "acl", "guard", "enforcement",
)
_ALLOW_PREFIXES = ("allow", "permit", "grant", "sanction", "approv", "accept", "ok")
_DENY_PREFIXES = (
    "deny", "denied", "block", "reject", "refus", "forbid", "violat",
    "escape", "unauthor", "prohibit", "disallow",
)
_NEGATED_ALLOW_RE = re.compile(
    r"\b(?:not|never)\s+(?:allowed?|permitted?|granted?|approved?)\b|"
    r"\b(?:operation|access|permission)\s+(?:is\s+)?not\s+(?:allowed?|permitted?)\b|"
    r"\bpermission\s+denied\b",
    re.IGNORECASE,
)


def policy_decision(blob: Any) -> Optional[bool]:
    """What did the enforcement layer decide about this operation?

        True   — explicitly ALLOWED  ("sandbox_policy": "allowed: scratch read")
        False  — explicitly DENIED   ("sandbox_policy": "sandbox_policy_denied")
        None   — no policy signal recorded

    Policy detectors need this to separate "the agent did something restricted
    and the sandbox PERMITTED it" (sanctioned use of the sandbox) from "the
    agent tried and was STOPPED" (a genuine escape attempt). Matching a tool
    NAME against a restricted list answers neither question.
    """
    if not isinstance(blob, dict):
        return None
    for key in _POLICY_KEYS:
        if key not in blob:
            continue
        value = blob[key]
        if isinstance(value, bool):
            return value
        token = str(value).strip().lower()
        # Check negated allow-language first. Substring/prefix allow matching
        # must never turn "operation not permitted" into an allow verdict.
        if _NEGATED_ALLOW_RE.search(token):
            return False
        if any(d in token for d in _DENY_PREFIXES):
            return False
        if token.startswith(_ALLOW_PREFIXES) or any(
            token.startswith(f"{a}ed") or f" {a}" in token for a in ("allow", "permit")
        ):
            return True
    return None


def all_operations_permitted(events: Sequence[Dict[str, Any]]) -> bool:
    """True when no event in ``events`` was denied, blocked, or errored.

    Used by policy/sandbox detectors, which must distinguish "the agent
    ATTEMPTED something restricted and was stopped" from "the agent performed
    an allowed operation that happens to touch a sensitive-sounding tool".
    """
    for event in events or ():
        if not isinstance(event, dict):
            continue
        if outcome_is_success(event) is False:
            return False
        for key in ("denied", "blocked", "rejected", "violation", "escaped"):
            if event.get(key):
                return False
    return True


# A clause that positively asserts the operation went fine.
_SUCCESS_ASSERTION_RE = re.compile(
    r"\b(?:"
    r"validation\s+(?:passed|succeeded|ok)|"
    r"(?:all|every)\s+\w+(?:\s+\w+)?\s+(?:passed|present|valid|populated|provided)|"
    r"passed\s+for\s+all|"
    r"completed\s+successfully|finished\s+successfully|ran\s+successfully|"
    r"(?:status|state|result|outcome)\s*[:=]\s*(?:success|succeeded|ok|passed|completed)|"
    r"workflow_status\s*[:=]\s*success|"
    r"no\s+(?:errors?|failures?|issues?|problems?|warnings?)\s+(?:were\s+)?"
    r"(?:found|detected|reported|raised|encountered)?|"
    r"\d+\s+of\s+\d+\s+\w+\s+passed"
    r")\b",
    re.IGNORECASE,
)

# An unambiguous failure marker. Its presence vetoes the clean-outcome verdict,
# so a mixed message ("validation passed for step 1; step 2 threw") is never
# suppressed.
_EXPLICIT_FAILURE_RE = re.compile(
    r"\b(?:"
    r"cannot\s+read\s+propert|is\s+not\s+defined|\bundefined\b|"
    r"\bmismatches?\b|traceback|stack\s*trace|"
    r"unhandled|uncaught|"
    r"(?:threw|throws|thrown|raised|hit|caught)\s+(?:an?\s+)?[\w.]*(?:exception|error)|"
    r"\b[A-Za-z_]+(?:Error|Exception)\b|"
    r"errno|econnrefused|etimedout|"
    r"required\s+(?:field|key|column)\s+(?:is\s+)?"
    r"(?:missing|absent|not\s+(?:present|provided|found))|"
    r"(?:missing|absent)\s+(?:a\s+)?(?:required\s+)?(?:field|key|column)|"
    r"(?:error|exception|failure)\s*[:=]\s*(?!0\b|none\b|null\b|\[\])\S|"
    r"(?:failed|errored|crashed|aborted|timed\s+out|rejected|denied)\b"
    r")",
    re.IGNORECASE,
)

_TYPE_COMPARISON_RE = re.compile(
    r"\bexpected\s+(?P<expected>string|number|array|object)\b"
    r"[^.\n;]{0,80}\b(?:but\s+)?(?:got|received|found)\s+"
    r"(?:an?\s+)?(?P<actual>string|number|array|object)\b",
    re.IGNORECASE,
)


def reports_clean_outcome(text: str) -> bool:
    """True when ``text`` positively asserts a clean run and shows no failure.

    Outcome awareness for rule tiers that mine free-text node/tool output for
    error SUBSTRINGS. A node that reports "Validation passed for all 3 items.
    Every required field is present" is describing a success; scoring its
    prose for the substring "required field" turns a passing validator into a
    schema error.

    Requires BOTH a positive success assertion AND the absence of any explicit
    failure marker, so a mixed message still falls through to the normal rules.
    """
    if not text:
        return False
    if reports_explicit_failure(text):
        return False
    return bool(_SUCCESS_ASSERTION_RE.search(text))


_FAILURE_TOKEN_RE = re.compile(
    r"\b(?:errors?|failures?|exceptions?|timeouts?|failed|aborted|denied|rejected)\b",
    re.IGNORECASE,
)


def reports_explicit_failure(text: str) -> bool:
    """True when text positively reports a failure rather than its absence."""
    if not text:
        return False
    for comparison in _TYPE_COMPARISON_RE.finditer(text):
        if comparison.group("expected").lower() != comparison.group("actual").lower():
            return True
    if _EXPLICIT_FAILURE_RE.search(text):
        return True
    for match in _FAILURE_TOKEN_RE.finditer(text):
        if not asserts_absence(text, match.start(), match.end()):
            return True
    return False


def driven_by_distinct_inputs(windows: Sequence[Sequence[str]]) -> bool:
    """True when each repetition was driven by DIFFERENT intervening input.

    The loop family (content fingerprints, tool-call repetition, node
    revisits, workflow cycles) all share one blind spot: they observe that an
    agent emitted the same thing N times and conclude it is stuck. But an
    agent that screens six contracts and correctly clears four of them emits
    the same deterministic verdict four times — over four DIFFERENT documents.
    That is a fan-out, not a loop.

    ``windows`` is the sequence of content windows sitting BETWEEN consecutive
    repetitions. When every window is non-empty and no two windows are
    identical, the repetitions were driven by distinct inputs and the
    repetition is not evidence of being stuck.

    A genuine loop has empty or identical intervening windows (the same
    request re-issued, or nothing new arriving between attempts), so this
    returns False and the detector fires as before.
    """
    normalised = []
    for window in windows:
        joined = " ".join(part.strip() for part in window if part and part.strip())
        joined = re.sub(r"\s+", " ", joined).strip().lower()
        if not joined:
            return False  # nothing happened between repeats — a real loop
        if reports_explicit_failure(joined):
            return False  # changing failure/timeout prose is not progress
        normalised.append(joined)
    if len(normalised) < 1:
        return False
    return len(set(normalised)) == len(normalised)


_DECLARED_TYPE_MAP: dict[str, type | tuple[type, ...]] = {
    "str": str, "string": str, "text": str,
    "int": int, "integer": int,
    "float": float, "number": (int, float),
    "bool": bool, "boolean": bool,
    "list": list, "array": list, "sequence": list,
    "dict": dict, "object": dict, "mapping": dict,
}

_OPTIONAL_RE = re.compile(r"^optional\[(.+)\]$|^(.+)\s*\|\s*none$", re.IGNORECASE)


_BOOL_STRINGS = {
    "true": True, "false": False, "yes": True, "no": False,
    "1": True, "0": False, "t": True, "f": False,
}

# Converting a zero-padded numeric string to a number discards information.
# Fields such as postal codes and account numbers commonly depend on those
# leading zeroes, so their conversion is not provably lossless without schema
# evidence that the field is numeric.
_LEADING_ZERO_NUMERIC_RE = re.compile(
    r"^[+-]?0\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$"
)


def _canonical_scalar(value: Any) -> Any:
    """Reduce a scalar to a comparable canonical form, or return a sentinel."""
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        try:
            number = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return ("other", value)
        return ("num", number) if number.is_finite() else ("other", value)
    if isinstance(value, str):
        text = value.strip()
        lowered = text.lower()
        if lowered in _BOOL_STRINGS:
            return ("bool", _BOOL_STRINGS[lowered])
        if _LEADING_ZERO_NUMERIC_RE.fullmatch(text):
            return ("str", text)
        try:
            number = Decimal(text)
            return ("num", number) if number.is_finite() else ("other", value)
        except (InvalidOperation, ValueError):
            return ("str", text)
    return ("other", value)


def is_lossless_coercion(previous: Any, current: Any) -> bool:
    """True when a type change PRESERVES the value.

    Type-drift rules flag any field whose Python type changed between
    snapshots. But an ETL / schema-migration / normalisation agent exists to
    change types: ``"68.00" -> 68.0``, ``"2" -> 2``, ``"true" -> True``. Those
    carry the same information in the target type and are the agent doing its
    job, not state corruption. A genuine corruption ("68.00" -> "N/A", 68.0 ->
    0) changes the value and is unaffected.

    Ambiguous cases return False so the caller keeps its existing behaviour.
    """
    prev_canon = _canonical_scalar(previous)
    curr_canon = _canonical_scalar(current)
    if prev_canon[0] == "other" or curr_canon[0] == "other":
        return False
    # A bool and the number 1/0 are NOT interchangeable for our purposes:
    # only accept a coercion when the canonical kind and value both agree.
    return prev_canon == curr_canon


def conforms_to_declared_type(value: Any, declared: Optional[str]) -> Optional[bool]:
    """Does ``value`` match the type the schema DECLARES for its field?

        True   — conforms (so a "type change" here is the schema working)
        False  — violates the declared type
        None   — no declaration, or a form we do not parse

    State-corruption rules flag any key whose Python type changed between
    snapshots. But a graph that declares ``selected_docs: Optional[list]`` and
    moves it from None to a list is doing exactly what it said it would; the
    type "changed" only in the sense that Optional means it must. Checking the
    declared schema separates a filled-in optional from real corruption.
    """
    if not declared or not isinstance(declared, str):
        return None
    text = declared.strip()
    optional = False
    match = _OPTIONAL_RE.match(text)
    if match:
        optional = True
        text = (match.group(1) or match.group(2) or "").strip()
    if value is None:
        return True if optional else None
    expected = _DECLARED_TYPE_MAP.get(text.lower())
    if expected is None:
        return None
    # bool is a subclass of int; keep them distinct.
    if isinstance(value, bool) and expected is not bool:
        return False
    return isinstance(value, expected)


def workflow_reached_success(states: Sequence[Any]) -> bool:
    """True when the final state of a run carries an explicit success outcome.

    A run that completed successfully is weak-to-no evidence for
    loop / recursion / resource-exhaustion detections, all of which are claims
    about a run that FAILED to make progress.
    """
    for state in reversed(list(states or ())):
        delta = getattr(state, "state_delta", None)
        if delta is None and isinstance(state, dict):
            delta = state.get("state_delta")
        verdict = outcome_is_success(delta) if isinstance(delta, dict) else None
        if verdict is not None:
            return verdict
    return False
