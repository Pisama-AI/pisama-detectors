"""Deterministic direct-contradiction checks for specification matching."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class _PolarityConstraint:
    negative: bool
    subject: frozenset[str]
    actions: frozenset[str]
    text: str


@dataclass(frozen=True)
class _NumericConstraint:
    bound: str
    strict: bool
    value: float
    unit: str
    text: str
    subject: frozenset[str]
    actions: frozenset[str]


@dataclass(frozen=True)
class _ScopeConstraint:
    scope: str
    subject: frozenset[str]
    actions: frozenset[str]
    text: str


@dataclass(frozen=True)
class _LanguageConstraint:
    language: str
    subject: frozenset[str]
    actions: frozenset[str]
    text: str


_STOP_WORDS = frozenset(
    "a all an and any are at be been being but can do does ensure every for from has have in "
    "include includes is may must need needs never no not of on only or shall should that the "
    "to using was were will with without".split()
)
_POLARITY_GENERIC_TERMS = frozenset({"api", "request", "service", "system"})
_NEGATIVE_POLARITY = re.compile(
    r"\b(?:(?:must|should|shall|can|could|will)\s+(?:not|never)|"
    r"(?:do|does|is|are)\s+not|cannot|can't|never|without|no)\b",
    re.IGNORECASE,
)
_POLARITY_SPLIT = re.compile(
    r"(?<!\d)[.!?;\n]+|\b(?:and|but)\s+"
    r"(?=(?:must|should|shall|can|cannot|can't|do|does|never|not)\b)",
    re.IGNORECASE,
)
_NUMERIC_CONSTRAINT = re.compile(
    r"\b(?:(?P<bound>at least|at most|no more than|no fewer than|"
    r"minimum(?: of)?|maximum(?: of)?|more than|fewer than|less than|"
    r"under|below|over|above|up to|exactly)\s+)?"
    r"(?P<value>\d+(?:\.\d+)?)\s*[- ]?\s*(?P<unit>%|[a-z]+)(?![a-z0-9_])",
    re.IGNORECASE,
)
_LOWER_BOUNDS = frozenset(
    {"at least", "no fewer than", "minimum", "minimum of", "more than", "over", "above"}
)
_UPPER_BOUNDS = frozenset(
    {
        "at most",
        "no more than",
        "maximum",
        "maximum of",
        "fewer than",
        "less than",
        "under",
        "below",
        "up to",
    }
)
_STRICT_BOUNDS = frozenset(
    {"more than", "over", "above", "fewer than", "less than", "under", "below"}
)
_NUMERIC_UNITS = frozenset(
    "% percent millisecond second minute hour day word character line item point step user "
    "request record row retry attempt node endpoint megabyte gigabyte".split()
)
_NUMERIC_UNIT_ALIASES = {
    "ms": "millisecond",
    "char": "character",
    "mb": "megabyte",
    "gb": "gigabyte",
}
_NUMERIC_SEGMENT_SPLIT = re.compile(r"(?<!\d)[.!?;,\n]+|\s+\band\b\s+", re.IGNORECASE)
_LANGUAGE_DIRECTIVE = re.compile(
    r"\b(?:(?:must|should|shall|need(?:s)? to|required? to) use|use|using|"
    r"written in|implemented in|built in|(?:implement|write|build|create)"
    r"[^.!?\n]{0,50}?\bin)\s+(?P<language>typescript|javascript|python|"
    r"java(?!script)|rust|golang|go|ruby|php|kotlin|swift|sql|c\+\+|c#|js|ts)"
    r"(?![a-z0-9_+#])",
    re.IGNORECASE,
)
_AUDIENCE = (
    r"(?:users?|customers?|admins?|administrators?|members?|tenants?|employees?|"
    r"viewers?|visitors?|people|persons?|accounts?|roles?)"
)
_ALL_AUDIENCE = re.compile(rf"\b(?:all|every)\s+{_AUDIENCE}\b", re.IGNORECASE)
_ONLY_AUDIENCE = re.compile(
    rf"(?:\bonly\s+(?:to\s+)?(?:selected\s+)?{_AUDIENCE}\b|"
    rf"\b(?:limited|restricted)\s+to\s+{_AUDIENCE}\b|"
    rf"\bto\s+{_AUDIENCE}\s+only(?=\s*(?:[.!?;]|$)))",
    re.IGNORECASE,
)
_AUDIENCE_TERMS = frozenset(
    "user customer admin administrator member tenant employee viewer visitor people person "
    "account role".split()
)
_SCOPE_GENERIC_TERMS = frozenset({"access", "accessible", "avail", "available", "view"})
_GENERIC_ACTORS = frozenset(
    {"api", "application", "app", "agent", "request", "service", "system", "worker"}
)
_ACTION_ALIASES = {
    "accessible": "access",
    "access": "access",
    "available": "access",
    "avail": "access",
    "view": "access",
    "views": "access",
    "build": "implement",
    "built": "implement",
    "create": "implement",
    "implement": "implement",
    "implemented": "implement",
    "write": "implement",
    "logging": "log",
    "logs": "log",
    "needed": "require",
    "needs": "require",
    "required": "require",
    "requires": "require",
    "using": "use",
    "uses": "use",
}
_MODAL_ACTION = re.compile(
    r"\b(?:must|should|shall|can|cannot|can't|could|may|will|"
    r"(?:do|does)\s+not|never|need(?:s)?\s+to|required\s+to)\s+"
    r"(?:not\s+|never\s+)?(?:be\s+)?(?P<action>[a-z]+)",
    re.IGNORECASE,
)
_LEADING_ACTION = re.compile(
    r"^\s*(?:the\s+)?(?:never\s+|(?:do|does)\s+not\s+)?(?P<action>[a-z]+)",
    re.IGNORECASE,
)
_DECLARATIVE_ACTION = re.compile(
    r"^\s*(?:the\s+)?[a-z][a-z0-9_-]*\s+"
    r"(?:(?:is|are|does|do|will)\s+(?:not\s+)?)?(?P<action>[a-z]+)",
    re.IGNORECASE,
)


def _stem(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix in (
        "ation",
        "ting",
        "ing",
        "ies",
        "ment",
        "ness",
        "able",
        "ible",
        "ive",
        "ous",
        "ful",
        "ed",
        "er",
        "es",
        "ly",
        "al",
        "s",
    ):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> frozenset[str]:
    return frozenset(
        _stem(token)
        for token in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.lower())
        if token not in _STOP_WORDS
    )


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right))


def _normalize_action(word: str) -> str:
    lowered = word.lower()
    return _ACTION_ALIASES.get(lowered, _ACTION_ALIASES.get(_stem(lowered), _stem(lowered)))


def _action_terms(text: str) -> frozenset[str]:
    modal_match = _MODAL_ACTION.search(text)
    if modal_match is not None:
        modal_action = modal_match.group("action")
        if modal_action.lower() not in {"at", "in", "of", "on", "to"}:
            return frozenset({_normalize_action(modal_action)})
        return frozenset()

    starts_with_subject = bool(
        re.match(r"^\s*(?:the|all|every|only)\b", text, re.IGNORECASE)
    )
    patterns = (
        (_DECLARATIVE_ACTION, _LEADING_ACTION)
        if starts_with_subject
        else (_LEADING_ACTION, _DECLARATIVE_ACTION)
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        action = match.group("action")
        if action.lower() not in {"at", "in", "of", "on", "to"}:
            return frozenset({_normalize_action(action)})
    return frozenset()


def _subject_terms(
    text: str,
    actions: frozenset[str],
    excluded: frozenset[str] = frozenset(),
) -> frozenset[str]:
    return frozenset(
        term
        for term in _terms(text)
        if term not in excluded
        and term not in _GENERIC_ACTORS
        and _normalize_action(term) not in actions
    )


def _constraint_relevance(
    left_subject: frozenset[str],
    right_subject: frozenset[str],
    left_actions: frozenset[str],
    right_actions: frozenset[str],
) -> float:
    if left_actions and right_actions:
        action_score = _similarity(left_actions, right_actions)
        if action_score < 0.5:
            return 0.0
    else:
        action_score = 0.5

    if left_subject and right_subject:
        subject_score = _similarity(left_subject, right_subject)
        if subject_score < 0.5:
            return 0.0
    elif not left_subject and not right_subject:
        subject_score = 1.0
    else:
        subject_score = 0.5

    return (action_score + subject_score) / 2


def _polarity_constraints(text: str) -> list[_PolarityConstraint]:
    constraints = []
    for raw_clause in _POLARITY_SPLIT.split(text):
        clause = raw_clause.strip()
        if not clause:
            continue
        subject_text = re.sub(
            r"\b(?:after|before|within)\b.*$",
            "",
            clause,
            flags=re.IGNORECASE,
        )
        actions = _action_terms(subject_text)
        subject = _subject_terms(subject_text, actions, _POLARITY_GENERIC_TERMS)
        if subject or actions:
            constraints.append(
                _PolarityConstraint(
                    negative=bool(_NEGATIVE_POLARITY.search(clause)),
                    subject=subject,
                    actions=actions,
                    text=clause,
                )
            )
    return constraints


def _polarity_reversals(user_intent: str, specification: str) -> list[str]:
    conflicts = []
    for intent in _polarity_constraints(user_intent):
        for spec in _polarity_constraints(specification):
            if (
                intent.negative != spec.negative
                and _constraint_relevance(
                    intent.subject,
                    spec.subject,
                    intent.actions,
                    spec.actions,
                )
                >= 0.5
            ):
                conflicts.append(
                    f"polarity reversal: {intent.text!r} conflicts with {spec.text!r}"
                )
                break
    return conflicts


def _normalize_unit(raw_unit: str) -> str:
    unit = raw_unit.lower()
    if unit in _NUMERIC_UNIT_ALIASES:
        return _NUMERIC_UNIT_ALIASES[unit]
    if unit.endswith("ies"):
        return f"{unit[:-3]}y"
    if unit.endswith("s"):
        return unit[:-1]
    return unit


def _numeric_constraints(text: str) -> list[_NumericConstraint]:
    constraints = []
    for segment in _NUMERIC_SEGMENT_SPLIT.split(text):
        for match in _NUMERIC_CONSTRAINT.finditer(segment):
            qualifier = " ".join((match.group("bound") or "").lower().split())
            bound = (
                "lower"
                if qualifier in _LOWER_BOUNDS
                else "upper"
                if qualifier in _UPPER_BOUNDS
                else "exact"
            )
            unit = _normalize_unit(match.group("unit"))
            if unit not in _NUMERIC_UNITS:
                continue
            ignored = {
                *re.findall(r"[a-z]+", qualifier),
                match.group("value"),
                _stem(match.group("unit").lower()),
                _stem(unit),
            }
            actions = _action_terms(segment)
            subject = _subject_terms(segment, actions, frozenset(ignored))
            constraints.append(
                _NumericConstraint(
                    bound=bound,
                    strict=qualifier in _STRICT_BOUNDS,
                    value=float(match.group("value")),
                    unit=unit,
                    text=match.group(0),
                    subject=subject,
                    actions=actions,
                )
            )
    return constraints


def _interval(
    constraint: _NumericConstraint,
) -> tuple[float, bool, float, bool]:
    if constraint.bound == "lower":
        return constraint.value, not constraint.strict, float("inf"), False
    if constraint.bound == "upper":
        return float("-inf"), False, constraint.value, not constraint.strict
    return constraint.value, True, constraint.value, True


def _contains(interval: tuple[float, bool, float, bool], value: float) -> bool:
    low, low_inclusive, high, high_inclusive = interval
    above_low = value > low or (value == low and low_inclusive)
    below_high = value < high or (value == high and high_inclusive)
    return above_low and below_high


def _disjoint(left: _NumericConstraint, right: _NumericConstraint) -> bool:
    left_interval = _interval(left)
    right_interval = _interval(right)
    boundary_low = max(left_interval[0], right_interval[0])
    boundary_high = min(left_interval[2], right_interval[2])
    if boundary_low != boundary_high:
        return boundary_low > boundary_high
    return not (_contains(left_interval, boundary_low) and _contains(right_interval, boundary_low))


def _numeric_reversals(user_intent: str, specification: str) -> list[str]:
    intent_constraints = _numeric_constraints(user_intent)
    spec_constraints = _numeric_constraints(specification)
    conflicts = []
    for intent in intent_constraints:
        relevant = [
            spec
            for spec in spec_constraints
            if spec.unit == intent.unit
            and _constraint_relevance(
                intent.subject,
                spec.subject,
                intent.actions,
                spec.actions,
            )
            >= 0.5
        ]
        conflicting = next((spec for spec in relevant if _disjoint(intent, spec)), None)
        if conflicting is not None:
            conflicts.append(
                f"numeric reversal: {intent.text!r} conflicts with {conflicting.text!r}"
            )
    return conflicts


def _normalize_language(language: str) -> str:
    aliases = {"js": "javascript", "ts": "typescript", "golang": "go"}
    lowered = language.lower()
    return aliases.get(lowered, lowered)


def _language_clauses(text: str) -> list[str]:
    clauses: list[str] = []
    for sentence in re.split(r"(?<!\d)[.!?;\n]+", text):
        if not sentence.strip():
            continue
        parts = re.split(
            r"\band\b(?=[^.!?;\n]{0,80}\b(?:typescript|javascript|python|java|rust|"
            r"golang|go|ruby|php|kotlin|swift|sql|c\+\+|c#|js|ts)\b)",
            sentence,
            flags=re.IGNORECASE,
        )
        clauses.extend(part.strip() for part in parts if part.strip())
    return clauses


def _language_constraints(text: str) -> list[_LanguageConstraint]:
    constraints = []
    for clause in _language_clauses(text):
        actions = _action_terms(clause)
        for match in _LANGUAGE_DIRECTIVE.finditer(clause):
            language = _normalize_language(match.group("language"))
            subject = _subject_terms(
                clause,
                actions,
                frozenset({_stem(language), language}),
            )
            constraints.append(
                _LanguageConstraint(
                    language=language,
                    subject=subject,
                    actions=actions,
                    text=clause,
                )
            )
    return constraints


def _languages_compatible(requested: str, specified: str) -> bool:
    return requested == specified or (requested == "javascript" and specified == "typescript")


def _language_reversals(user_intent: str, specification: str) -> list[str]:
    conflicts = []
    for intent in _language_constraints(user_intent):
        relevant = [
            spec
            for spec in _language_constraints(specification)
            if _constraint_relevance(
                intent.subject,
                spec.subject,
                intent.actions,
                spec.actions,
            )
            >= 0.5
        ]
        conflicting = next(
            (
                spec
                for spec in relevant
                if not _languages_compatible(intent.language, spec.language)
            ),
            None,
        )
        if conflicting is not None:
            conflicts.append(
                "language reversal: requested "
                f"{intent.language} for {intent.text!r}, specified "
                f"{conflicting.language} for {conflicting.text!r}"
            )
    return conflicts


def _scope_constraints(text: str) -> list[_ScopeConstraint]:
    constraints = []
    for raw_clause in re.split(r"(?<!\d)[.!?;\n]+", text):
        clause = raw_clause.strip()
        broad = bool(_ALL_AUDIENCE.search(clause))
        restricted = bool(_ONLY_AUDIENCE.search(clause))
        if broad == restricted:
            continue
        actions = _action_terms(clause)
        subject = _subject_terms(
            clause,
            actions,
            _AUDIENCE_TERMS | _SCOPE_GENERIC_TERMS,
        )
        if subject or actions:
            constraints.append(
                _ScopeConstraint(
                    scope="broad" if broad else "restricted",
                    subject=subject,
                    actions=actions,
                    text=clause,
                )
            )
    return constraints


def _scope_reversals(user_intent: str, specification: str) -> list[str]:
    conflicts = []
    for intent in _scope_constraints(user_intent):
        for spec in _scope_constraints(specification):
            if (
                intent.scope != spec.scope
                and _constraint_relevance(
                    intent.subject,
                    spec.subject,
                    intent.actions,
                    spec.actions,
                )
                >= 0.5
            ):
                conflicts.append(f"scope reversal: {intent.text!r} conflicts with {spec.text!r}")
                break
    return conflicts


def detect_direct_contradictions(user_intent: str, specification: str) -> list[str]:
    """Return explicit contradictions between matched requirements."""
    conflicts = [
        *_polarity_reversals(user_intent, specification),
        *_numeric_reversals(user_intent, specification),
        *_language_reversals(user_intent, specification),
        *_scope_reversals(user_intent, specification),
    ]
    return list(dict.fromkeys(conflicts))
