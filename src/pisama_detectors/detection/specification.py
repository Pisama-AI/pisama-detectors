"""
F1: Specification Mismatch Detection (MAST Taxonomy)
====================================================

Detects when a task specification doesn't match the user's original intent.
This occurs at the system design level when:
- The task decomposition loses critical requirements
- The specification is ambiguous or incomplete
- The task scope drifts from original intent

Version History:
- v1.0: Initial implementation
- v1.1: Improved for adversarial accuracy:
  - Code quality checks for code-related tasks
  - Numeric tolerance for approximate constraints (word counts, etc.)
  - Language mismatch detection (requested Python, got TypeScript)
  - Deprecated syntax detection
- v1.2: Phase 2 adversarial fixes:
  - Python deprecated syntax detection (cmp=, has_key, print statement)
  - JavaScript deprecated patterns (var, arguments.callee)
  - General deprecated API patterns
  - Improved word count tolerance handling
- v1.3: FPR reduction:
  - Semantic coverage using embeddings (reduces keyword false positives)
  - Reformulation detection (task restatement vs violation)
  - Stricter thresholds (coverage 0.80→0.65, ambiguity 3→4)
- v2.1: Keyword coverage improvements:
  - Sentence-level embedding comparison (not whole-text)
  - max(keyword, semantic) floor — embeddings only improve, never regress
  - Stem matching includes 3-char words (was >3, missed "run"↔"runs")
  - Added domain synonyms (catalog↔category)
- v2.9: Semantic divergence recalibration (Sprint 6 Phase BB):
  - Whole-text cosine similarity between intent and spec gates a new
    SCOPE_DRIFT flag when coverage is high but semantic distance is large
    (frequency/cadence/direction/scale mismatches).
  - Confidence now blends rule coverage with (1 - cosine_sim) to smooth
    the previously bimodal confidence distribution.
  - Existing-mismatch confidence floored by divergence so threshold grid
    search can find a stable optimum in the 0.30-0.50 band.
"""

# Detector version for tracking
DETECTOR_VERSION = "2.9"
DETECTOR_NAME = "SpecificationMismatchDetector"

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import re

logger = logging.getLogger(__name__)


class MismatchSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


class MismatchType(str, Enum):
    SCOPE_DRIFT = "scope_drift"
    MISSING_REQUIREMENT = "missing_requirement"
    AMBIGUOUS_SPEC = "ambiguous_spec"
    CONFLICTING_SPEC = "conflicting_spec"
    OVERSPECIFIED = "overspecified"


@dataclass
class SpecificationMismatchResult:
    detected: bool
    mismatch_type: Optional[MismatchType]
    severity: MismatchSeverity
    confidence: float
    requirement_coverage: float
    missing_requirements: list[str]
    ambiguous_elements: list[str]
    explanation: str
    suggested_fix: Optional[str] = None


class SpecificationMismatchDetector:
    """
    Detects F1: Specification Mismatch - task doesn't match user intent.

    Compares user intent with task specification to identify
    gaps, ambiguities, and scope drift.
    """

    # v1.1: Programming language patterns for language mismatch detection
    LANGUAGE_PATTERNS = {
        'python': (r'\bpython\b', [r'\bdef\s+\w+\s*\(', r'\bimport\s+\w+', r':\s*$', r'\bprint\s*\(']),
        'javascript': (r'\b(?:javascript|js)\b', [r'\bfunction\s+\w+', r'\bconst\s+\w+', r'\blet\s+\w+', r'=>']),
        'typescript': (r'\btypescript\b', [r':\s*\w+(?:\[\])?(?:\s*[,\)])', r'\binterface\s+\w+', r'<\w+>']),
        'java': (r'\bjava\b(?!script)', [r'\bpublic\s+class', r'\bprivate\s+\w+', r'\bvoid\s+\w+']),
        'sql': (r'\bsql\b', [r'\bSELECT\b', r'\bFROM\b', r'\bWHERE\b', r'\bINSERT\b']),
    }

    # v1.1: Code quality issues that indicate specification mismatch
    CODE_QUALITY_PATTERNS = [
        # Method reference without () - specifically common Python methods
        (r'\.(?:sort|append|extend|pop|remove|clear|reverse)\s*(?:\n|$|[^(])', "method_without_call"),
        (r'\bpass\s*$', "empty_implementation"),  # Python pass statement (placeholder)
        (r'//\s*TODO', "todo_marker"),
        (r'#\s*TODO', "todo_marker"),
        (r'raise\s+NotImplementedError', "not_implemented"),
        (r'throw\s+new\s+Error\(["\']not implemented', "not_implemented"),
        (r'\.\.\.\s*$', "ellipsis_placeholder"),  # ... as placeholder
    ]

    # v1.2: Deprecated syntax patterns by language
    DEPRECATED_SYNTAX_PATTERNS = {
        'python': [
            # Python 2 deprecated in Python 3
            (r'\bcmp\s*=', "deprecated_cmp_parameter"),  # sorted(..., cmp=func)
            (r'\.has_key\s*\(', "deprecated_has_key"),  # dict.has_key() -> use 'in'
            (r'\bprint\s+["\']', "deprecated_print_statement"),  # print "x" without parens
            (r'\braw_input\s*\(', "deprecated_raw_input"),  # raw_input() -> input()
            (r'\bexecfile\s*\(', "deprecated_execfile"),  # execfile() removed
            (r'\bxrange\s*\(', "deprecated_xrange"),  # xrange() -> range()
            (r'\.iteritems\s*\(', "deprecated_iteritems"),  # dict.iteritems() -> items()
            (r'\.iterkeys\s*\(', "deprecated_iterkeys"),  # dict.iterkeys() -> keys()
            (r'\.itervalues\s*\(', "deprecated_itervalues"),  # dict.itervalues() -> values()
            (r'\breduce\s*\((?![^)]*functools)', "deprecated_reduce"),  # reduce() needs functools
            (r'\bapply\s*\(', "deprecated_apply"),  # apply() removed
            (r'\bcoerce\s*\(', "deprecated_coerce"),  # coerce() removed
            (r'<>\s*', "deprecated_not_equal"),  # <> -> !=
            (r'\blong\s*\(', "deprecated_long"),  # long() -> int()
            (r'\bunicode\s*\(', "deprecated_unicode"),  # unicode() -> str()
        ],
        'javascript': [
            (r'\bvar\s+\w+', "deprecated_var"),  # var -> let/const (ES6+)
            (r'arguments\.callee', "deprecated_callee"),  # arguments.callee deprecated
            (r'with\s*\([^)]+\)\s*{', "deprecated_with"),  # with statement deprecated
            (r'\.substr\s*\(', "deprecated_substr"),  # substr -> substring/slice
            (r'escape\s*\(', "deprecated_escape"),  # escape() -> encodeURIComponent
            (r'unescape\s*\(', "deprecated_unescape"),  # unescape() -> decodeURIComponent
        ],
        'java': [
            (r'new\s+Date\s*\(\s*\d+\s*,', "deprecated_date_constructor"),  # Date(year,month,day) deprecated
            (r'\.getYear\s*\(', "deprecated_getYear"),  # getYear() -> getFullYear() - 1900
            (r'Thread\s*\.\s*stop\s*\(', "deprecated_thread_stop"),  # Thread.stop() deprecated
        ],
    }

    # v1.1: Numeric constraint patterns with tolerance
    NUMERIC_CONSTRAINT_PATTERNS = [
        (r'(\d+)[\s-]?word', "word_count"),
        (r'(\d+)[\s-]?character', "char_count"),
        (r'(\d+)[\s-]?line', "line_count"),
        (r'(\d+)[\s-]?item', "item_count"),
        (r'(\d+)[\s-]?point', "point_count"),
        (r'(\d+)[\s-]?step', "step_count"),
    ]

    # v1.1: Tolerance for approximate numeric constraints (percentage)
    NUMERIC_TOLERANCE = 0.10  # 10% tolerance

    # v1.3: Reformulation indicators - task restatement, not violation
    REFORMULATION_INDICATORS = [
        r"(?:task|goal|objective|mission):\s*",
        r"(?:working on|implementing|building|creating|developing)\s+",
        r"(?:as requested|as specified|as mentioned|as described|per request)",
        r"(?:proceeding with|starting|beginning|initiating)\s+",
        r"(?:the following|this involves|this includes|this requires)",
        r"(?:i will|i'll|let me|going to)\s+(?:implement|create|build|develop)",
        r"(?:here's|here is)\s+(?:the|my|a)\s+(?:plan|approach|solution)",
    ]

    # v1.4: Bonus/extra feature indicators - additions beyond requirements, not violations
    # These patterns indicate the agent completed the core task AND added extras
    BONUS_FEATURE_PATTERNS = [
        r"(?:also|additionally|as a bonus|bonus)\s+(?:added|included|implemented)",
        r"(?:extra|additional)\s+(?:feature|functionality|enhancement)",
        r"(?:i also|i've also|we also)\s+(?:added|included|implemented|created)",
        r"(?:went ahead|took the liberty)\s+(?:and|to)\s+(?:add|include)",
        r"(?:while i was at it|while working on this)",
        r"(?:for good measure|for convenience|for better ux)",
        r"(?:future-proofing|future proof|extensibility)",
        r"(?:nice to have|optional enhancement)",
        r"(?:beyond the requirements|exceeding requirements)",
    ]

    # v1.4: Benign scope expansion patterns - NOT violations
    BENIGN_EXPANSION_PATTERNS = [
        r"(?:error handling|validation|edge case|edge-case)",
        r"(?:logging|monitoring|observability)",
        r"(?:unit test|test coverage|integration test)",
        r"(?:documentation|docstring|readme|comment)",
        r"(?:type hint|typing|annotation)",
        r"(?:security|sanitization|input validation)",
    ]

    # Phase 1: Framework-specific thresholds to reduce false positives
    # v1.3: Lowered coverage thresholds (stricter about flagging)
    # v1.6: Raised thresholds across the board to reduce false positives
    FRAMEWORK_THRESHOLDS = {
        "ChatDev": {"spec_coverage": 0.60, "ambiguity_threshold": 5},
        "MetaGPT": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "AG2": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "Magentic": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "AutoGen": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "LangGraph": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
        "default": {"spec_coverage": 0.60, "ambiguity_threshold": 4},
    }

    def __init__(
        self,
        coverage_threshold: float = 0.60,  # v1.6: Raised from 0.50 — reduce FP in 50-60% zone
        ambiguity_threshold: int = 4,  # v1.3: Raised from 3
        framework: Optional[str] = None,
    ):
        self.framework = framework
        # Use framework-specific thresholds if available
        if framework and framework in self.FRAMEWORK_THRESHOLDS:
            thresholds = self.FRAMEWORK_THRESHOLDS[framework]
            self.coverage_threshold = thresholds["spec_coverage"]
            self.ambiguity_threshold = thresholds["ambiguity_threshold"]
        elif framework:
            # Unknown framework - use default
            thresholds = self.FRAMEWORK_THRESHOLDS["default"]
            self.coverage_threshold = thresholds["spec_coverage"]
            self.ambiguity_threshold = thresholds["ambiguity_threshold"]
        else:
            # No framework specified - use provided values
            self.coverage_threshold = coverage_threshold
            self.ambiguity_threshold = ambiguity_threshold

    def _extract_requirements(self, text: str) -> list[str]:
        requirements = []
        
        must_patterns = [
            r'must\s+([^.!?]+)',
            r'should\s+([^.!?]+)',
            r'need(?:s)?\s+to\s+([^.!?]+)',
            r'require(?:s|d)?\s+([^.!?]+)',
            r'has\s+to\s+([^.!?]+)',
            r'ensure\s+(?:that\s+)?([^.!?]+)',
        ]

        for pattern in must_patterns:
            matches = re.findall(pattern, text.lower())
            requirements.extend(matches)

        action_patterns = [
            r'(?:create|build|make|generate|design)\s+(?:a\s+)?([^.!?,]+)',
            r'(?:find|search|get|fetch)\s+([^.!?,]+)',
            r'(?:analyze|evaluate|assess)\s+([^.!?,]+)',
            r'(?:send|deliver|transmit)\s+([^.!?,]+)',
            # v1.5: broader action patterns
            r'(?:implement|set\s+up|configure|deploy|add)\s+(?:a\s+)?([^.!?,]+)',
            r'(?:monitor|track|log|alert)\s+([^.!?,]+)',
            r'(?:help\s+me|i\s+want\s+to|i\s+want\s+a)\s+([^.!?,]+)',
            r'(?:i\s+need)\s+(?:a\s+)?([^.!?,]+)',
            r'(?:migrate|convert|transform|clean\s+up)\s+([^.!?,]+)',
            # v2.1: additional verbs from error analysis
            r'(?:write|develop|plan|prepare|draft|outline)\s+(?:a\s+)?([^.!?,]+)',
            r'(?:process|handle|manage|run|execute)\s+([^.!?,]+)',
            r'(?:connect|integrate|link|sync(?:hronize)?)\s+([^.!?,]+)',
            r'(?:automate|schedule|optimize|improve)\s+([^.!?,]+)',
        ]

        for pattern in action_patterns:
            matches = re.findall(pattern, text.lower())
            requirements.extend(matches)
        
        return [r.strip() for r in requirements if len(r.strip()) > 3]

    def _extract_constraints(self, text: str) -> list[str]:
        constraints = []

        # v1.7: Tightened constraint patterns — require more specific phrases
        # to reduce phantom constraints from casual language.
        constraint_patterns = [
            # "do not/never" + verb (clear prohibition)
            r'(?:do not|don\'t|never|must not)\s+([^.!?,]+)',
            # "without X" (clear exclusion)
            r'without\s+(?:any\s+)?([^.!?,]+)',
            # "only/exclusively" (restriction)
            r'(?:only|exclusively)\s+([^.!?,]+)',
            # Quantitative constraints
            r'(?:at\s+(?:most|least))\s+([^.!?,]+)',
            r'(?:within|under|below|above)\s+(\d+[^.!?,]*)',
            # Temporal constraints
            r'(?:before|after|by)\s+([^.!?,]+)',
            r'(?:limit(?:ed)?\s+to)\s+([^.!?,]+)',
        ]

        for pattern in constraint_patterns:
            matches = re.findall(pattern, text.lower())
            constraints.extend(matches)

        # v1.7: Require longer constraint text (>8 chars) to filter noise
        return [c.strip() for c in constraints if len(c.strip()) > 8]

    def _is_task_reformulation(self, text: str) -> bool:
        """
        v1.3: Detect if text is reformulating/restating the task rather than violating it.

        This reduces false positives from agents that restate the task before working on it.
        """
        text_lower = text.lower()
        for pattern in self.REFORMULATION_INDICATORS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _has_bonus_features(self, text: str) -> bool:
        """
        v1.4: Detect if text indicates bonus/extra features were added.

        This reduces false positives when agents complete the task AND add extras.
        Adding error handling, tests, docs, etc. shouldn't be flagged as violations.
        """
        text_lower = text.lower()
        for pattern in self.BONUS_FEATURE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    def _has_benign_expansion(self, text: str) -> bool:
        """
        v1.4: Detect if text contains benign scope expansions.

        Patterns like adding error handling, tests, or documentation are
        generally helpful and shouldn't trigger specification mismatch.
        """
        text_lower = text.lower()
        for pattern in self.BENIGN_EXPANSION_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        return False

    # v2.7: Action stems that, when present in user intent, carry the primary
    # functional requirement. Their negation in the task specification means
    # the requirement is explicitly excluded — a severe mismatch regardless
    # of topical similarity.
    _REQUIREMENT_VERBS = frozenset({
        "alert", "alerts", "alerting", "notify", "notifies", "notifying",
        "notification", "signal", "warn", "warning",
        "check", "checks", "checking", "verify", "verifies", "verifying",
        "verification", "validate", "validates", "validating", "validation",
        "confirm", "confirms", "confirming", "confirmation",
        "review", "reviews", "reviewing", "approve", "approves", "approval",
        "moderate", "moderates", "moderating", "moderation",
        "flag", "flags", "flagging",
        "audit", "auditing",
        "authenticate", "authentication", "authorize", "authorization",
        "monitor", "monitoring", "track", "tracking", "log", "logging",
        "report", "reporting",
        "human", "manual", "supervised", "supervision",
        "resolution", "resolve", "resolves", "resolving",
        "interaction", "interact", "interacts",
    })

    # Negation patterns that signal the following phrase is excluded.
    _NEGATION_PATTERNS = [
        re.compile(r"without\s+(?:any\s+)?([a-zA-Z\s\-]{3,80})", re.IGNORECASE),
        re.compile(r"no\s+([a-zA-Z\s\-]{3,80}?)(?:,|\.|$)", re.IGNORECASE),
        re.compile(r"skip(?:s|ping)?\s+([a-zA-Z\s\-]{3,80})", re.IGNORECASE),
        re.compile(r"bypass(?:es|ing)?\s+([a-zA-Z\s\-]{3,80})", re.IGNORECASE),
        re.compile(r"ignor(?:e|es|ing)\s+([a-zA-Z\s\-]{3,80})", re.IGNORECASE),
        re.compile(r"disabl(?:e|es|ing|ed)\s+([a-zA-Z\s\-]{3,80})", re.IGNORECASE),
        re.compile(r"remov(?:e|es|ing|ed)\s+([a-zA-Z\s\-]{3,80})", re.IGNORECASE),
        re.compile(r"exclud(?:e|es|ing|ed)\s+([a-zA-Z\s\-]{3,80})", re.IGNORECASE),
        re.compile(r"instead\s+of\s+([a-zA-Z\s\-]{3,80})", re.IGNORECASE),
        re.compile(r"rather\s+than\s+([a-zA-Z\s\-]{3,80})", re.IGNORECASE),
        re.compile(r"only\s+(?:just\s+)?([a-zA-Z\s\-]{3,80}?)\s+without\s+", re.IGNORECASE),
    ]

    @staticmethod
    def _req_stem(word: str) -> str:
        """Strip common verb/noun inflections so 'alerts', 'alerting', 'alerted',
        and 'alertation' all map to a common stem. Only used for requirement-
        negation matching. "tion"/"sion" first so we don't double-strip
        ('validation' → 'validat', not 'validating' → 'valid')."""
        w = word.lower()
        for suffix in ("tion", "sion", "ing", "ed", "es", "s"):
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                return w[: -len(suffix)]
        return w

    def _detect_requirement_negation(
        self,
        user_intent: str,
        task_specification: str,
    ) -> list[str]:
        """v2.7: Return list of negation phrases in task_specification that
        strip a user-intent requirement verb. An empty list means no
        deterministic negation was found.

        Example: user wants "alerts when stock is low", spec says "without
        any alerting functionality" → returns ["alerting functionality"].
        """
        intent_tokens = {t for t in re.findall(r"[a-zA-Z]+", user_intent.lower()) if len(t) > 2}
        requirement_stems = {self._req_stem(w) for w in self._REQUIREMENT_VERBS}
        intent_stems = {self._req_stem(t) for t in intent_tokens}
        # Only negations referencing intent-verbs count; generic "without
        # errors" in a spec shouldn't fire unless the user asked for errors.
        intent_requirement_stems = intent_stems & requirement_stems
        if not intent_requirement_stems:
            return []

        negations: list[str] = []
        for pattern in self._NEGATION_PATTERNS:
            for match in pattern.finditer(task_specification):
                phrase = match.group(1).strip().lower()
                if not phrase:
                    continue
                phrase_stems = {self._req_stem(w) for w in re.findall(r"[a-zA-Z]+", phrase)}
                if phrase_stems & intent_requirement_stems:
                    negations.append(match.group(0).strip())
                    break  # One hit per pattern is enough
        return negations

    # v2.1: Words excluded from key phrase extraction — generic/vague terms
    # that don't carry specific meaning
    _PHRASE_STOP_WORDS = frozenset({
        # Standard stop words
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "has", "have", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "that", "this", "it", "its", "my",
        "our", "your", "their", "all", "each", "every", "any", "some", "no",
        "not", "so", "as", "if", "when", "than", "then", "also", "just",
        "about", "up", "out", "into", "over", "after", "before", "between",
        # Request-style verbs (user phrasing, not domain concepts)
        "need", "want", "help", "like", "wish", "make", "please",
        # Generic quality modifiers
        "simple", "basic", "good", "best", "easy", "quick", "fast",
        "better", "complete", "full", "proper", "right", "nice",
        # Generic tech nouns (too common to be specific)
        "feature", "system", "tool", "service", "application", "platform",
        "thing", "stuff", "part", "function", "ability", "capability",
        "supports", "feature", "functionality", "ability",
    })

    def _extract_key_phrases(self, text: str) -> list[str]:
        """
        v2.1: Extract important bigram phrases from text.

        Consecutive content words (both >3 chars, not in stop/generic set)
        form key phrases. These capture specific concepts like "email verification",
        "schema validation", "human agents" that individual word matching misses.
        """
        words = re.findall(r'[a-z][a-z\'-]+', text.lower())
        content_words = [
            (i, w) for i, w in enumerate(words)
            if len(w) > 3 and w not in self._PHRASE_STOP_WORDS
        ]

        phrases = []
        for j in range(len(content_words) - 1):
            idx1, w1 = content_words[j]
            idx2, w2 = content_words[j + 1]
            # Consecutive or separated by at most 1 stop word
            if idx2 - idx1 <= 2:
                phrases.append(f"{w1} {w2}")

        return phrases

    def _detect_missing_key_phrases(
        self,
        user_intent: str,
        spec_text: str,
    ) -> tuple[bool, list[str], float]:
        """
        v2.1: Detect when specific multi-word concepts from intent are absent
        from the spec, even when individual keyword overlap is high.

        Only flags phrases where BOTH words are absent from the spec — having
        one word present means the concept is at least partially covered.

        Returns (has_missing, missing_phrases, fraction_missing).
        """
        intent_phrases = self._extract_key_phrases(user_intent)
        if not intent_phrases:
            return False, [], 0.0

        spec_lower = spec_text.lower()
        spec_words = set(re.findall(r'[a-z]+', spec_lower))
        spec_stems = {self._stem(w) for w in spec_words if len(w) >= 3}

        missing = []
        for phrase in intent_phrases:
            w1, w2 = phrase.split()
            # v2.1: Both words must be absent for the phrase to be "missing"
            w1_found = (
                w1 in spec_lower or
                self._stem(w1) in spec_stems or
                any(syn in spec_lower for syn in self._expand_with_synonyms(w1))
            )
            w2_found = (
                w2 in spec_lower or
                self._stem(w2) in spec_stems or
                any(syn in spec_lower for syn in self._expand_with_synonyms(w2))
            )
            if not w1_found and not w2_found:
                missing.append(phrase)

        if not missing:
            return False, [], 0.0

        fraction = len(missing) / len(intent_phrases)
        # v2.4: Require at least 2 missing phrases AND 20% fraction to reduce
        # FPs on Q&A tasks (GAIA) where answers naturally omit some intent bigrams
        has_missing = len(missing) >= 2 and fraction >= 0.20
        return has_missing, missing, fraction

    @staticmethod
    def _extract_identifiers(text: str) -> set:
        """v2.6: Extract code identifiers (function names, class names, modules).

        Identifiers like `_separable`, `separability_matrix`, `astropy.modeling`
        are strong signals that two texts describe the same task even when
        they use completely different prose.
        """
        # snake_case identifiers
        snake_ids = set(re.findall(r'\b[a-z_][a-z0-9_]{4,}\b', text.lower()))
        # CamelCase identifiers
        camel_ids = set(re.findall(r'\b[A-Z][a-zA-Z0-9]{3,}[a-z][A-Z][a-zA-Z0-9]+\b', text))
        # Backtick-quoted identifiers (`function_name`)
        backtick_ids = set(re.findall(r'`([a-zA-Z_][a-zA-Z0-9_]+)`', text))
        # File paths and module references (foo.bar.baz)
        dotted_ids = set(re.findall(r'\b[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*){1,3}\b', text.lower()))
        # Filter common English words from snake_ids
        common = {
            'should', 'would', 'could', 'about', 'after', 'their', 'there', 'these',
            'those', 'where', 'which', 'while', 'first', 'other', 'every', 'before',
            'into', 'each', 'most', 'such', 'than', 'then', 'them', 'they', 'this',
            'that', 'with', 'when', 'will', 'have', 'been', 'being', 'were', 'from',
            'because', 'between', 'through', 'under', 'over', 'against', 'during',
            'without', 'within', 'across', 'around', 'above', 'below', 'beyond',
            'task', 'tasks', 'user', 'system', 'agent', 'agents', 'action', 'actions',
            'method', 'methods', 'result', 'results', 'output', 'outputs', 'input',
            'inputs', 'value', 'values', 'function', 'functions', 'feature', 'features',
        }
        return (snake_ids - common) | camel_ids | backtick_ids | dotted_ids

    def _identifier_overlap(self, intent: str, spec: str) -> float:
        """v2.6: Compute identifier overlap between two texts.

        High overlap means they reference the same code/concepts even if
        the prose is completely different.
        """
        intent_ids = self._extract_identifiers(intent)
        spec_ids = self._extract_identifiers(spec)
        if not intent_ids or not spec_ids:
            return 0.0
        overlap = intent_ids & spec_ids
        return len(overlap) / min(len(intent_ids), len(spec_ids))

    def _semantic_coverage(
        self,
        requirements: list[str],
        spec_text: str,
        threshold: float = 0.65,
    ) -> tuple[float, list[str]]:
        """
        v1.3/v2.1: Compute semantic coverage using embeddings.

        This improves on keyword matching by detecting semantic equivalence,
        e.g., "implement login" matches "build authentication system".

        v1.9: Split spec into sentences and compare each requirement against
        the best-matching sentence (max similarity) rather than the whole spec.
        v2.1: Keyword coverage as floor — max(semantic, keyword). Embeddings
        can only improve over keywords, never regress. This prevents the
        broken-embedding case (whole-text dilution giving 0.0) from overriding
        good keyword matches. TPs with high keyword overlap but semantic
        opposition (e.g., "escalates to human" vs "handles autonomously") are
        beyond deterministic detection and handled by the LLM judge tier.

        Falls back to keyword matching if embeddings unavailable.
        """
        if not requirements:
            return 1.0, []

        # v2.1: Always compute keyword coverage as baseline floor
        kw_coverage, kw_missing = self._compute_coverage(requirements, spec_text)

        try:
            from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder

            embedder = get_embedder()
            if not embedder:
                return kw_coverage, kw_missing

            # v1.9: Split spec into sentences for fine-grained comparison.
            # Comparing a short requirement against a long passage dilutes
            # the embedding and systematically underestimates similarity.
            spec_sentences = re.split(r'(?<!\d)[.!?]+\s+|\n+|(?:,\s*\(\d+\))', spec_text)
            spec_sentences = [s.strip() for s in spec_sentences if len(s.strip()) > 10]
            if not spec_sentences:
                spec_sentences = [spec_text[:8000]]

            # Embed all spec sentences
            sentence_embeddings = [
                embedder.encode(sent[:2000], is_query=False)
                for sent in spec_sentences
            ]

            covered = 0
            missing = []

            for req in requirements:
                req_embedding = embedder.encode(req, is_query=True)
                # v1.9: Max similarity across all spec sentences
                best_sim = max(
                    embedder.similarity(req_embedding, se)
                    for se in sentence_embeddings
                )
                if best_sim >= threshold:
                    covered += 1
                else:
                    missing.append(req)

            sem_coverage = covered / len(requirements)

            # v2.1: Take max of semantic and keyword coverage — embeddings
            # should only improve over keywords, never regress from them.
            if sem_coverage >= kw_coverage:
                return sem_coverage, missing
            else:
                return kw_coverage, kw_missing

        except Exception as e:
            logger.debug(f"Semantic coverage fallback to keywords: {e}")
            return kw_coverage, kw_missing

    def _semantic_divergence(self, user_intent: str, task_specification: str) -> float:
        """
        v2.9: Whole-text cosine similarity between intent and spec.

        Returns divergence in [0.0, 1.0] where 1.0 means completely different
        and 0.0 means identical. This is distinct from `_semantic_coverage`
        (which compares each extracted requirement to spec sentences).

        Why whole-text? FN cluster entries (frequency-mismatch,
        direction-mismatch, scale-mismatch) share vocabulary with the intent
        (so keyword coverage is ~1.0) but their overall meaning diverges.
        Whole-text embeddings capture that distributional shift.

        Returns 0.0 when embeddings are unavailable — the caller MUST treat
        0.0 as "no signal" (no flag, no confidence lift), never as "match".
        """
        if not user_intent or not task_specification:
            return 0.0
        try:
            from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder

            embedder = get_embedder()
            if not embedder:
                return 0.0
            # Cap input lengths — embeddings saturate on very long strings
            intent_vec = embedder.encode(user_intent[:4000], is_query=True)
            spec_vec = embedder.encode(task_specification[:4000], is_query=False)
            sim = embedder.similarity(intent_vec, spec_vec)
            return max(0.0, min(1.0, 1.0 - sim))
        except Exception as exc:
            logger.debug(f"Semantic divergence unavailable: {exc}")
            return 0.0

    # v2.9 calibration constants. Tuned on the combined golden datasets so
    # that divergence-flagged entries land cleanly above the calibration
    # CLI's threshold grid (which steps by 0.05 in [0.10, 0.90]). A blend
    # base of 0.45 with 0.35 range keeps flagged confidences in [0.45, 0.80]
    # — above any threshold ≤ 0.40 and below the 0.85 ceiling where the
    # legacy rule path already dominates.
    _DIVERGENCE_GATE = 0.28       # Only flag when whole-text divergence exceeds this
    _DIVERGENCE_NORM_LO = 0.25    # Normalisation window for blended confidence
    _DIVERGENCE_NORM_HI = 0.60

    def _blend_divergence_confidence(self, divergence: float, coverage: float) -> float:
        """v2.9: Map divergence into a bounded confidence in [0.30, 0.70].

        Used only when the divergence gate flags a SCOPE_DRIFT mismatch that
        the rule path missed. The blend produces a smooth middle-band signal
        instead of the historical conf=0.0/1.0 bimodality, letting the
        calibration grid search pick a stable threshold.
        """
        lo, hi = self._DIVERGENCE_NORM_LO, self._DIVERGENCE_NORM_HI
        norm = max(0.0, min(1.0, (divergence - lo) / max(1e-6, hi - lo)))
        base = 0.45 + 0.35 * norm
        # High keyword coverage + high divergence = vocabulary-match with
        # semantic-drift, the classic spec-mismatch pattern. Small bonus.
        if coverage >= 0.80 and divergence >= 0.30:
            base += 0.05
        return min(0.80, base)

    def _lift_confidence_with_divergence(self, raw_confidence: float, divergence: float) -> float:
        """v2.9: Floor an existing rule-detected confidence using divergence.

        The historical `confidence` output clustered at 0.10 or 0.95 (bimodal),
        which collapsed any threshold sweep. Lifting the floor based on
        semantic divergence spreads confidence across the middle bands without
        regressing high-confidence hits.
        """
        lo, hi = 0.15, 0.60
        norm = max(0.0, min(1.0, (divergence - lo) / max(1e-6, hi - lo)))
        floor = 0.35 + 0.30 * norm
        return max(raw_confidence, floor)

    def _detect_requested_language(self, intent: str) -> Optional[str]:
        """v1.1: Detect which programming language was requested."""
        intent_lower = intent.lower()
        for lang, (pattern, _) in self.LANGUAGE_PATTERNS.items():
            if re.search(pattern, intent_lower):
                return lang
        return None

    def _detect_output_language(self, output: str) -> Optional[str]:
        """v1.1: Detect which programming language is in the output."""
        for lang, (_, code_patterns) in self.LANGUAGE_PATTERNS.items():
            matches = sum(1 for p in code_patterns if re.search(p, output, re.MULTILINE))
            if matches >= 2:  # At least 2 language markers
                return lang
        return None

    def _check_code_quality(self, output: str) -> list[tuple]:
        """v1.1: Check for code quality issues that indicate incomplete/buggy code."""
        issues = []
        for pattern, issue_type in self.CODE_QUALITY_PATTERNS:
            matches = re.finditer(pattern, output, re.MULTILINE)
            for match in matches:
                # Get context around the match
                start = max(0, match.start() - 20)
                end = min(len(output), match.end() + 20)
                context = output[start:end].strip()
                issues.append((issue_type, match.group(), context))
        return issues

    def _check_deprecated_syntax(self, output: str, language: Optional[str]) -> list[tuple]:
        """v1.2: Check for deprecated syntax patterns in code output."""
        issues = []
        if not language or language not in self.DEPRECATED_SYNTAX_PATTERNS:
            return issues

        for pattern, issue_type in self.DEPRECATED_SYNTAX_PATTERNS[language]:
            matches = re.finditer(pattern, output, re.MULTILINE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(output), match.end() + 30)
                context = output[start:end].strip()
                issues.append((issue_type, match.group(), context))
        return issues

    def _extract_numeric_constraint(self, intent: str) -> Optional[tuple]:
        """v1.1: Extract numeric constraint from intent (e.g., '100-word')."""
        for pattern, constraint_type in self.NUMERIC_CONSTRAINT_PATTERNS:
            match = re.search(pattern, intent.lower())
            if match:
                return (int(match.group(1)), constraint_type)
        return None

    def _check_numeric_constraint(self, output: str, target: int, constraint_type: str) -> bool:
        """v1.1: Check if output meets numeric constraint within tolerance."""
        if constraint_type == "word_count":
            # Count words in output
            words = len(output.split())
            tolerance = int(target * self.NUMERIC_TOLERANCE)
            return abs(words - target) <= tolerance
        elif constraint_type == "line_count":
            lines = len(output.strip().split('\n'))
            tolerance = max(1, int(target * self.NUMERIC_TOLERANCE))
            return abs(lines - target) <= tolerance
        # For other types, be lenient
        return True

    def _is_code_task(self, intent: str) -> bool:
        """v1.1: Check if the task is code-related."""
        code_keywords = [
            r'\bcode\b', r'\bfunction\b', r'\bprogram\b', r'\bscript\b',
            r'\bimplement\b', r'\bwrite\s+(?:a\s+)?(?:python|javascript|java|sql)',
            r'\bcreate\s+(?:a\s+)?(?:function|class|method)',
        ]
        intent_lower = intent.lower()
        return any(re.search(kw, intent_lower) for kw in code_keywords)

    def _detect_ambiguities(self, text: str) -> list[str]:
        ambiguities = []
        
        vague_patterns = [
            (r'\b(some|several|many|few|various)\s+\w+', "vague quantity"),
            (r'\b(soon|later|eventually|sometime)\b', "vague timing"),
            (r'\b(good|better|best|nice|appropriate)\b', "subjective quality"),
            (r'\b(etc|and so on|and more|among others)\b', "incomplete list"),
            (r'\b(it|this|that)\b(?!\s+(?:is|are|was|will))', "ambiguous reference"),
            (r'\b(usually|typically|generally|normally)\b', "uncertain qualifier"),
            (r'\b(might|may|could|possibly|perhaps)\b', "uncertain action"),
            (r'\b(simple|easy|quick|basic)\b', "undefined complexity"),
        ]
        
        for pattern, issue_type in vague_patterns:
            if re.search(pattern, text.lower()):
                ambiguities.append(issue_type)
        
        return ambiguities

    # v2.2: Expansion phrases that indicate scope was explicitly expanded
    EXPANSION_PHRASES = [
        "also added", "additionally", "in addition to", "as a bonus",
        "plus", "on top of", "beyond what was asked", "extra feature",
        "i also", "we also", "also included", "also implemented",
    ]

    def _count_distinct_requirements(self, text: str) -> int:
        """Count distinct requirements using bullet points, numbered items,
        and 'and' conjunctions as delimiters."""
        count = 0
        # Count bullet points and numbered items
        bullet_count = len(re.findall(r'(?:^|\n)\s*(?:\d+[\.\/\)]\s|[-*\u2022]\s)', text))
        if bullet_count > 0:
            count = bullet_count
        else:
            # Count sentences that look like requirements (imperative or "must/should")
            sentences = re.split(r'[.!?]\s+|\n+', text)
            req_sentences = [
                s for s in sentences
                if len(s.strip()) > 10 and re.search(
                    r'\b(?:must|should|need|create|build|implement|add|set up|configure|ensure|include)\b',
                    s, re.IGNORECASE,
                )
            ]
            count = len(req_sentences)

        # Count "and" conjunctions joining distinct actions
        and_actions = re.findall(
            r'\b(?:and|,)\s+(?:also\s+)?(?:create|build|implement|add|set up|configure|ensure|include)\b',
            text, re.IGNORECASE,
        )
        count += len(and_actions)

        return max(1, count)

    def _detect_scope_expansion(self, user_intent: str, task_specification: str) -> Optional[str]:
        """v2.2: Detect when spec has 2x+ more requirements than intent (scope expansion).

        Also checks for explicit expansion phrases like 'also added', 'additionally', etc.

        Returns a description string if expansion detected, None otherwise.
        """
        # Check for explicit expansion phrases
        spec_lower = task_specification.lower()
        expansion_found = [
            phrase for phrase in self.EXPANSION_PHRASES
            if phrase in spec_lower
        ]

        # Count requirements in both
        intent_reqs = self._count_distinct_requirements(user_intent)
        spec_reqs = self._count_distinct_requirements(task_specification)

        # v2.5: Lowered from 3x/6 to 2x/4 — 3x was too strict, missed real scope
        # inflations (landing-page → e-commerce platform, CRM-dedupe → ML pipeline).
        # 2x multiplier balances recall gain vs medium-tier FP regression; min 4
        # keeps noise suppression for small synthetic specs.
        if spec_reqs >= intent_reqs * 2 and spec_reqs >= 4:
            return (
                f"scope_expansion: spec has {spec_reqs} requirements vs "
                f"{intent_reqs} in intent ({spec_reqs/intent_reqs:.1f}x expansion)"
            )

        # If explicit expansion phrases found AND spec has more requirements
        if expansion_found and spec_reqs > intent_reqs:
            return (
                f"scope_expansion: expansion phrases found ({', '.join(expansion_found[:3])}) "
                f"with {spec_reqs} spec requirements vs {intent_reqs} in intent"
            )

        return None

    # Synonym groups for requirement matching.
    # Each key maps to a set of semantically equivalent words.
    _SYNONYMS: dict[str, set[str]] = {
        # --- Verb synonyms ---
        "implement": {"build", "create", "develop", "make", "write", "code", "construct"},
        "build": {"implement", "create", "develop", "make", "write", "code", "construct"},
        "create": {"implement", "build", "develop", "make", "write", "generate"},
        "validate": {"verify", "check", "ensure", "confirm", "test", "assert"},
        "verify": {"validate", "check", "ensure", "confirm", "test"},
        "return": {"provide", "output", "give", "produce", "yield", "emit"},
        "provide": {"return", "output", "give", "produce", "supply"},
        "list": {"enumerate", "show", "display", "present", "outline"},
        "display": {"show", "render", "present", "list", "output"},
        "analyze": {"evaluate", "assess", "examine", "review", "inspect"},
        "update": {"modify", "change", "edit", "revise", "patch", "alter"},
        "delete": {"remove", "drop", "destroy", "clear", "purge", "erase"},
        "send": {"deliver", "transmit", "dispatch", "emit", "push"},
        "fetch": {"get", "retrieve", "load", "pull", "obtain", "query"},
        "store": {"save", "persist", "write", "record", "keep"},
        "parse": {"extract", "read", "decode", "interpret", "process"},
        "filter": {"select", "exclude", "narrow", "restrict", "screen"},
        "sort": {"order", "rank", "arrange", "organize"},
        "format": {"render", "transform", "convert", "style"},
        "handle": {"manage", "process", "deal", "address", "resolve"},
        "configure": {"setup", "config", "set", "initialize", "init"},
        "deploy": {"release", "publish", "ship", "launch", "rollout"},
        "test": {"verify", "validate", "check", "assert", "evaluate"},
        "log": {"record", "track", "audit", "trace", "monitor"},
        "notify": {"alert", "inform", "email", "message", "warn"},
        "authenticate": {"login", "auth", "verify", "authorize"},
        "register": {"signup", "enroll", "subscribe", "onboard"},
        "support": {"accept", "handle", "allow", "enable"},  # v1.3
        "resize": {"scale", "crop", "transform"},  # v1.3
        # --- Noun synonyms (v1.3: from error analysis) ---
        "purchase": {"order", "buy", "transaction", "checkout"},
        "order": {"purchase", "buy", "transaction"},
        "pipeline": {"workflow", "process", "flow", "chain"},
        "workflow": {"pipeline", "process", "flow"},
        "catalog": {"inventory", "collection", "listing", "directory", "category"},
        "category": {"catalog", "type", "classification"},
        "microservices": {"services", "microservice"},
        "services": {"microservices", "service"},
        "endpoint": {"route", "path", "url", "api"},
        "feature": {"functionality", "capability", "module"},
        "management": {"managing", "administration", "admin"},
        "processing": {"pipeline", "handling", "workflow"},
        "invoice": {"bill", "receipt"},
        "customer": {"user", "client"},
        "user": {"customer", "client", "account"},
    }

    # v1.3: Acronym expansion table
    _ACRONYMS: dict[str, set[str]] = {
        "crud": {"create", "read", "update", "delete"},
        "ci": {"continuous", "integration", "pipeline", "workflow", "actions"},
        "cd": {"continuous", "deployment", "delivery"},
        "api": {"endpoint", "route", "interface", "rest"},
        "sso": {"single", "sign", "login", "oauth", "authentication"},
        "auth": {"authentication", "authorization", "login"},
        "ui": {"interface", "frontend", "view", "page"},
        "db": {"database", "storage", "postgres", "mysql", "sql"},
        "etl": {"extract", "transform", "load"},
    }

    @staticmethod
    def _stem(word: str) -> str:
        """Minimal suffix-stripping stemmer for coverage matching."""
        # v2.1: Lowered guard from <=4 to <=3 so 4-char words like "runs"→"run"
        if len(word) <= 3:
            return word
        for suffix in ("ation", "ting", "ing", "ies", "ment", "ness",
                        "able", "ible", "ive", "ous", "ful",
                        "ed", "er", "es", "ly", "al", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[:-len(suffix)]
        return word

    def _expand_with_synonyms(self, word: str) -> set[str]:
        """Return the word plus any known synonyms, stems, and acronym expansions."""
        result = {word}
        # Direct synonyms
        result |= self._SYNONYMS.get(word, set())
        # Acronym expansion
        result |= self._ACRONYMS.get(word, set())
        # Stem-based matching: add the stem so it can match stemmed spec words
        stem = self._stem(word)
        if stem != word:
            result.add(stem)
            result |= self._SYNONYMS.get(stem, set())
        return result

    def _compute_coverage(
        self,
        intent_requirements: list[str],
        spec_text: str,
    ) -> tuple[float, list[str]]:
        if not intent_requirements:
            return 1.0, []

        spec_lower = spec_text.lower()
        # v1.3/v2.1: Pre-compute stemmed spec words for stem-based matching.
        # v2.1: Include 3-char words (was >3) so stems like "run"→"run" match
        spec_words = set(re.findall(r'[a-z]+', spec_lower))
        spec_stems = {self._stem(w) for w in spec_words if len(w) >= 3}
        covered = 0
        missing = []

        for req in intent_requirements:
            req_words = set(req.lower().split())
            req_words = {w for w in req_words if len(w) > 3}

            if not req_words:
                covered += 1
                continue

            # Check each requirement word against spec, allowing synonyms
            overlap = 0
            for w in req_words:
                expanded = self._expand_with_synonyms(w)
                # Check via substring (original) or stem match (v1.3)
                if any(syn in spec_lower for syn in expanded):
                    overlap += 1
                elif self._stem(w) in spec_stems:
                    overlap += 1

            coverage_ratio = overlap / len(req_words)

            if coverage_ratio >= 0.5:
                covered += 1
            else:
                missing.append(req)

        return covered / len(intent_requirements), missing

    # v2.3: Framework metadata patterns that inflate ambiguity/mismatch scores
    # v2.4: Expanded for ChatDev framework outputs — section headers, timestamps,
    #   file paths, and log-level prefixes cause false requirement mismatches.
    METADATA_PATTERNS = [
        r'\[Preprocessing\].*?\n',
        r'##\s+(?:Summary|Modified Files|Thinking|Code)\b.*?\n',  # ChatDev section headers
        r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}.*?\n',  # ISO timestamps (T or space separator)
        r'(?:config|src|lib)/\S+\.(?:py|js|ts|json)\b.*?\n',  # Source file paths
        r'\.py:\d+.*?\n',  # Python file references
        r'(?:INFO|DEBUG|WARNING|ERROR)\s+\S+.*?\n',  # Log level prefixes with logger name
    ]

    @staticmethod
    def _strip_metadata(text: str) -> str:
        """v2.3: Strip framework metadata that inflates ambiguity scores."""
        for pattern in SpecificationMismatchDetector.METADATA_PATTERNS:
            text = re.sub(pattern, '', text)
        return text.strip()

    def detect(
        self,
        user_intent: str,
        task_specification: str,
        original_request: Optional[str] = None,
    ) -> SpecificationMismatchResult:
        # v2.3: Strip framework metadata before analysis
        user_intent = self._strip_metadata(user_intent)
        task_specification = self._strip_metadata(task_specification)

        # v2.8: When task_specification is identical to user_intent (or a
        # pure extension of it), no mismatch is possible — the spec IS the
        # intent. This prevents false positives on sources that store the
        # same text in both fields (e.g. swebench_real where the GitHub
        # issue text is passed as both intent and spec).
        ui_norm = " ".join(user_intent.split())
        ts_norm = " ".join(task_specification.split())
        if ui_norm and (ui_norm == ts_norm or ts_norm.startswith(ui_norm)):
            return SpecificationMismatchResult(
                detected=False,
                mismatch_type=None,
                severity=MismatchSeverity.NONE,
                confidence=0.0,
                requirement_coverage=1.0,
                missing_requirements=[],
                ambiguous_elements=[],
                explanation="task_specification is identical to (or an extension of) user_intent — no mismatch possible",
            )

        # v2.5: Detect Q&A "Answer:" format for reduced sensitivity
        is_qa_answer = task_specification.strip().startswith("Answer:")

        # v2.9: Whole-text semantic divergence — used both to catch FNs the
        # rule path missed and to smooth confidence output. Computed once;
        # returns 0.0 if embeddings are unavailable (treated as no-signal).
        divergence = self._semantic_divergence(user_intent, task_specification)

        intent_requirements = self._extract_requirements(user_intent)
        intent_constraints = self._extract_constraints(user_intent)
        all_requirements = intent_requirements + intent_constraints

        # v1.3: Use semantic coverage (with fallback to keyword matching)
        coverage, missing = self._semantic_coverage(all_requirements, task_specification)

        ambiguities = self._detect_ambiguities(task_specification)

        # v1.3: Check if task_specification is a reformulation (not a violation)
        is_reformulation = self._is_task_reformulation(task_specification)

        # v1.4: Check for bonus features or benign expansions (not violations)
        has_bonus = self._has_bonus_features(task_specification)
        has_benign_expansion = self._has_benign_expansion(task_specification)

        mismatch_type = None
        detected = False
        code_issues = []

        # v1.1: Check numeric constraints with tolerance
        numeric_constraint = self._extract_numeric_constraint(user_intent)
        numeric_constraint_met = False
        if numeric_constraint:
            target, constraint_type = numeric_constraint
            if self._check_numeric_constraint(task_specification, target, constraint_type):
                # Constraint met within tolerance - this is the primary requirement
                numeric_constraint_met = True
            else:
                detected = True
                mismatch_type = MismatchType.MISSING_REQUIREMENT
                missing.append(f"{target}-{constraint_type.replace('_', ' ')}")

        # v1.1: Check code quality for code tasks
        if self._is_code_task(user_intent):
            code_issues = self._check_code_quality(task_specification)
            if code_issues:
                detected = True
                mismatch_type = MismatchType.MISSING_REQUIREMENT
                for issue_type, match, context in code_issues[:3]:
                    missing.append(f"code issue: {issue_type}")

        # v1.1: Check language mismatch (only flag if wrong language, not supersets)
        # v1.2: Also check for deprecated syntax
        requested_lang = self._detect_requested_language(user_intent)
        if requested_lang:
            # v1.2: Check deprecated syntax
            deprecated_issues = self._check_deprecated_syntax(task_specification, requested_lang)
            if deprecated_issues:
                detected = True
                mismatch_type = MismatchType.MISSING_REQUIREMENT
                for issue_type, match, context in deprecated_issues[:3]:
                    missing.append(f"deprecated syntax: {issue_type}")

            # v1.1: Check language mismatch
            output_lang = self._detect_output_language(task_specification)
            # TypeScript is superset of JavaScript - don't flag
            # Other language mismatches should be flagged
            if output_lang and output_lang != requested_lang:
                if not (requested_lang == 'javascript' and output_lang == 'typescript'):
                    detected = True
                    mismatch_type = MismatchType.MISSING_REQUIREMENT
                    missing.append(f"language mismatch: requested {requested_lang}, got {output_lang}")

        # v1.1: If numeric constraint is met, don't flag based on coverage/ambiguity
        # (the primary requirement was satisfied)
        # v1.3: Also skip if this is a reformulation of the task (not a violation)
        # v1.5: Tightened bonus/benign skip — require coverage >= 0.65 (was 0.5)
        # to prevent agents from masking missing requirements with extras
        # v2.5: Q&A "Answer:" entries have naturally lower coverage
        # (answers don't repeat the question). Only flag severe mismatches.
        # Q&A answers use completely different vocabulary from questions;
        # coverage will always be very low. Skip coverage check entirely.
        skip_coverage_check_qa = is_qa_answer

        # v2.6: Identifier overlap — when both texts reference the same
        # code identifiers (function names, module paths), they describe
        # the same task even if the prose is different.
        identifier_overlap = self._identifier_overlap(user_intent, task_specification)

        skip_coverage_check = skip_coverage_check_qa or (
            numeric_constraint_met or
            (is_reformulation and coverage >= 0.4) or  # v1.5: reformulation must still show some coverage
            (has_bonus and coverage >= 0.65) or  # Tightened: extras only skip at high coverage
            (has_benign_expansion and coverage >= 0.65) or  # Tightened: expansion only skip at high coverage
            (identifier_overlap >= 0.30)  # v2.6: 30%+ shared code identifiers = same task
        )
        if not skip_coverage_check:
            if coverage < self.coverage_threshold and not detected:
                detected = True
                mismatch_type = MismatchType.MISSING_REQUIREMENT
            elif len(ambiguities) >= self.ambiguity_threshold and not detected:
                detected = True
                mismatch_type = MismatchType.AMBIGUOUS_SPEC

        if original_request and user_intent != original_request:
            orig_reqs = self._extract_requirements(original_request)
            orig_coverage, orig_missing = self._compute_coverage(orig_reqs, task_specification)
            if orig_coverage < coverage - 0.2:
                detected = True
                mismatch_type = MismatchType.SCOPE_DRIFT
                missing.extend(orig_missing)

        # v2.1: Key phrase check — catches cases where individual keywords
        # overlap but specific multi-word concepts are absent from the spec.
        # v2.5: Skip for Q&A "Answer:" format — answers naturally use different
        # phrasing than questions, so key phrase overlap is unreliable.
        if not detected and not is_qa_answer:
            has_missing_phrases, missing_phrases, phrase_fraction = (
                self._detect_missing_key_phrases(user_intent, task_specification)
            )
            if has_missing_phrases:
                detected = True
                mismatch_type = MismatchType.MISSING_REQUIREMENT
                missing.extend(f"key phrase: {p}" for p in missing_phrases[:3])
                # Set coverage to reflect the key phrase gap
                coverage = min(coverage, 1.0 - phrase_fraction)

        # v2.2: Scope expansion detection — spec has many more requirements
        # than user intent, indicating the spec added unasked-for scope.
        if not detected:
            scope_expansion = self._detect_scope_expansion(user_intent, task_specification)
            if scope_expansion:
                detected = True
                mismatch_type = MismatchType.SCOPE_DRIFT
                missing.append(scope_expansion)

        # v2.7: Requirement negation — spec explicitly excludes a user-intent
        # requirement via phrases like "without X", "no X", "bypass X". Topical
        # embedding similarity cannot catch this (both texts are about the
        # same domain), so the deterministic negation check fires even when
        # semantic coverage is 1.0.
        if not detected and not is_qa_answer:
            negations = self._detect_requirement_negation(user_intent, task_specification)
            if negations:
                detected = True
                mismatch_type = MismatchType.MISSING_REQUIREMENT
                for neg in negations[:3]:
                    missing.append(f"requirement negated: {neg}")
                # v2.7: Requirement negation is a severe, deterministic signal
                # (not borderline) — cap coverage to ensure severe severity.
                coverage = min(coverage, 0.25)

        # v2.9: Semantic divergence gate — catches FN clusters where the
        # spec uses intent vocabulary (so coverage ≈ 1.0 and keyword checks
        # pass) but the meaning has shifted (frequency/cadence changes,
        # direction reversals, scale inflation, automation-vs-manual flips).
        # Skip for Q&A answers and for reformulations (those legitimately
        # rephrase the task using intent vocabulary).
        if (
            not detected
            and not is_qa_answer
            and not is_reformulation
            and divergence >= self._DIVERGENCE_GATE
        ):
            detected = True
            mismatch_type = MismatchType.SCOPE_DRIFT
            missing.append(
                f"semantic divergence: spec meaning diverges from intent "
                f"(divergence={divergence:.2f})"
            )

        if not detected:
            return SpecificationMismatchResult(
                detected=False,
                mismatch_type=None,
                severity=MismatchSeverity.NONE,
                confidence=coverage,
                requirement_coverage=coverage,
                missing_requirements=[],
                ambiguous_elements=[],
                explanation="Specification matches user intent",
            )

        if coverage < 0.3:
            severity = MismatchSeverity.SEVERE
        elif coverage < 0.5:
            severity = MismatchSeverity.MODERATE
        else:
            severity = MismatchSeverity.MINOR

        # Confidence scaled by severity — only high confidence for clear mismatches
        # v1.8: Gradient confidence — borderline cases get very low confidence
        # to separate from genuine violations. Coverage 0.50-0.60 maps to 0.10-0.20.
        if coverage < 0.3:
            confidence = min(0.95, 0.75 + (0.3 - coverage))
        elif coverage < 0.5:
            confidence = 0.60
        else:
            # Borderline coverage — gradient confidence, lower for higher coverage
            # 0.50 → 0.20, 0.55 → 0.15, 0.60 → 0.10
            confidence = max(0.10, 0.40 - coverage * 0.5)

        # v2.9: If the divergence gate was the sole trigger, emit a blended
        # confidence in [0.30, 0.80] instead of the coverage-derived value
        # (which would be near 0.10 because coverage is high).
        if mismatch_type == MismatchType.SCOPE_DRIFT and divergence >= self._DIVERGENCE_GATE:
            confidence = self._blend_divergence_confidence(divergence, coverage)
        else:
            # v2.9: For rule-detected mismatches, floor confidence by
            # divergence so moderately-uncertain hits don't collapse to 0.10
            # and get lost in the threshold grid search.
            confidence = self._lift_confidence_with_divergence(confidence, divergence)

        if mismatch_type == MismatchType.MISSING_REQUIREMENT:
            explanation = (
                f"Task specification missing {len(missing)} requirements from user intent. "
                f"Coverage: {coverage:.1%}"
            )
            fix = f"Add missing requirements to specification: {', '.join(missing[:3])}"
        elif mismatch_type == MismatchType.AMBIGUOUS_SPEC:
            explanation = (
                f"Task specification contains {len(ambiguities)} ambiguous elements: "
                f"{', '.join(ambiguities[:5])}"
            )
            fix = "Replace vague language with specific, measurable criteria"
        else:
            explanation = "Task specification has drifted from original user request"
            fix = "Re-align specification with original user intent"

        return SpecificationMismatchResult(
            detected=True,
            mismatch_type=mismatch_type,
            severity=severity,
            confidence=confidence,
            requirement_coverage=coverage,
            missing_requirements=missing[:10],
            ambiguous_elements=ambiguities,
            explanation=explanation,
            suggested_fix=fix,
        )

    def detect_from_trace(
        self,
        trace: dict,
    ) -> list[SpecificationMismatchResult]:
        results = []
        
        root_input = trace.get("input", {}).get("user_request", "")
        if not root_input:
            return results
        
        spans = trace.get("spans", [])
        for span in spans:
            task_spec = span.get("input", {}).get("task", "")
            if task_spec and len(task_spec) > 20:
                result = self.detect(
                    user_intent=root_input,
                    task_specification=task_spec,
                )
                if result.detected:
                    results.append(result)
        
        return results
