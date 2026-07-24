"""
F2: Poor Task Decomposition Detection (MAST Taxonomy)
=====================================================

Detects when task decomposition creates:
- Subtasks that are impossible or ill-defined
- Missing dependencies between subtasks
- Circular dependencies
- Subtasks that duplicate work
- Subtasks that are too large or too granular
- Subtasks that are too vague/non-actionable

Version History:
- v1.0: Initial implementation with dependency and granularity checks
- v1.1: Improved detection for common Phase 2 failure cases:
  - Added vagueness detection for non-actionable steps
  - Added complexity estimation for too-large subtasks
  - Task-aware granularity checking (complex tasks need more steps)
- v1.2: Reduced false positives for simple direct implementations:
  - Simple task detection (doesn't require decomposition)
  - Direct implementation detection (straightforward approach indicators)
  - Prose-style output handling (not flagging casual phrasing as decomposition)
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Detector version for tracking
DETECTOR_VERSION = "2.2"
DETECTOR_NAME = "TaskDecompositionDetector"

# v1.1: Words that indicate vague/non-actionable steps
VAGUE_INDICATORS = [
    "etc",
    "various",
    "miscellaneous",
    "general",
    "overall",
    "appropriate",
    "as needed",
    "if necessary",
    "possibly",
    "might",
    "maybe",
    "could potentially",
    "consider",
    "high-level",
    "broadly",
    "generally speaking",
    "explore options",
    "look into",
    "think about",
    "strategy",
    "approach",
    "framework",  # too abstract without specifics
]

# v1.1: Words that indicate a step is too complex/broad
COMPLEXITY_INDICATORS = [
    "entire",
    "complete",
    "full",
    "all",
    "whole",
    "comprehensive",
    "end-to-end",
    "everything",
    "system",
    "platform",
    "infrastructure",
    "architecture",
    "refactor",
    "redesign",
    "rebuild",
    "rewrite",
    "migrate",
    "transform",
    "overhaul",
]

# v1.1: Words that indicate a complex task requiring more decomposition
COMPLEX_TASK_INDICATORS = [
    "system",
    "platform",
    "application",
    "service",
    "authentication",
    "authorization",
    "database",
    "migration",
    "refactor",
    "integration",
    "infrastructure",
    "deployment",
    "pipeline",
]

# v1.2: Words that indicate a simple task that doesn't require decomposition
SIMPLE_TASK_INDICATORS = [
    "simple",
    "basic",
    "quick",
    "small",
    "minor",
    "single",
    "one",
    "just",
    "only",
    "add",
    "fix",
    "update",
    "change",
    "tweak",
    "button",
    "field",
    "input",
    "label",
    "text",
    "feature",
    "function",
    "method",
    "endpoint",
]

# v1.2: Patterns indicating direct implementation (not decomposition)
DIRECT_IMPLEMENTATION_PATTERNS = [
    r"\b(?:dive\s+right\s+in|straightforward|directly|right\s+away)\b",
    r"\b(?:simple|easy|quick)\s+(?:approach|implementation|solution)\b",
    r"\b(?:here\'s|here\s+is)\s+(?:the|my|a)\s+(?:code|implementation|solution)\b",
    r"\b(?:i\'ll|let\s+me)\s+(?:just|simply|directly)\b",
    r"\b(?:no\s+need\s+(?:to|for)|without\s+(?:the\s+)?need)\b",
    r"\bstraightforward\s+approach\b",
]


class DecompositionIssue(str, Enum):
    IMPOSSIBLE_SUBTASK = "impossible_subtask"
    MISSING_DEPENDENCY = "missing_dependency"
    CIRCULAR_DEPENDENCY = "circular_dependency"
    DUPLICATE_WORK = "duplicate_work"
    WRONG_GRANULARITY = "wrong_granularity"
    MISSING_SUBTASK = "missing_subtask"
    VAGUE_SUBTASK = "vague_subtask"  # v1.1: Non-actionable steps
    OVERLY_COMPLEX = "overly_complex"  # v1.1: Steps too large/broad
    IRRELEVANT_STEP = "irrelevant_step"  # v1.7: Steps unrelated to the task
    WRONG_ORDER = "wrong_order"  # v1.7: Logical dependency violated
    MISSING_REQUIREMENT = "missing_requirement"  # v1.7: Task requirement not covered


class DecompositionSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"


@dataclass
class Subtask:
    id: str
    description: str
    dependencies: list[str]
    assigned_agent: Optional[str] = None
    estimated_complexity: Optional[str] = None


@dataclass
class DecompositionResult:
    detected: bool
    issues: list[DecompositionIssue]
    severity: DecompositionSeverity
    confidence: float
    subtask_count: int
    problematic_subtasks: list[str]
    explanation: str
    suggested_fix: Optional[str] = None
    vague_count: int = 0  # v1.1: Number of vague steps
    complex_count: int = 0  # v1.1: Number of overly complex steps
    version: str = DETECTOR_VERSION


class TaskDecompositionDetector:
    """
    Detects F2: Poor Task Decomposition - subtasks ill-defined or impossible.

    Analyzes task breakdown for logical issues, dependency problems,
    and coverage gaps.
    """

    def __init__(
        self,
        min_subtasks: int = 2,
        max_subtasks: int = 15,  # v1.7: Lowered from 20 to catch over-decomposition
        check_dependencies: bool = True,
    ):
        self.min_subtasks = min_subtasks
        self.max_subtasks = max_subtasks
        self.check_dependencies = check_dependencies

    def _parse_subtasks(self, decomposition: str) -> list[Subtask]:
        subtasks = []

        # v1.3: Enhanced patterns to handle both newline-separated and
        # single-line formats (e.g. "Step 1: ... Step 2: ... Step 3: ...").
        # Uses lookahead to stop each capture at the next marker instead of
        # greedy [^\n]+ which swallows the entire line.
        patterns = [
            # Standard numbered lists: "1. xxx" or "1) xxx"
            (r"\d+[.)]\s*", r"\d+[.)]"),
            # Bullet points: "- xxx" or "* xxx" or "• xxx"
            (r"[-•*]\s+", r"[-•*]\s"),
            # Explicit step/task labels: "Step 1: xxx" or "Task: xxx"
            (r"(?:step|task|subtask)\s*\d*[:.]\s*", r"(?:step|task|subtask)\s*\d*[:.]"),
            # Phase labels: "Phase 1: xxx" or "Phase A: xxx"
            (
                r"(?:phase|part|stage)\s*[\dA-Za-z]*[:.]\s*",
                r"(?:phase|part|stage)\s*[\dA-Za-z]*[:.]",
            ),
            # Prose phases: "first, xxx" or "then, xxx" or "finally, xxx"
            (
                r"(?:first|second|third|then|next|finally|lastly)[,:]?\s+",
                r"(?:first|second|third|then|next|finally|lastly)[,:]?\s",
            ),
        ]

        items = []
        for prefix_pat, lookahead_pat in patterns:
            # Try newline-separated first (most common)
            nl_matches = re.findall(
                r"(?:^|\n)\s*" + prefix_pat + r"([^\n]+)",
                decomposition,
                re.IGNORECASE,
            )
            if len(nl_matches) >= 2:
                items = nl_matches
                break
            # Fallback: single-line with lookahead splitting
            sl_matches = re.findall(
                prefix_pat + r"(.*?)(?=\s*" + lookahead_pat + r"|$)",
                decomposition,
                re.IGNORECASE,
            )
            sl_matches = [m.strip() for m in sl_matches if m.strip()]
            if len(sl_matches) >= 2:
                items = sl_matches
                break

        # v1.1: Fallback - try to find colon-separated phrases that look like phases
        if not items:
            # Look for "xxx: yyy" patterns that might be phase descriptions
            colon_pattern = r"(?:^|\n)\s*([A-Z][^:]+):\s*([^\n]+)"
            colon_matches = re.findall(colon_pattern, decomposition)
            if colon_matches:
                items = [f"{label}: {desc}" for label, desc in colon_matches]

        for i, item in enumerate(items):
            deps = []
            dep_patterns = [
                r"(?:after|following|requires?|depends?\s+on)\s+(?:step|task)?\s*(\d+)",
                r"(?:once|when)\s+(?:step|task)?\s*(\d+)\s+(?:is\s+)?(?:complete|done)",
            ]
            for pattern in dep_patterns:
                dep_matches = re.findall(pattern, item.lower())
                deps.extend([f"task_{int(d) - 1}" for d in dep_matches if int(d) <= i])

            subtasks.append(
                Subtask(
                    id=f"task_{i}",
                    description=item.strip(),
                    dependencies=deps,
                )
            )

        return subtasks

    def _detect_impossible_subtasks(self, subtasks: list[Subtask]) -> list[str]:
        impossible_indicators = [
            "impossible",
            "cannot",
            "unable",
            "no way",
            "infeasible",
            "undefined",
            "unknown",
            "unclear",
            "ambiguous",
            "without access",
            "no information",
            "missing",
        ]

        problematic = []
        for subtask in subtasks:
            desc_lower = subtask.description.lower()
            for indicator in impossible_indicators:
                if indicator in desc_lower:
                    problematic.append(subtask.id)
                    break

        return problematic

    def _detect_circular_dependencies(self, subtasks: list[Subtask]) -> list[tuple[str, str]]:
        circular = []

        dep_map = {st.id: set(st.dependencies) for st in subtasks}

        for task_id, deps in dep_map.items():
            for dep in deps:
                if dep in dep_map and task_id in dep_map[dep]:
                    if (dep, task_id) not in circular:
                        circular.append((task_id, dep))

        return circular

    def _detect_duplicate_work(self, subtasks: list[Subtask]) -> list[tuple[str, str]]:
        duplicates = []

        for i, st1 in enumerate(subtasks):
            words1 = set(st1.description.lower().split())
            for j, st2 in enumerate(subtasks[i + 1 :], i + 1):
                words2 = set(st2.description.lower().split())
                if not words1 or not words2:
                    continue
                overlap = len(words1 & words2) / min(len(words1), len(words2))
                if overlap > 0.7:
                    duplicates.append((st1.id, st2.id))

        return duplicates

    # v1.9: Common articles/prepositions that shouldn't be treated as nouns
    _DEP_STOP_WORDS = frozenset(
        {"a", "an", "the", "it", "its", "to", "in", "on", "at", "by", "of", "for"}
    )

    # v2.2 (Sprint 8 LL): Porter-lite suffixes for dep-noun stemming — shares
    # the `strip_suffix` helper in `text_utils.py`. Longer suffixes first so
    # "tion" is tried before "ion".
    _DEP_STEM_SUFFIXES = (
        "ation",
        "ication",
        "ization",
        "tion",
        "sion",
        "ment",
        "ness",
        "ing",
        "ed",
        "es",
        "er",
        "ly",
        "al",
    )

    @classmethod
    def _dep_stem(cls, word: str) -> str:
        """v2.2 (Sprint 8 LL): Stem domain nouns for dep matching.

        Collapses "authentication"/"authenticate" → "authentic", "storage"/"stored" → "stor",
        so "create X" output dep is satisfied by "implement X authentication" input.
        """
        from pisama_detectors.detection.text_utils import (
            strip_plural_s,
            strip_ses_plural,
            strip_suffix,
        )

        word = strip_ses_plural(word, min_length=6)
        word = strip_plural_s(word, min_length=5)
        return strip_suffix(word, cls._DEP_STEM_SUFFIXES, min_remainder=4)

    def _detect_missing_dependencies(self, subtasks: list[Subtask]) -> list[str]:
        """v2.2 (Sprint 8 LL): Detect missing explicit dependencies between subtasks.

        Fixes:
        - filter stopwords from noun extraction
        - respect sequential ordering (if producer before consumer, implicit dep is met)
        - **stem both producer and consumer nouns** so domain-noun variants match
          ("create auth" satisfies "use authentication").
        """
        missing = []

        output_indicators = ["create", "generate", "produce", "build", "write"]
        input_indicators = ["use", "read", "process", "analyze", "with"]

        # Map output-noun-STEM to (step_id, step_index). Using stems lets
        # "implement authentication" satisfy a dep for a step that "uses auth".
        id_to_idx = {st.id: i for i, st in enumerate(subtasks)}
        outputs: dict[str, tuple[str, int]] = {}
        for st in subtasks:
            desc_lower = st.description.lower()
            for indicator in output_indicators:
                if indicator in desc_lower:
                    words = desc_lower.split()
                    idx = words.index(indicator) if indicator in words else -1
                    if idx >= 0 and idx + 1 < len(words):
                        noun = words[idx + 1].strip(".,;:()[]{}\"'")
                        if noun not in self._DEP_STOP_WORDS and len(noun) > 1:
                            outputs[self._dep_stem(noun)] = (
                                st.id,
                                id_to_idx.get(st.id, -1),
                            )

        for st in subtasks:
            desc_lower = st.description.lower()
            consumer_idx = id_to_idx.get(st.id, -1)
            for indicator in input_indicators:
                if indicator in desc_lower:
                    words = desc_lower.split()
                    idx = words.index(indicator) if indicator in words else -1
                    if idx >= 0 and idx + 1 < len(words):
                        needed = words[idx + 1].strip(".,;:()[]{}\"'")
                        stem = self._dep_stem(needed)
                        if stem in outputs:
                            producer_id, producer_idx = outputs[stem]
                            if producer_id == st.id:
                                continue
                            # v1.9: Sequential ordering satisfies implicit dependency
                            if producer_idx < consumer_idx:
                                continue
                            if producer_id not in st.dependencies:
                                missing.append(st.id)

        return missing

    def _detect_vague_subtasks(self, subtasks: list[Subtask]) -> list[str]:
        """v1.1: Detect subtasks that are too vague or non-actionable."""
        vague = []

        for subtask in subtasks:
            desc_lower = subtask.description.lower()

            # Check for vague indicators
            vague_count = sum(1 for ind in VAGUE_INDICATORS if ind in desc_lower)

            # Check for lack of specific action verbs
            action_verbs = [
                "create",
                "build",
                "implement",
                "write",
                "configure",
                "set up",
                "install",
                "deploy",
                "test",
                "validate",
                "define",
                "design",
                "develop",
                "add",
                "remove",
                "update",
                "modify",
                "fix",
                "integrate",
                "connect",
                "display",
                "show",
                "render",
                "format",
                "parse",  # v1.1: UI/data actions
                "fetch",
                "load",
                "save",
                "store",
                "delete",  # v1.1: data operations
                "call",
                "invoke",
                "execute",
                "run",
                "process",  # v1.1: execution actions
                "handle",
                "index",
                "containerize",
                "evaluate",  # v1.3: from error analysis
                "filter",
                "send",
                "register",
                "check",
                "apply",
                "generate",
                "schedule",
                "compress",
                "upload",
                "scan",
                "trigger",
                "track",
                "calculate",
                "archive",
                "stream",
                "provide",
                "return",
                "verify",
                "migrate",
                "optimize",
                "monitor",
                "extract",
                "transform",
                "publish",
                "subscribe",
                "query",
                "export",
                "import",
            ]
            has_action = any(verb in desc_lower for verb in action_verbs)

            # Vague if has multiple vague indicators OR no clear action
            if vague_count >= 2 or (vague_count >= 1 and not has_action):
                vague.append(subtask.id)
            elif not has_action and len(subtask.description.split()) < 5:
                # Very short without action verb is suspicious
                vague.append(subtask.id)

        return vague

    def _detect_overly_complex_subtasks(self, subtasks: list[Subtask]) -> list[str]:
        """v1.1: Detect subtasks that are too large/broad for a single step."""
        complex_steps = []

        for subtask in subtasks:
            desc_lower = subtask.description.lower()

            # Count complexity indicators
            complexity_count = sum(1 for ind in COMPLEXITY_INDICATORS if ind in desc_lower)

            # Multiple complexity indicators suggest step is too broad
            if complexity_count >= 2:
                complex_steps.append(subtask.id)

        return complex_steps

    def _is_complex_task(self, task_description: str) -> bool:
        """v1.1: Determine if task is complex and requires thorough decomposition."""
        task_lower = task_description.lower()
        return any(ind in task_lower for ind in COMPLEX_TASK_INDICATORS)

    def _is_simple_task(self, task_description: str) -> bool:
        """v1.2: Determine if task is simple and may not need decomposition."""
        task_lower = task_description.lower()
        # Simple if has simple indicators AND no complex indicators
        has_simple = any(ind in task_lower for ind in SIMPLE_TASK_INDICATORS)
        has_complex = self._is_complex_task(task_description)
        return has_simple and not has_complex

    def _is_direct_implementation(self, output: str) -> bool:
        """v1.2: Check if output indicates direct implementation (not decomposition)."""
        output_lower = output.lower()
        for pattern in DIRECT_IMPLEMENTATION_PATTERNS:
            if re.search(pattern, output_lower):
                return True
        return False

    def _get_min_subtasks_for_task(self, task_description: str) -> int:
        """v2.2 (Sprint 8 LL): Get minimum recommended subtasks based on task complexity.

        Tightened trigger for "too_few_subtasks": now requires BOTH a complex
        indicator AND a minimum description length (>= 8 words) before forcing a
        4-step minimum. Short task descriptions like "Build auth system" should
        not be penalized for having only 3 steps — they don't carry enough
        requirement signal to warrant 4+ decomposed subtasks.
        """
        task_words = task_description.split()
        if self._is_complex_task(task_description) and len(task_words) >= 8:
            return 4  # Complex AND substantive → 4-step minimum
        return self.min_subtasks  # Default (2) minimum

    # v1.7/v1.9: Common ordering dependency pairs — (prerequisite_keywords, dependent_keywords).
    # If a step matching dependent appears BEFORE a step matching prerequisite, it's wrong order.
    # v1.9: Removed overly broad pairs (create→process, load→transform) that caused FPs
    # on legitimate orderings like "process payment" before "create order record".
    _ORDERING_DEPS = [
        ({"build", "compile", "package"}, {"deploy", "release", "ship", "publish"}),
        (
            {"validate", "verify", "check", "lint", "test"},
            {"deploy", "release", "ship", "publish", "merge"},
        ),
        (
            {"create account", "register", "sign up", "create user"},
            {"send email", "send verification", "verification email", "welcome email"},
        ),
        (
            {"test", "run tests", "unit test", "integration test"},
            {"deploy", "release", "ship", "publish"},
        ),
        (
            {"design", "plan", "architect", "define schema"},
            {"implement", "build", "code", "develop"},
        ),
    ]

    # v1.7: Words/phrases universally relevant to any software task
    _GENERIC_DEV_WORDS = {
        # Verbs
        "create",
        "setup",
        "configure",
        "deploy",
        "test",
        "build",
        "implement",
        "update",
        "migrate",
        "integrate",
        "monitor",
        "validate",
        "write",
        "design",
        "plan",
        "review",
        "define",
        "handle",
        "process",
        "refactor",
        "optimize",
        "document",
        "install",
        "check",
        "verify",
        "authenticate",
        "authorize",
        "debug",
        "fix",
        "add",
        "remove",
        "parse",
        "send",
        "receive",
        "fetch",
        "store",
        "query",
        "run",
        "execute",
        "package",
        "lint",
        "compile",
        # Nouns (common across software tasks)
        "tests",
        "testing",
        "schema",
        "database",
        "endpoint",
        "endpoints",
        "form",
        "input",
        "validation",
        "security",
        "logging",
        "error",
        "documentation",
        "middleware",
        "server",
        "client",
        "model",
        "models",
        "controller",
        "service",
        "module",
        "config",
        "configuration",
        "pipeline",
        "workflow",
        "api",
        "data",
        "table",
        "tables",
    }

    def _detect_irrelevant_steps(
        self,
        subtasks: list[Subtask],
        task_description: str,
    ) -> list[str]:
        """v1.7: Detect steps whose content is unrelated to the task description.

        Only flags steps that belong to a clearly different domain (e.g. "pick a font"
        for a JWT migration task). Requires >=2 irrelevant steps making up >=40% of
        all steps to avoid flagging legitimate auxiliary steps.
        """
        task_words = set(w for w in re.findall(r"[a-z]+", task_description.lower()) if len(w) > 3)
        if not task_words:
            return []

        # Also include 2-word phrases from task
        task_lower = task_description.lower()
        task_bigrams = set()
        twords = task_lower.split()
        for i in range(len(twords) - 1):
            task_bigrams.add(f"{twords[i]} {twords[i + 1]}")

        candidates = []
        for subtask in subtasks:
            desc_lower = subtask.description.lower()
            step_words = set(w for w in re.findall(r"[a-z]+", desc_lower) if len(w) > 3)
            if not step_words:
                continue
            # Word overlap between step and task
            overlap = len(task_words & step_words)
            ratio = overlap / min(len(task_words), len(step_words))

            # Also check bigram overlap (handles "JWT tokens", "email verification")
            bigram_hit = any(bg in desc_lower for bg in task_bigrams)

            # Generic dev words that are universally relevant
            has_generic = bool(step_words & self._GENERIC_DEV_WORDS)

            if ratio < 0.15 and not bigram_hit and not has_generic:
                candidates.append(subtask.id)

        # Only flag if irrelevant steps are a significant fraction
        if len(candidates) >= 2 and len(candidates) / len(subtasks) >= 0.4:
            return candidates
        return []

    @staticmethod
    def _has_word(keyword: str, text: str) -> bool:
        """v1.9: Word-boundary aware keyword matching (prevents 'use' matching 'users')."""
        if " " in keyword:
            return keyword in text
        return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))

    def _detect_ordering_issues(self, subtasks: list[Subtask]) -> list[str]:
        """v1.7/v1.9: Detect steps in wrong logical order based on common dependency pairs.

        v1.9: Use word-boundary matching and require same-concern overlap
        to avoid flagging unrelated steps (e.g., 'Deploy ES' before 'Configure Logstash').
        """
        if len(subtasks) < 2:
            return []

        violations = []
        for dep_keywords, post_keywords in self._ORDERING_DEPS:
            # Find the earliest step matching the prerequisite
            prereq_idx = None
            for i, st in enumerate(subtasks):
                desc = st.description.lower()
                if any(self._has_word(kw, desc) for kw in dep_keywords):
                    prereq_idx = i
                    break

            # Find the earliest step matching the dependent
            dependent_idx = None
            for i, st in enumerate(subtasks):
                desc = st.description.lower()
                if any(self._has_word(kw, desc) for kw in post_keywords):
                    dependent_idx = i
                    break

            # Violation if dependent appears before prerequisite
            if prereq_idx is not None and dependent_idx is not None:
                if dependent_idx < prereq_idx:
                    violations.append(subtasks[dependent_idx].id)

        return list(set(violations))

    # v1.8: Stop words for missing requirement detection — generic task verbs
    # and nouns that don't represent domain-specific requirements.
    _REQ_STOP_WORDS = frozenset(
        {
            "build",
            "create",
            "implement",
            "setup",
            "develop",
            "design",
            "write",
            "deploy",
            "configure",
            "install",
            "update",
            "upgrade",
            "using",
            "existing",
            "based",
            "between",
            "across",
            "current",
            "system",
            "application",
            "feature",
            "service",
            "module",
            "component",
            "project",
            "platform",
            "solution",
            "architecture",
            "infrastructure",
            "should",
            "would",
            "could",
            "about",
            "their",
            "which",
            "where",
        }
    )

    _REQ_STEM_SUFFIXES = (
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ing",
        "ed",
        "er",
        "es",
        "ly",
        "al",
        "ous",
        "ive",
        "able",
        "ible",
    )

    @staticmethod
    def _req_stem(word: str) -> str:
        # v1.9: Handle "ses" plurals (processes→process) and skip double-s words
        from pisama_detectors.detection.text_utils import (
            strip_plural_s,
            strip_ses_plural,
            strip_suffix,
        )

        word = strip_ses_plural(word, min_length=6)
        word = strip_plural_s(word, min_length=5)
        return strip_suffix(word, TaskDecompositionDetector._REQ_STEM_SUFFIXES)

    def _detect_visual_only_decomposition(
        self,
        subtasks: list[Subtask],
        task_description: str,
    ) -> bool:
        """v1.10: Detect when a functional task gets only visual/styling steps.

        Example: 'Add a shopping cart feature' decomposed as:
        - Create shopping cart icon
        - Style with CSS
        - Add hover effects
        - Make responsive

        This is missing functional steps (add items, persistence, checkout).
        """
        task_lower = task_description.lower()
        # Is this a functional task?
        functional_indicators = [
            "feature",
            "functionality",
            "system",
            "service",
            "application",
            "implement",
            "build",
            "create",
            "develop",
            "integrate",
        ]
        if not any(ind in task_lower for ind in functional_indicators):
            return False

        # Check if subtasks are all visual/styling
        visual_keywords = {
            "icon",
            "css",
            "style",
            "hover",
            "color",
            "font",
            "layout",
            "responsive",
            "animation",
            "transition",
            "theme",
            "design",
            "visual",
            "ui",
            "frontend",
            "button",
            "modal",
            "banner",
        }
        functional_keywords = {
            "add",
            "remove",
            "delete",
            "update",
            "store",
            "save",
            "fetch",
            "retrieve",
            "process",
            "validate",
            "authenticate",
            "authorize",
            "send",
            "receive",
            "parse",
            "compute",
            "calculate",
            "database",
            "api",
            "endpoint",
            "persist",
            "cache",
            "queue",
        }

        all_step_text = " ".join(st.description.lower() for st in subtasks)
        step_words = set(re.findall(r"[a-z]+", all_step_text))

        visual_hits = len(step_words & visual_keywords)
        functional_hits = len(step_words & functional_keywords)

        # Visual-only if visual hits are dominant and functional hits are very low
        if visual_hits >= 2 and functional_hits <= 1:
            return True
        return False

    def _detect_missing_requirements(
        self,
        subtasks: list[Subtask],
        task_description: str,
    ) -> list[str]:
        """v1.7/v1.8: Detect key task requirements that no step addresses.

        v1.8: Added stop word filtering for generic task verbs/nouns
        and stem matching for better word coverage.
        """
        task_lower = task_description.lower()
        requirement_chunks = re.split(r"\b(?:with|and|,)\b", task_lower)

        all_step_text = " ".join(st.description.lower() for st in subtasks)
        # Pre-compute step word stems for matching
        step_words = set(re.findall(r"[a-z]{3,}", all_step_text))
        step_stems = {self._req_stem(w) for w in step_words}

        missing = []
        for chunk in requirement_chunks:
            chunk = chunk.strip()
            if len(chunk) < 5:
                continue
            # Extract significant words, filtering stop words
            chunk_words = set(
                w
                for w in re.findall(r"[a-z]+", chunk)
                if len(w) > 4 and w not in self._REQ_STOP_WORDS
            )
            if not chunk_words:
                continue
            # Check via substring OR stem matching
            found = 0
            for w in chunk_words:
                if w in all_step_text:
                    found += 1
                elif self._req_stem(w) in step_stems:
                    found += 1
            coverage = found / len(chunk_words)
            # v1.9: Require 3+ words — 2-word chunks are too often synonym gaps
            # (e.g., "virus scanning" vs decomposition using "scan with ClamAV")
            if coverage < 0.3 and len(chunk_words) >= 3:
                missing.append(chunk.strip()[:60])

        return missing

    def detect(
        self,
        task_description: str,
        decomposition: str | list[dict[str, Any] | str],
        agent_capabilities: Optional[dict[str, list[str]]] = None,
    ) -> DecompositionResult:
        # v2.0: Handle list-format decomposition (e.g. from AgentBench/GAIA traces)
        # Sprint 8 LL: Promote ONLY list-format with 1-2 items AND an existing
        # "Step/goal/Phase" prefix would NOT have been parsed. Well-formed
        # goal-lists (e.g. BabyAI "goal 1:" / "goal 2:") already have markers
        # the parser recognizes once joined; only truly thin "Agent output..."
        # single-item traces need a synthetic "Step 1:" prefix.
        if isinstance(decomposition, list):
            raw_strs = []
            for item in decomposition:
                if isinstance(item, dict):
                    raw_strs.append(str(item.get("subtask") or item.get("description") or item))
                else:
                    raw_strs.append(str(item))
            joined = "\n".join(raw_strs)
            # Only synthesize Step-prefixes for thin single-item traces that
            # otherwise defeat parsing. Multi-item lists with intrinsic
            # markers (goal N, step N, numbered bullets) are left alone.
            needs_synth = len(raw_strs) <= 1 and not re.search(
                r"(?:^|\n)\s*(?:\d+[.)]|[-•*]\s|step\s*\d*[:.]|phase\s|goal\s*\d)",
                joined,
                re.IGNORECASE,
            )
            if needs_synth:
                decomposition = "\n".join(f"Step {i + 1}: {s}" for i, s in enumerate(raw_strs))
            else:
                decomposition = joined

        # v2.1: Recognize SWE-bench / patch-plan format:
        #   "Repository: X\n Fix: ... \n Tests to pass: [...]"
        # These are valid structured plans, not vague decompositions.
        decomp_lower = decomposition.lower()
        has_fix_ref = any(
            kw in decomp_lower for kw in ["fix:", "patch:", "apply patch", "apply fix"]
        )
        has_test_ref = any(
            kw in decomp_lower for kw in ["tests to pass", "test_", "tests/test_", "::test_"]
        )
        if has_fix_ref and has_test_ref:
            return DecompositionResult(
                detected=False,
                issues=[],
                severity=DecompositionSeverity.NONE,
                confidence=0.8,
                subtask_count=1,
                problematic_subtasks=[],
                explanation="Patch-plan decomposition with fix target and test criteria",
                vague_count=0,
                complex_count=0,
            )

        # v1.2: Check for direct implementation of simple tasks first
        is_simple = self._is_simple_task(task_description)
        is_direct = self._is_direct_implementation(decomposition)

        # v1.2: Simple task with direct implementation = no decomposition needed
        if is_simple and is_direct:
            return DecompositionResult(
                detected=False,
                issues=[],
                severity=DecompositionSeverity.NONE,
                confidence=0.8,
                subtask_count=0,
                problematic_subtasks=[],
                explanation="Simple task handled with direct implementation approach",
                vague_count=0,
                complex_count=0,
            )

        subtasks = self._parse_subtasks(decomposition)

        # v1.1/v1.2: If no subtasks found, check task complexity
        if not subtasks:
            if self._is_complex_task(task_description):
                # Complex task without structured decomposition is a failure
                return DecompositionResult(
                    detected=True,
                    issues=[DecompositionIssue.WRONG_GRANULARITY],
                    severity=DecompositionSeverity.MODERATE,
                    confidence=0.7,
                    subtask_count=0,
                    problematic_subtasks=["no_structured_decomposition"],
                    explanation=f"Complex task '{task_description[:50]}...' lacks structured decomposition",
                    suggested_fix="Provide a clear numbered or bulleted list of subtasks",
                    vague_count=0,
                    complex_count=0,
                )
            else:
                # Simple task doesn't need decomposition
                return DecompositionResult(
                    detected=False,
                    issues=[],
                    severity=DecompositionSeverity.NONE,
                    confidence=0.5,
                    subtask_count=0,
                    problematic_subtasks=[],
                    explanation="No subtasks found (simple task may not need decomposition)",
                    vague_count=0,
                    complex_count=0,
                )

        # v1.2: Even with parsed subtasks, if task is simple and approach is direct,
        # don't penalize for minimal decomposition
        if is_simple and is_direct and len(subtasks) <= 3:
            return DecompositionResult(
                detected=False,
                issues=[],
                severity=DecompositionSeverity.NONE,
                confidence=0.7,
                subtask_count=len(subtasks),
                problematic_subtasks=[],
                explanation="Simple task with straightforward approach (minimal decomposition acceptable)",
                vague_count=0,
                complex_count=0,
            )

        issues = []
        problematic = []
        vague_count = 0
        complex_count = 0

        # v1.1: Use task-aware minimum subtasks
        min_required = self._get_min_subtasks_for_task(task_description)
        if len(subtasks) < min_required:
            issues.append(DecompositionIssue.WRONG_GRANULARITY)
            problematic.append("too_few_subtasks")
        elif len(subtasks) > self.max_subtasks:
            issues.append(DecompositionIssue.WRONG_GRANULARITY)
            problematic.append("too_many_subtasks")

        impossible = self._detect_impossible_subtasks(subtasks)
        if impossible:
            issues.append(DecompositionIssue.IMPOSSIBLE_SUBTASK)
            problematic.extend(impossible)

        circular = self._detect_circular_dependencies(subtasks)
        if circular:
            issues.append(DecompositionIssue.CIRCULAR_DEPENDENCY)
            for c1, c2 in circular:
                problematic.extend([c1, c2])

        duplicates = self._detect_duplicate_work(subtasks)
        if duplicates:
            issues.append(DecompositionIssue.DUPLICATE_WORK)
            for d1, d2 in duplicates:
                problematic.extend([d1, d2])

        if self.check_dependencies:
            missing_deps = self._detect_missing_dependencies(subtasks)
            if missing_deps:
                issues.append(DecompositionIssue.MISSING_DEPENDENCY)
                problematic.extend(missing_deps)

        # v1.1: Detect vague/non-actionable subtasks
        vague = self._detect_vague_subtasks(subtasks)
        if vague:
            issues.append(DecompositionIssue.VAGUE_SUBTASK)
            problematic.extend(vague)
            vague_count = len(vague)

        # v1.1: Detect overly complex subtasks
        complex_subtasks = self._detect_overly_complex_subtasks(subtasks)
        if complex_subtasks:
            issues.append(DecompositionIssue.OVERLY_COMPLEX)
            problematic.extend(complex_subtasks)
            complex_count = len(complex_subtasks)

        # v1.7: Detect steps unrelated to the task
        irrelevant = self._detect_irrelevant_steps(subtasks, task_description)
        if irrelevant:
            issues.append(DecompositionIssue.IRRELEVANT_STEP)
            problematic.extend(irrelevant)

        # v1.7: Detect steps in wrong logical order
        wrong_order = self._detect_ordering_issues(subtasks)
        if wrong_order:
            issues.append(DecompositionIssue.WRONG_ORDER)
            problematic.extend(wrong_order)

        # v1.7: Detect task requirements not addressed by any step
        missing_reqs = self._detect_missing_requirements(subtasks, task_description)
        if missing_reqs:
            issues.append(DecompositionIssue.MISSING_REQUIREMENT)
            problematic.extend(missing_reqs)

        # v1.10: Detect functional task decomposed only with visual/styling steps
        if self._detect_visual_only_decomposition(subtasks, task_description):
            issues.append(DecompositionIssue.MISSING_REQUIREMENT)
            problematic.append("functional_task_with_visual_only_steps")

        if not issues:
            return DecompositionResult(
                detected=False,
                issues=[],
                severity=DecompositionSeverity.NONE,
                confidence=0.9,
                subtask_count=len(subtasks),
                problematic_subtasks=[],
                explanation="Task decomposition appears valid",
                vague_count=0,
                complex_count=0,
            )

        if (
            DecompositionIssue.CIRCULAR_DEPENDENCY in issues
            or DecompositionIssue.IMPOSSIBLE_SUBTASK in issues
        ):
            severity = DecompositionSeverity.SEVERE
        elif len(issues) >= 2:
            severity = DecompositionSeverity.MODERATE
        else:
            severity = DecompositionSeverity.MINOR

        # Weighted confidence: serious structural issues get higher confidence
        # than common/minor issues like vague subtask names.
        # v1.3: Lowered weights for weak signals (vague subtask, wrong granularity)
        # to better separate true structural failures from stylistic issues.
        _issue_weights = {
            DecompositionIssue.CIRCULAR_DEPENDENCY: 0.50,
            DecompositionIssue.IMPOSSIBLE_SUBTASK: 0.40,
            DecompositionIssue.IRRELEVANT_STEP: 0.35,  # v1.7
            DecompositionIssue.WRONG_ORDER: 0.30,  # v1.7
            DecompositionIssue.OVERLY_COMPLEX: 0.25,
            DecompositionIssue.MISSING_DEPENDENCY: 0.25,
            DecompositionIssue.DUPLICATE_WORK: 0.25,
            DecompositionIssue.MISSING_REQUIREMENT: 0.25,  # v1.7
            DecompositionIssue.WRONG_GRANULARITY: 0.15,
            DecompositionIssue.MISSING_SUBTASK: 0.15,
            DecompositionIssue.VAGUE_SUBTASK: 0.10,
        }
        weighted_sum = sum(_issue_weights.get(i, 0.15) for i in issues)
        confidence = min(0.95, 0.15 + weighted_sum)

        issue_names = [i.value for i in issues]
        unique_problematic = list(set(problematic))[:5]
        explanation = (
            f"Task decomposition has {len(issues)} issues: {', '.join(issue_names)}. "
            f"Affected subtasks: {', '.join(unique_problematic)}"
        )

        fixes = []
        if DecompositionIssue.CIRCULAR_DEPENDENCY in issues:
            fixes.append("Break circular dependencies by reordering subtasks")
        if DecompositionIssue.IMPOSSIBLE_SUBTASK in issues:
            fixes.append("Redefine impossible subtasks with achievable scope")
        if DecompositionIssue.DUPLICATE_WORK in issues:
            fixes.append("Merge duplicate subtasks")
        if DecompositionIssue.MISSING_DEPENDENCY in issues:
            fixes.append("Add missing dependencies between subtasks")
        # v1.1: Fixes for new issue types
        if DecompositionIssue.VAGUE_SUBTASK in issues:
            fixes.append("Replace vague steps with specific, actionable tasks")
        if DecompositionIssue.OVERLY_COMPLEX in issues:
            fixes.append("Break down overly complex steps into smaller subtasks")
        if DecompositionIssue.WRONG_GRANULARITY in issues and "too_few_subtasks" in problematic:
            fixes.append(f"Add more subtasks (minimum {min_required} recommended for this task)")
        # v1.7: Fixes for new issue types
        if DecompositionIssue.IRRELEVANT_STEP in issues:
            fixes.append("Remove steps unrelated to the task and replace with relevant ones")
        if DecompositionIssue.WRONG_ORDER in issues:
            fixes.append("Reorder steps so prerequisites come before dependent steps")
        if DecompositionIssue.MISSING_REQUIREMENT in issues:
            fixes.append("Add steps covering all stated task requirements")

        return DecompositionResult(
            detected=True,
            issues=issues,
            severity=severity,
            confidence=confidence,
            subtask_count=len(subtasks),
            problematic_subtasks=list(set(problematic)),
            explanation=explanation,
            suggested_fix="; ".join(fixes) if fixes else None,
            vague_count=vague_count,
            complex_count=complex_count,
        )

    def detect_from_trace(
        self,
        trace: dict,
    ) -> list[DecompositionResult]:
        results = []

        spans = trace.get("spans", [])
        for span in spans:
            if span.get("type") == "planning" or "plan" in span.get("name", "").lower():
                task = span.get("input", {}).get("task", "")
                output = span.get("output", {}).get("content", "")

                if task and output:
                    result = self.detect(
                        task_description=task,
                        decomposition=output,
                    )
                    if result.detected:
                        results.append(result)

        return results
