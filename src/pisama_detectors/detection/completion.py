"""
F14: Completion Misjudgment Detection (MAST Taxonomy)
=====================================================

Detects when an agent incorrectly determines task completion.
This occurs in task verification when:
- Agent claims task is complete when it's not
- Agent delivers partial results as final
- Agent ignores incomplete subtasks
- Agent misses success criteria

Version History:
- v1.0: Initial implementation with explicit marker detection
- v1.1: Improved detection for adversarial cases:
  - Quantitative requirement detection ("all", "every", "complete")
  - Partial completeness hedges ("most", "core", "90%")
  - Qualifier detection ("appears", "seems", "on the surface")
  - Implicit completion claims (confident delivery)
- v1.2: Enhanced for Phase 2 adversarial accuracy:
  - More implicit completion patterns (comprehensive, fully covered)
  - Detection without explicit completion claim for quant requirements
  - "Planned" and "future" work detection as incomplete
  - Task-aware detection (MVP/prototype exemptions)
- v1.3: Phase 2 adversarial fixes:
  - Fixed percentage pattern regex (80% now matches)
  - JSON completion claim detection ("status": "complete")
  - JSON incomplete indicator detection ("documented": false)
  - Numeric ratio detection (8/10, documentedEndpoints: 8, total: 10)
  - Uncertainty language detection ("lingering", "might have missed")
- v1.5: Improved recall for quantitative-requirement tasks:
  - Structural incompleteness detection (list item count, missing sections)
  - Structural incompleteness registered as ensemble signal category
  - Quantitative-requirement exemption from 2-signal ensemble gate
"""

# Detector version for tracking
DETECTOR_VERSION = "2.1"
DETECTOR_NAME = "CompletionMisjudgmentDetector"

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CompletionSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class CompletionIssueType(str, Enum):
    PREMATURE_COMPLETION = "premature_completion"
    PARTIAL_DELIVERY = "partial_delivery"
    IGNORED_SUBTASKS = "ignored_subtasks"
    MISSED_CRITERIA = "missed_criteria"
    FALSE_SUCCESS_CLAIM = "false_success_claim"
    INCOMPLETE_VERIFICATION = "incomplete_verification"


@dataclass
class CompletionIssue:
    issue_type: CompletionIssueType
    description: str
    severity: CompletionSeverity
    missing_element: Optional[str] = None
    evidence: Optional[str] = None


@dataclass
class CompletionResult:
    detected: bool
    severity: CompletionSeverity
    confidence: float
    issues: List[CompletionIssue] = field(default_factory=list)
    completion_claimed: bool = False
    actual_completion_ratio: float = 1.0
    subtasks_total: int = 0
    subtasks_completed: int = 0
    success_criteria_met: int = 0
    success_criteria_total: int = 0
    explanation: str = ""
    suggested_fix: Optional[str] = None


class CompletionMisjudgmentDetector:
    """
    Detects F14: Completion Misjudgment - agent wrongly claims completion.

    Analyzes agent output against task requirements to identify
    premature or false completion claims.
    """

    # Patterns indicating completion claims
    COMPLETION_CLAIM_PATTERNS = [
        r"\b(?:task|work|job)\s+(?:is\s+)?(?:complete|completed|done|finished)\b",
        r"\b(?:i have|i\'ve)\s+(?:completed|finished|done)\b",
        r"\b(?:successfully|successfully)\s+(?:completed|finished|done)\b",
        r"\b(?:all\s+)?(?:tasks?|steps?|items?)\s+(?:are\s+)?(?:complete|done)\b",
        r"\b(?:mission\s+accomplished|job\s+done)\b",
        r"\bhere(?:\'s| is)\s+the\s+(?:final|completed|finished)\b",
        r"\b(?:that\'s|this\s+is)\s+everything\b",
        r"\bnothing\s+(?:else|more)\s+(?:to\s+do|needed|required)\b",
        # v1.7: Tightened subject patterns — only match task/work nouns, not any word
        r"\b(?:migration|implementation|setup|integration|feature|system|module|service|component|api|app|application|build|deployment|test|testing|code|refactor|fix|update|upgrade)\s+(?:is\s+)?(?:complete|completed|done|ready|finished|implemented|working|set up|configured|live)\b",
        r"\b(?:done|complete|completed|ready|finished|implemented)\s*[!.]",
    ]

    # Patterns indicating incomplete work
    INCOMPLETE_PATTERNS = [
        (r"\bTODO\b", "todo_marker"),
        (r"\bFIXME\b", "fixme_marker"),
        (r"\bHACK\b", "hack_marker"),
        (r"\bXXX\b", "xxx_marker"),
        (r"\b(?:not\s+yet|still\s+need|remaining|pending)\b", "pending_marker"),
        (r"\b(?:placeholder|stub|dummy|mock)\b", "placeholder"),
        (r"\b(?:will\s+be|to\s+be)\s+(?:implemented|added|done)\b", "future_work"),
        (r"\.{3,}|etc\.", "ellipsis"),
        (r"\b(?:part\s+\d|step\s+\d)\s+(?:of|/)\s+\d+", "partial_progress"),
        (r"\b(?:partial|incomplete|unfinished)\b", "explicit_incomplete"),
    ]

    # Patterns indicating errors or failures
    ERROR_PATTERNS = [
        r"\b(?:error|exception|failure|failed)\b",
        r"\b(?:could not|couldn\'t|unable to|cannot)\b",
        r"\b(?:crash|crashed|crashing)\b",
        r"\b(?:bug|broken|breaking)\b",
    ]

    # v1.1: Words in task that require 100% completion
    QUANTITATIVE_REQUIREMENTS = [
        "all",
        "every",
        "each",
        "complete",
        "full",
        "entire",
        "comprehensive",
        "thorough",
        "exhaustive",
        "total",
    ]

    # v1.1: Patterns indicating partial/incomplete work (hedges)
    PARTIAL_COMPLETION_PATTERNS = [
        (r"\b(?:most|majority|mainly|primarily|largely)\b", "partial_scope"),
        (
            r"\b(?:core|main|primary|key|essential)\s+(?:functionality|features?|parts?)\b",
            "core_only",
        ),
        (r"\b(?:basic|minimal|initial|preliminary)\b", "minimal_scope"),
        (
            r"(?<!\.)\b\d{1,2}%",
            "percentage_incomplete",
        ),  # v1.3/v1.7: skip decimal parts like "99.2%"
        (r"\b(?:some|several|few|certain)\s+(?:of|aspects?|parts?|areas?)\b", "partial_coverage"),
        (r"\b(?:focus(?:ed|ing)?|priorit(?:ized?|izing))\s+on\b", "selective_focus"),
        (r"\b(?:for now|at this point|currently|at the moment)\b", "temporal_limitation"),
        (r"\b(?:happy path|common case|typical scenario)\b", "limited_coverage"),
        # v1.3: Uncertainty language indicating incomplete
        (
            r"\b(?:lingering|might have missed|could still|probably missed)\b",
            "uncertainty_incomplete",
        ),
        (
            r"\b(?:couple of|few more|some more)\s+(?:edge cases?|cases?|tests?|items?)\b",
            "acknowledged_gaps",
        ),
    ]

    # v1.1: Qualifier patterns that suggest uncertainty about completion
    QUALIFIER_PATTERNS = [
        (
            r"\b(?:appears?|seems?|looks?)\s+(?:to be\s+)?(?:complete|done|finished|solid|good)\b",
            "appearance_qualifier",
        ),
        (r"\bon the surface\b", "surface_qualifier"),
        (
            r"\b(?:should|might|could)\s+(?:be\s+)?(?:complete|working|functional)\b",
            "uncertainty_qualifier",
        ),
        (r"\b(?:believe|think|assume)\s+(?:it\'?s?|we\'?re?|this is)\b", "belief_qualifier"),
        (r"\b(?:as far as|to the best of)\b", "limited_knowledge"),
    ]

    # v1.1: Implicit completion indicators (confident delivery without caveats)
    CONFIDENT_DELIVERY_PATTERNS = [
        r"\b(?:i\'ve|we\'ve|i have|we have)\s+(?:completed|finished|done|implemented)\b",
        r"\b(?:successfully|thoroughly|fully)\s+(?:completed|implemented|deployed)\b",
        r"\b(?:the|this)\s+(?:implementation|feature|system)\s+is\s+(?:ready|complete|done)\b",
        r"\bjust wrapped up\b",
        # v1.2: Additional implicit completion patterns
        r"\bfully\s+covered\b",
        r"\b(?:all|every)\s+(?:main|major|key|critical)\s+(?:modules?|components?|features?|endpoints?)\b",
        r"\b(?:operational|ready)\s+(?:and\s+ready\s+)?for\s+(?:review|deployment|production)\b",
        r"\b(?:we\'ve|i\'ve)\s+(?:pushed|deployed|shipped|released)\b",
        r"\bin\s+place\b",  # "monitoring in place"
    ]

    # v1.2: Progress language that implies work is ongoing (NOT complete)
    # These are NOT completion claims - they indicate partial progress
    PROGRESS_NOT_COMPLETE_PATTERNS = [
        r"\bgreat progress\b",
        r"\bgood progress\b",
        r"\bmaking progress\b",
        r"\bprogressing\s+(?:well|nicely)\b",
        r"\bcoming\s+along\b",
        r"\bcomprehensive\s+(?:overview|coverage)\b",  # overview != complete
    ]

    # v1.2: Patterns indicating planned/future work (not actually complete)
    # v1.6: Expanded verb list + contraction handling + deferred_work patterns
    PLANNED_WORK_PATTERNS = [
        (r"\b(?:coverage|tests?|testing)\s+planned\b", "planned_tests"),
        (
            r"\bwill\s+(?:be\s+)?(?:added|implemented|included|covered|optimized?|fixed|handled|addressed|resolved|completed|deployed|integrated|tested|updated|refactored)\b",
            "future_work",
        ),
        (
            r"\b(?:i'll|we'll|i\s+will|we\s+will)\s+(?:add|implement|include|cover|optimize|fix|handle|address|resolve|complete|deploy|integrate|test|update|refactor)\b",
            "future_work",
        ),
        (
            r"\b(?:next|later|future)\s+(?:phase|step|iteration|sprint|release|version)\b",
            "deferred_work",
        ),
        (r"\b(?:backlog|follow[- ]up|roadmap|post[- ]launch|post[- ]release)\b", "deferred_work"),
        (r"\b(?:to\s+be|tbd|coming\s+soon)\b", "pending_work"),
        (r"\b(?:placeholder|stub|mock)\s+(?:test|coverage|implementation)?\b", "stub_work"),
    ]

    # v2.2: Vague "readiness" phrases — agent implicitly claims the system
    # is done by calling it operational/ready/configured without specifics.
    # When paired with a multi-subtask task and a short output, this is an
    # implicit completion claim even though the COMPLETION_CLAIM patterns
    # don't match (they require task nouns like "migration is complete").
    VAGUE_READINESS_PATTERNS = [
        r"\b(?:is|now)\s+(?:operational|working|running|active|live|deployed|up|functional)\b",
        r"\b(?:is|now)\s+(?:set\s+up|configured|ready|in\s+place)\b",
        r"\b(?:setup|configuration)\s+is\s+(?:complete|done|finished)\b",
        r"\bready\s+to\s+use\b",
        # v2.2b: "{noun phrase} completed successfully" / "is finished"
        # — broad completion statement without task-noun anchor.
        r"\b(?:completed|finished)\s+successfully\b",
        r"\bis\s+(?:finished|done|complete)\b",
        r"\b(?:all|every)\s+\w+\s+(?:have\s+been|has\s+been)\s+(?:identified|documented|completed|addressed|resolved|processed|handled)\b",
    ]

    # v2.2c: Explicit failure admissions — agent literally says the work
    # failed. Strong standalone signal (no completion-claim required).
    EXPLICIT_FAILURE_PATTERNS = [
        r"\bagent\s+failed\b",
        r"\b(?:did\s+not|didn\'t)\s+(?:complete|finish|succeed|manage)\b",
        r"\b(?:was\s+)?unable\s+to\s+(?:complete|finish|resolve|find|locate|identify)\b",
        r"\btask\s+(?:has\s+)?(?:failed|not\s+completed|incomplete|unfinished)\b",
        r"\b(?:goal|task)\s+(?:goal\s+)?(?:not|has\s+not)\s+(?:been\s+)?(?:completed|achieved|met)\b",
        r"\bsearch\s+an?\s+non-?sensical\s+query\b",  # benchmark-specific failure marker
        r"\baction\s+error\b",
        r"\b(?:issue|problem|error)\s+at\s+step\s+\d+\b",
        r"\bfailed\s+(?:repeatedly|multiple\s+times)\b",
    ]

    # v2.2: Blocker / external-dependency phrases — agent explicitly defers
    # part of the work to another team or external action. Strong signal
    # that the task was not completed end-to-end.
    BLOCKER_PATTERNS = [
        (
            r"\b(?:opened|filed|raised|created)\s+(?:a\s+)?(?:ticket|issue|request)\b",
            "blocker_ticket",
        ),
        (r"\bneeds?\s+(?:IT|admin|ops|devops|manager|team|approval)\b", "blocker_approval"),
        (r"\b(?:waiting|blocked)\s+(?:for|on)\b", "blocker_waiting"),
        (r"\b(?:requires|needs)\s+\w+\s+(?:access|configuration|permission)\b", "blocker_access"),
        (r"\b(?:couldn\'t|could\s+not|unable\s+to)\s+(?:finish|complete)\b", "blocker_unable"),
    ]

    # v1.8: Honest partial completion phrases — agent acknowledges work is incomplete.
    # These reduce false positives because the agent is NOT falsely claiming completion.
    HONEST_PARTIAL_PATTERNS = [
        r"\bpartially done\b",
        r"\b\w+\s+complete,?\s+\w+\s+remaining\b",
        r"\bcompleted\s+\d+\s+of\s+\d+\b",
        r"\bstill working on\b",
        r"\bTODO:\b",
        r"\bin progress\b",
        r"\bnot yet (?:complete|finished|done)\b",
        r"\bwork in progress\b",
        r"\bremaining tasks?\b",
        r"\b\d+\s+(?:tasks?|items?)\s+left\b",
    ]

    # v1.2: Task types that are intentionally scoped (avoid false positives)
    SCOPED_TASK_PATTERNS = [
        r"\b(?:prototype|mvp|poc|proof\s+of\s+concept)\b",
        r"\b(?:quick|initial|rough|first)\s+(?:draft|version|pass|attempt)\b",
        r"\b(?:minimal|basic)\s+(?:version|implementation)\b",
        r"\bv0(?:\.\d+)?\b",  # v0.1, v0.2, etc.
    ]

    # v1.3: JSON-specific completion claim patterns
    JSON_COMPLETION_PATTERNS = [
        r'"status"\s*:\s*"[^"]*(?:complete|done|finished)[^"]*"',
        r'"(?:is_complete|completed|finished|done)"\s*:\s*true',
        r'"state"\s*:\s*"[^"]*(?:complete|success)[^"]*"',
    ]

    # v1.3: JSON-specific incomplete indicator patterns
    JSON_INCOMPLETE_PATTERNS = [
        (
            r'"(?:documented|hasExamples?|completed|done|tested|covered|implemented)"\s*:\s*false',
            "json_false_flag",
        ),
        (r'"(?:missing|pending|todo|incomplete)"\s*:\s*\[', "json_missing_list"),
        (r'"(?:coverage|completion)"\s*:\s*"?\d{1,2}%"?', "json_partial_coverage"),
    ]

    # v1.3: Numeric ratio patterns (e.g., "8/10", "documented: 8, total: 10")
    # v1.7: Added (?<!,) lookbehind to prevent matching inside comma-separated
    # numbers like "15,000 of 15,000" (was matched as "000 of 15" → 0/15).
    # v2.2: Also reject matches where either number is preceded by a digit —
    # e.g. "15,000 of 15,000" should not match "00 of 15" (the comma-only
    # lookbehind fails because "0" before the "00" has no comma).
    NUMERIC_RATIO_PATTERNS = [
        (r"(?<![\d,])(\d+)\s*/\s*(?<![\d,])(\d+)(?![\d,])", "explicit_ratio"),  # 8/10
        (
            r"(?<![\d,])(\d+)\s+(?:of|out of)\s+(?<![\d,])(\d+)(?![\d,])",
            "explicit_count",
        ),  # 8 of 10
        (
            r'"(?:documented|completed|done|tested)(?:Endpoints?|Items?|Tasks?)?"\s*:\s*(\d+).*?"(?:total)(?:Endpoints?|Items?|Tasks?)?"\s*:\s*(\d+)',
            "json_ratio",
        ),
    ]

    # Success criteria extraction patterns
    CRITERIA_PATTERNS = [
        r"(?:should|must|need to|required to)\s+(.+?)(?:\.|$)",
        r"(?:criteria|requirement|goal):\s*(.+?)(?:\.|$)",
        r"(?:success|completion)\s+(?:means?|requires?):\s*(.+?)(?:\.|$)",
        r"\d+\.\s+(.+?)(?:\n|$)",  # Numbered list items
    ]

    def __init__(
        self,
        strict_mode: bool = False,
        completion_threshold: float = 0.9,
    ):
        self.strict_mode = strict_mode
        self.completion_threshold = completion_threshold

    # v1.7.1: Partial progress phrases that look like completion claims but are not.
    # "I've completed 3 of 5" should NOT count as a full completion claim.
    _PARTIAL_PROGRESS_EXEMPTIONS = re.compile(
        r"(?:i have|i've)\s+(?:completed|finished|done)\s+\d+\s+(?:of|out of|/)\s+\d+",
        re.IGNORECASE,
    )

    # v1.9: Log/test separator patterns — lines matching these are visual
    # separators or timestamp prefixes and should NOT trigger completion claims.
    # Only match pure separator lines (not log-level prefixed content).
    _LOG_PATTERNS = [
        re.compile(r"^={5,}$"),  # ===== separator lines (full line)
        re.compile(r"^-{5,}$"),  # ----- separator lines (full line)
        re.compile(r"^\[?\d{4}-\d{2}-\d{2}[T ]"),  # ISO timestamp prefix
    ]

    def _detect_completion_claim(self, text: str) -> bool:
        """Detect if agent claims task completion.

        v1.7.1: Exempt partial-progress phrases like "completed 3 of 5" that
        explicitly acknowledge incomplete work.
        v1.9: Skip log/test output lines that match LOG_PATTERNS.
        """
        for pattern in self.COMPLETION_CLAIM_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # v1.9: Check if match is on a log/test line — skip if so
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                line = text[line_start:line_end].strip()
                if any(lp.search(line) for lp in self._LOG_PATTERNS):
                    continue

                # v1.7.1: Check if this match is part of a partial-progress phrase
                # e.g., "I've completed 3 of 5 endpoints" — NOT a full claim
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 30)
                window = text[start:end]
                if self._PARTIAL_PROGRESS_EXEMPTIONS.search(window):
                    continue
                return True
        return False

    def _detect_incomplete_markers(self, text: str) -> List[tuple]:
        """Detect markers indicating incomplete work.

        v1.7: Skip 'etc.' when inside parentheses — typically abbreviating
        an example list (e.g., 'standardized state abbreviations (CA, NY, etc.)')
        rather than indicating incomplete work.
        """
        markers = []

        for pattern, marker_type in self.INCOMPLETE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # v1.7: Skip "etc." inside parenthetical examples
                if marker_type == "ellipsis" and "etc" in match.group().lower():
                    # Check if preceded by an opening paren or comma+items
                    before = text[max(0, match.start() - 60) : match.start()]
                    if "(" in before and ")" not in before[before.rfind("(") :]:
                        continue  # Inside parentheses — skip

                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                markers.append((match.group(), marker_type, context))

        return markers

    # v1.6: Context phrases that neutralize error/failure matches
    # v1.7: Added "flagged", "reported", "logged" as post-match neutralizers
    _ERROR_NEUTRALIZERS = re.compile(
        r"\b(?:fixed|resolved|addressed|handled|corrected|recovered|cleared|eliminated)\b",
        re.IGNORECASE,
    )
    _ERROR_SUFFIX_NEUTRALIZERS = re.compile(
        r"\b(?:flagged|reported|logged|noted|documented|skipped|excluded)\b",
        re.IGNORECASE,
    )
    # v1.6: Compound terms where "error"/"failure" is a domain concept, not an actual error
    _ERROR_COMPOUND_SKIP = re.compile(
        r"\berror\s+(?:codes?|handling|messages?|types?|classes?|logging|recovery|pages?|boundaries?|rates?)\b"
        r"|\bfailure\s+(?:modes?|types?|handling|recovery|rates?|conditions?)\b"
        r"|\bsuccess\s*/\s*failure\b"
        r"|\bfail(?:ed)?\s+(?:gracefully|safely|over)\b",
        re.IGNORECASE,
    )

    def _detect_errors(self, text: str) -> List[str]:
        """Detect error/failure mentions in output.

        v1.6: Context-aware — skips domain terms ('error codes', 'failure modes',
        'success/failure') and matches neutralized by resolution language
        ('fixed all errors', 'resolved the failure').
        """
        errors = []

        for pattern in self.ERROR_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 40)
                end = min(len(text), match.end() + 40)
                context = text[start:end].strip()
                # v1.6: Skip domain compound terms
                local_start = max(0, match.start() - 10)
                local_end = min(len(text), match.end() + 20)
                local_ctx = text[local_start:local_end]
                if self._ERROR_COMPOUND_SKIP.search(local_ctx):
                    continue
                # v1.6: Skip if preceded by resolution language
                prefix_start = max(0, match.start() - 50)
                prefix = text[prefix_start : match.start()]
                if self._ERROR_NEUTRALIZERS.search(prefix):
                    continue
                # v1.7: Skip if followed by exception-handling language
                suffix_end = min(len(text), match.end() + 60)
                suffix = text[match.end() : suffix_end]
                if self._ERROR_SUFFIX_NEUTRALIZERS.search(suffix):
                    continue
                errors.append(context)

        return errors

    def _has_quantitative_requirement(self, task: str) -> bool:
        """v1.1: Check if task requires 100% completion."""
        task_lower = task.lower()
        return any(req in task_lower for req in self.QUANTITATIVE_REQUIREMENTS)

    def _detect_partial_completion(self, text: str) -> List[tuple]:
        """v1.1: Detect hedges indicating partial completion."""
        indicators = []
        for pattern, indicator_type in self.PARTIAL_COMPLETION_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                indicators.append((match.group(), indicator_type, context))
        return indicators

    def _detect_qualifiers(self, text: str) -> List[tuple]:
        """v1.1: Detect uncertainty qualifiers."""
        qualifiers = []
        for pattern, qualifier_type in self.QUALIFIER_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                qualifiers.append((match.group(), qualifier_type, context))
        return qualifiers

    def _detect_confident_delivery(self, text: str) -> bool:
        """v1.1: Detect implicit completion via confident delivery.

        v1.7.1: Exempt partial-progress phrases like "completed 3 of 5".
        v1.9: Skip log/test output lines.
        """
        for pattern in self.CONFIDENT_DELIVERY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # v1.9: Skip log/test lines
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                line = text[line_start:line_end].strip()
                if any(lp.search(line) for lp in self._LOG_PATTERNS):
                    continue

                # v1.7.1: Same partial-progress exemption as _detect_completion_claim
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 30)
                window = text[start:end]
                if self._PARTIAL_PROGRESS_EXEMPTIONS.search(window):
                    continue
                return True
        return False

    def _detect_planned_work(self, text: str) -> List[tuple]:
        """v1.2: Detect indicators of planned/future work (not actually complete)."""
        indicators = []
        for pattern, indicator_type in self.PLANNED_WORK_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                indicators.append((match.group(), indicator_type, context))
        return indicators

    def _is_intentionally_scoped_task(self, task: str) -> bool:
        """v1.2: Check if task is intentionally scoped (MVP, prototype, etc.)."""
        task_lower = task.lower()
        for pattern in self.SCOPED_TASK_PATTERNS:
            if re.search(pattern, task_lower):
                return True
        return False

    def _detect_progress_language(self, text: str) -> bool:
        """v1.2: Detect progress language that implies work is ongoing (not complete)."""
        for pattern in self.PROGRESS_NOT_COMPLETE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_honest_partial_reporting(self, text: str) -> bool:
        """v1.8: Detect honest partial completion reporting.

        If the agent explicitly acknowledges work is incomplete (e.g.,
        'partially done', 'completed N of M', 'still working on'),
        this is honest partial reporting — NOT a false completion claim.
        Reduces FP by recognizing transparent status updates.

        v2.1: Don't treat TODO/WIP inside code blocks as honest reporting —
        those are evidence of incompleteness, not status updates.
        """
        # Strip code blocks before checking patterns
        text_no_code = re.sub(r"```[\s\S]*?```", "", text)
        # Also strip inline code
        text_no_code = re.sub(r"`[^`]+`", "", text_no_code)

        for pattern in self.HONEST_PARTIAL_PATTERNS:
            if re.search(pattern, text_no_code, re.IGNORECASE):
                return True
        return False

    def _detect_explicit_failure(self, text: str) -> bool:
        """v2.2c: Detect explicit failure admissions ('agent failed', 'did not complete')."""
        for pattern in self.EXPLICIT_FAILURE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_vague_readiness(self, text: str) -> bool:
        """v2.2: Detect vague readiness claims like "is operational"."""
        for pattern in self.VAGUE_READINESS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_blockers(self, text: str) -> List[tuple]:
        """v2.2: Detect explicit blockers/external-dependency deferrals."""
        found = []
        for pattern, btype in self.BLOCKER_PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                start = max(0, m.start() - 30)
                end = min(len(text), m.end() + 30)
                found.append((m.group(), btype, text[start:end].strip()))
        return found

    def _detect_json_completion_claim(self, text: str) -> bool:
        """v1.3: Detect completion claims in JSON/structured output."""
        for pattern in self.JSON_COMPLETION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_json_incomplete(self, text: str) -> List[tuple]:
        """v1.3: Detect incomplete indicators in JSON/structured output."""
        indicators = []
        for pattern, indicator_type in self.JSON_INCOMPLETE_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                indicators.append((match.group(), indicator_type, context))
        return indicators

    def _detect_numeric_ratio(self, text: str) -> Optional[tuple]:
        """v1.3: Detect numeric ratios indicating partial completion (e.g., 8/10).

        Returns:
            tuple: (completed, total, ratio) if found, None otherwise
        """
        for pattern, ratio_type in self.NUMERIC_RATIO_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                try:
                    if ratio_type == "json_ratio":
                        completed = int(match.group(1))
                        total = int(match.group(2))
                    else:
                        completed = int(match.group(1))
                        total = int(match.group(2))

                    # Only flag if ratio is less than 100%
                    if total > 0 and completed < total:
                        ratio = completed / total
                        return (completed, total, ratio)
                except (ValueError, IndexError):
                    continue
        return None

    def _extract_success_criteria(self, task: str) -> List[str]:
        """Extract success criteria from task description."""
        criteria = []

        for pattern in self.CRITERIA_PATTERNS:
            matches = re.findall(pattern, task, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match.strip()) > 5:  # Filter very short matches
                    criteria.append(match.strip())

        return criteria

    # v1.7: Suffix list for criteria stemming (shared with _stem)
    _CRITERIA_STEM_SUFFIXES = (
        "ation",
        "tion",
        "sion",
        "ment",
        "ness",
        "ity",
        "ance",
        "ence",
        "ing",
        "ed",
        "er",
        "es",
        "ly",
        "al",
        "ous",
        "ive",
        "ful",
        "ize",
        "ise",
        "able",
        "ible",
    )

    @staticmethod
    def _criteria_stem(word: str) -> str:
        """Minimal suffix strip for criteria keyword matching."""
        from pisama_detectors.detection.text_utils import strip_suffix

        if word.endswith("s") and len(word) >= 4:
            word = word[:-1]
        return strip_suffix(word, CompletionMisjudgmentDetector._CRITERIA_STEM_SUFFIXES)

    def _check_criteria_met(
        self,
        criteria: List[str],
        output: str,
    ) -> tuple[int, int, List[str]]:
        """Check how many success criteria appear to be met.

        v1.7: Uses stem matching so 'paginated' matches 'pagination',
        'functional' matches 'filters', etc.
        """
        met = 0
        unmet = []

        output_lower = output.lower()
        # Pre-compute output stems for faster matching
        output_words = set(re.findall(r"[a-z]{3,}", output_lower))
        output_stems = {self._criteria_stem(w) for w in output_words}

        for criterion in criteria:
            # Extract key terms from criterion
            words = re.findall(r"\b\w{3,}\b", criterion.lower())
            key_words = [
                w
                for w in words
                if w
                not in {
                    "should",
                    "must",
                    "need",
                    "required",
                    "that",
                    "this",
                    "will",
                    "have",
                    "been",
                    "with",
                    "from",
                    "into",
                    "the",
                    "and",
                    "for",
                    "are",
                    "all",
                    "not",
                    "can",
                }
            ]

            if not key_words:
                continue

            # Check if key words appear in output via substring OR stem match
            matches = 0
            for w in key_words:
                if w in output_lower:
                    matches += 1
                elif self._criteria_stem(w) in output_stems:
                    matches += 1
            if matches >= len(key_words) * 0.5:  # At least 50% of key words
                met += 1
            else:
                unmet.append(criterion)

        return met, len(criteria), unmet

    def _analyze_subtasks(
        self,
        subtasks: List[Dict[str, Any]],
    ) -> tuple[int, int, List[str]]:
        """Analyze subtask completion status."""
        total = len(subtasks)
        completed = 0
        incomplete = []

        for subtask in subtasks:
            status = subtask.get("status", "").lower()
            name = subtask.get("name", subtask.get("description", "Unknown"))

            if status in ["complete", "completed", "done", "success", "passed"]:
                completed += 1
            else:
                incomplete.append(name)

        return completed, total, incomplete

    def _calculate_completion_ratio(
        self,
        output: str,
        task: str,
    ) -> float:
        """Estimate completion ratio based on output analysis."""
        # Count incomplete markers
        incomplete_markers = len(self._detect_incomplete_markers(output))

        # Count errors
        errors = len(self._detect_errors(output))

        # Base ratio
        if incomplete_markers + errors == 0:
            return 1.0

        # Penalize for incomplete markers and errors
        penalty = min(0.5, incomplete_markers * 0.1 + errors * 0.15)
        return max(0.0, 1.0 - penalty)

    def _detect_structural_incompleteness(self, task: str, output: str) -> list:
        """Detect structural incompleteness in the output.

        Checks whether the output satisfies explicit numeric or sectional
        requirements mentioned in the task description.
        """
        issues: list = []

        # Check for requested list item count
        count_patterns = [
            r"(?:list|provide|give|name|identify|enumerate)\s+(\d+)",
            r"(\d+)\s+(?:items|examples|points|reasons|steps|features|recommendations)",
            r"top\s+(\d+)",
        ]

        requested_count = None
        for pattern in count_patterns:
            match = re.search(pattern, task.lower())
            if match:
                requested_count = int(match.group(1))
                break

        if requested_count and requested_count > 1:
            # Count actual list items in output
            bullet_pattern = r"(?:^|\n)\s*(?:\d+[\.\/\)]\s|[-*\u2022]\s)"
            actual_items = len(re.findall(bullet_pattern, output))

            if actual_items > 0 and actual_items < requested_count:
                ratio = actual_items / requested_count
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description=f"Task requests {requested_count} items but output contains only {actual_items}",
                        severity=CompletionSeverity.MODERATE
                        if ratio >= 0.5
                        else CompletionSeverity.SEVERE,
                        evidence=f"requested={requested_count}, actual={actual_items}, ratio={ratio:.2f}",
                    )
                )

        # v1.5: Check for missing sections mentioned in the task
        # Look for explicit section names the task requires (e.g., "include
        # introduction, methodology, results, and conclusion")
        section_patterns = [
            # "include X, Y, and Z sections"
            r"(?:include|cover|write|add|provide|create)\s+(?:a\s+)?(?:sections?\s+(?:on|for|about)\s+)?(.+?)(?:\s+section[s]?)?(?:\.|$)",
            # "sections: X, Y, Z"
            r"sections?\s*:\s*(.+?)(?:\.|$)",
        ]

        task_lower = task.lower()
        output_lower = output.lower()

        for pattern in section_patterns:
            match = re.search(pattern, task_lower)
            if match:
                # Extract potential section names from comma/and-separated list
                section_text = match.group(1)
                # Split on commas and "and"
                parts = re.split(r",\s*|\s+and\s+", section_text)
                # Filter to reasonable section names (2-40 chars, no full sentences)
                section_names = [
                    p.strip().strip("\"'")
                    for p in parts
                    if 2 <= len(p.strip()) <= 40 and " is " not in p and " are " not in p
                ]
                if len(section_names) >= 2:
                    missing = [s for s in section_names if s not in output_lower]
                    if missing and len(missing) < len(section_names):
                        # Some sections present, some missing — partial delivery
                        issues.append(
                            CompletionIssue(
                                issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                                description=f"Task mentions sections but {len(missing)} missing from output: {', '.join(missing[:3])}",
                                severity=CompletionSeverity.MODERATE,
                                evidence=f"required={len(section_names)}, missing={len(missing)}",
                            )
                        )
                    break  # Only use the first matching pattern

        return issues

    def _detect_enumerated_coverage(self, task: str, output: str) -> list:
        """Detect when task enumerates specific items (A, B, and C) but output
        only covers a subset."""
        issues: list = []

        # Extract enumerated items from task:
        #   "with Google, GitHub, and Microsoft"
        #   "including login, logout, and token refresh"
        #   "for CSV, JSON, and PDF formats"
        enum_patterns = [
            r"(?:with|including|for|supporting|using|like)\s+(.+?)(?:\.|$)",
            r"(?:implement|build|create|add|set up)\s+(.+?)(?:\.|$)",
        ]

        task_lower = task.lower()
        output_lower = output.lower()

        for pattern in enum_patterns:
            match = re.search(pattern, task_lower)
            if not match:
                continue
            fragment = match.group(1)
            # Split on commas and "and"/"or"
            parts = re.split(r",\s*|\s+(?:and|or)\s+", fragment)
            # v2.2: Strip residual leading "and "/"or " conjunctions that
            # survive comma-first splits (e.g. "disk, and request metrics"
            # → ["disk", "and request metrics"] → ["disk", "request metrics"]).
            cleaned = []
            for p in parts:
                s = p.strip()
                s = re.sub(r"^(?:and|or)\s+", "", s)
                cleaned.append(s)
            # Filter to meaningful items (2+ chars, not common filler words)
            items = [
                s
                for s in cleaned
                if len(s) >= 2
                and s
                not in (
                    "the",
                    "a",
                    "an",
                    "all",
                    "each",
                    "every",
                    "other",
                )
            ]
            if len(items) >= 3:
                found = sum(1 for item in items if item in output_lower)
                if found < len(items):
                    missing = [i for i in items if i not in output_lower]
                    severity = (
                        CompletionSeverity.SEVERE if found == 0 else CompletionSeverity.MODERATE
                    )
                    issues.append(
                        CompletionIssue(
                            issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                            description=(
                                f"Task enumerates {len(items)} items but output "
                                f"only covers {found}: missing {', '.join(missing[:3])}"
                            ),
                            severity=severity,
                            evidence=f"items={items}, found={found}",
                        )
                    )
                    break  # One match is enough

        return issues

    # -----------------------------------------------------------------
    # Subtask name → status inference
    # -----------------------------------------------------------------
    _STEM_SUFFIXES = (
        "tion",
        "ing",
        "ment",
        "ness",
        "ity",
        "ed",
        "er",
        "es",
        "ly",
        "al",
        "ous",
        "ive",
        "ful",
        "ize",
        "ise",
    )

    @staticmethod
    def _stem_word(word: str) -> str:
        """Minimal suffix strip for subtask inference matching."""
        from pisama_detectors.detection.text_utils import strip_suffix

        if word.endswith("s") and len(word) >= 4:
            word = word[:-1]
        return strip_suffix(word, CompletionMisjudgmentDetector._STEM_SUFFIXES)

    @staticmethod
    def infer_subtask_status(
        subtask_names: List[str],
        agent_output: str,
    ) -> List[Dict[str, Any]]:
        """Convert string subtask names to dicts with inferred status.

        Infers whether each subtask was completed by checking the agent
        output for matching words (via substring and stem overlap), with
        negation-context detection to avoid false completions.

        Args:
            subtask_names: Plain string subtask names.
            agent_output: The agent's response to check against.

        Returns:
            List of ``{"name": ..., "status": "completed"|"pending"}`` dicts.
        """
        output_lower = agent_output.lower()
        output_words = set(re.findall(r"[a-z]{3,}", output_lower))
        stem = CompletionMisjudgmentDetector._stem_word
        output_stems = {stem(w) for w in output_words}

        result: List[Dict[str, Any]] = []
        for name in subtask_names:
            parts = re.split(r"[_\s\-]+", name.lower())
            name_words = {w for w in parts if len(w) >= 3}
            overlap = sum(1 for w in name_words if w in output_lower or stem(w) in output_stems)
            ratio = overlap / max(len(name_words), 1)
            status = "completed" if ratio >= 0.5 else "pending"
            # Check for negation near matched words
            if status == "completed" and overlap > 0:
                for w in name_words:
                    pos = output_lower.find(w)
                    if pos >= 0:
                        ctx = output_lower[max(0, pos - 20) : pos]
                        if re.search(r"\bnot\s+(?:yet|currently)?\b", ctx):
                            status = "pending"
                            break
            result.append({"name": name, "status": status})
        return result

    def detect(
        self,
        task: str,
        agent_output: str,
        subtasks: Optional[List[Any]] = None,
        success_criteria: Optional[List[str]] = None,
        expected_outputs: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> CompletionResult:
        """
        Detect completion misjudgment.

        Args:
            task: The task the agent was asked to perform
            agent_output: The agent's response/output
            subtasks: List of subtasks — either dicts with 'name'/'status'
                keys, or plain strings (auto-inferred via agent_output)
            success_criteria: List of success criteria to check
            expected_outputs: List of expected output elements
            context: Additional context

        Returns:
            CompletionResult with detection outcome
        """
        # v2.2: Normalize success_criteria — some callers pass a plain string
        # instead of a list. Iterating a string yields per-character "criteria"
        # which broke criteria matching (criteria_met=0 always). For SHORT
        # strings (typical SDK usage) wrap as a single-element list. For long
        # benchmark-style strings (ARB/SWE-bench with an entire task echoed
        # into the criterion) wrap and also flag for the external-grader path.
        if isinstance(success_criteria, str):
            success_criteria = [success_criteria] if success_criteria.strip() else None
        if isinstance(expected_outputs, str):
            expected_outputs = [expected_outputs] if expected_outputs.strip() else None

        # Auto-convert string subtask names to dicts
        subtasks_were_inferred = False
        if subtasks:
            string_names = [s for s in subtasks if isinstance(s, str)]
            if string_names:
                dict_items = [s for s in subtasks if isinstance(s, dict)]
                inferred = self.infer_subtask_status(string_names, agent_output)
                subtasks = dict_items + inferred
                # v2.2: If >= 50% of subtasks were name-inferred, treat the
                # inference as a weak signal (heuristic word-overlap is noisy).
                if len(string_names) >= len(subtasks) * 0.5:
                    subtasks_were_inferred = True
        issues = []

        # Check if agent claims completion (explicit or implicit)
        completion_claimed = self._detect_completion_claim(agent_output)

        # v1.1: Check for implicit completion via confident delivery
        confident_delivery = self._detect_confident_delivery(agent_output)
        if confident_delivery and not completion_claimed:
            completion_claimed = True  # Treat confident delivery as implicit claim

        # v1.3: Check for JSON-specific completion claims
        json_completion = self._detect_json_completion_claim(agent_output)
        if json_completion and not completion_claimed:
            completion_claimed = True  # Treat JSON completion status as claim

        # v2.2: Vague readiness phrase ("is operational", "ready to use") on
        # a short output covering a multi-subtask task — treat as implicit
        # completion claim so the rest of the pipeline can evaluate it.
        vague_readiness = self._detect_vague_readiness(agent_output)
        output_word_count = len(agent_output.split())
        subtask_count = len(subtasks) if subtasks else 0
        if (
            not completion_claimed
            and vague_readiness
            and subtask_count >= 3
            and output_word_count < 40
        ):
            completion_claimed = True

        # v2.2: Explicit blockers — "opened a ticket", "needs IT access",
        # "waiting for", "couldn't finish". Strong signal of incomplete
        # delivery, regardless of completion claim.
        blockers = self._detect_blockers(agent_output)

        # v2.2c: Explicit failure admission ("agent failed", "did not complete")
        explicit_failure = self._detect_explicit_failure(agent_output)

        # Detect incomplete markers
        incomplete_markers = self._detect_incomplete_markers(agent_output)

        # v1.3: Detect JSON-specific incomplete indicators
        json_incomplete = self._detect_json_incomplete(agent_output)

        # v1.3: Detect numeric ratios (e.g., 8/10)
        numeric_ratio = self._detect_numeric_ratio(agent_output)

        # Detect errors
        errors = self._detect_errors(agent_output)

        # v1.1: Detect partial completion hedges and qualifiers
        # v1.7: Also check criteria text for quantitative requirements
        has_quant_req = self._has_quantitative_requirement(task)
        if not has_quant_req and success_criteria:
            criteria_text = " ".join(success_criteria)
            has_quant_req = self._has_quantitative_requirement(criteria_text)
        partial_indicators = self._detect_partial_completion(agent_output)
        qualifiers = self._detect_qualifiers(agent_output)

        # v1.8: Detect honest partial reporting (reduces FP)
        honest_partial = self._detect_honest_partial_reporting(agent_output)

        # v1.2: Detect planned/future work indicators and check for scoped tasks
        planned_work = self._detect_planned_work(agent_output)
        is_scoped_task = self._is_intentionally_scoped_task(task)
        has_progress_language = self._detect_progress_language(agent_output)

        # Calculate completion ratio
        completion_ratio = self._calculate_completion_ratio(agent_output, task)

        # v1.1: Penalize completion ratio if partial indicators found with quantitative requirement
        if has_quant_req and partial_indicators:
            penalty = min(0.3, len(partial_indicators) * 0.1)
            completion_ratio = max(0.0, completion_ratio - penalty)

        # Analyze subtasks if provided
        subtasks_completed = 0
        subtasks_total = 0
        incomplete_subtasks = []

        if subtasks:
            subtasks_completed, subtasks_total, incomplete_subtasks = self._analyze_subtasks(
                subtasks
            )
            if subtasks_total > 0:
                subtask_ratio = subtasks_completed / subtasks_total
                # v2.2: When statuses are inferred from names, clip the
                # completion_ratio penalty — the word-overlap heuristic is
                # noisy and 1-2 missed subtasks out of 4-6 is commonly wrong.
                if subtasks_were_inferred:
                    # v2.3: Name-inference with word-overlap is a noisy
                    # heuristic — outputs frequently describe the same work
                    # using synonyms. Clamp the subtask_ratio so it cannot
                    # drive completion_ratio below 0.5 on inference alone.
                    # A true "premature completion" still has to come from
                    # other signals (blockers, explicit_failure, numeric
                    # ratio, planned_work, enumerated_coverage, etc.).
                    if subtask_ratio >= 0.75:
                        subtask_ratio = 0.95
                    elif subtask_ratio >= 0.5:
                        subtask_ratio = 0.85
                    else:
                        subtask_ratio = max(subtask_ratio, 0.5)
                completion_ratio = min(completion_ratio, subtask_ratio)

        # Check success criteria
        criteria = success_criteria or self._extract_success_criteria(task)
        criteria_met = 0
        criteria_total = 0
        unmet_criteria = []

        if criteria:
            criteria_met, criteria_total, unmet_criteria = self._check_criteria_met(
                criteria, agent_output
            )
            if criteria_total > 0:
                criteria_ratio = criteria_met / criteria_total
                completion_ratio = min(completion_ratio, criteria_ratio)

        # Check expected outputs
        if expected_outputs:
            output_lower = agent_output.lower()
            found_outputs = sum(1 for exp in expected_outputs if exp.lower() in output_lower)
            if len(expected_outputs) > 0:
                output_ratio = found_outputs / len(expected_outputs)
                completion_ratio = min(completion_ratio, output_ratio)
                if found_outputs < len(expected_outputs):
                    missing = [e for e in expected_outputs if e.lower() not in output_lower]
                    for m in missing[:3]:  # Top 3 missing
                        issues.append(
                            CompletionIssue(
                                issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                                description=f"Expected output missing: {m}",
                                severity=CompletionSeverity.MODERATE,
                                missing_element=m,
                            )
                        )

        # Detect misjudgment
        # v2.3: When subtask completion was computed from name-inference
        # alone (no other mismatch signal), downgrade severity from
        # SEVERE → MODERATE so ensemble voting (2+ categories) applies.
        # Outputs that describe the same work using synonyms of subtask
        # names produced many SEVERE FPs at this gate.
        if completion_claimed and completion_ratio < self.completion_threshold:
            premature_severity = CompletionSeverity.SEVERE
            if (
                subtasks_were_inferred
                and not incomplete_markers
                and not errors
                and not (numeric_ratio and numeric_ratio[2] < 1.0)
                and not planned_work
                and not qualifiers
            ):
                premature_severity = CompletionSeverity.MODERATE
            issues.append(
                CompletionIssue(
                    issue_type=CompletionIssueType.PREMATURE_COMPLETION,
                    description=f"Agent claimed completion at {completion_ratio:.0%} actual progress",
                    severity=premature_severity,
                    evidence=f"Completion ratio: {completion_ratio:.2f}",
                )
            )

        # Check for incomplete markers with completion claim
        if completion_claimed and incomplete_markers:
            for marker, marker_type, context in incomplete_markers[:3]:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.INCOMPLETE_VERIFICATION,
                        description=f"Incomplete marker found despite completion claim: {marker}",
                        severity=CompletionSeverity.MODERATE,
                        evidence=context,
                    )
                )

        # Check for errors with completion claim
        if completion_claimed and errors:
            issues.append(
                CompletionIssue(
                    issue_type=CompletionIssueType.FALSE_SUCCESS_CLAIM,
                    description=f"Errors detected despite completion claim ({len(errors)} error mentions)",
                    severity=CompletionSeverity.SEVERE,
                    evidence=errors[0] if errors else None,
                )
            )

        # Check for incomplete subtasks with completion claim.
        # v2.2: When subtask statuses are inferred from names (heuristic
        # word-overlap), a single miss is unreliable — require at least
        # 2 pending subtasks OR >= 40% pending to fire.
        if completion_claimed and incomplete_subtasks:
            fire_ignored = True
            if subtasks_were_inferred and subtasks_total > 0:
                pending_ratio = len(incomplete_subtasks) / subtasks_total
                # v2.2: Require at least 2 pending OR >= 40% pending for
                # inferred subtask statuses — the name-inference heuristic
                # is noisy, so single-miss firings produced FPs on cases
                # like "migration" inferred pending when the agent actually
                # completed that work.
                if len(incomplete_subtasks) < 2 and pending_ratio < 0.4:
                    fire_ignored = False
            if fire_ignored:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.IGNORED_SUBTASKS,
                        description=f"{len(incomplete_subtasks)} subtasks incomplete despite completion claim",
                        severity=CompletionSeverity.SEVERE,
                        missing_element=", ".join(incomplete_subtasks[:3]),
                    )
                )

        # Check for unmet criteria with completion claim
        if completion_claimed and unmet_criteria:
            issues.append(
                CompletionIssue(
                    issue_type=CompletionIssueType.MISSED_CRITERIA,
                    description=f"{len(unmet_criteria)} success criteria not met",
                    severity=CompletionSeverity.MODERATE,
                    missing_element=unmet_criteria[0] if unmet_criteria else None,
                )
            )

        # v1.1: Check for partial completion with quantitative requirement
        if has_quant_req and partial_indicators and completion_claimed:
            for indicator, indicator_type, context in partial_indicators[:2]:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description=f"Task requires 100% but output indicates partial: '{indicator}'",
                        severity=CompletionSeverity.MODERATE,
                        evidence=context,
                    )
                )

        # v1.1: Check for uncertainty qualifiers with completion claim
        if qualifiers and completion_claimed:
            for qualifier, qualifier_type, context in qualifiers[:2]:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.INCOMPLETE_VERIFICATION,
                        description=f"Uncertainty qualifier suggests incomplete: '{qualifier}'",
                        severity=CompletionSeverity.MINOR,
                        evidence=context,
                    )
                )

        # v1.2: Check for planned/future work (indicates incomplete even with completion claim)
        if planned_work and completion_claimed:
            for indicator, indicator_type, context in planned_work[:2]:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PREMATURE_COMPLETION,
                        description=f"Work marked as planned/future but completion claimed: '{indicator}'",
                        severity=CompletionSeverity.MODERATE,
                        evidence=context,
                    )
                )

        # v1.2: For quantitative requirements, detect failure even without explicit claim
        # If task requires "all" and output shows partial work, that's a failure
        if has_quant_req and not is_scoped_task and not completion_claimed:
            # Check if there are partial indicators or qualifiers
            if partial_indicators:
                for indicator, indicator_type, context in partial_indicators[:2]:
                    issues.append(
                        CompletionIssue(
                            issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                            description=f"Task requires 100% completion but partial work indicated: '{indicator}'",
                            severity=CompletionSeverity.MODERATE,
                            evidence=context,
                        )
                    )
            if qualifiers:
                for qualifier, qualifier_type, context in qualifiers[:2]:
                    issues.append(
                        CompletionIssue(
                            issue_type=CompletionIssueType.INCOMPLETE_VERIFICATION,
                            description=f"Task requires certainty but uncertainty expressed: '{qualifier}'",
                            severity=CompletionSeverity.MODERATE,
                            evidence=context,
                        )
                    )
            # v1.2: Progress language without completion = not done
            if has_progress_language:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PREMATURE_COMPLETION,
                        description="Task requires 100% but uses progress language (implies ongoing work)",
                        severity=CompletionSeverity.MODERATE,
                        evidence="Progress language detected without explicit completion claim",
                    )
                )

        # v1.3: Check for JSON incomplete indicators with completion claim
        if completion_claimed and json_incomplete:
            for indicator, indicator_type, context in json_incomplete[:2]:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description=f"JSON output shows incomplete items: '{indicator}'",
                        severity=CompletionSeverity.MODERATE,
                        evidence=context,
                    )
                )

        # v1.3: Check for numeric ratios showing incomplete (with or without completion claim)
        if numeric_ratio and has_quant_req and not is_scoped_task:
            completed, total, ratio = numeric_ratio
            if ratio < 1.0:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description=f"Task requires 100% but output shows {completed}/{total} ({ratio * 100:.0f}%)",
                        severity=CompletionSeverity.MODERATE,
                        evidence=f"Numeric ratio detected: {completed}/{total}",
                    )
                )

        # v1.3: JSON incomplete without explicit claim but with quantitative requirement
        if json_incomplete and has_quant_req and not is_scoped_task and not completion_claimed:
            for indicator, indicator_type, context in json_incomplete[:2]:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description=f"Task requires 100% but JSON shows incomplete: '{indicator}'",
                        severity=CompletionSeverity.MODERATE,
                        evidence=context,
                    )
                )

        # v1.7: Detect deferred/incomplete work without explicit completion claim
        # Agents that say "X is on the backlog" or "not yet configured" are
        # delivering partial work — create MODERATE issues so the ensemble gate
        # can consider them alongside other signals.
        if not completion_claimed and not is_scoped_task:
            if planned_work and any(p[1] == "deferred_work" for p in planned_work):
                deferred = [p for p in planned_work if p[1] == "deferred_work"]
                for indicator, _, context in deferred[:2]:
                    issues.append(
                        CompletionIssue(
                            issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                            description=f"Work explicitly deferred: '{indicator}'",
                            severity=CompletionSeverity.MODERATE,
                            evidence=context,
                        )
                    )
            if incomplete_markers and (partial_indicators or planned_work):
                for marker, marker_type, context in incomplete_markers[:2]:
                    issues.append(
                        CompletionIssue(
                            issue_type=CompletionIssueType.INCOMPLETE_VERIFICATION,
                            description=f"Incomplete marker without completion claim: '{marker}'",
                            severity=CompletionSeverity.MINOR,
                            evidence=context,
                        )
                    )

        # v2.0: Detect truncated output — output that ends mid-code or mid-sentence
        # is clearly incomplete. This is a strong signal (SEVERE) because the
        # deliverable is literally cut off.
        if len(agent_output) > 500:
            tail = agent_output[-30:].rstrip()
            # Truncated if ends with "..." (explicit truncation marker)
            is_truncated = tail.endswith("...")
            if is_truncated:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description="Output appears truncated (ends with ellipsis)",
                        severity=CompletionSeverity.SEVERE,
                        evidence=f"Output tail: ...{tail}",
                    )
                )

        # v2.1: Task complexity vs output brevity — complex multi-requirement
        # tasks with suspiciously short outputs suggest incomplete delivery
        if completion_claimed and not is_scoped_task:
            task_requirements = sum(
                1
                for kw in [
                    "and",
                    "also",
                    "additionally",
                    "furthermore",
                    "plus",
                ]
                if kw in task.lower()
            )
            task_requirement_count = max(1, task_requirements + 1)
            output_word_count = len(agent_output.split())
            # Expect ~50 words per requirement minimum
            if task_requirement_count >= 3 and output_word_count < task_requirement_count * 40:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description=f"Task has {task_requirement_count} requirements but output is only {output_word_count} words",
                        severity=CompletionSeverity.MODERATE,
                        evidence=f"requirements~={task_requirement_count}, words={output_word_count}",
                    )
                )

        # v2.2b: Short output claiming readiness for a multi-subtask task —
        # agent says "X is ready / completed successfully" in < 40 words for a
        # task with 3+ subtasks. This is an implicit false-completion signal
        # because the output can't plausibly cover the subtask set.
        if vague_readiness and subtask_count >= 3 and output_word_count < 40 and not is_scoped_task:
            issues.append(
                CompletionIssue(
                    issue_type=CompletionIssueType.PREMATURE_COMPLETION,
                    description=(
                        f"Vague readiness claim in {output_word_count}-word output "
                        f"for task with {subtask_count} subtasks"
                    ),
                    severity=CompletionSeverity.MODERATE,
                    evidence="vague_readiness + short_output",
                )
            )

        # v2.2d: External-grader criteria — entries from benchmark datasets
        # (ALFRED/ARB/SWE-bench) use "Complete: <task>" or "Agent resolves..."
        # as the sole success criterion. These mean "external grader will
        # verify completion". If no completion claim is made AND the output
        # doesn't mention the key task nouns, the agent likely didn't finish.
        is_external_grader = False
        is_swe_bench_grader = False
        is_arb_grader = False
        if success_criteria and len(success_criteria) == 1:
            single = success_criteria[0].lower()
            if single.startswith("complete:") or single.strip() in (
                "completed",
                "complete",
                "success",
            ):
                is_external_grader = True
            elif single.startswith("agent resolves"):
                is_swe_bench_grader = True
            elif single.startswith("agent successfully completes"):
                is_arb_grader = True
        if (
            is_external_grader
            and not completion_claimed
            and not is_scoped_task
            and len(agent_output) > 30
        ):
            # Check whether the output overlaps with the task nouns (gives
            # the agent the benefit of the doubt on direct answers). If
            # overlap is very low, treat as incomplete.
            task_nouns = set(
                w
                for w in re.findall(r"[a-z]{4,}", task.lower())
                if w
                not in {
                    "task",
                    "from",
                    "with",
                    "that",
                    "this",
                    "have",
                    "been",
                    "agent",
                    "complete",
                    "completed",
                    "needs",
                    "should",
                    "must",
                    "would",
                    "could",
                    "could",
                    "will",
                    "when",
                    "where",
                    "what",
                    "conduct",
                    "create",
                    "build",
                    "implement",
                    "setup",
                    "set",
                    "using",
                    "based",
                    "there",
                    "their",
                    "about",
                    "these",
                    "those",
                }
            )
            output_tokens = set(re.findall(r"[a-z]{4,}", agent_output.lower()))
            if task_nouns:
                overlap = len(task_nouns & output_tokens) / len(task_nouns)
                if overlap < 0.25:
                    issues.append(
                        CompletionIssue(
                            issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                            description=f"External-grader task but output has low task-noun overlap ({overlap:.0%})",
                            severity=CompletionSeverity.MODERATE,
                            evidence=f"overlap={overlap:.2f}",
                        )
                    )

        # v2.2d: ARB grader — successful runs are short (median 4 steps);
        # failed runs exhaust many steps (median 29). Fire when step count
        # is high and no completion claim.
        if is_arb_grader and not completion_claimed and not is_scoped_task:
            step_nums = [int(m.group(1)) for m in re.finditer(r"Step\s+(\d+)", agent_output)]
            max_step = max(step_nums) if step_nums else 0
            if max_step >= 15:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description=f"ARB task exhausted {max_step} steps without completion",
                        severity=CompletionSeverity.MODERATE,
                        evidence=f"max_step={max_step}",
                    )
                )

        # v2.2d: SWE-bench grader — successful runs include a final "submit"
        # action. Only fire when there's NO submit AND explicit problem/error
        # language in the output tail (last 500 chars) AND the agent got
        # blocked mid-investigation.
        if (
            is_swe_bench_grader
            and not completion_claimed
            and not is_scoped_task
            and len(agent_output) > 100
        ):
            output_lower = agent_output.lower()
            tail_lower = output_lower[-500:]
            has_submit = "submit" in output_lower
            # v2.2e: Tighter — require BLOCKING language in the tail, not just
            # any error/issue mention (swe_comp outputs frequently mention
            # "error" even in successful fixes while referring to the bug).
            has_blocking_tail = any(
                p in tail_lower
                for p in (
                    "cannot",
                    "unable to",
                    "still need",
                    "not sure",
                    "doesn't seem",
                    "does not seem",
                )
            )
            if not has_submit and has_blocking_tail:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description="SWE-bench task without submit action",
                        severity=CompletionSeverity.MODERATE,
                        evidence="no_submit_action",
                    )
                )

        # v2.2c: Explicit failure signal — agent admits failure
        if explicit_failure and not is_scoped_task:
            issues.append(
                CompletionIssue(
                    issue_type=CompletionIssueType.FALSE_SUCCESS_CLAIM,
                    description="Agent explicitly admits failure/incomplete work",
                    severity=CompletionSeverity.SEVERE,
                    evidence="explicit_failure pattern matched",
                )
            )

        # v2.2: Blocker signal — partial delivery with external dependency
        if blockers and not is_scoped_task:
            for indicator, btype, context in blockers[:2]:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.PARTIAL_DELIVERY,
                        description=f"Agent deferred work via blocker: '{indicator}'",
                        severity=CompletionSeverity.MODERATE,
                        evidence=context,
                    )
                )

        # v1.5: Detect structural incompleteness (e.g., fewer list items than requested)
        structural_issues = self._detect_structural_incompleteness(task, agent_output)
        if structural_issues:
            issues.extend(structural_issues)

        # v1.5: Detect enumerated items coverage gap
        enum_issues = self._detect_enumerated_coverage(task, agent_output)
        if enum_issues:
            issues.extend(enum_issues)

        # v1.9: Structural subtask gap detection — if subtasks provided and
        # agent claims done but covers < 25% of subtask names, flag it.
        # Only fires when subtask analysis shows all "completed" (no prior
        # IGNORED_SUBTASKS issue) to avoid double-flagging.
        # v2.2: Tightened threshold (was 50%) + use stem matching like
        # infer_subtask_status so "provider" matches "providers". Skip the
        # check entirely when subtasks were name-inferred AND the analyzer
        # already saw them all as "completed" — that's strong signal of
        # coverage and the raw-substring check adds FPs via tokenization.
        if subtasks and len(subtasks) > 1 and completion_claimed and not incomplete_subtasks:
            subtask_names = [
                s.get("name", s.get("description", "")) if isinstance(s, dict) else str(s)
                for s in subtasks
            ]
            output_lower = agent_output.lower()
            output_words = set(re.findall(r"[a-z]{3,}", output_lower))
            output_stems = {self._stem_word(w) for w in output_words}
            mentioned = 0
            for st in subtask_names:
                words = [w for w in re.split(r"[_\s\-]+", st.lower()) if len(w) >= 3]
                if not words:
                    continue
                if any(w in output_lower or self._stem_word(w) in output_stems for w in words):
                    mentioned += 1
            coverage = mentioned / len(subtask_names) if subtask_names else 1.0
            if coverage < 0.4:
                issues.append(
                    CompletionIssue(
                        issue_type=CompletionIssueType.IGNORED_SUBTASKS,
                        description=f"Agent claims completion but only mentions {mentioned}/{len(subtask_names)} subtasks ({coverage:.0%} coverage)",
                        severity=CompletionSeverity.MODERATE,
                        evidence=f"subtask_coverage={coverage:.2f}",
                    )
                )

        # v1.6: Completion by absence — if task has explicit criteria
        # and output addresses fewer than a third, flag even without claim.
        # v2.2: Raise threshold from 0.5 to 0.34 and require either a
        # completion claim OR >=3 criteria. Keyword-based criteria matching
        # is noisy; firing on 49% coverage with 2 criteria (i.e. 1/2 missing)
        # produced many FPs on narrative outputs that implicitly cover
        # criteria using synonyms.
        if criteria_total >= 3 and criteria_met < criteria_total * 0.34 and not is_scoped_task:
            issues.append(
                CompletionIssue(
                    issue_type=CompletionIssueType.MISSED_CRITERIA,
                    description=f"Output addresses only {criteria_met}/{criteria_total} success criteria",
                    severity=CompletionSeverity.MODERATE,
                    evidence=f"Unmet: {', '.join(unmet_criteria[:3])}",
                )
            )

        # v1.2: Reduce false positives for scoped tasks
        if is_scoped_task and issues:
            # Filter out minor issues for intentionally scoped tasks
            issues = [
                i
                for i in issues
                if i.severity in [CompletionSeverity.SEVERE, CompletionSeverity.CRITICAL]
            ]

        # v1.4: Ensemble voting — require 2+ distinct signal categories
        # to trigger detection for non-critical issues.  A single signal
        # category (e.g., only "partial_indicators" without a completion
        # claim) produces too many false positives.
        set(i.issue_type for i in issues)
        max_issue_severity = max((i.severity for i in issues), default=CompletionSeverity.NONE)
        if issues and max_issue_severity not in [
            CompletionSeverity.SEVERE,
            CompletionSeverity.CRITICAL,
        ]:
            # Count distinct signal categories (not just issue types)
            signal_categories = set()
            if completion_claimed:
                signal_categories.add("completion_claim")
            if incomplete_markers:
                signal_categories.add("incomplete_markers")
            if partial_indicators:
                signal_categories.add("partial_indicators")
            if qualifiers:
                signal_categories.add("qualifiers")
            if numeric_ratio and numeric_ratio[2] < 1.0:
                signal_categories.add("numeric_ratio")
            if json_incomplete:
                signal_categories.add("json_incomplete")
            if planned_work:
                signal_categories.add("planned_work")
            if errors:
                signal_categories.add("errors")
            if incomplete_subtasks:
                signal_categories.add("incomplete_subtasks")
            if structural_issues:
                signal_categories.add("structural_incompleteness")
            if enum_issues:
                signal_categories.add("enumerated_coverage")
            if blockers:
                signal_categories.add("blockers")
            if vague_readiness and output_word_count < 40 and subtask_count >= 3:
                signal_categories.add("vague_short_output")
            if explicit_failure:
                signal_categories.add("explicit_failure")
            if is_external_grader and any("External-grader" in i.description for i in issues):
                signal_categories.add("external_grader_mismatch")
            if is_swe_bench_grader and any("SWE-bench" in i.description for i in issues):
                signal_categories.add("swe_bench_no_submit")
            if is_arb_grader and any("ARB task exhausted" in i.description for i in issues):
                signal_categories.add("arb_step_exhaustion")

            # v1.5: Exempt quantitative-requirement cases: a single strong signal
            # (partial_indicators, numeric_ratio, or structural_incompleteness)
            # is sufficient when the task has explicit numeric requirements.
            has_quant_exemption = has_quant_req and (
                "partial_indicators" in signal_categories
                or "numeric_ratio" in signal_categories
                or "incomplete_subtasks" in signal_categories
                or "structural_incompleteness" in signal_categories
                or "enumerated_coverage" in signal_categories
            )
            # Enumerated coverage + completion claim is always strong enough
            if (
                "enumerated_coverage" in signal_categories
                and "completion_claim" in signal_categories
            ):
                has_quant_exemption = True
            # v1.6: Completion claim + any incompleteness signal is strong
            if "completion_claim" in signal_categories and (
                "incomplete_markers" in signal_categories
                or "planned_work" in signal_categories
                or "errors" in signal_categories
                or "json_incomplete" in signal_categories
            ):
                has_quant_exemption = True
            # v1.6: Structural or enumerated coverage alone is strong evidence
            if (
                "structural_incompleteness" in signal_categories
                or "enumerated_coverage" in signal_categories
            ):
                has_quant_exemption = True
            # v1.7: Deferred work (backlog, next sprint) alone is strong evidence
            if planned_work and any(p[1] == "deferred_work" for p in planned_work):
                has_quant_exemption = True
            # v2.2: Explicit blocker (opened ticket, waiting for approval)
            # is strong standalone evidence of incomplete delivery.
            if blockers:
                has_quant_exemption = True
            # v2.2b: Vague readiness + short output + multi-subtask task
            # alone is sufficient evidence.
            if "vague_short_output" in signal_categories:
                has_quant_exemption = True
            # v2.2c: Explicit failure alone is strong standalone evidence.
            if "explicit_failure" in signal_categories:
                has_quant_exemption = True
            if "external_grader_mismatch" in signal_categories:
                has_quant_exemption = True
            if "swe_bench_no_submit" in signal_categories:
                has_quant_exemption = True
            if "arb_step_exhaustion" in signal_categories:
                has_quant_exemption = True

            if len(signal_categories) < 2 and not has_quant_exemption:
                issues = []  # Suppress detection — single signal insufficient

        # v1.8: If agent honestly reports partial progress (e.g., "partially done",
        # "completed N of M", "still working on", "TODO:"), suppress detection —
        # honest partial reporting is not a completion misjudgment.
        # v2.2: But only when there are no blockers — a ticket for IT access is
        # not "transparent status reporting", it's a deferred deliverable.
        if honest_partial and not completion_claimed and not blockers:
            # Agent is transparently reporting partial progress — not a false claim
            issues = []

        # Determine result
        detected = len(issues) > 0

        if not detected:
            return CompletionResult(
                detected=False,
                severity=CompletionSeverity.NONE,
                confidence=0.8,
                completion_claimed=completion_claimed,
                actual_completion_ratio=completion_ratio,
                subtasks_total=subtasks_total,
                subtasks_completed=subtasks_completed,
                success_criteria_met=criteria_met,
                success_criteria_total=criteria_total,
                explanation="No completion misjudgment detected",
            )

        # Calculate severity
        if any(i.severity == CompletionSeverity.CRITICAL for i in issues):
            severity = CompletionSeverity.CRITICAL
        elif any(i.severity == CompletionSeverity.SEVERE for i in issues):
            severity = CompletionSeverity.SEVERE
        elif any(i.severity == CompletionSeverity.MODERATE for i in issues):
            severity = CompletionSeverity.MODERATE
        else:
            severity = CompletionSeverity.MINOR

        # Calculate confidence based on evidence strength and signal diversity
        signal_count = sum(
            [
                bool(completion_claimed),
                bool(incomplete_markers),
                bool(errors),
                bool(incomplete_subtasks),
                bool(partial_indicators),
                bool(planned_work),
                bool(structural_issues),
                bool(enum_issues),
                bool(numeric_ratio and numeric_ratio[2] < 1.0),
                bool(json_incomplete),
            ]
        )
        if signal_count >= 4:
            base_confidence = 0.85
        elif signal_count >= 3:
            base_confidence = 0.75
        elif signal_count >= 2:
            base_confidence = 0.65
        else:
            base_confidence = 0.55
        # Boost for severe issues
        if any(
            i.severity in (CompletionSeverity.SEVERE, CompletionSeverity.CRITICAL) for i in issues
        ):
            base_confidence = max(base_confidence, 0.75)
        confidence = min(0.95, base_confidence)

        # Build explanation
        issue_types = set(i.issue_type.value for i in issues)
        explanation = (
            f"Detected completion misjudgment: {', '.join(issue_types)}. "
            f"Actual completion: {completion_ratio:.0%}"
        )

        # Suggest fix
        if any(i.issue_type == CompletionIssueType.IGNORED_SUBTASKS for i in issues):
            suggested_fix = (
                "Ensure agent tracks and completes all subtasks before claiming completion"
            )
        elif any(i.issue_type == CompletionIssueType.MISSED_CRITERIA for i in issues):
            suggested_fix = "Implement success criteria verification before completion"
        elif any(i.issue_type == CompletionIssueType.FALSE_SUCCESS_CLAIM for i in issues):
            suggested_fix = "Add error detection to prevent false success claims"
        else:
            suggested_fix = "Add comprehensive completion verification to agent workflow"

        return CompletionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            issues=issues,
            completion_claimed=completion_claimed,
            actual_completion_ratio=completion_ratio,
            subtasks_total=subtasks_total,
            subtasks_completed=subtasks_completed,
            success_criteria_met=criteria_met,
            success_criteria_total=criteria_total,
            explanation=explanation,
            suggested_fix=suggested_fix,
        )


# Singleton instance
completion_detector = CompletionMisjudgmentDetector()
