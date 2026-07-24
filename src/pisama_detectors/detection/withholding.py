"""
F8: Information Withholding Detection (MAST Taxonomy)
======================================================

Detects when an agent doesn't share critical information with peers.
This occurs in inter-agent communication when:
- Agent discovers important information but doesn't pass it on
- Agent summarizes away critical details
- Agent selectively omits negative findings
- Agent's output is significantly less informative than its internal state

Version History:
- v1.0: Initial implementation
- v1.1: Task-aware detection to reduce false positives:
  - Recognize summary/list tasks expect condensed output
  - Links to full details = NOT withholding
  - Organized/structured output = NOT withholding
- v1.2: FPR reduction:
  - Expanded condensed output task patterns
  - Importance weighting for critical patterns
  - Semantic retention check using embeddings
  - Role-based threshold adjustment
"""

# Detector version for tracking
DETECTOR_VERSION = "1.2"
DETECTOR_NAME = "InformationWithholdingDetector"

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set

# v1.6: Reasoning trace markers — if internal_state shows deliberate reasoning
# process, the agent is meant to expose its thinking, not forward it verbatim.
REASONING_MARKERS = [
    r"\b(?:thinking|plan|approach|reasoning|strategy|analysis)\s*:",
    r"\b(?:step \d|first|then|next|finally)\b.*\b(?:step \d|then|next|finally)\b",
    r"\blet me (?:think|consider|analyze|plan)\b",
]

logger = logging.getLogger(__name__)


class WithholdingSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class WithholdingType(str, Enum):
    CRITICAL_OMISSION = "critical_omission"
    DETAIL_LOSS = "detail_loss"
    NEGATIVE_SUPPRESSION = "negative_suppression"
    SELECTIVE_REPORTING = "selective_reporting"
    CONTEXT_STRIPPING = "context_stripping"


@dataclass
class WithholdingIssue:
    issue_type: WithholdingType
    withheld_info: str
    severity: WithholdingSeverity
    description: str


@dataclass
class WithholdingResult:
    detected: bool
    severity: WithholdingSeverity
    confidence: float
    issues: List[WithholdingIssue] = field(default_factory=list)
    information_retention_ratio: float = 1.0
    critical_items_found: int = 0
    critical_items_reported: int = 0
    explanation: str = ""
    suggested_fix: Optional[str] = None


class InformationWithholdingDetector:
    """
    Detects F8: Information Withholding - agent hides info from peers.

    Compares agent's internal findings with what it communicates to
    downstream agents or final output.
    """

    # Critical information patterns that should be passed on
    CRITICAL_PATTERNS = [
        (r"\b(error|failure|exception|bug|issue|problem)\b", "error_condition"),
        (r"\b(warning|caution|alert|risk|danger)\b", "warning"),
        (r"\b(critical|urgent|important|essential|crucial)\b", "priority_marker"),
        (r"\b(security|vulnerability|exploit|breach)\b", "security_issue"),
        (r"\b(deadline|due date|expires?|timeout)\b", "time_constraint"),
        (r"\b(cost|price|fee|charge|expense)\s*[:=]?\s*\$?\d+", "financial_info"),
        (r"\b(blocked|blocker|impediment|obstacle)\b", "blocker"),
        (r"\b(deprecated|obsolete|outdated|legacy)\b", "deprecation"),
        (r"\bnot\s+(?:working|functional|available|supported)\b", "unavailability"),
        (r"\b(failed|unsuccessful|unable|cannot)\b", "failure_indicator"),
    ]

    # v1.2: Importance weights for critical patterns (1.0 = critical, 0.0 = informational)
    CRITICAL_PATTERNS_WEIGHTED = [
        (r"\b(error|failure|exception|bug)\b", "error_condition", 1.0),  # Critical
        (r"\b(security|vulnerability|exploit|breach)\b", "security_issue", 1.0),  # Critical
        (r"\b(blocked|blocker|impediment)\b", "blocker", 0.9),  # High
        (r"\b(issue|problem)\b", "issue", 0.8),  # High
        (r"\b(warning|caution|alert|risk|danger)\b", "warning", 0.7),  # Medium
        (r"\b(critical|urgent|important)\b", "priority_marker", 0.6),  # Context-dependent
        (r"\b(deadline|due date|expires?|timeout)\b", "time_constraint", 0.5),  # Medium-low
        (r"\b(cost|price|fee|charge|expense)\s*[:=]?\s*\$?\d+", "financial_info", 0.5),
        (r"\b(deprecated|obsolete|outdated|legacy)\b", "deprecation", 0.4),  # Low
        (r"\bnot\s+(?:working|functional|available|supported)\b", "unavailability", 0.6),
        (r"\b(failed|unsuccessful|unable|cannot)\b", "failure_indicator", 0.7),
    ]

    # Patterns indicating negative findings
    NEGATIVE_FINDING_PATTERNS = [
        r"\b(no|not|none|neither|never)\s+\w+",
        r"\b(missing|absent|lacking|without)\b",
        r"\b(incorrect|wrong|invalid|bad)\b",
        r"\b(rejected|denied|refused|declined)\b",
        r"\b(insufficient|inadequate|incomplete)\b",
    ]

    # Patterns indicating summarization/reduction
    SUMMARY_PATTERNS = [
        r"\b(in summary|to summarize|in brief|briefly)\b",
        r"\b(the main point|key takeaway|bottom line)\b",
        r"\b(essentially|basically|simply put)\b",
        r"\b(overall|in general|generally speaking)\b",
    ]

    # v1.1: Tasks that explicitly request summary/list format (not withholding)
    SUMMARY_TASK_PATTERNS = [
        r"\bsummar(?:y|ize|ise)\b",
        r"\bbrief(?:ly)?\b",
        r"\boverview\b",
        r"\bkey\s+(?:points?|findings?|takeaways?)\b",
        r"\bhighlights?\b",
        r"\blist\b",
    ]

    # v1.2: Expanded patterns for condensed output tasks
    CONDENSED_OUTPUT_TASKS = [
        r"\b(?:get|give|provide)\s+(?:me\s+)?(?:the\s+)?(?:main|key|important|top)\b",
        r"\b(?:executive\s+)?summar(?:y|ize|ise)\b",
        r"\b(?:brief|short|concise|quick)\s+(?:overview|summary|report)\b",
        r"\btl;?dr\b",
        r"\bhighlights?\b",
        r"\b(?:bullet\s+)?(?:points?|list)\b",
        r"\bconclusion\b",
        r"\b(?:bottom\s+line|takeaway)\b",
        r"\b(?:in\s+)?one\s+(?:sentence|paragraph|word)\b",
        r"\babstract\b",
        r"\b(?:max(?:imum)?|under|less\s+than)\s+\d+\s*(?:word|char|line)",
        r"\b(?:keep\s+it\s+)?(?:short|brief|concise)\b",
    ]

    # v1.2: Roles that are expected to summarize (not withholding)
    SUMMARIZING_ROLES = [
        "manager",
        "coordinator",
        "reporter",
        "summarizer",
        "summary",
        "lead",
        "pm",
        "director",
        "executive",
        "supervisor",
    ]

    # v1.1: Patterns indicating full information is accessible (not withholding)
    FULL_ACCESS_PATTERNS = [
        r"\bsee\s+(?:full|complete|detailed)\b",
        r"\bfull\s+(?:details?|report|analysis)\s+(?:at|in|available)\b",
        r"\bmore\s+(?:details?|information)\s+(?:at|in|available)\b",
        r"\blink(?:s|ed)?\s+(?:to|below|above)\b",
        r"\b(?:attached|appendix|supplement)\b",
        r"\brefer(?:ence)?\s+to\b",
        r"\bfor\s+(?:more|additional|complete)\s+(?:details?|information)\b",
    ]

    # v1.1: Patterns indicating organized/structured output
    STRUCTURED_OUTPUT_PATTERNS = [
        r"(?:^|\n)\s*[-•*]\s+",  # Bullet points
        r"(?:^|\n)\s*\d+[.)]\s+",  # Numbered list
        r"(?:^|\n)\s*#{1,6}\s+",  # Markdown headers
        r'"[^"]+":\s*[{\[]',  # JSON structure
        r"\n\s+\w+:\s+",  # Key-value pairs
    ]

    def __init__(
        self,
        critical_retention_threshold: float = 0.8,
        detail_retention_threshold: float = 0.6,
    ):
        self.critical_retention_threshold = critical_retention_threshold
        self.detail_retention_threshold = detail_retention_threshold

    def _extract_critical_items(self, text: str) -> List[tuple]:
        """Extract critical information items from text."""
        items = []

        for pattern, item_type in self.CRITICAL_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get surrounding context (50 chars each side)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end].strip()
                items.append((match.group(), item_type, context))

        return items

    def _extract_negative_findings(self, text: str) -> List[str]:
        """Extract negative findings from text."""
        findings = []

        for pattern in self.NEGATIVE_FINDING_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 30)
                end = min(len(text), match.end() + 30)
                context = text[start:end].strip()
                findings.append(context)

        return findings

    def _is_summary_task(self, task: str) -> bool:
        """v1.1: Check if task explicitly requests summary/list format."""
        if not task:
            return False
        for pattern in self.SUMMARY_TASK_PATTERNS:
            if re.search(pattern, task, re.IGNORECASE):
                return True
        return False

    def _is_condensed_output_expected(self, task: str, context: str = "") -> bool:
        """v1.2: Determine if task expects condensed/summarized output."""
        combined = f"{task} {context}".lower()

        for pattern in self.CONDENSED_OUTPUT_TASKS:
            if re.search(pattern, combined, re.IGNORECASE):
                return True

        return False

    def _is_summarizing_role(self, role: Optional[str]) -> bool:
        """v1.2: Check if agent role is expected to summarize."""
        if not role:
            return False
        role_lower = role.lower()
        return any(r in role_lower for r in self.SUMMARIZING_ROLES)

    def _semantic_information_retained(
        self,
        internal_info: str,
        output: str,
        threshold: float = 0.75,  # v1.3: Default — original threshold
    ) -> bool:
        """
        v1.2: Check if information is semantically retained even with different wording.

        Uses embedding similarity to detect if key information from internal state
        is present in output, even if paraphrased.
        """
        if not internal_info or not output:
            return True  # Can't determine, assume retained

        try:
            from pisama_detectors.detection.shared_embedder import (
                get_shared_embedder as get_embedder,
            )

            embedder = get_embedder()
            if not embedder:
                return False  # Can't check, return conservative answer

            internal_emb = embedder.encode(internal_info[:4000], is_query=True)
            output_emb = embedder.encode(output[:4000], is_query=False)
            similarity = embedder.similarity(internal_emb, output_emb)

            return similarity >= threshold

        except Exception as e:
            logger.debug(f"Semantic retention check failed: {e}")
            return False  # Can't check, return conservative answer

    def _has_full_access_indicators(self, text: str) -> bool:
        """v1.1: Check if output provides access to full information."""
        for pattern in self.FULL_ACCESS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _is_structured_output(self, text: str) -> bool:
        """v1.1: Check if output is well-structured (organized, not withheld)."""
        structure_count = 0
        for pattern in self.STRUCTURED_OUTPUT_PATTERNS:
            if re.search(pattern, text, re.MULTILINE):
                structure_count += 1
        return structure_count >= 2  # Multiple structure indicators

    def _calculate_information_density(self, text: str) -> float:
        """Calculate information density of text."""
        if not text:
            return 0.0

        words = text.split()
        if not words:
            return 0.0

        # Count substantive words (not stopwords)
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "and",
            "but",
            "if",
            "or",
            "because",
            "until",
            "while",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "what",
            "which",
        }

        substantive = sum(1 for w in words if w.lower() not in stopwords and len(w) > 2)
        return substantive / len(words)

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract named entities and key terms."""
        entities = set()

        # Numbers with units
        numbers = re.findall(
            r"\b\d+(?:\.\d+)?(?:\s*(?:%|dollars?|USD|EUR|GB|MB|KB|ms|seconds?|minutes?|hours?|days?))?",
            text,
        )
        entities.update(numbers)

        # Capitalized terms (potential names/entities)
        caps = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        entities.update(caps)

        # Technical terms
        tech = re.findall(
            r"\b(?:API|URL|HTTP|JSON|XML|SQL|SDK|REST|OAuth|JWT|AWS|GCP|Azure)\b",
            text,
            re.IGNORECASE,
        )
        entities.update(e.upper() for e in tech)

        # Quoted strings
        quoted = re.findall(r'"([^"]+)"', text)
        entities.update(quoted)

        return entities

    def _check_item_retention(
        self,
        internal_items: List[tuple],
        output_text: str,
    ) -> tuple[int, int, List[str]]:
        """Check how many critical items are retained in output."""
        found = len(internal_items)
        retained = 0
        missing = []

        output_lower = output_text.lower()

        for item, item_type, context in internal_items:
            # Check if item or its context appears in output
            item_found = item.lower() in output_lower
            context_words = set(context.lower().split())
            context_overlap = len(context_words & set(output_lower.split())) / max(
                len(context_words), 1
            )

            if item_found or context_overlap > 0.5:
                retained += 1
            else:
                missing.append(f"{item_type}: {item}")

        return found, retained, missing

    def detect(
        self,
        internal_state: str,
        agent_output: str,
        task_context: Optional[str] = None,
        downstream_requirements: Optional[List[str]] = None,
        agent_role: Optional[str] = None,  # v1.2: Agent role for threshold adjustment
    ) -> WithholdingResult:
        """
        Detect information withholding.

        Args:
            internal_state: Agent's internal findings/state (e.g., tool results, reasoning)
            agent_output: What the agent actually communicated
            task_context: The task the agent was performing
            downstream_requirements: What downstream agents need
            agent_role: Role of the agent (e.g., "manager", "coordinator")

        Returns:
            WithholdingResult with detection outcome
        """
        issues = []

        if not internal_state or not agent_output:
            return WithholdingResult(
                detected=False,
                severity=WithholdingSeverity.NONE,
                confidence=0.5,
                explanation="Insufficient data for withholding detection",
            )

        # v1.1: Check for false positive indicators
        is_summary_task = self._is_summary_task(task_context or internal_state)
        has_full_access = self._has_full_access_indicators(agent_output)
        is_structured = self._is_structured_output(agent_output)

        # v1.2: Extended false positive checks
        is_condensed_expected = self._is_condensed_output_expected(
            task_context or "", internal_state
        )
        is_summarizing_role = self._is_summarizing_role(agent_role)
        is_semantically_retained = self._semantic_information_retained(internal_state, agent_output)

        # v1.2: More lenient thresholds for summarizing contexts
        effective_critical_threshold = self.critical_retention_threshold
        effective_detail_threshold = self.detail_retention_threshold

        if is_summarizing_role or is_condensed_expected:
            effective_critical_threshold = 0.6  # Was 0.8
            effective_detail_threshold = 0.4  # Was 0.6

        # v1.6: Reasoning trace detection — if internal_state contains
        # deliberate reasoning markers, the agent is showing its process.
        # Reasoning traces are not meant to be forwarded verbatim.
        reasoning_marker_count = sum(
            1 for pattern in REASONING_MARKERS if re.search(pattern, internal_state, re.IGNORECASE)
        )
        if reasoning_marker_count >= 2:
            effective_critical_threshold *= 0.7  # More lenient

        # Extract critical items from internal state
        internal_critical = self._extract_critical_items(internal_state)
        found, retained, missing_critical = self._check_item_retention(
            internal_critical, agent_output
        )

        # Check critical item retention
        if found > 0:
            retention_ratio = retained / found
        else:
            retention_ratio = 1.0

        # Detect critical omissions
        # v1.2: Use effective threshold and check semantic retention
        if retention_ratio < effective_critical_threshold and missing_critical:
            # v1.2: If semantically retained, don't flag as withholding
            if is_semantically_retained:
                logger.debug("Critical items not keyword-matched but semantically retained")
            else:
                for missed in missing_critical[:5]:  # Top 5 missing
                    issues.append(
                        WithholdingIssue(
                            issue_type=WithholdingType.CRITICAL_OMISSION,
                            withheld_info=missed,
                            severity=WithholdingSeverity.SEVERE,
                            description=f"Critical information not passed on: {missed}",
                        )
                    )

        # Check for negative finding suppression
        internal_negatives = self._extract_negative_findings(internal_state)
        output_negatives = self._extract_negative_findings(agent_output)

        # v1.4: Reverted buffer to +2 — MINOR confidence fix handles FP/TP separation
        if len(internal_negatives) > len(output_negatives) + 2:
            suppressed_count = len(internal_negatives) - len(output_negatives)
            issues.append(
                WithholdingIssue(
                    issue_type=WithholdingType.NEGATIVE_SUPPRESSION,
                    withheld_info=f"{suppressed_count} negative findings",
                    severity=WithholdingSeverity.MODERATE,
                    description=f"Agent suppressed {suppressed_count} negative findings from output",
                )
            )

        # Check entity retention
        internal_entities = self._extract_entities(internal_state)
        output_entities = self._extract_entities(agent_output)

        if internal_entities:
            entity_retention = len(internal_entities & output_entities) / len(internal_entities)
            # v1.2: Use effective threshold
            if entity_retention < effective_detail_threshold:
                lost_entities = internal_entities - output_entities
                issues.append(
                    WithholdingIssue(
                        issue_type=WithholdingType.DETAIL_LOSS,
                        withheld_info=f"{len(lost_entities)} entities/details",
                        severity=WithholdingSeverity.MINOR,
                        description=f"Lost {len(lost_entities)} specific details in output",
                    )
                )

        # Check for over-summarization
        internal_density = self._calculate_information_density(internal_state)
        output_density = self._calculate_information_density(agent_output)

        if internal_density > 0 and output_density < internal_density * 0.5:
            if re.search(r"|".join(p for p in self.SUMMARY_PATTERNS), agent_output, re.IGNORECASE):
                issues.append(
                    WithholdingIssue(
                        issue_type=WithholdingType.CONTEXT_STRIPPING,
                        withheld_info="excessive summarization",
                        severity=WithholdingSeverity.MINOR,
                        description="Output significantly less detailed than internal state",
                    )
                )

        # v1.1: Reduce false positives based on task type and output characteristics
        if issues:
            # If task explicitly asks for summary/list, don't flag summarization as withholding
            # v1.2: Also check condensed output expectations and summarizing roles
            if is_summary_task or is_condensed_expected or is_summarizing_role:
                issues = [
                    i
                    for i in issues
                    if i.issue_type
                    not in [
                        WithholdingType.CONTEXT_STRIPPING,
                        WithholdingType.DETAIL_LOSS,
                    ]
                ]

            # If output provides links/references to full info, don't flag as withholding
            if has_full_access:
                issues = [
                    i
                    for i in issues
                    if i.severity
                    in [
                        WithholdingSeverity.SEVERE,
                        WithholdingSeverity.CRITICAL,
                    ]
                ]  # Only keep critical issues

            # If output is well-structured, reduce severity of detail loss
            if is_structured:
                for i, issue in enumerate(issues):
                    if issue.issue_type == WithholdingType.DETAIL_LOSS:
                        # Structured output likely just organized info differently
                        issues[i] = WithholdingIssue(
                            issue_type=issue.issue_type,
                            withheld_info=issue.withheld_info,
                            severity=WithholdingSeverity.NONE,  # Don't flag
                            description=issue.description,
                        )
                issues = [i for i in issues if i.severity != WithholdingSeverity.NONE]

        # Determine result
        detected = len(issues) > 0

        if not detected:
            return WithholdingResult(
                detected=False,
                severity=WithholdingSeverity.NONE,
                confidence=0.8,
                information_retention_ratio=retention_ratio,
                critical_items_found=found,
                critical_items_reported=retained,
                explanation="No information withholding detected",
            )

        # Calculate severity
        if any(i.severity == WithholdingSeverity.CRITICAL for i in issues):
            severity = WithholdingSeverity.CRITICAL
        elif any(i.severity == WithholdingSeverity.SEVERE for i in issues):
            severity = WithholdingSeverity.SEVERE
        elif any(i.severity == WithholdingSeverity.MODERATE for i in issues):
            severity = WithholdingSeverity.MODERATE
        else:
            severity = WithholdingSeverity.MINOR

        # Calculate confidence — v1.5: Reverted to original formula.
        # Previous severity-based splits (v1.2-v1.4) hurt recall badly.
        # Original gives all detections confidence 0.70+ → LLM judge decides
        # TP vs FP now that the prompt bug is fixed (was getting empty data).
        confidence = min(0.95, 0.6 + (len(issues) * 0.1))

        # Build explanation
        issue_types = set(i.issue_type.value for i in issues)
        explanation = (
            f"Detected {len(issues)} withholding issue(s): {', '.join(issue_types)}. "
            f"Critical info retention: {retention_ratio:.1%}"
        )

        # Suggest fix
        if any(i.issue_type == WithholdingType.CRITICAL_OMISSION for i in issues):
            suggested_fix = "Ensure agent passes all critical findings to downstream agents"
        elif any(i.issue_type == WithholdingType.NEGATIVE_SUPPRESSION for i in issues):
            suggested_fix = "Configure agent to report negative findings, not just positive ones"
        else:
            suggested_fix = "Review agent's summarization settings to preserve more detail"

        return WithholdingResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            issues=issues,
            information_retention_ratio=retention_ratio,
            critical_items_found=found,
            critical_items_reported=retained,
            explanation=explanation,
            suggested_fix=suggested_fix,
        )


# Singleton instance
withholding_detector = InformationWithholdingDetector()
