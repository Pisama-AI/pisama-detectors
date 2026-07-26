"""Hallucination detection for LLM agent outputs."""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pisama_detectors._config import get_settings
from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder

settings = get_settings()


@dataclass
class HallucinationResult:
    detected: bool
    confidence: float
    hallucination_type: Optional[str]
    evidence: List[str]
    grounding_score: float
    details: Dict[str, Any] = field(default_factory=dict)
    raw_score: Optional[float] = None
    calibration_info: Optional[Dict[str, Any]] = None


@dataclass
class SourceDocument:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class HallucinationDetector:
    def __init__(
        self,
        grounding_threshold: Optional[float] = None,
        confidence_scaling: float = 1.0,
    ):
        self._embedder = None
        self.grounding_threshold = grounding_threshold or 0.65
        self.confidence_scaling = confidence_scaling
        self.citation_pattern = re.compile(
            r"\[(?P<bracket>\d+)\]|"
            r"\(source:?\s*(?P<source>[^)]+)\)|"
            r"\{\{cite:\s*(?P<template>[^}]+)\}\}",
            re.IGNORECASE,
        )
        self.confidence_phrases = [
            "I'm not sure",
            "I don't know",
            "I cannot confirm",
            "I believe",
            "I think",
            "possibly",
            "perhaps",
            "might be",
            "as far as I know",
            "to my knowledge",
        ]
        self.definitive_phrases = [
            "definitely",
            "certainly",
            "absolutely",
            "always",
            "never",
            "100%",
            "guaranteed",
            "proven fact",
            "undoubtedly",
        ]

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedder()
        return self._embedder

    def _calibrate_confidence(
        self,
        grounding_score: float,
        evidence_count: int,
        has_sources: bool,
        fabrication_score: float,
    ) -> Tuple[float, Dict[str, Any]]:
        """Calibrate confidence based on evidence quality."""
        base_confidence = 0.5

        if has_sources:
            base_confidence += 0.2

        evidence_factor = min(0.15, evidence_count * 0.03)
        base_confidence += evidence_factor

        grounding_factor = (1 - grounding_score) * 0.15
        base_confidence += grounding_factor

        fabrication_factor = (1 - fabrication_score) * 0.1
        base_confidence += fabrication_factor

        calibrated = min(0.99, base_confidence * self.confidence_scaling)

        calibration_info = {
            "base_confidence": round(base_confidence, 4),
            "evidence_count": evidence_count,
            "has_sources": has_sources,
            "grounding_factor": round(grounding_factor, 4),
            "fabrication_factor": round(fabrication_factor, 4),
        }

        return round(calibrated, 4), calibration_info

    def detect_hallucination(
        self,
        output: str,
        sources: Optional[List[SourceDocument]] = None,
        context: Optional[str] = None,
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> HallucinationResult:
        evidence: List[str] = []
        details: Dict[str, Any] = {}

        grounding_score = 1.0
        hallucination_type = None

        # v1.6: Strip Q&A format prefixes — "Question: ... Answer: ..." output
        # should only be evaluated on the answer part. The question is
        # context, not a claim, and drags down grounding score.
        import re as _re

        _qa_match = _re.search(r"(?:^|\n)\s*Answer:\s*", output, _re.IGNORECASE)
        if _qa_match:
            output = output[_qa_match.end() :].strip()

        if sources:
            source_score, source_evidence = self._check_source_grounding(output, sources)
            grounding_score = min(grounding_score, source_score)
            evidence.extend(source_evidence)
            details["source_grounding_score"] = source_score

            # v1.5: Track short-source flag for diagnostics. The effective
            # threshold adjustment is handled below.
            all_sources_short = all(len(s.content) < 200 for s in sources)
            if all_sources_short:
                details["short_sources"] = True

        if context:
            context_score, context_evidence = self._check_context_consistency(output, context)
            grounding_score = min(grounding_score, context_score)
            evidence.extend(context_evidence)
            details["context_consistency_score"] = context_score

        if tool_results:
            tool_score, tool_evidence = self._check_tool_result_consistency(output, tool_results)
            grounding_score = min(grounding_score, tool_score)
            evidence.extend(tool_evidence)
            details["tool_result_score"] = tool_score

        # v1.7: Instructional/educational text gets a fabrication pass —
        # how-to guides, tutorials, and explanations legitimately introduce
        # new terms/concepts that aren't in the source.
        import re as _re2

        instructional_patterns = [
            r"\b(?:here\'?s how|to do this|the steps are|you can|first,?\s+\w+,?\s+then)\b",
            r"\b(?:for example|such as|in this case|this means)\b",
            r"\b(?:install|import|configure|set up|create a|run the)\b",
        ]
        is_instructional = (
            sum(1 for p in instructional_patterns if _re2.search(p, output, _re2.IGNORECASE)) >= 2
        )
        fabrication_boost = 0.15 if is_instructional else 0.0

        fabrication_score, fabrication_evidence = self._detect_fabricated_facts(output)
        fabrication_score = min(1.0, fabrication_score + fabrication_boost)
        if fabrication_score < 0.95:
            evidence.extend(fabrication_evidence)
            details["fabrication_indicators"] = fabrication_evidence
            if fabrication_score < 0.7:
                grounding_score = min(grounding_score, fabrication_score)

        # v1.7: Self-negotiation detection — agent identifies issues then
        # rationalizes them away. Pattern from Anthropic harness research:
        # evaluator agents "identify legitimate issues, then talk themselves
        # into deciding they weren't a big deal and approve the work anyway."
        self_negotiation_score, self_negotiation_evidence = self._detect_self_negotiation(output)
        if self_negotiation_score > 0:
            evidence.extend(self_negotiation_evidence)
            details["self_negotiation_score"] = self_negotiation_score
            # Drag grounding down when strong self-negotiation detected
            if self_negotiation_score >= 0.5:
                grounding_score = min(grounding_score, 1.0 - self_negotiation_score * 0.4)

        # v1.4: Make citation check optional — if no citation pattern exists
        # in the output AND sources are plain text strings, skip the citation
        # check entirely to avoid FP on simple Q&A.
        has_citation_patterns = bool(self.citation_pattern.search(output))
        sources_are_plain = (
            sources is not None and all(len(s.content) < 200 and not s.metadata for s in sources)
            if sources
            else False
        )
        if has_citation_patterns or not sources_are_plain:
            citation_score, citation_evidence = self._check_citation_validity(output, sources)
            if citation_evidence:
                evidence.extend(citation_evidence)
                details["citation_issues"] = citation_evidence
                grounding_score = min(grounding_score, citation_score)

        confidence_score = self._analyze_confidence_calibration(output)
        details["confidence_calibration"] = confidence_score

        detected = grounding_score < self.grounding_threshold

        if detected:
            if details.get("source_grounding_score", 1.0) < 0.5:
                hallucination_type = "ungrounded_claim"
            elif details.get("tool_result_score", 1.0) < 0.5:
                hallucination_type = "tool_result_contradiction"
            elif details.get("citation_issues"):
                hallucination_type = "invalid_citation"
            elif fabrication_score < 0.5:
                hallucination_type = "fabricated_fact"
            elif details.get("self_negotiation_score", 0) >= 0.5:
                hallucination_type = "self_negotiation"
            else:
                hallucination_type = "general_hallucination"

        raw_score = grounding_score
        calibrated_confidence, calibration_info = self._calibrate_confidence(
            grounding_score=grounding_score,
            evidence_count=len(evidence),
            has_sources=sources is not None and len(sources) > 0,
            fabrication_score=fabrication_score,
        )

        return HallucinationResult(
            detected=detected,
            confidence=calibrated_confidence,
            hallucination_type=hallucination_type,
            evidence=evidence,
            grounding_score=grounding_score,
            details=details,
            raw_score=raw_score,
            calibration_info=calibration_info,
        )

    def _check_source_grounding(
        self,
        output: str,
        sources: List[SourceDocument],
    ) -> Tuple[float, List[str]]:
        if not sources:
            return 1.0, []

        evidence = []

        output_sentences = self._split_sentences(output)
        if not output_sentences:
            return 1.0, []

        source_texts = [s.content for s in sources]
        sentence_similarities, sentence_contradictions = self._source_similarity_scores(
            output_sentences,
            source_texts,
        )

        grounded_count = 0
        hedging_words = {
            "however",
            "though",
            "note",
            "should",
            "could",
            "might",
            "further",
            "additionally",
            "also",
            "needed",
            "workup",
            "suggest",
            "recommend",
            "consider",
            "possible",
            "likely",
            "typically",
            "generally",
            "often",
            "usually",
            "may",
        }
        # Phrases that indicate the agent is honestly admitting uncertainty
        uncertainty_phrases = [
            "i don't have",
            "i don't know",
            "i'm not sure",
            "i cannot confirm",
            "not available",
            "no information",
            "not specified",
            "not mentioned",
            "no details",
            "no specific",
            "unclear whether",
            "unable to confirm",
            "based on the available",
            "from what i can see",
            "it appears that",
            "it seems",
            "this suggests",
        ]
        for i, max_sim in enumerate(sentence_similarities):
            sent_lower = output_sentences[i].lower()
            sent_stripped = sent_lower.lstrip()

            # Sentences admitting uncertainty are inherently grounded (honest)
            is_uncertainty = any(p in sent_lower for p in uncertainty_phrases)
            if is_uncertainty:
                grounded_count += 1
                continue

            # Sentences that hedge, qualify, or continue a previous thought
            is_hedging = any(w in sent_lower for w in hedging_words)
            # Transition-initial sentences ("Then, ...", "Next, ...", etc.)
            # are continuations that don't need strict grounding
            transition_starts = (
                "then,",
                "then ",
                "next,",
                "next ",
                "finally,",
                "finally ",
                "after that,",
                "second,",
                "third,",
                "in summary,",
                "overall,",
                "in conclusion,",
            )
            is_transition = any(sent_stripped.startswith(t) for t in transition_starts)
            grounding_bar = 0.40 if (is_hedging or is_transition) else 0.55

            if max_sim >= grounding_bar:
                grounded_count += 1
            elif sentence_contradictions[i]:
                evidence.append(
                    f"Claim contradicts provided source: '{output_sentences[i][:80]}...'"
                )
            elif max_sim < 0.4:
                evidence.append(f"Ungrounded claim: '{output_sentences[i][:80]}...'")

        embedding_ratio = grounded_count / len(output_sentences)

        # Entity/numerical cross-check: penalize numbers and named entities
        # in output that don't appear in any source text.
        source_blob_lower = " ".join(source_texts).lower()
        source_blob_original = " ".join(source_texts)
        novelty_penalty = self._compute_novelty_penalty(
            self._strip_citations(output),
            source_blob_lower,
            source_blob_original,
        )
        if novelty_penalty > 0:
            evidence.append(
                f"Output introduces {novelty_penalty:.0%} novel numbers/entities not in sources"
            )

        grounding_ratio = max(0.0, embedding_ratio - novelty_penalty)
        return grounding_ratio, evidence

    def _source_similarity_scores(
        self,
        output_sentences: List[str],
        source_texts: List[str],
    ) -> Tuple[List[float], List[bool]]:
        source_clauses = [
            clause
            for source in source_texts
            for clause in self._split_source_clauses(source)
        ]
        embedder = self.embedder
        clean_output_sentences = [self._strip_citations(sentence) for sentence in output_sentences]
        output_embeddings = source_embeddings = None
        if embedder is not None and source_clauses:
            embeddings = embedder.encode(clean_output_sentences + source_clauses)
            output_embeddings = embeddings[: len(output_sentences)]
            source_embeddings = embeddings[len(output_sentences) :]

        sentence_similarities: List[float] = []
        sentence_contradictions: List[bool] = []
        for index, sentence in enumerate(output_sentences):
            clean_sentence = clean_output_sentences[index]
            similarities = []
            contradicted = False
            for source_index, source_clause in enumerate(source_clauses):
                if self._claims_contradict(clean_sentence, source_clause):
                    contradicted = True
                    similarities.append(0.0)
                    continue

                lexical_score, shared_terms = self._claim_support_score(
                    clean_sentence,
                    source_clause,
                )
                if lexical_score <= 0:
                    similarities.append(0.0)
                    continue

                if embedder is not None:
                    assert output_embeddings is not None
                    assert source_embeddings is not None
                    semantic_score = embedder.similarity(
                        output_embeddings[index],
                        source_embeddings[source_index],
                    )
                    if lexical_score >= 0.6 or shared_terms >= 2:
                        lexical_score = max(lexical_score, semantic_score)
                similarities.append(lexical_score)
            sentence_similarities.append(max(similarities, default=0.0))
            sentence_contradictions.append(contradicted)
        return sentence_similarities, sentence_contradictions

    _GROUNDING_STOP_WORDS = frozenset(
        "a all an and any are as at be been being by can could did do does every for from "
        "had has have in is it may might must of on only or shall should that the their "
        "there these they this those to was were will with would".split()
    )
    _CLAIM_TERM_ALIASES = {
        "accessible": "access",
        "available": "access",
        "located": "locate",
        "location": "locate",
        "needed": "require",
        "needs": "require",
        "required": "require",
        "requires": "require",
        "using": "use",
        "uses": "use",
        "increased": "increase",
        "increases": "increase",
        "rising": "increase",
        "rose": "increase",
        "decreased": "decrease",
        "decreases": "decrease",
        "falling": "decrease",
        "fell": "decrease",
    }
    _CLAIM_ANTONYMS = (
        frozenset({"increase", "decrease"}),
        frozenset({"allow", "deny"}),
        frozenset({"enable", "disable"}),
        frozenset({"include", "exclude"}),
        frozenset({"before", "after"}),
        frozenset({"above", "below"}),
    )
    _CLAIM_NEGATION = re.compile(
        r"\b(?:not|never|no|none|without|cannot|can't|doesn't|does\s+not|"
        r"do\s+not|isn't|is\s+not|aren't|are\s+not|must\s+not|should\s+not)\b",
        re.IGNORECASE,
    )
    _CLAIM_AUDIENCE = (
        r"(?:users?|customers?|admins?|administrators?|members?|tenants?|employees?|"
        r"viewers?|visitors?|people|persons?|accounts?|roles?)"
    )
    _CLAIM_BROAD_SCOPE = re.compile(
        rf"\b(?:all|every|any)\s+{_CLAIM_AUDIENCE}\b",
        re.IGNORECASE,
    )
    _CLAIM_RESTRICTED_SCOPE = re.compile(
        rf"(?:\bonly\s+(?:to\s+)?(?:selected\s+)?{_CLAIM_AUDIENCE}\b|"
        rf"\bto\s+{_CLAIM_AUDIENCE}\s+only\b)",
        re.IGNORECASE,
    )
    _CLAIM_MODAL_ACTION = re.compile(
        r"\b(?:must|should|shall|can|cannot|can't|could|may|will|"
        r"(?:do|does|is|are)\s+not|never)\s+"
        r"(?:not\s+|never\s+)?(?:be\s+)?(?P<action>[a-z]+)",
        re.IGNORECASE,
    )
    _CLAIM_DECLARATIVE_ACTION = re.compile(
        r"^\s*(?:the\s+)?[a-z][a-z0-9_-]*\s+"
        r"(?:(?:is|are|does|do|will)\s+(?:not\s+)?)?(?P<action>[a-z]+)",
        re.IGNORECASE,
    )

    @classmethod
    def _claim_stem(cls, word: str) -> str:
        if word in cls._CLAIM_TERM_ALIASES:
            return cls._CLAIM_TERM_ALIASES[word]
        if word.endswith("ies") and len(word) > 4:
            return f"{word[:-3]}y"
        if word.endswith("ing") and len(word) > 5:
            return word[:-3]
        if word.endswith("ed") and len(word) > 4:
            return word[:-2]
        if word.endswith("es") and len(word) > 4:
            return word[:-2]
        if word.endswith("s") and len(word) > 3:
            return word[:-1]
        return word

    @classmethod
    def _claim_terms(cls, text: str) -> set[str]:
        return {
            cls._claim_stem(token)
            for token in re.findall(r"[a-z]+|\d+(?:\.\d+)?", text.casefold())
            if token not in cls._GROUNDING_STOP_WORDS
        }

    @classmethod
    def _claim_action(cls, text: str) -> Optional[str]:
        modal = cls._CLAIM_MODAL_ACTION.search(text)
        if modal is not None:
            action = modal.group("action").casefold()
            if action not in {"at", "in", "of", "on", "to"}:
                return cls._claim_stem(action)
        declarative = cls._CLAIM_DECLARATIVE_ACTION.search(text)
        if declarative is not None:
            action = declarative.group("action").casefold()
            if action not in {"at", "in", "of", "on", "to"}:
                return cls._claim_stem(action)
        return None

    @classmethod
    def _claim_numbers(cls, text: str) -> set[str]:
        return set(re.findall(r"(?<![a-z0-9])\d+(?:[.,]\d+)*(?:%|[kmb])?", text.casefold()))

    @classmethod
    def _numbers_match(cls, claim: str, source: str) -> bool:
        claim_numbers = cls._claim_numbers(claim)
        if not claim_numbers:
            return True
        source_numbers = cls._claim_numbers(source)
        return claim_numbers <= source_numbers

    @classmethod
    def _claim_pair_relevance(cls, claim: str, source: str) -> float:
        claim_terms = cls._claim_terms(claim) - cls._claim_numbers(claim)
        source_terms = cls._claim_terms(source) - cls._claim_numbers(source)
        if not claim_terms or not source_terms:
            return 0.0
        claim_action = cls._claim_action(claim)
        source_action = cls._claim_action(source)
        if claim_action and source_action and claim_action != source_action:
            if frozenset({claim_action, source_action}) not in cls._CLAIM_ANTONYMS:
                return 0.0
        return len(claim_terms & source_terms) / max(len(claim_terms), len(source_terms))

    @classmethod
    def _claims_contradict(cls, claim: str, source: str) -> bool:
        overlap = cls._claim_pair_relevance(claim, source)
        if overlap < 0.5:
            return False

        claim_action = cls._claim_action(claim)
        source_action = cls._claim_action(source)
        opposite_action = (
            claim_action is not None
            and source_action is not None
            and frozenset({claim_action, source_action}) in cls._CLAIM_ANTONYMS
        )
        opposite_polarity = bool(cls._CLAIM_NEGATION.search(claim)) != bool(
            cls._CLAIM_NEGATION.search(source)
        )
        opposite_scope = (
            bool(cls._CLAIM_BROAD_SCOPE.search(claim))
            and bool(cls._CLAIM_RESTRICTED_SCOPE.search(source))
        ) or (
            bool(cls._CLAIM_RESTRICTED_SCOPE.search(claim))
            and bool(cls._CLAIM_BROAD_SCOPE.search(source))
        )
        numeric_conflict = not cls._numbers_match(claim, source)
        return opposite_action or opposite_polarity or opposite_scope or numeric_conflict

    def _strip_citations(self, text: str) -> str:
        return " ".join(self.citation_pattern.sub("", text).split())

    @classmethod
    def _split_source_clauses(cls, text: str) -> List[str]:
        clauses = re.split(
            r"(?<=[.!?])\s+|[;\n]+|\b(?:and|but)\s+"
            r"(?=(?:the|this|that|it|must|should|shall|can|cannot|do|does|never)\b)",
            text,
            flags=re.IGNORECASE,
        )
        return [clause.strip() for clause in clauses if len(clause.strip()) >= 3]

    def _claim_support_score(self, claim: str, source: str) -> Tuple[float, int]:
        clean_claim = self._strip_citations(claim)
        claim_terms = self._claim_terms(clean_claim)
        source_terms = self._claim_terms(source)
        if not claim_terms or not source_terms or not self._numbers_match(clean_claim, source):
            return 0.0, 0

        shared = claim_terms & source_terms
        coverage = len(shared) / len(claim_terms)
        normalized_claim = " ".join(
            re.findall(r"[a-z]+|\d+(?:\.\d+)?", clean_claim.casefold())
        )
        normalized_source = " ".join(
            re.findall(r"[a-z]+|\d+(?:\.\d+)?", source.casefold())
        )
        if normalized_claim and normalized_claim in normalized_source:
            return 1.0, len(shared)
        if coverage >= 0.6 and (len(shared) >= 2 or coverage >= 0.8):
            return coverage, len(shared)
        return 0.0, len(shared)

    @staticmethod
    def _compute_novelty_penalty(
        output: str, source_blob_lower: str, source_blob_original: str = ""
    ) -> float:
        """Compute a penalty for numbers and proper nouns in output absent from sources."""
        # v1.3: Improved number extraction — also captures unit-attached numbers
        # like "10mg", "$500K", "25%".  The original \b\d[\d,.]*\b missed these
        # because \b requires a word-boundary between digits and letters.
        _num_pat = r"(?:\$)?\d[\d,.]*[KMBkmb%]?"
        output_numbers = set(re.findall(_num_pat, output))
        source_numbers = set(re.findall(_num_pat, source_blob_lower))

        # v1.3: Approximate number matching — a "novel" number that is within
        # 5% of a source number (e.g. 99 vs 99.2, rounding) is not truly novel.
        def _parse_num(s: str) -> float | None:
            cleaned = s.strip("$%KMBkmb,")
            try:
                val = float(cleaned)
                if s.endswith(("K", "k")):
                    val *= 1000
                elif s.endswith(("M", "m")):
                    val *= 1_000_000
                elif s.endswith(("B", "b")):
                    val *= 1_000_000_000
                return val
            except ValueError:
                return None

        def _has_approx_match(num_str: str) -> bool:
            val = _parse_num(num_str)
            if val is None:
                return False
            for src in source_numbers:
                src_val = _parse_num(src)
                if src_val is not None and src_val != 0:
                    if abs(val - src_val) / abs(src_val) < 0.05:
                        return True
            return False

        novel_numbers = output_numbers - source_numbers
        # Ignore trivially small numbers (1-digit) and years already in source
        novel_numbers = {n for n in novel_numbers if len(n) > 1}
        # v1.3: Remove approximate matches (rounding, unit conversion)
        novel_numbers = {n for n in novel_numbers if not _has_approx_match(n)}

        # Extract capitalized multi-word names (potential proper nouns)
        # Use original-case source text for name extraction (regex needs uppercase)
        output_names = set(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", output))
        source_names = set(
            re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", source_blob_original or source_blob_lower)
        )

        # v1.3: Also extract ALL-CAPS acronyms (AWS, GKE, EKS, etc.)
        output_acronyms = set(re.findall(r"\b[A-Z]{2,}\b", output))
        source_acronyms = set(
            re.findall(r"\b[A-Z]{2,}\b", source_blob_original or source_blob_lower)
        )
        novel_acronyms = output_acronyms - source_acronyms

        # Check lowercase match and substring containment (e.g. "San Francisco"
        # in source should match "San Francisco Bay Area" in output).
        source_names_lower = {n.lower() for n in source_names}

        def _name_is_known(name: str) -> bool:
            nl = name.lower()
            if nl in source_names_lower:
                return True
            # Check if any source name is a prefix/substring of the output name
            for sn in source_names_lower:
                if sn in nl or nl in sn:
                    return True
            return False

        novel_names = {n for n in output_names if not _name_is_known(n)}

        total_claims = max(1, len(output_numbers) + len(output_names) + len(output_acronyms))
        novel_count = len(novel_numbers) + len(novel_names) + len(novel_acronyms)
        if novel_count == 0:
            return 0.0
        # Penalty proportional to fraction of novel entities, scaled aggressively.
        # v1.1: Slightly moderated from original — require 2+ for full floor penalty.
        # v1.3: Single novel entity floor lowered to 0.06 (from 0.10) to reduce FP.
        base_penalty = (novel_count / total_claims) * 0.8
        if novel_count >= 2:
            floor_penalty = min(0.7, novel_count * 0.13)
        else:
            # Single novel entity: lower floor penalty to reduce FP
            floor_penalty = 0.06
        return min(0.7, max(base_penalty, floor_penalty))

    def _check_context_consistency(
        self,
        output: str,
        context: str,
    ) -> Tuple[float, List[str]]:
        evidence = []

        if self.embedder is None:
            output_tokens = set(re.findall(r"\b[a-z0-9]+\b", output.lower()))
            context_tokens = set(re.findall(r"\b[a-z0-9]+\b", context.lower()))
            if not output_tokens or not context_tokens:
                return 0.0, ["Output or context has no comparable tokens"]
            similarity = len(output_tokens & context_tokens) / len(output_tokens | context_tokens)
            if similarity < 0.4:
                evidence.append("Output has low lexical similarity to context")
            return similarity, evidence

        output_emb = self.embedder.encode(output)
        context_emb = self.embedder.encode(context)

        similarity = self.embedder.similarity(output_emb, context_emb)

        if similarity < 0.4:
            evidence.append("Output has low semantic similarity to context")

        return similarity, evidence

    def _check_tool_result_consistency(
        self,
        output: str,
        tool_results: List[Dict[str, Any]],
    ) -> Tuple[float, List[str]]:
        evidence = []
        output.lower()

        contradictions = 0.0
        total_checks = 0

        for result in tool_results:
            tool_name = result.get("tool", "unknown")
            tool_output = str(result.get("result", ""))

            if not tool_output:
                continue

            total_checks += 1

            numbers_in_tool = re.findall(r"\b\d+\.?\d*\b", tool_output)
            for num in numbers_in_tool[:5]:
                if num in output:
                    continue
                try:
                    num_val = float(num)
                    if num_val > 100:
                        pattern = rf"\b{int(num_val)}\b"
                        if not re.search(pattern, output):
                            contradictions += 0.5
                            evidence.append(
                                f"Tool '{tool_name}' returned {num} but value not reflected in output"
                            )
                except ValueError:
                    pass

        if total_checks == 0:
            return 1.0, []

        consistency_score = max(0, 1 - (contradictions / total_checks))
        return consistency_score, evidence

    def _detect_self_negotiation(self, output: str) -> Tuple[float, List[str]]:
        """Detect self-negotiation: agent identifies issues then dismisses them.

        Pattern: evaluator identifies problems but rationalizes them as acceptable.
        From Anthropic: "agents identify legitimate issues, then talk themselves
        into deciding they weren't a big deal and approve the work anyway."
        """
        evidence = []
        score = 0.0

        lower = output.lower()

        # Problem-acknowledgment phrases
        problem_phrases = [
            "issue",
            "bug",
            "error",
            "problem",
            "flaw",
            "defect",
            "failure",
            "missing",
            "incorrect",
            "broken",
            "not working",
            "doesn't work",
            "not implemented",
            "incomplete",
            "wrong",
        ]
        # Dismissal phrases that follow problem acknowledgment
        dismissal_phrases = [
            "however this is acceptable",
            "but this is minor",
            "not a big deal",
            "shouldn't be a concern",
            "can be ignored",
            "within acceptable",
            "doesn't affect the overall",
            "still meets the requirements",
            "overall the quality is",
            "despite this, I would still",
            "I'll approve",
            "passing this",
            "while not perfect",
            "good enough",
            "the issues are minor",
            "these are edge cases",
            "negligible impact",
            "won't affect most users",
        ]

        problem_count = sum(1 for p in problem_phrases if p in lower)
        dismissal_count = sum(1 for d in dismissal_phrases if d in lower)

        if problem_count >= 2 and dismissal_count >= 1:
            score = min((problem_count * 0.15 + dismissal_count * 0.25), 1.0)
            evidence.append(
                f"Self-negotiation: {problem_count} problems identified but "
                f"{dismissal_count} dismissal patterns found"
            )

        # High self-rating despite issues
        rating_match = re.search(r"(?:score|rating|grade)[:\s]*(\d+)\s*/\s*(\d+)", lower)
        if rating_match and problem_count >= 2:
            given = int(rating_match.group(1))
            total = int(rating_match.group(2))
            if total > 0 and given / total >= 0.8:
                score = max(score, 0.7)
                evidence.append(
                    f"High self-rating ({given}/{total}) despite {problem_count} identified issues"
                )

        return score, evidence

    def _detect_fabricated_facts(self, output: str) -> Tuple[float, List[str]]:
        evidence = []
        score = 1.0

        # (pattern, description, penalty) — higher penalty for strong signals
        specific_patterns = [
            (r"founded in \d{4}", "Specific founding year", 0.1),
            (r"according to (?:a )?\d{4} (?:study|report|survey)", "Specific study reference", 0.1),
            (
                r"\d+(?:\.\d+)?% of (?:people|users|companies|developers|workers)",
                "Specific percentage statistic",
                0.1,
            ),
            (r"(?:Dr\.|Professor) [A-Z][a-z]+ [A-Z][a-z]+", "Named expert", 0.15),
            (r"published in (?:the )?[A-Z][a-zA-Z\s]+ Journal", "Journal reference", 0.15),
            # v1.6: Journal/publication fabrication — catch fake journal names
            (
                r"(?:Journal|Review|Proceedings) of [A-Z][a-zA-Z\s]{5,}",
                "Journal/publication name",
                0.12,
            ),
            (
                r"(?:International|American|European|Global) (?:Journal|Review|Institute)",
                "International publication",
                0.12,
            ),
            (
                r"\b(?:meta-analysis|longitudinal study|survey)\b.*\bcovering\s+[\d,]+",
                "Study scope claim",
                0.1,
            ),
            (r"\b\d{4}\s+(?:report|study|survey|analysis)\b", "Year-specific study", 0.08),
            # Fabricated URL patterns
            (
                r"https?://(?:www\.)?[a-z]+(?:-[a-z]+)+\.(?:com|org|io)/[a-z0-9/-]+",
                "Specific URL",
                0.1,
            ),
            (
                r"https?://(?:docs|api|support)\.[a-z]+\.(?:com|org)/[a-z0-9/-]+",
                "Documentation URL",
                0.1,
            ),
            (r"(?:visit|see|check|refer to)\s+https?://\S+", "Referenced URL", 0.1),
            # Causal/conclusion fabrication — strong signals, higher penalties
            (r"directly (?:drove|caused|led to|resulted in)", "Fabricated causal claim", 0.2),
            (
                r"exceeding (?:internal |expected )?(?:projections|estimates|targets)",
                "Fabricated comparison",
                0.2,
            ),
            (r"scientists predict|experts forecast|analysts expect", "Fabricated prediction", 0.2),
            (
                r"which (?:scientists|researchers|experts) (?:predict|expect|forecast)",
                "Attributed prediction",
                0.2,
            ),
        ]

        fabrication_count = 0
        for pattern, desc, penalty in specific_patterns:
            matches = re.findall(pattern, output)
            if matches:
                fabrication_count += len(matches)
                for match in matches[:2]:
                    evidence.append(f"Potential fabrication ({desc}): '{match}'")
                    score -= penalty

        # v1.6: Multiple fabrication signals compound — 3+ signals is strong evidence
        if fabrication_count >= 3:
            evidence.append(
                f"Multiple fabrication indicators ({fabrication_count}) — high hallucination risk"
            )
            score -= 0.15

        definitive_count = sum(
            1 for phrase in self.definitive_phrases if phrase.lower() in output.lower()
        )
        if definitive_count >= 3:
            evidence.append(
                f"High definitiveness ({definitive_count} definitive phrases) may indicate overconfidence"
            )
            score -= 0.15

        # v1.5: Superlative/extreme claim detection — catches fabricated guarantees
        extreme_patterns = [
            (r"\b(?:guarantee[ds]?|guaranteed)\b", "Guarantee claim"),
            (r"\b(?:unlimited|infinite)\b", "Unlimited claim"),
            (
                r"\b(?:personally|officially)\s+(?:approved|confirmed|announced)\b",
                "Personal authority claim",
            ),
            (r"\b\d{3,}%\b", "Extreme percentage (100%+)"),
            (
                r"\b(?:free|no[\s-]cost|zero[\s-]cost)\s+(?:for|to)\s+(?:all|every|international)\b",
                "Universal free claim",
            ),
            (r"\b(?:immediately|effective\s+immediately)\b", "Immediate effect claim"),
            (
                r"\b(?:replaced|replacing|abolish|declared)\s+.*\b(?:currency|dollar|official)\b",
                "Institutional change claim",
            ),
        ]
        extreme_count = 0
        for pattern, desc in extreme_patterns:
            if re.search(pattern, output, re.IGNORECASE):
                extreme_count += 1
                evidence.append(f"Extreme claim ({desc})")
                score -= 0.15
        if extreme_count >= 2:
            evidence.append(f"Multiple extreme claims ({extreme_count}) — high fabrication risk")
            score -= 0.1

        # Cap total fabrication penalty at 0.5 to prevent over-detection
        return max(0.5, score) if fabrication_count <= 1 else max(0, score), evidence

    def _check_citation_validity(
        self,
        output: str,
        sources: Optional[List[SourceDocument]],
    ) -> Tuple[float, List[str]]:
        evidence = []

        citations = list(self.citation_pattern.finditer(output))
        if not citations:
            return 1.0, []

        if sources is None:
            if citations:
                evidence.append(
                    f"Output contains {len(citations)} citations but no sources provided"
                )
                return 0.5, evidence
            return 1.0, []

        num_sources = len(sources)
        source_labels = {
            str(value).strip().casefold()
            for source in sources
            for key in ("id", "label", "name", "title", "source", "url")
            if (value := source.metadata.get(key)) is not None
        }
        invalid_citations = 0

        for match in citations:
            reference = next(
                value
                for value in (
                    match.group("bracket"),
                    match.group("source"),
                    match.group("template"),
                )
                if value is not None
            ).strip()
            if reference.isdigit():
                if not 1 <= int(reference) <= num_sources:
                    invalid_citations += 1
                    evidence.append(f"Citation {match.group(0)} references non-existent source")
            elif reference.casefold() not in source_labels:
                invalid_citations += 1
                evidence.append(f"Citation {match.group(0)} references non-existent source")

        if invalid_citations > 0:
            return 0.0, evidence

        return 1.0, []

    def _analyze_confidence_calibration(self, output: str) -> float:
        output_lower = output.lower()

        uncertainty_count = sum(
            1 for phrase in self.confidence_phrases if phrase.lower() in output_lower
        )
        definitive_count = sum(
            1 for phrase in self.definitive_phrases if phrase.lower() in output_lower
        )

        if definitive_count > uncertainty_count + 2:
            return 0.6
        elif uncertainty_count > definitive_count:
            return 0.9
        return 0.75

    def _split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if len(s.strip()) >= 3]


hallucination_detector = HallucinationDetector()
