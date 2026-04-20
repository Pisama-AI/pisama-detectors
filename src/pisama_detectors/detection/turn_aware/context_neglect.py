"""
F7: Context Neglect Detection
=============================

Detects when agents improperly utilize context from:
1. Previous turns in the conversation
2. User instructions and requirements
3. Tool/system outputs
"""

import re
import logging
from typing import List, Optional, Dict, Any

from ._base import (
    TurnSnapshot,
    TurnAwareDetector,
    TurnAwareDetectionResult,
    TurnAwareSeverity,
)
from ._embedding_mixin import EmbeddingMixin

logger = logging.getLogger(__name__)


class TurnAwareContextNeglectDetector(EmbeddingMixin, TurnAwareDetector):
    """Detects F7: Context Neglect across conversation turns.

    Analyzes whether agents properly utilize context from:
    1. Previous turns in the conversation
    2. User instructions and requirements
    3. Tool/system outputs

    Uses accumulated context tracking to detect when information
    is "lost" or ignored as the conversation progresses.

    Enhanced with semantic embeddings (v2.0):
    - Embedding-based context utilization scoring
    - Semantic similarity to detect topic alignment even with different words
    - Information density analysis for substantive responses

    Based on MAST research: FM-2.2 Context Neglect (Category 2: Execution Failures)
    """

    name = "TurnAwareContextNeglectDetector"
    version = "2.0"  # Semantic enhancement
    supported_failure_modes = ["F7"]

    # Code patterns that indicate a code response
    CODE_PATTERNS = [
        "def ", "class ", "import ", "from ", "function ",
        "const ", "let ", "var ", "return ", "if (", "if(",
        "for (", "for(", "while ", "=>", "```", "{", "}",
        "self.", "this.", "async ", "await ",
    ]

    # Explicit neglect indicators - agent explicitly ignoring or misunderstanding
    # Enhanced for MAST benchmark patterns
    NEGLECT_INDICATORS = [
        "instead", "rather than", "not what you asked",
        "different topic", "unrelated", "i'll analyze",
        "let me look at", "i'll check",
        # Added for better MAST recall
        "i'll focus on", "let me try", "actually",
        "ignore", "skip", "disregard", "missing the point",
        "that's not", "not related", "off-topic",
        "weather", "temperature",  # Common wrong-topic responses
    ]

    def __init__(
        self,
        utilization_threshold: float = 0.08,  # Raised from 0.03 to reduce FPs (16.2% FPR)
        min_context_length: int = 40,  # Reduced to catch shorter context issues
        check_user_instructions: bool = True,
        check_tool_outputs: bool = True,
        require_explicit_neglect: bool = False,  # Enable implicit detection for F7
        min_issues_to_flag: int = 2,  # Raised from 1 to reduce FPs
    ):
        self.utilization_threshold = utilization_threshold
        self.min_context_length = min_context_length
        self.check_user_instructions = check_user_instructions
        self.check_tool_outputs = check_tool_outputs
        self.require_explicit_neglect = require_explicit_neglect
        self.min_issues_to_flag = min_issues_to_flag

    def detect(
        self,
        turns: List[TurnSnapshot],
        conversation_metadata: Optional[Dict[str, Any]] = None,
    ) -> TurnAwareDetectionResult:
        """Detect context neglect across conversation turns."""
        if len(turns) < 2:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.0,
                failure_mode=None,
                explanation="Too few turns to analyze",
                detector_name=self.name,
            )

        # Separate turns by participant type
        user_turns = [t for t in turns if t.participant_type == "user"]
        agent_turns = [t for t in turns if t.participant_type == "agent"]
        tool_turns = [t for t in turns if t.participant_type == "tool"]
        system_turns = [t for t in turns if t.participant_type == "system"]

        neglect_issues = []
        affected_turns = []

        # Detect if this is a multi-agent conversation (no user turns, multiple agents)
        is_multi_agent = not user_turns and len(agent_turns) >= 3

        if is_multi_agent:
            # MULTI-AGENT MODE: Check if agents ignore accumulated context
            # In ChatDev/MetaGPT, context neglect = agent ignores key info from prior turns
            neglect_issues.extend(self._check_multi_agent_context_neglect(turns, agent_turns))
            for issue in neglect_issues:
                affected_turns.append(issue.get("turn", 0))
        else:
            # TRADITIONAL MODE: User-agent conversation
            # Handle systems where task comes from system prompt
            synthetic_user_turns = []
            agents_to_check = agent_turns
            if user_turns:
                synthetic_user_turns = user_turns
            elif system_turns and any(len(t.content) > 50 for t in system_turns):
                synthetic_user_turns = [t for t in system_turns if len(t.content) > 50]
            elif agent_turns:
                synthetic_user_turns = [agent_turns[0]]
                agents_to_check = agent_turns[1:]

            # Check 1: Agent responses vs user instructions
            if self.check_user_instructions and synthetic_user_turns and agents_to_check:
                for agent_turn in agents_to_check:
                    prior_user_turns = [
                        u for u in synthetic_user_turns
                        if u.turn_number < agent_turn.turn_number
                    ]
                    if not prior_user_turns:
                        continue

                    immediate_user = max(prior_user_turns, key=lambda u: u.turn_number)
                    user_context = immediate_user.content

                    if self._is_code_response(agent_turn.content):
                        if self._is_code_request(user_context):
                            continue

                    has_explicit_neglect = self._has_explicit_neglect(
                        user_context, agent_turn.content
                    )

                    if has_explicit_neglect:
                        neglect_issues.append({
                            "type": "explicit_neglect",
                            "turn": agent_turn.turn_number,
                            "description": f"Agent turn {agent_turn.turn_number} explicitly ignored user request",
                        })
                        affected_turns.append(agent_turn.turn_number)
                    elif not self.require_explicit_neglect:
                        utilization = self._compute_utilization(
                            user_context, agent_turn.content
                        )
                        if utilization < self.utilization_threshold:
                            neglect_issues.append({
                                "type": "user_instruction_neglect",
                                "turn": agent_turn.turn_number,
                                "utilization": utilization,
                                "description": f"Agent turn {agent_turn.turn_number} poorly utilized user context",
                            })
                            affected_turns.append(agent_turn.turn_number)

        # Check 2: Agent responses vs tool outputs
        if self.check_tool_outputs and tool_turns and agent_turns:
            for agent_turn in agent_turns:
                # Find tool turns immediately before this agent turn
                prior_tool_turns = [
                    t for t in tool_turns
                    if t.turn_number < agent_turn.turn_number
                    and t.turn_number >= agent_turn.turn_number - 3  # Within 3 turns
                ]
                if prior_tool_turns:
                    tool_context = "\n".join(t.content for t in prior_tool_turns)
                    if len(tool_context) >= self.min_context_length:
                        # For tool outputs, check if key data is referenced
                        if self._ignores_tool_data(tool_context, agent_turn.content):
                            neglect_issues.append({
                                "type": "tool_output_neglect",
                                "turn": agent_turn.turn_number,
                                "description": f"Agent turn {agent_turn.turn_number} ignored tool output",
                            })
                            affected_turns.append(agent_turn.turn_number)

        # NOTE: Removed context_drift check - repetition is F5's job, not F7

        # Require minimum issues to reduce false positives
        if len(neglect_issues) < self.min_issues_to_flag:
            return TurnAwareDetectionResult(
                detected=False,
                severity=TurnAwareSeverity.NONE,
                confidence=0.8,
                failure_mode=None,
                explanation=f"Insufficient evidence ({len(neglect_issues)} issue(s), need {self.min_issues_to_flag}+)",
                detector_name=self.name,
            )

        # Determine severity based on number and type of issues
        if len(neglect_issues) >= 3:
            severity = TurnAwareSeverity.SEVERE
        elif len(neglect_issues) >= 2 or any(i["type"] == "explicit_neglect" for i in neglect_issues):
            severity = TurnAwareSeverity.MODERATE
        else:
            severity = TurnAwareSeverity.MINOR

        confidence = min(0.95, 0.5 + len(neglect_issues) * 0.15)

        return TurnAwareDetectionResult(
            detected=True,
            severity=severity,
            confidence=confidence,
            failure_mode="F7",
            explanation=f"Context neglect detected in {len(neglect_issues)} instances",
            affected_turns=list(set(affected_turns)),
            evidence={
                "issues": neglect_issues,
                "total_turns": len(turns),
                "agent_turns": len(agent_turns),
            },
            suggested_fix=(
                "Ensure agent prompts explicitly reference accumulated conversation context. "
                "Add context window management to prevent information loss."
            ),
            detector_name=self.name,
        )

    def _is_code_response(self, content: str) -> bool:
        """Check if response contains code."""
        content_lower = content.lower()
        code_pattern_count = sum(1 for p in self.CODE_PATTERNS if p.lower() in content_lower)
        return code_pattern_count >= 2

    def _is_code_request(self, content: str) -> bool:
        """Check if user is asking for code."""
        code_keywords = [
            "write", "code", "function", "implement", "create",
            "program", "script", "fix", "debug", "add", "python",
            "javascript", "java", "def", "class", "method",
        ]
        content_lower = content.lower()
        return any(kw in content_lower for kw in code_keywords)

    def _has_explicit_neglect(self, user_context: str, agent_response: str) -> bool:
        """Check for explicit signs of ignoring user context."""
        response_lower = agent_response.lower()
        context_lower = user_context.lower()

        # Check if agent mentions doing something else
        for indicator in self.NEGLECT_INDICATORS:
            if indicator in response_lower:
                # Verify it's not the user's correction
                if indicator not in context_lower:
                    return True

        # Check for topic mismatch: user asks about X, agent talks about Y
        # Extract key nouns from both
        user_topics = self._extract_topics(context_lower)
        agent_topics = self._extract_topics(response_lower)

        if user_topics and agent_topics:
            overlap = user_topics & agent_topics
            if len(overlap) == 0 and not self._is_code_response(agent_response):
                # No topic overlap and not a code response = likely neglect
                return True

        return False

    def _extract_topics(self, text: str) -> set:
        """Extract main topic words from text."""
        # Common domain-specific keywords that indicate topic
        # Enhanced for MAST benchmark diversity
        topic_indicators = {
            "sales", "data", "analysis", "report", "weather", "temperature",
            "calculator", "function", "code", "implementation", "database",
            "user", "authentication", "api", "server", "client", "file",
            "error", "bug", "test", "performance", "security", "config",
            # Added for better MAST coverage
            "upload", "download", "login", "register", "password", "email",
            "todo", "task", "list", "game", "chat", "message", "search",
            "product", "order", "cart", "payment", "invoice", "customer",
            "document", "image", "video", "audio", "pdf", "export", "import",
        }
        words = set(text.split())
        return words & topic_indicators

    def _ignores_tool_data(self, tool_output: str, agent_response: str) -> bool:
        """Check if agent ignores important data from tool output."""
        # Extract numbers from tool output
        numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', tool_output))
        response_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', agent_response))

        # If tool output has significant numbers and none appear in response
        if len(numbers) >= 3 and len(numbers & response_numbers) == 0:
            return True

        return False

    def _compute_utilization(self, context: str, output: str) -> float:
        """Compute how much of the context is reflected in the output.

        Enhanced with semantic similarity (v2.0):
        - Uses embedding similarity when available (more accurate)
        - Falls back to word overlap for speed or when unavailable
        """
        # Try semantic similarity first (more accurate for different phrasings)
        similarity = self.semantic_similarity(context, output)
        if similarity >= 0:  # Embeddings available
            return similarity

        # Fallback to keyword-based
        context_words = set(w.lower() for w in context.split() if len(w) > 3)
        output_words = set(w.lower() for w in output.split() if len(w) > 3)

        if not context_words:
            return 1.0

        overlap = context_words & output_words
        return len(overlap) / len(context_words)

    def _check_semantic_context_alignment(
        self,
        user_context: str,
        agent_response: str,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Check if agent response semantically aligns with user context.

        Returns dict with alignment analysis.
        """
        similarity = self.semantic_similarity(user_context, agent_response)

        if similarity < 0:  # Embeddings unavailable
            return {"available": False}

        # Also check information density of response
        response_density = self.compute_information_density(agent_response)

        return {
            "available": True,
            "similarity": similarity,
            "aligned": similarity >= threshold,
            "response_density": response_density,
            "low_density": response_density < 0.3,  # Very sparse response
        }

    def _check_multi_agent_context_neglect(
        self,
        all_turns: List[TurnSnapshot],
        agent_turns: List[TurnSnapshot],
    ) -> List[Dict[str, Any]]:
        """Check for context neglect in multi-agent conversations.

        In multi-agent systems (ChatDev, MetaGPT), F7 occurs when:
        1. Agents don't reference key information from earlier turns
        2. Low semantic coherence between task and later execution
        3. Evidence of "forgotten" context or repeated clarifications

        Detection approach:
        - Track key topic words from task description
        - Check semantic coherence across conversation
        - Detect low information transfer from early to late turns
        """
        issues = []

        if len(agent_turns) < 5:
            return issues

        # Step 1: Extract task topic from early turns (first 2-3 substantive turns)
        early_turns = [t for t in agent_turns[:4] if len(t.content) > 100]
        if not early_turns:
            return issues

        task_content = " ".join(t.content for t in early_turns)
        task_keywords = self._extract_task_keywords(task_content)

        if len(task_keywords) < 3:
            return issues

        # Step 2: Check middle turns for context coherence
        # F7 often manifests in the middle of conversation where agents "forget"
        mid_start = len(agent_turns) // 3
        mid_end = 2 * len(agent_turns) // 3
        middle_turns = agent_turns[mid_start:mid_end] if mid_end > mid_start else []

        for turn in middle_turns:
            if len(turn.content) < 50:
                continue

            turn_lower = turn.content.lower()
            turn_words = set(turn_lower.split())

            # Check keyword overlap with task
            overlap = len(task_keywords & turn_words)
            overlap_ratio = overlap / len(task_keywords)

            # Very low overlap in a substantive turn suggests context neglect
            if len(turn.content) > 200 and overlap_ratio < 0.1:
                # Additional check: is this turn about something completely different?
                if not self._is_code_response(turn.content):
                    issues.append({
                        "type": "low_context_coherence",
                        "turn": turn.turn_number,
                        "description": f"Turn {turn.turn_number} has low coherence with task context",
                        "overlap_ratio": overlap_ratio,
                    })

        # Step 3: Check for explicit neglect indicators
        neglect_indicators = [
            "forgot", "forgotten", "didn't mention", "wasn't clear",
            "misunderstood", "wrong assumption", "actually",
            "wait", "hold on", "let me reconsider", "missed",
        ]

        for i, turn in enumerate(agent_turns[3:], start=3):
            turn_lower = turn.content.lower()
            for indicator in neglect_indicators:
                if indicator in turn_lower:
                    # Check if this seems like catching a mistake
                    context = turn_lower[max(0, turn_lower.find(indicator)-30):turn_lower.find(indicator)+50]
                    if any(x in context for x in ["requirement", "task", "should", "need", "must"]):
                        issues.append({
                            "type": "explicit_neglect_recovery",
                            "turn": turn.turn_number,
                            "description": f"Agent catches forgotten context: '{indicator}'",
                        })
                        break

        # Step 4: Check for re-asking questions (strong signal)
        question_patterns = [
            r"what (?:is|are|should|would)",
            r"how (?:do|should|would|can)",
            r"could you (?:clarify|explain|tell)",
            r"can you (?:clarify|explain|tell)",
        ]

        for i, turn in enumerate(agent_turns[4:], start=4):
            turn_lower = turn.content.lower()
            for pattern in question_patterns:
                if re.search(pattern, turn_lower):
                    # Extract what's being asked
                    match = re.search(pattern + r"\s+(\w+(?:\s+\w+){0,3})", turn_lower)
                    if match:
                        asked_topic = match.group(1) if match.lastindex else ""
                        # Check if this was discussed in earlier turns
                        earlier = " ".join(t.content.lower() for t in agent_turns[:i])
                        topic_words = [w for w in asked_topic.split() if len(w) > 3]
                        if topic_words:
                            found = sum(1 for w in topic_words if w in earlier)
                            if found >= len(topic_words) * 0.6:
                                issues.append({
                                    "type": "re_asks_discussed_topic",
                                    "turn": turn.turn_number,
                                    "description": f"Re-asks about already discussed: {asked_topic[:30]}",
                                })
                                break

        # Prioritize and return issues
        # Strong signals: explicit_neglect_recovery, re_asks_discussed_topic
        # Weak signals: low_context_coherence
        strong_issues = [i for i in issues if i["type"] in ("explicit_neglect_recovery", "re_asks_discussed_topic")]
        weak_issues = [i for i in issues if i["type"] == "low_context_coherence"]

        if strong_issues:
            return strong_issues[:2]
        elif len(weak_issues) >= 2:
            # Multiple low coherence turns = likely context neglect
            return weak_issues[:2]

        return []

    def _extract_task_keywords(self, text: str) -> set:
        """Extract key task-related keywords from early conversation.

        Focuses on technical terms, product names, and specific requirements
        that should be maintained throughout the conversation.
        """
        keywords = set()
        text_lower = text.lower()

        # Extract programming-related terms
        prog_terms = re.findall(r'\b(python|java|javascript|react|django|flask|api|database|sql|cli|gui|web|app|file|data|user|input|output|function|class|method)\b', text_lower)
        keywords.update(prog_terms)

        # Extract quoted terms (often specific requirements)
        quoted = re.findall(r'"([^"]+)"', text)
        for q in quoted:
            keywords.update(w.lower() for w in q.split() if len(w) > 3)

        # Extract CamelCase terms (class/component names)
        camel = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', text)
        keywords.update(c.lower() for c in camel)

        # Extract snake_case terms
        snake = re.findall(r'\b([a-z]+_[a-z_]+)\b', text)
        keywords.update(snake)

        # Extract key nouns that appear multiple times
        words = re.findall(r'\b[a-z]{4,}\b', text_lower)
        word_counts = {}
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1
        frequent = [w for w, c in word_counts.items() if c >= 2]
        keywords.update(frequent[:10])

        # Remove common stop words
        stop_words = {'that', 'this', 'with', 'from', 'have', 'will', 'been', 'were', 'they', 'their', 'would', 'could', 'should', 'about', 'into', 'more', 'some', 'such', 'than', 'then', 'them', 'when', 'where', 'which', 'while', 'your'}
        keywords -= stop_words

        return keywords

    def _extract_requirements(self, text: str) -> List[str]:
        """Extract requirements/constraints from task description."""
        requirements = []
        text_lower = text.lower()

        # Patterns that indicate requirements
        req_patterns = [
            r"must\s+(\w+(?:\s+\w+){0,5})",
            r"should\s+(\w+(?:\s+\w+){0,5})",
            r"need(?:s)?\s+to\s+(\w+(?:\s+\w+){0,5})",
            r"require(?:s|d)?\s+(\w+(?:\s+\w+){0,5})",
            r"implement\s+(\w+(?:\s+\w+){0,5})",
            r"ensure\s+(\w+(?:\s+\w+){0,5})",
            r"include\s+(\w+(?:\s+\w+){0,5})",
        ]

        for pattern in req_patterns:
            matches = re.findall(pattern, text_lower)
            requirements.extend(matches)

        return list(set(requirements))[:10]  # Limit to top 10

    def _extract_key_entities(self, text: str) -> set:
        """Extract key named entities from text."""
        # Simple entity extraction - focus on technical terms
        words = re.findall(r'\b[A-Z][a-zA-Z]+(?:[A-Z][a-zA-Z]+)*\b', text)  # CamelCase
        words += re.findall(r'\b[a-z]+_[a-z_]+\b', text)  # snake_case
        return set(words)

    def _check_for_contradiction(self, prior_context: str, current_turn: str) -> Optional[str]:
        """Check if current turn contradicts prior context."""
        prior_lower = prior_context.lower()
        current_lower = current_turn.lower()

        # Contradiction patterns
        if "cli" in prior_lower and "gui" in current_lower:
            if "instead of cli" in current_lower or "not cli" in current_lower:
                return "Changed from CLI to GUI requirement"

        if "web" in prior_lower and "desktop" in current_lower:
            if "instead" in current_lower or "change to" in current_lower:
                return "Changed platform requirement"

        # Check for explicit contradictions
        contradiction_markers = ["actually", "instead", "rather than", "change to", "not what"]
        for marker in contradiction_markers:
            if marker in current_lower:
                # Check if it's contradicting something from prior context
                idx = current_lower.find(marker)
                context_around = current_lower[max(0, idx-50):idx+50]
                if any(word in context_around for word in ["requirement", "task", "original", "specification"]):
                    return f"Explicit change detected near '{marker}'"

        return None

    def _is_decision_turn(self, content: str) -> bool:
        """Check if this turn makes a decision that should consider requirements."""
        decision_indicators = [
            "i will", "i'll", "let me", "let's", "we will", "we'll",
            "implementing", "creating", "building", "developing",
            "the approach", "my plan", "the solution", "the design",
            "decided to", "choosing", "selected",
        ]
        content_lower = content.lower()
        return any(ind in content_lower for ind in decision_indicators)

    def _check_missing_requirements(
        self,
        requirements: List[str],
        current_content: str,
        prior_turns: List[TurnSnapshot],
    ) -> List[str]:
        """Check which requirements are not addressed in the conversation so far."""
        if not requirements:
            return []

        # Build context from all prior turns
        all_content = current_content.lower()
        for turn in prior_turns:
            all_content += " " + turn.content.lower()

        missing = []
        for req in requirements:
            req_words = set(req.lower().split())
            # Check if requirement words appear in the conversation
            words_found = sum(1 for w in req_words if w in all_content)
            if words_found < len(req_words) * 0.5:  # Less than half the words found
                missing.append(req)

        return missing[:3]  # Return top 3 missing

    def _repeats_prior_work(
        self,
        current_content: str,
        prior_turns: List[TurnSnapshot],
    ) -> bool:
        """Check if current turn is repeating work from prior turns."""
        if len(prior_turns) < 2:
            return False

        current_lower = current_content.lower()

        # Check for phrases indicating repetition
        repeat_phrases = [
            "let me implement", "let me create", "let me write",
            "i will implement", "i will create", "i will write",
        ]

        for phrase in repeat_phrases:
            if phrase in current_lower:
                # Check if something similar was already done
                for prior in prior_turns[-5:]:  # Check last 5 turns
                    prior_lower = prior.content.lower()
                    if phrase.replace("let me ", "i ") in prior_lower or phrase in prior_lower:
                        return True

        return False
