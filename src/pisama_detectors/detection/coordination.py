from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# v1.8: Common English words that are frequently capitalized but are NOT names.
# Used to filter name-substitution detection in _detect_data_corruption_relay.
# Without this, casual dialogue ("Hey, remember...") and tool-call syntax
# ("[Function call: ...]") trigger false positives on MemGPT/function-executor
# traces. All entries must be multi-letter to match the `[A-Z][a-z]+` regex.
_COMMON_ENGLISH_CAPS = frozenset(
    {
        "Hey",
        "Hi",
        "Hello",
        "Yes",
        "No",
        "Okay",
        "Ok",
        "Sure",
        "Thanks",
        "Thank",
        "Please",
        "Well",
        "Yeah",
        "Yep",
        "Nope",
        "Now",
        "Let",
        "Looks",
        "Seems",
        "My",
        "Your",
        "Our",
        "Their",
        "His",
        "Her",
        "Its",
        "You",
        "We",
        "They",
        "She",
        "He",
        "It",
        "This",
        "That",
        "These",
        "Those",
        "What",
        "Where",
        "When",
        "Why",
        "How",
        "Who",
        "Which",
        "The",
        "More",
        "Most",
        "Less",
        "Least",
        "Some",
        "Any",
        "All",
        "Every",
        "Would",
        "Could",
        "Should",
        "Must",
        "May",
        "Might",
        "Can",
        "Will",
        "Shall",
        "Do",
        "Did",
        "Does",
        "Have",
        "Has",
        "Had",
        "Am",
        "Is",
        "Are",
        "Was",
        "Were",
        "Here",
        "There",
        "Then",
        "Also",
        "Still",
        "Just",
        "Only",
        "Function",
        "Error",
        "Warning",
        "Note",
        "Info",
        "Debug",
        "Status",
        "Success",
        "True",
        "False",
        "None",
        "Null",
        "Call",
        "Message",
        "Response",
        "Request",
        "Reply",
    }
)


@dataclass
class Message:
    from_agent: str
    to_agent: str
    content: str
    timestamp: float
    acknowledged: bool = False


@dataclass
class CoordinationIssue:
    issue_type: str
    agents_involved: List[str]
    message: str
    severity: str


@dataclass
class CoordinationAnalysisResult:
    healthy: bool
    issues: List[CoordinationIssue]
    metrics: Dict[str, float]
    detected: bool = False
    confidence: float = 0.0
    issue_count: int = 0
    raw_score: Optional[float] = None
    calibration_info: Optional[Dict[str, Any]] = None


class CoordinationAnalyzer:
    def __init__(self, confidence_scaling: float = 1.0):
        self.message_timeout_seconds = 30.0
        self.max_back_forth_count = (
            8  # v1.5: raised from 5 (pipeline agents naturally exchange more)
        )
        self.confidence_scaling = confidence_scaling

    def analyze_coordination(
        self,
        messages: List[Message],
        agent_ids: List[str],
    ) -> CoordinationAnalysisResult:
        issues = []

        issues.extend(self._detect_ignored_messages(messages))
        issues.extend(self._detect_information_withholding(messages, agent_ids))
        issues.extend(self._detect_excessive_back_forth(messages))
        issues.extend(self._detect_circular_delegation(messages))
        # v1.4: New detection methods
        issues.extend(self._detect_conflicting_instructions(messages))
        issues.extend(self._detect_duplicate_dispatch(messages))
        issues.extend(self._detect_data_corruption_relay(messages))
        issues.extend(self._detect_ordering_violations(messages))
        issues.extend(self._detect_excessive_delegation(messages))
        issues.extend(self._detect_resource_contention(messages))
        issues.extend(self._detect_rapid_instruction_change(messages))
        issues.extend(self._detect_response_delay(messages))
        issues.extend(self._detect_indirect_delegation(messages))
        # v1.7: Ported from agent_teams + escalation_loop
        issues.extend(self._detect_lead_hoarding(messages, agent_ids))
        issues.extend(self._detect_silent_agent(messages, agent_ids))
        issues.extend(self._detect_stale_handoff_loop(messages))

        metrics = self._compute_metrics(messages, agent_ids)

        return CoordinationAnalysisResult(
            healthy=len([i for i in issues if i.severity in ["high", "critical"]]) == 0,
            issues=issues,
            metrics=metrics,
        )

    def _is_pipeline_topology(self, messages: List[Message], agent_ids: List[str]) -> bool:
        """v1.5: Detect if agents communicate in a linear chain (A→B→C→...).
        Pipeline topologies should not be penalized for limited communication breadth
        or indirect delegation patterns."""
        if len(agent_ids) < 3:
            return False
        # Build adjacency: who sends to whom
        senders_per_agent: Dict[str, Set[str]] = defaultdict(set)
        recipients_per_agent: Dict[str, Set[str]] = defaultdict(set)
        for msg in messages:
            senders_per_agent[msg.to_agent].add(msg.from_agent)
            recipients_per_agent[msg.from_agent].add(msg.to_agent)
        # Pipeline: most agents have ≤2 unique communication partners (1 sender + 1 recipient)
        pipeline_agents = 0
        for agent in agent_ids:
            partners = senders_per_agent.get(agent, set()) | recipients_per_agent.get(agent, set())
            if len(partners) <= 2:
                pipeline_agents += 1
        return pipeline_agents >= len(agent_ids) * 0.6

    def analyze_coordination_with_confidence(
        self,
        messages: List[Message],
        agent_ids: List[str],
    ) -> CoordinationAnalysisResult:
        issues = []
        is_pipeline = self._is_pipeline_topology(messages, agent_ids)

        issues.extend(self._detect_ignored_messages(messages))
        # v1.5: Suppress withholding and indirect delegation in pipeline topologies
        if not is_pipeline:
            issues.extend(self._detect_information_withholding(messages, agent_ids))
        issues.extend(self._detect_excessive_back_forth(messages))
        issues.extend(self._detect_circular_delegation(messages))
        # v1.4: New detection methods
        issues.extend(self._detect_conflicting_instructions(messages))
        issues.extend(self._detect_duplicate_dispatch(messages))
        issues.extend(self._detect_data_corruption_relay(messages))
        issues.extend(self._detect_ordering_violations(messages))
        issues.extend(self._detect_excessive_delegation(messages))
        issues.extend(self._detect_resource_contention(messages))
        issues.extend(self._detect_rapid_instruction_change(messages))
        issues.extend(self._detect_response_delay(messages))
        if not is_pipeline:
            issues.extend(self._detect_indirect_delegation(messages))
        # v1.6: Content-based stalled progress detection
        issues.extend(self._detect_stalled_progress(messages))
        # v1.6: Duplicate response detection (MAST pattern)
        issues.extend(self._detect_duplicate_responses(messages))
        # v1.6: Discussion-without-progress (ChatDev/MetaGPT patterns)
        issues.extend(self._detect_discussion_without_progress(messages))
        # v1.9: Repeated-content cycle (Magentic/AutoGen agent re-introduction patterns)
        issues.extend(self._detect_repeated_content_cycle(messages))
        # v1.7: Ported from agent_teams + escalation_loop on consolidation pass
        issues.extend(self._detect_lead_hoarding(messages, agent_ids))
        issues.extend(self._detect_silent_agent(messages, agent_ids))
        issues.extend(self._detect_stale_handoff_loop(messages))

        metrics = self._compute_metrics(messages, agent_ids)

        max_severity = "low"
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for issue in issues:
            sev = issue.severity
            if sev in severity_counts:
                severity_counts[sev] += 1
            if self._severity_rank(sev) > self._severity_rank(max_severity):
                max_severity = sev

        raw_score = self._calculate_raw_score(issues, severity_counts, metrics)
        confidence, calibration_info = self._calibrate_confidence(
            issues=issues,
            severity_counts=severity_counts,
            max_severity=max_severity,
            metrics=metrics,
            raw_score=raw_score,
        )

        healthy = len([i for i in issues if i.severity in ["high", "critical"]]) == 0

        return CoordinationAnalysisResult(
            healthy=healthy,
            issues=issues,
            metrics=metrics,
            detected=len(issues) > 0,
            confidence=confidence,
            issue_count=len(issues),
            raw_score=raw_score,
            calibration_info=calibration_info,
        )

    def _severity_rank(self, severity: str) -> int:
        ranks = {"low": 0, "medium": 1, "high": 2, "critical": 3}
        return ranks.get(severity, 0)

    def _calculate_raw_score(
        self,
        issues: List[CoordinationIssue],
        severity_counts: Dict[str, int],
        metrics: Dict[str, float],
    ) -> float:
        if not issues:
            return 0.0

        issue_score = (
            severity_counts.get("low", 0) * 0.1
            + severity_counts.get("medium", 0) * 0.25
            + severity_counts.get("high", 0) * 0.4
            + severity_counts.get("critical", 0) * 0.6
        )

        ack_rate = metrics.get("acknowledgment_rate", 1.0)
        health_penalty = (1.0 - ack_rate) * 0.2

        return min(1.0, issue_score + health_penalty)

    def _calibrate_confidence(
        self,
        issues: List[CoordinationIssue],
        severity_counts: Dict[str, int],
        max_severity: str,
        metrics: Dict[str, float],
        raw_score: float,
    ) -> Tuple[float, Dict[str, Any]]:
        if not issues:
            return 0.0, {
                "issue_count": 0,
                "severity_counts": severity_counts,
                "max_severity": "none",
                "raw_score": 0.0,
                "confidence_scaling": self.confidence_scaling,
            }

        severity_weight = {
            "low": 0.4,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95,
        }.get(max_severity, 0.5)

        issue_types = set(i.issue_type for i in issues)
        diversity_factor = min(1.0, len(issue_types) / 4)

        issue_factor = min(0.25, len(issues) * 0.05)

        ack_rate = metrics.get("acknowledgment_rate", 1.0)
        health_factor = (1.0 - ack_rate) * 0.15

        # v1.1: Boosted weights — many true positives were getting low confidence
        # causing them to fall below the optimized threshold.
        base_confidence = (
            severity_weight * 0.40
            + raw_score * 0.35
            + diversity_factor * 0.10
            + issue_factor
            + health_factor
        )
        # v1.6: Tightened floors — medium at 0.30 caused 87% FP on medium-difficulty
        # cases. Now require 2+ distinct issue types for medium to pass the floor.
        severity_floor = {"low": 0.15, "medium": 0.20, "high": 0.45, "critical": 0.65}
        if max_severity == "medium" and len(issue_types) < 2:
            # Single medium-severity signal is weak — don't inflate confidence
            pass
        else:
            base_confidence = max(base_confidence, severity_floor.get(max_severity, 0.20))

        calibrated = min(0.99, base_confidence * self.confidence_scaling)

        calibration_info = {
            "issue_count": len(issues),
            "severity_counts": severity_counts,
            "max_severity": max_severity,
            "severity_weight": severity_weight,
            "diversity_factor": round(diversity_factor, 4),
            "issue_types": list(issue_types),
            "acknowledgment_rate": ack_rate,
            "raw_score": round(raw_score, 4),
            "confidence_scaling": self.confidence_scaling,
        }

        return round(calibrated, 4), calibration_info

    def _detect_ignored_messages(self, messages: List[Message]) -> List[CoordinationIssue]:
        issues = []

        # v1.6: Identify "broadcast destinations" — agents that are receive-only
        # like 'system', 'logger', 'audit'. Messages to these don't need replies.
        broadcast_targets = {"system", "logger", "audit", "log", "broadcast", "monitor", "tracer"}

        # v1.6: Check if any message in the trace has explicit acknowledged=True
        # If yes, the field is reliable. If not, the field is missing/unreliable.
        has_ack_data = any(m.acknowledged for m in messages)

        def _ts_key(m):
            try:
                return float(m.timestamp)
            except (TypeError, ValueError):
                return 0.0

        # Track flagged pairs to avoid duplicate issues for same sender→recipient
        flagged_pairs = set()

        for i, msg in enumerate(messages):
            # Skip messages to broadcast destinations
            if msg.to_agent.lower() in broadcast_targets:
                continue
            # Skip self-messages
            if msg.from_agent == msg.to_agent:
                continue
            # Skip if already flagged this pair
            pair = (msg.from_agent, msg.to_agent)
            if pair in flagged_pairs:
                continue

            # Check for ANY activity from the recipient after this message
            msg_ts = _ts_key(msg)
            recipient_activity = [
                m for m in messages if _ts_key(m) > msg_ts and m.from_agent == msg.to_agent
            ]

            # Conditions for flagging as ignored:
            # A) acknowledged=False explicitly set (test case path)
            # B) No recipient activity AND this isn't the last message in a long trace
            should_flag = False
            severity = "medium"

            if not msg.acknowledged and not recipient_activity:
                # Recipient never replies AFTER this message.
                recipient_speaks_anywhere = any(m.from_agent == msg.to_agent for m in messages)
                if not recipient_speaks_anywhere:
                    # Recipient never speaks at all — clear ignored signal
                    should_flag = True
                    severity = "medium" if has_ack_data else "low"
                elif has_ack_data:
                    # Recipient speaks but explicitly didn't ack THIS message
                    should_flag = True
                    severity = "medium"
                # If !has_ack_data and recipient speaks elsewhere, don't flag —
                # the absence of ack is unreliable signal in real traces.

            if should_flag:
                flagged_pairs.add(pair)
                issues.append(
                    CoordinationIssue(
                        issue_type="ignored_message",
                        agents_involved=[msg.from_agent, msg.to_agent],
                        message=f"Message from {msg.from_agent} to {msg.to_agent} was not acknowledged",
                        severity=severity,
                    )
                )

        # v1.1: Detect content-based ignored messages — receiver asks for the
        # same information that was already sent to them (message was lost/ignored
        # despite being "acknowledged").
        for i, msg in enumerate(messages):
            later_from_recipient = [
                m for m in messages if m.timestamp > msg.timestamp and m.from_agent == msg.to_agent
            ]
            for reply in later_from_recipient:
                reply_lower = reply.content.lower()
                msg_lower = msg.content.lower()
                # If reply asks for what was already provided, it's a lost message
                request_phrases = [
                    "please provide",
                    "requesting",
                    "still waiting",
                    "no .* received",
                    "where is",
                    "send me",
                    "need the",
                    "waiting for",
                ]
                import re as _re

                is_repeat_request = any(_re.search(p, reply_lower) for p in request_phrases)
                # Check content overlap — is the reply asking about the same topic?
                msg_words = set(w for w in msg_lower.split() if len(w) > 4)
                reply_words = set(w for w in reply_lower.split() if len(w) > 4)
                if msg_words and reply_words:
                    overlap = len(msg_words & reply_words) / min(len(msg_words), len(reply_words))
                else:
                    overlap = 0
                if is_repeat_request and overlap > 0.2:
                    issues.append(
                        CoordinationIssue(
                            issue_type="message_lost",
                            agents_involved=[msg.from_agent, msg.to_agent],
                            message=f"Message from {msg.from_agent} appears lost — {msg.to_agent} re-requests same info",
                            severity="high",
                        )
                    )
                    break  # One detection per message pair

        return issues

    def _detect_information_withholding(
        self,
        messages: List[Message],
        agent_ids: List[str],
    ) -> List[CoordinationIssue]:
        issues = []

        communication_matrix: Dict[str, Set[str]] = defaultdict(set)
        # v1.3: Also track which agents have sent at least one message
        # and which agents have been addressed by others.
        agents_that_sent: Set[str] = set()
        agents_addressed: Dict[str, Set[str]] = defaultdict(set)
        for msg in messages:
            communication_matrix[msg.from_agent].add(msg.to_agent)
            agents_that_sent.add(msg.from_agent)
            agents_addressed[msg.to_agent].add(msg.from_agent)

        for agent in agent_ids:
            # v1.3: Only flag agents that have sent at least one message.
            # Agents that only receive (terminal nodes in pipelines, fan-out
            # receivers, pub/sub subscribers) should not be penalized for
            # limited communication breadth.
            if agent not in agents_that_sent:
                continue

            recipients = communication_matrix.get(agent, set())
            potential_recipients = set(agent_ids) - {agent}

            # v1.3: Also skip if the agent has been addressed by at most 1
            # other agent (pipeline topology: each agent talks to next in chain).
            addressed_by = agents_addressed.get(agent, set())
            if len(addressed_by) <= 1 and len(recipients) >= 1:
                continue

            if len(recipients) < len(potential_recipients) * 0.5 and len(potential_recipients) > 1:
                missing = potential_recipients - recipients
                issues.append(
                    CoordinationIssue(
                        issue_type="limited_communication",
                        agents_involved=[agent] + list(missing),
                        message=f"Agent {agent} has not communicated with: {missing}",
                        severity="low",
                    )
                )

        return issues

    def _detect_excessive_back_forth(self, messages: List[Message]) -> List[CoordinationIssue]:
        issues = []

        pair_exchanges: Dict[tuple, int] = defaultdict(int)
        for msg in messages:
            pair = tuple(sorted([msg.from_agent, msg.to_agent]))
            pair_exchanges[pair] += 1

        for pair, count in pair_exchanges.items():
            # v1.8: Skip agent↔tool-executor pairs. Tool-calling naturally
            # produces many round-trips (MemGPT patterns hit 10+ exchanges per
            # turn) and these are not coordination failures.
            if self._is_tool_agent(pair[0]) or self._is_tool_agent(pair[1]):
                continue
            if count > self.max_back_forth_count:
                issues.append(
                    CoordinationIssue(
                        issue_type="excessive_back_forth",
                        agents_involved=list(pair),
                        message=f"Agents {pair[0]} and {pair[1]} have exchanged {count} messages (threshold: {self.max_back_forth_count})",
                        severity="medium",
                    )
                )

        return issues

    @staticmethod
    def _is_tool_agent(agent_id: str) -> bool:
        """v1.8: Return True if this agent name looks like a tool/function
        executor rather than a coordinating agent. Tool round-trips shouldn't
        be penalized the same way as agent-to-agent chatter."""
        lower = agent_id.lower()
        tool_markers = (
            "function_executor",
            "tool_executor",
            "function_caller",
            "tool_caller",
            "_executor",
            "_tool",
            "tools",
            "toolkit",
            "function_runner",
            "tool_runner",
        )
        return any(marker in lower for marker in tool_markers)

    def _detect_circular_delegation(self, messages: List[Message]) -> List[CoordinationIssue]:
        issues = []

        import re as _re

        delegation_graph: Dict[str, List[str]] = defaultdict(list)
        for msg in messages:
            content_lower = msg.content.lower()
            # v1.1: Use both exact substring and regex for delegation detection.
            # Regex handles cases like "pass this to" or "hand it off to".
            delegation_phrases = [
                "delegate",
                "hand off",
                "handoff",
                "take over",
                "your turn",
                "assign to",
                "forward to",
                "escalate to",
                "handle this",
                "proceed with",
                "can you handle",
                "please take",
                "transfer to",
                "route to",
            ]
            delegation_regexes = [
                r"pass\s+(?:\w+\s+)?to\b",  # "pass to", "pass this to", "pass it to"
                r"hand\s+\w+\s+off\b",  # "hand it off"
                r"delegat\w*\s+(?:this|it|to)\b",  # "delegating to", "delegated this"
            ]
            has_delegation = any(phrase in content_lower for phrase in delegation_phrases) or any(
                _re.search(pat, content_lower) for pat in delegation_regexes
            )
            if has_delegation:
                delegation_graph[msg.from_agent].append(msg.to_agent)

        for start_agent in delegation_graph:
            visited = set()
            stack = [start_agent]

            while stack:
                current = stack.pop()
                if current in visited:
                    issues.append(
                        CoordinationIssue(
                            issue_type="circular_delegation",
                            agents_involved=list(visited),
                            message=f"Circular delegation detected involving {visited}",
                            severity="high",
                        )
                    )
                    break
                visited.add(current)
                stack.extend(delegation_graph.get(current, []))

        return issues

    # ── v1.4: New detection methods ────────────────────────────────────────

    def _detect_conflicting_instructions(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.4: Detect when multiple agents send contradictory instructions to the same agent."""
        issues = []
        # Group messages by recipient within short time windows
        by_recipient: Dict[str, List[Message]] = defaultdict(list)
        for msg in messages:
            by_recipient[msg.to_agent].append(msg)

        conflict_pairs = [
            (
                {"update", "set", "change", "enable", "activate", "start", "add"},
                {"delete", "remove", "disable", "deactivate", "stop", "drop"},
            ),
            (
                {"create", "insert", "save", "write", "open"},
                {"delete", "remove", "destroy", "close", "drop"},
            ),
            ({"lock"}, {"unlock"}),
            ({"approve", "accept"}, {"reject", "deny", "decline"}),
        ]

        for recipient, msgs in by_recipient.items():
            for i in range(len(msgs)):
                for j in range(i + 1, len(msgs)):
                    m1, m2 = msgs[i], msgs[j]
                    if m1.from_agent == m2.from_agent:
                        continue  # Same sender — might be an update, not conflict
                    try:
                        if abs(float(m1.timestamp) - float(m2.timestamp)) > 5.0:
                            continue  # Too far apart in time
                    except (TypeError, ValueError):
                        pass  # Non-numeric timestamps
                    w1 = set(m1.content.lower().split())
                    w2 = set(m2.content.lower().split())
                    for pos_set, neg_set in conflict_pairs:
                        if (w1 & pos_set and w2 & neg_set) or (w1 & neg_set and w2 & pos_set):
                            issues.append(
                                CoordinationIssue(
                                    issue_type="conflicting_instructions",
                                    agents_involved=[m1.from_agent, m2.from_agent, recipient],
                                    message=f"Conflicting instructions to {recipient}: '{m1.content[:50]}' vs '{m2.content[:50]}'",
                                    severity="high",
                                )
                            )
                            break
        return issues

    def _detect_duplicate_dispatch(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.4: Detect when the same task is dispatched to multiple agents."""
        issues = []
        # Group by sender
        by_sender: Dict[str, List[Message]] = defaultdict(list)
        for msg in messages:
            by_sender[msg.from_agent].append(msg)

        for sender, msgs in by_sender.items():
            for i in range(len(msgs)):
                for j in range(i + 1, len(msgs)):
                    m1, m2 = msgs[i], msgs[j]
                    if m1.to_agent == m2.to_agent:
                        continue  # Same recipient — not a duplicate dispatch
                    # Compare timestamps if numeric, skip if string/incompatible
                    try:
                        if abs(float(m1.timestamp) - float(m2.timestamp)) > 2.0:
                            continue
                    except (TypeError, ValueError):
                        pass  # Non-numeric timestamps — fall through to content check
                    # Check content similarity
                    w1 = set(m1.content.lower().split())
                    w2 = set(m2.content.lower().split())
                    if not w1 or not w2:
                        continue
                    overlap = len(w1 & w2) / min(len(w1), len(w2))
                    if overlap >= 0.7:
                        issues.append(
                            CoordinationIssue(
                                issue_type="duplicate_dispatch",
                                agents_involved=[sender, m1.to_agent, m2.to_agent],
                                message=f"Same task dispatched to {m1.to_agent} and {m2.to_agent}: '{m1.content[:50]}'",
                                severity="medium",
                            )
                        )
        return issues

    def _detect_data_corruption_relay(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.4: Detect when data values change during relay between agents."""
        issues = []
        import re as _re

        # Find relay chains: A→B then B→C with overlapping topic (C must differ from A)
        for i, m1 in enumerate(messages):
            for m2 in messages:
                if m2.timestamp <= m1.timestamp:
                    continue
                if m2.from_agent != m1.to_agent:
                    continue  # Not a relay
                if m2.to_agent == m1.from_agent:
                    continue  # Reply, not relay to third party
                # Check topic overlap
                w1 = set(w for w in m1.content.lower().split() if len(w) > 3)
                w2 = set(w for w in m2.content.lower().split() if len(w) > 3)
                if not w1 or not w2:
                    continue
                topic_overlap = len(w1 & w2) / min(len(w1), len(w2))
                if topic_overlap < 0.2:
                    continue  # Different topic — not a relay

                # Extract key-value pairs or named entities
                kv1 = dict(_re.findall(r"(\w+):\s*(\w+)", m1.content))
                kv2 = dict(_re.findall(r"(\w+):\s*(\w+)", m2.content))
                if kv1 and kv2:
                    shared_keys = set(kv1.keys()) & set(kv2.keys())
                    for key in shared_keys:
                        if kv1[key] != kv2[key]:
                            issues.append(
                                CoordinationIssue(
                                    issue_type="data_corruption_relay",
                                    agents_involved=[m1.from_agent, m1.to_agent, m2.to_agent],
                                    message=f"Data corrupted in relay: {key} changed from '{kv1[key]}' to '{kv2[key]}'",
                                    severity="high",
                                )
                            )
                            return issues  # One detection per chain

                # Also check for proper noun substitution (names)
                # Only match names NOT at the start of a sentence to avoid
                # false positives from sentence-initial capitalization
                def _extract_names(text: str) -> Set[str]:
                    # Find capitalized words that aren't the first word
                    # of the message, following sentence-ending punctuation,
                    # or common English words that are frequently capitalized
                    # (Hey, Function, Yes, My, etc. — see _COMMON_ENGLISH_CAPS).
                    names = set()
                    for m in _re.finditer(r"\b[A-Z][a-z]+\b", text):
                        pos = m.start()
                        if pos == 0:
                            continue  # First word of message
                        before = text[max(0, pos - 3) : pos].strip()
                        if before and before[-1] in ".!?":
                            continue  # First word of sentence
                        word = m.group()
                        if word in _COMMON_ENGLISH_CAPS:
                            continue  # v1.8: Common English cap, not a name
                        names.add(word)
                    return names

                names1 = _extract_names(m1.content)
                names2 = _extract_names(m2.content)
                # v1.8: Require at least one shared name before flagging
                # substitution — without an anchor, the two messages are
                # not demonstrably relaying the same entity's context.
                shared_names = names1 & names2
                if names1 and names2 and shared_names and topic_overlap > 0.3:
                    changed_names = names1 - names2
                    new_names = names2 - names1
                    if changed_names and new_names:
                        issues.append(
                            CoordinationIssue(
                                issue_type="data_corruption_relay",
                                agents_involved=[m1.from_agent, m1.to_agent, m2.to_agent],
                                message=f"Name changed in relay: {changed_names} became {new_names}",
                                severity="high",
                            )
                        )
                        return issues
        return issues

    def _detect_ordering_violations(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.4: Detect when sequential steps complete out of order."""
        issues = []
        import re as _re

        # Find orchestrator dispatching sequential steps
        step_dispatches: Dict[str, List[Tuple[Message, int]]] = defaultdict(list)
        for msg in messages:
            # Look for step/phase numbering in content
            match = _re.search(r"\b(?:step|phase|task)\s*(\d+)", msg.content, _re.IGNORECASE)
            if match:
                step_num = int(match.group(1))
                step_dispatches[msg.from_agent].append((msg, step_num))

        for sender, dispatches in step_dispatches.items():
            dispatches.sort(key=lambda x: x[1])  # Sort by step number
            for i in range(len(dispatches) - 1):
                msg_early, step_early = dispatches[i]
                msg_late, step_late = dispatches[i + 1]
                # Check if the later step's recipient responded before the earlier step's
                early_response = next(
                    (
                        m
                        for m in messages
                        if m.from_agent == msg_early.to_agent and m.timestamp > msg_early.timestamp
                    ),
                    None,
                )
                late_response = next(
                    (
                        m
                        for m in messages
                        if m.from_agent == msg_late.to_agent and m.timestamp > msg_late.timestamp
                    ),
                    None,
                )
                if late_response and (
                    not early_response or late_response.timestamp < early_response.timestamp
                ):
                    issues.append(
                        CoordinationIssue(
                            issue_type="ordering_violation",
                            agents_involved=[sender, msg_early.to_agent, msg_late.to_agent],
                            message=f"Step {step_late} completed before step {step_early}",
                            severity="medium",
                        )
                    )
        return issues

    def _detect_excessive_delegation(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.4: Detect when a task is forwarded through a chain without being worked on."""
        issues = []
        # Build forwarding chains: A→B, B→C, C→D with similar content
        chains: List[List[Message]] = []
        for msg in messages:
            # Try to extend an existing chain
            extended = False
            for chain in chains:
                last = chain[-1]
                if msg.from_agent == last.to_agent and msg.timestamp > last.timestamp:
                    w_last = set(last.content.lower().split())
                    w_msg = set(msg.content.lower().split())
                    if w_last and w_msg:
                        overlap = len(w_last & w_msg) / min(len(w_last), len(w_msg))
                        if overlap >= 0.6:
                            chain.append(msg)
                            extended = True
                            break
            if not extended:
                chains.append([msg])

        for chain in chains:
            if len(chain) >= 3:  # 3+ forwards = excessive
                agents = [chain[0].from_agent] + [m.to_agent for m in chain]
                issues.append(
                    CoordinationIssue(
                        issue_type="excessive_delegation",
                        agents_involved=agents,
                        message=f"Task forwarded through {len(chain)} agents without work: {' → '.join(agents)}",
                        severity="medium",
                    )
                )
        return issues

    def _detect_resource_contention(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.4: Detect when multiple agents compete for the same resource."""
        issues = []
        import re as _re

        resource_verbs = {"lock", "acquire", "reserve", "claim", "allocate", "write"}
        # Find messages requesting resource access
        resource_requests: List[Tuple[Message, str]] = []
        for msg in messages:
            lower = msg.content.lower()
            has_verb = any(v in lower for v in resource_verbs)
            if has_verb:
                # Extract resource name (word after verb, or common patterns)
                for v in resource_verbs:
                    match = _re.search(v + r"\s+(?:to\s+)?(?:resource\s+)?(\w+)", lower)
                    if match:
                        resource_requests.append((msg, match.group(1)))
                        break

        # Check for contention: different agents requesting same resource
        for i in range(len(resource_requests)):
            for j in range(i + 1, len(resource_requests)):
                m1, r1 = resource_requests[i]
                m2, r2 = resource_requests[j]
                if r1 == r2 and m1.from_agent != m2.from_agent:
                    try:
                        if abs(float(m1.timestamp) - float(m2.timestamp)) >= 5.0:
                            continue
                    except (TypeError, ValueError):
                        pass  # Non-numeric — assume contention possible
                    if True:
                        issues.append(
                            CoordinationIssue(
                                issue_type="resource_contention",
                                agents_involved=[m1.from_agent, m2.from_agent],
                                message=f"Resource contention: {m1.from_agent} and {m2.from_agent} both requesting '{r1}'",
                                severity="high",
                            )
                        )
        return issues

    def _detect_rapid_instruction_change(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.4: Detect when an agent cancels/overrides a recent instruction."""
        issues = []
        cancel_words = {
            "cancel",
            "instead",
            "disregard",
            "ignore",
            "scratch",
            "stop",
            "abort",
            "nevermind",
            "never mind",
            "actually",
        }
        # Group by (from_agent, to_agent) pair
        by_pair: Dict[Tuple[str, str], List[Message]] = defaultdict(list)
        for msg in messages:
            by_pair[(msg.from_agent, msg.to_agent)].append(msg)

        for pair, msgs in by_pair.items():
            # v1.8: Skip user/human senders and tool-agent pairs. Users
            # saying "actually" or "instead" in follow-ups is conversational
            # clarification, not rapid instruction change. Tool round-trips
            # also produce cancel-word content (e.g. status: aborted) that
            # shouldn't trigger this check.
            sender_lower = pair[0].lower()
            if sender_lower in ("user", "human"):
                continue
            if self._is_tool_agent(pair[0]) or self._is_tool_agent(pair[1]):
                continue
            try:
                msgs_sorted = sorted(msgs, key=lambda m: float(m.timestamp))
            except (TypeError, ValueError):
                msgs_sorted = list(msgs)
            for i in range(len(msgs_sorted) - 1):
                m1, m2 = msgs_sorted[i], msgs_sorted[i + 1]
                try:
                    if float(m2.timestamp) - float(m1.timestamp) > 3.0:
                        continue  # Not rapid
                except (TypeError, ValueError):
                    pass  # Non-numeric — fall through to content check
                lower2 = m2.content.lower()
                if any(w in lower2 for w in cancel_words):
                    issues.append(
                        CoordinationIssue(
                            issue_type="rapid_instruction_change",
                            agents_involved=list(pair),
                            message=f"Rapid instruction change from {pair[0]} to {pair[1]}: '{m2.content[:50]}'",
                            severity="low",
                        )
                    )
        return issues

    def _detect_stalled_progress(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.6: Detect coordination failures via stalled progress markers in content.

        MAST traces show patterns like 'Progress: 0/7 steps completed (0%)' or
        'Status: 0 completed, 0 in progress, 0 blocked, 7 not started' which
        indicate the workflow is stalled — agents broadcasting status without
        actual progress.
        """
        import re as _re

        issues = []
        stall_patterns = [
            _re.compile(r"(\d+)\s*/\s*(\d+)\s+steps\s+completed", _re.IGNORECASE),
            _re.compile(
                r"(\d+)\s+completed,\s*(\d+)\s+in\s+progress,\s*(\d+)\s+blocked,\s*(\d+)\s+not\s+started",
                _re.IGNORECASE,
            ),
            _re.compile(r"progress[:\s]+0\.0%|0%\s+complete", _re.IGNORECASE),
        ]
        for msg in messages:
            content = msg.content
            for pat in stall_patterns:
                m = pat.search(content)
                if m:
                    groups = m.groups()
                    if not groups:
                        continue
                    try:
                        completed = int(groups[0])
                        # 0 completed = stalled
                        if completed == 0 and len(groups) >= 2:
                            total = int(groups[1])
                            if total >= 3:  # Real workflow with steps
                                issues.append(
                                    CoordinationIssue(
                                        issue_type="stalled_progress",
                                        agents_involved=[msg.from_agent, msg.to_agent],
                                        message=f"Workflow stalled: 0/{total} steps completed",
                                        severity="high",
                                    )
                                )
                                break
                    except (ValueError, IndexError):
                        pass
        return issues

    def _detect_duplicate_responses(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.6: Detect agents producing duplicate/repeated responses.

        MAST traces include explicit warnings like 'Observed duplicate responses.
        Consider new strategies and avoid repeating ineffective approaches.'
        """
        issues = []
        for msg in messages:
            content_lower = msg.content.lower()
            if "duplicate response" in content_lower or "repeating ineffective" in content_lower:
                issues.append(
                    CoordinationIssue(
                        issue_type="duplicate_responses",
                        agents_involved=[msg.from_agent, msg.to_agent],
                        message=f"Duplicate responses observed: {msg.content[:80]}",
                        severity="high",
                    )
                )
        return issues

    def _detect_discussion_without_progress(
        self, messages: List[Message]
    ) -> List[CoordinationIssue]:
        """v1.6: Detect long discussions where agents agree but don't make decisions.

        MAST framework patterns show CEO/CPO/CTO discussing without actually
        producing concrete output. Indicators:
        - Long conversations (10+ messages)
        - Multiple agents agreeing without dissent
        - High repetition of agreement phrases
        - No code/output blocks in messages
        """
        if len(messages) < 10:
            return []

        issues = []
        agreement_phrases = [
            "i agree",
            "agreed",
            "good idea",
            "great idea",
            "sounds good",
            "makes sense",
            "i fully agree",
            "fully agree",
            "i concur",
            "concur",
            "absolutely",
            "i think you",
            "good point",
            "excellent point",
        ]

        agreement_count = 0
        has_concrete_output = False
        for msg in messages:
            content_lower = msg.content.lower()
            if any(phrase in content_lower for phrase in agreement_phrases):
                agreement_count += 1
            # Look for concrete output: code blocks, file names, function definitions
            if "```" in msg.content or "def " in msg.content or "function " in msg.content:
                has_concrete_output = True
            # Also: look for file paths, URLs, specific deliverables
            if any(
                marker in content_lower
                for marker in [".py", ".js", ".json", ".yaml", ".md", "http://", "https://"]
            ):
                has_concrete_output = True

        # If 30%+ of messages are agreement and no concrete output → stalled discussion
        agreement_rate = agreement_count / len(messages)
        if agreement_rate >= 0.30 and not has_concrete_output:
            issues.append(
                CoordinationIssue(
                    issue_type="discussion_without_progress",
                    agents_involved=list(set(m.from_agent for m in messages))[:5],
                    message=f"Long discussion ({len(messages)} msgs) with high agreement rate ({agreement_rate:.0%}) but no concrete output",
                    severity="medium",
                )
            )

        return issues

    def _detect_repeated_content_cycle(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.9: Detect agents emitting verbatim-identical content repeatedly.

        In Magentic / AutoGen / MetaGPT traces, coordination stalls often
        manifest as agents re-broadcasting the same description or status line
        without advancing. `_detect_duplicate_responses` requires explicit
        "duplicate response" warnings in content; `_detect_stalled_progress`
        requires specific progress markers. Neither catches the pattern where
        an agent simply outputs the exact same string three or more times.

        Signal: trace has >=6 messages AND a non-tool agent emits the same
        >=30-char content prefix 3+ times. Skips tool-executor agents whose
        protocol responses (e.g. `status: OK`) are expected to repeat.
        """
        if len(messages) < 6:
            return []
        per_sender_content: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for m in messages:
            if self._is_tool_agent(m.from_agent):
                continue
            key = (m.content or "").strip()[:200]
            if len(key) <= 30:
                continue
            per_sender_content[m.from_agent][key] += 1
        for sender, counts in per_sender_content.items():
            if not counts:
                continue
            top_key, top_count = max(counts.items(), key=lambda kv: kv[1])
            if top_count >= 3:
                return [
                    CoordinationIssue(
                        issue_type="repeated_content_cycle",
                        agents_involved=[sender],
                        message=(
                            f"Agent '{sender}' emitted the same content {top_count} times "
                            f"— stalled re-broadcasting pattern"
                        ),
                        severity="medium",
                    )
                ]
        return []

    def _detect_response_delay(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.4: Detect unusually long delays between request and response."""
        issues = []

        # Find request-response pairs
        def _ts(m):
            try:
                return float(m.timestamp)
            except (TypeError, ValueError):
                return 0.0

        for msg in messages:
            responses = [
                m
                for m in messages
                if m.from_agent == msg.to_agent
                and m.to_agent == msg.from_agent
                and _ts(m) > _ts(msg)
            ]
            if responses:
                first_response = min(responses, key=_ts)
                delay = _ts(first_response) - _ts(msg)
                if delay > 10.0:  # 10 seconds threshold
                    issues.append(
                        CoordinationIssue(
                            issue_type="slow_response",
                            agents_involved=[msg.from_agent, msg.to_agent],
                            message=f"Slow response from {msg.to_agent}: {delay:.0f}s delay",
                            severity="low",
                        )
                    )
        return issues

    def _detect_indirect_delegation(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.4: Detect when an intermediary re-delegates and response bypasses them."""
        issues = []
        # Pattern: A→B, B→C, C→A (B is bypassed in the response)
        for m1 in messages:
            for m2 in messages:
                if m2.timestamp <= m1.timestamp:
                    continue
                if m2.from_agent != m1.to_agent:
                    continue  # m2 must be from m1's recipient
                if m2.to_agent == m1.from_agent:
                    continue  # Direct response, not re-delegation
                # m1: A→B, m2: B→C — check if C replies to A
                for m3 in messages:
                    if m3.timestamp <= m2.timestamp:
                        continue
                    if m3.from_agent == m2.to_agent and m3.to_agent == m1.from_agent:
                        # C→A: bypassing B — the triangle pattern itself is
                        # strong evidence. Use prefix matching for topic check
                        # (handles summarize/summary, deploy/deployment, etc.)
                        w1 = set(w[:5] for w in m1.content.lower().split() if len(w) > 3)
                        w3 = set(w[:5] for w in m3.content.lower().split() if len(w) > 3)
                        prefix_overlap = len(w1 & w3)
                        if prefix_overlap > 0 or len(messages) <= 4:
                            issues.append(
                                CoordinationIssue(
                                    issue_type="indirect_delegation",
                                    agents_involved=[m1.from_agent, m1.to_agent, m2.to_agent],
                                    message=f"{m1.to_agent} re-delegated to {m2.to_agent} who replied directly to {m1.from_agent}",
                                    severity="low",
                                )
                            )
                            return issues  # One per trace
        return issues

    def _detect_lead_hoarding(
        self, messages: List[Message], agent_ids: List[str]
    ) -> List[CoordinationIssue]:
        """v1.7: Detect when one agent dominates the message volume.

        Ported from agent_teams.py: in a team of 2+ agents, if one agent
        sends >60% of the messages, it suggests "lead hoarding" — the lead
        is doing most of the work and other teammates are underutilized.
        """
        if len(agent_ids) < 2:
            return []
        msg_counts: Dict[str, int] = defaultdict(int)
        for m in messages:
            sender = m.from_agent
            if sender:
                msg_counts[sender] += 1
        if not msg_counts:
            return []
        total = sum(msg_counts.values())
        if total < 5:
            return []  # Too few messages to draw a conclusion
        max_sender, max_count = max(msg_counts.items(), key=lambda kv: kv[1])
        share = max_count / total
        if share > 0.60:
            return [
                CoordinationIssue(
                    issue_type="lead_hoarding",
                    agents_involved=[max_sender],
                    message=(
                        f"Agent '{max_sender}' sent {max_count}/{total} messages "
                        f"({share:.0%}) — other teammates are underutilized"
                    ),
                    severity="medium",
                )
            ]
        return []

    def _detect_silent_agent(
        self, messages: List[Message], agent_ids: List[str]
    ) -> List[CoordinationIssue]:
        """v1.7: Detect agents who appear in agent_ids but produce no output.

        Ported from agent_teams.py: if an agent is part of the team but
        never emits a message, it's a "silent agent" — assigned but never
        engaged. Any non-empty message counts as engagement: short
        acknowledgments are valid coordination signals, not silence.
        """
        if len(agent_ids) < 2:
            return []
        # Build set of agents who emitted at least one non-empty message.
        active_senders: Set[str] = set()
        for m in messages:
            if m.from_agent and m.content and m.content.strip():
                active_senders.add(m.from_agent)
        # System/lead aren't expected to have output of their own
        team_agents = {a for a in agent_ids if a not in ("lead", "system")}
        silent = team_agents - active_senders
        if silent and len(team_agents) > 1:
            silent_ratio = len(silent) / len(team_agents)
            if silent_ratio > 0.30:
                return [
                    CoordinationIssue(
                        issue_type="silent_agent",
                        agents_involved=sorted(silent),
                        message=(
                            f"{len(silent)}/{len(team_agents)} team agents "
                            f"({silent_ratio:.0%}) produced no output"
                        ),
                        severity="medium",
                    )
                ]
        return []

    def _detect_stale_handoff_loop(self, messages: List[Message]) -> List[CoordinationIssue]:
        """v1.7: Detect circular handoffs where the handoff content stays similar.

        Ported from escalation_loop: when agents pass the same issue back and
        forth (A→B→A or A→B→C→A) AND the content of consecutive handoffs has
        high word overlap (>=70%), it's an unresolved escalation loop. The
        existing _detect_circular_delegation only checks for the cycle; this
        adds the staleness check.
        """
        if len(messages) < 3:
            return []
        # Find all (sender, recipient) message lists
        handoffs_by_pair: Dict[tuple, List[Message]] = defaultdict(list)
        for m in messages:
            handoffs_by_pair[(m.from_agent, m.to_agent)].append(m)
        # Find pairs with reverse partner that also has messages
        loop_signal: List[CoordinationIssue] = []
        seen_loops = set()
        for (sender, recipient), msgs in handoffs_by_pair.items():
            reverse_msgs = handoffs_by_pair.get((recipient, sender), [])
            if not reverse_msgs:
                continue
            loop_key = tuple(sorted([sender, recipient]))
            if loop_key in seen_loops:
                continue
            # Look at the first 3 messages in the loop and check if their
            # content has high pairwise word overlap (= no progress).
            ordered = sorted(msgs + reverse_msgs, key=lambda m: m.timestamp)[:3]
            if len(ordered) < 3:
                continue
            word_sets = [set(w.lower() for w in m.content.split() if len(w) > 3) for m in ordered]
            if not all(word_sets):
                continue
            overlaps = []
            for i in range(len(word_sets) - 1):
                a, b = word_sets[i], word_sets[i + 1]
                if not (a or b):
                    continue
                jaccard = len(a & b) / max(len(a | b), 1)
                overlaps.append(jaccard)
            if overlaps and sum(overlaps) / len(overlaps) >= 0.70:
                loop_signal.append(
                    CoordinationIssue(
                        issue_type="stale_handoff_loop",
                        agents_involved=[sender, recipient],
                        message=(
                            f"Circular handoffs between '{sender}' and "
                            f"'{recipient}' with stale content (no progress)"
                        ),
                        severity="high",
                    )
                )
                seen_loops.add(loop_key)
        return loop_signal

    # v2.0 Sprint 7 Phase FF-2 constants. Tuned against the coordination
    # golden dataset: stuck-pair mean-sim >= 0.80 gave P=0.88 on 15 flipped
    # positives; >= 0.70 dropped precision below the 0.88 floor. Held as a
    # standalone signal rather than a rule-path issue because integrating
    # into severity_counts inflated calibrated confidence across many
    # traces, hurting overall F1.
    _STUCK_PAIR_GATE = 0.80  # avg cosine similarity across pair messages
    _STUCK_PAIR_MIN_MSGS = 3  # require 3+ messages in the pair

    def semantic_stuck_pair_signal(self, messages: List[Message]) -> float:
        """v2.0 Sprint 7 Phase FF-2: semantic stuck-pair similarity signal.

        For each unordered agent pair with 3+ exchanges (tool executors
        excluded), compute the mean pairwise cosine similarity of the
        messages they exchanged. Returns the maximum such similarity
        across all qualifying pairs.

        Returns 0.0 when:
        - Fewer than 3 messages overall
        - No qualifying non-tool pair has 3+ exchanges
        - Embedder is unavailable or raises

        The caller is expected to treat values above ~0.80 as evidence of
        a semantically stuck exchange (retry loop, mutual-request
        deadlock, state desync) and lift confidence accordingly. Values
        below that are inconclusive — healthy task-focused pairs routinely
        produce 0.50-0.70.
        """
        if len(messages) < 3:
            return 0.0
        from collections import defaultdict as _dd

        pair_contents: Dict[tuple, List[str]] = _dd(list)
        for m in messages:
            f, t = str(m.from_agent or ""), str(m.to_agent or "")
            if not f or not t or f == t:
                continue
            if self._is_tool_agent(f) or self._is_tool_agent(t):
                continue
            content = (m.content or "").strip()
            if len(content) < 10:
                continue
            pair = tuple(sorted([f, t]))
            pair_contents[pair].append(content[:2000])

        if not pair_contents:
            return 0.0

        # Lazy embedder import so tests that don't exercise this path don't
        # pay the model-load cost.
        try:
            from pisama_detectors.detection.shared_embedder import get_shared_embedder

            embedder = get_shared_embedder()
            if not embedder:
                return 0.0
        except Exception:
            return 0.0

        best_sim = 0.0
        for pair, contents in pair_contents.items():
            if len(contents) < self._STUCK_PAIR_MIN_MSGS:
                continue
            try:
                vecs = [embedder.encode(c, is_query=False) for c in contents]
            except Exception:
                continue
            sims: List[float] = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sims.append(embedder.similarity(vecs[i], vecs[j]))
            if not sims:
                continue
            mean_sim = sum(sims) / len(sims)
            if mean_sim > best_sim:
                best_sim = mean_sim

        return float(best_sim)

    def _compute_metrics(
        self,
        messages: List[Message],
        agent_ids: List[str],
    ) -> Dict[str, float]:
        if not messages or not agent_ids:
            return {}

        total_messages = len(messages)
        acknowledged = sum(1 for m in messages if m.acknowledged)

        unique_pairs = len(set(tuple(sorted([m.from_agent, m.to_agent])) for m in messages))
        max_pairs = len(agent_ids) * (len(agent_ids) - 1) / 2

        return {
            "acknowledgment_rate": acknowledged / total_messages if total_messages > 0 else 0,
            "communication_density": unique_pairs / max_pairs if max_pairs > 0 else 0,
            "messages_per_agent": total_messages / len(agent_ids) if agent_ids else 0,
        }


coordination_analyzer = CoordinationAnalyzer()
