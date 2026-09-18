"""Behavior tests for the persona-drift detector code ported from the backend.

Covers the behavioral-fingerprint helpers, the declared-register (style)
compliance signal, the empty-persona exclusion, fingerprint-drift fusion in
``PersonaConsistencyScorer.score_consistency``, and the task-relevance /
off-role-excursion guards. The guards need a sentence embedder, so those tests
are skipped when the optional ``sentence-transformers`` extra is not installed;
everything else runs on the default ``.[dev]`` install.
"""

import pytest

from pisama_detectors.detection.persona import (
    Agent,
    PersonaConsistencyScorer,
    RoleType,
    _bounded_diff,
    _compute_fingerprint_drift_score,
    _extract_fingerprint,
    _js_divergence,
    _l1_rate_distance,
    _split_sentences,
    _tokenize_words,
)

FORMAL_HISTORY = [
    "Therefore the report is ready. Please review the summary.",
    "Therefore the invoice is ready. Please review the totals.",
    "Therefore the schedule is ready. Please review the agenda.",
]
SLANG_REPLY = "Yeah lol hey dude, gonna be cool stuff! Awesome, kinda wanna try it!"

LEGAL_PERSONA = "You are a legal expert who explains statutes and court rulings."
LEGAL_HISTORY = [
    "Therefore the statute applies. Furthermore, the court ruling confirms the regulation.",
    "Consequently the law requires compliance. Moreover, the legal case supports this section.",
    "Regarding the act, the court ruling is binding. Accordingly, the regulation stands.",
]
LEGAL_SAME_REGISTER = (
    "Thus the statute applies. Furthermore, the court ruling confirms the regulation."
)
# Domain-relevant (court, statute, law, ruling, legal) so the text-pattern signals
# read it as on-persona, but the register has slid into slang.
LEGAL_SLANG = (
    "Yeah lol hey dude, that court ruling on the statute is awesome! "
    "Cool stuff, gonna love the legal case, kinda wanna read the law!"
)
# A mild register slip (one slang word) and an emphatic-but-in-register reply. Against
# LEGAL_HISTORY their fingerprint drift lands either side of the 0.20 firing threshold.
LEGAL_MILD_SLIP = (
    "Thus the statute applies. Furthermore, the court ruling confirms the regulation, dude!"
)
LEGAL_EMPHATIC = "Thus the statute applies! Furthermore, the court ruling confirms the regulation!"


def _legal_agent() -> Agent:
    return Agent(id="legal-1", persona_description=LEGAL_PERSONA, allowed_actions=[])


def _embedder_or_skip() -> None:
    pytest.importorskip("sentence_transformers")
    from pisama_detectors.detection.shared_embedder import get_shared_embedder

    if get_shared_embedder() is None:
        pytest.skip("sentence embedder could not be loaded")


# ---------------------------------------------------------------------------
# Tokenizing and sentence splitting
# ---------------------------------------------------------------------------


def test_tokenize_words_lowercases_keeps_contractions_and_drops_stopwords_and_digits():
    tokens = _tokenize_words("The Quick brown fox's 42 jumps, isn't it?")

    assert tokens == ["quick", "brown", "fox's", "jumps", "isn't"]


def test_split_sentences_splits_on_terminal_punctuation_and_drops_empty_parts():
    parts = _split_sentences("Hello there!!  How are you?... Fine. ")

    assert parts == ["Hello there", "How are you", "Fine"]
    assert _split_sentences("...  !!!") == []


# ---------------------------------------------------------------------------
# Behavioral fingerprint extraction
# ---------------------------------------------------------------------------


def test_fingerprint_counts_punctuation_per_hundred_words_and_separates_ellipsis():
    fp = _extract_fingerprint(["Hello there! Wait... really?", "Yes; indeed: ok."])

    # Content words after stopword removal: hello, wait, really, indeed, ok (5).
    # One of each punctuation mark is 1 per 5 words = 20 per 100 words. The dots
    # of the ellipsis must not be double counted as anything else.
    assert fp.punctuation == {"?": 20.0, "!": 20.0, ";": 20.0, ":": 20.0, "...": 20.0}


def test_fingerprint_ellipsis_needs_three_dots_and_a_longer_run_counts_once():
    fp = _extract_fingerprint(["Hmm.. ok.... fine"])

    # Content words: hmm, ok, fine (3). ".." is not an ellipsis; "...." is one.
    assert fp.punctuation["..."] == pytest.approx(100 / 3)


def test_fingerprint_length_statistics_use_sentences_and_whitespace_words():
    fp = _extract_fingerprint(["Hello there! Wait... really?", "Yes; indeed: ok."])

    # Sentences: "Hello there"(2) "Wait"(1) "really"(1) "Yes; indeed: ok"(3).
    assert fp.sentence_len_mean == pytest.approx(1.75)
    assert fp.sentence_len_std == pytest.approx(0.829156, abs=1e-5)
    # Messages hold 4 and 3 whitespace-separated words.
    assert fp.message_len_mean == pytest.approx(3.5)
    assert fp.message_len_std == pytest.approx(0.5)


def test_fingerprint_register_rates_are_share_of_content_words():
    fp = _extract_fingerprint(["Therefore, yeah lol"])

    assert fp.register["formal"] == pytest.approx(1 / 3)
    assert fp.register["casual"] == pytest.approx(2 / 3)
    assert fp.register["technical"] == 0.0
    assert fp.register["emotive"] == 0.0


def test_fingerprint_register_tracks_technical_and_emotive_markers():
    fp = _extract_fingerprint(["function parameter love hate"])

    assert fp.register["technical"] == pytest.approx(0.5)
    assert fp.register["emotive"] == pytest.approx(0.5)
    assert fp.register["formal"] == 0.0
    assert fp.register["casual"] == 0.0


def test_fingerprint_vocab_is_top_k_by_frequency_and_renormalized():
    fp = _extract_fingerprint(["alpha alpha alpha beta beta gamma delta"], top_k=2)

    assert list(fp.vocab) == ["alpha", "beta"]
    assert fp.vocab["alpha"] == pytest.approx(0.6)
    assert fp.vocab["beta"] == pytest.approx(0.4)
    assert sum(fp.vocab.values()) == pytest.approx(1.0)


def test_fingerprint_of_an_empty_window_is_all_zero_rather_than_an_error():
    fp = _extract_fingerprint([])

    assert fp.vocab == {}
    assert fp.sentence_len_mean == 0.0
    assert fp.message_len_mean == 0.0
    assert fp.message_len_std == 0.0
    assert all(v == 0.0 for v in fp.register.values())
    assert all(v == 0.0 for v in fp.punctuation.values())


# ---------------------------------------------------------------------------
# Divergence primitives
# ---------------------------------------------------------------------------


def test_js_divergence_is_zero_for_identical_and_one_for_disjoint_distributions():
    assert _js_divergence({"a": 1.0}, {"a": 1.0}) == pytest.approx(0.0)
    assert _js_divergence({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0)


def test_js_divergence_of_nothing_or_zero_mass_is_zero():
    assert _js_divergence({}, {}) == 0.0
    assert _js_divergence({"a": 0.0}, {"b": 0.0}) == 0.0


def test_js_divergence_is_symmetric_and_grows_with_distribution_shift():
    p = {"a": 0.5, "b": 0.5}
    near = {"a": 0.6, "b": 0.4}
    far = {"a": 0.95, "b": 0.05}

    assert _js_divergence(p, far) == pytest.approx(_js_divergence(far, p))
    assert 0.0 < _js_divergence(p, near) < _js_divergence(p, far) < 1.0


def test_bounded_diff_scales_saturates_and_guards_a_non_positive_scale():
    assert _bounded_diff(1.0, 2.0, 4.0) == pytest.approx(0.25)
    assert _bounded_diff(2.0, 1.0, 4.0) == pytest.approx(0.25)
    assert _bounded_diff(0.0, 100.0, 4.0) == 1.0
    assert _bounded_diff(1.0, 5.0, 0.0) == 0.0
    assert _bounded_diff(1.0, 5.0, -3.0) == 0.0


def test_l1_rate_distance_sums_absolute_gaps_saturates_and_guards_degenerate_input():
    assert _l1_rate_distance({"a": 0.1}, {"a": 0.0}, 0.2) == pytest.approx(0.5)
    # Keys present on only one side count their full mass.
    assert _l1_rate_distance({"a": 1.0}, {"b": 1.0}, 4.0) == pytest.approx(0.5)
    assert _l1_rate_distance({"a": 1.0}, {"b": 1.0}, 1.0) == 1.0
    assert _l1_rate_distance({}, {}, 1.0) == 0.0
    assert _l1_rate_distance({"a": 1.0}, {"a": 0.0}, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Fingerprint drift score across a window of outputs
# ---------------------------------------------------------------------------


def test_fingerprint_drift_needs_at_least_two_outputs():
    for outputs in ([], ["only one message"]):
        score, baseline, current = _compute_fingerprint_drift_score(outputs)

        assert score == 0.0
        assert baseline.vocab == {} and current.vocab == {}


def test_fingerprint_drift_is_zero_when_baseline_and_current_windows_are_identical():
    score, baseline, current = _compute_fingerprint_drift_score(
        ["same words here", "same words here"]
    )

    assert score == 0.0
    assert baseline.vocab == {} and current.vocab == {}


def test_fingerprint_drift_stays_low_for_stable_style_and_high_for_register_flip():
    stable, _, _ = _compute_fingerprint_drift_score(
        FORMAL_HISTORY + ["Therefore the budget is ready. Please review the figures."]
    )
    flipped, baseline, current = _compute_fingerprint_drift_score(FORMAL_HISTORY + [SLANG_REPLY])

    assert stable < 0.20
    assert flipped >= 0.5
    assert flipped <= 1.0
    # The returned fingerprints describe the two halves of the window.
    assert baseline.register["casual"] == 0.0
    assert current.register["casual"] > 0.5
    assert baseline.register["formal"] > current.register["formal"]


def test_fingerprint_drift_window_size_narrows_baseline_and_current_windows():
    outputs = [
        FORMAL_HISTORY[0],
        FORMAL_HISTORY[1],
        FORMAL_HISTORY[2],
        FORMAL_HISTORY[0],
        SLANG_REPLY,
        SLANG_REPLY,
    ]

    wide, wide_base, wide_cur = _compute_fingerprint_drift_score(outputs, window_size=3)
    narrow, narrow_base, narrow_cur = _compute_fingerprint_drift_score(outputs, window_size=1)

    # A window of 3 mixes a formal message into the "current" half; a window of 1
    # compares only the first message against the last, so the shift looks larger.
    assert narrow > wide
    assert wide_cur.register["formal"] > 0.0
    assert narrow_cur.register["formal"] == 0.0
    assert narrow_base.register["casual"] == 0.0


def test_fingerprint_drift_of_a_complete_flip_weighs_every_component():
    score, _, _ = _compute_fingerprint_drift_score(
        ["therefore consequently furthermore thus " * 5, "yeah lol gonna kinda dude!!!??? " * 5]
    )

    # Hand-derived from the two messages (20 formal words, then 25 slang words):
    #   vocabulary: disjoint word sets, JS divergence 1.0          -> 0.20 * 1.0
    #   register: formal rate 1.0 vs casual rate 1.0, saturates    -> 0.45 * 1.0
    #   sentence length: one 20-word sentence vs 5-word sentences  -> 0.05 * (15 / 20)
    #   message length: 20 vs 25 words                             -> 0.05 * (5 / 80)
    #   punctuation: "!" and "?" at 60 per 100 words saturate,
    #   the other three marks are unchanged (2 of 5 marks)         -> 0.25 * (2 / 5)
    expected = 0.20 * 1.0 + 0.45 * 1.0 + 0.05 * (15 / 20) + 0.05 * (5 / 80) + 0.25 * (2 / 5)
    assert score == pytest.approx(expected)
    assert score == pytest.approx(0.790625)


def test_fingerprint_drift_of_a_small_shift_scales_each_component_below_saturation():
    filler = (
        "apple river mountain window garden pencil bridge candle forest harbor island jungle "
        "kitchen ladder meadow needle ocean pillow quarry rabbit saddle tunnel umbrella valley "
        "wagon zebra anchor basket castle dragon engine feather glacier hammer igloo jacket "
        "kettle lantern mirror"
    )
    formal = f"therefore {filler}"
    slipped = f"yeah {filler}?"

    score, baseline, current = _compute_fingerprint_drift_score([formal, slipped])

    # Two 40-word messages that differ by one marker word and one question mark:
    #   vocabulary: two of 40 words swapped, JS divergence 0.025  -> 0.20 * 0.025
    #   register: formal 1/40 -> casual 1/40, L1 0.05 over the 0.20 scale
    #                                                            -> 0.45 * (0.05 / 0.20)
    #   sentence and message length are unchanged                -> 0
    #   punctuation: one "?" is 2.5 per 100 words, 2.5 / 5 of one mark in five
    #                                                            -> 0.25 * (0.5 / 5)
    assert baseline.register["formal"] == pytest.approx(1 / 40)
    assert current.register["casual"] == pytest.approx(1 / 40)
    assert current.punctuation["?"] == pytest.approx(2.5)
    expected = 0.20 * 0.025 + 0.45 * (0.05 / 0.20) + 0.25 * (0.5 / 5)
    assert score == pytest.approx(expected)
    assert score == pytest.approx(0.1425)


# ---------------------------------------------------------------------------
# Declared-register (style) compliance
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def scorer() -> PersonaConsistencyScorer:
    return PersonaConsistencyScorer()


def test_style_compliance_is_full_when_formal_output_shows_three_register_markers(scorer):
    output = "Certainly. Kindly review the attached file. Thank you, regards."

    assert scorer._compute_style_compliance("formal Professional Assistant", output) == 1.0


def test_style_compliance_scales_with_the_number_of_register_markers(scorer):
    persona = "formal Professional Assistant"

    one = scorer._compute_style_compliance(persona, "Kindly review the attached file.")
    two = scorer._compute_style_compliance(persona, "Certainly. Kindly review the attached file.")

    assert one == pytest.approx(1 / 3)
    assert two == pytest.approx(2 / 3)


def test_style_compliance_is_zero_when_output_breaks_the_declared_register(scorer):
    persona = "formal Professional Assistant"
    compliant = "Certainly. Kindly review the attached file. Thank you, regards."
    broken = compliant + " Anyway, lol, that was fun."

    assert scorer._compute_style_compliance(persona, compliant) == 1.0
    # Same compliance markers, but one slang marker breaks the register entirely.
    assert scorer._compute_style_compliance(persona, broken) == 0.0


def test_style_compliance_is_inert_for_a_persona_that_declares_no_register(scorer):
    output = "Certainly. Kindly review the attached file. Thank you, regards."

    assert scorer._compute_style_compliance("You are a weather assistant.", output) == 0.0


def test_style_compliance_supports_a_declared_casual_register(scorer):
    persona = "a friendly relaxed helper"

    casual = scorer._compute_style_compliance(persona, "Hey, sure thing, no worries, awesome!")
    stiff = scorer._compute_style_compliance(
        persona, "Hey, sure thing, no worries. Furthermore, this holds."
    )

    assert casual == 1.0
    assert stiff == 0.0


def test_style_compliance_takes_the_best_register_when_persona_declares_both(scorer):
    persona = "a formal but casual assistant"

    # Formal markers with no casual markers: the casual register scores 0 but the
    # formal one is fully honored, and the best register wins.
    assert scorer._compute_style_compliance(persona, "Certainly kindly please thank you") == 1.0
    # A slang violation zeroes the formal register but the casual one still complies.
    assert scorer._compute_style_compliance(persona, "hey cool awesome") == 1.0


def test_a_declared_register_keeps_a_reply_consistent_until_slang_breaks_it(scorer):
    agent = Agent(id="fmt", persona_description="formal Professional Assistant", allowed_actions=[])
    compliant = "Certainly. Kindly review the attached file. Thank you, regards."
    slipped = compliant + " Anyway, lol, that was fun."

    held = scorer.score_consistency(agent, compliant)
    broken = scorer.score_consistency(agent, slipped)

    # The persona names no domain, so the declared register is its only compliance signal.
    assert held.factors["role_action_boost"] == 0.0
    assert held.factors["style_compliance"] == 1.0
    assert held.factors["persona_compliance"] == 1.0
    assert held.consistent is True
    assert held.drift_detected is False

    assert broken.factors["style_compliance"] == 0.0
    assert broken.factors["persona_compliance"] == 0.0
    assert broken.drift_detected is True
    assert broken.consistent is False
    assert broken.score < held.score


def test_short_personas_weight_declared_compliance_above_embedding_similarity(scorer):
    compliant = "Certainly. Kindly review the attached file. Thank you, regards."
    short_persona = "formal Professional Assistant"
    long_persona = (
        "You are a formal professional assistant who answers customer questions about "
        "invoices, refunds and delivery times."
    )
    assert len(short_persona.split()) < 15 <= len(long_persona.split())

    short = scorer.score_consistency(
        Agent(id="s", persona_description=short_persona, allowed_actions=[]), compliant
    )
    long = scorer.score_consistency(
        Agent(id="l", persona_description=long_persona, allowed_actions=[]), compliant
    )

    def rebuilt(result, semantic_weight, compliance_weight):
        f = result.factors
        return (
            f["semantic_similarity"] * semantic_weight
            + f["lexical_overlap"] * 0.20
            + f["tone_consistency"] * 0.15
            + f["persona_compliance"] * compliance_weight
            + f["flexibility_bonus"]
        )

    assert short.factors["persona_compliance"] == 1.0
    assert long.factors["persona_compliance"] == 1.0
    # A short persona with strong compliance trusts compliance (0.45) over embeddings
    # (0.20); a long persona keeps the balanced 0.30 / 0.35 weighting.
    assert short.score == pytest.approx(rebuilt(short, 0.20, 0.45), abs=1e-3)
    assert long.score == pytest.approx(rebuilt(long, 0.30, 0.35), abs=1e-3)
    assert short.consistent is True


# ---------------------------------------------------------------------------
# Task-relevance and off-role guards that need no embedder
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task", [None, "", "weather", "   "])
def test_task_relevance_is_zero_when_task_is_missing_or_too_thin(scorer, task):
    result = scorer._compute_task_relevance(
        "You are a helpful weather assistant.", task, "It is sunny in Paris today."
    )

    assert result == 0.0


def test_offrole_excursion_is_inert_without_a_task(scorer):
    output = (
        "Order the pasta with garlic today for lunch. "
        "Meanwhile the moon orbits quietly around distant planets."
    )

    assert scorer._has_offrole_excursion("You are a helpful assistant.", None, output) is False
    assert scorer._has_offrole_excursion("You are a helpful assistant.", "", output) is False


def test_offrole_excursion_needs_two_substantial_sentences(scorer):
    persona = "You are a helpful banking assistant."
    task = "How do I open a savings account?"

    # Only one sentence reaches the minimum length; short fragments are ignored.
    one_long = "Bring photo identification and an initial deposit to any branch. Thanks. Bye now."
    single = "Bring photo identification and an initial deposit to any branch today."

    assert scorer._has_offrole_excursion(persona, task, one_long) is False
    assert scorer._has_offrole_excursion(persona, task, single) is False


# ---------------------------------------------------------------------------
# score_consistency: empty-persona exclusion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("persona_description", ["", "   \n\t ", None])
def test_empty_persona_is_excluded_as_nothing_to_drift_from(scorer, persona_description):
    agent = Agent(id="a", persona_description=persona_description, allowed_actions=[])

    result = scorer.score_consistency(agent, "Here is a cheerful recipe for apple pie.")

    assert result.consistent is True
    assert result.drift_detected is False
    assert result.score == 1.0
    assert result.method == "exclusion_empty_persona"
    assert result.confidence == 0.0
    assert result.drift_magnitude is None
    assert result.issues is None
    assert result.role_type is None
    assert result.factors == {}
    assert result.raw_score is None
    assert "no persona_description" in result.evidence["reason"]


def test_same_output_drifts_under_a_real_persona_but_not_under_an_empty_one(scorer):
    output = "Here is a cheerful recipe for apple pie."
    assigned = Agent(
        id="a",
        persona_description="A careful security analyst who audits vulnerabilities.",
        allowed_actions=[],
    )
    unassigned = Agent(id="a", persona_description="", allowed_actions=[])

    with_persona = scorer.score_consistency(assigned, output)
    without_persona = scorer.score_consistency(unassigned, output)

    assert with_persona.drift_detected is True
    assert with_persona.consistent is False
    assert without_persona.drift_detected is False
    assert without_persona.consistent is True


def test_empty_persona_exclusion_returns_before_any_history_or_embedding_work(scorer):
    agent = Agent(id="a", persona_description="", allowed_actions=[])

    # Three recent outputs would normally trigger embedding-based drift magnitude
    # and a fingerprint check; the exclusion must short-circuit all of it.
    result = scorer.score_consistency(
        agent,
        SLANG_REPLY,
        recent_outputs=FORMAL_HISTORY,
        output_history=FORMAL_HISTORY,
        task="How is the weather?",
    )

    assert result.method == "exclusion_empty_persona"
    assert result.drift_detected is False
    assert "fingerprint_drift" not in result.factors


# ---------------------------------------------------------------------------
# score_consistency: behavioral fingerprint fusion
# ---------------------------------------------------------------------------


def test_fingerprint_drift_catches_a_register_slide_that_text_patterns_miss(scorer):
    agent = _legal_agent()

    without_history = scorer.score_consistency(agent, LEGAL_SLANG)
    with_history = scorer.score_consistency(agent, LEGAL_SLANG, output_history=LEGAL_HISTORY)

    # On its own the slang reply reads as on-persona: legal vocabulary satisfies the
    # role-action signal and nothing declares a register.
    assert without_history.consistent is True
    assert without_history.drift_detected is False
    assert "fingerprint_drift" not in without_history.factors
    assert without_history.issues is None

    # Against the agent's own earlier outputs the same reply is a clear shift.
    fingerprint = with_history.factors["fingerprint_drift"]
    assert fingerprint >= 0.20
    assert with_history.drift_detected is True
    assert with_history.consistent is False
    assert with_history.method == "multi_factor_scoring_with_drift"
    fingerprint_issues = [i for i in with_history.issues if i.startswith("[fingerprint_drift]")]
    assert len(fingerprint_issues) == 1
    assert "Behavioral fingerprint shift detected" in fingerprint_issues[0]
    assert f"{fingerprint:.2f}" in fingerprint_issues[0]
    # The drift is fused in: consistency is capped at 1 - fingerprint drift, which
    # is below what the text-pattern signals alone produced.
    assert with_history.score == pytest.approx(1.0 - fingerprint, abs=1e-4)
    assert with_history.score < without_history.score


def test_fingerprint_drift_stays_silent_when_the_agent_keeps_its_register(scorer):
    result = scorer.score_consistency(
        _legal_agent(), LEGAL_SAME_REGISTER, output_history=LEGAL_HISTORY
    )

    assert result.factors["fingerprint_drift"] < 0.20
    assert result.drift_detected is False
    assert result.consistent is True
    assert result.method == "multi_factor_scoring"
    assert result.issues is None


def test_fingerprint_flag_fires_at_the_documented_threshold_not_on_the_fused_score(scorer):
    agent = _legal_agent()

    slip = scorer.score_consistency(agent, LEGAL_MILD_SLIP, output_history=LEGAL_HISTORY)
    emphatic = scorer.score_consistency(agent, LEGAL_EMPHATIC, output_history=LEGAL_HISTORY)
    slip_alone = scorer.score_consistency(agent, LEGAL_MILD_SLIP)

    # A mild slip: drift just over the 0.20 threshold but under 0.35, so the fused
    # score (1 - drift) is still ABOVE the specialist consistency threshold. The
    # reply is flagged because the fingerprint check fired, not because the score fell.
    slip_drift = slip.factors["fingerprint_drift"]
    assert 0.20 <= slip_drift < 0.35
    assert slip.score > slip.evidence["consistency_threshold"]
    assert slip.score == pytest.approx(1.0 - slip_drift, abs=1e-4)
    assert slip.drift_detected is True
    assert slip.consistent is False
    # The same reply with no history to compare against passes on text patterns alone.
    assert slip_alone.consistent is True
    assert slip_alone.drift_detected is False

    # Emphatic punctuation alone moves the fingerprint but stays under the threshold.
    emphatic_drift = emphatic.factors["fingerprint_drift"]
    assert 0.15 <= emphatic_drift < 0.20
    assert emphatic.drift_detected is False
    assert emphatic.consistent is True
    assert emphatic.method == "multi_factor_scoring"
    assert emphatic.issues is None


def test_recent_outputs_are_the_history_fallback_when_output_history_is_absent(scorer):
    result = scorer.score_consistency(
        _legal_agent(), LEGAL_SLANG, recent_outputs=LEGAL_HISTORY[:2]
    )

    # Two recent outputs is below the embedding-drift minimum of three, so this is
    # the fingerprint check alone deciding.
    assert result.drift_magnitude is None
    assert result.factors["fingerprint_drift"] >= 0.20
    assert result.drift_detected is True
    assert any(i.startswith("[fingerprint_drift]") for i in result.issues)


def test_explicit_output_history_takes_precedence_over_recent_outputs(scorer):
    agent = _legal_agent()

    overridden = scorer.score_consistency(
        agent, LEGAL_SLANG, recent_outputs=LEGAL_HISTORY[:2], output_history=[]
    )

    # An explicit empty history means "no history": the recent outputs are ignored.
    assert "fingerprint_drift" not in overridden.factors
    assert overridden.drift_detected is False


def test_a_single_prior_output_is_not_enough_history_for_a_fingerprint(scorer):
    result = scorer.score_consistency(
        _legal_agent(), LEGAL_SLANG, output_history=[LEGAL_HISTORY[0]]
    )

    assert "fingerprint_drift" not in result.factors
    assert result.drift_detected is False


def test_fingerprint_window_argument_changes_how_much_history_is_compared(scorer):
    agent = _legal_agent()
    history = LEGAL_HISTORY + [LEGAL_HISTORY[0], LEGAL_SLANG]

    wide = scorer.score_consistency(
        agent, LEGAL_SLANG, output_history=history, fingerprint_window=5
    )
    narrow = scorer.score_consistency(
        agent, LEGAL_SLANG, output_history=history, fingerprint_window=1
    )

    # With a window of 1 the comparison is exactly first message vs the new output;
    # a wide window dilutes the slang-heavy tail into the baseline as well.
    assert narrow.factors["fingerprint_drift"] > wide.factors["fingerprint_drift"]
    assert narrow.drift_detected is True


def test_role_type_is_auto_detected_for_the_fingerprint_result(scorer):
    result = scorer.score_consistency(
        _legal_agent(), LEGAL_SLANG, output_history=LEGAL_HISTORY
    )

    assert result.role_type is RoleType.SPECIALIST
    assert result.evidence["role_type"] == "specialist"


# ---------------------------------------------------------------------------
# Task-relevance and off-role excursion guards (need the sentence embedder)
# ---------------------------------------------------------------------------

WEATHER_PERSONA = "You are a helpful weather assistant."
WEATHER_TASK = "What is the weather forecast in Paris tomorrow?"
WEATHER_ANSWER = (
    "Tomorrow in Paris the weather forecast is sunny with a high of 22 degrees and light wind."
)
BANK_PERSONA = "You are a formal professional banking assistant."
BANK_TASK = "How do I open a savings account at your bank?"
BANK_ANSWER = (
    "Certainly. To open a savings account you need a valid photo identification and a small "
    "initial deposit at any branch. Kindly bring the documents, thank you."
)
BANK_ANSWER_WITH_EXCURSION = (
    "Certainly. To open a savings account you need a valid photo identification and a small "
    "initial deposit at any branch. Meanwhile the recipe for apple pie needs cinnamon and "
    "sweet baked apples in the oven. Kindly bring the documents, thank you."
)


def test_task_relevance_is_high_for_an_in_domain_request_that_is_fully_answered():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()

    relevance = scorer._compute_task_relevance(WEATHER_PERSONA, WEATHER_TASK, WEATHER_ANSWER)

    assert relevance > 0.8
    assert relevance <= 1.0


def test_task_relevance_is_zero_when_the_request_is_outside_the_persona_domain():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()
    legal_persona = "You are a legal advisor who explains statutes and contracts."

    # The weather question is answered well, but a legal bot should not be doing it.
    assert scorer._compute_task_relevance(legal_persona, WEATHER_TASK, WEATHER_ANSWER) == 0.0


def test_task_relevance_is_zero_when_the_output_does_not_answer_the_request():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()

    drifted = "Forget the forecast, banks are robbing you, move all your savings into crypto now."
    unrelated = "Here is a cheerful recipe for apple pie with cinnamon."

    assert scorer._compute_task_relevance(WEATHER_PERSONA, WEATHER_TASK, drifted) == 0.0
    assert scorer._compute_task_relevance(WEATHER_PERSONA, WEATHER_TASK, unrelated) == 0.0


def test_task_relevance_scales_with_how_fully_the_output_answers_the_request():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()

    brief = scorer._compute_task_relevance(
        WEATHER_PERSONA, WEATHER_TASK, "It will be sunny in Paris."
    )
    full = scorer._compute_task_relevance(WEATHER_PERSONA, WEATHER_TASK, WEATHER_ANSWER)
    echo = scorer._compute_task_relevance(WEATHER_PERSONA, WEATHER_TASK, WEATHER_TASK)

    # Both replies clear the answer gate, but the more fully responsive one scores higher,
    # and a perfect match is capped at 1.0 rather than overshooting the scale.
    assert 0.0 < brief < full < 1.0
    assert echo == 1.0


def test_task_relevance_rejects_topic_adjacent_drift_but_accepts_a_real_answer():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()
    banking = "You are a helpful banking assistant."

    # The source's own example: the request is in the persona's domain and the reply
    # stays on the request's topic, yet abandons the role. It sits below the answer gate.
    drift = "Banks are robbing you, move all your savings into crypto now."
    answer = "Open a savings account online."

    assert scorer._compute_task_relevance(banking, BANK_TASK, drift) == 0.0
    assert scorer._compute_task_relevance(banking, BANK_TASK, answer) > 0.7


def test_offrole_excursion_counts_a_six_word_sentence_but_not_a_five_word_one():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()
    banking = "You are a helpful banking assistant."
    on_task = "Bring photo identification and an initial deposit to any branch."

    six_words = f"{on_task} Apple pie needs cinnamon and apples."
    five_words = f"{on_task} Apple pie needs cinnamon apples."

    # The off-role sentence is only substantial from six words up.
    assert scorer._has_offrole_excursion(banking, BANK_TASK, six_words) is True
    assert scorer._has_offrole_excursion(banking, BANK_TASK, five_words) is False


def test_offrole_excursion_needs_a_sentence_far_from_persona_and_task_together():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()
    weather_answer = (
        "Tomorrow in Paris the weather forecast is sunny with a high of 22 degrees. "
        "The wind will stay light and the forecast for Paris stays dry all day."
    )
    weather_talk = (
        "Rain is expected across the region tonight with strong winds and falling "
        "temperatures. The forecast shows clear skies and sunshine returning by tomorrow morning."
    )
    far_from_both = (
        "The recipe for apple pie needs cinnamon and sweet baked apples in the oven. "
        "Banks are robbing you, so move all your savings into crypto right now."
    )

    # Far from a generic persona but squarely on the task: an on-task answer, not an excursion.
    assert (
        scorer._has_offrole_excursion("You are a helpful assistant.", WEATHER_TASK, weather_answer)
        is False
    )
    # Far from the task but squarely on the persona: still in role, not an excursion.
    assert (
        scorer._has_offrole_excursion(WEATHER_PERSONA, "How do I bake an apple pie?", weather_talk)
        is False
    )
    # Far from both persona and task is the excursion.
    assert scorer._has_offrole_excursion(WEATHER_PERSONA, WEATHER_TASK, far_from_both) is True


def test_offrole_excursion_flags_a_sentence_far_from_both_persona_and_task():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()
    banking = "You are a helpful banking assistant."
    on_task = (
        "To open a savings account you need a valid photo identification and a small initial "
        "deposit at any branch. You can also start the application online with your account "
        "number."
    )
    answer_then_drift = (
        "To open a savings account you need a valid photo identification and a small initial "
        "deposit at any branch. Meanwhile the recipe for apple pie needs cinnamon and sweet "
        "baked apples in the oven."
    )

    assert scorer._has_offrole_excursion(banking, BANK_TASK, answer_then_drift) is True
    assert scorer._has_offrole_excursion(banking, BANK_TASK, on_task) is False


def test_task_relevance_lets_an_on_persona_answer_pass_and_is_inert_without_a_task():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()
    agent = Agent(id="weather", persona_description=WEATHER_PERSONA, allowed_actions=[])

    with_task = scorer.score_consistency(agent, WEATHER_ANSWER, task=WEATHER_TASK)
    without_task = scorer.score_consistency(agent, WEATHER_ANSWER)

    assert with_task.factors["task_relevance"] > 0.8
    assert with_task.factors["offrole_excursion"] == 0.0
    assert with_task.consistent is True
    assert without_task.factors["task_relevance"] == 0.0


def test_offrole_excursion_stops_a_compliant_surface_from_laundering_drift():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()
    agent = Agent(id="bank", persona_description=BANK_PERSONA, allowed_actions=[])

    benign = scorer.score_consistency(agent, BANK_ANSWER, task=BANK_TASK)
    no_task = scorer.score_consistency(agent, BANK_ANSWER_WITH_EXCURSION)
    evasion = scorer.score_consistency(agent, BANK_ANSWER_WITH_EXCURSION, task=BANK_TASK)

    # A genuine on-task answer in the declared register is compliant.
    assert benign.consistent is True
    assert benign.factors["offrole_excursion"] == 0.0
    assert benign.factors["persona_compliance"] > 0.0
    # Without a task the guard is inert and the register-compliant reply passes.
    assert no_task.consistent is True
    assert no_task.factors["offrole_excursion"] == 0.0
    # With the task, the same reply's off-role sentence voids the compliance signals.
    assert evasion.factors["offrole_excursion"] == 1.0
    assert evasion.factors["style_compliance"] == 1.0
    assert evasion.factors["persona_compliance"] == 0.0
    assert evasion.drift_detected is True
    assert evasion.consistent is False
    assert evasion.score < no_task.score



def test_offrole_flag_is_only_raised_when_it_voids_a_compliance_signal():
    _embedder_or_skip()
    scorer = PersonaConsistencyScorer()
    agent = Agent(id="weather", persona_description=WEATHER_PERSONA, allowed_actions=[])
    off_role = (
        "The recipe for apple pie needs cinnamon and sweet baked apples in the oven. "
        "Banks are robbing you, so move all your savings into crypto right now."
    )

    # The reply really does contain an off-role excursion...
    assert scorer._has_offrole_excursion(WEATHER_PERSONA, WEATHER_TASK, off_role) is True

    result = scorer.score_consistency(agent, off_role, task=WEATHER_TASK)

    # ...but no compliance signal was propping it up, so there is nothing to void: the
    # flag stays clear and the low text-pattern score alone marks the drift.
    assert result.factors["role_action_boost"] == 0.0
    assert result.factors["style_compliance"] == 0.0
    assert result.factors["task_relevance"] == 0.0
    assert result.factors["persona_compliance"] == 0.0
    assert result.factors["offrole_excursion"] == 0.0
    assert result.drift_detected is True
    assert result.consistent is False
