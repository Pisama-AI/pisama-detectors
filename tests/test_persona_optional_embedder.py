"""PersonaConsistencyScorer must degrade, not crash, without the embedding extra.

`get_shared_embedder()` returns None when the optional sentence-transformers
extra is not installed. Several scorer paths used to call `.encode` on that None
and raise AttributeError. Absence is modelled with a real subclass whose
`embedder` is None (exactly what `get_shared_embedder()` returns), so these run
identically with or without the extra installed.
"""

from typing import Any

import pytest

from pisama_detectors.detection.persona import Agent, PersonaConsistencyScorer

WEATHER = Agent(
    id="weather",
    persona_description=(
        "You are a helpful weather assistant that answers questions about forecasts"
    ),
    allowed_actions=[],
)
LEGAL = Agent(
    id="legal",
    persona_description="You are a legal contract reviewer who analyses clauses and liability",
    allowed_actions=[],
)
ON_PERSONA_OUTPUT = "The forecast for tomorrow is sunny with light wind."
TASK = "What is the weather going to be tomorrow in Paris?"
RECENT = ["Rain expected today.", "It is cloudy right now.", "Windy conditions later."]
MULTI_SENTENCE_OFF_ROLE = (
    "The forecast for tomorrow is sunny with light wind across the region. "
    "Meanwhile the quarterly earnings report shows strong revenue growth overall. "
    "Investors should consider rebalancing their portfolios into technology stocks."
)


class NoEmbedderScorer(PersonaConsistencyScorer):
    @property
    def embedder(self) -> Any:
        return None


def _summary(result: Any) -> tuple[Any, ...]:
    return (result.raw_score, result.consistent, result.drift_detected)


def test_task_is_inert_without_embedder() -> None:
    scorer = NoEmbedderScorer()

    with_task = scorer.score_consistency(WEATHER, ON_PERSONA_OUTPUT, task=TASK)
    without_task = scorer.score_consistency(WEATHER, ON_PERSONA_OUTPUT)

    assert _summary(with_task) == _summary(without_task)
    assert with_task.factors["task_relevance"] == 0.0
    assert with_task.factors["offrole_excursion"] == 0.0


def test_task_relevance_returns_zero_without_embedder() -> None:
    scorer = NoEmbedderScorer()

    assert (
        scorer._compute_task_relevance(WEATHER.persona_description, TASK, ON_PERSONA_OUTPUT)
        == 0.0
    )


def test_offrole_excursion_is_not_claimed_without_embedder() -> None:
    scorer = NoEmbedderScorer()

    assert (
        scorer._has_offrole_excursion(WEATHER.persona_description, TASK, MULTI_SENTENCE_OFF_ROLE)
        is False
    )


def test_recent_outputs_fall_back_to_score_based_drift_without_embedder() -> None:
    scorer = NoEmbedderScorer()

    with_history = scorer.score_consistency(WEATHER, ON_PERSONA_OUTPUT, recent_outputs=RECENT)
    without_history = scorer.score_consistency(WEATHER, ON_PERSONA_OUTPUT)

    assert with_history.drift_magnitude is None
    assert "drift_magnitude" not in with_history.factors
    assert _summary(with_history) == _summary(without_history)


def test_role_usurpation_abstains_without_embedder() -> None:
    scorer = NoEmbedderScorer()
    legal_text = "The indemnification clause limits liability under this contract."

    assert scorer.detect_role_usurpation(WEATHER, legal_text, [WEATHER, LEGAL]) is None


def test_real_scorer_never_raises_whether_or_not_the_extra_is_installed() -> None:
    scorer = PersonaConsistencyScorer()

    scorer.score_consistency(
        WEATHER, MULTI_SENTENCE_OFF_ROLE, task=TASK, recent_outputs=RECENT
    )
    scorer.detect_role_usurpation(WEATHER, ON_PERSONA_OUTPUT, [WEATHER, LEGAL])


def test_embedder_paths_still_work_when_the_extra_is_installed() -> None:
    pytest.importorskip("sentence_transformers")
    scorer = PersonaConsistencyScorer()
    if scorer.embedder is None:
        pytest.skip("semantic model unavailable")
    legal_text = "The indemnification clause limits liability under this contract."

    assert scorer.detect_role_usurpation(WEATHER, legal_text, [WEATHER, LEGAL]) == "legal"
    assert scorer.detect_role_usurpation(WEATHER, ON_PERSONA_OUTPUT, [WEATHER, LEGAL]) is None
    with_history = scorer.score_consistency(WEATHER, ON_PERSONA_OUTPUT, recent_outputs=RECENT)
    assert isinstance(with_history.drift_magnitude, float)
    assert "drift_magnitude" in with_history.factors
