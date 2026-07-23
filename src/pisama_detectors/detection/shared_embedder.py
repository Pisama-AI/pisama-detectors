"""Shared embedder singleton for OSS detectors.

Loads `all-MiniLM-L6-v2` via sentence-transformers on first use and wraps
the raw model in a thin numpy-returning adapter so detectors get a stable
`.encode(...) -> np.ndarray` + `.similarity(e1, e2) -> float` contract.

Pisama Cloud overrides this module to route through Voyage AI first.
"""

import logging
from typing import List, Union

import numpy as np

logger = logging.getLogger(__name__)

_embedder = None


class _MiniLMEmbedder:
    """Numpy-returning wrapper around SentenceTransformer.

    The backend uses an EmbeddingService with this same interface; detectors
    depend on `.encode()` returning numpy arrays and `.similarity()` returning
    a plain float. Calling raw SentenceTransformer.encode() can return torch
    tensors in newer versions, breaking downstream `round(...)` and arithmetic.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        single = isinstance(texts, str)
        text_list = [texts] if single else list(texts)
        embeddings = self._model.encode(
            text_list,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings[0] if single else embeddings

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        e1 = np.asarray(embedding1).flatten()
        e2 = np.asarray(embedding2).flatten()
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(e1, e2) / (norm1 * norm2))


def get_shared_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Get the shared embedder singleton."""
    global _embedder

    if _embedder is not None:
        return _embedder

    try:
        logger.info(f"Loading local SentenceTransformer: {model_name}")
        _embedder = _MiniLMEmbedder(model_name)
        return _embedder
    except ModuleNotFoundError:
        logger.debug("Semantic detection is unavailable; install pisama-detectors[semantic]")
        return None
    except Exception as e:
        logger.warning(f"Local SentenceTransformer failed: {e}")
        return None


def clear_shared_embedder():
    """Free the shared embedder from memory."""
    global _embedder
    _embedder = None
