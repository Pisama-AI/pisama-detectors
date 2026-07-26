"""Shared embedder singleton for OSS detectors.

Loads `all-MiniLM-L6-v2` via sentence-transformers on first use and wraps
the raw model in a thin numpy-returning adapter so detectors get a stable
`.encode(...) -> np.ndarray` + `.similarity(e1, e2) -> float` contract.

Pisama Cloud overrides this module to route through Voyage AI first.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

_embedder = None
DEFAULT_MINILM_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MINILM_REPOSITORY = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
DEFAULT_MINILM_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


class _MiniLMEmbedder:
    """Numpy-returning wrapper around SentenceTransformer.

    The backend uses an EmbeddingService with this same interface; detectors
    depend on `.encode()` returning numpy arrays and `.similarity()` returning
    a plain float. Calling raw SentenceTransformer.encode() can return torch
    tensors in newer versions, breaking downstream `round(...)` and arithmetic.
    """

    def __init__(self, model_name: str, revision: Optional[str] = None):
        from sentence_transformers import SentenceTransformer

        model_path = model_name
        if revision is not None:
            from huggingface_hub import hf_hub_download

            repository = (
                DEFAULT_MINILM_REPOSITORY
                if model_name == DEFAULT_MINILM_MODEL
                else model_name
            )
            downloaded_files = [
                hf_hub_download(
                    repo_id=repository,
                    filename=filename,
                    revision=revision,
                )
                for filename in DEFAULT_MINILM_FILES
            ]
            model_path = str(
                next(
                    Path(filename).parent
                    for filename in downloaded_files
                    if Path(filename).name == "modules.json"
                )
            )

        # Loading the pinned snapshot by local path keeps compatibility with
        # sentence-transformers 2.2, whose constructor had no revision keyword.
        self._model = SentenceTransformer(model_path)

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
        similarity = float(np.dot(e1, e2) / (norm1 * norm2))
        return max(-1.0, min(1.0, similarity))


def get_shared_embedder(
    model_name: str = DEFAULT_MINILM_MODEL,
):
    """Get the shared embedder singleton.

    The bundled MiniLM default is downloaded at a reviewed immutable revision.
    Custom model names retain their existing unpinned behavior.
    """
    global _embedder

    if _embedder is not None:
        return _embedder

    try:
        effective_revision = (
            DEFAULT_MINILM_REVISION if model_name == DEFAULT_MINILM_MODEL else None
        )
        logger.info(
            "Loading local SentenceTransformer: %s (revision=%s)",
            model_name,
            effective_revision or "default",
        )
        _embedder = _MiniLMEmbedder(model_name, revision=effective_revision)
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
