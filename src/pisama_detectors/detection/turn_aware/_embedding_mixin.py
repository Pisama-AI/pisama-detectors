"""
Embedding Mixin for Turn-Aware Detectors
========================================

Provides embedding-based semantic analysis capabilities for detectors.
"""

import logging
from typing import List, Optional, Dict, Any

from ._base import EMBEDDING_SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

# Embedding availability flag (lazy-loaded)
EMBEDDING_AVAILABLE = None


def _check_embedding_available() -> bool:
    """Check if embedding service is available (lazy load)."""
    global EMBEDDING_AVAILABLE
    if EMBEDDING_AVAILABLE is None:
        try:
            from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder
            embedder = get_embedder()
            # Try a quick encode to verify it works
            _ = embedder.encode("test")
            EMBEDDING_AVAILABLE = True
            logger.info("Embedding service available for semantic detection")
        except Exception as e:
            EMBEDDING_AVAILABLE = False
            logger.warning(f"Embedding service not available, using keyword fallback: {e}")
    return EMBEDDING_AVAILABLE


class EmbeddingMixin:
    """Mixin providing embedding-based semantic analysis for detectors.

    Based on STATE_OF_THE_ART_DETECTOR_DESIGN.md recommendations:
    - Tier 2 detection using embedding similarity
    - Semantic drift detection for task alignment
    - Information density analysis
    """

    _embedder = None
    _embedder_lock = None  # Class-level lock for thread safety

    @classmethod
    def _get_embedder_lock(cls):
        """Get or create the class-level lock for embedder initialization."""
        if cls._embedder_lock is None:
            import threading
            cls._embedder_lock = threading.RLock()
        return cls._embedder_lock

    @property
    def embedder(self):
        """Lazy-load embedding service (thread-safe)."""
        lock = self._get_embedder_lock()
        with lock:
            if self._embedder is None and _check_embedding_available():
                try:
                    from pisama_detectors.detection.shared_embedder import get_shared_embedder as get_embedder
                    self._embedder = get_embedder()
                except Exception:
                    pass
        return self._embedder

    def semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.

        Returns:
            Cosine similarity score (0-1), or -1 if embeddings unavailable
        """
        if not self.embedder:
            return -1.0

        try:
            emb1 = self.embedder.encode(text1, is_query=True)
            emb2 = self.embedder.encode(text2, is_query=False)
            return self.embedder.similarity(emb1, emb2)
        except Exception as e:
            logger.debug(f"Embedding similarity failed: {e}")
            return -1.0

    def batch_semantic_similarity(
        self,
        query: str,
        passages: List[str]
    ) -> List[float]:
        """Compute semantic similarity between query and multiple passages.

        Returns:
            List of similarity scores, or empty list if unavailable
        """
        if not self.embedder or not passages:
            return []

        try:
            query_emb = self.embedder.encode_query(query)
            passage_embs = self.embedder.encode_passages(passages)
            similarities = self.embedder.batch_similarity(query_emb, passage_embs)
            return similarities.tolist()
        except Exception as e:
            logger.debug(f"Batch embedding similarity failed: {e}")
            return []

    def detect_semantic_drift(
        self,
        reference: str,
        responses: List[str],
        threshold: float = EMBEDDING_SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """Detect semantic drift from reference text across multiple responses.

        Based on MAST research: embedding similarity < 0.7 indicates significant drift.

        Returns:
            Dict with drift analysis including scores and drifted indices
        """
        similarities = self.batch_semantic_similarity(reference, responses)

        if not similarities:
            return {"available": False}

        drifted_indices = [i for i, sim in enumerate(similarities) if sim < threshold]
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Detect progressive drift (similarity decreasing over time)
        progressive = False
        if len(similarities) >= 3:
            first_half = similarities[:len(similarities)//2]
            second_half = similarities[len(similarities)//2:]
            first_avg = sum(first_half) / len(first_half) if first_half else 0
            second_avg = sum(second_half) / len(second_half) if second_half else 0
            progressive = second_avg < first_avg - 0.1  # 10% degradation

        return {
            "available": True,
            "similarities": similarities,
            "avg_similarity": avg_similarity,
            "drifted_indices": drifted_indices,
            "drift_detected": len(drifted_indices) > 0,
            "progressive_drift": progressive,
            "threshold": threshold,
        }

    def compute_information_density(self, text: str) -> float:
        """Estimate information density of text.

        Higher values = more substantive content.
        Based on: unique terms, sentence complexity, specificity markers.
        """
        if not text:
            return 0.0

        words = text.lower().split()
        if not words:
            return 0.0

        # Unique word ratio
        unique_ratio = len(set(words)) / len(words)

        # Specificity markers (numbers, technical terms, proper nouns)
        import re
        numbers = len(re.findall(r'\b\d+(?:\.\d+)?\b', text))
        technical = len(re.findall(r'\b[A-Z][a-z]*[A-Z]\w*\b', text))  # camelCase

        # Sentence complexity (words per sentence)
        sentences = max(1, text.count('.') + text.count('!') + text.count('?'))
        words_per_sentence = len(words) / sentences

        # Combine metrics
        density = (
            unique_ratio * 0.4 +
            min(1.0, numbers / 10) * 0.2 +
            min(1.0, technical / 5) * 0.2 +
            min(1.0, words_per_sentence / 20) * 0.2
        )

        return min(1.0, density)

    def contrastive_similarity(
        self,
        anchor: str,
        positive: str,
        negative: str,
        mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute contrastive similarity using triplet comparison.

        Uses TRACE framework approach: anchor should be more similar to
        positive than to negative by a margin.

        Args:
            anchor: Reference text (e.g., the trace being analyzed)
            positive: Example of expected behavior
            negative: Example of failure/anomaly
            mode: MAST failure mode for model selection

        Returns:
            Dict with similarity scores and classification
        """
        if not self.embedder:
            return {"available": False}

        try:
            result = self.embedder.compute_contrastive_score(
                anchor=anchor,
                positive=positive,
                negative=negative,
            )
            result["available"] = True

            # Classify based on which example is closer
            if result["pos_sim"] > result["neg_sim"]:
                result["classification"] = "normal"
                result["confidence"] = result["margin_score"]
            else:
                result["classification"] = "anomaly"
                result["confidence"] = -result["margin_score"]

            return result
        except Exception as e:
            logger.debug(f"Contrastive similarity failed: {e}")
            return {"available": False, "error": str(e)}

    def batch_semantic_drift(
        self,
        reference: str,
        responses: List[str],
        window_size: int = 3,
        threshold: float = EMBEDDING_SIMILARITY_THRESHOLD,
    ) -> Dict[str, Any]:
        """Detect semantic drift using efficient batch processing.

        Optimized for long traces by:
        1. Using sliding window for local drift detection
        2. Computing trend line for progressive drift
        3. Identifying sudden drift points

        Args:
            reference: Original task/context text
            responses: List of agent responses in order
            window_size: Size of sliding window for local analysis
            threshold: Similarity threshold for drift detection

        Returns:
            Dict with comprehensive drift analysis
        """
        if not self.embedder or not responses:
            return {"available": False}

        try:
            # Get similarities using parent method
            similarities = self.batch_semantic_similarity(reference, responses)
            if not similarities:
                return {"available": False}

            n = len(similarities)

            # Find drift points (sudden drops)
            drift_points = []
            for i in range(1, n):
                if similarities[i-1] - similarities[i] > 0.15:  # 15% drop
                    drift_points.append({
                        "index": i,
                        "drop": similarities[i-1] - similarities[i],
                        "before": similarities[i-1],
                        "after": similarities[i],
                    })

            # Sliding window analysis
            window_avgs = []
            for i in range(n - window_size + 1):
                window = similarities[i:i + window_size]
                window_avgs.append(sum(window) / len(window))

            # Compute trend (linear regression slope)
            if n >= 3:
                x = list(range(n))
                x_mean = sum(x) / n
                y_mean = sum(similarities) / n
                numerator = sum((x[i] - x_mean) * (similarities[i] - y_mean) for i in range(n))
                denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
                slope = numerator / denominator if denominator != 0 else 0
            else:
                slope = 0

            # Classify drift severity
            avg_sim = sum(similarities) / n
            drifted_count = sum(1 for s in similarities if s < threshold)

            if slope < -0.05 and drifted_count > n // 2:
                severity = "severe"
            elif slope < -0.02 or drifted_count > n // 3:
                severity = "moderate"
            elif drifted_count > 0:
                severity = "mild"
            else:
                severity = "none"

            return {
                "available": True,
                "similarities": similarities,
                "avg_similarity": avg_sim,
                "min_similarity": min(similarities),
                "max_similarity": max(similarities),
                "trend_slope": slope,
                "drift_points": drift_points,
                "window_averages": window_avgs,
                "drifted_count": drifted_count,
                "drifted_indices": [i for i, s in enumerate(similarities) if s < threshold],
                "severity": severity,
                "progressive_drift": slope < -0.02,
                "sudden_drift": len(drift_points) > 0,
            }
        except Exception as e:
            logger.debug(f"Batch semantic drift failed: {e}")
            return {"available": False, "error": str(e)}
