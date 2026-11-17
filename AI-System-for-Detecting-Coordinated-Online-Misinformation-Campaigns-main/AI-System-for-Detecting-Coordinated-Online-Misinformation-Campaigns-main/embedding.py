"""Sentence embedding helpers using scikit-learn TF-IDF + SVD (torch-free)."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer


# ─── Lightweight model wrapper compatible with the rest of the codebase ────────

class _TFIDFEmbedder:
    """Drop-in replacement for SentenceTransformer using TF-IDF + SVD LSA."""

    N_COMPONENTS = 128  # embedding dimensionality

    def __init__(self):
        self._pipeline: Pipeline | None = None
        self._fitted = False

    def _build(self, texts: list[str]):
        self._pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                strip_accents="unicode",
                analyzer="word",
                ngram_range=(1, 2),
                max_features=20_000,
                sublinear_tf=True,
            )),
            ("svd", TruncatedSVD(n_components=self.N_COMPONENTS, random_state=42)),
            ("norm", Normalizer(copy=False)),
        ])
        self._pipeline.fit(texts)
        self._fitted = True

    def encode(
        self,
        texts: list[str],
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.N_COMPONENTS))

        if not self._fitted:
            self._build(texts)

        vecs = self._pipeline.transform(texts)
        return np.array(vecs)


# ─── Public API (same signatures as before) ────────────────────────────────────

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> _TFIDFEmbedder:
    """Return a lightweight TF-IDF embedder (no PyTorch / DLL required)."""
    return _TFIDFEmbedder()


def embed_texts(model: _TFIDFEmbedder, texts: list[str]) -> np.ndarray:
    """Encode text posts into embedding vectors."""
    if not texts:
        return np.empty((0, _TFIDFEmbedder.N_COMPONENTS))
    return model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
