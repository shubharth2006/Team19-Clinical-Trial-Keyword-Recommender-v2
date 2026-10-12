"""Project model interface and term-similarity utilities based on the cited method."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import norm


class AbstractLatentFactorModel(ABC):
    """Interface for a model that estimates missing term weights in study records."""

    def __init__(self, tau: float, k: float, alpha: float, lambda_: float):
        self._tau = tau
        self._k = k
        self._alpha = alpha
        self._lambda = lambda_
        self._b = None
        self._W_theta = None
        self._training_history = []
        self.mse_test = None
        self.apk_test = None

    def _calculate_neighbor_weight_matrix(self, utility: csc_matrix) -> np.ndarray:
        """Create term-to-term weights from cosine similarity over study records."""
        row_norms = np.asarray(norm(utility, ord=2, axis=1)).reshape(-1)
        row_norms[row_norms == 0] = 1.0
        normalized_terms = utility.multiply((1.0 / row_norms).reshape(-1, 1))
        cosine_similarity = normalized_terms.dot(normalized_terms.transpose()).toarray()

        distances = 1.0 - cosine_similarity
        affinities = np.exp(self._tau * np.power(distances, self._k))
        np.fill_diagonal(affinities, 0.0)
        return affinities

    @abstractmethod
    def train(self, *args, **kwargs):
        """Fit model parameters to a term-by-study utility matrix."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, values: np.ndarray) -> np.ndarray:
        """Return non-negative estimated utility values."""
        raise NotImplementedError
