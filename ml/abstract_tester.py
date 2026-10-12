"""Common masking behaviour for the recommender's evaluation routines."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import csc_matrix

from ml.abstract_model import AbstractLatentFactorModel


class AbstractTester(ABC):
    """Base class for evaluating recovery of partially hidden study terms."""

    def __init__(
        self,
        model: AbstractLatentFactorModel,
        fraction_of_masking: float = 0.5,
    ):
        if not 0 <= fraction_of_masking < 1:
            raise ValueError("fraction_of_masking must be in [0, 1).")
        self._model = model
        self._fraction_of_masking = fraction_of_masking

    def choose_mask_indices(self, values: np.ndarray) -> np.ndarray:
        """Return a same-shaped mask with a reproducible fraction hidden by the caller's RNG."""
        array = np.asarray(values)
        known_positions = np.flatnonzero(array.reshape(-1) != 0)
        n_to_hide = int(len(known_positions) * self._fraction_of_masking)

        mask = np.ones(array.size, dtype=float)
        if n_to_hide:
            hidden_positions = np.random.choice(
                known_positions, size=n_to_hide, replace=False
            )
            mask[hidden_positions] = 0.0
        return mask.reshape(array.shape)

    @abstractmethod
    def test(self, values: np.ndarray):
        """Evaluate a single study vector."""
        raise NotImplementedError

    @abstractmethod
    def __call__(self, documents: csc_matrix):
        """Evaluate all non-empty study columns."""
        raise NotImplementedError
