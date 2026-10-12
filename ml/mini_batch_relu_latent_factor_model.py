"""Mini-batch ReLU recommender trained over a term-by-study utility matrix."""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm

from ml.abstract_model import AbstractLatentFactorModel


class MiniBatchReLuLatentFactorModel(AbstractLatentFactorModel):
    """Learn non-negative term scores from term-neighbour weights and study records."""

    def __init__(
        self,
        tau: float = -2,
        k: float = 2,
        alpha: float = 0.01,
        lambda_: float = 0.0001,
        batch_size: int = 10000,
        max_epoch: int = 1000,
    ):
        super().__init__(tau, k, alpha, lambda_)
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if max_epoch < 1:
            raise ValueError("max_epoch must be positive.")
        self._batch_size = int(batch_size)
        self._max_epoch = int(max_epoch)

    @staticmethod
    def _relu(linear_scores: np.ndarray) -> np.ndarray:
        return np.maximum(linear_scores, 0.0)

    def _forward_dense(self, records: np.ndarray) -> np.ndarray:
        return self._relu(self._W_theta @ records + self._b)

    def _objective(self, records: np.ndarray) -> float:
        errors = self._forward_dense(records) - records
        return float(np.square(errors).sum() / (2 * records.shape[1]))

    def train(self, train_r: csc_matrix):
        """Fit model coefficients with mini-batch gradient descent and L2 penalty."""
        if train_r.shape[1] == 0:
            raise ValueError("Training matrix must contain at least one study.")

        targets = np.asarray(train_r.toarray(), dtype=np.float64)
        term_count, study_count = targets.shape

        neighbour_weights = self._calculate_neighbor_weight_matrix(train_r)
        coefficients = np.zeros((term_count, term_count), dtype=np.float64)
        bias = np.zeros((term_count, 1), dtype=np.float64)
        self._W_theta = neighbour_weights * coefficients
        self._b = bias
        self._training_history = [self._objective(targets)]
        print(f"Loss before training: {self._training_history[-1]:0.8E}")

        for epoch in range(self._max_epoch):
            starts = range(0, study_count, self._batch_size)
            progress = tqdm(starts, total=math.ceil(study_count / self._batch_size))
            for start in progress:
                stop = min(start + self._batch_size, study_count)
                batch = targets[:, start:stop]
                prediction_before_relu = self._W_theta @ batch + bias

                # The derivative is zero where ReLU clips the model output.
                residual = np.zeros_like(prediction_before_relu)
                active = prediction_before_relu >= 0
                residual[active] = prediction_before_relu[active] - batch[active]

                scale = self._alpha / batch.shape[1]
                coefficient_gradient = (
                    scale * neighbour_weights * (residual @ batch.T)
                    + self._lambda * coefficients
                )
                bias_gradient = scale * residual.sum(axis=1, keepdims=True)

                coefficients -= coefficient_gradient
                bias -= bias_gradient
                self._W_theta = neighbour_weights * coefficients

                movement = np.square(coefficient_gradient).sum() + np.square(bias_gradient).sum()
                progress.set_description(
                    f"Epoch {epoch + 1}/{self._max_epoch} update={movement:.3e}"
                )
            progress.close()

            epoch_loss = self._objective(targets)
            self._training_history.append(epoch_loss)
            print(f"Loss after epoch {epoch + 1}: {epoch_loss:0.8E}")

        self._W_theta = neighbour_weights * coefficients
        self._b = bias

    def predict(self, r: np.ndarray) -> np.ndarray:
        """Estimate missing term weights for one or more study vectors."""
        if self._W_theta is None or self._b is None:
            raise RuntimeError("Call train() before predict().")
        values = np.asarray(r)
        return self._relu(self._W_theta @ values + self._b)
