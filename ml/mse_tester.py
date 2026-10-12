"""Mean-squared reconstruction evaluation for hidden terms."""
from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm

from ml.abstract_model import AbstractLatentFactorModel
from ml.abstract_tester import AbstractTester


class MseTester(AbstractTester):
    """Report squared reconstruction error over all and relevant matrix entries."""

    def test(self, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        actual = np.asarray(r)
        partial = actual * self.choose_mask_indices(actual)
        estimated = self._model.predict(partial)

        squared_error = np.square(estimated - actual)
        relevant = (estimated > 0) | (actual > 0)
        return squared_error.reshape(-1), squared_error[relevant].reshape(-1)

    def __call__(self, docs: csc_matrix) -> tuple[float, float]:
        all_errors = []
        relevant_errors = []
        for column in tqdm(range(docs.shape[1])):
            record = docs[:, column].toarray()
            if np.any(record):
                every_entry, relevant_entry = self.test(record)
                all_errors.extend(every_entry.tolist())
                relevant_errors.extend(relevant_entry.tolist())

        results = (
            float(np.mean(all_errors)) if all_errors else 0.0,
            float(np.mean(relevant_errors)) if relevant_errors else 0.0,
        )
        self._model.mse_test = results
        return results
