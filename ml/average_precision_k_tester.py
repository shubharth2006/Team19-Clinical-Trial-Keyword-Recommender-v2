"""Top-k recovery evaluation for hidden MeSH terms."""
from __future__ import annotations

import heapq

import numpy as np
from scipy.sparse import csc_matrix
from tqdm.auto import tqdm

from ml.abstract_model import AbstractLatentFactorModel
from ml.abstract_tester import AbstractTester


class AveragePrecisionKTester(AbstractTester):
    """Measure top-k precision while keeping the legacy class name for the CLI."""

    def __init__(
        self,
        model: AbstractLatentFactorModel,
        k: int = 3,
        fraction_of_masking: float = 0.8,
    ):
        super().__init__(model, fraction_of_masking)
        if k < 1:
            raise ValueError("k must be positive.")
        self._k = int(k)

    def test(self, r: np.ndarray) -> tuple[float, float]:
        original = np.asarray(r)
        masked = original * self.choose_mask_indices(original)
        candidate_mask = masked == 0
        scores = self._model.predict(masked)

        candidate_ids = np.flatnonzero(
            candidate_mask.reshape(-1) & (scores.reshape(-1) > 0)
        )
        ranked_pairs = [(int(i), float(scores.reshape(-1)[i])) for i in candidate_ids]
        best = heapq.nlargest(self._k, ranked_pairs, key=lambda pair: pair[1])
        if not best:
            return 0.0, 0.0

        original_flat = original.reshape(-1)
        hits = sum(original_flat[index] != 0 for index, _ in best)
        model_precision = float(hits / len(best))

        eligible_ids = np.flatnonzero(candidate_mask.reshape(-1))
        random_ids = np.random.choice(eligible_ids, size=len(best))
        random_precision = float(np.count_nonzero(original_flat[random_ids]) / len(random_ids))
        return model_precision, random_precision

    def __call__(self, docs: csc_matrix) -> tuple[float, float]:
        model_scores = []
        random_scores = []
        for column in tqdm(range(docs.shape[1])):
            record = docs[:, column].toarray()
            if np.any(record):
                model_score, random_score = self.test(record)
                model_scores.append(model_score)
                random_scores.append(random_score)

        means = (
            float(np.mean(model_scores)) if model_scores else 0.0,
            float(np.mean(random_scores)) if random_scores else 0.0,
        )
        self._model.apk_test = means
        return means
