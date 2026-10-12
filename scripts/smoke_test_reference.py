"""Offline smoke test for the reference model (no network/data download required)."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.sparse import csc_matrix

from scripts.run_reference_model import build_utility_matrix
from ml.mini_batch_relu_latent_factor_model import MiniBatchReLuLatentFactorModel
from ml.mse_tester import MseTester
from ml.average_precision_k_tester import AveragePrecisionKTester


def main():
    documents = [
        ["Arthritis, Rheumatoid", "Methotrexate", "Autoimmune Diseases"],
        ["Arthritis, Rheumatoid", "Autoimmune Diseases"],
        ["Lung Neoplasms", "Chemotherapy"],
        ["Lung Neoplasms", "Neoplasms", "Chemotherapy"],
        ["Arthritis, Rheumatoid", "Methotrexate"],
        ["Lung Neoplasms", "Neoplasms"],
        ["Autoimmune Diseases", "Methotrexate"],
        ["Lung Neoplasms", "Chemotherapy"],
        ["Arthritis, Rheumatoid", "Autoimmune Diseases"],
        ["Lung Neoplasms", "Neoplasms"],
    ]
    matrix, vocab, _, _ = build_utility_matrix(documents, min_doc_freq=1, max_terms=8)
    assert matrix.shape == (len(vocab), len(documents))
    assert matrix.nnz > 0

    train = csc_matrix(matrix[:, :8])
    validation = csc_matrix(matrix[:, 8:9])
    test = csc_matrix(matrix[:, 9:])
    model = MiniBatchReLuLatentFactorModel(
        tau=-0.001, k=1000, alpha=0.1, lambda_=0.01,
        batch_size=4, max_epoch=2,
    )
    model.train(train)

    pred = model.predict(test.toarray())
    assert pred.shape == test.shape
    assert np.all(np.isfinite(pred))
    assert np.all(pred >= 0)

    np.random.seed(42)
    mse, mser = MseTester(model)(validation)
    np.random.seed(43)
    apk, random_apk = AveragePrecisionKTester(
        model, k=3, fraction_of_masking=0.8
    )(validation)
    assert all(np.isfinite(x) for x in (mse, mser, apk, random_apk))
    print("Reference-model smoke test passed.")
    print(f"matrix={matrix.shape}, vocabulary={len(vocab)}, nnz={matrix.nnz}")
    print(f"validation MSE={mse:.6f}; MSER={mser:.6f}; AP@3={apk:.4f}; random AP@3={random_apk:.4f}")


if __name__ == "__main__":
    main()
