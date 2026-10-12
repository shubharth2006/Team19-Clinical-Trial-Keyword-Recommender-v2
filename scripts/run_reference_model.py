"""Run the project's clinical-trial MeSH recommendation experiment.

The model and evaluation routines in ml/ implement the method described in the
cited research paper. This runner adapts that method to a public dataset whose
records already contain MeSH annotations.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csc_matrix, save_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_utils import load_csv
from ml.mini_batch_relu_latent_factor_model import MiniBatchReLuLatentFactorModel
from ml.mse_tester import MseTester
from ml.average_precision_k_tester import AveragePrecisionKTester


def build_utility_matrix(documents: list[list[str]], min_doc_freq: int = 5, max_terms: int = 134):
    """Build the reference-style term-by-document log TF-IDF utility matrix.

    The data source provides MeSH annotations instead of the original XML
    annotation pipeline. Terms are binary per study (one occurrence), so the
    log-TF component is log10(2). Document frequency determines log-IDF.
    """
    if not documents:
        raise ValueError("No documents were provided.")
    n_docs = len(documents)
    doc_freq: Counter[str] = Counter()
    for terms in documents:
        doc_freq.update(set(terms))

    candidates = [
        term for term, count in doc_freq.items()
        if count >= min_doc_freq and count < n_docs
    ]
    candidates.sort(key=lambda term: (-doc_freq[term], term))
    if max_terms > 0:
        candidates = candidates[:max_terms]
    if len(candidates) < 2:
        raise ValueError(
            f"Only {len(candidates)} terms survived filtering. "
            "Use more documents, lower --min-doc-freq, or increase --max-terms."
        )

    term_to_idx = {term: i for i, term in enumerate(candidates)}
    idf = {term: float(np.log10(n_docs / doc_freq[term])) for term in candidates}
    row_idx, col_idx, values = [], [], []
    for col, terms in enumerate(documents):
        for term in set(terms):
            idx = term_to_idx.get(term)
            if idx is None:
                continue
            weight = float(np.log10(2.0) * idf[term])
            if weight > 0:
                row_idx.append(idx)
                col_idx.append(col)
                values.append(weight)

    matrix = csc_matrix(
        (np.asarray(values, dtype=np.float32), (row_idx, col_idx)),
        shape=(len(candidates), n_docs),
        dtype=np.float32,
    )
    return matrix, candidates, doc_freq, idf


def topk_precision_for_masked_terms(train_R, eval_R, fraction, k, seed):
    """Popularity baseline on the same masked-term recovery task."""
    rng = np.random.default_rng(seed)
    popularity = np.asarray((train_R > 0).sum(axis=1)).ravel().astype(np.float64)
    per_doc = []
    for j in range(eval_R.shape[1]):
        r = eval_R[:, j].toarray().ravel()
        positives = np.flatnonzero(r > 0)
        if positives.size == 0:
            continue
        masked = r.copy()
        count = int(len(positives) * fraction)
        hidden = rng.choice(positives, size=min(count, len(positives)), replace=False)
        masked[hidden] = 0
        candidates = np.flatnonzero(masked == 0)
        if not len(candidates):
            continue
        ranking_scores = popularity.copy()
        ranking_scores[masked > 0] = -np.inf
        ranked = candidates[np.argsort(ranking_scores[candidates])[::-1]][:k]
        per_doc.append(float(np.sum(r[ranked] > 0) / max(len(ranked), 1)))
    return float(np.mean(per_doc)) if per_doc else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train the reference-aligned ReLU collaborative-filtering model.")
    parser.add_argument("--input", default=str(ROOT / "data" / "sample_trials.csv"))
    parser.add_argument("--max-documents", type=int, default=0, help="Optional cap; 0 uses all input rows")
    parser.add_argument("--min-doc-freq", type=int, default=1)
    parser.add_argument("--max-terms", type=int, default=134)
    parser.add_argument("--tau", type=float, default=-1e-3)
    parser.add_argument("--k-neighbor", type=float, default=1e3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-fraction", type=float, default=0.8)
    args = parser.parse_args()

    if not 0.0 < args.mask_fraction < 1.0:
        parser.error("--mask-fraction must be between 0 and 1")

    df = load_csv(args.input)
    if args.max_documents > 0 and len(df) > args.max_documents:
        df = df.sample(n=args.max_documents, random_state=args.seed).reset_index(drop=True)
    docs = df["mesh_terms_list"].tolist()

    U, vocabulary, doc_freq, idf = build_utility_matrix(
        docs, min_doc_freq=args.min_doc_freq, max_terms=args.max_terms
    )

    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(U.shape[1])
    n_train = int(0.8 * len(indices))
    n_val = int(0.1 * len(indices))
    if n_train < 1 or n_val < 1 or n_train + n_val >= len(indices):
        parser.error("Need at least 10 usable records for a non-empty 80:10:10 split.")
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_R = U[:, train_idx].tocsc()
    val_R = U[:, val_idx].tocsc()
    test_R = U[:, test_idx].tocsc()

    model = MiniBatchReLuLatentFactorModel(
        tau=args.tau,
        k=args.k_neighbor,
        alpha=args.alpha,
        lambda_=args.lambda_reg,
        batch_size=args.batch_size,
        max_epoch=args.epochs,
    )
    model.train(train_R)

    # Reproducible masking in the original-style metrics.
    np.random.seed(args.seed + 101)
    val_mse, val_mser = MseTester(model)(val_R)
    np.random.seed(args.seed + 102)
    val_apk, val_random = AveragePrecisionKTester(
        model, k=3, fraction_of_masking=args.mask_fraction
    )(val_R)

    np.random.seed(args.seed + 201)
    test_mse, test_mser = MseTester(model)(test_R)
    np.random.seed(args.seed + 202)
    test_apk, test_random = AveragePrecisionKTester(
        model, k=3, fraction_of_masking=args.mask_fraction
    )(test_R)

    # Additional popularity baseline using exactly the same held-out records.
    test_popularity = topk_precision_for_masked_terms(
        train_R, test_R, args.mask_fraction, 3, args.seed + 204
    )

    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    with (out_dir / "reference_model.pkl").open("wb") as handle:
        pickle.dump(model, handle)
    with (out_dir / "mesh_vocab.json").open("w", encoding="utf-8") as handle:
        json.dump({
            "vocabulary": vocabulary,
            "idf": idf,
            "document_frequency": {term: int(doc_freq[term]) for term in vocabulary},
            "num_documents": int(U.shape[1]),
            "matrix_shape": list(U.shape),
            "source_file": str(Path(args.input).name),
        }, handle, ensure_ascii=False, indent=2)
    save_npz(out_dir / "utility_matrix.npz", U)

    summary = {
        "input_file": str(Path(args.input).name),
        "documents_used": int(U.shape[1]),
        "vocabulary_size": int(U.shape[0]),
        "matrix_shape_terms_by_documents": list(U.shape),
        "split": {"train": int(train_R.shape[1]), "validation": int(val_R.shape[1]), "test": int(test_R.shape[1])},
        "parameters": {
            "tau": args.tau, "k": args.k_neighbor, "alpha": args.alpha,
            "lambda": args.lambda_reg, "epochs": args.epochs,
            "batch_size": args.batch_size, "seed": args.seed,
            "mask_fraction": args.mask_fraction,
        },
        "validation": {"mse": float(val_mse), "mser": float(val_mser), "apk3": float(val_apk), "random_apk3": float(val_random)},
        "test": {
            "mse": float(test_mse), "mser": float(test_mser),
            "apk3": float(test_apk), "random_apk3": float(test_random),
            "popularity_apk3": float(test_popularity),
        },
        "training_loss": [float(value) for value in model._training_history],
        "interpretation_note": "This dataset adapter consumes pre-assigned MeSH condition/intervention terms. Results are not directly comparable with the original XML text-matching corpus.",
    }
    with (results_dir / "reference_experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    pd.DataFrame([{
        "documents": summary["documents_used"],
        "terms": summary["vocabulary_size"],
        "train_docs": train_R.shape[1],
        "validation_docs": val_R.shape[1],
        "test_docs": test_R.shape[1],
        "validation_mse": val_mse,
        "validation_mser": val_mser,
        "validation_apk3": val_apk,
        "validation_random_apk3": val_random,
        "test_mse": test_mse,
        "test_mser": test_mser,
        "test_apk3": test_apk,
        "test_random_apk3": test_random,
        "test_popularity_apk3": test_popularity,
        "tau": args.tau, "k": args.k_neighbor, "alpha": args.alpha,
        "lambda_reg": args.lambda_reg, "epochs": args.epochs,
        "mask_fraction": args.mask_fraction, "seed": args.seed,
    }]).to_csv(results_dir / "reference_metrics.csv", index=False)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved model and matrix artifacts under: {out_dir}")
    print(f"Saved experiment summary under: {results_dir}")


if __name__ == "__main__":
    main()
