"""Recommend MeSH keywords from a free-text query using the trained reference model."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Import the model module before unpickling; Python pickles classes by module path.
from ml.mini_batch_relu_latent_factor_model import MiniBatchReLuLatentFactorModel  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="Clinical trial query text")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--model", default=str(ROOT / "output" / "reference_model.pkl"))
    parser.add_argument("--vocab", default=str(ROOT / "output" / "mesh_vocab.json"))
    args = parser.parse_args()

    model_path, vocab_path = Path(args.model), Path(args.vocab)
    if not model_path.exists() or not vocab_path.exists():
        raise SystemExit("Train the reference model first: python scripts/run_reference_model.py")
    with model_path.open("rb") as handle:
        model = pickle.load(handle)
    with vocab_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)

    vocab = meta["vocabulary"]
    idf = meta["idf"]
    n_docs = int(meta["num_documents"])
    query_lower = re.sub(r"\s+", " ", args.query.lower()).strip()
    matched = []
    for term in sorted(vocab, key=lambda term: (-len(term), term)):
        if re.search(r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)", query_lower):
            matched.append(term)

    vector = np.zeros((len(vocab), 1), dtype=np.float32)
    for term in matched:
        idx = vocab.index(term)
        vector[idx, 0] = np.log10(2.0) * float(idf[term])

    scores = model.predict(vector).ravel()
    for term in matched:
        scores[vocab.index(term)] = -np.inf
    order = np.argsort(scores)[::-1]
    recs = [(vocab[i], float(scores[i])) for i in order if np.isfinite(scores[i])][:args.top_k]

    print("Query:", args.query)
    print("Detected MeSH terms:", ", ".join(matched) if matched else "(none)")
    print("Recommended terms:")
    if not recs:
        print("  No terms scored above zero. Try a query containing one or more exact MeSH phrases.")
    for rank, (term, score) in enumerate(recs, 1):
        print(f"  {rank}. {term} (score={score:.6f})")


if __name__ == "__main__":
    main()
