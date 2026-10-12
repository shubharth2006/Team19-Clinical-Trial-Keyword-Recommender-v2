from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_terms(value) -> list[str]:
    """Parse list/JSON/list-like/text fields of MeSH terms into clean strings."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            text = str(item.get("term", "") if isinstance(item, dict) else item).strip()
            if text:
                result.append(text)
        return sorted(set(result))
    if isinstance(value, dict):
        terms: list[str] = []
        def walk(node):
            if isinstance(node, dict):
                if node.get("term"):
                    terms.append(str(node["term"]).strip())
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
        walk(value)
        return sorted({term for term in terms if term})

    text = str(value).strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(text)
            if isinstance(obj, (list, dict)):
                return parse_terms(obj)
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            pass
    return sorted({x.strip() for x in re.split(r"\s*[|;]\s*", text) if x.strip()})


def load_csv(path: str | Path) -> pd.DataFrame:
    # Only read columns used by the matrix builder. Clinical-trial text can be
    # large, especially when running on hundreds of thousands of records.
    columns = pd.read_csv(path, nrows=0).columns
    usecols = [name for name in ("nct_id", "mesh_terms") if name in columns]
    df = pd.read_csv(path, usecols=usecols)
    if "mesh_terms" not in df.columns:
        raise ValueError(
            "Dataset must contain a 'mesh_terms' column. "
            f"Available columns: {', '.join(map(str, df.columns))}"
        )
    df["mesh_terms_list"] = df["mesh_terms"].apply(parse_terms)
    df = df[df["mesh_terms_list"].map(len) > 0].reset_index(drop=True)
    if df.empty:
        raise ValueError("No records with non-empty MeSH terms were found.")
    return df


def filter_vocabulary(
    df: pd.DataFrame,
    min_doc_freq: int = 1,
    max_terms: int | None = 250,
) -> tuple[pd.DataFrame, list[str]]:
    """Keep frequent MeSH terms to keep the term-by-document model tractable."""
    counts: dict[str, int] = {}
    for terms in df["mesh_terms_list"]:
        for term in set(terms):
            counts[term] = counts.get(term, 0) + 1
    vocab = [term for term, count in counts.items() if count >= min_doc_freq]
    vocab.sort(key=lambda term: (-counts[term], term))
    if max_terms is not None and max_terms > 0:
        vocab = vocab[:max_terms]
    keep = set(vocab)
    out = df.copy()
    out["mesh_terms_list"] = out["mesh_terms_list"].apply(
        lambda terms: [term for term in terms if term in keep]
    )
    out = out[out["mesh_terms_list"].map(len) > 0].reset_index(drop=True)
    return out, vocab


def extract_terms_from_query(query: str, vocabulary: list[str]) -> list[str]:
    """Exact-match controlled MeSH phrases, preferring longer terms."""
    q = re.sub(r"\s+", " ", query.lower()).strip()
    matches = []
    for term in sorted(vocabulary, key=lambda x: (-len(x), x)):
        pattern = r"(?<!\w)" + re.escape(term.lower()) + r"(?!\w)"
        if re.search(pattern, q):
            matches.append(term)
    return matches


def query_vector_from_terms(
    terms: list[str],
    vocabulary: list[str],
    n_docs: int,
    doc_freq: np.ndarray | None = None,
) -> np.ndarray:
    """Build a single query vector on the corpus TF-IDF scale."""
    vector = np.zeros(len(vocabulary), dtype=np.float32)
    if doc_freq is None:
        return vector
    term_to_idx = {term: idx for idx, term in enumerate(vocabulary)}
    for term in terms:
        idx = term_to_idx.get(term)
        if idx is not None:
            df = max(float(doc_freq[idx]), 1.0)
            vector[idx] = np.log10(2.0) * np.log10(max(n_docs, 1) / df)
    return vector
