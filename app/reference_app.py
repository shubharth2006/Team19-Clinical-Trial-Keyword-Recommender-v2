from __future__ import annotations

import json
import pickle
from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "output" / "reference_model.pkl"
VOCAB_PATH = ROOT / "output" / "mesh_vocab.json"

st.set_page_config(page_title="Clinical Trial Keyword Recommendation", page_icon="🧬", layout="wide")
st.title("Clinical Trial MeSH Keyword Recommendation")
st.caption("Reference-aligned collaborative-filtering reproduction · Problem 14 · Team 19")


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        return None
    with MODEL_PATH.open("rb") as handle:
        model = pickle.load(handle)
    with VOCAB_PATH.open(encoding="utf-8") as handle:
        vocab = json.load(handle)
    return model, vocab


def aliases_for_term(term: str) -> set[str]:
    term = term.lower().strip()
    aliases = {term}
    if "," in term:
        left, right = term.split(",", 1)
        aliases.add((right.strip() + " " + left.strip()).lower())
    return aliases


artifacts = load_artifacts()
if artifacts is None:
    st.warning("The trained reference model is not available yet.")
    st.code(
        "python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv\n"
        "python scripts/run_reference_model.py --input data/clinical_trials_5000.csv --min-doc-freq 5 --max-terms 134"
    )
    st.stop()

model, meta = artifacts
vocab = meta["vocabulary"]
idf = meta["idf"]
n_docs = int(meta["num_documents"])

st.write(f"**Training corpus:** {n_docs:,} studies · **Vocabulary:** {len(vocab)} MeSH terms")
query = st.text_area(
    "Clinical-trial query",
    "rheumatoid arthritis methotrexate autoimmune diseases",
    height=110,
)
top_k = st.slider("Number of recommendations", min_value=3, max_value=10, value=3)

if st.button("Recommend MeSH terms", type="primary"):
    q = re.sub(r"\s+", " ", query.lower()).strip()
    matched = []
    for term in sorted(vocab, key=lambda item: (-len(item), item)):
        for alias in aliases_for_term(term):
            if re.search(r"(?<!\w)" + re.escape(alias) + r"(?!\w)", q):
                matched.append(term)
                break
    matched = list(dict.fromkeys(matched))

    vector = np.zeros((len(vocab), 1), dtype=np.float32)
    for term in matched:
        vector[vocab.index(term), 0] = np.log10(2.0) * float(idf[term])

    scores = model.predict(vector).ravel()
    for term in matched:
        scores[vocab.index(term)] = -np.inf

    ranked = [(vocab[i], float(scores[i])) for i in np.argsort(scores)[::-1]
              if np.isfinite(scores[i])][:top_k]

    left, right = st.columns([1, 1.3])
    with left:
        st.subheader("Detected MeSH terms")
        if matched:
            for term in matched:
                st.write(f"- {term}")
        else:
            st.info("No exact MeSH phrases matched. Try the comma-reversed MeSH form, a diagnosis, or an intervention.")
    with right:
        st.subheader("Recommended terms")
        if ranked:
            st.dataframe(
                pd.DataFrame(
                    [{"Rank": i, "Recommended MeSH term": term, "Model score": round(score, 6)}
                     for i, (term, score) in enumerate(ranked, 1)]
                ),
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No recommendation was produced.")

st.caption("Educational information-retrieval prototype only. This is not clinical advice.")
