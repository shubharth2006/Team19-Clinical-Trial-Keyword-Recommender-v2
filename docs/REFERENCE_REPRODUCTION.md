# Method implementation and reproduction notes

## Research basis

The project implements the collaborative-filtering method described by Xiao Zhou in *Collaborative Filtering on Keywords Recommendation for Clinical Trial Records* (Stanford CS229 Project, 2020): https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf.

The report is the source for the research method and mathematical framing. This repository contains its own model and evaluation implementation; it does not claim to introduce a new collaborative-filtering algorithm.

## Current data pipeline

The original research describes processing historical ClinicalTrials.gov XML and using a MeSH trie to recognize terms in selected text sections. For easier reproducibility, the current project consumes pre-assigned MeSH condition/intervention annotations from the public Hugging Face clinical-trials dataset. This avoids requiring the historical XML archive and annual MeSH descriptor files.

This is not identical in data semantics: the current vocabulary is built from pre-assigned condition/intervention MeSH annotations, rather than every MeSH synonym matched against all six original text attributes. The metrics are results for this adapted dataset and must not be compared numerically as though they were produced by the original corpus.

## Implementation overview

- `ml/abstract_model.py` defines the model interface and computes term-neighbour weights from cosine similarity.
- `ml/mini_batch_relu_latent_factor_model.py` trains the ReLU prediction layer using mini-batch gradient descent and L2 regularization.
- `ml/abstract_tester.py`, `ml/mse_tester.py`, and `ml/average_precision_k_tester.py` implement masked-term evaluation.
- `scripts/run_reference_model.py` builds the utility matrix, creates the data split, runs evaluation, adds a popularity baseline, and saves artifacts.
- `scripts/reference_recommend.py` and `app/reference_app.py` provide CLI and Streamlit inference.

## Reproducibility choices and limitations

- A fixed random seed controls the document split and evaluation masking.
- Saved model and vocabulary artifacts make the demo repeatable.
- Record and vocabulary caps keep the experiment tractable on a student laptop.
- A popularity baseline supplements the random baseline.
- Vocabulary and IDF are currently calculated before the train/validation/test split. A stricter evaluation should fit them on training data only.
- Query matching is based on exact controlled-vocabulary phrases, not general semantic understanding.
- Results are from a single seeded run; evaluate additional seeds and queries before making broader performance claims.
