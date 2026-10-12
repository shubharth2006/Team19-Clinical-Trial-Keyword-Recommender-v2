# Reference-aligned experiment: 55,634 studies

## Execution

- Run: [GitHub Actions - 55,634-record reproduction](https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender/actions/runs/38137159221)
- Dataset: 55,634 shuffled-stream records; each saved record contains only pre-assigned condition/intervention MeSH terms
- Vocabulary: 134 terms
- Split: 44,507 train / 5,563 validation / 5,564 test
- Parameters: tau=-0.001, k=1000, alpha=0.1, lambda=0.01, 5 epochs, seed=42
- Masking: 80% of positive terms in held-out records
- Verification: downloader, model training, evaluation, saved-model inference, and Streamlit startup health check all passed.

## Held-out results

| Metric | Collaborative filtering | Popularity baseline | Random baseline |
|---|---:|---:|---:|
| Top-3 precision (reference AP@K tester) | 0.02792 | 0.01165 | 0.00472 |
| MSE | 0.00473 | - | - |
| MSER | 0.00473 | - | - |

Validation Top-3 precision was 0.02866, compared with random 0.00664. Validation MSE/MSER was 0.00481. Training loss decreased from 0.19330 before updates to 0.19072 after five epochs.

For this one seed, the top-3 precision is about 2.40 times the popularity baseline and 5.92 times the random baseline. These are descriptive ratios for this run only. The absolute value of the metric is still low.

## Recommendation smoke test

The command-line interface loaded the saved model and produced these suggestions for the broad query "Neoplasms":

1. Breast Neoplasms (0.010830)
2. Syndrome (0.010607)
3. Diabetes Mellitus (0.010438)

The first recommendation is topically related; the other two are weaker matches. This verifies the inference path but also shows why qualitative relevance checks across several queries are still required.

## Scope and limitations

This experiment implements the assigned collaborative-filtering method described in the cited research paper. Our adapter uses pre-assigned MeSH annotations for conditions/interventions from the current Hugging Face dataset, rather than rebuilding the original historical XML and MeSH-trie matching across six free-text fields. Therefore, the data pipeline is not identical and these scores must not be presented as directly comparable with the original repository's scores.

The run used one random seed and five epochs. It supports that the training/evaluation/demo path runs and that the model beat simple ranking baselines on this run; it does not establish robust performance across data samples or query types.


The downloader now exports only the condition and intervention labels used to qualify each record, keeping the data filter and model input consistent.
