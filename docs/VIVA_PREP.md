# Viva preparation: Problem 14

## One-sentence explanation

The project recommends additional controlled medical subject headings (MeSH terms) for a clinical-trial search by learning co-occurrence structure between medical terms across trial records.

## Pipeline

1. Read clinical-trial records and parse their MeSH terms.
2. Count term occurrences by study and filter rare terms to bound vocabulary size.
3. Build a log-scaled TF-IDF utility matrix with terms as rows and trials as columns.
4. Split the data into train, validation and test records.
5. Calculate cosine similarity between term rows and form a term-neighbor matrix W.
6. Fit a ReLU prediction layer with L2 regularization using mini-batch gradient descent.
7. Hide a portion of known terms in held-out studies and rank candidate terms.
8. Compare Top-3 recovery against popularity and random baselines.

## Likely questions

### Why MeSH?

MeSH is a controlled biomedical vocabulary. It reduces variation caused by using different free-text phrases for the same concept.

### Why TF-IDF?

The term-document representation gives more weight to terms that distinguish a study and less weight to terms that appear in many studies.

### Why collaborative filtering?

The model learns co-occurrence relationships across studies, so a term can receive a high score because it is related to the terms already present in the query.

### Why mask terms?

For evaluation, masking simulates an incomplete keyword set. The model is tested on whether it ranks the hidden known terms highly.

### Why use baselines?

A popularity baseline tests whether the model does better than simply recommending frequent terms. A random baseline gives a simple reference point.

### What is the main limitation?

The adapter uses pre-assigned condition/intervention MeSH terms rather than recreating the original XML and MeSH-trie matching across six text fields. The query matcher relies on exact phrases, Top-3 precision is low in absolute terms, and the results come from one seed. The vocabulary and IDF values are also calculated before the train/validation/test split, so a stricter evaluation should fit them using training records only.

### What would improve it next?

Evaluate additional random seeds and a wider range of queries; fit the vocabulary and IDF on training records only; inspect recommendations for topical relevance; and compare with additional retrieval baselines.
