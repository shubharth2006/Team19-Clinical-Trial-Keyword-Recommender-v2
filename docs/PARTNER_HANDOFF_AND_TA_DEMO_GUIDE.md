# Team 19 Partner Handoff and TA Demo Guide

**Course:** UE24CS352A Machine Learning Mini-Project  
**Problem 14:** Collaborative Filtering on Keywords Recommendation for Clinical Trial Records  
**Team 19:** Shubharth Upadhyay (PES2UG24CS499) and Suraj Acharya (PES2UG24CS539)

This guide is for both teammates. Use it together with README.md, docs/REFERENCE_REPRODUCTION.md, docs/VIVA_PREP.md, results/REFERENCE_RUN_55634.md, the report PDF, and the nine-slide deck.

## 1. The one-minute project explanation

Clinical-trial search can miss relevant records when a searcher does not enter all the medical subject headings used to index a study. This project takes a query containing recognizable MeSH terms and ranks additional candidate MeSH terms that may expand the search.

The method creates a term-by-trial TF-IDF utility matrix, computes relationships between terms from their trial co-occurrence patterns, and trains a ReLU predictor with mini-batch gradient descent and L2 regularization. Evaluation hides some known terms from held-out records and measures whether the model retrieves them in its top three. It is an information-retrieval prototype, not a medical decision system.

**Research basis and implementation:** this is a reproduction/adaptation of the method described in Xiao Zhou's Stanford CS229 paper, not a new algorithm. The model and evaluation modules in this repository are implemented for this project. Project-specific work also integrates the current dataset, creates the streaming adapter, runs the reproducible experiment, adds a popularity baseline, saves inference artifacts, exposes a CLI and Streamlit demo, and tests/documents the workflow. Describe each teammate's real contributions accurately.

## 2. Give your teammate access to the real source of truth

Repository: https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender

The repository is private. The owner should open GitHub repository Settings, add the teammate under Collaborators/Manage access, and have the teammate accept the invitation. Then the teammate should clone the repository. Send this guide, the report, presentation, and the trained-run artifact ZIP as supporting files, but keep the GitHub repository as the authoritative source code.

Important: avoid using any earlier ZIP of the compact prototype as the current source. The reference-aligned code is the version in the repository's main branch.

## 3. Fastest way to launch the already-trained demo

Use this when rehearsing with the TA and you do not want to download and train the full dataset during the demonstration.

1. Clone the private repository and enter its folder.
2. Download Team19_Clinical_Trial_CF_55634_Run_Artifacts.zip.
3. Extract it into the repository root while preserving the folder structure. It should create output/reference_model.pkl, output/mesh_vocab.json, results/reference_metrics.csv, and results/reference_experiment.json.
4. Create/activate a Python environment and install requirements (commands below).
5. In PowerShell, run:
   
       python scripts/reference_recommend.py --query "Neoplasms" --top-k 3
   
6. Then run:
   
       streamlit run app/reference_app.py
   
7. Open the local URL printed by Streamlit, usually http://localhost:8501. Try the default query and then an exact MeSH phrase such as Neoplasms. Press the recommendation button.

The 55,634-record workflow returned these sample CLI suggestions for “Neoplasms”: Breast Neoplasms, Syndrome, and Diabetes Mellitus. Only the first is clearly topical; this is useful for demonstrating both the working interface and a limitation. Results can differ if you retrain the model.

Artifact source run: https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender/actions/runs/38137159221

## 4. Re-run the full experiment on Windows PowerShell

This reproduces training, not just inference. It requires an internet connection and time to download the dataset.

From the repository folder:

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python scripts/smoke_test_reference.py

If PowerShell blocks activation, use this for the current terminal only:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
    .venv\Scripts\Activate.ps1

Download 55,634 eligible studies:

    python scripts/download_hf_sample.py --rows 55634 --out data/clinical_trials_55634.csv --seed 42

Train and evaluate:

    python scripts/run_reference_model.py --input data/clinical_trials_55634.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000 --seed 42

Inspect recommendations and launch the app:

    python scripts/reference_recommend.py --query "Neoplasms" --top-k 3
    streamlit run app/reference_app.py

Expected output locations:
- output/reference_model.pkl — trained Python model
- output/mesh_vocab.json — vocabulary, inverse document-frequency values, document frequencies, and corpus size
- output/utility_matrix.npz — term-by-trial utility matrix
- results/reference_metrics.csv — one-row metric table
- results/reference_experiment.json — parameters, split sizes, metrics, and training-loss history

If download fails, retry later and check the public Hugging Face dataset status. Do not commit the downloaded dataset or the binary model to the source repository.

## 5. End-to-end pipeline: understand every directory

### Data acquisition — scripts/download_hf_sample.py

Streams records from louisbrulenaudet/clinical-trials. It shuffles with seed 42 by default, selects rows with usable MeSH condition/intervention annotations, serializes nested fields to CSV, and writes incrementally instead of storing the full source dataset in RAM. Its default compact export contains nct_id, mesh_terms, and source_url; optional text fields can be included for inspection.

### Parsing — src/data_utils.py

Reads the CSV, parses the nested JSON/list-like MeSH field, extracts term strings, removes empty records, and returns a normalized term list for every trial. It also contains utility functions for vocabulary filtering and exact phrase matching.

### Matrix construction and experiment orchestration — scripts/run_reference_model.py

- Reads the parsed studies and counts how many studies contain each term.
- Keeps sufficiently frequent terms and caps the vocabulary at 134 for the reference-scale run.
- Builds a sparse term-by-trial utility matrix.
- Splits the records deterministically into 80% training, 10% validation, and 10% test using seed 42.
- Instantiates and trains the reference-aligned model.
- Evaluates the validation and test sets with masked-term reconstruction.
- Adds a popularity baseline and writes model, vocabulary, matrix, and metric artifacts.

### Reference model — ml/mini_batch_relu_latent_factor_model.py and ml/abstract_model.py

These project modules compute a cosine-similarity-derived term-neighbour matrix, learn a parameter matrix and bias, and predict non-negative utility scores with ReLU. The mathematical method follows the cited research paper; the code in this repository implements the model and evaluator.

### Evaluation — ml/abstract_tester.py, ml/mse_tester.py, ml/average_precision_k_tester.py

The tester hides a fraction of known terms in each held-out trial, runs prediction, and compares predictions with the original terms. MSE measures reconstruction error over entries; MSER measures error on the relevant entries as defined in the tester. The top-k tester calculates top-3 recovery and also produces a random baseline. The orchestration script adds a popularity ranking that uses term prevalence in the training set.

### Inference and interface — scripts/reference_recommend.py and app/reference_app.py

The CLI and Streamlit UI load the saved model and vocabulary. The query matcher looks for exact vocabulary phrases (with limited alias handling in the UI), builds a query vector on the model's TF-IDF scale, predicts term scores, removes already-matched terms, and returns the highest-scoring candidates. This is not a general-purpose semantic language model. It relies on the selected vocabulary and phrase matching.

### Reproducibility — .github/workflows/

The reference pipeline smoke-test workflow compiles code and tests model/CLI execution on a bundled sample. The data-run workflow downloads a selectable sample size, fits/evaluates the model, checks inference, checks Streamlit startup, and uploads artifacts. The 55,634-record data-run completed successfully.

## 6. Model math you should be able to explain

Let N be the number of studies, tf(i,j) the count of term i in study j, and df(i) the number of studies containing term i.

Utility matrix:

    U(i,j) = log10(1 + tf(i,j)) * log10(N / df(i))

For the current annotation dataset, terms are treated as present/absent per study, so tf is effectively 1 when the term appears. Terms appearing in every document have IDF 0 and are not useful for ranking; the runner excludes them.

The model normalizes term rows, calculates their pairwise cosine similarity, and builds neighbor weights:

    W(i,j) = exp(tau * (1 - cosine(U_i, U_j))^k), with W(i,i) = 0

Prediction:

    R_hat = ReLU((W elementwise-multiplied-by Theta) R + b)

Theta and b are trained by mini-batch gradient descent; L2 regularization is applied to Theta. ReLU sets negative predictions to zero. W encodes term-term relationships; R is the term-by-study matrix; b is a per-term bias.

Reference-scale configuration: tau = -0.001, k = 1000, alpha = 0.1, lambda = 0.01, batch size = 10,000, 5 epochs, seed = 42.

## 7. How to interpret the real 55,634-record result

The measured split was 44,507 training, 5,563 validation, and 5,564 test records, using 134 terms and masking 80% of positive terms for evaluation.

| Test metric | Result |
|---|---:|
| Collaborative-filtering Top-3 precision / reference AP@3 | 0.02792 |
| Popularity baseline Top-3 precision | 0.01165 |
| Random baseline Top-3 precision | 0.00472 |
| MSE | 0.00473 |
| MSER | 0.00473 |

Validation Top-3 precision was 0.02866; validation random was 0.00664. Training loss decreased from 0.19330 before updates to 0.19072 after five epochs.

Interpretation: the model beat both simple baselines in this one seeded run. The absolute Top-3 precision is still low. The data adapter uses pre-assigned MeSH annotations instead of rebuilding the original historical XML and MeSH-trie matching across six free-text attributes. Therefore, these metrics are not directly comparable with the original project's scores. One seed does not establish robust generalization.

A further methodological caveat is that this runner constructs the vocabulary and IDF weights before its train/validation/test split. For a stricter future experiment, vocabulary/IDF should be fitted on training data only and then applied unchanged to validation/test data. If asked about it, acknowledge it as a limitation rather than hiding it.

## 8. TA demonstration sequence (about 6 minutes)

### Before the TA arrives
- Verify that the repo is accessible and the app launches from the trained artifacts.
- Keep the report PDF, presentation, successful GitHub Actions run, and metric CSV open.
- Avoid waiting for the full dataset download during the review. Use the saved model for the live demo; show the completed workflow run to prove the training path was actually exercised.

### Recommended speaking order
1. Slides 1-2 (45 seconds): state the problem, input, and output.
2. Slides 3-4 (60 seconds): explain the research method, dataset source, and why the dataset adapter differs from the historic XML pipeline.
3. Slides 5-6 (90 seconds): explain utility matrix, term neighbor weights, ReLU prediction, masking, Top-3, MSE/MSER.
4. Slide 7 (30 seconds): point to the source files and the successful CI workflow.
5. Slide 8 (60 seconds): report the measured metrics accurately, include the baselines, and mention the low absolute precision and data limitation.
6. Slide 9 (30 seconds): conclusion and next steps.
7. Live app (60-90 seconds): enter a query, show matched terms and recommendations. Then point to results/REFERENCE_RUN_55634.md and the GitHub Actions run.

Never say the model diagnoses patients or that the suggestions are medically validated. It is a search-keyword prototype only.

## 9. Likely viva questions and defensible answers

**What problem does it solve?**  
It proposes additional MeSH terms from an incomplete clinical-trial query to help keyword-based retrieval.

**What is MeSH?**  
Medical Subject Headings, a controlled biomedical vocabulary used to describe medical concepts consistently.

**Why use TF-IDF?**  
A term that distinguishes some studies should carry more weight than a term appearing almost everywhere.

**What is collaborative filtering here?**  
It learns term relationships from their occurrence patterns across many trial records and uses those relationships to score other terms.

**Why cosine similarity?**  
It compares the direction of two term-occurrence vectors across studies, so co-occurrence patterns can be compared without their raw magnitude dominating.

**Why ReLU?**  
It clips negative prediction scores to zero, giving non-negative outputs for ranking/reconstruction.

**What do tau and k control?**  
They control the shape of the exponential transformation used to turn term cosine similarities into neighbor weights. alpha is the learning rate; lambda controls L2 regularization strength.

**Why mask terms?**  
Masking simulates an incomplete keyword set. The evaluator checks whether hidden terms can be recovered among the highest-ranked candidates.

**What does Top-3 precision mean?**  
It is the fraction of the returned top three candidates that were among the terms hidden from the original held-out record. The evaluator keeps a legacy AP@K class name for compatibility, but the calculation is precision-at-k rather than ranking-sensitive average precision.

**Why compare with popularity and random?**  
They show whether the model does better than recommending frequent terms or selecting candidates randomly.

**Is this identical to the original paper's experiment?**  
No. Our implementation follows the paper's method, but the data adapter consumes pre-assigned MeSH annotations rather than reproducing the historical XML and MeSH-trie matching pipeline. We disclose this difference in the report and repository.

**What are the limitations?**  
Low absolute Top-3 precision, one seed, limited vocabulary, exact phrase matching, different dataset semantics, and vocabulary/IDF calculated before the record split.

**What would you improve next?**  
Use train-only vocabulary/IDF, evaluate multiple seeds, inspect relevance on a set of queries, compare against additional baselines, and make query matching more robust.

**Which parts did the team implement?**  
The repository includes the model and evaluation modules, dataset adapter, experiment orchestration, CLI, interface, artifact workflow, automated checks, and documentation. The underlying method is based on the cited paper. Be specific about each teammate's actual contributions; do not invent individual work.

## 10. Two-person rehearsal and contribution record

Each teammate should be able to:
- run both the CLI and the Streamlit application;
- explain the model equation in plain language;
- find where the dataset is downloaded, parsed, split, trained, evaluated, and saved;
- reproduce the headline metrics from the recorded run;
- explain the implementation and dataset adaptation without hesitation;
- describe their own concrete contributions truthfully.

Before the review, write down who handled which real tasks (for example, data inspection, running the experiment, reviewing output, app integration, documentation, slides, or code walkthrough). Both teammates should still understand the whole pipeline because the evaluation may ask either person about any component.

## 11. Send this checklist to your teammate

- [ ] Accept private GitHub collaborator invitation.
- [ ] Clone the current repository; do not use the earlier prototype ZIP as the source of truth.
- [ ] Download the report, slides, this guide, and Team19_Clinical_Trial_CF_55634_Run_Artifacts.zip.
- [ ] Set up the venv and dependencies.
- [ ] Extract artifacts into repo root and run CLI inference.
- [ ] Launch Streamlit and rehearse one query.
- [ ] Read results/REFERENCE_RUN_55634.md and docs/REFERENCE_REPRODUCTION.md.
- [ ] Practise slides 3-8 and the limitations/attribution questions.
- [ ] Record actual individual contributions honestly.

## References

- Zhou, X. Collaborative Filtering on Keywords Recommendation for Clinical Trial Records, Stanford CS229 Project, 2020: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf
- Dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials
- Team repository: https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender
- Completed reference-scale run: https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender/actions/runs/38137159221
