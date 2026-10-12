# Collaborative Filtering on Keywords Recommendation for Clinical Trial Records

**UE24CS352A - Machine Learning Mini-Project | Problem 14 | Team 19**  
Shubharth Upadhyay (PES2UG24CS499) and Suraj Acharya (PES2UG24CS539)

This project implements and evaluates a collaborative-filtering approach for recommending Medical Subject Headings (MeSH) for clinical-trial records. The model follows the method described in Xiao Zhou's Stanford CS229 project report; the underlying research method is not claimed as a new algorithm.

> Educational information-retrieval prototype only; not a clinical decision-support system.

## Implementation

The main experiment is `scripts/run_reference_model.py`. The model and evaluation logic are implemented in the `ml/` package.

- MeSH term-by-trial log-TF-IDF utility matrix
- Cosine-similarity-derived term-neighbour weights
- ReLU prediction with a trainable parameter matrix and bias
- Mini-batch gradient descent with L2 regularization
- Masked-term evaluation using Top-3 precision, MSE and MSER
- Random baseline from the evaluator and a separate popularity baseline
- Saved model/vocabulary artifacts for CLI inference and the Streamlit demo

Project-specific integration includes the dataset adapter, deterministic experiment runner, popularity baseline, saved outputs, CLI, Streamlit interface, automated checks and documentation. The implementation follows the published method rather than proposing a new algorithm. See `docs/REFERENCE_REPRODUCTION.md` for the implementation notes and dataset differences.

## Quick start: use the saved 55,634-record model

1. Create and activate a virtual environment.
2. Install dependencies using `pip install -r requirements.txt`.
3. Download the run artifact from [the completed GitHub Actions run](https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender/actions/runs/38137159221).
4. Extract `Team19_Clinical_Trial_CF_55634_Run_Artifacts.zip` into the repository root, preserving paths so that `output/` and `results/` are created.
5. Run CLI inference and launch the app:

       python scripts/reference_recommend.py --query "Neoplasms" --top-k 3
       streamlit run app/reference_app.py

Open the local Streamlit URL, typically http://localhost:8501.

## Reproduce the experiment

Recommended Windows PowerShell sequence:

    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    python scripts/smoke_test_reference.py
    python scripts/download_hf_sample.py --rows 55634 --out data/clinical_trials_55634.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_55634.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000 --seed 42
    python scripts/reference_recommend.py --query "Neoplasms" --top-k 3
    streamlit run app/reference_app.py

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current terminal, then activate again. Downloading the dataset requires internet access.

## Measured run

A 55,634-record experiment completed in GitHub Actions, including download, training, evaluation, CLI inference, Streamlit startup, and artifact upload.

- Split: 44,507 train / 5,563 validation / 5,564 test
- Vocabulary: 134 terms; seed: 42; epochs: 5; masked terms: 80%
- Test Top-3 precision: collaborative filtering 0.02792, popularity 0.01165, random 0.00472
- Test MSE/MSER: 0.00473
- Validation Top-3 precision: 0.02866; random 0.00664

Full run notes: `results/REFERENCE_RUN_55634.md`  
Metrics CSV: `results/reference_metrics_55634.csv`  
Experiment JSON: `results/reference_experiment_55634.json`  
Completed workflow: https://github.com/shubharth2006/Team19-Clinical-Trial-Keyword-Recommender/actions/runs/38137159221

These results are from one seed and the adapted dataset; they are not directly comparable with results from the original paper's data pipeline. Absolute Top-3 precision remains low.

## Dataset adaptation and limitations

Dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials

The current pipeline uses pre-assigned MeSH annotations rather than reproducing the original historical XML and MeSH-trie matching across six free-text fields. The utility-matrix vocabulary and IDF are computed before the train/validation/test split; a stricter evaluation should fit them on training records only. Query matching relies on exact controlled-vocabulary phrases, so the interface is not a general semantic search model.

The app demonstrates the pipeline, but recommendation relevance should be checked across several queries before claiming strong quality.

## Team handoff and TA preparation

- Start with `docs/PARTNER_HANDOFF_AND_TA_DEMO_GUIDE.md` for Windows setup, file map, equations, demonstration plan and viva questions.
- Read `docs/VIVA_PREP.md` and `docs/REFERENCE_REPRODUCTION.md`.
- Keep the private repository as the source of truth; add faculty and assigned TAs through GitHub repository access.
- Describe the work each teammate actually completed.

## References

1. Zhou, X. *Collaborative Filtering on Keywords Recommendation for Clinical Trial Records*, Stanford CS229 Project, 2020: https://cs229.stanford.edu/proj2020spr/report/Zhou.pdf
2. Dataset: https://huggingface.co/datasets/louisbrulenaudet/clinical-trials
3. ClinicalTrials.gov: https://clinicaltrials.gov/
