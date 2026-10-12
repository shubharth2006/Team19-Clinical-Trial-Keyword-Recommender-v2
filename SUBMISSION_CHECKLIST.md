# Team 19 - Problem 14 Submission Checklist

## Project structure
- [x] Private GitHub repository
- [x] README with setup and run instructions
- [x] Research paper cited and implementation documentation included
- [x] Model and evaluation modules implemented in the project repository
- [x] Reference-aligned ReLU collaborative-filtering model and masked MSE/AP@K evaluators
- [x] Dataset adapter, experiment runner, inference CLI and Streamlit demo
- [x] Offline smoke-test and end-to-end CLI test in GitHub Actions
- [x] 55,634-record real-data experiment completed; measured metrics recorded
- [x] Recommendation CLI inference and Streamlit startup passed after training
- [x] Two-page PDF and nine-slide presentation updated with measured results and adaptation disclosure

## Remaining before/at hand-in
- [x] Run the updated data workflow: training, recommendation CLI inference and Streamlit startup health check all completed successfully
- [x] Inspect the sample recommendation output and note its relevance limitation
- [ ] Run at least one additional seed if time permits
- [ ] Ensure both team members can explain the model equation, data adapter, metrics and limitations
- [ ] Add faculty and assigned TAs to the private repository
- [ ] Submit the PDF and presentation through the course submission channel

## Main commands

    pip install -r requirements.txt
    python scripts/smoke_test_reference.py
    python scripts/download_hf_sample.py --rows 5000 --out data/clinical_trials_5000.csv --seed 42
    python scripts/run_reference_model.py --input data/clinical_trials_5000.csv --min-doc-freq 5 --max-terms 134 --epochs 5 --batch-size 10000 --seed 42
    python scripts/reference_recommend.py --query "Neoplasms" --top-k 3
    streamlit run app/reference_app.py

## Measured experiment

See results/REFERENCE_RUN_55634.md. The completed 55,634-record run used 44,507/5,563/5,564 train/validation/test records and a 134-term MeSH vocabulary. Test Top-3 precision was 0.02792 for collaborative filtering, 0.01165 for popularity, and 0.00472 for random selection. This is one seeded result with an adapted MeSH-label schema, not directly comparable with results from the paper's original data pipeline.

## Dataset limitation

The current adapter uses pre-assigned MeSH condition/intervention labels rather than reproducing the original XML-to-MeSH-trie matching across all six text attributes. Keep that limitation visible in the write-up and presentation.
