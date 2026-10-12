"""Stream a sample of clinical-trial records from the Hugging Face dataset.

Use --rows 0 to write every source study with MeSH terms. The script streams
records and writes directly to CSV so it does not accumulate all studies in RAM.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CORE_FIELDS = ["nct_id", "mesh_terms", "source_url"]
TEXT_FIELDS = [
    "brief_title", "official_title", "brief_summary",
    "detailed_description", "eligibility_criteria", "conditions", "keywords",
]


def serialise(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def main():
    ap = argparse.ArgumentParser(description="Stream ClinicalTrials.gov-derived studies with MeSH terms.")
    ap.add_argument("--rows", type=int, default=5000,
                    help="Number of qualifying records to save; use 0 for all")
    ap.add_argument("--out", default="data/clinical_trials_5000.csv")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-buffer", type=int, default=10000,
                    help="Streaming shuffle buffer; set to 0 to preserve source order")
    ap.add_argument("--include-text", action="store_true",
                    help="Include titles, summaries, eligibility and other text fields in the CSV")
    args = ap.parse_args()
    fields = CORE_FIELDS + (TEXT_FIELDS if args.include_text else [])
    if args.rows < 0:
        ap.error("--rows must be zero (all rows) or a positive integer")

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dependencies first: pip install datasets pandas pyarrow") from exc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen = 0
    dataset = load_dataset(
        "louisbrulenaudet/clinical-trials",
        split=args.split,
        streaming=True,
    )
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    def annotated_terms(mesh):
        found = []
        if isinstance(mesh, dict):
            for group in ("conditions", "interventions"):
                for item in (mesh.get(group) or []):
                    if isinstance(item, dict) and item.get("term"):
                        term = str(item["term"]).strip()
                        if term:
                            found.append(term)
        return sorted(set(found))

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in dataset:
            seen += 1
            mesh = row.get("mesh_terms")
            terms = annotated_terms(mesh)
            if not terms:
                continue
            record = {key: serialise(row.get(key)) for key in fields}
            # Export only the annotation groups used to select eligible records.
            record["mesh_terms"] = serialise(terms)
            if row.get("nct_id"):
                record["source_url"] = "https://clinicaltrials.gov/study/" + str(row["nct_id"])
            writer.writerow(record)
            count += 1
            if count % 500 == 0:
                handle.flush()
                print(f"Scanned {seen:,} records; saved {count:,} with MeSH terms")
            if args.rows and count >= args.rows:
                break

    print(f"Saved {count:,} studies with MeSH terms to {out}")
    if args.rows == 0:
        print("Completed full streaming pass over the selected split.")


if __name__ == "__main__":
    main()
