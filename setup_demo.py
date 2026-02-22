#!/usr/bin/env python3
"""
Setup script for AMR-Guard Demo
Initializes the database and imports all data.

Drug interactions CSV is sourced from the Kaggle dataset:
  https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions

On Kaggle: attach the dataset via "Add data" — it will be mounted automatically.
Locally:   place ~/.kaggle/kaggle.json or run:
             kaggle datasets download -d mghobashy/drug-drug-interactions --unzip -p docs/drug_safety/
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    print("=" * 60)
    print("AMR-Guard Demo Setup")
    print("AMR-Guard: Infection Lifecycle Orchestrator")
    print("=" * 60)
    print()

    # Step 1: Import structured data into SQLite
    print("Step 1: Importing structured data into SQLite...")
    print("-" * 40)

    from src.db.import_data import import_all_data

    # Limit interactions to 50k for faster demo setup
    structured_results = import_all_data(interactions_limit=50000)

    # Step 2: Import docs into ChromaDB
    print("\nStep 2: Importing documents into ChromaDB (Vector Store)...")
    print("-" * 40)

    from src.db.vector_store import import_all_vectors

    vector_results = import_all_vectors()

    # Summary
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\nData imported:")
    print(f"  - EML Antibiotics:    {structured_results.get('eml_antibiotics', 0):>6} records")
    print(f"  - ATLAS Susceptibility:{structured_results.get('atlas_susceptibility', 0):>5} records")
    print(f"  - MIC Breakpoints:    {structured_results.get('mic_breakpoints', 0):>6} records")
    print(f"  - Drug Interactions:  {structured_results.get('drug_interactions', 0):>6} records")
    print(f"  - IDSA Guidelines:    {vector_results.get('idsa_guidelines', 0):>6} chunks")
    print(f"  - MIC Reference:      {vector_results.get('mic_reference', 0):>6} chunks")

    if structured_results.get('drug_interactions', 0) == 0:
        print()
        print("  ⚠  Drug interactions were not imported.")
        print("     Dataset: https://www.kaggle.com/datasets/mghobashy/drug-drug-interactions")
        print("     • On Kaggle: add the dataset via the notebook UI ('Add data').")
        print("     • Locally:   kaggle datasets download -d mghobashy/drug-drug-interactions \\")
        print("                    --unzip -p docs/drug_safety/")

    print()
    print("To run the app:  uv run streamlit run app.py")
    print()


if __name__ == "__main__":
    main()
