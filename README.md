---
title: AMR-Guard
emoji: ⚕️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.45.0"
app_file: app.py
pinned: true
license: apache-2.0
tags:
  - healthcare
  - medical
  - antimicrobial-resistance
  - clinical-decision-support
  - streamlit
  - llm
  - medgemma
short_description: Multi-agent clinical decision support for antimicrobial stewardship
---

# AMR-Guard: Infection Lifecycle Orchestrator

A multi-agent clinical decision-support system for antimicrobial stewardship, submitted to the **[MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)**.

Powered by **MedGemma** (4B multimodal + 27B text) and **TxGemma** — HAI-DEF models from Google.

---

## What it does

AMR-Guard guides clinicians through two stages of infection management with a **dynamic, site-aware patient form** that adapts its fields based on the selected infection site.

**Stage 1 — Empirical** (no lab results yet)
Patient history → risk factor analysis → empirical antibiotic recommendation

**Stage 2 — Targeted** (lab results available)
Lab report upload (PDF / image, any language) → pathogen & MIC extraction → resistance trend analysis → targeted prescription with drug interaction screening

A unique capability is **MIC creep detection**: the system flags when a pathogen's Minimum Inhibitory Concentration has risen ≥4-fold across admissions — even while the lab still reports "Susceptible" — giving clinicians a 6–18 month early warning before formal treatment failure.

---

## Key Features

| Feature | Details |
|---------|---------|
| **Dynamic form** | Fields adapt to infection site (urinary, respiratory, bloodstream, skin, intra-abdominal, CNS) |
| **Contextual suspected source** | Dropdown options change based on infection site (e.g. CAP / HAP / VAP for respiratory) |
| **Conditional creatinine** | Shown prominently for systemic infections; optional toggle for skin / intra-abdominal |
| **Lab file upload** | Upload PDF or image (PNG/JPG/TIFF) — PDF text extracted via pypdf; images sent to MedGemma vision |
| **MIC creep detection** | ≥4-fold MIC rise flagged before clinical resistance develops |
| **WHO AWaRe stewardship** | ACCESS → WATCH → RESERVE prescribing hierarchy enforced |
| **Drug interaction screening** | 191 K+ interactions from DDInter 2.0 |
| **Renal dose adjustment** | Cockcroft-Gault CrCl → 5-tier dose adjustment |

---

## Agent Pipeline

```
Patient form ──► Agent 1: Intake Historian  ──► (no lab) ──────────────────────────────► Agent 4: Clinical Pharmacologist ──► Prescription
                       │                                                                              ▲
                       └──► (lab uploaded) ──► Agent 2: Vision Specialist ──► Agent 3: Trend Analyst ──┘
```

| # | Agent | Model | Role |
|---|-------|-------|------|
| 1 | Intake Historian | MedGemma 4B IT | Parse EHR notes, calculate CrCl (Cockcroft-Gault), identify MDR risk factors |
| 2 | Vision Specialist | MedGemma 4B IT (multimodal) | Extract pathogen names + MIC values from lab images / PDFs in **any language** |
| 3 | Trend Analyst | MedGemma 27B Text IT | Detect MIC creep, compute resistance velocity against EUCAST v16.0 breakpoints |
| 4 | Clinical Pharmacologist | MedGemma 4B IT + TxGemma 9B | Select antibiotic + dose, apply WHO AWaRe stewardship, screen drug interactions |

**Orchestration:** LangGraph state machine with conditional routing
**Knowledge base:** SQLite (EUCAST breakpoints, WHO AWaRe, ATLAS surveillance, DDInter interactions) + ChromaDB (IDSA guidelines, WHO GLASS — semantic RAG)

---

## Hugging Face Spaces Deployment

> **Recommended deployment target.** Provides a persistent URL, native Streamlit support, GPU access, and multi-user access out of the box.

### Requirements
- A HF Space with **GPU hardware** (T4 for MedGemma 4B; A10G or better for MedGemma 27B)
- HF access granted to [MedGemma](https://huggingface.co/google/medgemma-4b-it) and [TxGemma](https://huggingface.co/google/txgemma-2b-predict)

### Steps

**1. Create a new Space**

Go to [huggingface.co/new-space](https://huggingface.co/new-space) and select:
- SDK: **Streamlit**
- Hardware: **T4 (GPU)** (free tier, limited quota) or **A10G**

**2. Push this repository**

```bash
git remote set-url space https://huggingface.co/spaces/<your-username>/amr-guard 2>/dev/null || git remote add space https://huggingface.co/spaces/<your-username>/amr-guard
git push space master
```

**3. Add Space Secrets**

In your Space → Settings → Variables and Secrets, add:

| Secret name | Value | Notes |
|-------------|-------|-------|
| `MEDIC_LOCAL_MEDGEMMA_4B_MODEL` | `google/medgemma-4b-it` | Required |
| `MEDIC_LOCAL_MEDGEMMA_27B_MODEL` | `google/medgemma-4b-it` | Use 4B fallback on T4 |
| `MEDIC_LOCAL_TXGEMMA_9B_MODEL` | `google/txgemma-2b-predict` | Required |
| `MEDIC_LOCAL_TXGEMMA_2B_MODEL` | `google/txgemma-2b-predict` | Required |
| `MEDIC_QUANTIZATION` | `4bit` | Required |
| `MEDIC_ENV` | `production` | Required |
| `HF_TOKEN` | Your HF access token | Required (gated models) |

**4. First boot — knowledge base initialisation**

`app.py` detects the HF Spaces environment (`SPACE_ID` env var) and automatically runs `setup_demo.py` on first boot to build the SQLite + ChromaDB knowledge base. This takes ~2–5 minutes once and requires no manual steps.

> Enable **Persistent Storage** (Space Settings → Persistent Storage) so the knowledge base survives restarts. Without it, setup runs on every cold boot (~2 min overhead).

**5. Drug interaction dataset (optional)**

To enable full drug interaction screening, place `db_drug_interactions.csv` in `docs/drug_safety/` before pushing, or after deployment open the Space terminal and run:

```bash
kaggle datasets download -d mghobashy/drug-drug-interactions --unzip -p docs/drug_safety/
python setup_demo.py
```

---

## Local Setup

### Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- HuggingFace account with access to MedGemma and TxGemma

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Minimum required settings in `.env`:

```bash
MEDIC_LOCAL_MEDGEMMA_4B_MODEL=google/medgemma-4b-it
MEDIC_LOCAL_MEDGEMMA_27B_MODEL=google/medgemma-4b-it   # 4B fallback if < 24 GB VRAM
MEDIC_LOCAL_TXGEMMA_9B_MODEL=google/txgemma-2b-predict
MEDIC_LOCAL_TXGEMMA_2B_MODEL=google/txgemma-2b-predict
MEDIC_QUANTIZATION=4bit
```

### 3. Authenticate with HuggingFace

```bash
uv run huggingface-cli login
```

### 4. Build the knowledge base

```bash
uv run python setup_demo.py
```

Ingests EUCAST breakpoints, WHO AWaRe classification, IDSA guidelines, ATLAS surveillance data, and DDInter drug interactions into SQLite + ChromaDB. Source files are in `docs/` — generated database is written to `data/` (gitignored).

### 5. Run the app

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Kaggle Reproduction

The full pipeline can also be reproduced on a free Kaggle T4 GPU (16 GB VRAM):

1. Open [`notebooks/kaggle_medic_demo.ipynb`](notebooks/kaggle_medic_demo.ipynb) in Kaggle
2. Add the `mghobashy/drug-drug-interactions` dataset to the notebook
3. Add your HuggingFace token as a Kaggle secret named `HF_TOKEN`
4. Run all cells — the notebook clones this repo, installs dependencies, builds the knowledge base, and launches the app via a public tunnel

Models run with 4-bit quantization on T4 (MedGemma 4B + TxGemma 2B).

---

## Dynamic Form — Field Reference

The Patient Analysis form adapts based on the selected **Primary infection site**.

| Infection site | Site-specific fields | Creatinine |
|---|---|---|
| **Urinary** | Catheter status, urinary symptoms, urine appearance | Always shown |
| **Respiratory** | O₂ saturation, ventilation status, cough type, sputum character | Always shown |
| **Bloodstream** | Central line, temperature, heart rate, respiratory rate, WBC, lactate, shock status | Always shown |
| **Skin** | Wound type, cellulitis extent, abscess, foreign body | Optional (renal flag) |
| **Intra-abdominal** | Pain location, peritonitis signs, perforation suspected, ascites | Optional (renal flag) |
| **CNS** | CSF obtained, neurological symptoms, recent neurosurgery, GCS score | Always shown |
| **Other** | No site-specific fields | Optional (renal flag) |

The **Suspected source** dropdown adapts contextually (e.g., respiratory → CAP / HAP / VAP / Aspiration / ...).

**Lab / Culture Results** accepts three input modes:
- **None** — empirical pathway only
- **Upload file** — PDF (text extracted via pypdf) or image (PNG/JPG/TIFF sent to MedGemma vision)
- **Paste text** — manual copy-paste from a lab system

---

## Knowledge Base Sources

All data is open-access — no registration required except where noted.

| Source | Contents | Used for |
|--------|----------|---------|
| [EUCAST v16.0](https://www.eucast.org/bacteria/clinical-breakpoints-and-interpretation/) | Clinical breakpoint tables | MIC interpretation, creep detection |
| [WHO AWaRe 2024](https://aware.essentialmeds.org) | Access / Watch / Reserve classification | Antibiotic stewardship |
| [IDSA AMR Guidance 2024](https://www.idsociety.org/practice-guideline/amr-guidance/) | Treatment guidelines PDF | Empirical therapy RAG |
| [Pfizer ATLAS](https://atlas-surveillance.com) *(free registration)* | 6.5M MIC surveillance measurements | Resistance patterns RAG |
| [WHO GLASS](https://worldhealthorg.shinyapps.io/glass-dashboard/) | 23M+ AMR episodes, 141 countries | Global resistance context |
| [DDInter 2.0](https://ddinter2.scbdd.com) | 191,000+ drug-drug interactions | Interaction screening |
| [OpenFDA](https://api.fda.gov/drug/label.json) | Drug labeling / contraindications | Safety RAG |

---

## Project Structure

```
amr-guard/
├── app.py                  # Streamlit UI — all four pages
├── setup_demo.py           # One-command knowledge base setup
├── requirements.txt        # pip requirements (HF Spaces / CI)
├── packages.txt            # apt system packages (HF Spaces)
├── pyproject.toml          # Full dependency spec (managed by uv)
├── .env.example            # Environment variable template
│
├── src/
│   ├── agents.py           # Four agent implementations
│   ├── form_config.py      # Dynamic form field definitions per infection site
│   ├── graph.py            # LangGraph orchestrator + conditional routing
│   ├── loader.py           # Model loading: multimodal + causal LM + vision inference
│   ├── prompts.py          # System and user prompts for all agents
│   ├── rag.py              # ChromaDB ingestion and retrieval helpers
│   ├── state.py            # InfectionState TypedDict schema
│   ├── utils.py            # CrCl calculator, MIC creep detection
│   ├── config.py           # Pydantic settings (reads from .env / Space Secrets)
│   ├── tools/
│   │   ├── antibiotic_tools.py   # WHO AWaRe lookups, MIC interpretation
│   │   ├── resistance_tools.py   # Pathogen resistance pattern queries
│   │   ├── safety_tools.py       # Drug interaction screening
│   │   └── rag_tools.py          # Guideline retrieval wrappers
│   └── db/
│       ├── schema.sql            # SQLite table definitions
│       ├── database.py           # Connection and query helpers
│       ├── import_data.py        # ETL: Excel/CSV/PDF → SQLite
│       └── vector_store.py       # ChromaDB ingestion
│
├── docs/                   # Source data files (committed — used by setup_demo.py)
│   ├── antibiotic_guidelines/   # WHO AWaRe Excel exports, IDSA PDF
│   ├── mic_breakpoints/         # EUCAST v16.0 breakpoint tables
│   ├── pathogen_resistance/     # ATLAS susceptibility data
│   └── drug_safety/             # DDInter drug interaction CSV
│
├── notebooks/
│   └── kaggle_medic_demo.ipynb  # Full reproducible Kaggle notebook
│
└── tests/
    └── test_pipeline.py         # Agent and pipeline unit tests
```

---

> **Research demo only.** Not validated for clinical use. All recommendations must be reviewed by a licensed clinician before any patient-care decision.
