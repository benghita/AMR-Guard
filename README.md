# AMR-Guard: Infection Lifecycle Orchestrator

A multi-agent clinical decision-support system for antimicrobial stewardship, submitted to the **[MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)**.

Powered by **MedGemma** (4B multimodal + 27B text) and **TxGemma** — HAI-DEF models from Google.

---

## What it does

AMR-Guard guides clinicians through two stages of infection management:

**Stage 1 — Empirical** (no lab results yet)
Patient history → risk factor analysis → empirical antibiotic recommendation

**Stage 2 — Targeted** (lab results available)
Lab report image or PDF (any language) → pathogen & MIC extraction → resistance trend analysis → targeted prescription with drug interaction screening

A unique capability is **MIC creep detection**: the system flags when a pathogen's Minimum Inhibitory Concentration has risen ≥4-fold across admissions — even while the lab still reports "Susceptible" — giving clinicians a 6–18 month early warning before formal treatment failure.

---

## Agent Pipeline

```
Patient form ──► Agent 1: Intake Historian  ──► (no lab) ──────────────────► Agent 4: Clinical Pharmacologist ──► Prescription
                       │                                                                  ▲
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

## Requirements

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- HuggingFace account with access granted to:
  - [MedGemma](https://huggingface.co/google/medgemma-4b-it)
  - [TxGemma](https://huggingface.co/google/txgemma-2b-predict)
---

## Setup

### 1. Install dependencies

```bash
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`. Minimum required settings:

```bash
# Local model IDs (HuggingFace)
MEDIC_LOCAL_MEDGEMMA_4B_MODEL=google/medgemma-4b-it
MEDIC_LOCAL_MEDGEMMA_27B_MODEL=google/medgemma-4b-it   # use 4B as fallback if <24 GB VRAM
MEDIC_LOCAL_TXGEMMA_9B_MODEL=google/txgemma-2b-predict
MEDIC_LOCAL_TXGEMMA_2B_MODEL=google/txgemma-2b-predict
```

### 3. Authenticate with HuggingFace

```bash
uv run huggingface-cli login
```

### 4. Build the knowledge base

Ingests EUCAST breakpoints, WHO AWaRe classification, IDSA guidelines, ATLAS surveillance data, and DDInter drug interactions into SQLite + ChromaDB:

```bash
uv run python setup_demo.py
```

This reads the source data files in `docs/` and writes to `data/` (gitignored, generated locally).

### 5. Run the app

```bash
uv run streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Kaggle Reproduction

The full pipeline can be reproduced on a free Kaggle T4 GPU (16 GB VRAM):

1. Open [`notebooks/kaggle_medic_demo.ipynb`](notebooks/kaggle_medic_demo.ipynb) in Kaggle
2. Add the `mghobashy/drug-drug-interactions` dataset to the notebook
3. Add your HuggingFace token as a Kaggle secret named `HF_TOKEN`
4. Run all cells — the notebook clones this repo, installs dependencies, builds the knowledge base, and launches the app via a public tunnel

Models run with 4-bit quantization on T4 (MedGemma 4B + TxGemma 2B).

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
medic-amr-guard/
├── app.py                  # Streamlit UI (single-file, all four stages)
├── setup_demo.py           # One-command knowledge base setup
├── pyproject.toml          # Dependencies (managed by uv)
├── .env.example            # Environment variable template
│
├── src/
│   ├── agents.py           # Four agent implementations
│   ├── graph.py            # LangGraph orchestrator + conditional routing
│   ├── loader.py           # Model loading: local HuggingFace causal LMs
│   ├── prompts.py          # System and user prompts for all agents
│   ├── rag.py              # ChromaDB ingestion and retrieval helpers
│   ├── state.py            # InfectionState TypedDict schema
│   ├── utils.py            # CrCl calculator, MIC creep detection
│   ├── config.py           # Pydantic settings (reads from .env)
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