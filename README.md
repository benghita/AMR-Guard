# Med-I-C — Infection Lifecycle Orchestrator

**MedGemma Impact Challenge submission** | Deadline: Feb 24, 2026

A multi-agent clinical decision-support system for antimicrobial stewardship, powered by MedGemma and TxGemma (HAI-DEF models).

---

## What it does

Med-I-C guides clinicians through two stages of infection management:

- **Stage 1 — Empirical** (no lab results yet): Patient history → empirical antibiotic recommendation
- **Stage 2 — Targeted** (lab results available): Lab report image/PDF (any language) → pathogen extraction → MIC trend analysis → targeted prescription

Key capability: **MIC creep detection** — flags rising resistance trends before the lab formally reports "Resistant", giving clinicians a 6–18 month early-warning window.

---

## Architecture

```
Patient Data ──▶ [Agent 1: Intake Historian]  ──▶ (no lab) ──▶ [Agent 4: Clinical Pharmacologist]
                         │                                                     ▲
                         └──▶ (lab uploaded) ──▶ [Agent 2: Vision Specialist]  │
                                                          │                    │
                                                 [Agent 3: Trend Analyst] ─────┘
```

| Agent | Model | Role |
|-------|-------|------|
| Intake Historian | MedGemma 4B IT | Parse EHR notes, calculate CrCl, identify MDR risk factors |
| Vision Specialist | MedGemma 4B IT (multimodal) | Extract pathogen + MICs from lab images/PDFs in any language |
| Trend Analyst | MedGemma 27B Text IT | Detect MIC creep, compute resistance velocity vs EUCAST breakpoints |
| Clinical Pharmacologist | MedGemma 4B IT + TxGemma 9B | Select antibiotic, dose, check drug interactions, apply WHO AWaRe |

**Orchestration:** LangGraph | **Knowledge base:** SQLite (EUCAST, WHO AWaRe, ATLAS, DDInter) + ChromaDB (IDSA guidelines, WHO GLASS) | **UI:** Streamlit

---

## Quick Start

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for package management
- HuggingFace account with MedGemma and TxGemma access granted

### 1. Clone and install

```bash
git clone https://github.com/your-org/Med-I-C
cd Med-I-C
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — set HUGGINGFACE_TOKEN and choose backend (local or vertex)
```

### 3. Build knowledge base

```bash
uv run python setup_demo.py
```

This downloads and ingests all open-access data sources (EUCAST breakpoints, WHO AWaRe, IDSA guidelines, DDInter drug interactions) into SQLite and ChromaDB.

### 4. Run the app

```bash
uv run streamlit run app.py
```

---

## Kaggle Reproduction

Open `notebooks/kaggle_medic_demo.ipynb` in Kaggle:

1. Add the `mghobashy/drug-drug-interactions` dataset
2. Add your HuggingFace token as a Kaggle secret (`HF_TOKEN`)
3. Run all cells — the notebook sets up the full environment and launches the app via a public tunnel

Tested on Kaggle T4 GPU (16 GB VRAM) using 4-bit quantization for MedGemma 4B and TxGemma 2B.

---

## Data Sources (all open-access)

| Source | Use |
|--------|-----|
| [EUCAST v16.0](https://www.eucast.org) | Clinical breakpoint tables for MIC interpretation |
| [WHO AWaRe 2024](https://aware.essentialmeds.org) | Antibiotic stewardship classification |
| [IDSA AMR Guidance 2024](https://www.idsociety.org/practice-guideline/amr-guidance/) | Treatment guidelines (RAG) |
| [Pfizer ATLAS](https://atlas-surveillance.com) | 6.5M MIC surveillance measurements |
| [WHO GLASS](https://worldhealthorg.shinyapps.io/glass-dashboard/) | 23M+ global AMR surveillance episodes |
| [DDInter 2.0](https://ddinter2.scbdd.com) | 191,000+ drug-drug interactions |
| [OpenFDA](https://api.fda.gov/drug/label.json) | Drug labeling and safety data |

---

## Model Licenses

You must accept the model licenses on HuggingFace before use:
- MedGemma: https://huggingface.co/google/medgemma-4b-it
- TxGemma: https://huggingface.co/google/txgemma-9b-predict

---

## Project Structure

```
Med-I-C/
├── app.py                  # Streamlit UI
├── setup_demo.py           # One-command knowledge base setup
├── src/
│   ├── agents.py           # 4 agent implementations
│   ├── graph.py            # LangGraph orchestrator + conditional routing
│   ├── loader.py           # Model loading (local / Vertex AI / 4-bit quant)
│   ├── prompts.py          # System and user prompts for all agents
│   ├── rag.py              # ChromaDB ingestion and retrieval
│   ├── state.py            # InfectionState schema (TypedDict)
│   ├── utils.py            # CrCl calculator, MIC creep detection
│   ├── config.py           # Pydantic settings
│   ├── tools/              # Antibiotic, resistance, safety, RAG query tools
│   └── db/                 # SQLite schema, import scripts, vector store
├── docs/                   # Source data files (EUCAST xlsx, IDSA pdf, etc.)
├── notebooks/
│   └── kaggle_medic_demo.ipynb
└── tests/
    └── test_pipeline.py
```

---

## Competition Writeup

See [WRITEUP.md](WRITEUP.md) for the full 3-page submission document.
