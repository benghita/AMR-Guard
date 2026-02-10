# AMR-Guard: Infection Lifecycle Orchestrator — Execution Plan

## Competition
- **Challenge:** [MedGemma Impact Challenge](https://www.kaggle.com/competitions/med-gemma-impact-challenge)
- **Deadline:** February 24, 2026
- **Submission:** 3-min video + 3-page writeup + reproducible source code
- **Requirement:** ≥1 HAI-DEF model (MedGemma, TxGemma, etc.)
- **Target award:** Agentic Workflow

---

## 1. Project Structure

```
Med-I-C/
├── pyproject.toml
├── .env.example
├── src/
│   ├── config.py              # Settings, env vars
│   ├── state.py               # InfectionState schema
│   ├── graph.py               # LangGraph orchestrator
│   ├── loader.py              # Model loading (local/Vertex/quantized)
│   ├── prompts.py             # Prompt templates
│   ├── rag.py                 # ChromaDB ingest + retrieval
│   ├── agents.py              # All 4 agents in one module
│   └── utils.py               # CrCl calculator, MIC analysis, formatters
├── app.py                     # Streamlit frontend (single file)
├── scripts/
│   └── ingest_data.py         # Download & ingest real data into ChromaDB
├── tests/
│   └── test_pipeline.py
└── data/                      # Downloaded real-world data (gitignored)
    └── chroma_db/
```

---

## 2. Multi-Agent Architecture (4 Agents, 2 Stages)

**Stage 1 — Empirical Advisor** (no lab results): Agent 1 → Agent 4
**Stage 2 — Targeted Sentinel** (lab results available): Agent 1 → Agent 2 → Agent 3 → Agent 4

| Agent | Role | Model | Fallback |
|-------|------|-------|----------|
| 1. Intake Historian | Parse patient data, risk factors, CrCl | MedGemma 4B IT | Vertex AI API |
| 2. Vision Specialist | Lab report (image/PDF, any language) → structured JSON in English | MedGemma 4B IT (multimodal) | Vertex AI API |
| 3. Trend Analyst | MIC creep detection, resistance velocity | MedGemma 27B Text IT | Vertex API / 4B fallback |
| 4. Clinical Pharmacologist | Final Rx + safety checks | MedGemma 4B + TxGemma 9B (safety only) | TxGemma 2B |

**Note on TxGemma:** Drug discovery model (toxicity, molecular properties), NOT clinical prescribing. Used only as safety checker. MedGemma + RAG handles antibiotic selection.

**Note on Vision Specialist:** Accepts lab reports in any format (image: PNG/JPG, PDF) and any language. Uses MedGemma multimodal to extract culture & sensitivity data, pathogen identification, MIC values, and S/I/R interpretations. Outputs standardized English JSON regardless of input language. Supports:
- Culture & sensitivity reports
- Antibiogram reports
- Microbiology lab results
- Blood culture reports
- Urine culture reports

---

## 3. Data Sources (Real-World, Open-Access Only)

### RAG Collection 1: `antibiotic_guidelines`

| Source | URL | Format | Access |
|--------|-----|--------|--------|
| **WHO AWaRe Classification** | https://aware.essentialmeds.org/groups | Web/structured DB | Free, no registration |
| **IDSA AMR Guidance** | https://www.idsociety.org/practice-guideline/amr-guidance/ | PDF/HTML | Free download |
| **NICE Antimicrobial Guidelines** | https://www.nice.org.uk/guidance/conditions-and-diseases/infections/antimicrobial-stewardship | JSON via API (`https://api.nice.org.uk/`) | Free API key |
| **WHO Medically Important Antimicrobials** | https://cdn.who.int/media/docs/default-source/gcp/who-mia-list-2024-lv.pdf | PDF | CC BY-NC-SA 3.0 |

### RAG Collection 2: `mic_breakpoints`

| Source | URL | Format | Access |
|--------|-----|--------|--------|
| **EUCAST Clinical Breakpoints v16.0** | https://www.eucast.org/bacteria/clinical-breakpoints-and-interpretation/clinical-breakpoint-tables/ | Excel/PDF | Free download, updated annually |
| **CDC Reference AST Data** | https://www.cdc.gov/healthcare-associated-infections/php/lab-resources/reference-ast-data.html | Tables | Free |

### RAG Collection 3: `drug_safety`

| Source | URL | Format | Access |
|--------|-----|--------|--------|
| **OpenFDA Drug Labeling API** | `https://api.fda.gov/drug/label.json` | REST JSON | Free, no key needed |
| **DailyMed** | `https://dailymed.nlm.nih.gov/dailymed/services/` | REST JSON/XML + bulk ZIP | Free |
| **RxNorm API** | `https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html` | REST JSON/XML | Free |
| **DDInter 2.0** (drug-drug interactions) | https://ddinter2.scbdd.com | Web + downloadable | Free, 302K DDI records |
| **DrugBank Open** | https://go.drugbank.com/releases/latest | XML/CSV | Free academic license |

### RAG Collection 4: `pathogen_resistance`

| Source | URL | Format | Access |
|--------|-----|--------|--------|
| **WHO GLASS Dashboard** | https://worldhealthorg.shinyapps.io/glass-dashboard/ | Downloadable datasets | Free, 23M+ episodes, 141 countries |
| **ATLAS (Pfizer Surveillance)** | https://atlas-surveillance.com | MIC data via Vivli | Free registration, 6.5M MICs |
| **EARS-Net / ECDC** | https://www.ecdc.europa.eu/en/about-us/networks/disease-networks-and-laboratory-networks/ears-net-data | Tables/reports | Free |
| **CARD Database** | https://card.mcmaster.ca/download | JSON/FASTA | CC-BY 4.0, 8.5K ontology terms |
| **NCBI NDARO** | https://www.ncbi.nlm.nih.gov/pathogens/antimicrobial-resistance/ | FTP + BigQuery | Free |
| **SENTRY Program** | https://ghdx.healthdata.org/record/sentry-antimicrobial-surveillance-program-data-2012-2018 | Downloadable via GHDx | Free |
| **CDC AR Threat Reports** | https://www.cdc.gov/antimicrobial-resistance/data-research/threats/index.html | PDF/data | Free |

### Lab Report Samples (for Vision Specialist testing)

| Source | Description | Access |
|--------|-------------|--------|
| **Public hospital antibiograms** | Annual susceptibility reports (Stanford, Washington State) | Free PDF downloads |
| **Real-world lab report templates** | Culture & sensitivity report formats from various countries/languages | Collected from open hospital publications |

### Embedding Model
`all-MiniLM-L6-v2` (384-dim, runs on CPU)

### Data Ingestion Strategy (`scripts/ingest_data.py`)
1. Download EUCAST Excel breakpoint tables → parse → chunk → embed into `mic_breakpoints`
2. Fetch OpenFDA labels for key antibiotics → extract warnings/interactions → embed into `drug_safety`
3. Download DDInter bulk data → parse interactions → embed into `drug_safety`
4. Fetch IDSA/WHO guidelines text → chunk → embed into `antibiotic_guidelines`
5. Download GLASS/ATLAS resistance rates → parse by organism/region → embed into `pathogen_resistance`
6. Download CARD resistance gene data → embed into `pathogen_resistance`

---

## 4. Development Phases

### Phase 1: Foundation (Days 1–4)
- `uv init`, install deps, set up `config.py`, `state.py`, `loader.py`
- Build `scripts/ingest_data.py`: download real data from APIs/files above
- Ingest into ChromaDB, verify RAG retrieval
- Verify MedGemma 4B loads and responds
- **Milestone:** Model loads, RAG returns real guideline/breakpoint data

### Phase 2: Stage 1 MVP (Days 5–8)
- Implement Agent 1 (Intake Historian) + Agent 4 (Clinical Pharmacologist)
- Wire into LangGraph (Stage 1 path only)
- Build Streamlit app with patient form + prescription card
- **Milestone:** Patient data → empirical antibiotic recommendation

### Phase 3: Stage 2 — Vision + Trends (Days 9–14)
- Implement Agent 2 (Vision Specialist): lab report (image/PDF, any language) → structured English JSON
- Implement Agent 3 (Trend Analyst): MIC creep detection with real EUCAST breakpoints
- Wire full 4-agent LangGraph with conditional routing
- Add lab uploader to Streamlit
- **Milestone:** Full 4-agent pipeline runs end-to-end

### Phase 4: Polish + Testing (Days 15–19)
- Test with real breakpoint data, real guideline retrieval
- Add safety alerts, error handling, loading states
- Test on Kaggle notebook environment
- **Milestone:** Robust, demo-ready

### Phase 5: Submission (Days 20–24)
- Record 3-min demo video, write 3-page writeup
- Create reproducible Kaggle notebook
- **Submit by Feb 24**

---

## 5. Key Files to Create

| File | What it does |
|------|-------------|
| `pyproject.toml` | Dependencies: langgraph, langchain-core, chromadb, streamlit, transformers, torch, accelerate, bitsandbytes, sentence-transformers, google-cloud-aiplatform, Pillow, pydantic, python-dotenv, openpyxl, requests |
| `src/state.py` | `InfectionState` TypedDict: patient info, CrCl, lab results, MIC data, recommendations, alerts |
| `src/config.py` | Pydantic Settings: model IDs, deploy prefs, API keys, ChromaDB path |
| `src/loader.py` | `run_inference()`: auto-select local vs Vertex, 4-bit quantization fallback |
| `src/prompts.py` | 4 agent prompts + TxGemma safety prompt |
| `src/rag.py` | ChromaDB ingest + `get_context_string()` retriever |
| `src/agents.py` | `intake_historian.run()`, `vision_specialist.run()`, `trend_analyst.run()`, `clinical_pharmacologist.run()` |
| `src/utils.py` | `calculate_crcl()`, `detect_mic_creep()`, `format_prescription_card()` |
| `src/graph.py` | LangGraph `StateGraph` with conditional routing (Stage 1 vs 2) |
| `app.py` | Streamlit: patient form, lab uploader, results display |
| `scripts/ingest_data.py` | Download + parse + embed real data from EUCAST/OpenFDA/GLASS/CARD/DDInter |

---

## 6. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| MedGemma 27B too large | Vertex AI API; fallback to 4B with enhanced prompt |
| TxGemma bad at clinical Rx | Safety checker only; MedGemma + RAG for drug selection |
| Lab report parsing unreliable | Few-shot prompts; PDF text extraction pre-processing; manual JSON entry fallback in UI; multi-language prompt instructions |
| Real data ingestion failures | Cache downloaded files; fallback to cached ChromaDB |
| Kaggle reproducibility | Pin versions; include download script; test on fresh kernel |

---

## 7. Verification

1. `uv run python scripts/ingest_data.py` → verify real data ingested into ChromaDB
2. `uv run streamlit run app.py` → UI loads
3. Enter patient data (UTI case) → empirical recommendation cites real IDSA/WHO guidelines
4. Upload lab report (image or PDF, test with non-English sample) → structured English JSON extraction
5. MIC creep case → alert fires using real EUCAST breakpoints
6. Drug interaction case (warfarin + antibiotics) → safety alert from real DDInter/OpenFDA data
7. Test on Kaggle notebook → reproducible
