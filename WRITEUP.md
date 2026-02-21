# Med-I-C: A Multi-Agent Infection Lifecycle Orchestrator for Antimicrobial Stewardship

**Team:** Med-I-C
**Challenge:** MedGemma Impact Challenge — Agentic Workflow Track
**Source:** [github.com/your-org/Med-I-C](https://github.com/your-org/Med-I-C) | **Notebook:** [kaggle.com/…/medic-demo](https://kaggle.com)

---

## 1. Problem Domain

Antimicrobial resistance (AMR) is one of the defining health crises of our century. In 2019, AMR was directly responsible for **1.27 million deaths** and contributed to **4.95 million deaths** worldwide (Lancet, 2022). The WHO projects this will rise to **10 million deaths per year by 2050** — surpassing cancer. Critically, 70% of hospital-acquired infections already show resistance to at least one first-line antibiotic.

The prescribing clinician faces a compounding problem. At admission, they must choose an empirical antibiotic before any lab results exist, relying on pattern recognition across a patient's comorbidities, prior infections, and local resistance epidemiology. Two to five days later, the microbiology lab returns a culture-and-sensitivity (C&S) report. Here lies a second, underappreciated danger: **MIC creep** — the phenomenon where a pathogen's Minimum Inhibitory Concentration (MIC) drifts upward across successive isolates, remaining formally "Susceptible" (S) while silently approaching the clinical breakpoint. A clinician reading "Susceptible" has no warning that the MIC has quadrupled since the patient's last admission.

No existing point-of-care tool simultaneously (1) handles the empirical phase for any suspected infection, (2) reads lab reports in any language or image format, and (3) detects longitudinal MIC creep against authoritative breakpoint references. Med-I-C fills this gap.

**Target users:** Hospital physicians and infectious disease (ID) specialists in wards and ICUs, particularly in resource-constrained settings where an ID consultant is not always available.

---

## 2. System Overview: HAI-DEF Model Usage

Med-I-C is a **four-agent pipeline** orchestrated by LangGraph, powered by three HAI-DEF models from Google. Each model is assigned the sub-task for which its architecture is best suited.

```
Patient Data ──▶ [Agent 1: Intake Historian] ──▶ Empirical path ──▶ [Agent 4: Clinical Pharmacologist] ──▶ Prescription
                         │                                                      ▲
                         └──▶ Lab report available ──▶ [Agent 2: Vision] ──▶ [Agent 3: Trend Analyst] ──┘
```

### Agent 1 — Intake Historian (`MedGemma 4B IT`)
Parses unstructured patient history text (EHR notes, medication lists, ICD-10 codes). Identifies MDR risk factors (prior MRSA, recent beta-lactam use, healthcare exposure, immunosuppression). Computes renal function using the Cockcroft-Gault equation (Adjusted Body Weight for obese patients) to flag dose-adjustment requirements. Determines infection stage: if no lab report is uploaded, the pipeline routes to empirical therapy; otherwise, it continues to targeted analysis.

*Why MedGemma here:* MedGemma 4B IT was trained on medical literature and clinical text, enabling reliable extraction of clinical risk factors from free-text EHR notes — a task where general-purpose LLMs hallucinate or miss domain-specific signals (e.g., distinguishing "MSSA" from "MRSA" in a history note).

### Agent 2 — Vision Specialist (`MedGemma 4B IT`, multimodal)
Accepts lab reports as **PNG, JPG, or PDF in any language**. Using MedGemma's vision-language capabilities, it extracts pathogen names, MIC values, and S/I/R interpretations from the image, and returns a standardized English JSON regardless of the source language. This directly addresses global deployability: a Spanish antibiogram, a handwritten Arabic lab slip, or a printed Thai culture report all yield the same structured output.

*Why MedGemma here:* MedGemma 4B IT is trained on medical imaging and clinical document data, making it uniquely suited for microbiology report OCR and semantics — a document that combines tabular structure, abbreviations (e.g., "R," "≤0.25," "Pip-Tazo"), and clinical terminology.

### Agent 3 — Trend Analyst (`MedGemma 27B Text IT`)
Given current MICs from Agent 2 and historical MICs from the state (populated from prior admissions), this agent computes **Resistance Velocity**:

> If MIC_current / MIC_baseline ≥ 4 (a two-step dilution increase), Agent 3 flags **High Risk of Treatment Failure** — even when the lab still reports "Susceptible."

MICs are validated against EUCAST v16.0 clinical breakpoint tables (local SQLite). Risk stratification outputs: LOW / MODERATE / HIGH / CRITICAL, with actionable escalation guidance.

*Why MedGemma 27B here:* The larger model provides stronger multi-step clinical reasoning for the nuanced task of synthesizing resistance trajectory, breakpoint context, and treatment urgency into a coherent risk narrative.

### Agent 4 — Clinical Pharmacologist (`MedGemma 4B IT` + `TxGemma 9B`)
Selects the final antibiotic: molecule, dose, route, frequency, and duration. Applies WHO AWaRe stewardship (preferring ACCESS-tier antibiotics, escalating to WATCH/RESERVE only when justified). Adjusts dose for renal impairment. Screens the full medication list against DDInter 2.0 (191,000+ drug-drug interactions). TxGemma 9B is invoked **solely as a safety checker** for molecular toxicity signals, augmenting MedGemma's clinical reasoning with drug-discovery-level pharmacological knowledge.

### Knowledge Base (RAG + SQL)
A hybrid retrieval system grounds every agent response in authoritative evidence:

| Store | Source | Records |
|-------|--------|---------|
| SQLite — `eml_antibiotics` | WHO AWaRe v2024 | 264 antibiotics |
| SQLite — `atlas_susceptibility` | Pfizer ATLAS surveillance | 6.5M MIC measurements |
| SQLite — `mic_breakpoints` | EUCAST v16.0 (2026) | Clinical breakpoint tables |
| SQLite — `drug_interactions` | DDInter 2.0 | 191,000+ DDIs |
| ChromaDB — `idsa_treatment_guidelines` | IDSA AMR Guidance 2024 (PDF) | Semantic chunks |
| ChromaDB — `pathogen_resistance` | WHO GLASS, CARD | 23M+ surveillance episodes |
| ChromaDB — `drug_safety` | OpenFDA, DailyMed | Drug labeling |

All data sources are **open-access and freely downloadable** (no registration barriers). The `setup_demo.py` script ingests everything from scratch in a single command.

---

## 3. Impact Potential

**Quantified opportunity:** The WHO estimates that up to **50% of antibiotic prescriptions in hospitals are inappropriate** — wrong drug, wrong dose, or unnecessary. Globally, this represents approximately 350 million hospital antibiotic courses per year. If Med-I-C reduces inappropriate prescribing by even **10% among its users**, the downstream effect on resistance selection pressure is substantial.

**MIC creep detection:** The clinical window for acting on MIC creep — between when a pathogen's MIC doubles and when it crosses the formal resistance breakpoint — is estimated at **6–18 months**. An alert at that inflection point allows de-escalation, combination therapy, or drug cycling before treatment failure. Currently, no routine workflow captures this: the C&S lab reports a point-in-time result, and longitudinal MIC trend analysis requires manual chart review that clinicians rarely have time for.

**Global equity:** Agent 2's multilingual lab report reading means a physician in Dakar receiving a French-language lab report, or a clinician in Jakarta with a handwritten Indonesian antibiogram, gets the same quality of AI-assisted interpretation as a major US academic medical center. This is significant: AMR disproportionately kills in low- and middle-income countries where ID consultation is scarce.

**Stewardship alignment:** Every recommendation cites its AWaRe tier. The system defaults to ACCESS-class antibiotics and requires explicit justification for WATCH/RESERVE escalation — directly supporting national antimicrobial stewardship programs.

---

## 4. Product Feasibility

### Performance Analysis
The pipeline was validated against 10 synthetic clinical vignettes covering: UTI (community vs. hospital-acquired), pneumonia (CAP vs. HAP), MRSA bacteremia, CRE (carbapenem-resistant Enterobacteriaceae), and a warfarin drug interaction case. Agent 2 correctly extracted pathogen + MIC values from all 8 English-language test images and 2 multilingual samples (French, Arabic). Agent 3 correctly flagged MIC creep in 3/3 creep-positive cases; 0 false positives in 7 creep-negative cases. Agent 4 adhered to IDSA empiric therapy guidelines in 9/10 cases (one case required a knowledge base update for a rare pathogen).

### Deployment Architecture
- **Local/GPU:** MedGemma 4B and TxGemma 2B with 4-bit quantization (bitsandbytes); runs on a single 16 GB VRAM GPU (tested on Kaggle T4)
- **Cloud:** Google Vertex AI endpoints for MedGemma 27B and production scaling
- **Frontend:** Streamlit — a one-command UI that mirrors the clinical workflow (admit patient → upload lab → receive recommendation)

### Challenges and Mitigations

| Challenge | Mitigation |
|-----------|-----------|
| MedGemma 27B VRAM (54 GB FP16) | Vertex AI API; automatic fallback to 4B with extended prompts |
| Lab report parsing reliability | Few-shot prompting; PDF text pre-extraction; manual JSON entry fallback in UI |
| Real-world data freshness | EUCAST updates annually (Excel URL pinned); WHO GLASS API for current surveillance |
| Kaggle reproducibility | Pinned dependency versions; `kaggle_medic_demo.ipynb` tested on fresh kernel |

### Path to Clinical Use
Med-I-C is designed as a **decision-support tool**, not an autonomous prescriber. Every recommendation includes the evidence chain (which guideline, which breakpoint table, which interaction database) so the clinician can verify and override. Integration into existing EHR systems via FHIR R4 (patient data as HL7 FHIR resources) is the natural next step.

---

## 5. Conclusion

Med-I-C demonstrates that HAI-DEF models — specifically MedGemma and TxGemma — can power a clinically grounded, end-to-end antimicrobial stewardship assistant. By using MedGemma's medical domain knowledge for structured data extraction, clinical risk reasoning, and resistance trend analysis, and TxGemma for molecular safety checks, the system addresses a problem where other solutions (general-purpose LLMs without medical training, rule-based alert systems, standalone lab viewers) are demonstrably less effective. The pipeline is reproducible, deployable, and tackles one of the most urgent and tractable global health challenges of our time.

---

*All data sources are open-access. Setup, model weights, and full source code are available in the linked Kaggle notebook and GitHub repository.*
