# Med-I-C Knowledge Storage Strategy

## Overview

This document defines how each document in the `docs/` folder will be stored and queried to support the **AMR-Guard: Infection Lifecycle Orchestrator** workflow.

---

## Document Classification Summary

| Document | Type | Storage | Purpose in Workflow |
|----------|------|---------|---------------------|
| EML exports (ACCESS/RESERVE/WATCH) | XLSX | **SQLite** | Antibiotic classification & stewardship |
| ATLAS Susceptibility Data | XLSX | **SQLite** | Pathogen resistance patterns |
| MIC Breakpoint Tables | XLSX | **SQLite** | Susceptibility interpretation |
| Drug Interactions | CSV | **SQLite** | Drug safety screening |
| IDSA Guidance (ciae403.pdf) | PDF | **ChromaDB** | Clinical treatment guidelines |
| MIC Breakpoint Tables (PDF) | PDF | **ChromaDB** | Reference documentation |

---

## Part 1: Structured Data (SQLite)

### 1.1 EML Antibiotic Classification Tables

**Source Files:**
- `antibiotic_guidelines/EML export ACCESS group.xlsx`
- `antibiotic_guidelines/EML export RESERVE group.xlsx`
- `antibiotic_guidelines/EML export WATCH group.xlsx`

**Database Table: `eml_antibiotics`**

```sql
CREATE TABLE eml_antibiotics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    medicine_name TEXT NOT NULL,
    who_category TEXT NOT NULL,  -- 'ACCESS', 'RESERVE', 'WATCH'
    eml_section TEXT,
    formulations TEXT,
    indication TEXT,
    atc_codes TEXT,
    combined_with TEXT,
    status TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_medicine_name ON eml_antibiotics(medicine_name);
CREATE INDEX idx_who_category ON eml_antibiotics(who_category);
CREATE INDEX idx_atc_codes ON eml_antibiotics(atc_codes);
```

**Usage in Workflow:**
- **Agent 1 (Intake Historian):** Query to identify antibiotic stewardship category
- **Agent 4 (Clinical Pharmacologist):** Suggest ACCESS antibiotics first, escalate to WATCH/RESERVE only when necessary

---

### 1.2 ATLAS Pathogen Susceptibility Data

**Source File:** `pathogen_resistance/ATLAS Susceptibility Data Export.xlsx`

**Database Tables:**

```sql
CREATE TABLE atlas_susceptibility_percent (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pathogen TEXT NOT NULL,
    antibiotic TEXT NOT NULL,
    region TEXT,
    year INTEGER,
    susceptibility_percent REAL,
    sample_size INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE atlas_susceptibility_absolute (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pathogen TEXT NOT NULL,
    antibiotic TEXT NOT NULL,
    region TEXT,
    year INTEGER,
    susceptible_count INTEGER,
    intermediate_count INTEGER,
    resistant_count INTEGER,
    total_isolates INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pathogen ON atlas_susceptibility_percent(pathogen);
CREATE INDEX idx_antibiotic ON atlas_susceptibility_percent(antibiotic);
CREATE INDEX idx_pathogen_abs ON atlas_susceptibility_absolute(pathogen);
```

**Usage in Workflow:**
- **Agent 1 (Empirical Phase):** Retrieve local/regional resistance patterns for empirical therapy
- **Agent 3 (Trend Analyst):** Compare current MIC with population-level trends

---

### 1.3 MIC Breakpoint Tables

**Source File:** `mic_breakpoints/v_16.0__BreakpointTables.xlsx`

**Database Tables:**

```sql
CREATE TABLE mic_breakpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pathogen_group TEXT NOT NULL,  -- e.g., 'Enterobacterales', 'Staphylococcus'
    antibiotic TEXT NOT NULL,
    route TEXT,  -- 'IV', 'Oral', 'Topical'
    mic_susceptible REAL,  -- S breakpoint (mg/L)
    mic_resistant REAL,    -- R breakpoint (mg/L)
    disk_susceptible REAL, -- Zone diameter (mm)
    disk_resistant REAL,
    notes TEXT,
    eucast_version TEXT DEFAULT '16.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dosage_guidance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    antibiotic TEXT NOT NULL,
    standard_dose TEXT,
    high_dose TEXT,
    renal_adjustment TEXT,
    notes TEXT
);

CREATE INDEX idx_bp_pathogen ON mic_breakpoints(pathogen_group);
CREATE INDEX idx_bp_antibiotic ON mic_breakpoints(antibiotic);
```

**Usage in Workflow:**
- **Agent 2 (Vision Specialist):** Validate extracted MIC values against breakpoints
- **Agent 3 (Trend Analyst):** Interpret S/I/R classification from MIC values
- **Agent 4 (Clinical Pharmacologist):** Use dosage guidance for prescriptions

---

### 1.4 Drug Interactions Database

**Source File:** `drug_safety/db_drug_interactions.csv`

**Database Table:**

```sql
CREATE TABLE drug_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_1 TEXT NOT NULL,
    drug_2 TEXT NOT NULL,
    interaction_description TEXT,
    severity TEXT,  -- Derived: 'major', 'moderate', 'minor'
    mechanism TEXT, -- Derived from description
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_drug_1 ON drug_interactions(drug_1);
CREATE INDEX idx_drug_2 ON drug_interactions(drug_2);
CREATE INDEX idx_severity ON drug_interactions(severity);

-- View for bidirectional lookup
CREATE VIEW drug_interaction_lookup AS
SELECT drug_1, drug_2, interaction_description, severity FROM drug_interactions
UNION ALL
SELECT drug_2, drug_1, interaction_description, severity FROM drug_interactions;
```

**Usage in Workflow:**
- **Agent 4 (Clinical Pharmacologist):** Check for interactions with patient's current medications
- **Safety Alerts:** Flag potential toxicity issues

---

## Part 2: Unstructured Data (ChromaDB)

### 2.1 IDSA Clinical Guidelines

**Source File:** `antibiotic_guidelines/ciae403.pdf`

**ChromaDB Collection: `idsa_treatment_guidelines`**

```python
collection_config = {
    "name": "idsa_treatment_guidelines",
    "metadata": {
        "source": "IDSA 2024 Guidance",
        "doi": "10.1093/cid/ciae403",
        "version": "2024"
    },
    "embedding_function": "sentence-transformers/all-MiniLM-L6-v2"
}

# Document chunking strategy
chunk_config = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "separators": ["\n\n", "\n", ". "],
    "metadata_fields": ["section", "pathogen_type", "recommendation_type"]
}
```

**Metadata Schema per Chunk:**
```python
{
    "section": "Treatment Recommendations",
    "pathogen_type": "ESBL-E | CRE | CRAB | DTR-PA | S.maltophilia",
    "recommendation_strength": "Strong | Conditional",
    "evidence_quality": "High | Moderate | Low",
    "page_number": int
}
```

**Usage in Workflow:**
- **Agent 1 (Empirical Phase):** Retrieve treatment recommendations for suspected pathogens
- **Agent 4 (Clinical Pharmacologist):** Provide evidence-based justification for antibiotic selection

---

### 2.2 MIC Breakpoint Reference (PDF)

**Source File:** `mic_breakpoints/v_16.0_Breakpoint_Tables.pdf`

**ChromaDB Collection: `mic_reference_docs`**

```python
collection_config = {
    "name": "mic_reference_docs",
    "metadata": {
        "source": "EUCAST Breakpoint Tables",
        "version": "16.0"
    },
    "embedding_function": "sentence-transformers/all-MiniLM-L6-v2"
}
```

**Usage in Workflow:**
- **Supplementary Context:** Provide detailed explanations for breakpoint interpretations
- **Edge Cases:** Handle unusual pathogens or antibiotic combinations not in structured tables

---

## Part 3: Query Tools Definition

### Tool 1: `query_antibiotic_info`

**Purpose:** Retrieve antibiotic classification and formulation details

```python
def query_antibiotic_info(
    antibiotic_name: str,
    include_category: bool = True,
    include_formulations: bool = True
) -> dict:
    """
    Query EML antibiotic database for classification and details.

    Args:
        antibiotic_name: Name of the antibiotic (partial match supported)
        include_category: Include WHO stewardship category
        include_formulations: Include available formulations

    Returns:
        dict with antibiotic details, category, indications

    Used by: Agent 1, Agent 4
    """
```

**SQL Query:**
```sql
SELECT medicine_name, who_category, formulations, indication, combined_with
FROM eml_antibiotics
WHERE LOWER(medicine_name) LIKE LOWER(?)
ORDER BY who_category;  -- ACCESS first, then WATCH, then RESERVE
```

---

### Tool 2: `query_resistance_pattern`

**Purpose:** Get susceptibility data for pathogen-antibiotic combinations

```python
def query_resistance_pattern(
    pathogen: str,
    antibiotic: str = None,
    region: str = None,
    year: int = None
) -> dict:
    """
    Query ATLAS susceptibility data for resistance patterns.

    Args:
        pathogen: Pathogen name (e.g., "E. coli", "K. pneumoniae")
        antibiotic: Optional specific antibiotic to check
        region: Optional geographic region filter
        year: Optional year filter (defaults to most recent)

    Returns:
        dict with susceptibility percentages and trends

    Used by: Agent 1 (Empirical), Agent 3 (Trend Analysis)
    """
```

**SQL Query:**
```sql
SELECT antibiotic, susceptibility_percent, sample_size, year
FROM atlas_susceptibility_percent
WHERE LOWER(pathogen) LIKE LOWER(?)
  AND (antibiotic = ? OR ? IS NULL)
  AND (region = ? OR ? IS NULL)
ORDER BY year DESC, susceptibility_percent DESC;
```

---

### Tool 3: `interpret_mic_value`

**Purpose:** Classify MIC as S/I/R based on EUCAST breakpoints

```python
def interpret_mic_value(
    pathogen: str,
    antibiotic: str,
    mic_value: float,
    route: str = "IV"
) -> dict:
    """
    Interpret MIC value against EUCAST breakpoints.

    Args:
        pathogen: Pathogen name or group
        antibiotic: Antibiotic name
        mic_value: MIC value in mg/L
        route: Administration route (IV, Oral)

    Returns:
        dict with interpretation (S/I/R), breakpoint values, dosing notes

    Used by: Agent 2, Agent 3
    """
```

**SQL Query:**
```sql
SELECT mic_susceptible, mic_resistant, notes
FROM mic_breakpoints
WHERE LOWER(pathogen_group) LIKE LOWER(?)
  AND LOWER(antibiotic) LIKE LOWER(?)
  AND (route = ? OR route IS NULL);
```

**Interpretation Logic:**
```python
if mic_value <= mic_susceptible:
    return "Susceptible"
elif mic_value > mic_resistant:
    return "Resistant"
else:
    return "Intermediate (Susceptible, Increased Exposure)"
```

---

### Tool 4: `check_drug_interactions`

**Purpose:** Screen for drug-drug interactions

```python
def check_drug_interactions(
    target_drug: str,
    patient_medications: list[str],
    severity_filter: str = None
) -> list[dict]:
    """
    Check for interactions between target drug and patient's medications.

    Args:
        target_drug: Antibiotic being considered
        patient_medications: List of patient's current medications
        severity_filter: Optional filter ('major', 'moderate', 'minor')

    Returns:
        list of interaction dicts with severity and description

    Used by: Agent 4 (Safety Check)
    """
```

**SQL Query:**
```sql
SELECT drug_1, drug_2, interaction_description, severity
FROM drug_interaction_lookup
WHERE LOWER(drug_1) LIKE LOWER(?)
  AND LOWER(drug_2) IN (SELECT LOWER(value) FROM json_each(?))
  AND (severity = ? OR ? IS NULL)
ORDER BY severity DESC;
```

---

### Tool 5: `search_clinical_guidelines`

**Purpose:** RAG search over IDSA guidelines for treatment recommendations

```python
def search_clinical_guidelines(
    query: str,
    pathogen_filter: str = None,
    n_results: int = 5
) -> list[dict]:
    """
    Semantic search over IDSA clinical guidelines.

    Args:
        query: Natural language query about treatment
        pathogen_filter: Optional pathogen type filter
        n_results: Number of results to return

    Returns:
        list of relevant guideline excerpts with metadata

    Used by: Agent 1 (Empirical), Agent 4 (Justification)
    """
```

**ChromaDB Query:**
```python
results = collection.query(
    query_texts=[query],
    n_results=n_results,
    where={"pathogen_type": pathogen_filter} if pathogen_filter else None,
    include=["documents", "metadatas", "distances"]
)
```

---

### Tool 6: `calculate_mic_trend`

**Purpose:** Analyze MIC creep over time

```python
def calculate_mic_trend(
    patient_id: str,
    pathogen: str,
    antibiotic: str,
    historical_mics: list[dict]  # [{date, mic_value}, ...]
) -> dict:
    """
    Calculate resistance velocity and MIC trend.

    Args:
        patient_id: Patient identifier
        pathogen: Identified pathogen
        antibiotic: Target antibiotic
        historical_mics: List of historical MIC readings

    Returns:
        dict with trend analysis, resistance_velocity, risk_level

    Used by: Agent 3 (Trend Analyst)
    """
```

**Logic:**
```python
# Calculate resistance velocity
if len(historical_mics) >= 2:
    baseline_mic = historical_mics[0]["mic_value"]
    current_mic = historical_mics[-1]["mic_value"]

    ratio = current_mic / baseline_mic

    if ratio >= 4:  # Two-step dilution increase
        risk_level = "HIGH"
        alert = "MIC Creep Detected - Risk of Treatment Failure"
    elif ratio >= 2:
        risk_level = "MODERATE"
        alert = "MIC Trending Upward - Monitor Closely"
    else:
        risk_level = "LOW"
        alert = None
```

---

## Part 4: Workflow Integration

### Stage 1: Empirical Phase (Before Lab Results)

```
Input: Patient history, symptoms, infection site
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 1: Intake Historian (MedGemma 1.5)               │
│  ├── Tool: search_clinical_guidelines()                 │
│  │   └── ChromaDB: idsa_treatment_guidelines            │
│  ├── Tool: query_resistance_pattern()                   │
│  │   └── SQLite: atlas_susceptibility_percent           │
│  └── Tool: query_antibiotic_info()                      │
│      └── SQLite: eml_antibiotics                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 4: Clinical Pharmacologist (TxGemma)             │
│  ├── Tool: check_drug_interactions()                    │
│  │   └── SQLite: drug_interactions                      │
│  └── Tool: query_antibiotic_info() [dosing]             │
│      └── SQLite: eml_antibiotics + dosage_guidance      │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output: Empirical therapy recommendation with safety check
```

### Stage 2: Targeted Phase (After Lab Results)

```
Input: Lab report (antibiogram image/PDF)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 2: Vision Specialist (MedGemma 4B)               │
│  ├── Extract: Pathogen name, MIC values                 │
│  └── Tool: interpret_mic_value()                        │
│      └── SQLite: mic_breakpoints                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 3: Trend Analyst (MedGemma 27B)                  │
│  ├── Tool: calculate_mic_trend()                        │
│  │   └── Patient historical data + current MIC          │
│  └── Tool: query_resistance_pattern()                   │
│      └── SQLite: atlas_susceptibility (population data) │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Agent 4: Clinical Pharmacologist (TxGemma)             │
│  ├── Tool: search_clinical_guidelines()                 │
│  │   └── ChromaDB: idsa_treatment_guidelines            │
│  ├── Tool: check_drug_interactions()                    │
│  │   └── SQLite: drug_interactions                      │
│  └── Generate: Final prescription with justification    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Output: Targeted therapy with MIC trend analysis & safety alerts
```

---

## Part 5: Implementation Checklist

### SQLite Setup
- [ ] Create database schema with all tables
- [ ] Import EML Excel files (ACCESS, RESERVE, WATCH)
- [ ] Import ATLAS susceptibility data (both sheets)
- [ ] Import MIC breakpoint tables (41 sheets)
- [ ] Import drug interactions CSV
- [ ] Add severity classification to interactions
- [ ] Create indexes for efficient queries

### ChromaDB Setup
- [ ] Initialize ChromaDB persistent storage
- [ ] Process ciae403.pdf with chunking strategy
- [ ] Process MIC breakpoint PDF
- [ ] Add metadata to all chunks
- [ ] Test semantic search queries

### Tool Implementation
- [ ] Implement `query_antibiotic_info()`
- [ ] Implement `query_resistance_pattern()`
- [ ] Implement `interpret_mic_value()`
- [ ] Implement `check_drug_interactions()`
- [ ] Implement `search_clinical_guidelines()`
- [ ] Implement `calculate_mic_trend()`
- [ ] Create unified tool interface for LangGraph

---

## File Structure

```
Med-I-C/
├── docs/                          # Source documents
├── data/
│   ├── medic.db                   # SQLite database
│   └── chroma/                    # ChromaDB persistent storage
├── src/
│   ├── db/
│   │   ├── schema.sql             # Database schema
│   │   └── import_data.py         # Data import scripts
│   ├── tools/
│   │   ├── antibiotic_tools.py    # query_antibiotic_info, interpret_mic
│   │   ├── resistance_tools.py   # query_resistance_pattern, calculate_mic_trend
│   │   ├── safety_tools.py       # check_drug_interactions
│   │   └── rag_tools.py          # search_clinical_guidelines
│   └── agents/
│       ├── intake_historian.py    # Agent 1
│       ├── vision_specialist.py   # Agent 2
│       ├── trend_analyst.py       # Agent 3
│       └── clinical_pharmacologist.py  # Agent 4
└── KNOWLEDGE_STORAGE_STRATEGY.md  # This document
```
