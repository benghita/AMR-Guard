-- Med-I-C Database Schema
-- AMR-Guard: Infection Lifecycle Orchestrator

-- EML Antibiotic Classification Table
CREATE TABLE IF NOT EXISTS eml_antibiotics (
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

CREATE INDEX IF NOT EXISTS idx_eml_medicine_name ON eml_antibiotics(medicine_name);
CREATE INDEX IF NOT EXISTS idx_eml_who_category ON eml_antibiotics(who_category);
CREATE INDEX IF NOT EXISTS idx_eml_atc_codes ON eml_antibiotics(atc_codes);

-- ATLAS Susceptibility Data (Percent)
CREATE TABLE IF NOT EXISTS atlas_susceptibility (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    species TEXT,
    family TEXT,
    antibiotic TEXT,
    percent_susceptible REAL,
    percent_intermediate REAL,
    percent_resistant REAL,
    total_isolates INTEGER,
    year INTEGER,
    region TEXT,
    source TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_atlas_species ON atlas_susceptibility(species);
CREATE INDEX IF NOT EXISTS idx_atlas_antibiotic ON atlas_susceptibility(antibiotic);
CREATE INDEX IF NOT EXISTS idx_atlas_family ON atlas_susceptibility(family);

-- MIC Breakpoints Table
CREATE TABLE IF NOT EXISTS mic_breakpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    pathogen_group TEXT NOT NULL,
    antibiotic TEXT NOT NULL,
    route TEXT,
    mic_susceptible REAL,
    mic_resistant REAL,
    disk_susceptible REAL,
    disk_resistant REAL,
    notes TEXT,
    eucast_version TEXT DEFAULT '16.0',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_bp_pathogen ON mic_breakpoints(pathogen_group);
CREATE INDEX IF NOT EXISTS idx_bp_antibiotic ON mic_breakpoints(antibiotic);

-- Dosage Guidance Table
CREATE TABLE IF NOT EXISTS dosage_guidance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    antibiotic TEXT NOT NULL,
    standard_dose TEXT,
    high_dose TEXT,
    renal_adjustment TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_dosage_antibiotic ON dosage_guidance(antibiotic);

-- Drug Interactions Table
CREATE TABLE IF NOT EXISTS drug_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    drug_1 TEXT NOT NULL,
    drug_2 TEXT NOT NULL,
    interaction_description TEXT,
    severity TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_di_drug_1 ON drug_interactions(drug_1);
CREATE INDEX IF NOT EXISTS idx_di_drug_2 ON drug_interactions(drug_2);
CREATE INDEX IF NOT EXISTS idx_di_severity ON drug_interactions(severity);

-- View for bidirectional drug interaction lookup
CREATE VIEW IF NOT EXISTS drug_interaction_lookup AS
SELECT id, drug_1, drug_2, interaction_description, severity FROM drug_interactions
UNION ALL
SELECT id, drug_2 as drug_1, drug_1 as drug_2, interaction_description, severity FROM drug_interactions;

-- Patient History Table (for demo purposes)
CREATE TABLE IF NOT EXISTS patient_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id TEXT NOT NULL,
    infection_date DATE,
    pathogen TEXT,
    antibiotic TEXT,
    mic_value REAL,
    interpretation TEXT,
    outcome TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_ph_patient ON patient_history(patient_id);
CREATE INDEX IF NOT EXISTS idx_ph_pathogen ON patient_history(pathogen);
