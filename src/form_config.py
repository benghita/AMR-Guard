"""
Declarative field definitions for the dynamic Patient Analysis form.

Each infection site maps to a list of site-specific fields and contextual
suspected-source options. Universal fields (age, sex, weight, height,
creatinine, medications, allergies, comorbidities, risk factors) are always
shown and are NOT listed here.
"""

SITE_SPECIFIC_FIELDS: dict[str, list[dict]] = {
    "urinary": [
        {
            "key": "catheter_status",
            "label": "Catheter status",
            "type": "selectbox",
            "options": [
                "No catheter",
                "Indwelling (Foley)",
                "Intermittent",
                "Suprapubic",
                "Recently removed (<48 h)",
            ],
        },
        {
            "key": "urinary_symptoms",
            "label": "Urinary symptoms",
            "type": "multiselect",
            "options": [
                "Dysuria",
                "Frequency",
                "Urgency",
                "Hematuria",
                "Suprapubic pain",
                "Flank pain",
                "Fever / chills",
            ],
        },
        {
            "key": "urine_appearance",
            "label": "Urine appearance",
            "type": "selectbox",
            "options": ["Clear", "Cloudy", "Turbid", "Malodorous", "Hematuria"],
        },
    ],
    "respiratory": [
        {
            "key": "o2_saturation",
            "label": "O\u2082 Saturation (%)",
            "type": "number_input",
            "min": 50.0,
            "max": 100.0,
            "default": 97.0,
            "step": 0.5,
        },
        {
            "key": "ventilation_status",
            "label": "Ventilation status",
            "type": "selectbox",
            "options": [
                "Room air",
                "Supplemental O\u2082 (nasal cannula)",
                "Supplemental O\u2082 (mask)",
                "Non-invasive (BiPAP / CPAP)",
                "Mechanical ventilation",
            ],
        },
        {
            "key": "cough_type",
            "label": "Cough type",
            "type": "selectbox",
            "options": ["None", "Dry", "Productive", "Hemoptysis"],
        },
        {
            "key": "sputum_character",
            "label": "Sputum character",
            "type": "selectbox",
            "options": [
                "None",
                "Clear / white",
                "Yellow",
                "Green / purulent",
                "Rust-colored",
                "Blood-tinged",
            ],
        },
    ],
    "bloodstream": [
        {
            "key": "central_line_present",
            "label": "Central line present",
            "type": "checkbox",
            "default": False,
        },
        {
            "key": "temperature_c",
            "label": "Temperature (\u00b0C)",
            "type": "number_input",
            "min": 34.0,
            "max": 43.0,
            "default": 38.5,
            "step": 0.1,
        },
        {
            "key": "heart_rate_bpm",
            "label": "Heart rate (bpm)",
            "type": "number_input",
            "min": 30,
            "max": 250,
            "default": 90,
            "step": 1,
        },
        {
            "key": "respiratory_rate",
            "label": "Respiratory rate (/min)",
            "type": "number_input",
            "min": 5,
            "max": 60,
            "default": 18,
            "step": 1,
        },
        {
            "key": "wbc_count",
            "label": "WBC count (\u00d710\u2079/L)",
            "type": "number_input",
            "min": 0.0,
            "max": 100.0,
            "default": 12.0,
            "step": 0.1,
        },
        {
            "key": "lactate_mmol",
            "label": "Lactate (mmol/L)",
            "type": "number_input",
            "min": 0.0,
            "max": 30.0,
            "default": 1.0,
            "step": 0.1,
        },
        {
            "key": "shock_status",
            "label": "Shock status",
            "type": "selectbox",
            "options": [
                "No shock",
                "Compensated (SBP > 90, tachycardia)",
                "Septic shock (vasopressors required)",
            ],
        },
    ],
    "skin": [
        {
            "key": "wound_type",
            "label": "Wound type",
            "type": "selectbox",
            "options": [
                "Laceration",
                "Ulcer (diabetic / pressure)",
                "Bite (animal / human)",
                "Surgical site",
                "Burn",
                "Abscess",
                "Cellulitis (no wound)",
            ],
        },
        {
            "key": "cellulitis_extent",
            "label": "Cellulitis extent",
            "type": "selectbox",
            "options": [
                "None",
                "Localized (< 5 cm)",
                "Moderate (5\u201310 cm)",
                "Extensive (> 10 cm)",
                "Rapidly spreading",
            ],
        },
        {
            "key": "abscess_present",
            "label": "Abscess present",
            "type": "checkbox",
            "default": False,
        },
        {
            "key": "foreign_body",
            "label": "Foreign body / implant",
            "type": "checkbox",
            "default": False,
        },
    ],
    "intra-abdominal": [
        {
            "key": "abdominal_pain_location",
            "label": "Pain location",
            "type": "selectbox",
            "options": [
                "Diffuse",
                "RUQ",
                "LUQ",
                "RLQ",
                "LLQ",
                "Epigastric",
                "Periumbilical",
            ],
        },
        {
            "key": "peritonitis_signs",
            "label": "Peritonitis signs",
            "type": "multiselect",
            "options": [
                "Guarding",
                "Rebound tenderness",
                "Rigidity",
                "Absent bowel sounds",
            ],
        },
        {
            "key": "perforation_suspected",
            "label": "Perforation suspected",
            "type": "checkbox",
            "default": False,
        },
        {
            "key": "ascites",
            "label": "Ascites present",
            "type": "checkbox",
            "default": False,
        },
    ],
    "CNS": [
        {
            "key": "csf_obtained",
            "label": "CSF obtained",
            "type": "checkbox",
            "default": False,
        },
        {
            "key": "neuro_symptoms",
            "label": "Neurological symptoms",
            "type": "multiselect",
            "options": [
                "Headache",
                "Neck stiffness",
                "Photophobia",
                "Altered mental status",
                "Seizures",
                "Focal deficits",
            ],
        },
        {
            "key": "recent_neurosurgery",
            "label": "Recent neurosurgery",
            "type": "checkbox",
            "default": False,
        },
        {
            "key": "gcs_score",
            "label": "GCS score",
            "type": "number_input",
            "min": 3,
            "max": 15,
            "default": 15,
            "step": 1,
        },
    ],
    "other": [],
}


# Sites where serum creatinine is shown prominently in demographics.
# For all other sites a "renal impairment?" toggle is shown instead.
CREATININE_PROMINENT_SITES: frozenset[str] = frozenset(
    {"urinary", "bloodstream", "CNS", "respiratory"}
)

SUSPECTED_SOURCE_OPTIONS: dict[str, list[str]] = {
    "urinary": [
        "Community-acquired UTI",
        "Catheter-associated UTI (CAUTI)",
        "Complicated UTI",
        "Pyelonephritis",
        "Urosepsis",
        "Other",
    ],
    "respiratory": [
        "Community-acquired pneumonia (CAP)",
        "Hospital-acquired pneumonia (HAP)",
        "Ventilator-associated pneumonia (VAP)",
        "Aspiration pneumonia",
        "Lung abscess",
        "Empyema",
        "Other",
    ],
    "bloodstream": [
        "Primary bacteremia",
        "Catheter-related BSI (CRBSI)",
        "Secondary bacteremia (from known source)",
        "Endocarditis",
        "Unknown source",
        "Other",
    ],
    "skin": [
        "Cellulitis",
        "Surgical site infection",
        "Diabetic foot infection",
        "Bite wound infection",
        "Necrotizing fasciitis",
        "Abscess",
        "Other",
    ],
    "intra-abdominal": [
        "Appendicitis",
        "Cholecystitis / cholangitis",
        "Diverticulitis",
        "Peritonitis (SBP)",
        "Post-surgical",
        "Liver abscess",
        "Other",
    ],
    "CNS": [
        "Community-acquired meningitis",
        "Post-neurosurgical meningitis",
        "Healthcare-associated ventriculitis",
        "Brain abscess",
        "Other",
    ],
    "other": [],
}
