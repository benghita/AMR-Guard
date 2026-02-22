
from typing import Dict, List, Literal, NotRequired, Optional, TypedDict


class LabResult(TypedDict, total=False):
    """A single lab value with optional reference range and flag."""
    name: str
    value: str
    unit: NotRequired[Optional[str]]
    reference_range: NotRequired[Optional[str]]
    flag: NotRequired[Optional[Literal["low", "normal", "high", "critical"]]]


class MICDatum(TypedDict, total=False):
    """A single MIC measurement for one organism–antibiotic pair."""
    organism: str
    antibiotic: str
    mic_value: str
    mic_unit: NotRequired[Optional[str]]
    interpretation: NotRequired[Optional[Literal["S", "I", "R"]]]
    breakpoint_source: NotRequired[Optional[str]]  # e.g. "EUCAST v16.0"
    year: NotRequired[Optional[int]]
    site: NotRequired[Optional[str]]  # e.g. "blood", "urine"


class Recommendation(TypedDict, total=False):
    """Final clinical recommendation assembled by Agent 4."""
    primary_antibiotic: Optional[str]
    backup_antibiotic: NotRequired[Optional[str]]
    dose: Optional[str]
    route: Optional[str]
    frequency: Optional[str]
    duration: Optional[str]
    rationale: Optional[str]
    references: NotRequired[List[str]]
    safety_alerts: NotRequired[List[str]]


class InfectionState(TypedDict, total=False):
    """
    Shared state object passed between all agents in the pipeline.

    All keys are optional so each agent only needs to populate its own outputs.
    """

    # Patient identity & demographics
    patient_id: NotRequired[Optional[str]]
    age_years: NotRequired[Optional[float]]
    sex: NotRequired[Optional[Literal["male", "female", "other", "unknown"]]]
    weight_kg: NotRequired[Optional[float]]
    height_cm: NotRequired[Optional[float]]

    # Clinical context
    suspected_source: NotRequired[Optional[str]]  # e.g. "community UTI"
    comorbidities: NotRequired[List[str]]
    medications: NotRequired[List[str]]
    allergies: NotRequired[List[str]]
    infection_site: NotRequired[Optional[str]]
    country_or_region: NotRequired[Optional[str]]

    # Renal function & vitals
    serum_creatinine_mg_dl: NotRequired[Optional[float]]
    creatinine_clearance_ml_min: NotRequired[Optional[float]]
    vitals: NotRequired[Dict[str, str]]  # flexible key/value, e.g. {"BP": "120/80"}

    # Lab data & MICs
    labs_raw_text: NotRequired[Optional[str]]   # raw OCR or pasted text
    labs_image_bytes: NotRequired[Optional[bytes]]  # uploaded image (PNG/JPG/TIFF) for vision model
    labs_parsed: NotRequired[List[LabResult]]
    mic_data: NotRequired[List[MICDatum]]
    mic_trend_summary: NotRequired[Optional[str]]

    # Routing flags set by agents
    stage: NotRequired[Literal["empirical", "targeted"]]
    route_to_vision: NotRequired[bool]
    route_to_trend_analyst: NotRequired[bool]

    # Agent outputs
    intake_notes: NotRequired[Optional[str]]       # Agent 1
    vision_notes: NotRequired[Optional[str]]       # Agent 2
    trend_notes: NotRequired[Optional[str]]        # Agent 3
    pharmacology_notes: NotRequired[Optional[str]] # Agent 4
    recommendation: NotRequired[Optional[Recommendation]]

    # RAG context & safety
    rag_context: NotRequired[Optional[str]]
    guideline_sources: NotRequired[List[str]]
    breakpoint_sources: NotRequired[List[str]]
    safety_warnings: NotRequired[List[str]]

    # Diagnostics
    errors: NotRequired[List[str]]
    debug_log: NotRequired[List[str]]

