"""
Four-agent pipeline for the infection lifecycle workflow.

Agent 1 - Intake Historian:     parse patient data, calculate CrCl, identify AMR risk factors
Agent 2 - Vision Specialist:    extract organisms and MIC values from lab reports
Agent 3 - Trend Analyst:        detect MIC creep and resistance velocity
Agent 4 - Clinical Pharmacologist: generate final antibiotic recommendation with safety checks
"""

import json
import logging
from typing import Optional

from .config import get_settings
from .loader import run_inference, run_inference_with_image, TextModelName
from .prompts import (
    INTAKE_HISTORIAN_SYSTEM,
    INTAKE_HISTORIAN_PROMPT,
    VISION_SPECIALIST_SYSTEM,
    VISION_SPECIALIST_PROMPT,
    TREND_ANALYST_SYSTEM,
    TREND_ANALYST_PROMPT,
    CLINICAL_PHARMACOLOGIST_SYSTEM,
    CLINICAL_PHARMACOLOGIST_PROMPT,
    TXGEMMA_SAFETY_PROMPT,
)
from .rag import get_context_for_agent
from .state import InfectionState
from .utils import (
    calculate_crcl,
    get_renal_dose_category,
    safe_json_parse,
    normalize_organism_name,
    normalize_antibiotic_name,
)

logger = logging.getLogger(__name__)


def run_intake_historian(state: InfectionState) -> InfectionState:
    """Parse patient data, calculate CrCl, identify MDR risk factors, and set the treatment stage."""
    logger.info("Running Intake Historian agent...")

    crcl = None
    if all([state.get("age_years"), state.get("weight_kg"), state.get("serum_creatinine_mg_dl"), state.get("sex")]):
        try:
            crcl = calculate_crcl(
                age_years=state["age_years"],
                weight_kg=state["weight_kg"],
                serum_creatinine_mg_dl=state["serum_creatinine_mg_dl"],
                sex=state["sex"],
                use_ibw=True,
                height_cm=state.get("height_cm"),
            )
            state["creatinine_clearance_ml_min"] = crcl
            logger.info(f"Calculated CrCl: {crcl} mL/min")
        except Exception as e:
            logger.warning(f"Could not calculate CrCl: {e}")
            state.setdefault("errors", []).append(f"CrCl calculation error: {e}")

    patient_data = _format_patient_data(state)
    query = f"treatment {state.get('suspected_source', '')} {state.get('infection_site', '')}"
    rag_context = get_context_for_agent(
        agent_name="intake_historian",
        query=query,
        patient_context={"pathogen_type": state.get("suspected_source")},
    )

    site_vitals_str = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}" for k, v in state.get("vitals", {}).items()
    ) or "None provided"

    prompt = f"{INTAKE_HISTORIAN_SYSTEM}\n\n{INTAKE_HISTORIAN_PROMPT.format(
        patient_data=patient_data,
        medications=', '.join(state.get('medications', [])) or 'None reported',
        allergies=', '.join(state.get('allergies', [])) or 'No known allergies',
        infection_site=state.get('infection_site', 'Unknown'),
        suspected_source=state.get('suspected_source', 'Unknown'),
        site_vitals=site_vitals_str,
        rag_context=rag_context,
    )}"

    try:
        response = run_inference(prompt=prompt, model_name="medgemma_4b", max_new_tokens=1024, temperature=0.2)
        parsed = safe_json_parse(response)
        if parsed:
            state["intake_notes"] = json.dumps(parsed, indent=2)
            if parsed.get("creatinine_clearance_ml_min") and crcl is None:
                state["creatinine_clearance_ml_min"] = parsed["creatinine_clearance_ml_min"]
            state["stage"] = parsed.get("recommended_stage", "empirical")
        else:
            state["intake_notes"] = response
            state["stage"] = "empirical"

        # Route to vision only if lab text was provided
        state["route_to_vision"] = bool(state.get("labs_raw_text"))

    except Exception as e:
        logger.error(f"Intake Historian error: {e}")
        state.setdefault("errors", []).append(f"Intake Historian error: {e}")
        state["intake_notes"] = f"Error: {e}"
        state["stage"] = "empirical"

    logger.info(f"Intake Historian complete. Stage: {state.get('stage')}")
    return state


def run_vision_specialist(state: InfectionState) -> InfectionState:
    """Extract pathogen names, MIC values, and S/I/R interpretations from lab report text."""
    logger.info("Running Vision Specialist agent...")

    labs_raw = state.get("labs_raw_text", "")
    labs_image_bytes = state.get("labs_image_bytes")

    if not labs_raw and not labs_image_bytes:
        logger.info("No lab data to process, skipping Vision Specialist")
        state["vision_notes"] = "No lab data provided"
        state["route_to_trend_analyst"] = False
        return state

    # Determine input modality and prepare prompt content description
    if labs_image_bytes:
        source_format = "image"
        language = "Auto-detected"
        report_content = "See attached image — extract all lab data visible in the image."
    else:
        source_format = "text"
        language = "English (assumed)"
        report_content = labs_raw

    rag_context = get_context_for_agent(
        agent_name="vision_specialist",
        query="culture sensitivity susceptibility interpretation",
        patient_context={},
    )

    prompt = f"{VISION_SPECIALIST_SYSTEM}\n\n{VISION_SPECIALIST_PROMPT.format(
        report_content=report_content,
        source_format=source_format,
        language=language,
    )}"

    try:
        if labs_image_bytes:
            from io import BytesIO
            from PIL import Image as PILImage
            image = PILImage.open(BytesIO(labs_image_bytes)).convert("RGB")
            logger.info(f"Running vision inference on uploaded image ({image.size})")
            response = run_inference_with_image(
                prompt=prompt, image=image, model_name="medgemma_4b", max_new_tokens=2048, temperature=0.1
            )
        else:
            response = run_inference(prompt=prompt, model_name="medgemma_4b", max_new_tokens=2048, temperature=0.1)
        parsed = safe_json_parse(response)
        if parsed:
            state["vision_notes"] = json.dumps(parsed, indent=2)

            organisms = parsed.get("identified_organisms", [])
            susceptibility = parsed.get("susceptibility_results", [])

            mic_data = [
                {
                    "organism": normalize_organism_name(r.get("organism", "")),
                    "antibiotic": normalize_antibiotic_name(r.get("antibiotic", "")),
                    "mic_value": str(r.get("mic_value", "")),
                    "mic_unit": r.get("mic_unit", "mg/L"),
                    "interpretation": r.get("interpretation"),
                }
                for r in susceptibility
            ]

            state["mic_data"] = mic_data
            state["labs_parsed"] = [
                {
                    "name": org.get("organism_name", "Unknown"),
                    "value": org.get("colony_count", ""),
                    "flag": "pathogen" if org.get("significance") == "pathogen" else None,
                }
                for org in organisms
            ]
            state["route_to_trend_analyst"] = len(mic_data) > 0

            critical = parsed.get("critical_findings", [])
            if critical:
                state.setdefault("safety_warnings", []).extend(critical)
        else:
            state["vision_notes"] = response
            state["route_to_trend_analyst"] = False

    except Exception as e:
        logger.error(f"Vision Specialist error: {e}")
        state.setdefault("errors", []).append(f"Vision Specialist error: {e}")
        state["vision_notes"] = f"Error: {e}"
        state["route_to_trend_analyst"] = False

    logger.info(f"Vision Specialist complete. MIC data points: {len(state.get('mic_data', []))}")
    return state


def run_trend_analyst(state: InfectionState) -> InfectionState:
    """Analyze MIC trends per organism-antibiotic pair and flag high-risk creep."""
    logger.info("Running Trend Analyst agent...")

    mic_data = state.get("mic_data", [])
    if not mic_data:
        logger.info("No MIC data to analyze, skipping Trend Analyst")
        state["trend_notes"] = "No MIC data available for trend analysis"
        return state

    trend_results = []

    for mic in mic_data:
        organism = mic.get("organism", "Unknown")
        antibiotic = mic.get("antibiotic", "Unknown")

        rag_context = get_context_for_agent(
            agent_name="trend_analyst",
            query=f"breakpoint {organism} {antibiotic}",
            patient_context={
                "organism": organism,
                "antibiotic": antibiotic,
                "region": state.get("country_or_region"),
            },
        )

        # Single time-point history — trend analysis requires historical data in production
        mic_history = [{"date": "current", "mic_value": mic.get("mic_value", "0")}]

        prompt = f"{TREND_ANALYST_SYSTEM}\n\n{TREND_ANALYST_PROMPT.format(
            organism=organism,
            antibiotic=antibiotic,
            mic_history=json.dumps(mic_history, indent=2),
            breakpoint_data=rag_context,
            resistance_context='Regional data not available',
        )}"

        try:
            # Agent 3 is designed for MedGemma 27B; on limited GPU the env var maps this to 4B
            response = run_inference(
                prompt=prompt,
                model_name="medgemma_27b",
                max_new_tokens=1024,
                temperature=0.2,
            )
            parsed = safe_json_parse(response)
            if parsed:
                trend_results.append(parsed)
                risk_level = parsed.get("risk_level", "LOW")
                if risk_level in ["HIGH", "CRITICAL"]:
                    warning = f"MIC trend alert for {organism}/{antibiotic}: {parsed.get('recommendation', 'Review needed')}"
                    state.setdefault("safety_warnings", []).append(warning)
            else:
                trend_results.append({"raw_response": response})

        except Exception as e:
            logger.error(f"Trend analysis error for {organism}/{antibiotic}: {e}")
            trend_results.append({"error": str(e)})

    state["trend_notes"] = json.dumps(trend_results, indent=2)

    high_risk_count = sum(1 for t in trend_results if t.get("risk_level") in ["HIGH", "CRITICAL"])
    state["mic_trend_summary"] = f"Analyzed {len(trend_results)} organism-antibiotic pairs. High-risk findings: {high_risk_count}"

    logger.info(f"Trend Analyst complete. {state['mic_trend_summary']}")
    return state


def run_clinical_pharmacologist(state: InfectionState) -> InfectionState:
    """Synthesize all agent outputs into a final antibiotic recommendation with safety checks."""
    logger.info("Running Clinical Pharmacologist agent...")

    intake_summary = state.get("intake_notes", "No intake data")
    lab_results = state.get("vision_notes", "No lab data")
    trend_analysis = state.get("trend_notes", "No trend data")

    query = f"treatment {state.get('suspected_source', '')} antibiotic recommendation"
    rag_context = get_context_for_agent(
        agent_name="clinical_pharmacologist",
        query=query,
        patient_context={"proposed_antibiotic": None},
    )

    site_vitals_str = "\n".join(
        f"- {k.replace('_', ' ').title()}: {v}" for k, v in state.get("vitals", {}).items()
    ) or "None provided"

    prompt = f"{CLINICAL_PHARMACOLOGIST_SYSTEM}\n\n{CLINICAL_PHARMACOLOGIST_PROMPT.format(
        intake_summary=intake_summary,
        lab_results=lab_results,
        trend_analysis=trend_analysis,
        age=state.get('age_years', 'Unknown'),
        weight=state.get('weight_kg', 'Unknown'),
        crcl=state.get('creatinine_clearance_ml_min', 'Unknown'),
        allergies=', '.join(state.get('allergies', [])) or 'No known allergies',
        current_medications=', '.join(state.get('medications', [])) or 'None reported',
        infection_site=state.get('infection_site', 'Unknown'),
        suspected_source=state.get('suspected_source', 'Unknown'),
        severity=state.get('intake_notes', {}).get('infection_severity', 'Unknown') if isinstance(state.get('intake_notes'), dict) else 'Unknown',
        site_vitals=site_vitals_str,
        rag_context=rag_context,
    )}"

    try:
        response = run_inference(prompt=prompt, model_name="medgemma_4b", max_new_tokens=2048, temperature=0.2)
        parsed = safe_json_parse(response)
        if parsed:
            state["pharmacology_notes"] = json.dumps(parsed, indent=2)

            primary = parsed.get("primary_recommendation", {})
            recommendation = {
                "primary_antibiotic": primary.get("antibiotic"),
                "dose": primary.get("dose"),
                "route": primary.get("route"),
                "frequency": primary.get("frequency"),
                "duration": primary.get("duration"),
                "rationale": parsed.get("rationale"),
                "references": parsed.get("guideline_references", []),
                "safety_alerts": [a.get("message") for a in parsed.get("safety_alerts", [])],
            }

            alt = parsed.get("alternative_recommendation", {})
            if alt.get("antibiotic"):
                recommendation["backup_antibiotic"] = alt.get("antibiotic")

            state["recommendation"] = recommendation

            for alert in parsed.get("safety_alerts", []):
                if alert.get("level") in ["WARNING", "CRITICAL"]:
                    state.setdefault("safety_warnings", []).append(alert.get("message"))

            if primary.get("antibiotic"):
                safety_result = _run_txgemma_safety_check(
                    antibiotic=primary.get("antibiotic"),
                    dose=primary.get("dose"),
                    route=primary.get("route"),
                    duration=primary.get("duration"),
                    age=state.get("age_years"),
                    crcl=state.get("creatinine_clearance_ml_min"),
                    medications=state.get("medications", []),
                )
                if safety_result:
                    state.setdefault("debug_log", []).append(f"TxGemma safety: {safety_result}")
        else:
            state["pharmacology_notes"] = response
            state["recommendation"] = {"rationale": response}

    except Exception as e:
        logger.error(f"Clinical Pharmacologist error: {e}")
        state.setdefault("errors", []).append(f"Clinical Pharmacologist error: {e}")
        state["pharmacology_notes"] = f"Error: {e}"

    logger.info("Clinical Pharmacologist complete.")
    return state


def _format_patient_data(state: InfectionState) -> str:
    """Format patient fields from state into a readable string for prompt injection."""
    lines = []

    if state.get("patient_id"):
        lines.append(f"Patient ID: {state['patient_id']}")

    demographics = []
    if state.get("age_years"):
        demographics.append(f"{state['age_years']} years old")
    if state.get("sex"):
        demographics.append(state["sex"])
    if demographics:
        lines.append(f"Demographics: {', '.join(demographics)}")

    if state.get("weight_kg"):
        lines.append(f"Weight: {state['weight_kg']} kg")
    if state.get("height_cm"):
        lines.append(f"Height: {state['height_cm']} cm")

    if state.get("serum_creatinine_mg_dl"):
        lines.append(f"Serum Creatinine: {state['serum_creatinine_mg_dl']} mg/dL")
    if state.get("creatinine_clearance_ml_min"):
        crcl = state["creatinine_clearance_ml_min"]
        category = get_renal_dose_category(crcl)
        lines.append(f"CrCl: {crcl} mL/min ({category})")

    if state.get("comorbidities"):
        lines.append(f"Comorbidities: {', '.join(state['comorbidities'])}")

    if state.get("vitals"):
        lines.append("Site-Specific Assessment:")
        for k, v in state["vitals"].items():
            label = k.replace("_", " ").title()
            lines.append(f"  - {label}: {v}")

    return "\n".join(lines) if lines else "No patient data available"


def _run_txgemma_safety_check(
    antibiotic: str,
    dose: Optional[str],
    route: Optional[str],
    duration: Optional[str],
    age: Optional[float],
    crcl: Optional[float],
    medications: list,
) -> Optional[str]:
    """Run a supplementary TxGemma toxicology check on the proposed prescription."""
    try:
        prompt = TXGEMMA_SAFETY_PROMPT.format(
            antibiotic=antibiotic,
            dose=dose or "Not specified",
            route=route or "Not specified",
            duration=duration or "Not specified",
            age=age or "Unknown",
            crcl=crcl or "Unknown",
            medications=", ".join(medications) if medications else "None",
        )
        # Agent 4 safety check uses TxGemma 9B; on limited GPU the env var maps this to 2B
        return run_inference(prompt=prompt, model_name="txgemma_9b", max_new_tokens=256, temperature=0.1)
    except Exception as e:
        logger.warning(f"TxGemma safety check failed: {e}")
        return None


AGENTS = {
    "intake_historian": run_intake_historian,
    "vision_specialist": run_vision_specialist,
    "trend_analyst": run_trend_analyst,
    "clinical_pharmacologist": run_clinical_pharmacologist,
}


def run_agent(agent_name: str, state: InfectionState) -> InfectionState:
    """Dispatch to a named agent."""
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}")
    return AGENTS[agent_name](state)
