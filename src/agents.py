"""
Multi-Agent System.

Implements the 4 specialized agents for the infection lifecycle workflow:
- Agent 1: Intake Historian - Parse patient data, risk factors, calculate CrCl
- Agent 2: Vision Specialist - Extract structured data from lab reports
- Agent 3: Trend Analyst - Detect MIC creep and resistance velocity
- Agent 4: Clinical Pharmacologist - Final Rx recommendations + safety checks
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from .config import get_settings
from .loader import run_inference, TextModelName
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


# =============================================================================
# AGENT 1: INTAKE HISTORIAN
# =============================================================================

def run_intake_historian(state: InfectionState) -> InfectionState:
    """
    Agent 1: Parse patient data, calculate CrCl, identify risk factors.

    Input state fields used:
        - age_years, weight_kg, height_cm, sex
        - serum_creatinine_mg_dl
        - medications, allergies, comorbidities
        - suspected_source, infection_site

    Output state fields updated:
        - creatinine_clearance_ml_min
        - intake_notes
        - stage (empirical/targeted)
        - route_to_vision
    """
    logger.info("Running Intake Historian agent...")

    # Calculate CrCl if we have the required data
    crcl = None
    if all([
        state.get("age_years"),
        state.get("weight_kg"),
        state.get("serum_creatinine_mg_dl"),
        state.get("sex"),
    ]):
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

    # Build patient data string for prompt
    patient_data = _format_patient_data(state)

    # Get RAG context
    query = f"treatment {state.get('suspected_source', '')} {state.get('infection_site', '')}"
    rag_context = get_context_for_agent(
        agent_name="intake_historian",
        query=query,
        patient_context={
            "pathogen_type": state.get("suspected_source"),
        },
    )

    # Format the prompt
    prompt = f"{INTAKE_HISTORIAN_SYSTEM}\n\n{INTAKE_HISTORIAN_PROMPT.format(
        patient_data=patient_data,
        medications=', '.join(state.get('medications', [])) or 'None reported',
        allergies=', '.join(state.get('allergies', [])) or 'No known allergies',
        infection_site=state.get('infection_site', 'Unknown'),
        suspected_source=state.get('suspected_source', 'Unknown'),
        rag_context=rag_context,
    )}"

    # Run inference
    try:
        response = run_inference(
            prompt=prompt,
            model_name="medgemma_4b",
            max_new_tokens=1024,
            temperature=0.2,
        )

        # Parse response
        parsed = safe_json_parse(response)
        if parsed:
            state["intake_notes"] = json.dumps(parsed, indent=2)

            # Update state from parsed response
            if parsed.get("creatinine_clearance_ml_min") and crcl is None:
                state["creatinine_clearance_ml_min"] = parsed["creatinine_clearance_ml_min"]

            # Determine stage
            recommended_stage = parsed.get("recommended_stage", "empirical")
            state["stage"] = recommended_stage

            # Route to vision if we have lab data to process
            state["route_to_vision"] = bool(state.get("labs_raw_text"))
        else:
            state["intake_notes"] = response
            state["stage"] = "empirical"
            state["route_to_vision"] = bool(state.get("labs_raw_text"))

    except Exception as e:
        logger.error(f"Intake Historian error: {e}")
        state.setdefault("errors", []).append(f"Intake Historian error: {e}")
        state["intake_notes"] = f"Error: {e}"
        state["stage"] = "empirical"

    logger.info(f"Intake Historian complete. Stage: {state.get('stage')}")
    return state


# =============================================================================
# AGENT 2: VISION SPECIALIST
# =============================================================================

def run_vision_specialist(state: InfectionState) -> InfectionState:
    """
    Agent 2: Extract structured data from lab reports (text, images, PDFs).

    Input state fields used:
        - labs_raw_text (extracted text from lab report)

    Output state fields updated:
        - labs_parsed
        - mic_data
        - vision_notes
        - route_to_trend_analyst
    """
    logger.info("Running Vision Specialist agent...")

    labs_raw = state.get("labs_raw_text", "")
    if not labs_raw:
        logger.info("No lab data to process, skipping Vision Specialist")
        state["vision_notes"] = "No lab data provided"
        state["route_to_trend_analyst"] = False
        return state

    # Detect language (simplified - in production would use langdetect)
    language = "English (assumed)"

    # Get RAG context for lab interpretation
    rag_context = get_context_for_agent(
        agent_name="vision_specialist",
        query="culture sensitivity susceptibility interpretation",
        patient_context={},
    )

    # Format the prompt
    prompt = f"{VISION_SPECIALIST_SYSTEM}\n\n{VISION_SPECIALIST_PROMPT.format(
        report_content=labs_raw,
        source_format='text',
        language=language,
    )}"

    # Run inference
    try:
        response = run_inference(
            prompt=prompt,
            model_name="medgemma_4b",
            max_new_tokens=2048,
            temperature=0.1,
        )

        # Parse response
        parsed = safe_json_parse(response)
        if parsed:
            state["vision_notes"] = json.dumps(parsed, indent=2)

            # Extract organisms and susceptibility data
            organisms = parsed.get("identified_organisms", [])
            susceptibility = parsed.get("susceptibility_results", [])

            # Convert to MICDatum format
            mic_data = []
            for result in susceptibility:
                mic_datum = {
                    "organism": normalize_organism_name(result.get("organism", "")),
                    "antibiotic": normalize_antibiotic_name(result.get("antibiotic", "")),
                    "mic_value": str(result.get("mic_value", "")),
                    "mic_unit": result.get("mic_unit", "mg/L"),
                    "interpretation": result.get("interpretation"),
                }
                mic_data.append(mic_datum)

            state["mic_data"] = mic_data
            state["labs_parsed"] = [{
                "name": org.get("organism_name", "Unknown"),
                "value": org.get("colony_count", ""),
                "flag": "pathogen" if org.get("significance") == "pathogen" else None,
            } for org in organisms]

            # Route to trend analyst if we have MIC data
            state["route_to_trend_analyst"] = len(mic_data) > 0

            # Check for critical findings
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


# =============================================================================
# AGENT 3: TREND ANALYST
# =============================================================================

def run_trend_analyst(state: InfectionState) -> InfectionState:
    """
    Agent 3: Analyze MIC trends and detect resistance velocity.

    Input state fields used:
        - mic_data (current MIC readings)
        - Historical MIC data (if available)

    Output state fields updated:
        - mic_trend_summary
        - trend_notes
        - safety_warnings (if high risk detected)
    """
    logger.info("Running Trend Analyst agent...")

    mic_data = state.get("mic_data", [])
    if not mic_data:
        logger.info("No MIC data to analyze, skipping Trend Analyst")
        state["trend_notes"] = "No MIC data available for trend analysis"
        return state

    # For each organism-antibiotic pair, analyze trends
    trend_results = []

    for mic in mic_data:
        organism = mic.get("organism", "Unknown")
        antibiotic = mic.get("antibiotic", "Unknown")

        # Get RAG context for breakpoints
        rag_context = get_context_for_agent(
            agent_name="trend_analyst",
            query=f"breakpoint {organism} {antibiotic}",
            patient_context={
                "organism": organism,
                "antibiotic": antibiotic,
                "region": state.get("country_or_region"),
            },
        )

        # Format MIC history (in production, would pull from database)
        mic_history = [{"date": "current", "mic_value": mic.get("mic_value", "0")}]

        # Format prompt
        prompt = f"{TREND_ANALYST_SYSTEM}\n\n{TREND_ANALYST_PROMPT.format(
            organism=organism,
            antibiotic=antibiotic,
            mic_history=json.dumps(mic_history, indent=2),
            breakpoint_data=rag_context,
            resistance_context='Regional data not available',
        )}"

        try:
            response = run_inference(
                prompt=prompt,
                model_name="medgemma_27b",  # Agent 3: MedGemma 27B per PLAN.md (env maps to 4B on limited GPU)
                max_new_tokens=1024,
                temperature=0.2,
            )

            parsed = safe_json_parse(response)
            if parsed:
                trend_results.append(parsed)

                # Add safety warning if high/critical risk
                risk_level = parsed.get("risk_level", "LOW")
                if risk_level in ["HIGH", "CRITICAL"]:
                    warning = f"MIC trend alert for {organism}/{antibiotic}: {parsed.get('recommendation', 'Review needed')}"
                    state.setdefault("safety_warnings", []).append(warning)
            else:
                trend_results.append({"raw_response": response})

        except Exception as e:
            logger.error(f"Trend analysis error for {organism}/{antibiotic}: {e}")
            trend_results.append({"error": str(e)})

    # Summarize trends
    state["trend_notes"] = json.dumps(trend_results, indent=2)

    # Create summary
    high_risk_count = sum(1 for t in trend_results if t.get("risk_level") in ["HIGH", "CRITICAL"])
    state["mic_trend_summary"] = f"Analyzed {len(trend_results)} organism-antibiotic pairs. High-risk findings: {high_risk_count}"

    logger.info(f"Trend Analyst complete. {state['mic_trend_summary']}")
    return state


# =============================================================================
# AGENT 4: CLINICAL PHARMACOLOGIST
# =============================================================================

def run_clinical_pharmacologist(state: InfectionState) -> InfectionState:
    """
    Agent 4: Generate final antibiotic recommendation with safety checks.

    Input state fields used:
        - intake_notes, vision_notes, trend_notes
        - age_years, weight_kg, creatinine_clearance_ml_min
        - allergies, medications
        - infection_site, suspected_source

    Output state fields updated:
        - recommendation
        - pharmacology_notes
        - safety_warnings (additional alerts)
    """
    logger.info("Running Clinical Pharmacologist agent...")

    # Gather all previous agent outputs
    intake_summary = state.get("intake_notes", "No intake data")
    lab_results = state.get("vision_notes", "No lab data")
    trend_analysis = state.get("trend_notes", "No trend data")

    # Get RAG context
    query = f"treatment {state.get('suspected_source', '')} antibiotic recommendation"
    rag_context = get_context_for_agent(
        agent_name="clinical_pharmacologist",
        query=query,
        patient_context={
            "proposed_antibiotic": None,  # Will be determined by agent
        },
    )

    # Format prompt
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
        rag_context=rag_context,
    )}"

    try:
        response = run_inference(
            prompt=prompt,
            model_name="medgemma_4b",
            max_new_tokens=2048,
            temperature=0.2,
        )

        parsed = safe_json_parse(response)
        if parsed:
            state["pharmacology_notes"] = json.dumps(parsed, indent=2)

            # Build recommendation
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

            # Add alternative if provided
            alt = parsed.get("alternative_recommendation", {})
            if alt.get("antibiotic"):
                recommendation["backup_antibiotic"] = alt.get("antibiotic")

            state["recommendation"] = recommendation

            # Add safety alerts to state
            for alert in parsed.get("safety_alerts", []):
                if alert.get("level") in ["WARNING", "CRITICAL"]:
                    state.setdefault("safety_warnings", []).append(alert.get("message"))

            # Run TxGemma safety check (optional)
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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _format_patient_data(state: InfectionState) -> str:
    """Format patient data for prompt injection."""
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
        vitals_str = ", ".join(f"{k}: {v}" for k, v in state["vitals"].items())
        lines.append(f"Vitals: {vitals_str}")

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
    """
    Run TxGemma safety check (supplementary).

    TxGemma is used only for safety validation, not primary recommendations.
    """
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

        response = run_inference(
            prompt=prompt,
            model_name="txgemma_9b",  # Agent 4 safety: TxGemma 9B per PLAN.md (env maps to 2B on limited GPU)
            max_new_tokens=256,
            temperature=0.1,
        )

        return response

    except Exception as e:
        logger.warning(f"TxGemma safety check failed: {e}")
        return None


# =============================================================================
# AGENT REGISTRY
# =============================================================================

AGENTS = {
    "intake_historian": run_intake_historian,
    "vision_specialist": run_vision_specialist,
    "trend_analyst": run_trend_analyst,
    "clinical_pharmacologist": run_clinical_pharmacologist,
}


def run_agent(agent_name: str, state: InfectionState) -> InfectionState:
    """
    Run a specific agent by name.

    Args:
        agent_name: Name of the agent to run
        state: Current infection state

    Returns:
        Updated infection state
    """
    if agent_name not in AGENTS:
        raise ValueError(f"Unknown agent: {agent_name}")

    return AGENTS[agent_name](state)


__all__ = [
    "run_intake_historian",
    "run_vision_specialist",
    "run_trend_analyst",
    "run_clinical_pharmacologist",
    "run_agent",
    "AGENTS",
]
