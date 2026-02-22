"""
LangGraph orchestrator for the infection lifecycle workflow.

Stage 1 (empirical - no lab results):
    Intake Historian → Clinical Pharmacologist

Stage 2 (targeted - lab results available):
    Intake Historian → Vision Specialist → [Trend Analyst →] Clinical Pharmacologist
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from .agents import (
    run_intake_historian,
    run_vision_specialist,
    run_trend_analyst,
    run_clinical_pharmacologist,
)
from .state import InfectionState

logger = logging.getLogger(__name__)


def route_after_intake(state: InfectionState) -> Literal["vision_specialist", "clinical_pharmacologist"]:
    """Route to Vision Specialist if we have lab text to parse; otherwise go straight to pharmacologist."""
    if state.get("stage") == "targeted" and state.get("route_to_vision"):
        logger.info("Graph: routing to Vision Specialist (targeted path)")
        return "vision_specialist"
    logger.info("Graph: routing to Clinical Pharmacologist (empirical path)")
    return "clinical_pharmacologist"


def route_after_vision(state: InfectionState) -> Literal["trend_analyst", "clinical_pharmacologist"]:
    """Route to Trend Analyst if Vision Specialist extracted MIC values."""
    if state.get("route_to_trend_analyst"):
        logger.info("Graph: routing to Trend Analyst")
        return "trend_analyst"
    logger.info("Graph: skipping Trend Analyst (no MIC data)")
    return "clinical_pharmacologist"


def build_infection_graph() -> StateGraph:
    """Build and return the compiled LangGraph for the infection pipeline."""
    graph = StateGraph(InfectionState)

    graph.add_node("intake_historian", run_intake_historian)
    graph.add_node("vision_specialist", run_vision_specialist)
    graph.add_node("trend_analyst", run_trend_analyst)
    graph.add_node("clinical_pharmacologist", run_clinical_pharmacologist)

    graph.set_entry_point("intake_historian")

    graph.add_conditional_edges(
        "intake_historian",
        route_after_intake,
        {"vision_specialist": "vision_specialist", "clinical_pharmacologist": "clinical_pharmacologist"},
    )
    graph.add_conditional_edges(
        "vision_specialist",
        route_after_vision,
        {"trend_analyst": "trend_analyst", "clinical_pharmacologist": "clinical_pharmacologist"},
    )

    graph.add_edge("trend_analyst", "clinical_pharmacologist")
    graph.add_edge("clinical_pharmacologist", END)

    return graph


def run_pipeline(patient_data: dict, labs_raw_text: str | None = None) -> InfectionState:
    """
    Run the full infection pipeline and return the final state.

    Pass labs_raw_text to trigger the targeted (Stage 2) pathway.
    Without it, only the empirical (Stage 1) pathway runs.
    """
    labs_image_bytes: bytes | None = patient_data.get("labs_image_bytes")
    has_lab_input = bool(labs_raw_text or labs_image_bytes)

    initial_state: InfectionState = {
        "age_years": patient_data.get("age_years"),
        "weight_kg": patient_data.get("weight_kg"),
        "height_cm": patient_data.get("height_cm"),
        "sex": patient_data.get("sex"),
        "serum_creatinine_mg_dl": patient_data.get("serum_creatinine_mg_dl"),
        "medications": patient_data.get("medications", []),
        "allergies": patient_data.get("allergies", []),
        "comorbidities": patient_data.get("comorbidities", []),
        "infection_site": patient_data.get("infection_site"),
        "suspected_source": patient_data.get("suspected_source"),
        "country_or_region": patient_data.get("country_or_region"),
        "vitals": patient_data.get("vitals", {}),
        "stage": "targeted" if has_lab_input else "empirical",
        "errors": [],
        "safety_warnings": [],
    }

    if labs_raw_text:
        initial_state["labs_raw_text"] = labs_raw_text
    if labs_image_bytes:
        initial_state["labs_image_bytes"] = labs_image_bytes

    logger.info(f"Starting pipeline (stage: {initial_state['stage']}, lab_text={bool(labs_raw_text)}, lab_image={bool(labs_image_bytes)})")
    logger.info(f"Patient: {patient_data.get('age_years')}y, {patient_data.get('sex')}, infection: {patient_data.get('infection_site')}")
    
    try:
        compiled = build_infection_graph().compile()
        logger.info("Graph compiled successfully")
        final_state = compiled.invoke(initial_state)
        logger.info("Pipeline complete")
        return final_state
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        initial_state["errors"].append(f"Pipeline error: {str(e)}")
        return initial_state


def run_empirical_pipeline(patient_data: dict) -> InfectionState:
    """Shorthand for run_pipeline without lab data (Stage 1)."""
    return run_pipeline(patient_data)


def run_targeted_pipeline(patient_data: dict, labs_raw_text: str) -> InfectionState:
    """Shorthand for run_pipeline with lab data (Stage 2)."""
    return run_pipeline(patient_data, labs_raw_text=labs_raw_text)


def get_graph_mermaid() -> str:
    """Return a Mermaid diagram of the graph (for documentation and debugging)."""
    try:
        return build_infection_graph().compile().get_graph().draw_mermaid()
    except Exception:
        return """
graph TD
    A[intake_historian] --> B{route_after_intake}
    B -->|targeted + lab data| C[vision_specialist]
    B -->|empirical| E[clinical_pharmacologist]
    C --> D{route_after_vision}
    D -->|has MIC data| F[trend_analyst]
    D -->|no MIC data| E
    F --> E
    E --> G[END]
"""
