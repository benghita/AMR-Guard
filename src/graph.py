"""
LangGraph Orchestrator for Med-I-C Multi-Agent System.

Implements the infection lifecycle workflow with conditional routing:

Stage 1 (Empirical - no lab results):
    Intake Historian -> Clinical Pharmacologist

Stage 2 (Targeted - lab results available):
    Intake Historian -> Vision Specialist -> Trend Analyst -> Clinical Pharmacologist
"""

from __future__ import annotations

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


# =============================================================================
# NODE FUNCTIONS (Wrapper for agents)
# =============================================================================

def intake_historian_node(state: InfectionState) -> InfectionState:
    """Node 1: Run Intake Historian agent."""
    logger.info("Graph: Executing Intake Historian node")
    return run_intake_historian(state)


def vision_specialist_node(state: InfectionState) -> InfectionState:
    """Node 2: Run Vision Specialist agent."""
    logger.info("Graph: Executing Vision Specialist node")
    return run_vision_specialist(state)


def trend_analyst_node(state: InfectionState) -> InfectionState:
    """Node 3: Run Trend Analyst agent."""
    logger.info("Graph: Executing Trend Analyst node")
    return run_trend_analyst(state)


def clinical_pharmacologist_node(state: InfectionState) -> InfectionState:
    """Node 4: Run Clinical Pharmacologist agent."""
    logger.info("Graph: Executing Clinical Pharmacologist node")
    return run_clinical_pharmacologist(state)


# =============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# =============================================================================

def route_after_intake(state: InfectionState) -> Literal["vision_specialist", "clinical_pharmacologist"]:
    """
    Determine routing after Intake Historian.

    Routes to Vision Specialist if:
    - stage is "targeted" AND
    - route_to_vision is True (i.e., we have lab data to process)

    Otherwise routes directly to Clinical Pharmacologist (empirical path).
    """
    stage = state.get("stage", "empirical")
    has_lab_data = state.get("route_to_vision", False)

    if stage == "targeted" and has_lab_data:
        logger.info("Graph: Routing to Vision Specialist (targeted path)")
        return "vision_specialist"
    else:
        logger.info("Graph: Routing to Clinical Pharmacologist (empirical path)")
        return "clinical_pharmacologist"


def route_after_vision(state: InfectionState) -> Literal["trend_analyst", "clinical_pharmacologist"]:
    """
    Determine routing after Vision Specialist.

    Routes to Trend Analyst if:
    - route_to_trend_analyst is True (i.e., we have MIC data to analyze)

    Otherwise skips to Clinical Pharmacologist.
    """
    should_analyze_trends = state.get("route_to_trend_analyst", False)

    if should_analyze_trends:
        logger.info("Graph: Routing to Trend Analyst")
        return "trend_analyst"
    else:
        logger.info("Graph: Skipping Trend Analyst, routing to Clinical Pharmacologist")
        return "clinical_pharmacologist"


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_infection_graph() -> StateGraph:
    """
    Build the LangGraph StateGraph for the infection lifecycle workflow.

    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph with InfectionState as the state schema
    graph = StateGraph(InfectionState)

    # Add nodes
    graph.add_node("intake_historian", intake_historian_node)
    graph.add_node("vision_specialist", vision_specialist_node)
    graph.add_node("trend_analyst", trend_analyst_node)
    graph.add_node("clinical_pharmacologist", clinical_pharmacologist_node)

    # Set entry point
    graph.set_entry_point("intake_historian")

    # Add conditional edges from intake_historian
    graph.add_conditional_edges(
        "intake_historian",
        route_after_intake,
        {
            "vision_specialist": "vision_specialist",
            "clinical_pharmacologist": "clinical_pharmacologist",
        }
    )

    # Add conditional edges from vision_specialist
    graph.add_conditional_edges(
        "vision_specialist",
        route_after_vision,
        {
            "trend_analyst": "trend_analyst",
            "clinical_pharmacologist": "clinical_pharmacologist",
        }
    )

    # Add edge from trend_analyst to clinical_pharmacologist
    graph.add_edge("trend_analyst", "clinical_pharmacologist")

    # Add edge from clinical_pharmacologist to END
    graph.add_edge("clinical_pharmacologist", END)

    return graph


def compile_graph():
    """
    Build and compile the graph for execution.

    Returns:
        Compiled graph that can be invoked with .invoke(state)
    """
    graph = build_infection_graph()
    return graph.compile()


# =============================================================================
# EXECUTION HELPERS
# =============================================================================

def run_pipeline(
    patient_data: dict,
    labs_raw_text: str | None = None,
) -> InfectionState:
    """
    Run the full infection lifecycle pipeline.

    This is the main entry point for executing the multi-agent workflow.

    Args:
        patient_data: Dict containing patient information:
            - age_years: Patient age
            - weight_kg: Patient weight
            - sex: "male" or "female"
            - serum_creatinine_mg_dl: Serum creatinine (optional)
            - medications: List of current medications
            - allergies: List of allergies
            - comorbidities: List of comorbidities
            - infection_site: Site of infection
            - suspected_source: Suspected pathogen/source

        labs_raw_text: Raw text from lab report (if available).
                      If provided, triggers targeted (Stage 2) pathway.

    Returns:
        Final InfectionState with recommendation

    Example:
        >>> state = run_pipeline(
        ...     patient_data={
        ...         "age_years": 65,
        ...         "weight_kg": 70,
        ...         "sex": "male",
        ...         "serum_creatinine_mg_dl": 1.2,
        ...         "medications": ["metformin", "lisinopril"],
        ...         "allergies": ["penicillin"],
        ...         "infection_site": "urinary",
        ...         "suspected_source": "community UTI",
        ...     },
        ...     labs_raw_text="E. coli isolated. Ciprofloxacin MIC: 0.5 mg/L (S)"
        ... )
        >>> print(state["recommendation"]["primary_antibiotic"])
    """
    # Build initial state from patient data
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
    }

    # Add lab data if provided
    if labs_raw_text:
        initial_state["labs_raw_text"] = labs_raw_text
        initial_state["stage"] = "targeted"
    else:
        initial_state["stage"] = "empirical"

    # Compile and run the graph
    logger.info(f"Starting pipeline execution (stage: {initial_state['stage']})")

    compiled_graph = compile_graph()
    final_state = compiled_graph.invoke(initial_state)

    logger.info("Pipeline execution complete")

    return final_state


def run_empirical_pipeline(patient_data: dict) -> InfectionState:
    """
    Run Stage 1 (Empirical) pipeline only.

    Shorthand for run_pipeline without lab data.
    """
    return run_pipeline(patient_data, labs_raw_text=None)


def run_targeted_pipeline(patient_data: dict, labs_raw_text: str) -> InfectionState:
    """
    Run Stage 2 (Targeted) pipeline with lab data.

    Shorthand for run_pipeline with lab data.
    """
    return run_pipeline(patient_data, labs_raw_text=labs_raw_text)


# =============================================================================
# VISUALIZATION (for debugging)
# =============================================================================

def get_graph_mermaid() -> str:
    """
    Get Mermaid diagram representation of the graph.

    Useful for documentation and debugging.
    """
    graph = build_infection_graph()
    try:
        return graph.compile().get_graph().draw_mermaid()
    except Exception:
        # Fallback: return manual diagram
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


__all__ = [
    "build_infection_graph",
    "compile_graph",
    "run_pipeline",
    "run_empirical_pipeline",
    "run_targeted_pipeline",
    "get_graph_mermaid",
]
