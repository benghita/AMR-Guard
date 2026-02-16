"""Med-I-C Query Tools for AMR-Guard Workflow."""

from .antibiotic_tools import (
    query_antibiotic_info,
    get_antibiotics_by_category,
    get_antibiotic_for_indication,
    interpret_mic_value,
    get_breakpoints_for_pathogen,
)

from .resistance_tools import (
    query_resistance_pattern,
    get_most_effective_antibiotics,
    get_resistance_trend,
    calculate_mic_trend,
    get_pathogen_families,
    get_pathogens_by_family,
)

from .safety_tools import (
    check_drug_interactions,
    check_single_interaction,
    get_all_interactions_for_drug,
    get_major_interactions_for_drug,
    screen_antibiotic_safety,
    get_interaction_statistics,
)

from .rag_tools import (
    search_clinical_guidelines,
    search_mic_reference_docs,
    get_treatment_recommendation,
    explain_mic_interpretation,
    get_empirical_therapy_guidance,
)

__all__ = [
    # Antibiotic tools
    "query_antibiotic_info",
    "get_antibiotics_by_category",
    "get_antibiotic_for_indication",
    "interpret_mic_value",
    "get_breakpoints_for_pathogen",

    # Resistance tools
    "query_resistance_pattern",
    "get_most_effective_antibiotics",
    "get_resistance_trend",
    "calculate_mic_trend",
    "get_pathogen_families",
    "get_pathogens_by_family",

    # Safety tools
    "check_drug_interactions",
    "check_single_interaction",
    "get_all_interactions_for_drug",
    "get_major_interactions_for_drug",
    "screen_antibiotic_safety",
    "get_interaction_statistics",

    # RAG tools
    "search_clinical_guidelines",
    "search_mic_reference_docs",
    "get_treatment_recommendation",
    "explain_mic_interpretation",
    "get_empirical_therapy_guidance",
]
