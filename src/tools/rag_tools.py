"""RAG tools for querying clinical guidelines via ChromaDB."""

from src.db.vector_store import search_guidelines, search_mic_reference


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
            Options: 'ESBL-E', 'CRE', 'CRAB', 'DTR-PA', 'S.maltophilia', 'AmpC-E', 'General'
        n_results: Number of results to return

    Returns:
        List of relevant guideline excerpts with metadata

    Used by: Agent 1 (Empirical), Agent 4 (Justification)
    """
    results = search_guidelines(query, n_results, pathogen_filter)

    # Format for agent consumption
    formatted = []
    for r in results:
        formatted.append({
            "content": r.get("content", ""),
            "pathogen_type": r.get("metadata", {}).get("pathogen_type", "General"),
            "source": r.get("metadata", {}).get("source", "IDSA Guidelines"),
            "relevance_score": 1 - r.get("distance", 1)  # Convert distance to similarity
        })

    return formatted


def search_mic_reference_docs(query: str, n_results: int = 3) -> list[dict]:
    """
    Search MIC breakpoint reference documentation.

    Args:
        query: Query about MIC interpretation or breakpoints
        n_results: Number of results to return

    Returns:
        List of relevant reference excerpts
    """
    results = search_mic_reference(query, n_results)

    formatted = []
    for r in results:
        formatted.append({
            "content": r.get("content", ""),
            "source": r.get("metadata", {}).get("source", "EUCAST Breakpoints"),
            "relevance_score": 1 - r.get("distance", 1)
        })

    return formatted


def get_treatment_recommendation(
    pathogen: str,
    infection_site: str = None,
    patient_factors: list[str] = None
) -> dict:
    """
    Get treatment recommendation by searching guidelines.

    Args:
        pathogen: Identified or suspected pathogen
        infection_site: Location of infection (e.g., "urinary", "respiratory")
        patient_factors: List of patient factors (e.g., ["renal impairment", "pregnancy"])

    Returns:
        Treatment recommendation with guideline citations
    """
    # Build comprehensive query
    query_parts = [f"treatment for {pathogen} infection"]

    if infection_site:
        query_parts.append(f"in {infection_site}")

    if patient_factors:
        query_parts.append(f"considering {', '.join(patient_factors)}")

    query = " ".join(query_parts)

    # Search guidelines
    results = search_clinical_guidelines(query, n_results=5)

    # Try to determine pathogen category
    pathogen_category = None
    pathogen_lower = pathogen.lower()

    pathogen_mapping = {
        "ESBL-E": ["esbl", "extended-spectrum", "e. coli", "klebsiella"],
        "CRE": ["carbapenem-resistant", "cre", "carbapenemase"],
        "CRAB": ["acinetobacter", "crab"],
        "DTR-PA": ["pseudomonas", "dtr"],
        "S.maltophilia": ["stenotrophomonas", "maltophilia"],
    }

    for category, keywords in pathogen_mapping.items():
        for keyword in keywords:
            if keyword in pathogen_lower:
                pathogen_category = category
                break

    # Search with pathogen filter if category identified
    if pathogen_category:
        filtered_results = search_clinical_guidelines(
            query, pathogen_filter=pathogen_category, n_results=3
        )
        if filtered_results:
            results = filtered_results + results[:2]  # Combine results

    return {
        "query": query,
        "pathogen_category": pathogen_category or "General",
        "recommendations": results[:5],
        "note": "These recommendations are from IDSA 2024 guidelines. Always verify with current institutional protocols."
    }


def explain_mic_interpretation(
    pathogen: str,
    antibiotic: str,
    mic_value: float
) -> dict:
    """
    Get detailed explanation for MIC interpretation from reference docs.

    Args:
        pathogen: Pathogen name
        antibiotic: Antibiotic name
        mic_value: The MIC value to interpret

    Returns:
        Detailed explanation with reference citations
    """
    query = f"MIC breakpoint interpretation for {antibiotic} against {pathogen}"

    results = search_mic_reference_docs(query, n_results=3)

    return {
        "query": query,
        "mic_value": mic_value,
        "reference_excerpts": results,
        "note": "Refer to current EUCAST v16.0 breakpoint tables for official interpretation."
    }


def get_empirical_therapy_guidance(
    infection_type: str,
    risk_factors: list[str] = None
) -> dict:
    """
    Get empirical therapy guidance for an infection type.

    Args:
        infection_type: Type of infection (e.g., "UTI", "pneumonia", "sepsis")
        risk_factors: List of risk factors (e.g., ["prior MRSA", "recent antibiotics"])

    Returns:
        Empirical therapy recommendations
    """
    query_parts = [f"empirical therapy for {infection_type}"]

    if risk_factors:
        query_parts.append(f"with risk factors: {', '.join(risk_factors)}")

    query = " ".join(query_parts)

    results = search_clinical_guidelines(query, n_results=5)

    return {
        "infection_type": infection_type,
        "risk_factors": risk_factors or [],
        "recommendations": results,
        "note": "Empirical therapy should be de-escalated based on culture results."
    }
