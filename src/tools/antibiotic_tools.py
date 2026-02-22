"""Antibiotic query tools for AMR-Guard workflow."""

from typing import Optional
from src.db.database import execute_query


def query_antibiotic_info(
    antibiotic_name: str,
    include_category: bool = True,
    include_formulations: bool = True
) -> list[dict]:
    """
    Query EML antibiotic database for classification and details.

    Args:
        antibiotic_name: Name of the antibiotic (partial match supported)
        include_category: Include WHO stewardship category
        include_formulations: Include available formulations

    Returns:
        List of matching antibiotics with details

    Used by: Agent 1, Agent 4
    """
    query = """
        SELECT
            medicine_name,
            who_category,
            eml_section,
            formulations,
            indication,
            atc_codes,
            combined_with,
            status
        FROM eml_antibiotics
        WHERE LOWER(medicine_name) LIKE LOWER(?)
        ORDER BY
            CASE who_category
                WHEN 'ACCESS' THEN 1
                WHEN 'WATCH' THEN 2
                WHEN 'RESERVE' THEN 3
                ELSE 4
            END
    """

    results = execute_query(query, (f"%{antibiotic_name}%",))

    # Filter columns based on parameters
    if not include_category or not include_formulations:
        filtered_results = []
        for r in results:
            filtered = dict(r)
            if not include_category:
                filtered.pop('who_category', None)
            if not include_formulations:
                filtered.pop('formulations', None)
            filtered_results.append(filtered)
        return filtered_results

    return results


def get_antibiotics_by_category(category: str) -> list[dict]:
    """
    Get all antibiotics in a specific WHO category.

    Args:
        category: WHO category ('ACCESS', 'WATCH', 'RESERVE')

    Returns:
        List of antibiotics in that category
    """
    query = """
        SELECT medicine_name, indication, formulations, atc_codes
        FROM eml_antibiotics
        WHERE UPPER(who_category) = UPPER(?)
        ORDER BY medicine_name
    """

    return execute_query(query, (category,))


def get_antibiotic_for_indication(indication_keyword: str) -> list[dict]:
    """
    Find antibiotics based on indication keywords.

    Args:
        indication_keyword: Keyword to search in indications

    Returns:
        List of matching antibiotics with indications
    """
    query = """
        SELECT
            medicine_name,
            who_category,
            indication,
            formulations
        FROM eml_antibiotics
        WHERE LOWER(indication) LIKE LOWER(?)
        ORDER BY
            CASE who_category
                WHEN 'ACCESS' THEN 1
                WHEN 'WATCH' THEN 2
                WHEN 'RESERVE' THEN 3
                ELSE 4
            END
    """

    return execute_query(query, (f"%{indication_keyword}%",))


def interpret_mic_value(
    pathogen: str,
    antibiotic: str,
    mic_value: float,
    route: str = None
) -> dict:
    """
    Interpret MIC value against EUCAST breakpoints.

    Args:
        pathogen: Pathogen name or group
        antibiotic: Antibiotic name
        mic_value: MIC value in mg/L
        route: Administration route (IV, Oral)

    Returns:
        Dict with interpretation (S/I/R), breakpoint values, clinical notes

    Used by: Agent 2, Agent 3
    """
    query = """
        SELECT
            pathogen_group,
            antibiotic,
            mic_susceptible,
            mic_resistant,
            notes,
            route
        FROM mic_breakpoints
        WHERE LOWER(pathogen_group) LIKE LOWER(?)
          AND LOWER(antibiotic) LIKE LOWER(?)
        LIMIT 1
    """

    results = execute_query(query, (f"%{pathogen}%", f"%{antibiotic}%"))

    if not results:
        return {
            "interpretation": "UNKNOWN",
            "message": f"No breakpoint found for {antibiotic} against {pathogen}",
            "mic_value": mic_value,
            "breakpoints": None
        }

    bp = results[0]
    mic_s = bp.get('mic_susceptible')
    mic_r = bp.get('mic_resistant')

    # Determine interpretation
    if mic_s is not None and mic_value <= mic_s:
        interpretation = "SUSCEPTIBLE"
        message = f"MIC ({mic_value} mg/L) ≤ S breakpoint ({mic_s} mg/L)"
    elif mic_r is not None and mic_value > mic_r:
        interpretation = "RESISTANT"
        message = f"MIC ({mic_value} mg/L) > R breakpoint ({mic_r} mg/L)"
    elif mic_s is not None and mic_r is not None:
        interpretation = "INTERMEDIATE"
        message = f"MIC ({mic_value} mg/L) between S ({mic_s}) and R ({mic_r}) breakpoints"
    else:
        interpretation = "UNKNOWN"
        message = "Incomplete breakpoint data"

    return {
        "interpretation": interpretation,
        "message": message,
        "mic_value": mic_value,
        "breakpoints": {
            "susceptible": mic_s,
            "resistant": mic_r
        },
        "pathogen_group": bp.get('pathogen_group'),
        "notes": bp.get('notes')
    }


def get_breakpoints_for_pathogen(pathogen: str) -> list[dict]:
    """
    Get all available breakpoints for a pathogen.

    Args:
        pathogen: Pathogen name or group

    Returns:
        List of antibiotic breakpoints for the pathogen
    """
    query = """
        SELECT
            antibiotic,
            mic_susceptible,
            mic_resistant,
            route,
            notes
        FROM mic_breakpoints
        WHERE LOWER(pathogen_group) LIKE LOWER(?)
        ORDER BY antibiotic
    """

    return execute_query(query, (f"%{pathogen}%",))
