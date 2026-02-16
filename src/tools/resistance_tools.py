"""Resistance pattern and trend analysis tools for Med-I-C workflow."""

from typing import Optional
from src.db.database import execute_query


def query_resistance_pattern(
    pathogen: str,
    antibiotic: str = None,
    region: str = None,
    year: int = None
) -> list[dict]:
    """
    Query ATLAS susceptibility data for resistance patterns.

    Args:
        pathogen: Pathogen name (e.g., "E. coli", "K. pneumoniae")
        antibiotic: Optional specific antibiotic to check
        region: Optional geographic region filter
        year: Optional year filter (defaults to most recent)

    Returns:
        List of susceptibility records with percentages

    Used by: Agent 1 (Empirical), Agent 3 (Trend Analysis)
    """
    conditions = ["LOWER(species) LIKE LOWER(?)"]
    params = [f"%{pathogen}%"]

    if antibiotic:
        conditions.append("LOWER(antibiotic) LIKE LOWER(?)")
        params.append(f"%{antibiotic}%")

    if region:
        conditions.append("LOWER(region) LIKE LOWER(?)")
        params.append(f"%{region}%")

    if year:
        conditions.append("year = ?")
        params.append(year)

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT
            species,
            family,
            antibiotic,
            percent_susceptible,
            percent_intermediate,
            percent_resistant,
            total_isolates,
            year,
            region
        FROM atlas_susceptibility
        WHERE {where_clause}
        ORDER BY year DESC, percent_susceptible DESC
        LIMIT 50
    """

    return execute_query(query, tuple(params))


def get_most_effective_antibiotics(
    pathogen: str,
    min_susceptibility: float = 80.0,
    limit: int = 10
) -> list[dict]:
    """
    Find antibiotics with highest susceptibility for a pathogen.

    Args:
        pathogen: Pathogen name
        min_susceptibility: Minimum susceptibility percentage (default 80%)
        limit: Maximum number of results

    Returns:
        List of effective antibiotics sorted by susceptibility
    """
    query = """
        SELECT
            antibiotic,
            AVG(percent_susceptible) as avg_susceptibility,
            SUM(total_isolates) as total_samples,
            MAX(year) as latest_year
        FROM atlas_susceptibility
        WHERE LOWER(species) LIKE LOWER(?)
          AND percent_susceptible >= ?
        GROUP BY antibiotic
        ORDER BY avg_susceptibility DESC
        LIMIT ?
    """

    return execute_query(query, (f"%{pathogen}%", min_susceptibility, limit))


def get_resistance_trend(
    pathogen: str,
    antibiotic: str
) -> list[dict]:
    """
    Get resistance trend over time for pathogen-antibiotic combination.

    Args:
        pathogen: Pathogen name
        antibiotic: Antibiotic name

    Returns:
        List of yearly susceptibility data
    """
    query = """
        SELECT
            year,
            AVG(percent_susceptible) as avg_susceptibility,
            AVG(percent_resistant) as avg_resistance,
            SUM(total_isolates) as total_samples
        FROM atlas_susceptibility
        WHERE LOWER(species) LIKE LOWER(?)
          AND LOWER(antibiotic) LIKE LOWER(?)
          AND year IS NOT NULL
        GROUP BY year
        ORDER BY year ASC
    """

    return execute_query(query, (f"%{pathogen}%", f"%{antibiotic}%"))


def calculate_mic_trend(
    historical_mics: list[dict],
    current_mic: float = None
) -> dict:
    """
    Calculate resistance velocity and MIC trend from historical data.

    Args:
        historical_mics: List of historical MIC readings [{"date": ..., "mic_value": ...}, ...]
        current_mic: Optional current MIC value (if not in historical_mics)

    Returns:
        Dict with trend analysis, resistance_velocity, risk_level

    Used by: Agent 3 (Trend Analyst)

    Logic:
        - If MIC increases by 4x (two-step dilution), flag HIGH risk
        - If MIC increases by 2x (one-step dilution), flag MODERATE risk
        - Otherwise, LOW risk
    """
    if not historical_mics:
        return {
            "risk_level": "UNKNOWN",
            "message": "No historical MIC data available",
            "trend": None,
            "velocity": None
        }

    # Sort by date if available
    sorted_mics = sorted(
        historical_mics,
        key=lambda x: x.get('date', '0')
    )

    mic_values = [m['mic_value'] for m in sorted_mics if m.get('mic_value')]

    if current_mic:
        mic_values.append(current_mic)

    if len(mic_values) < 2:
        return {
            "risk_level": "UNKNOWN",
            "message": "Insufficient MIC history (need at least 2 values)",
            "trend": None,
            "velocity": None,
            "values": mic_values
        }

    baseline_mic = mic_values[0]
    latest_mic = mic_values[-1]

    # Avoid division by zero
    if baseline_mic == 0:
        baseline_mic = 0.001

    ratio = latest_mic / baseline_mic

    # Calculate velocity (fold change per time point)
    velocity = ratio ** (1 / (len(mic_values) - 1)) if len(mic_values) > 1 else 1

    # Determine trend direction
    if ratio > 1.5:
        trend = "INCREASING"
    elif ratio < 0.67:
        trend = "DECREASING"
    else:
        trend = "STABLE"

    # Determine risk level
    if ratio >= 4:
        risk_level = "HIGH"
        alert = "MIC CREEP DETECTED - Two-step dilution increase. High risk of treatment failure even if currently 'Susceptible'."
    elif ratio >= 2:
        risk_level = "MODERATE"
        alert = "MIC trending upward (one-step dilution increase). Monitor closely and consider alternative agents."
    elif trend == "INCREASING":
        risk_level = "LOW"
        alert = "Slight MIC increase observed. Continue current therapy with monitoring."
    else:
        risk_level = "LOW"
        alert = "MIC stable or decreasing. Current therapy appears effective."

    return {
        "risk_level": risk_level,
        "alert": alert,
        "trend": trend,
        "velocity": round(velocity, 2),
        "ratio": round(ratio, 2),
        "baseline_mic": baseline_mic,
        "current_mic": latest_mic,
        "data_points": len(mic_values),
        "values": mic_values
    }


def get_pathogen_families() -> list[dict]:
    """Get list of unique pathogen families in the database."""
    query = """
        SELECT DISTINCT family, COUNT(DISTINCT species) as species_count
        FROM atlas_susceptibility
        WHERE family IS NOT NULL AND family != ''
        GROUP BY family
        ORDER BY species_count DESC
    """
    return execute_query(query)


def get_pathogens_by_family(family: str) -> list[dict]:
    """Get all pathogens in a specific family."""
    query = """
        SELECT DISTINCT species
        FROM atlas_susceptibility
        WHERE LOWER(family) LIKE LOWER(?)
        ORDER BY species
    """
    return execute_query(query, (f"%{family}%",))
