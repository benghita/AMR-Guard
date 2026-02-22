"""Drug safety and interaction tools for AMR-Guard workflow."""

from typing import Optional
from src.db.database import execute_query


def check_drug_interactions(
    target_drug: str,
    patient_medications: list[str],
    severity_filter: str = None
) -> list[dict]:
    """
    Check for interactions between target drug and patient's medications.

    Args:
        target_drug: Antibiotic being considered
        patient_medications: List of patient's current medications
        severity_filter: Optional filter ('major', 'moderate', 'minor')

    Returns:
        List of interaction dicts with severity and description

    Used by: Agent 4 (Safety Check)
    """
    if not patient_medications:
        return []

    # Build query with proper parameter handling
    placeholders = ','.join(['?' for _ in patient_medications])

    conditions = [f"LOWER(drug_2) IN ({placeholders})"]
    params = [med.lower() for med in patient_medications]

    # Add target drug condition
    conditions.append("LOWER(drug_1) LIKE LOWER(?)")
    params.append(f"%{target_drug}%")

    if severity_filter:
        conditions.append("severity = ?")
        params.append(severity_filter)

    where_clause = " AND ".join(conditions)

    query = f"""
        SELECT
            drug_1,
            drug_2,
            interaction_description,
            severity
        FROM drug_interaction_lookup
        WHERE {where_clause}
        ORDER BY
            CASE severity
                WHEN 'major' THEN 1
                WHEN 'moderate' THEN 2
                WHEN 'minor' THEN 3
                ELSE 4
            END
    """

    return execute_query(query, tuple(params))


def check_single_interaction(drug_1: str, drug_2: str) -> Optional[dict]:
    """
    Check for interaction between two specific drugs.

    Args:
        drug_1: First drug name
        drug_2: Second drug name

    Returns:
        Interaction details or None if no interaction found
    """
    query = """
        SELECT
            drug_1,
            drug_2,
            interaction_description,
            severity
        FROM drug_interaction_lookup
        WHERE (LOWER(drug_1) LIKE LOWER(?) AND LOWER(drug_2) LIKE LOWER(?))
        LIMIT 1
    """

    results = execute_query(query, (f"%{drug_1}%", f"%{drug_2}%"))
    return results[0] if results else None


def get_all_interactions_for_drug(drug: str) -> list[dict]:
    """
    Get all known interactions for a specific drug.

    Args:
        drug: Drug name to check

    Returns:
        List of all interactions involving this drug
    """
    query = """
        SELECT
            drug_1,
            drug_2,
            interaction_description,
            severity
        FROM drug_interaction_lookup
        WHERE LOWER(drug_1) LIKE LOWER(?)
        ORDER BY
            CASE severity
                WHEN 'major' THEN 1
                WHEN 'moderate' THEN 2
                WHEN 'minor' THEN 3
                ELSE 4
            END
        LIMIT 100
    """

    return execute_query(query, (f"%{drug}%",))


def get_major_interactions_for_drug(drug: str) -> list[dict]:
    """
    Get only major interactions for a specific drug.

    Args:
        drug: Drug name to check

    Returns:
        List of major severity interactions
    """
    query = """
        SELECT
            drug_1,
            drug_2,
            interaction_description
        FROM drug_interaction_lookup
        WHERE LOWER(drug_1) LIKE LOWER(?)
          AND severity = 'major'
        LIMIT 50
    """

    return execute_query(query, (f"%{drug}%",))


def screen_antibiotic_safety(
    antibiotic: str,
    patient_medications: list[str],
    patient_allergies: list[str] = None
) -> dict:
    """
    Comprehensive safety screening for an antibiotic choice.

    Args:
        antibiotic: Proposed antibiotic
        patient_medications: List of current medications
        patient_allergies: List of known allergies (optional)

    Returns:
        Safety assessment with interactions and alerts

    Used by: Agent 4 (Clinical Pharmacologist)
    """
    safety_report = {
        "antibiotic": antibiotic,
        "safe_to_use": True,
        "alerts": [],
        "interactions": [],
        "allergy_warnings": []
    }

    # Check drug interactions
    interactions = check_drug_interactions(antibiotic, patient_medications)

    if interactions:
        safety_report["interactions"] = interactions

        # Check for major interactions
        major = [i for i in interactions if i.get('severity') == 'major']
        moderate = [i for i in interactions if i.get('severity') == 'moderate']

        if major:
            safety_report["safe_to_use"] = False
            safety_report["alerts"].append({
                "level": "CRITICAL",
                "message": f"Found {len(major)} major drug interaction(s). Review required before prescribing."
            })

        if moderate:
            safety_report["alerts"].append({
                "level": "WARNING",
                "message": f"Found {len(moderate)} moderate drug interaction(s). Consider dose adjustment or monitoring."
            })

    # Check allergies (basic check for cross-reactivity)
    if patient_allergies:
        antibiotic_lower = antibiotic.lower()

        # Common antibiotic class cross-reactivity patterns
        cross_reactivity = {
            "penicillin": ["amoxicillin", "ampicillin", "piperacillin", "cephalosporin"],
            "cephalosporin": ["ceftriaxone", "cefotaxime", "ceftazidime", "cefepime"],
            "sulfa": ["sulfamethoxazole", "trimethoprim-sulfamethoxazole", "bactrim"],
            "fluoroquinolone": ["ciprofloxacin", "levofloxacin", "moxifloxacin"],
        }

        for allergy in patient_allergies:
            allergy_lower = allergy.lower()

            # Direct match
            if allergy_lower in antibiotic_lower:
                safety_report["safe_to_use"] = False
                safety_report["allergy_warnings"].append({
                    "level": "CRITICAL",
                    "message": f"Patient has documented allergy to {allergy}. CONTRAINDICATED."
                })

            # Cross-reactivity check
            for allergen, related in cross_reactivity.items():
                if allergen in allergy_lower:
                    for related_drug in related:
                        if related_drug in antibiotic_lower:
                            safety_report["alerts"].append({
                                "level": "WARNING",
                                "message": f"Potential cross-reactivity: Patient allergic to {allergy}, {antibiotic} is in related class."
                            })

    # Summary
    if safety_report["safe_to_use"]:
        safety_report["summary"] = "No critical safety concerns identified."
    else:
        safety_report["summary"] = "SAFETY CONCERNS IDENTIFIED - Review required before prescribing."

    return safety_report


def get_interaction_statistics() -> dict:
    """Get statistics about the drug interaction database."""
    queries = {
        "total": "SELECT COUNT(*) as count FROM drug_interactions",
        "major": "SELECT COUNT(*) as count FROM drug_interactions WHERE severity = 'major'",
        "moderate": "SELECT COUNT(*) as count FROM drug_interactions WHERE severity = 'moderate'",
        "minor": "SELECT COUNT(*) as count FROM drug_interactions WHERE severity = 'minor'",
    }

    stats = {}
    for key, query in queries.items():
        result = execute_query(query)
        stats[key] = result[0]['count'] if result else 0

    return stats
