"""
Utility functions for Med-I-C multi-agent system.

Includes:
- Creatinine Clearance (CrCl) calculator
- MIC trend analysis and creep detection
- Prescription card formatter
- Data validation helpers
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Literal, Optional, Tuple


# =============================================================================
# CREATININE CLEARANCE CALCULATOR
# =============================================================================

def calculate_crcl(
    age_years: float,
    weight_kg: float,
    serum_creatinine_mg_dl: float,
    sex: Literal["male", "female"],
    use_ibw: bool = False,
    height_cm: Optional[float] = None,
) -> float:
    """
    Calculate Creatinine Clearance using the Cockcroft-Gault equation.

    Formula:
        CrCl = [(140 - age) × weight × (0.85 if female)] / (72 × SCr)

    Args:
        age_years: Patient age in years
        weight_kg: Actual body weight in kg
        serum_creatinine_mg_dl: Serum creatinine in mg/dL
        sex: Patient sex ("male" or "female")
        use_ibw: If True, use Ideal Body Weight instead of actual weight
        height_cm: Height in cm (required if use_ibw=True)

    Returns:
        Estimated CrCl in mL/min
    """
    if serum_creatinine_mg_dl <= 0:
        raise ValueError("Serum creatinine must be positive")

    if age_years <= 0 or weight_kg <= 0:
        raise ValueError("Age and weight must be positive")

    # Calculate weight to use
    weight = weight_kg
    if use_ibw and height_cm:
        weight = calculate_ibw(height_cm, sex)
        # Use adjusted body weight if actual weight > IBW
        if weight_kg > weight * 1.3:
            weight = calculate_adjusted_bw(weight, weight_kg)

    # Cockcroft-Gault equation
    crcl = ((140 - age_years) * weight) / (72 * serum_creatinine_mg_dl)

    # Apply sex factor
    if sex == "female":
        crcl *= 0.85

    return round(crcl, 1)


def calculate_ibw(height_cm: float, sex: Literal["male", "female"]) -> float:
    """
    Calculate Ideal Body Weight using the Devine formula.

    Args:
        height_cm: Height in centimeters
        sex: Patient sex

    Returns:
        Ideal body weight in kg
    """
    height_inches = height_cm / 2.54
    height_over_60 = max(0, height_inches - 60)

    if sex == "male":
        ibw = 50 + 2.3 * height_over_60
    else:
        ibw = 45.5 + 2.3 * height_over_60

    return round(ibw, 1)


def calculate_adjusted_bw(ibw: float, actual_weight: float) -> float:
    """
    Calculate Adjusted Body Weight for obese patients.

    Formula: AdjBW = IBW + 0.4 × (Actual - IBW)
    """
    return round(ibw + 0.4 * (actual_weight - ibw), 1)


def get_renal_dose_category(crcl: float) -> str:
    """
    Categorize renal function for dosing purposes.

    Returns:
        Renal function category
    """
    if crcl >= 90:
        return "normal"
    elif crcl >= 60:
        return "mild_impairment"
    elif crcl >= 30:
        return "moderate_impairment"
    elif crcl >= 15:
        return "severe_impairment"
    else:
        return "esrd"


# =============================================================================
# MIC TREND ANALYSIS
# =============================================================================

def calculate_mic_trend(
    mic_values: List[Dict[str, Any]],
    susceptible_breakpoint: Optional[float] = None,
    resistant_breakpoint: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyze MIC trend over time and detect MIC creep.

    Args:
        mic_values: List of dicts with 'date' and 'mic_value' keys
        susceptible_breakpoint: S breakpoint (optional)
        resistant_breakpoint: R breakpoint (optional)

    Returns:
        Dict with trend analysis results
    """
    if len(mic_values) < 2:
        return {
            "trend": "insufficient_data",
            "risk_level": "UNKNOWN",
            "alert": "Need at least 2 MIC values for trend analysis",
        }

    # Extract MIC values
    mics = [float(v["mic_value"]) for v in mic_values]

    baseline_mic = mics[0]
    current_mic = mics[-1]

    # Calculate fold change
    if baseline_mic > 0:
        fold_change = current_mic / baseline_mic
    else:
        fold_change = float("inf")

    # Calculate trend
    if len(mics) >= 3:
        # Linear regression slope
        n = len(mics)
        x_mean = (n - 1) / 2
        y_mean = sum(mics) / n
        numerator = sum((i - x_mean) * (mics[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0

        if slope > 0.5:
            trend = "increasing"
        elif slope < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"
    else:
        if current_mic > baseline_mic * 1.5:
            trend = "increasing"
        elif current_mic < baseline_mic * 0.67:
            trend = "decreasing"
        else:
            trend = "stable"

    # Calculate resistance velocity (fold change per time point)
    velocity = fold_change ** (1 / (len(mics) - 1)) if len(mics) > 1 else 1.0

    # Determine risk level
    risk_level, alert = _assess_mic_risk(
        current_mic, baseline_mic, fold_change, trend,
        susceptible_breakpoint, resistant_breakpoint
    )

    return {
        "baseline_mic": baseline_mic,
        "current_mic": current_mic,
        "ratio": round(fold_change, 2),
        "trend": trend,
        "velocity": round(velocity, 3),
        "risk_level": risk_level,
        "alert": alert,
        "n_readings": len(mics),
    }


def _assess_mic_risk(
    current_mic: float,
    baseline_mic: float,
    fold_change: float,
    trend: str,
    s_breakpoint: Optional[float],
    r_breakpoint: Optional[float],
) -> Tuple[str, str]:
    """
    Assess risk level based on MIC trends and breakpoints.

    Returns:
        Tuple of (risk_level, alert_message)
    """
    # If we have breakpoints, use them for risk assessment
    if s_breakpoint is not None and r_breakpoint is not None:
        margin = s_breakpoint / current_mic if current_mic > 0 else float("inf")

        if current_mic > r_breakpoint:
            return "CRITICAL", f"MIC ({current_mic}) exceeds resistant breakpoint ({r_breakpoint}). Organism is RESISTANT."

        if current_mic > s_breakpoint:
            return "HIGH", f"MIC ({current_mic}) exceeds susceptible breakpoint ({s_breakpoint}). Consider alternative therapy."

        if margin < 2:
            if trend == "increasing":
                return "HIGH", f"MIC approaching breakpoint (margin: {margin:.1f}x) with increasing trend. High risk of resistance emergence."
            else:
                return "MODERATE", f"MIC close to breakpoint (margin: {margin:.1f}x). Monitor closely."

        if margin < 4:
            if trend == "increasing":
                return "MODERATE", f"MIC rising with {margin:.1f}x margin to breakpoint. Consider enhanced monitoring."
            else:
                return "LOW", "MIC stable with adequate margin to breakpoint."

        return "LOW", "MIC well below breakpoint with good safety margin."

    # Without breakpoints, use fold change and trend
    if fold_change >= 8:
        return "CRITICAL", f"MIC increased {fold_change:.1f}-fold from baseline. Urgent review needed."

    if fold_change >= 4:
        return "HIGH", f"MIC increased {fold_change:.1f}-fold from baseline. High risk of treatment failure."

    if fold_change >= 2:
        if trend == "increasing":
            return "MODERATE", f"MIC increased {fold_change:.1f}-fold with rising trend. Enhanced monitoring recommended."
        else:
            return "LOW", f"MIC increased {fold_change:.1f}-fold but trend is {trend}."

    if trend == "increasing":
        return "MODERATE", "MIC showing upward trend. Continue monitoring."

    return "LOW", "MIC stable or decreasing. Current therapy appropriate."


def detect_mic_creep(
    organism: str,
    antibiotic: str,
    mic_history: List[Dict[str, Any]],
    breakpoints: Dict[str, float],
) -> Dict[str, Any]:
    """
    Detect MIC creep for a specific organism-antibiotic pair.

    Args:
        organism: Pathogen name
        antibiotic: Antibiotic name
        mic_history: Historical MIC values with dates
        breakpoints: Dict with 'susceptible' and 'resistant' keys

    Returns:
        Comprehensive MIC creep analysis
    """
    trend_analysis = calculate_mic_trend(
        mic_history,
        susceptible_breakpoint=breakpoints.get("susceptible"),
        resistant_breakpoint=breakpoints.get("resistant"),
    )

    # Add organism/antibiotic context
    trend_analysis["organism"] = organism
    trend_analysis["antibiotic"] = antibiotic
    trend_analysis["breakpoint_susceptible"] = breakpoints.get("susceptible")
    trend_analysis["breakpoint_resistant"] = breakpoints.get("resistant")

    # Calculate time to resistance estimate
    if trend_analysis["trend"] == "increasing" and trend_analysis["velocity"] > 1.0:
        current = trend_analysis["current_mic"]
        s_bp = breakpoints.get("susceptible")
        if s_bp and current < s_bp:
            # Estimate doublings needed to reach breakpoint
            doublings_needed = math.log2(s_bp / current) if current > 0 else 0
            # Estimate time based on velocity
            if trend_analysis["velocity"] > 1.0:
                log_velocity = math.log(trend_analysis["velocity"]) / math.log(2)
                if log_velocity > 0:
                    time_estimate = doublings_needed / log_velocity
                    trend_analysis["estimated_readings_to_resistance"] = round(time_estimate, 1)

    return trend_analysis


# =============================================================================
# PRESCRIPTION FORMATTER
# =============================================================================

def format_prescription_card(recommendation: Dict[str, Any]) -> str:
    """
    Format a recommendation into a readable prescription card.

    Args:
        recommendation: Dict with recommendation details

    Returns:
        Formatted prescription card as string
    """
    lines = []
    lines.append("=" * 50)
    lines.append("ANTIBIOTIC PRESCRIPTION")
    lines.append("=" * 50)

    primary = recommendation.get("primary_recommendation", recommendation)

    lines.append(f"\nDRUG: {primary.get('antibiotic', 'N/A')}")
    lines.append(f"DOSE: {primary.get('dose', 'N/A')}")
    lines.append(f"ROUTE: {primary.get('route', 'N/A')}")
    lines.append(f"FREQUENCY: {primary.get('frequency', 'N/A')}")
    lines.append(f"DURATION: {primary.get('duration', 'N/A')}")

    if primary.get("aware_category"):
        lines.append(f"WHO AWaRe: {primary.get('aware_category')}")

    # Dose adjustments
    adjustments = recommendation.get("dose_adjustments", {})
    if adjustments.get("renal") and adjustments["renal"] != "None needed":
        lines.append(f"\nRENAL ADJUSTMENT: {adjustments['renal']}")
    if adjustments.get("hepatic") and adjustments["hepatic"] != "None needed":
        lines.append(f"HEPATIC ADJUSTMENT: {adjustments['hepatic']}")

    # Safety alerts
    alerts = recommendation.get("safety_alerts", [])
    if alerts:
        lines.append("\n" + "-" * 50)
        lines.append("SAFETY ALERTS:")
        for alert in alerts:
            level = alert.get("level", "INFO")
            marker = {"CRITICAL": "[!!!]", "WARNING": "[!!]", "INFO": "[i]"}.get(level, "[?]")
            lines.append(f"  {marker} {alert.get('message', '')}")

    # Monitoring
    monitoring = recommendation.get("monitoring_parameters", [])
    if monitoring:
        lines.append("\n" + "-" * 50)
        lines.append("MONITORING:")
        for param in monitoring:
            lines.append(f"  - {param}")

    # Rationale
    if recommendation.get("rationale"):
        lines.append("\n" + "-" * 50)
        lines.append("RATIONALE:")
        lines.append(f"  {recommendation['rationale']}")

    lines.append("\n" + "=" * 50)

    return "\n".join(lines)


# =============================================================================
# JSON PARSING HELPERS
# =============================================================================

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    Safely parse JSON from agent output, handling common issues.

    Attempts to extract JSON from text that may contain markdown code blocks
    or other formatting.
    """
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    import re

    json_patterns = [
        r"```json\s*\n?(.*?)\n?```",  # ```json ... ```
        r"```\s*\n?(.*?)\n?```",       # ``` ... ```
        r"\{[\s\S]*\}",                 # Raw JSON object
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def validate_agent_output(output: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate that agent output contains required fields.

    Args:
        output: Agent output dict
        required_fields: List of required field names

    Returns:
        Tuple of (is_valid, list_of_missing_fields)
    """
    missing = [field for field in required_fields if field not in output]
    return len(missing) == 0, missing


# =============================================================================
# DATA NORMALIZATION
# =============================================================================

def normalize_antibiotic_name(name: str) -> str:
    """
    Normalize antibiotic name to standard format.
    """
    # Common name mappings
    mappings = {
        "amox": "amoxicillin",
        "amox/clav": "amoxicillin-clavulanate",
        "augmentin": "amoxicillin-clavulanate",
        "pip/tazo": "piperacillin-tazobactam",
        "zosyn": "piperacillin-tazobactam",
        "tmp/smx": "trimethoprim-sulfamethoxazole",
        "bactrim": "trimethoprim-sulfamethoxazole",
        "cipro": "ciprofloxacin",
        "levo": "levofloxacin",
        "moxi": "moxifloxacin",
        "vanc": "vancomycin",
        "vanco": "vancomycin",
        "mero": "meropenem",
        "imi": "imipenem",
        "gent": "gentamicin",
        "tobra": "tobramycin",
        "ceftriax": "ceftriaxone",
        "rocephin": "ceftriaxone",
        "cefepime": "cefepime",
        "maxipime": "cefepime",
    }

    normalized = name.lower().strip()
    return mappings.get(normalized, normalized)


def normalize_organism_name(name: str) -> str:
    """
    Normalize organism name to standard format.
    """
    name = name.strip()

    # Common abbreviations
    abbreviations = {
        "e. coli": "Escherichia coli",
        "e.coli": "Escherichia coli",
        "k. pneumoniae": "Klebsiella pneumoniae",
        "k.pneumoniae": "Klebsiella pneumoniae",
        "p. aeruginosa": "Pseudomonas aeruginosa",
        "p.aeruginosa": "Pseudomonas aeruginosa",
        "s. aureus": "Staphylococcus aureus",
        "s.aureus": "Staphylococcus aureus",
        "mrsa": "Staphylococcus aureus (MRSA)",
        "mssa": "Staphylococcus aureus (MSSA)",
        "enterococcus": "Enterococcus species",
        "vre": "Enterococcus (VRE)",
    }

    lower_name = name.lower()
    return abbreviations.get(lower_name, name)


__all__ = [
    "calculate_crcl",
    "calculate_ibw",
    "calculate_adjusted_bw",
    "get_renal_dose_category",
    "calculate_mic_trend",
    "detect_mic_creep",
    "format_prescription_card",
    "safe_json_parse",
    "validate_agent_output",
    "normalize_antibiotic_name",
    "normalize_organism_name",
]
