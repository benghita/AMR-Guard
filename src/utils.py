"""
Utility functions for clinical calculations and data parsing.

- Creatinine Clearance (CrCl) via Cockcroft-Gault
- MIC trend analysis and creep detection
- Prescription card formatter
- JSON parsing and data normalization helpers
"""

import json
import math
import re
from typing import Any, Dict, List, Literal, Optional, Tuple


# --- CrCl calculator ---

def calculate_crcl(
    age_years: float,
    weight_kg: float,
    serum_creatinine_mg_dl: float,
    sex: Literal["male", "female"],
    use_ibw: bool = False,
    height_cm: Optional[float] = None,
) -> float:
    """
    Cockcroft-Gault equation.

    CrCl = [(140 - age) × weight × (0.85 if female)] / (72 × SCr)

    When use_ibw=True and height is given, uses Ideal Body Weight.
    For obese patients (actual > 1.3 × IBW), switches to Adjusted Body Weight.
    Returns CrCl in mL/min.
    """
    if serum_creatinine_mg_dl <= 0:
        raise ValueError("Serum creatinine must be positive")
    if age_years <= 0 or weight_kg <= 0:
        raise ValueError("Age and weight must be positive")

    weight = weight_kg
    if use_ibw and height_cm:
        ibw = calculate_ibw(height_cm, sex)
        weight = calculate_adjusted_bw(ibw, weight_kg) if weight_kg > ibw * 1.3 else ibw

    crcl = ((140 - age_years) * weight) / (72 * serum_creatinine_mg_dl)
    if sex == "female":
        crcl *= 0.85

    return round(crcl, 1)


def calculate_ibw(height_cm: float, sex: Literal["male", "female"]) -> float:
    """
    Devine formula for Ideal Body Weight.

    Male: 50 kg + 2.3 kg per inch over 5 feet
    Female: 45.5 kg + 2.3 kg per inch over 5 feet
    """
    height_over_60_inches = max(0, height_cm / 2.54 - 60)
    base = 50 if sex == "male" else 45.5
    return round(base + 2.3 * height_over_60_inches, 1)


def calculate_adjusted_bw(ibw: float, actual_weight: float) -> float:
    """
    Adjusted Body Weight for obese patients.

    AdjBW = IBW + 0.4 × (Actual - IBW)
    """
    return round(ibw + 0.4 * (actual_weight - ibw), 1)


def get_renal_dose_category(crcl: float) -> str:
    """Map CrCl value to a dosing category string."""
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


# --- MIC trend analysis ---

def calculate_mic_trend(
    mic_values: List[Dict[str, Any]],
    susceptible_breakpoint: Optional[float] = None,
    resistant_breakpoint: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Analyze a list of MIC readings over time.

    Requires at least 2 readings. Uses linear regression slope for trend
    direction when >= 3 points are available; falls back to ratio comparison
    for exactly 2 points.
    """
    if len(mic_values) < 2:
        return {
            "trend": "insufficient_data",
            "risk_level": "UNKNOWN",
            "alert": "Need at least 2 MIC values for trend analysis",
        }

    mics = [float(v["mic_value"]) for v in mic_values]
    baseline_mic = mics[0]
    current_mic = mics[-1]
    fold_change = (current_mic / baseline_mic) if baseline_mic > 0 else float("inf")

    if len(mics) >= 3:
        n = len(mics)
        x_mean = (n - 1) / 2
        y_mean = sum(mics) / n
        numerator = sum((i - x_mean) * (mics[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        slope = numerator / denominator if denominator != 0 else 0
        trend = "increasing" if slope > 0.5 else "decreasing" if slope < -0.5 else "stable"
    else:
        trend = "increasing" if current_mic > baseline_mic * 1.5 else "decreasing" if current_mic < baseline_mic * 0.67 else "stable"

    # Fold change per time step (geometric rate of change)
    velocity = fold_change ** (1 / (len(mics) - 1)) if len(mics) > 1 else 1.0

    risk_level, alert = _assess_mic_risk(
        current_mic, baseline_mic, fold_change, trend,
        susceptible_breakpoint, resistant_breakpoint,
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
    Assign a risk level (LOW/MODERATE/HIGH/CRITICAL) based on breakpoints and fold change.

    Prefers breakpoint-based assessment when breakpoints are available.
    Falls back to fold-change thresholds otherwise.
    """
    if s_breakpoint is not None and r_breakpoint is not None:
        margin = s_breakpoint / current_mic if current_mic > 0 else float("inf")

        if current_mic > r_breakpoint:
            return "CRITICAL", f"MIC ({current_mic}) exceeds resistant breakpoint ({r_breakpoint}). Organism is RESISTANT."
        if current_mic > s_breakpoint:
            return "HIGH", f"MIC ({current_mic}) exceeds susceptible breakpoint ({s_breakpoint}). Consider alternative therapy."
        if margin < 2:
            if trend == "increasing":
                return "HIGH", f"MIC approaching breakpoint (margin: {margin:.1f}x) with increasing trend. High risk of resistance emergence."
            return "MODERATE", f"MIC close to breakpoint (margin: {margin:.1f}x). Monitor closely."
        if margin < 4:
            if trend == "increasing":
                return "MODERATE", f"MIC rising with {margin:.1f}x margin to breakpoint. Consider enhanced monitoring."
            return "LOW", "MIC stable with adequate margin to breakpoint."
        return "LOW", "MIC well below breakpoint with good safety margin."

    # No breakpoints — use fold change thresholds from EUCAST MIC creep criteria
    if fold_change >= 8:
        return "CRITICAL", f"MIC increased {fold_change:.1f}-fold from baseline. Urgent review needed."
    if fold_change >= 4:
        return "HIGH", f"MIC increased {fold_change:.1f}-fold from baseline. High risk of treatment failure."
    if fold_change >= 2:
        if trend == "increasing":
            return "MODERATE", f"MIC increased {fold_change:.1f}-fold with rising trend. Enhanced monitoring recommended."
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

    Augments calculate_mic_trend with a time-to-resistance estimate
    when the MIC is rising and a susceptible breakpoint is available.
    """
    result = calculate_mic_trend(
        mic_history,
        susceptible_breakpoint=breakpoints.get("susceptible"),
        resistant_breakpoint=breakpoints.get("resistant"),
    )

    result["organism"] = organism
    result["antibiotic"] = antibiotic
    result["breakpoint_susceptible"] = breakpoints.get("susceptible")
    result["breakpoint_resistant"] = breakpoints.get("resistant")

    # Estimate how many more time-points until MIC reaches the susceptible breakpoint
    if result["trend"] == "increasing" and result["velocity"] > 1.0:
        current = result["current_mic"]
        s_bp = breakpoints.get("susceptible")
        if s_bp and current < s_bp:
            doublings_needed = math.log2(s_bp / current) if current > 0 else 0
            log_velocity = math.log(result["velocity"]) / math.log(2)
            if log_velocity > 0:
                result["estimated_readings_to_resistance"] = round(doublings_needed / log_velocity, 1)

    return result


# --- Prescription formatter ---

def format_prescription_card(recommendation: Dict[str, Any]) -> str:
    """Format a recommendation dict as a plain-text prescription card."""
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

    adjustments = recommendation.get("dose_adjustments", {})
    if adjustments.get("renal") and adjustments["renal"] != "None needed":
        lines.append(f"\nRENAL ADJUSTMENT: {adjustments['renal']}")
    if adjustments.get("hepatic") and adjustments["hepatic"] != "None needed":
        lines.append(f"HEPATIC ADJUSTMENT: {adjustments['hepatic']}")

    alerts = recommendation.get("safety_alerts", [])
    if alerts:
        lines.append("\n" + "-" * 50)
        lines.append("SAFETY ALERTS:")
        for alert in alerts:
            level = alert.get("level", "INFO")
            marker = {"CRITICAL": "[!!!]", "WARNING": "[!!]", "INFO": "[i]"}.get(level, "[?]")
            lines.append(f"  {marker} {alert.get('message', '')}")

    monitoring = recommendation.get("monitoring_parameters", [])
    if monitoring:
        lines.append("\n" + "-" * 50)
        lines.append("MONITORING:")
        for param in monitoring:
            lines.append(f"  - {param}")

    if recommendation.get("rationale"):
        lines.append("\n" + "-" * 50)
        lines.append("RATIONALE:")
        lines.append(f"  {recommendation['rationale']}")

    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


# --- JSON parsing ---

def safe_json_parse(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the first JSON object from a string.

    Handles model output that may wrap JSON in markdown code fences.
    Returns None if no valid JSON is found.
    """
    if not text:
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in [r"```json\s*\n?(.*?)\n?```", r"```\s*\n?(.*?)\n?```", r"\{[\s\S]*\}"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(1) if match.lastindex else match.group(0)
                return json.loads(json_str)
            except (json.JSONDecodeError, IndexError):
                continue

    return None


def validate_agent_output(output: Dict[str, Any], required_fields: List[str]) -> Tuple[bool, List[str]]:
    """Return (is_valid, missing_fields) for an agent output dict."""
    missing = [f for f in required_fields if f not in output]
    return len(missing) == 0, missing


# --- Name normalization ---

def normalize_antibiotic_name(name: str) -> str:
    """Map common abbreviations and brand names to standard antibiotic names."""
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
    return mappings.get(name.lower().strip(), name.lower().strip())


def normalize_organism_name(name: str) -> str:
    """Map common abbreviations to full organism names."""
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
    return abbreviations.get(name.strip().lower(), name.strip())
