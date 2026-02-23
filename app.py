"""
AMR-Guard — Gradio Interface (ZeroGPU compatible)
Infection Lifecycle Orchestrator · Multi-Agent Clinical Decision Support
"""

import json
import logging
import os
import subprocess
import sys
import traceback
from io import BytesIO
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging early so all module-level loggers emit to stdout.
# force=True reconfigures the root logger even if already set by an import.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)

# ── huggingface_hub compatibility shim ───────────────────────────────────────
# Older gradio versions (pre-5.7) import HfFolder from huggingface_hub in
# oauth.py. HfFolder was removed in huggingface_hub >= 0.25. Patch it back
# in-memory before importing gradio so the old oauth.py can find it.
try:
    from huggingface_hub import HfFolder as _check  # noqa: F401
except ImportError:
    import huggingface_hub as _hfh

    class _HfFolder:
        @staticmethod
        def get_token():
            return os.environ.get("HF_TOKEN") or _hfh.get_token()

        @staticmethod
        def save_token(token: str) -> None:  # noqa: ARG004
            pass

        @staticmethod
        def delete_token() -> None:
            pass

    _hfh.HfFolder = _HfFolder

# ── HuggingFace Spaces: auto-build knowledge base on first boot ───────────────
_DB_PATH = PROJECT_ROOT / os.getenv("MEDIC_DATA_DIR", "data") / "amr_guard.db"
if os.environ.get("SPACE_ID") and not _DB_PATH.exists():
    subprocess.run([sys.executable, str(PROJECT_ROOT / "setup_demo.py")], check=False)

import gradio as gr
import pandas as pd

# ── Gradio boolean-schema safety patch ───────────────────────────────────────
# Gradio <5.7 walks JSON Schemas and does `if "const" in schema:` without
# guarding against boolean schemas (valid in JSON Schema spec but not a dict).
# sdk_version is now >=5.25.0 (bug fixed upstream) but keep this as a guard.
try:
    import gradio.utils as _gr_utils
    _orig_get_type = getattr(_gr_utils, "get_type", None)
    if _orig_get_type:
        def _safe_get_type(schema, *a, **kw):
            if not isinstance(schema, dict):
                return "other"
            return _orig_get_type(schema, *a, **kw)
        _gr_utils.get_type = _safe_get_type
except Exception:
    pass
try:
    import gradio.route_utils as _gr_ru
    for _fn_name in ("get_type", "_json_schema_to_python_type", "json_schema_to_python_type"):
        _fn = getattr(_gr_ru, _fn_name, None)
        if _fn:
            def _safe_fn(schema, *a, _f=_fn, **kw):
                if not isinstance(schema, dict):
                    return "other"
                return _f(schema, *a, **kw)
            setattr(_gr_ru, _fn_name, _safe_fn)
except Exception:
    pass

from src.config import get_settings
from src.form_config import CREATININE_PROMINENT_SITES, SITE_SPECIFIC_FIELDS, SUSPECTED_SOURCE_OPTIONS
from src.loader import run_inference  # noqa: F401 – registers @spaces.GPU with ZeroGPU at startup
from src.tools import (
    calculate_mic_trend,
    get_empirical_therapy_guidance,
    get_most_effective_antibiotics,
    interpret_mic_value,
    screen_antibiotic_safety,
    search_clinical_guidelines,
)

# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
body, .gradio-container { font-family: 'Inter', sans-serif !important; }

.med-banner {
    background: linear-gradient(135deg, #0b2545 0%, #1a4a8a 100%);
    padding: 22px 30px; border-radius: 12px; margin-bottom: 20px;
}
.med-banner h1 { color: #fff; font-size: 1.9rem; font-weight: 700; margin: 0; }
.med-banner p  { color: #9ec4f0; font-size: 0.95rem; margin: 4px 0 0; }

.section-title {
    font-size: 1.1rem; font-weight: 600; color: #6b8fc4;
    border-bottom: 2px solid #1a4a8a; padding-bottom: 6px; margin: 16px 0 12px;
}

.stat-cards {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px;
}
.stat-card {
    background: #fff; border: 1px solid #dde4ee; border-top: 3px solid #1a4a8a;
    border-radius: 10px; padding: 18px 20px; text-align: center;
}
.stat-card .label { color: #6b7a99; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.04em; }
.stat-card .value { color: #0b2545; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
.stat-card .sub   { color: #9ec4f0; font-size: 0.75rem; margin-top: 2px; }

.agent-step {
    background: #f4f7fc; border-left: 4px solid #1a4a8a;
    border-radius: 8px; padding: 14px 16px; margin-bottom: 10px;
}
.agent-step .num  { color: #1a4a8a; font-weight: 700; font-size: 0.85rem; }
.agent-step .name { color: #0b2545; font-weight: 600; }
.agent-step .desc { color: #5a6680; font-size: 0.85rem; margin-top: 4px; }

.badge-high     { background:#fff0f0; border-left:4px solid #c0392b; color:#7b1d1d;
    padding:10px 14px; border-radius:6px; margin-bottom:6px; }
.badge-moderate { background:#fff8ee; border-left:4px solid #e67e22; color:#7a4a00;
    padding:10px 14px; border-radius:6px; margin-bottom:6px; }
.badge-low      { background:#f0fff4; border-left:4px solid #27ae60; color:#145a32;
    padding:10px 14px; border-radius:6px; margin-bottom:6px; }
.badge-info     { background:#eaf3ff; border-left:4px solid #1a4a8a; color:#0b2545;
    padding:10px 14px; border-radius:6px; margin-bottom:6px; }

.rx-card {
    background: #f4f7fc; border: 1px solid #c5d3e8; border-radius: 10px;
    padding: 22px 24px; font-size: 0.9rem; line-height: 1.7;
}
.rx-card .rx-symbol { font-size: 2rem; color: #1a4a8a; font-weight: 700; }
.rx-card .rx-drug   { font-size: 1.2rem; font-weight: 700; color: #0b2545; }

.disclaimer {
    background: #fff8ee; border: 1px solid #f0c080; border-radius: 8px;
    padding: 12px 16px; font-size: 0.78rem; color: #7a5000; margin-top: 20px;
}
"""

BANNER_HTML = """
<div class="med-banner">
  <div>
    <h1>⚕ AMR-Guard</h1>
    <p>Infection Lifecycle Orchestrator &nbsp;·&nbsp; Multi-Agent Clinical Decision Support</p>
  </div>
</div>
"""

INFECTION_SITES = ["urinary", "respiratory", "bloodstream", "skin", "intra-abdominal", "CNS", "other"]


# ── HTML result builders ───────────────────────────────────────────────────────

def _parse_notes(raw):
    if not raw or raw in ("No lab data provided", "No MIC data available for trend analysis", ""):
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return None


def _build_rec_html(result: dict) -> str:
    rec = result.get("recommendation") or {}
    if not rec:
        return '<div class="badge-info">No recommendation generated.</div>'
    primary  = rec.get("primary_antibiotic", "—")
    dose     = rec.get("dose", "—")
    route    = rec.get("route", "—")
    freq     = rec.get("frequency", "—")
    duration = rec.get("duration", "—")
    alt      = rec.get("backup_antibiotic", "")
    rationale = rec.get("rationale", "")
    refs     = rec.get("references", [])
    alt_html = f"<br><strong>Alternative:</strong> {alt}" if alt else ""
    rat_html = f"<br><br><strong>Clinical rationale</strong><br>{rationale}" if rationale else ""
    ref_html = ""
    if refs:
        items = "".join(f"<li>{r}</li>" for r in refs)
        ref_html = f"<br><strong>References</strong><ul style='margin:4px 0 0 16px'>{items}</ul>"
    return f"""
<div class="rx-card">
  <div class="rx-symbol">℞</div>
  <div class="rx-drug">{primary}</div><br>
  <strong>Dose:</strong> {dose} &nbsp;·&nbsp;
  <strong>Route:</strong> {route} &nbsp;·&nbsp;
  <strong>Frequency:</strong> {freq} &nbsp;·&nbsp;
  <strong>Duration:</strong> {duration}
  {alt_html}{rat_html}{ref_html}
</div>"""


def _build_intake_html(result: dict) -> str:
    intake = _parse_notes(result.get("intake_notes", ""))
    crcl   = result.get("creatinine_clearance_ml_min")
    html   = ""
    if isinstance(intake, dict):
        v       = crcl or intake.get("creatinine_clearance_ml_min", 0)
        sev     = intake.get("infection_severity", "")
        pathway = intake.get("recommended_stage", "")
        cells   = ""
        if v:
            cells += f"<td style='padding:8px 16px 8px 0'><strong>CrCl</strong><br>{float(v):.1f} mL/min</td>"
        if sev:
            cells += f"<td style='padding:8px 16px'><strong>Severity</strong><br>{sev.capitalize()}</td>"
        if pathway:
            cells += f"<td style='padding:8px 16px'><strong>Pathway</strong><br>{pathway.capitalize()}</td>"
        if cells:
            html += f"<table style='margin-bottom:12px'><tr>{cells}</tr></table>"
        if intake.get("patient_summary"):
            html += f'<div class="badge-info">{intake["patient_summary"]}</div>'
        if intake.get("renal_dose_adjustment_needed"):
            html += '<div class="badge-moderate" style="margin-top:8px">⚠ Renal dose adjustment required</div>'
        if intake.get("identified_risk_factors"):
            items = "".join(f"<li>{rf}</li>" for rf in intake["identified_risk_factors"])
            html += f"<br><strong>Identified risk factors</strong><ul style='margin:4px 0 0 16px'>{items}</ul>"
    elif crcl:
        html = f"<strong>CrCl:</strong> {float(crcl):.1f} mL/min"
    else:
        html = '<div class="badge-info">Intake summary not available.</div>'
    return html


def _build_lab_html_and_df(result: dict) -> tuple[str, pd.DataFrame]:
    vision = _parse_notes(result.get("vision_notes", ""))
    trend  = _parse_notes(result.get("trend_notes", ""))
    html   = ""
    df     = pd.DataFrame()

    if vision is None:
        html += '<div class="badge-info">No lab data processed. Provide lab results to activate the targeted pathway.</div>'
    else:
        v = vision if isinstance(vision, dict) else {}
        if v.get("specimen_type"):
            html += f"<strong>Specimen:</strong> {v['specimen_type'].capitalize()}<br>"
        if v.get("extraction_confidence") is not None:
            conf  = float(v["extraction_confidence"])
            color = "#27ae60" if conf >= 0.85 else "#e67e22" if conf >= 0.6 else "#c0392b"
            html += (f'<div class="badge-info">Extraction confidence: '
                     f'<span style="color:{color};font-weight:700">{conf:.0%}</span></div>')
        orgs = v.get("identified_organisms", [])
        if orgs:
            items = "".join(
                f"<li><strong>{o.get('organism_name','?')}</strong>"
                + (f" — {o.get('significance','')}" if o.get("significance") else "")
                + "</li>"
                for o in orgs
            )
            html += f"<br><strong>Identified organisms</strong><ul style='margin:4px 0 0 16px'>{items}</ul>"
        sus = v.get("susceptibility_results", [])
        if sus:
            rows = [
                {
                    "Organism": e.get("organism", ""),
                    "Antibiotic": e.get("antibiotic", ""),
                    "MIC (mg/L)": str(e.get("mic_value", "")),
                    "Result": e.get("interpretation", ""),
                }
                for e in sus
            ]
            df = pd.DataFrame(rows)

    if trend:
        html += "<hr><strong>MIC Trend Analysis</strong><br>"
        items = trend if isinstance(trend, list) else [trend]
        for item in items:
            if not isinstance(item, dict):
                html += f"<p>{item}</p>"
                continue
            risk = item.get("risk_level", "UNKNOWN").upper()
            css  = {"HIGH": "badge-high", "MODERATE": "badge-moderate"}.get(risk, "badge-low")
            icon = {"HIGH": "🚨", "MODERATE": "⚠"}.get(risk, "✓")
            org  = item.get("organism", "")
            ab   = item.get("antibiotic", "")
            label = f"{org} / {ab} — " if (org or ab) else ""
            html += (f'<div class="{css}">{icon} <strong>{label}{risk}</strong><br>'
                     f'<span style="font-size:0.88rem">{item.get("recommendation","")}</span></div>')
    return html, df


def _build_safety_html(result: dict) -> str:
    warnings = result.get("safety_warnings", [])
    errors   = result.get("errors", [])
    html = "".join(f'<div class="badge-high">⚠ {w}</div>' for w in warnings)
    if not warnings:
        html = '<div class="badge-low">✓ No safety concerns identified.</div>'
    html += "".join(f'<div class="badge-high" style="margin-top:6px">Error: {e}</div>' for e in errors)
    return html


def _demo_result(patient_data: dict, has_labs: bool) -> dict:
    result = {
        "stage": "targeted" if has_labs else "empirical",
        "creatinine_clearance_ml_min": 58.3,
        "intake_notes": json.dumps({
            "patient_summary": (
                f"{patient_data.get('age_years')}-year-old {patient_data.get('sex')} "
                f"· {patient_data.get('suspected_source', 'infection')}"
            ),
            "creatinine_clearance_ml_min": 58.3,
            "renal_dose_adjustment_needed": True,
            "identified_risk_factors": patient_data.get("comorbidities", []),
            "infection_severity": "moderate",
            "recommended_stage": "targeted" if has_labs else "empirical",
        }),
        "recommendation": {
            "primary_antibiotic": "Ciprofloxacin",
            "dose": "500 mg",
            "route": "Oral",
            "frequency": "Every 12 hours",
            "duration": "7 days",
            "backup_antibiotic": "Nitrofurantoin 100 mg MR BD × 5 days",
            "rationale": (
                "Community-acquired UTI with moderate renal impairment (CrCl 58 mL/min). "
                "Ciprofloxacin provides broad Gram-negative coverage. "
                "No dose adjustment required above CrCl 30 mL/min."
            ),
            "references": ["IDSA UTI Guidelines 2024", "EUCAST Breakpoint Tables v16.0"],
        },
        "safety_warnings": [],
        "errors": [],
    }
    if has_labs:
        result["vision_notes"] = json.dumps({
            "specimen_type": "urine",
            "identified_organisms": [{"organism_name": "Escherichia coli", "significance": "pathogen"}],
            "susceptibility_results": [
                {"organism": "E. coli", "antibiotic": "Ciprofloxacin", "mic_value": 0.25, "interpretation": "S"},
                {"organism": "E. coli", "antibiotic": "Nitrofurantoin", "mic_value": 16, "interpretation": "S"},
                {"organism": "E. coli", "antibiotic": "Ampicillin", "mic_value": ">32", "interpretation": "R"},
            ],
            "extraction_confidence": 0.95,
        })
        result["trend_notes"] = json.dumps([{
            "organism": "E. coli", "antibiotic": "Ciprofloxacin",
            "risk_level": "LOW", "recommendation": "No MIC creep detected.",
        }])
    return result


# ── Site change / lab method handlers ─────────────────────────────────────────

def update_site_ui(site):
    grp_updates  = [gr.update(visible=(s == site)) for s in INFECTION_SITES]
    src_choices  = SUSPECTED_SOURCE_OPTIONS.get(site, []) or ["Other"]
    prominent    = site in CREATININE_PROMINENT_SITES
    return (
        *grp_updates,
        gr.update(choices=src_choices, value=src_choices[0]),
        gr.update(visible=prominent),       # creatinine_main
        gr.update(visible=not prominent),   # renal_flag
        gr.update(visible=False),           # creatinine_optional (reset hidden)
    )


def toggle_optional_creatinine(flag):
    return gr.update(visible=bool(flag))


def toggle_lab_inputs(method):
    return (
        gr.update(visible=(method == "Upload file (PDF / image)")),
        gr.update(visible=(method == "Paste lab text")),
    )


# ── Pipeline function ──────────────────────────────────────────────────────────
# Site-specific field order (matches component creation order in the Blocks):
#   urinary      : sf0  sf1  sf2                          (3 fields)
#   respiratory  : sf3  sf4  sf5  sf6                     (4 fields)
#   bloodstream  : sf7  sf8  sf9  sf10  sf11  sf12  sf13  (7 fields)
#   skin         : sf14 sf15 sf16 sf17                    (4 fields)
#   intra-abdom  : sf18 sf19 sf20 sf21                    (4 fields)
#   CNS          : sf22 sf23 sf24 sf25                    (4 fields)

def run_pipeline_ui(
    age, weight, height, sex,
    creatinine_main, renal_flag, creatinine_optional,
    infection_site, suspected_source,
    # urinary
    sf0, sf1, sf2,
    # respiratory
    sf3, sf4, sf5, sf6,
    # bloodstream
    sf7, sf8, sf9, sf10, sf11, sf12, sf13,
    # skin
    sf14, sf15, sf16, sf17,
    # intra-abdominal
    sf18, sf19, sf20, sf21,
    # CNS
    sf22, sf23, sf24, sf25,
    # medical history
    medications, allergies, comorbidities, risk_factors,
    # lab
    lab_method, lab_file, lab_paste,
    progress=gr.Progress(),
):
    # Creatinine
    if infection_site in CREATININE_PROMINENT_SITES:
        creatinine = creatinine_main
    else:
        creatinine = creatinine_optional if renal_flag else None

    # Site-specific vitals
    site_vitals: dict = {}
    if infection_site == "urinary":
        site_vitals = {
            "catheter_status": str(sf0 or ""),
            "urinary_symptoms": ", ".join(sf1) if sf1 else "",
            "urine_appearance": str(sf2 or ""),
        }
    elif infection_site == "respiratory":
        site_vitals = {
            "o2_saturation": str(sf3 or ""),
            "ventilation_status": str(sf4 or ""),
            "cough_type": str(sf5 or ""),
            "sputum_character": str(sf6 or ""),
        }
    elif infection_site == "bloodstream":
        site_vitals = {
            "central_line_present": "Yes" if sf7 else "No",
            "temperature_c": str(sf8 or ""),
            "heart_rate_bpm": str(sf9 or ""),
            "respiratory_rate": str(sf10 or ""),
            "wbc_count": str(sf11 or ""),
            "lactate_mmol": str(sf12 or ""),
            "shock_status": str(sf13 or ""),
        }
    elif infection_site == "skin":
        site_vitals = {
            "wound_type": str(sf14 or ""),
            "cellulitis_extent": str(sf15 or ""),
            "abscess_present": "Yes" if sf16 else "No",
            "foreign_body": "Yes" if sf17 else "No",
        }
    elif infection_site == "intra-abdominal":
        site_vitals = {
            "abdominal_pain_location": str(sf18 or ""),
            "peritonitis_signs": ", ".join(sf19) if sf19 else "",
            "perforation_suspected": "Yes" if sf20 else "No",
            "ascites": "Yes" if sf21 else "No",
        }
    elif infection_site == "CNS":
        site_vitals = {
            "csf_obtained": "Yes" if sf22 else "No",
            "neuro_symptoms": ", ".join(sf23) if sf23 else "",
            "recent_neurosurgery": "Yes" if sf24 else "No",
            "gcs_score": str(sf25 or ""),
        }

    # Lab file handling
    labs_raw_text   = None
    labs_image_bytes = None
    if lab_method == "Upload file (PDF / image)" and lab_file is not None:
        file_path = lab_file if isinstance(lab_file, str) else lab_file.name
        ext = file_path.rsplit(".", 1)[-1].lower()
        with open(file_path, "rb") as fh:
            file_bytes = fh.read()
        if ext == "pdf":
            try:
                import pypdf
                reader   = pypdf.PdfReader(BytesIO(file_bytes))
                extracted = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
                if extracted:
                    labs_raw_text = extracted
                else:
                    labs_image_bytes = file_bytes
            except Exception:
                labs_image_bytes = file_bytes
        else:
            labs_image_bytes = file_bytes
    elif lab_method == "Paste lab text" and lab_paste:
        labs_raw_text = lab_paste.strip() or None

    patient_data = {
        "age_years":            float(age or 65),
        "weight_kg":            float(weight or 70),
        "height_cm":            float(height or 170),
        "sex":                  sex or "male",
        "serum_creatinine_mg_dl": float(creatinine) if creatinine else None,
        "infection_site":       infection_site,
        "suspected_source":     suspected_source or f"{infection_site} infection",
        "medications":          [m.strip() for m in (medications or "").split("\n") if m.strip()],
        "allergies":            [a.strip() for a in (allergies or "").split("\n") if a.strip()],
        "comorbidities":        list(comorbidities or []) + list(risk_factors or []),
        "vitals":               site_vitals,
        "labs_image_bytes":     labs_image_bytes,
    }

    has_labs = bool(labs_raw_text or labs_image_bytes)
    stages   = (
        ["Intake Historian", "Vision Specialist", "Trend Analyst", "Clinical Pharmacologist"]
        if has_labs else ["Intake Historian", "Clinical Pharmacologist"]
    )
    for i, name in enumerate(stages):
        progress((i + 0.5) / len(stages), desc=f"Running: {name}…")

    try:
        from src.graph import run_pipeline
        result = run_pipeline(patient_data, labs_raw_text)
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("Pipeline failed — falling back to demo result.\n%s", tb)
        result = _demo_result(patient_data, has_labs)
        result["errors"].append(f"Pipeline error: {e}")
        result["recommendation"] = {}  # suppress the hardcoded drug from showing

    progress(1.0, desc="Complete")

    rec_html          = _build_rec_html(result)
    intake_html       = _build_intake_html(result)
    lab_html, lab_df  = _build_lab_html_and_df(result)
    safety_html       = _build_safety_html(result)

    return rec_html, intake_html, lab_html, lab_df, safety_html, gr.update(visible=True)


# ── Clinical Tools handlers ────────────────────────────────────────────────────

def switch_tool(tool):
    tools = ["Empirical Advisor", "MIC Interpreter", "MIC Trend Analysis", "Drug Safety Check"]
    return [gr.update(visible=(t == tool)) for t in tools]


def run_empirical(infection_type, pathogen, risk):
    guidance = get_empirical_therapy_guidance(infection_type, list(risk or []))
    html = ""
    for i, rec in enumerate(guidance.get("recommendations", [])[:3], 1):
        score   = rec.get("relevance_score", 0)
        content = rec.get("content", "")
        source  = rec.get("source", "IDSA Guidelines 2024")
        html += (f'<div class="badge-info"><strong>Excerpt {i}</strong>'
                 f' (relevance {score:.2f})<br>{content}<br><em>Source: {source}</em></div>')
    if pathogen:
        effective = get_most_effective_antibiotics(pathogen, min_susceptibility=70)
        if effective:
            items = "".join(
                f"<li><strong>{ab.get('antibiotic')}</strong>"
                f" — {ab.get('avg_susceptibility', 0):.1f}% susceptible</li>"
                for ab in effective[:6]
            )
            html += f"<br><strong>Resistance data — {pathogen}</strong><ul style='margin:4px 0 0 16px'>{items}</ul>"
        else:
            html += '<div class="badge-info">No resistance data available for this pathogen.</div>'
    return html or '<div class="badge-info">No results found.</div>'


def run_mic_interpret(pathogen, antibiotic, mic):
    if not pathogen or not antibiotic:
        return '<div class="badge-info">Enter pathogen and antibiotic.</div>'
    result = interpret_mic_value(pathogen, antibiotic, float(mic or 1.0))
    interp = result.get("interpretation", "UNKNOWN")
    msg    = result.get("message", "")
    if interp == "SUSCEPTIBLE":
        return f'<div class="badge-low"><strong>Susceptible (S)</strong> — {msg}</div>'
    if interp == "RESISTANT":
        return f'<div class="badge-high"><strong>Resistant (R)</strong> — {msg}</div>'
    return f'<div class="badge-moderate"><strong>Intermediate (I)</strong> — {msg}</div>'


def update_mic_inputs(n):
    return [gr.update(visible=(i < int(n))) for i in range(6)]


def run_mic_trend(n, m0, m1, m2, m3, m4, m5):
    vals = [m0, m1, m2, m3, m4, m5][: int(n)]
    mic_values = [{"date": f"T{i}", "mic_value": float(v or 1.0)} for i, v in enumerate(vals)]
    result = calculate_mic_trend(mic_values)
    risk   = result.get("risk_level", "UNKNOWN")
    alert  = result.get("alert", "")
    css    = {"HIGH": "badge-high", "MODERATE": "badge-moderate"}.get(risk, "badge-low")
    icon   = {"HIGH": "🚨", "MODERATE": "⚠"}.get(risk, "✓")
    base   = result.get("baseline_mic", "—")
    curr   = result.get("current_mic", "—")
    ratio  = result.get("ratio", "—")
    return f"""
<div class="{css}">{icon} <strong>{risk} RISK</strong> — {alert}</div>
<br>
<table><tr>
<td style='padding:8px 24px 8px 0'><strong>Baseline MIC</strong><br>{base} mg/L</td>
<td style='padding:8px 24px'><strong>Current MIC</strong><br>{curr} mg/L</td>
<td style='padding:8px 24px'><strong>Fold change</strong><br>{ratio}×</td>
</tr></table>"""


def run_drug_safety(ab, meds, allergies_txt):
    if not ab:
        return '<div class="badge-info">Enter an antibiotic to check.</div>'
    med_list     = [m.strip() for m in (meds or "").split("\n") if m.strip()]
    allergy_list = [a.strip() for a in (allergies_txt or "").split("\n") if a.strip()]
    result = screen_antibiotic_safety(ab, med_list, allergy_list)
    if result.get("safe_to_use"):
        html = '<div class="badge-low">✓ No critical safety concerns identified.</div>'
    else:
        html = '<div class="badge-high">⚠ Safety concerns identified — review required.</div>'
    html += "".join(
        f'<div class="badge-moderate" style="margin-top:8px">⚠ {a.get("message","")}</div>'
        for a in result.get("alerts", [])
    )
    return html


def run_guidelines_search(query, pathogen_filter):
    if not query:
        return '<div class="badge-info">Enter a search query.</div>'
    filt    = None if pathogen_filter == "All" else pathogen_filter
    results = search_clinical_guidelines(query, pathogen_filter=filt, n_results=5)
    if not results:
        return ('<div class="badge-info">No results found. Try broader search terms or '
                'check that the knowledge base has been initialised.</div>')
    html = ""
    for i, r in enumerate(results, 1):
        score   = r.get("relevance_score", 0)
        content = r.get("content", "")
        source  = r.get("source", "")
        src_str = f"<br><em>Source: {source}</em>" if source else ""
        html += (f'<div class="badge-info"><strong>Result {i}</strong>'
                 f' · relevance {score:.2f}<br>{content}{src_str}</div>')
    return html


# ── Widget factory for site-specific fields ────────────────────────────────────

def _make_site_widget(field):
    ftype = field["type"]
    label = field["label"]
    if ftype == "selectbox":
        return gr.Dropdown(choices=field["options"], value=field["options"][0], label=label)
    if ftype == "multiselect":
        return gr.CheckboxGroup(choices=field["options"], label=label)
    if ftype == "number_input":
        return gr.Number(
            value=field.get("default", 0), label=label,
            minimum=field.get("min"), maximum=field.get("max"),
        )
    if ftype == "checkbox":
        return gr.Checkbox(value=field.get("default", False), label=label)
    return gr.Textbox(label=label)


# ── Models table (build-time) ─────────────────────────────────────────────────

_s = get_settings()
OVERVIEW_MODELS_MD = f"""
| Agent | Role | Model |
|---|---|---|
| 1, 2, 4 | Clinical reasoning | `{_s.medgemma_4b_model or "google/medgemma-4b-it"}` |
| 3 | Trend analysis | `{_s.medgemma_27b_model or "google/medgemma-27b-text-it"}` |
| 4 (safety) | Pharmacology check | `{_s.txgemma_9b_model or "google/txgemma-9b-predict"}` |
| — | Semantic retrieval | `{_s.embedding_model_name}` |
| — | Inference backend | HuggingFace Transformers · {_s.quantization} quant |
"""

# ── Gradio Blocks ─────────────────────────────────────────────────────────────

with gr.Blocks(theme=gr.themes.Soft(), css=CSS, title="AMR-Guard") as demo:
    gr.HTML(BANNER_HTML)

    with gr.Tabs():

        # ── Tab 1: Overview ────────────────────────────────────────────────────
        with gr.Tab("Overview"):
            gr.HTML("""
<div class="section-title">System Overview</div>
<div class="stat-cards">
  <div class="stat-card">
    <div class="label">WHO AWaRe</div><div class="value">264</div><div class="sub">antibiotics classified</div>
  </div>
  <div class="stat-card">
    <div class="label">EUCAST</div><div class="value">v16.0</div><div class="sub">breakpoint tables</div>
  </div>
  <div class="stat-card">
    <div class="label">IDSA</div><div class="value">2024</div><div class="sub">treatment guidelines</div>
  </div>
  <div class="stat-card">
    <div class="label">DDInter</div><div class="value">191K+</div><div class="sub">drug interactions</div>
  </div>
</div>
<div class="section-title">Agent Pipeline</div>
""")
            with gr.Row():
                with gr.Column():
                    gr.HTML("""
<p><strong>Stage 1 — Empirical</strong> <em>(no lab results yet)</em></p>
<div class="agent-step"><div class="num">Agent 01</div><div class="name">Intake Historian</div>
<div class="desc">Parses patient data, calculates CrCl, identifies MDR risk factors</div></div>
<div class="agent-step"><div class="num">Agent 04</div><div class="name">Clinical Pharmacologist</div>
<div class="desc">Empirical antibiotic selection · WHO AWaRe · safety screening</div></div>
""")
                with gr.Column():
                    gr.HTML("""
<p><strong>Stage 2 — Targeted</strong> <em>(culture / sensitivity available)</em></p>
<div class="agent-step"><div class="num">Agent 01</div><div class="name">Intake Historian</div>
<div class="desc">Same as Stage 1</div></div>
<div class="agent-step"><div class="num">Agent 02</div><div class="name">Vision Specialist</div>
<div class="desc">Extracts structured data from lab reports (any language / format)</div></div>
<div class="agent-step"><div class="num">Agent 03</div><div class="name">Trend Analyst</div>
<div class="desc">Detects MIC creep · calculates resistance velocity</div></div>
<div class="agent-step"><div class="num">Agent 04</div><div class="name">Clinical Pharmacologist</div>
<div class="desc">Targeted recommendation informed by susceptibility data</div></div>
""")
            gr.HTML('<div class="section-title">AI Models (Local)</div>')
            gr.Markdown(OVERVIEW_MODELS_MD)
            gr.HTML(
                '<div class="disclaimer">⚠ <strong>Research demo only.</strong> '
                "Not validated for clinical use. All recommendations must be reviewed "
                "by a licensed clinician before any patient-care decision.</div>"
            )

        # ── Tab 2: Patient Analysis ────────────────────────────────────────────
        with gr.Tab("Patient Analysis"):
            gr.HTML('<div class="section-title">Patient Analysis Pipeline</div>')

            # Demographics row
            with gr.Row():
                with gr.Column(scale=1):
                    age    = gr.Number(value=65,   label="Age (years)",   minimum=0,   maximum=120, precision=0)
                    weight = gr.Number(value=70.0, label="Weight (kg)",   minimum=1.0, maximum=300.0)
                    height = gr.Number(value=170.0,label="Height (cm)",   minimum=50.0,maximum=250.0)
                with gr.Column(scale=1):
                    sex               = gr.Dropdown(choices=["male", "female"], value="male", label="Biological sex")
                    creatinine_main   = gr.Number(value=1.2, label="Serum Creatinine (mg/dL)",
                                                  minimum=0.1, maximum=20.0, visible=True)
                    renal_flag        = gr.Checkbox(label="Known renal impairment / CKD?", visible=False)
                    creatinine_optional = gr.Number(value=1.2, label="Serum Creatinine (mg/dL)",
                                                    minimum=0.1, maximum=20.0, visible=False)
                with gr.Column(scale=1):
                    infection_site  = gr.Dropdown(choices=INFECTION_SITES, value="urinary",
                                                  label="Primary infection site")
                    _init_src = SUSPECTED_SOURCE_OPTIONS.get("urinary", [])
                    suspected_source = gr.Dropdown(choices=_init_src,
                                                   value=_init_src[0] if _init_src else None,
                                                   label="Suspected source")

            # Site-specific field groups (pre-rendered, one per site)
            site_groups: dict = {}
            # Component lists per site (in field declaration order)
            u_comps:  list = []  # 3 components
            r_comps:  list = []  # 4 components
            b_comps:  list = []  # 7 components
            sk_comps: list = []  # 4 components
            ia_comps: list = []  # 4 components
            cn_comps: list = []  # 4 components

            for site in INFECTION_SITES:
                fields = SITE_SPECIFIC_FIELDS.get(site, [])
                with gr.Group(visible=(site == "urinary")) as grp:
                    if fields:
                        gr.HTML(f'<div class="section-title">{site.title()} — Assessment</div>')
                        with gr.Row():
                            for field in fields:
                                comp = _make_site_widget(field)
                                if site == "urinary":
                                    u_comps.append(comp)
                                elif site == "respiratory":
                                    r_comps.append(comp)
                                elif site == "bloodstream":
                                    b_comps.append(comp)
                                elif site == "skin":
                                    sk_comps.append(comp)
                                elif site == "intra-abdominal":
                                    ia_comps.append(comp)
                                elif site == "CNS":
                                    cn_comps.append(comp)
                site_groups[site] = grp

            # Flatten all site components in fixed order for fn inputs
            all_site_inputs = u_comps + r_comps + b_comps + sk_comps + ia_comps + cn_comps

            # Medical history
            gr.HTML('<div class="section-title">Medical History</div>')
            with gr.Row():
                with gr.Column():
                    medications = gr.Textbox(
                        label="Current medications (one per line)",
                        placeholder="Metformin\nLisinopril", lines=4,
                    )
                    allergies = gr.Textbox(
                        label="Drug allergies (one per line)",
                        placeholder="Penicillin\nSulfa", lines=3,
                    )
                with gr.Column():
                    comorbidities = gr.CheckboxGroup(
                        choices=["Diabetes", "CKD", "Heart Failure", "COPD",
                                 "Immunocompromised", "Recent Surgery", "Malignancy", "Liver Disease"],
                        label="Comorbidities",
                    )
                    risk_factors = gr.CheckboxGroup(
                        choices=["Prior MRSA", "Recent antibiotics (<90 d)", "Healthcare-associated",
                                 "Recent hospitalisation", "Nursing home", "Prior MDR infection"],
                        label="MDR risk factors",
                    )

            # Lab input
            gr.HTML('<div class="section-title">Lab / Culture Results '
                    '<small>(optional — triggers targeted pathway)</small></div>')
            lab_method = gr.Radio(
                choices=["None — empirical pathway only", "Upload file (PDF / image)", "Paste lab text"],
                value="None — empirical pathway only",
                label="Input method",
            )
            lab_file  = gr.File(
                label="Lab report",
                file_types=[".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"],
                visible=False,
            )
            lab_paste = gr.Textbox(
                label="Lab report text",
                placeholder=(
                    "Organism: Escherichia coli\n"
                    "Ciprofloxacin: S  MIC 0.25\n"
                    "Nitrofurantoin: S  MIC 16\n"
                    "Ampicillin: R  MIC >32"
                ),
                lines=5, visible=False,
            )

            run_btn = gr.Button("Run Agent Pipeline", variant="primary")

            # Results (hidden until pipeline completes)
            with gr.Group(visible=False) as results_group:
                gr.HTML('<div class="section-title">Results</div>')
                with gr.Tabs():
                    with gr.Tab("Recommendation"):
                        rec_out    = gr.HTML()
                    with gr.Tab("Patient Summary"):
                        intake_out = gr.HTML()
                    with gr.Tab("Lab Analysis"):
                        lab_html_out = gr.HTML()
                        lab_df_out   = gr.DataFrame(label="Susceptibility Table", wrap=True)
                    with gr.Tab("Safety"):
                        safety_out = gr.HTML()

            # ── Wiring ──
            infection_site.change(
                fn=update_site_ui,
                inputs=[infection_site],
                outputs=[
                    *[site_groups[s] for s in INFECTION_SITES],
                    suspected_source,
                    creatinine_main,
                    renal_flag,
                    creatinine_optional,
                ],
            )
            renal_flag.change(
                fn=toggle_optional_creatinine,
                inputs=[renal_flag],
                outputs=[creatinine_optional],
            )
            lab_method.change(
                fn=toggle_lab_inputs,
                inputs=[lab_method],
                outputs=[lab_file, lab_paste],
            )
            run_btn.click(
                fn=run_pipeline_ui,
                inputs=[
                    age, weight, height, sex,
                    creatinine_main, renal_flag, creatinine_optional,
                    infection_site, suspected_source,
                    *all_site_inputs,
                    medications, allergies, comorbidities, risk_factors,
                    lab_method, lab_file, lab_paste,
                ],
                outputs=[rec_out, intake_out, lab_html_out, lab_df_out, safety_out, results_group],
            )

        # ── Tab 3: Clinical Tools ──────────────────────────────────────────────
        with gr.Tab("Clinical Tools"):
            gr.HTML('<div class="section-title">Clinical Tools</div>')
            tool_sel = gr.Dropdown(
                choices=["Empirical Advisor", "MIC Interpreter", "MIC Trend Analysis", "Drug Safety Check"],
                value="Empirical Advisor",
                label="Select tool",
            )

            # Empirical Advisor
            with gr.Group(visible=True) as grp_ea:
                with gr.Row():
                    with gr.Column(scale=3):
                        ea_infection = gr.Dropdown(
                            choices=["Urinary Tract Infection", "Pneumonia", "Sepsis",
                                     "Skin / Soft Tissue", "Intra-abdominal", "Meningitis"],
                            value="Urinary Tract Infection", label="Infection type",
                        )
                        ea_pathogen = gr.Textbox(
                            label="Suspected pathogen (optional)",
                            placeholder="e.g., Klebsiella pneumoniae",
                        )
                        ea_risk = gr.CheckboxGroup(
                            choices=["Prior MRSA", "Recent antibiotics (<90 d)", "Healthcare-associated",
                                     "Immunocompromised", "Renal impairment", "Prior MDR"],
                            label="Risk factors",
                        )
                    with gr.Column(scale=1):
                        gr.HTML("""
<div class="badge-info"><strong>WHO AWaRe</strong><br>
<span style="color:#145a32">●</span> Access — first-line<br>
<span style="color:#7a4a00">●</span> Watch — second-line<br>
<span style="color:#7b1d1d">●</span> Reserve — last resort</div>""")
                ea_btn = gr.Button("Get recommendation", variant="primary")
                ea_out = gr.HTML()

            # MIC Interpreter
            with gr.Group(visible=False) as grp_mi:
                with gr.Row():
                    with gr.Column():
                        mi_pathogen  = gr.Textbox(label="Pathogen",   placeholder="e.g., Escherichia coli")
                        mi_antibiotic= gr.Textbox(label="Antibiotic", placeholder="e.g., Ciprofloxacin")
                        mi_mic       = gr.Number(value=1.0, label="MIC value (mg/L)", minimum=0.001, maximum=1024.0)
                    with gr.Column():
                        gr.HTML("""
<div class="badge-info" style="margin-top:28px"><strong>Interpretation guide</strong><br><br>
<strong>S</strong> Susceptible — antibiotic is effective<br>
<strong>I</strong> Intermediate — effective at higher doses<br>
<strong>R</strong> Resistant — do not use</div>""")
                mi_btn = gr.Button("Interpret", variant="primary")
                mi_out = gr.HTML()

            # MIC Trend Analysis
            with gr.Group(visible=False) as grp_mt:
                mt_n = gr.Slider(minimum=2, maximum=6, value=3, step=1,
                                 label="Number of historical readings")
                with gr.Row():
                    mt_m = [
                        gr.Number(value=float(2 ** i), label=f"MIC {i+1} (mg/L)",
                                  minimum=0.001, maximum=256.0, visible=(i < 3))
                        for i in range(6)
                    ]
                mt_btn = gr.Button("Analyse trend", variant="primary")
                mt_out = gr.HTML()
                mt_n.change(fn=update_mic_inputs, inputs=[mt_n], outputs=mt_m)

            # Drug Safety Check
            with gr.Group(visible=False) as grp_ds:
                with gr.Row():
                    with gr.Column():
                        ds_ab   = gr.Textbox(label="Antibiotic to check",
                                             placeholder="e.g., Ciprofloxacin")
                        ds_meds = gr.Textbox(label="Concurrent medications",
                                             placeholder="Warfarin\nMetformin\nAmlodipine", lines=4)
                    with gr.Column():
                        ds_allergies = gr.Textbox(label="Known allergies",
                                                  placeholder="Penicillin\nSulfa", lines=3)
                ds_btn = gr.Button("Check safety", variant="primary")
                ds_out = gr.HTML()

            tool_sel.change(
                fn=switch_tool, inputs=[tool_sel],
                outputs=[grp_ea, grp_mi, grp_mt, grp_ds],
            )
            ea_btn.click(fn=run_empirical,     inputs=[ea_infection, ea_pathogen, ea_risk], outputs=[ea_out])
            mi_btn.click(fn=run_mic_interpret, inputs=[mi_pathogen, mi_antibiotic, mi_mic], outputs=[mi_out])
            mt_btn.click(fn=run_mic_trend,     inputs=[mt_n, *mt_m],                       outputs=[mt_out])
            ds_btn.click(fn=run_drug_safety,   inputs=[ds_ab, ds_meds, ds_allergies],      outputs=[ds_out])

        # ── Tab 4: Guidelines ──────────────────────────────────────────────────
        with gr.Tab("Guidelines"):
            gr.HTML('<div class="section-title">Clinical Guidelines Search</div>')
            with gr.Row():
                gl_query  = gr.Textbox(
                    label="Search query",
                    placeholder="e.g., ESBL E. coli UTI treatment carbapenems",
                    scale=3,
                )
                gl_filter = gr.Dropdown(
                    choices=["All", "ESBL-E", "CRE", "CRAB", "DTR-PA"],
                    value="All", label="Filter by pathogen", scale=1,
                )
            gl_btn = gr.Button("Search", variant="primary")
            gl_out = gr.HTML()
            gr.HTML(
                '<div class="disclaimer">Sources: IDSA Treatment Guidelines 2024 · '
                "EUCAST Breakpoint Tables v16.0 · WHO EML · DDInter drug interaction database.</div>"
            )
            gl_btn.click(fn=run_guidelines_search, inputs=[gl_query, gl_filter], outputs=[gl_out])


if __name__ == "__main__":
    demo.launch()
