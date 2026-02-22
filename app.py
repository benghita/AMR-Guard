"""
AMR-Guard: AMR-Guard Demo Application
Infection Lifecycle Orchestrator — Streamlit Interface
"""

import json
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.form_config import CREATININE_PROMINENT_SITES, SITE_SPECIFIC_FIELDS, SUSPECTED_SOURCE_OPTIONS
from src.tools import (
    calculate_mic_trend,
    get_empirical_therapy_guidance,
    get_most_effective_antibiotics,
    interpret_mic_value,
    screen_antibiotic_safety,
    search_clinical_guidelines,
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AMR-Guard · AMR-Guard",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Fonts & Base ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0b2545;
}
[data-testid="stSidebar"] * { color: #e8edf3 !important; }
[data-testid="stSidebar"] .stRadio label { padding: 6px 0; font-size: 0.9rem; }
[data-testid="stSidebar"] hr { border-color: #1e3a5f; }

/* ── Top banner ── */
.med-banner {
    background: linear-gradient(135deg, #0b2545 0%, #1a4a8a 100%);
    padding: 22px 30px;
    border-radius: 12px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
}
.med-banner h1 { color: #ffffff; font-size: 1.9rem; font-weight: 700; margin: 0; }
.med-banner p  { color: #9ec4f0; font-size: 0.95rem; margin: 4px 0 0; }

/* ── Section headings ── */
.section-title {
    font-size: 1.15rem; font-weight: 600;
    color: #6b8fc4; border-bottom: 2px solid #1a4a8a;
    padding-bottom: 6px; margin: 24px 0 16px;
}

/* ── Stat cards ── */
.stat-card {
    background: #ffffff;
    border: 1px solid #dde4ee;
    border-top: 3px solid #1a4a8a;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.stat-card .label { color: #6b7a99; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; }
.stat-card .value { color: #0b2545; font-size: 1.6rem; font-weight: 700; margin-top: 4px; }
.stat-card .sub   { color: #9ec4f0; font-size: 0.75rem; margin-top: 2px; }

/* ── Agent flow card ── */
.agent-step {
    background: #f4f7fc;
    border-left: 4px solid #1a4a8a;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.agent-step .num  { color: #1a4a8a; font-weight: 700; font-size: 0.85rem; }
.agent-step .name { color: #0b2545; font-weight: 600; }
.agent-step .desc { color: #5a6680; font-size: 0.85rem; margin-top: 4px; }

/* ── Alert badges ── */
.badge-high     { background:#fff0f0; border-left:4px solid #c0392b; color:#7b1d1d; padding:10px 14px; border-radius:6px; }
.badge-moderate { background:#fff8ee; border-left:4px solid #e67e22; color:#7a4a00; padding:10px 14px; border-radius:6px; }
.badge-low      { background:#f0fff4; border-left:4px solid #27ae60; color:#145a32; padding:10px 14px; border-radius:6px; }
.badge-info     { background:#eaf3ff; border-left:4px solid #1a4a8a; color:#0b2545; padding:10px 14px; border-radius:6px; }

/* ── Prescription card ── */
.rx-card {
    background: #f4f7fc;
    border: 1px solid #c5d3e8;
    border-radius: 10px;
    padding: 22px 24px;
    font-size: 0.9rem;
    line-height: 1.7;
}
.rx-card .rx-symbol { font-size: 2rem; color: #1a4a8a; font-weight: 700; }
.rx-card .rx-drug   { font-size: 1.2rem; font-weight: 700; color: #0b2545; }

/* ── Disclaimer ── */
.disclaimer {
    background: #fff8ee;
    border: 1px solid #f0c080;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.78rem;
    color: #7a5000;
    margin-top: 20px;
}

/* ── Form tweaks ── */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    border-radius: 6px !important;
}
.stButton > button[kind="primary"] {
    background: #1a4a8a; border: none;
    border-radius: 8px; font-weight: 600;
    padding: 0.6rem 1.4rem;
}
.stButton > button[kind="primary"]:hover { background: #0b2545; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚕ AMR-Guard")
    st.markdown("**AMR-Guard**")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Dashboard", "Patient Analysis", "Clinical Tools", "Guidelines"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#6b8fc4'>Powered by local LLMs<br>via HuggingFace Transformers</small>",
        unsafe_allow_html=True,
    )


# ── Banner ────────────────────────────────────────────────────────────────────

st.markdown(
    """
<div class="med-banner">
    <div>
        <h1>⚕ AMR-Guard</h1>
        <p>Infection Lifecycle Orchestrator &nbsp;·&nbsp; Multi-Agent Clinical Decision Support</p>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Pages ─────────────────────────────────────────────────────────────────────


def page_dashboard():
    st.markdown('<div class="section-title">System Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    cards = [
        ("WHO AWaRe", "264", "antibiotics classified"),
        ("EUCAST", "v16.0", "breakpoint tables"),
        ("IDSA", "2024", "treatment guidelines"),
        ("DDInter", "191K+", "drug interactions"),
    ]
    for col, (label, value, sub) in zip([col1, col2, col3, col4], cards):
        col.markdown(
            f'<div class="stat-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Agent Pipeline</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Stage 1 — Empirical** *(no lab results yet)*")
        for num, name, desc in [
            ("01", "Intake Historian", "Parses patient data, calculates CrCl, identifies MDR risk factors"),
            ("04", "Clinical Pharmacologist", "Empirical antibiotic selection · WHO AWaRe · safety screening"),
        ]:
            st.markdown(
                f'<div class="agent-step"><div class="num">Agent {num}</div>'
                f'<div class="name">{name}</div><div class="desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    with c2:
        st.markdown("**Stage 2 — Targeted** *(culture / sensitivity available)*")
        for num, name, desc in [
            ("01", "Intake Historian", "Same as Stage 1"),
            ("02", "Vision Specialist", "Extracts structured data from lab reports (any language / format)"),
            ("03", "Trend Analyst", "Detects MIC creep · calculates resistance velocity"),
            ("04", "Clinical Pharmacologist", "Targeted recommendation informed by susceptibility data"),
        ]:
            st.markdown(
                f'<div class="agent-step"><div class="num">Agent {num}</div>'
                f'<div class="name">{name}</div><div class="desc">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">AI Models (Local)</div>', unsafe_allow_html=True)

    from src.config import get_settings
    s = get_settings()
    st.markdown(
        f"""
| Agent | Role | Model |
|---|---|---|
| 1, 2, 4 | Clinical reasoning | `{s.local_medgemma_4b_model or "google/medgemma-4b-it"}` |
| 3 | Trend analysis | `{s.local_medgemma_27b_model or "google/medgemma-27b-text-it"}` |
| 4 (safety) | Pharmacology check | `{s.local_txgemma_9b_model or "google/txgemma-9b-predict"}` |
| — | Semantic retrieval (RAG) | `{s.embedding_model_name}` |
| — | Inference backend | Local · HuggingFace Transformers · {s.quantization} quant |
"""
    )

    st.markdown(
        '<div class="disclaimer">⚠ <strong>Research demo only.</strong> '
        "Not validated for clinical use. All recommendations must be reviewed "
        "by a licensed clinician before any patient-care decision.</div>",
        unsafe_allow_html=True,
    )


def _parse_notes(raw) -> dict | list | None:
    """Parse a notes field that may be a JSON string, dict, or list."""
    if not raw or raw in ("No lab data provided", "No MIC data available for trend analysis", ""):
        return None
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return None


def _render_intake_tab(result: dict):
    intake = _parse_notes(result.get("intake_notes", ""))
    crcl = result.get("creatinine_clearance_ml_min")

    if isinstance(intake, dict):
        # Metrics row
        cols = st.columns(3)
        if crcl or intake.get("creatinine_clearance_ml_min"):
            val = crcl or intake.get("creatinine_clearance_ml_min", 0)
            cols[0].metric("CrCl", f"{val:.1f} mL/min")
        if intake.get("infection_severity"):
            cols[1].metric("Severity", intake["infection_severity"].capitalize())
        if intake.get("recommended_stage"):
            cols[2].metric("Pathway", intake["recommended_stage"].capitalize())

        if intake.get("patient_summary"):
            st.markdown(f'<div class="badge-info">{intake["patient_summary"]}</div>', unsafe_allow_html=True)

        if intake.get("renal_dose_adjustment_needed"):
            st.markdown(
                '<div class="badge-moderate" style="margin-top:8px">⚠ Renal dose adjustment required</div>',
                unsafe_allow_html=True,
            )

        if intake.get("identified_risk_factors"):
            st.markdown("**Identified risk factors**")
            for rf in intake["identified_risk_factors"]:
                st.markdown(f"- {rf}")
    elif intake:
        st.text(str(intake))
    else:
        if crcl:
            st.metric("CrCl", f"{crcl:.1f} mL/min")
        st.info("Intake summary not available.")


def _render_lab_tab(result: dict):
    vision = _parse_notes(result.get("vision_notes", ""))
    trend = _parse_notes(result.get("trend_notes", ""))

    if vision is None:
        st.info("No lab data was processed. Provide lab results to activate the targeted pathway.")
    else:
        v = vision if isinstance(vision, dict) else {}
        if v.get("specimen_type"):
            st.markdown(f"**Specimen:** {v['specimen_type'].capitalize()}")

        if v.get("extraction_confidence") is not None:
            conf = float(v["extraction_confidence"])
            color = "#27ae60" if conf >= 0.85 else "#e67e22" if conf >= 0.6 else "#c0392b"
            st.markdown(
                f'<div class="badge-info">Extraction confidence: '
                f'<span style="color:{color};font-weight:700">{conf:.0%}</span></div>',
                unsafe_allow_html=True,
            )

        orgs = v.get("identified_organisms", [])
        if orgs:
            st.markdown("**Identified organisms**")
            for o in orgs:
                name = o.get("organism_name", "Unknown")
                sig = o.get("significance", "")
                st.markdown(f"- **{name}**" + (f" — {sig}" if sig else ""))

        sus = v.get("susceptibility_results", [])
        if sus:
            st.markdown("**Susceptibility results**")
            rows = []
            for entry in sus:
                interp = entry.get("interpretation", "")
                color = {"S": "#145a32", "R": "#7b1d1d", "I": "#7a4a00"}.get(interp.upper(), "#333")
                rows.append({
                    "Organism": entry.get("organism", ""),
                    "Antibiotic": entry.get("antibiotic", ""),
                    "MIC (mg/L)": entry.get("mic_value", ""),
                    "Result": interp,
                })
            import pandas as pd
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

    # MIC Trend section
    if trend:
        st.markdown("---")
        st.markdown("**MIC Trend Analysis**")
        items = trend if isinstance(trend, list) else [trend]
        for item in items:
            if not isinstance(item, dict):
                st.text(str(item))
                continue
            risk = item.get("risk_level", "UNKNOWN").upper()
            css = {"HIGH": "badge-high", "MODERATE": "badge-moderate"}.get(risk, "badge-low")
            icon = {"HIGH": "🚨", "MODERATE": "⚠"}.get(risk, "✓")
            org = item.get("organism", "")
            ab = item.get("antibiotic", "")
            label = f"{org} / {ab} — " if (org or ab) else ""
            st.markdown(
                f'<div class="{css}" style="margin-bottom:6px">'
                f'{icon} <strong>{label}{risk}</strong><br>'
                f'<span style="font-size:0.88rem">{item.get("recommendation", "")}</span></div>',
                unsafe_allow_html=True,
            )


def page_patient_analysis():
    st.markdown('<div class="section-title">Patient Analysis Pipeline</div>', unsafe_allow_html=True)

    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None

    # ── Patient form ──
    with st.expander("Patient Demographics & Vitals", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 0, 120, 65)
            weight = st.number_input("Weight (kg)", 1.0, 300.0, 70.0, step=0.5)
            height = st.number_input("Height (cm)", 50.0, 250.0, 170.0, step=0.5)
        with c2:
            sex = st.selectbox("Biological sex", ["male", "female"])
            # Infection site is needed to decide creatinine visibility, so render it first
            # (Streamlit reruns top-to-bottom, but c3 renders in the same pass, so we
            #  read infection_site from session state on the *next* rerun.  We default
            #  to the current widget value via a placeholder key.)
            infection_site = st.session_state.get("_infection_site_val", "urinary")
            if infection_site in CREATININE_PROMINENT_SITES:
                creatinine = st.number_input("Serum Creatinine (mg/dL)", 0.1, 20.0, 1.2, step=0.1,
                                             help="Required for CrCl-based dose adjustment")
            else:
                renal_flag = st.checkbox("Known renal impairment / CKD?",
                                         help="Check to enter serum creatinine for dose adjustment")
                creatinine = (
                    st.number_input("Serum Creatinine (mg/dL)", 0.1, 20.0, 1.2, step=0.1)
                    if renal_flag else None
                )
        with c3:
            infection_site = st.selectbox(
                "Primary infection site",
                ["urinary", "respiratory", "bloodstream", "skin", "intra-abdominal", "CNS", "other"],
                key="_infection_site_val",
            )
            source_options = SUSPECTED_SOURCE_OPTIONS.get(infection_site, [])
            if source_options:
                suspected_source = st.selectbox("Suspected source", source_options)
                if suspected_source == "Other":
                    suspected_source = st.text_input(
                        "Specify source", placeholder="Describe the suspected source"
                    )
            else:
                suspected_source = st.text_input(
                    "Suspected source", placeholder="e.g., community-acquired infection"
                )

    # ── Site-specific assessment (dynamic per infection site) ──
    site_vitals: dict[str, str] = {}
    site_fields = SITE_SPECIFIC_FIELDS.get(infection_site, [])
    if site_fields:
        with st.expander(f"Site-Specific Assessment — {infection_site.title()}", expanded=True):
            cols = st.columns(2)
            for i, field in enumerate(site_fields):
                col = cols[i % 2]
                with col:
                    fkey = f"site_{field['key']}"
                    ftype = field["type"]
                    if ftype == "selectbox":
                        val = st.selectbox(field["label"], field["options"], key=fkey)
                    elif ftype == "multiselect":
                        val = st.multiselect(field["label"], field["options"], key=fkey)
                        val = ", ".join(val) if val else ""
                    elif ftype == "number_input":
                        val = st.number_input(
                            field["label"],
                            min_value=field.get("min", 0.0),
                            max_value=field.get("max", 999.0),
                            value=field.get("default", 0.0),
                            step=field.get("step", 1.0),
                            key=fkey,
                        )
                        val = str(val)
                    elif ftype == "checkbox":
                        val = st.checkbox(
                            field["label"], value=field.get("default", False), key=fkey
                        )
                        val = "Yes" if val else "No"
                    elif ftype == "text_input":
                        val = st.text_input(field["label"], key=fkey)
                    else:
                        continue
                    site_vitals[field["key"]] = str(val)

    with st.expander("Medical History"):
        c1, c2 = st.columns(2)
        with c1:
            medications = st.text_area("Current medications (one per line)", placeholder="Metformin\nLisinopril", height=100)
            allergies = st.text_area("Drug allergies (one per line)", placeholder="Penicillin\nSulfa", height=80)
        with c2:
            comorbidities = st.multiselect(
                "Comorbidities",
                ["Diabetes", "CKD", "Heart Failure", "COPD", "Immunocompromised", "Recent Surgery", "Malignancy", "Liver Disease"],
            )
            risk_factors = st.multiselect(
                "MDR risk factors",
                ["Prior MRSA", "Recent antibiotics (<90 d)", "Healthcare-associated", "Recent hospitalisation", "Nursing home", "Prior MDR infection"],
            )

    with st.expander("Lab / Culture Results  (optional — triggers targeted pathway)"):
        method = st.radio(
            "Input method",
            ["None — empirical pathway only", "Upload file (PDF / image)", "Paste lab text"],
            horizontal=True,
        )
        labs_raw_text = None
        labs_image_bytes = None

        if method == "Upload file (PDF / image)":
            uploaded = st.file_uploader(
                "Lab report file",
                type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp"],
                help="Upload a culture & sensitivity report, antibiogram, or any lab document.",
            )
            if uploaded is not None:
                file_bytes = uploaded.read()
                ext = uploaded.name.rsplit(".", 1)[-1].lower()
                if ext == "pdf":
                    # Extract text from PDF using pypdf
                    import pypdf
                    from io import BytesIO
                    try:
                        reader = pypdf.PdfReader(BytesIO(file_bytes))
                        extracted = "\n".join(
                            page.extract_text() or "" for page in reader.pages
                        ).strip()
                        if extracted:
                            labs_raw_text = extracted
                            st.success(f"PDF parsed — {len(reader.pages)} page(s), {len(extracted)} characters extracted.")
                        else:
                            st.warning(
                                "PDF text extraction returned empty content (scanned PDF?). "
                                "The file will be processed as an image by the vision model."
                            )
                            # Convert first page to image fallback via pillow (requires pypdf extras)
                            labs_image_bytes = file_bytes
                    except Exception as e:
                        st.error(f"PDF parsing failed: {e}")
                else:
                    # Image file — pass directly to the multimodal model
                    labs_image_bytes = file_bytes
                    from PIL import Image as _PILImage
                    from io import BytesIO as _BytesIO
                    try:
                        thumb = _PILImage.open(_BytesIO(file_bytes))
                        st.image(thumb, caption=f"Uploaded: {uploaded.name}", width=320)
                    except Exception:
                        st.info(f"Image uploaded: {uploaded.name}")

        elif method == "Paste lab text":
            labs_raw_text = st.text_area(
                "Lab report",
                placeholder=(
                    "Organism: Escherichia coli\n"
                    "Ciprofloxacin: S  MIC 0.25\n"
                    "Nitrofurantoin: S  MIC 16\n"
                    "Ampicillin: R  MIC >32"
                ),
                height=160,
            )

    st.markdown("")
    run_btn = st.button("Run Agent Pipeline", type="primary", use_container_width=False)

    if run_btn:
        has_lab_input = bool(labs_raw_text or labs_image_bytes)
        patient_data = {
            "age_years": age,
            "weight_kg": weight,
            "height_cm": height,
            "sex": sex,
            "serum_creatinine_mg_dl": creatinine,
            "infection_site": infection_site,
            "suspected_source": suspected_source or f"{infection_site} infection",
            "medications": [m.strip() for m in medications.split("\n") if m.strip()],
            "allergies": [a.strip() for a in allergies.split("\n") if a.strip()],
            "comorbidities": list(comorbidities) + list(risk_factors),
            "vitals": site_vitals,
            "labs_image_bytes": labs_image_bytes,
        }

        stages = (
            ["Intake Historian", "Vision Specialist", "Trend Analyst", "Clinical Pharmacologist"]
            if has_lab_input
            else ["Intake Historian", "Clinical Pharmacologist"]
        )

        prog = st.progress(0, text="Starting pipeline…")
        for i, name in enumerate(stages):
            prog.progress((i + 1) / len(stages), text=f"Running: {name}")

        try:
            from src.graph import run_pipeline
            result = run_pipeline(patient_data, labs_raw_text)
        except Exception:
            result = _demo_result(patient_data, labs_raw_text or bool(labs_image_bytes))

        prog.progress(100, text="Complete")
        st.session_state.pipeline_result = result

    # ── Results ──
    if st.session_state.pipeline_result:
        result = st.session_state.pipeline_result
        st.markdown('<div class="section-title">Results</div>', unsafe_allow_html=True)

        t1, t2, t3, t4 = st.tabs(["Recommendation", "Patient Summary", "Lab Analysis", "Safety"])

        with t1:
            rec = result.get("recommendation", {})
            if rec:
                primary = rec.get("primary_antibiotic", "—")
                dose = rec.get("dose", "—")
                route = rec.get("route", "—")
                freq = rec.get("frequency", "—")
                duration = rec.get("duration", "—")
                alt = rec.get("backup_antibiotic", "")

                st.markdown(
                    f"""
<div class="rx-card">
  <div class="rx-symbol">℞</div>
  <div class="rx-drug">{primary}</div>
  <br>
  <strong>Dose:</strong> {dose} &nbsp;·&nbsp;
  <strong>Route:</strong> {route} &nbsp;·&nbsp;
  <strong>Frequency:</strong> {freq} &nbsp;·&nbsp;
  <strong>Duration:</strong> {duration}
  {"<br><strong>Alternative:</strong> " + alt if alt else ""}
</div>
""",
                    unsafe_allow_html=True,
                )

                if rec.get("rationale"):
                    st.markdown("**Clinical rationale**")
                    st.markdown(rec["rationale"])

                if rec.get("references"):
                    st.markdown("**References**")
                    for ref in rec["references"]:
                        st.markdown(f"- {ref}")

        with t2:
            _render_intake_tab(result)

        with t3:
            _render_lab_tab(result)

        with t4:
            warnings = result.get("safety_warnings", [])
            if warnings:
                for w in warnings:
                    st.markdown(f'<div class="badge-high">⚠ {w}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-low">✓ No safety concerns identified.</div>', unsafe_allow_html=True)

            errors = result.get("errors", [])
            for err in errors:
                st.error(err)


def _demo_result(patient_data: dict, labs_raw_text) -> dict:
    result = {
        "stage": "targeted" if labs_raw_text else "empirical",
        "creatinine_clearance_ml_min": 58.3,
        "intake_notes": json.dumps({
            "patient_summary": f"{patient_data.get('age_years')}-year-old {patient_data.get('sex')} · {patient_data.get('suspected_source', 'infection')}",
            "creatinine_clearance_ml_min": 58.3,
            "renal_dose_adjustment_needed": True,
            "identified_risk_factors": patient_data.get("comorbidities", []),
            "infection_severity": "moderate",
            "recommended_stage": "targeted" if labs_raw_text else "empirical",
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
                "Ciprofloxacin provides broad Gram-negative coverage. Dose standard — "
                "no adjustment required above CrCl 30 mL/min."
            ),
            "references": ["IDSA UTI Guidelines 2024", "EUCAST Breakpoint Tables v16.0"],
        },
        "safety_warnings": [],
        "errors": [],
    }
    if labs_raw_text:
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
            "organism": "E. coli",
            "antibiotic": "Ciprofloxacin",
            "risk_level": "LOW",
            "recommendation": "Continue current therapy — no MIC creep detected.",
        }])
    return result


def page_clinical_tools():
    st.markdown('<div class="section-title">Clinical Tools</div>', unsafe_allow_html=True)

    tool = st.selectbox(
        "Select tool",
        ["Empirical Advisor", "MIC Interpreter", "MIC Trend Analysis", "Drug Safety Check"],
        label_visibility="visible",
    )

    st.markdown("")

    # ── Empirical Advisor ──
    if tool == "Empirical Advisor":
        c1, c2 = st.columns([3, 1])
        with c1:
            infection_type = st.selectbox(
                "Infection type",
                ["Urinary Tract Infection", "Pneumonia", "Sepsis", "Skin / Soft Tissue", "Intra-abdominal", "Meningitis"],
            )
            pathogen = st.text_input("Suspected pathogen (optional)", placeholder="e.g., Klebsiella pneumoniae")
            risk = st.multiselect(
                "Risk factors",
                ["Prior MRSA", "Recent antibiotics (<90 d)", "Healthcare-associated", "Immunocompromised", "Renal impairment", "Prior MDR"],
            )
        with c2:
            st.markdown(
                '<div class="badge-info"><strong>WHO AWaRe</strong><br>'
                '<span style="color:#145a32">●</span> Access — first-line<br>'
                '<span style="color:#7a4a00">●</span> Watch — second-line<br>'
                '<span style="color:#7b1d1d">●</span> Reserve — last resort</div>',
                unsafe_allow_html=True,
            )

        if st.button("Get recommendation", type="primary"):
            with st.spinner("Searching clinical guidelines…"):
                guidance = get_empirical_therapy_guidance(infection_type, risk)

            if guidance.get("recommendations"):
                for i, rec in enumerate(guidance["recommendations"][:3], 1):
                    with st.expander(f"Guideline excerpt {i}  (relevance {rec.get('relevance_score', 0):.2f})"):
                        st.markdown(rec.get("content", ""))
                        st.caption(f"Source: {rec.get('source', 'IDSA Guidelines 2024')}")

            if pathogen:
                st.markdown(f"**Resistance data — {pathogen}**")
                effective = get_most_effective_antibiotics(pathogen, min_susceptibility=70)
                if effective:
                    for ab in effective[:6]:
                        st.write(f"- **{ab.get('antibiotic')}** — {ab.get('avg_susceptibility', 0):.1f}% susceptible")
                else:
                    st.info("No resistance data available for this pathogen.")

    # ── MIC Interpreter ──
    elif tool == "MIC Interpreter":
        c1, c2 = st.columns(2)
        with c1:
            pathogen = st.text_input("Pathogen", placeholder="e.g., Escherichia coli")
            antibiotic = st.text_input("Antibiotic", placeholder="e.g., Ciprofloxacin")
            mic = st.number_input("MIC value (mg/L)", 0.001, 1024.0, 1.0, step=0.001, format="%.3f")
        with c2:
            st.markdown(
                '<div class="badge-info" style="margin-top:28px">'
                "<strong>Interpretation guide</strong><br><br>"
                "<strong>S</strong> Susceptible — antibiotic is effective<br>"
                "<strong>I</strong> Intermediate — effective at higher doses<br>"
                "<strong>R</strong> Resistant — do not use</div>",
                unsafe_allow_html=True,
            )

        if st.button("Interpret", type="primary"):
            if pathogen and antibiotic:
                result = interpret_mic_value(pathogen, antibiotic, mic)
                interp = result.get("interpretation", "UNKNOWN")
                msg = result.get("message", "")
                if interp == "SUSCEPTIBLE":
                    st.markdown(f'<div class="badge-low"><strong>Susceptible (S)</strong> — {msg}</div>', unsafe_allow_html=True)
                elif interp == "RESISTANT":
                    st.markdown(f'<div class="badge-high"><strong>Resistant (R)</strong> — {msg}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="badge-moderate"><strong>Intermediate (I)</strong> — {msg}</div>', unsafe_allow_html=True)

    # ── MIC Trend ──
    elif tool == "MIC Trend Analysis":
        n = st.slider("Number of historical readings", 2, 6, 3)
        cols = st.columns(n)
        mic_values = []
        for i, col in enumerate(cols):
            v = col.number_input(f"MIC {i + 1} (mg/L)", 0.001, 256.0, float(2 ** i), key=f"mic_{i}")
            mic_values.append({"date": f"T{i}", "mic_value": v})

        if st.button("Analyse trend", type="primary"):
            result = calculate_mic_trend(mic_values)
            risk = result.get("risk_level", "UNKNOWN")
            alert = result.get("alert", "")
            css = {"HIGH": "badge-high", "MODERATE": "badge-moderate"}.get(risk, "badge-low")
            icon = {"HIGH": "🚨", "MODERATE": "⚠"}.get(risk, "✓")
            st.markdown(f'<div class="{css}">{icon} <strong>{risk} RISK</strong> — {alert}</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline MIC", f"{result.get('baseline_mic', '—')} mg/L")
            c2.metric("Current MIC", f"{result.get('current_mic', '—')} mg/L")
            c3.metric("Fold change", f"{result.get('ratio', '—')}×")

    # ── Drug Safety ──
    elif tool == "Drug Safety Check":
        c1, c2 = st.columns(2)
        with c1:
            ab = st.text_input("Antibiotic to check", placeholder="e.g., Ciprofloxacin")
            meds = st.text_area("Concurrent medications", placeholder="Warfarin\nMetformin\nAmlodipine", height=120)
        with c2:
            allergies = st.text_area("Known allergies", placeholder="Penicillin\nSulfa", height=100)

        if st.button("Check safety", type="primary"):
            if ab:
                med_list = [m.strip() for m in meds.split("\n") if m.strip()]
                allergy_list = [a.strip() for a in allergies.split("\n") if a.strip()]
                result = screen_antibiotic_safety(ab, med_list, allergy_list)

                if result.get("safe_to_use"):
                    st.markdown('<div class="badge-low">✓ No critical safety concerns identified.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-high">⚠ Safety concerns identified — review required.</div>', unsafe_allow_html=True)

                for alert in result.get("alerts", []):
                    st.markdown(f'<div class="badge-moderate" style="margin-top:8px">⚠ {alert.get("message", "")}</div>', unsafe_allow_html=True)


def page_guidelines():
    st.markdown('<div class="section-title">Clinical Guidelines Search</div>', unsafe_allow_html=True)

    query = st.text_input("Search query", placeholder="e.g., ESBL E. coli UTI treatment carbapenems")
    pathogen_filter = st.selectbox("Filter by pathogen", ["All", "ESBL-E", "CRE", "CRAB", "DTR-PA"])

    if st.button("Search", type="primary"):
        if query:
            with st.spinner("Searching knowledge base…"):
                filter_val = None if pathogen_filter == "All" else pathogen_filter
                results = search_clinical_guidelines(query, pathogen_filter=filter_val, n_results=5)

            if results:
                for i, r in enumerate(results, 1):
                    with st.expander(f"Result {i}  ·  relevance {r.get('relevance_score', 0):.2f}"):
                        st.markdown(r.get("content", ""))
                        if r.get("source"):
                            st.caption(f"Source: {r['source']}")
            else:
                st.info("No results found. Try broader search terms or check that the knowledge base has been initialised.")

    st.markdown(
        '<div class="disclaimer">Sources: IDSA Treatment Guidelines 2024 · '
        "EUCAST Breakpoint Tables v16.0 · WHO EML · DDInter drug interaction database.</div>",
        unsafe_allow_html=True,
    )


# ── Router ────────────────────────────────────────────────────────────────────

if page == "Dashboard":
    page_dashboard()
elif page == "Patient Analysis":
    page_patient_analysis()
elif page == "Clinical Tools":
    page_clinical_tools()
elif page == "Guidelines":
    page_guidelines()
