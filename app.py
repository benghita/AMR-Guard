"""
Med-I-C: AMR-Guard Demo Application
Infection Lifecycle Orchestrator - Streamlit Interface

Multi-Agent Architecture powered by MedGemma via LangGraph
"""

import streamlit as st
import sys
import json
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools import (
    interpret_mic_value,
    get_most_effective_antibiotics,
    calculate_mic_trend,
    screen_antibiotic_safety,
    search_clinical_guidelines,
    get_empirical_therapy_guidance,
)
from src.utils import format_prescription_card

# Page configuration
st.set_page_config(
    page_title="Med-I-C: AMR-Guard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
    }
    .agent-card {
        background-color: #F5F5F5;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #1E88E5;
    }
    .agent-active {
        border-left-color: #4CAF50;
        background-color: #E8F5E9;
    }
    .agent-complete {
        border-left-color: #9E9E9E;
        background-color: #FAFAFA;
    }
    .risk-high {
        background-color: #FFCDD2;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #D32F2F;
    }
    .risk-moderate {
        background-color: #FFE0B2;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #F57C00;
    }
    .risk-low {
        background-color: #C8E6C9;
        padding: 10px;
        border-radius: 5px;
        border-left: 4px solid #388E3C;
    }
    .prescription-card {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        font-family: monospace;
        white-space: pre-wrap;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<p class="main-header">🦠 Med-I-C: AMR-Guard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Infection Lifecycle Orchestrator - Multi-Agent System</p>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Overview",
            "🤖 Agent Pipeline",
            "💊 Empirical Advisor",
            "🔬 Lab Interpretation",
            "📊 MIC Trend Analysis",
            "⚠️ Drug Safety Check",
            "📚 Clinical Guidelines"
        ]
    )

    if page == "🏠 Overview":
        show_overview()
    elif page == "🤖 Agent Pipeline":
        show_agent_pipeline()
    elif page == "💊 Empirical Advisor":
        show_empirical_advisor()
    elif page == "🔬 Lab Interpretation":
        show_lab_interpretation()
    elif page == "📊 MIC Trend Analysis":
        show_mic_trend_analysis()
    elif page == "⚠️ Drug Safety Check":
        show_drug_safety()
    elif page == "📚 Clinical Guidelines":
        show_guidelines_search()


def show_overview():
    st.header("System Overview")

    st.markdown("""
    **AMR-Guard** is a multi-agent AI system that orchestrates the complete infection treatment lifecycle,
    from initial empirical therapy to targeted treatment based on lab results.
    """)

    # Architecture diagram
    st.subheader("Multi-Agent Architecture")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### Stage 1: Empirical Phase
        **Path:** Agent 1 → Agent 4

        *Before lab results are available*

        1. **Intake Historian** (Agent 1)
           - Parses patient demographics & history
           - Calculates CrCl for renal dosing
           - Identifies risk factors for MDR

        2. **Clinical Pharmacologist** (Agent 4)
           - Recommends empirical antibiotics
           - Applies WHO AWaRe principles
           - Performs safety checks
        """)

    with col2:
        st.markdown("""
        ### Stage 2: Targeted Phase
        **Path:** Agent 1 → Agent 2 → Agent 3 → Agent 4

        *When lab/culture results are available*

        1. **Intake Historian** (Agent 1)
        2. **Vision Specialist** (Agent 2)
           - Extracts data from lab reports
           - Supports any language/format
        3. **Trend Analyst** (Agent 3)
           - Detects MIC creep patterns
           - Calculates resistance velocity
        4. **Clinical Pharmacologist** (Agent 4)
        """)

    st.divider()

    # Knowledge sources
    st.subheader("Knowledge Sources")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("WHO AWaRe", "264", "antibiotics classified")
    with col2:
        st.metric("EUCAST", "v16.0", "breakpoint tables")
    with col3:
        st.metric("IDSA", "2024", "treatment guidelines")
    with col4:
        st.metric("DDInter", "191K+", "drug interactions")

    # Model info
    st.subheader("AI Models")
    st.markdown("""
    | Agent | Primary Model | Fallback |
    |-------|---------------|----------|
    | Intake Historian | MedGemma 4B IT | Vertex AI API |
    | Vision Specialist | MedGemma 4B IT (multimodal) | Vertex AI API |
    | Trend Analyst | MedGemma 4B IT | Vertex AI API |
    | Clinical Pharmacologist | MedGemma 4B + TxGemma 2B (safety) | Vertex AI API |
    """)


def show_agent_pipeline():
    st.header("🤖 Multi-Agent Pipeline")
    st.markdown("*Run the complete infection lifecycle workflow*")

    # Initialize session state
    if "pipeline_result" not in st.session_state:
        st.session_state.pipeline_result = None

    # Patient Information Form
    with st.expander("Patient Information", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=65)
            weight = st.number_input("Weight (kg)", min_value=1.0, max_value=300.0, value=70.0)
            height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0)

        with col2:
            sex = st.selectbox("Sex", ["male", "female"])
            creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.1, max_value=20.0, value=1.2)

        with col3:
            infection_site = st.selectbox(
                "Infection Site",
                ["urinary", "respiratory", "bloodstream", "skin", "intra-abdominal", "CNS", "other"]
            )
            suspected_source = st.text_input(
                "Suspected Source",
                placeholder="e.g., community UTI, hospital-acquired pneumonia"
            )

    with st.expander("Medical History"):
        col1, col2 = st.columns(2)

        with col1:
            medications = st.text_area(
                "Current Medications (one per line)",
                placeholder="Metformin\nLisinopril\nAspirin",
                height=100
            )
            allergies = st.text_area(
                "Allergies (one per line)",
                placeholder="Penicillin\nSulfa",
                height=100
            )

        with col2:
            comorbidities = st.multiselect(
                "Comorbidities",
                ["Diabetes", "CKD", "Heart Failure", "COPD", "Immunocompromised",
                 "Recent Surgery", "Malignancy", "Liver Disease"]
            )
            risk_factors = st.multiselect(
                "MDR Risk Factors",
                ["Prior MRSA infection", "Recent antibiotic use (<90 days)",
                 "Healthcare-associated", "Recent hospitalization",
                 "Nursing home resident", "Prior MDR infection"]
            )

    # Lab Data (Optional - triggers Stage 2)
    with st.expander("Lab Results (Optional - triggers targeted pathway)"):
        lab_input_method = st.radio(
            "Input Method",
            ["None (Empirical only)", "Paste Lab Text", "Upload File"],
            horizontal=True
        )

        labs_raw_text = None

        if lab_input_method == "Paste Lab Text":
            labs_raw_text = st.text_area(
                "Lab Report Text",
                placeholder="""Example:
Culture: Urine
Organism: Escherichia coli
Colony Count: >100,000 CFU/mL

Susceptibility:
Ampicillin: R (MIC >32)
Ciprofloxacin: S (MIC 0.25)
Nitrofurantoin: S (MIC 16)
Trimethoprim-Sulfamethoxazole: R (MIC >4)""",
                height=200
            )

        elif lab_input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload Lab Report (PDF or Image)",
                type=["pdf", "png", "jpg", "jpeg"]
            )
            if uploaded_file:
                st.info("File uploaded. Text extraction will be performed by the Vision Specialist agent.")
                # In production, would extract text here
                labs_raw_text = f"[Uploaded file: {uploaded_file.name}]"

    # Run Pipeline Button
    st.divider()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        run_pipeline_btn = st.button(
            "🚀 Run Agent Pipeline",
            type="primary",
            use_container_width=True
        )

    if run_pipeline_btn:
        # Build patient data
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
        }

        # Show pipeline progress
        st.subheader("Pipeline Execution")

        # Agent progress indicators
        agents = [
            ("Intake Historian", "Analyzing patient data..."),
            ("Vision Specialist", "Processing lab results...") if labs_raw_text else None,
            ("Trend Analyst", "Analyzing MIC trends...") if labs_raw_text else None,
            ("Clinical Pharmacologist", "Generating recommendations..."),
        ]
        agents = [a for a in agents if a is not None]

        progress_bar = st.progress(0)
        status_text = st.empty()

        # Simulate pipeline execution (in production, would call actual pipeline)
        try:
            # Try to import and run the actual pipeline
            from src.graph import run_pipeline

            for i, (agent_name, status_msg) in enumerate(agents):
                status_text.text(f"Agent {i+1}/{len(agents)}: {agent_name} - {status_msg}")
                progress_bar.progress((i + 1) / len(agents))

            # Run the actual pipeline
            result = run_pipeline(patient_data, labs_raw_text)
            st.session_state.pipeline_result = result

        except Exception as e:
            st.error(f"Pipeline execution error: {e}")
            st.info("Running in demo mode with simulated output...")

            # Demo mode - simulate results
            st.session_state.pipeline_result = _generate_demo_result(patient_data, labs_raw_text)

        progress_bar.progress(100)
        status_text.text("Pipeline complete!")

    # Display Results
    if st.session_state.pipeline_result:
        result = st.session_state.pipeline_result

        st.divider()
        st.subheader("Pipeline Results")

        # Tabs for different result sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Recommendation",
            "👤 Patient Summary",
            "🔬 Lab Analysis",
            "⚠️ Safety Alerts"
        ])

        with tab1:
            rec = result.get("recommendation", {})
            if rec:
                st.markdown("### Antibiotic Recommendation")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Primary:** {rec.get('primary_antibiotic', 'N/A')}")
                    st.markdown(f"**Dose:** {rec.get('dose', 'N/A')}")
                    st.markdown(f"**Route:** {rec.get('route', 'N/A')}")
                    st.markdown(f"**Frequency:** {rec.get('frequency', 'N/A')}")
                    st.markdown(f"**Duration:** {rec.get('duration', 'N/A')}")

                with col2:
                    if rec.get("backup_antibiotic"):
                        st.markdown(f"**Alternative:** {rec.get('backup_antibiotic')}")

                st.markdown("---")
                st.markdown("**Rationale:**")
                st.markdown(rec.get("rationale", "No rationale provided"))

                if rec.get("references"):
                    st.markdown("**References:**")
                    for ref in rec["references"]:
                        st.markdown(f"- {ref}")

        with tab2:
            st.markdown("### Patient Assessment")
            intake_notes = result.get("intake_notes", "")
            if intake_notes:
                try:
                    intake_data = json.loads(intake_notes) if isinstance(intake_notes, str) else intake_notes
                    st.json(intake_data)
                except:
                    st.text(intake_notes)

            if result.get("creatinine_clearance_ml_min"):
                st.metric("Calculated CrCl", f"{result['creatinine_clearance_ml_min']} mL/min")

        with tab3:
            st.markdown("### Laboratory Analysis")

            vision_notes = result.get("vision_notes", "No lab data processed")
            if vision_notes and vision_notes != "No lab data provided":
                try:
                    vision_data = json.loads(vision_notes) if isinstance(vision_notes, str) else vision_notes
                    st.json(vision_data)
                except:
                    st.text(vision_notes)

            trend_notes = result.get("trend_notes", "")
            if trend_notes and trend_notes != "No MIC data available for trend analysis":
                st.markdown("#### MIC Trend Analysis")
                try:
                    trend_data = json.loads(trend_notes) if isinstance(trend_notes, str) else trend_notes
                    st.json(trend_data)
                except:
                    st.text(trend_notes)

        with tab4:
            st.markdown("### Safety Alerts")

            warnings = result.get("safety_warnings", [])
            if warnings:
                for warning in warnings:
                    st.warning(f"⚠️ {warning}")
            else:
                st.success("No safety concerns identified")

            errors = result.get("errors", [])
            if errors:
                st.markdown("#### Errors")
                for error in errors:
                    st.error(error)


def _generate_demo_result(patient_data: dict, labs_raw_text: str | None) -> dict:
    """Generate demo result when actual pipeline is not available."""
    result = {
        "stage": "targeted" if labs_raw_text else "empirical",
        "creatinine_clearance_ml_min": 58.3,
        "intake_notes": json.dumps({
            "patient_summary": f"65-year-old male with {patient_data.get('suspected_source', 'infection')}",
            "creatinine_clearance_ml_min": 58.3,
            "renal_dose_adjustment_needed": True,
            "identified_risk_factors": patient_data.get("comorbidities", []),
            "infection_severity": "moderate",
            "recommended_stage": "targeted" if labs_raw_text else "empirical",
        }),
        "recommendation": {
            "primary_antibiotic": "Ciprofloxacin",
            "dose": "500mg",
            "route": "PO",
            "frequency": "Every 12 hours",
            "duration": "7 days",
            "backup_antibiotic": "Nitrofurantoin",
            "rationale": "Based on suspected community-acquired UTI with moderate renal impairment. Ciprofloxacin provides good coverage for common uropathogens. Dose adjusted for CrCl 58 mL/min.",
            "references": ["IDSA UTI Guidelines 2024", "EUCAST Breakpoint Tables v16.0"],
            "safety_alerts": [],
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
            ],
            "extraction_confidence": 0.95,
        })
        result["trend_notes"] = json.dumps([{
            "organism": "E. coli",
            "antibiotic": "Ciprofloxacin",
            "risk_level": "LOW",
            "recommendation": "Continue current therapy",
        }])

    return result


def show_empirical_advisor():
    st.header("💊 Empirical Advisor")
    st.markdown("*Get empirical therapy recommendations before lab results*")

    col1, col2 = st.columns([2, 1])

    with col1:
        infection_type = st.selectbox(
            "Infection Type",
            ["Urinary Tract Infection", "Pneumonia", "Sepsis",
             "Skin/Soft Tissue", "Intra-abdominal", "Meningitis"]
        )

        suspected_pathogen = st.text_input(
            "Suspected Pathogen (optional)",
            placeholder="e.g., E. coli, Klebsiella pneumoniae"
        )

        risk_factors = st.multiselect(
            "Risk Factors",
            ["Prior MRSA infection", "Recent antibiotic use (<90 days)",
             "Healthcare-associated", "Immunocompromised",
             "Renal impairment", "Prior MDR infection"]
        )

    with col2:
        st.markdown("**WHO AWaRe Categories**")
        st.markdown("""
        - **ACCESS**: First-line, low resistance
        - **WATCH**: Higher resistance potential
        - **RESERVE**: Last resort antibiotics
        """)

    if st.button("Get Recommendation", type="primary"):
        with st.spinner("Searching guidelines..."):
            guidance = get_empirical_therapy_guidance(
                infection_type,
                risk_factors
            )

            st.subheader("Guideline Recommendations")

            if guidance.get("recommendations"):
                for i, rec in enumerate(guidance["recommendations"][:3], 1):
                    with st.expander(f"Excerpt {i} (Relevance: {rec.get('relevance_score', 0):.2f})"):
                        st.markdown(rec.get("content", ""))
                        st.caption(f"Source: {rec.get('source', 'IDSA Guidelines')}")

            if suspected_pathogen:
                st.subheader(f"Resistance Data: {suspected_pathogen}")
                effective = get_most_effective_antibiotics(suspected_pathogen, min_susceptibility=70)

                if effective:
                    for ab in effective[:5]:
                        st.write(f"- **{ab.get('antibiotic')}**: {ab.get('avg_susceptibility', 0):.1f}% susceptible")
                else:
                    st.info("No resistance data found.")


def show_lab_interpretation():
    st.header("🔬 Lab Interpretation")
    st.markdown("*Interpret antibiogram MIC values*")

    col1, col2 = st.columns(2)

    with col1:
        pathogen = st.text_input("Pathogen", placeholder="e.g., Escherichia coli")
        antibiotic = st.text_input("Antibiotic", placeholder="e.g., Ciprofloxacin")
        mic_value = st.number_input("MIC (mg/L)", min_value=0.001, max_value=1024.0, value=1.0)

    with col2:
        st.markdown("**Interpretation Guide**")
        st.markdown("""
        - **S**: Susceptible - antibiotic effective
        - **I**: Intermediate - may work at higher doses
        - **R**: Resistant - do not use
        """)

    if st.button("Interpret", type="primary"):
        if pathogen and antibiotic:
            result = interpret_mic_value(pathogen, antibiotic, mic_value)
            interpretation = result.get("interpretation", "UNKNOWN")

            if interpretation == "SUSCEPTIBLE":
                st.success(f"✅ {interpretation}")
            elif interpretation == "RESISTANT":
                st.error(f"❌ {interpretation}")
            else:
                st.warning(f"⚠️ {interpretation}")

            st.markdown(f"**Details:** {result.get('message', '')}")


def show_mic_trend_analysis():
    st.header("📊 MIC Trend Analysis")
    st.markdown("*Detect MIC creep over time*")

    num_readings = st.slider("Historical readings", 2, 6, 3)

    mic_values = []
    cols = st.columns(num_readings)

    for i, col in enumerate(cols):
        mic = col.number_input(f"MIC {i+1}", min_value=0.001, max_value=256.0, value=float(2 ** i), key=f"mic_{i}")
        mic_values.append({"date": f"T{i}", "mic_value": mic})

    if st.button("Analyze", type="primary"):
        result = calculate_mic_trend(mic_values)
        risk_level = result.get("risk_level", "UNKNOWN")

        if risk_level == "HIGH":
            st.markdown(f'<div class="risk-high">🚨 HIGH RISK: {result.get("alert", "")}</div>', unsafe_allow_html=True)
        elif risk_level == "MODERATE":
            st.markdown(f'<div class="risk-moderate">⚠️ MODERATE: {result.get("alert", "")}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low">✅ LOW RISK: {result.get("alert", "")}</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Baseline", f"{result.get('baseline_mic', 'N/A')} mg/L")
        col2.metric("Current", f"{result.get('current_mic', 'N/A')} mg/L")
        col3.metric("Fold Change", f"{result.get('ratio', 'N/A')}x")


def show_drug_safety():
    st.header("⚠️ Drug Safety Check")

    col1, col2 = st.columns(2)

    with col1:
        antibiotic = st.text_input("Antibiotic", placeholder="e.g., Ciprofloxacin")
        current_meds = st.text_area("Current Medications", placeholder="Warfarin\nMetformin", height=150)

    with col2:
        allergies = st.text_area("Allergies", placeholder="Penicillin\nSulfa", height=100)

    if st.button("Check Safety", type="primary"):
        if antibiotic:
            medications = [m.strip() for m in current_meds.split("\n") if m.strip()]
            allergy_list = [a.strip() for a in allergies.split("\n") if a.strip()]

            result = screen_antibiotic_safety(antibiotic, medications, allergy_list)

            if result.get("safe_to_use"):
                st.success("✅ No critical safety concerns")
            else:
                st.error("❌ Safety concerns identified")

            for alert in result.get("alerts", []):
                st.warning(f"⚠️ {alert.get('message', '')}")


def show_guidelines_search():
    st.header("📚 Clinical Guidelines")

    query = st.text_input("Search", placeholder="e.g., ESBL E. coli UTI treatment")
    pathogen_filter = st.selectbox("Pathogen Filter", ["All", "ESBL-E", "CRE", "CRAB", "DTR-PA"])

    if st.button("Search", type="primary"):
        if query:
            filter_val = None if pathogen_filter == "All" else pathogen_filter
            results = search_clinical_guidelines(query, pathogen_filter=filter_val, n_results=5)

            if results:
                for i, r in enumerate(results, 1):
                    with st.expander(f"Result {i} (Relevance: {r.get('relevance_score', 0):.2f})"):
                        st.markdown(r.get("content", ""))
            else:
                st.info("No results found.")


if __name__ == "__main__":
    main()
