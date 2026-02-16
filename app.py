"""
Med-I-C: AMR-Guard Demo Application
Infection Lifecycle Orchestrator - Streamlit Interface
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tools import (
    query_antibiotic_info,
    get_antibiotics_by_category,
    interpret_mic_value,
    get_breakpoints_for_pathogen,
    query_resistance_pattern,
    get_most_effective_antibiotics,
    calculate_mic_trend,
    check_drug_interactions,
    screen_antibiotic_safety,
    search_clinical_guidelines,
    get_treatment_recommendation,
    get_empirical_therapy_guidance,
)

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
    st.markdown('<p class="sub-header">Infection Lifecycle Orchestrator Demo</p>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        [
            "🏠 Overview",
            "💊 Stage 1: Empirical Advisor",
            "🔬 Stage 2: Lab Interpretation",
            "📊 MIC Trend Analysis",
            "⚠️ Drug Safety Check",
            "📚 Clinical Guidelines Search"
        ]
    )

    if page == "🏠 Overview":
        show_overview()
    elif page == "💊 Stage 1: Empirical Advisor":
        show_empirical_advisor()
    elif page == "🔬 Stage 2: Lab Interpretation":
        show_lab_interpretation()
    elif page == "📊 MIC Trend Analysis":
        show_mic_trend_analysis()
    elif page == "⚠️ Drug Safety Check":
        show_drug_safety()
    elif page == "📚 Clinical Guidelines Search":
        show_guidelines_search()


def show_overview():
    st.header("System Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Stage 1: Empirical Phase")
        st.markdown("""
        **The "First 24 Hours"**

        Before lab results are available, the system:
        - Analyzes patient history and risk factors
        - Suggests empirical antibiotics based on:
            - Suspected pathogen
            - Local resistance patterns
            - WHO stewardship guidelines (ACCESS → WATCH → RESERVE)
        - Checks drug interactions with current medications
        """)

    with col2:
        st.subheader("Stage 2: Targeted Phase")
        st.markdown("""
        **The "Lab Interpretation"**

        Once antibiogram is available, the system:
        - Interprets MIC values against EUCAST breakpoints
        - Detects "MIC Creep" from historical data
        - Refines antibiotic selection
        - Provides evidence-based treatment recommendations
        """)

    st.divider()

    st.subheader("Knowledge Sources")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("WHO EML", "264", "antibiotics classified")
    with col2:
        st.metric("ATLAS Data", "10K+", "susceptibility records")
    with col3:
        st.metric("Breakpoints", "41", "pathogen groups")
    with col4:
        st.metric("Interactions", "191K+", "drug pairs")


def show_empirical_advisor():
    st.header("💊 Stage 1: Empirical Advisor")
    st.markdown("*Recommend empirical therapy before lab results*")

    col1, col2 = st.columns([2, 1])

    with col1:
        infection_type = st.selectbox(
            "Infection Type",
            ["Urinary Tract Infection (UTI)", "Pneumonia", "Sepsis",
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
        st.markdown("**WHO Stewardship Categories**")
        st.markdown("""
        - **ACCESS**: First-line, low resistance
        - **WATCH**: Higher resistance potential
        - **RESERVE**: Last resort antibiotics
        """)

    if st.button("Get Empirical Recommendation", type="primary"):
        with st.spinner("Searching guidelines and resistance data..."):
            # Get recommendations from guidelines
            guidance = get_empirical_therapy_guidance(
                infection_type.split("(")[0].strip(),
                risk_factors
            )

            st.subheader("Recommendations")

            if guidance.get("recommendations"):
                for i, rec in enumerate(guidance["recommendations"][:3], 1):
                    with st.expander(f"Guideline Excerpt {i} (Relevance: {rec.get('relevance_score', 0):.2f})"):
                        st.markdown(rec.get("content", ""))
                        st.caption(f"Source: {rec.get('source', 'IDSA Guidelines')}")

            # If pathogen specified, show resistance patterns
            if suspected_pathogen:
                st.subheader(f"Resistance Patterns for {suspected_pathogen}")

                effective = get_most_effective_antibiotics(suspected_pathogen, min_susceptibility=70)

                if effective:
                    st.markdown("**Most Effective Antibiotics (>70% susceptibility)**")
                    for ab in effective[:5]:
                        st.write(f"- **{ab.get('antibiotic')}**: {ab.get('avg_susceptibility', 0):.1f}% susceptible")
                else:
                    st.info("No resistance data found for this pathogen.")


def show_lab_interpretation():
    st.header("🔬 Stage 2: Lab Interpretation")
    st.markdown("*Interpret antibiogram MIC values*")

    col1, col2 = st.columns(2)

    with col1:
        pathogen = st.text_input(
            "Identified Pathogen",
            placeholder="e.g., Escherichia coli, Pseudomonas aeruginosa"
        )

        antibiotic = st.text_input(
            "Antibiotic",
            placeholder="e.g., Ciprofloxacin, Meropenem"
        )

        mic_value = st.number_input(
            "MIC Value (mg/L)",
            min_value=0.001,
            max_value=1024.0,
            value=1.0,
            step=0.5
        )

    with col2:
        st.markdown("**How to Read Results**")
        st.markdown("""
        - **S (Susceptible)**: MIC ≤ breakpoint - antibiotic likely effective
        - **I (Intermediate)**: May work with higher doses
        - **R (Resistant)**: MIC > breakpoint - do not use
        """)

    if st.button("Interpret MIC", type="primary"):
        if pathogen and antibiotic:
            with st.spinner("Checking breakpoints..."):
                result = interpret_mic_value(pathogen, antibiotic, mic_value)

                interpretation = result.get("interpretation", "UNKNOWN")

                if interpretation == "SUSCEPTIBLE":
                    st.success(f"✅ **{interpretation}**")
                elif interpretation == "RESISTANT":
                    st.error(f"❌ **{interpretation}**")
                elif interpretation == "INTERMEDIATE":
                    st.warning(f"⚠️ **{interpretation}**")
                else:
                    st.info(f"❓ **{interpretation}**")

                st.markdown(f"**Details:** {result.get('message', '')}")

                if result.get("breakpoints"):
                    bp = result["breakpoints"]
                    st.markdown(f"""
                    **Breakpoints:**
                    - S ≤ {bp.get('susceptible', 'N/A')} mg/L
                    - R > {bp.get('resistant', 'N/A')} mg/L
                    """)

                if result.get("notes"):
                    st.info(f"**Note:** {result.get('notes')}")
        else:
            st.warning("Please enter both pathogen and antibiotic names.")


def show_mic_trend_analysis():
    st.header("📊 MIC Trend Analysis")
    st.markdown("*Detect MIC creep over time*")

    st.markdown("""
    Enter historical MIC values to detect resistance velocity.
    **MIC Creep**: A gradual increase in MIC that may predict treatment failure
    even when the organism is still classified as "Susceptible".
    """)

    # Input for historical MICs
    num_readings = st.slider("Number of historical readings", 2, 6, 3)

    mic_values = []
    cols = st.columns(num_readings)

    for i, col in enumerate(cols):
        with col:
            mic = col.number_input(
                f"MIC {i+1}",
                min_value=0.001,
                max_value=256.0,
                value=float(2 ** i),  # Default: 1, 2, 4, ...
                key=f"mic_{i}"
            )
            mic_values.append({"date": f"T{i}", "mic_value": mic})

    if st.button("Analyze Trend", type="primary"):
        result = calculate_mic_trend(mic_values)

        risk_level = result.get("risk_level", "UNKNOWN")

        if risk_level == "HIGH":
            st.markdown(f'<div class="risk-high"><strong>🚨 HIGH RISK</strong><br>{result.get("alert", "")}</div>',
                       unsafe_allow_html=True)
        elif risk_level == "MODERATE":
            st.markdown(f'<div class="risk-moderate"><strong>⚠️ MODERATE RISK</strong><br>{result.get("alert", "")}</div>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-low"><strong>✅ LOW RISK</strong><br>{result.get("alert", "")}</div>',
                       unsafe_allow_html=True)

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Baseline MIC", f"{result.get('baseline_mic', 'N/A')} mg/L")
        with col2:
            st.metric("Current MIC", f"{result.get('current_mic', 'N/A')} mg/L")
        with col3:
            st.metric("Fold Change", f"{result.get('ratio', 'N/A')}x")

        st.markdown(f"**Trend:** {result.get('trend', 'N/A')}")
        st.markdown(f"**Resistance Velocity:** {result.get('velocity', 'N/A')}x per time point")


def show_drug_safety():
    st.header("⚠️ Drug Safety Check")
    st.markdown("*Screen for drug interactions*")

    col1, col2 = st.columns(2)

    with col1:
        antibiotic = st.text_input(
            "Proposed Antibiotic",
            placeholder="e.g., Ciprofloxacin"
        )

        current_meds = st.text_area(
            "Current Medications (one per line)",
            placeholder="Warfarin\nMetformin\nAmlodipine",
            height=150
        )

    with col2:
        allergies = st.text_area(
            "Known Allergies (one per line)",
            placeholder="Penicillin\nSulfa",
            height=100
        )

    if st.button("Check Safety", type="primary"):
        if antibiotic:
            medications = [m.strip() for m in current_meds.split("\n") if m.strip()]
            allergy_list = [a.strip() for a in allergies.split("\n") if a.strip()]

            with st.spinner("Checking interactions..."):
                result = screen_antibiotic_safety(antibiotic, medications, allergy_list)

                if result.get("safe_to_use"):
                    st.success("✅ No critical safety concerns identified")
                else:
                    st.error("❌ SAFETY CONCERNS IDENTIFIED")

                # Show alerts
                if result.get("alerts"):
                    st.subheader("Alerts")
                    for alert in result["alerts"]:
                        level = alert.get("level", "WARNING")
                        if level == "CRITICAL":
                            st.error(f"🚨 {alert.get('message', '')}")
                        else:
                            st.warning(f"⚠️ {alert.get('message', '')}")

                # Show allergy warnings
                if result.get("allergy_warnings"):
                    st.subheader("Allergy Warnings")
                    for warn in result["allergy_warnings"]:
                        st.error(f"🚫 {warn.get('message', '')}")

                # Show interactions
                if result.get("interactions"):
                    st.subheader("Drug Interactions Found")
                    for interaction in result["interactions"][:5]:
                        severity = interaction.get("severity", "unknown")
                        icon = "🔴" if severity == "major" else "🟡" if severity == "moderate" else "🟢"
                        st.markdown(f"""
                        {icon} **{interaction.get('drug_1')}** ↔ **{interaction.get('drug_2')}**
                        - Severity: {severity.upper()}
                        - {interaction.get('interaction_description', '')}
                        """)
        else:
            st.warning("Please enter an antibiotic name.")


def show_guidelines_search():
    st.header("📚 Clinical Guidelines Search")
    st.markdown("*Search IDSA treatment guidelines*")

    query = st.text_input(
        "Search Query",
        placeholder="e.g., treatment for ESBL E. coli UTI"
    )

    pathogen_filter = st.selectbox(
        "Filter by Pathogen Type (optional)",
        ["All", "ESBL-E", "CRE", "CRAB", "DTR-PA", "S.maltophilia", "AmpC-E"]
    )

    if st.button("Search Guidelines", type="primary"):
        if query:
            with st.spinner("Searching clinical guidelines..."):
                filter_value = None if pathogen_filter == "All" else pathogen_filter

                results = search_clinical_guidelines(query, pathogen_filter=filter_value, n_results=5)

                if results:
                    st.subheader(f"Found {len(results)} relevant excerpts")

                    for i, result in enumerate(results, 1):
                        with st.expander(
                            f"Result {i} - {result.get('pathogen_type', 'General')} "
                            f"(Relevance: {result.get('relevance_score', 0):.2f})"
                        ):
                            st.markdown(result.get("content", ""))
                            st.caption(f"Source: {result.get('source', 'IDSA Guidelines')}")
                else:
                    st.info("No results found. Try a different query or remove the filter.")
        else:
            st.warning("Please enter a search query.")


if __name__ == "__main__":
    main()
