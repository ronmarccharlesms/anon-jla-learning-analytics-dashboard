"""
dashboard.py
============
Streamlit front-end for the FEU High School Academic Performance Dashboard.

This file is the sole consumer of analysis_engine.py.  All statistical
computation, feature engineering, and modelling logic lives in the engine;
this file handles only page layout, widget state, caching, and chart
rendering.

Page structure
--------------
General Analysis — SHS (sidebar page 1)
    Tab 1 — Overview & Trends
        School-year filter, strand KPI tiles, Subject Performance Extremes,
        Grade Distribution box plot, Grade Density KDE overlay.

    Tab 2 — Subject Deep Dive
        Single-subject skew-normal deep-dive, Head-to-Head comparison.

    Tab 3 — Correlations
        Cohort filters, Correlation Heatmap, Scatter Grid (All / Top 20% /
        Bottom 20%).

    Tab 4 — Predictive Outlook
        Macro RF model validation expander, strand/grade/semester prediction
        controls, risk KPI summary, prediction bar chart, full prediction table.

General Analysis — JHS (sidebar page 2)
    Same analytical categories, using JHS quarter and annual-subject-grade logic.

Student Profile (sidebar page 3)
    Search by name or SIS ID (minimum 3 characters).
    Identity KPI card (8 metrics).
    Academic Trajectory (Growth Curve) + Performance Radar (Spider Chart).
    Subject-Level Peer Comparison (Dumbbell Plot).
    Predictive Grade Outlook (current semester, micro RF model).
    Future Term Forecast (untaken curriculum subjects, same micro RF model).
    Full Academic Transcript (sortable table).

Guide (sidebar page 5)
    Faculty and administrator operating reference with visual feature maps.

Caching strategy
----------------
@st.cache_resource on get_data(scope)     — one active SHS, JHS, or combined DataFrame.
@st.cache_resource on get_macro_models()  — non-serialisable sklearn objects.
@st.cache_resource on get_micro_models()  — non-serialisable sklearn objects.

The @st.cache_resource functions use a leading underscore on the DataFrame
parameter (_df) to prevent Streamlit from attempting to hash it.

Deployment
----------
Restricted to institution-authorised users in compliance with the Philippine
Data Privacy Act of 2012 (Republic Act No. 10173).  A public replication
repository with synthetic data is available at [Repository URL].
"""
# for streamlit dashboard
# dashboard.py

import streamlit as st
import analysis_engine as engine
import grade12_prediction_test as grade12_test
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.express as px
import re

# chart_config_global — applied to every st.plotly_chart() call in the file.
# scrollZoom is disabled because accidental scroll-wheel zoom is disorienting
# for non-technical users reviewing charts during board meetings.
# Six mode-bar buttons are removed to reduce UI clutter; autoScale and
# resetScale are retained for users who need to restore the default view.
chart_config_global = {
    'scrollZoom': False,
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': [
        'zoom2d', 'pan2d', 'select2d', 'lasso2d', 'zoomIn2d', 'zoomOut2d'
    ]
}

# Match the SHS longitudinal boxplot palette for the experimental TEST charts.
TEST_BOLD_COLORS = px.colors.qualitative.Bold
TEST_STATUS_COLORS = {
    'Likely Improved': TEST_BOLD_COLORS[0],
    'Likely Maintained': TEST_BOLD_COLORS[1],
    'Likely Declined': TEST_BOLD_COLORS[2],
    'Uncertain': TEST_BOLD_COLORS[3],
}
TEST_MODEL_COLORS = {
    'Ordinal Logistic Regression': TEST_BOLD_COLORS[0],
    'Regularized Multinomial Logistic': TEST_BOLD_COLORS[1],
    'Random Forest Classifier': TEST_BOLD_COLORS[2],
    'Extra Trees Classifier': TEST_BOLD_COLORS[3],
    'Probability Consensus': TEST_BOLD_COLORS[4],
}

TEST_PAGE = "TEST"


def render_location_strip(level, active_tab, sequence):
    """Render a compact visual breadcrumb for the current dashboard view."""
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; align-items:center; gap:.4rem; margin:.15rem 0 .8rem; padding:.45rem .65rem; border:1px solid rgba(128,128,128,.35); border-radius:.45rem; font-size:.86rem; color:inherit;">
          <span style="opacity:.7;">Dashboard</span>
          <span>›</span>
          <strong>{level}</strong>
          <span>›</span>
          <strong>{active_tab}</strong>
          <span style="opacity:.7;">·</span>
          <span style="opacity:.78;">{sequence}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.set_page_config(layout="wide", page_title="FEU High School Academic Performance Dashboard")

# Canvas LMS gradebook export files — one per semester, ordered chronologically.
# The tenth file (AY 2025-2026, Semester 2) is added when the semester completes.
csv_files = [
    "gb_2021-2022_1.csv", "gb_2021-2022_2.csv",
    "gb_2022-2023_1.csv", "gb_2022-2023_2.csv",
    "gb_2023-2024_1.csv", "gb_2023-2024_2.csv",
    "gb_2024-2025_1.csv", "gb_2024-2025_2.csv",
    "gb_2025-2026_1.csv", "gb_2025-2026_2.csv"
]
# JHS files are discovered so new quarter exports do not require a code edit.
# The engine accepts the standardized CSV convention and the existing legacy
# Excel filenames during the migration.
jhs_files = engine.discover_jhs_files("JHS_historical-grades")

@st.cache_resource(max_entries=1)
def get_data(scope="combined"):
    """Load only the active data scope and retain one shared in-memory frame.

    The Guide does not need academic data. General-analysis pages load only
    their school level, while Student Profile loads the combined corpus. A
    one-entry resource cache prevents separate SHS, JHS, and combined frames
    from accumulating as users move between pages.
    """
    if scope == "shs":
        return engine.load_combined_data(csv_files, [])
    if scope == "jhs":
        return engine.load_combined_data([], jhs_files)
    return engine.load_combined_data(csv_files, jhs_files)

@st.cache_resource
def get_macro_models(_df):
    """Train cohort-level RF models once per session.

    @st.cache_resource is required (not @st.cache_data) because sklearn model
    objects are not serialisable.  The leading underscore on _df prevents
    Streamlit from attempting to hash the DataFrame argument.
    """
    return engine.train_macro_model(_df)

@st.cache_resource
def get_micro_models(_df):
    """Train student-subject RF models once per session.  See get_macro_models."""
    return engine.train_micro_model(_df)

@st.cache_resource
def get_jhs_models(_df):
    """Train the separate quarter-aware JHS RF models once per session."""
    return engine.train_jhs_model(_df)


@st.cache_data(show_spinner=False)
def get_grade12_test_data():
    """Load the isolated external-grade prediction study once per session."""
    return grade12_test.load_study_data(".")

# Sidebar: page navigation, attribution, and admin cache control.
page = st.sidebar.radio(
    "Navigation", [
        "General Analysis — SHS", "General Analysis — JHS",
        "Student Profile", "Guide", TEST_PAGE
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by: <br>**Ron Marc Charles Sombillo, MSc**", unsafe_allow_html=True)
st.sidebar.caption("Education Technology Coordinator <br>FEU High School 2025", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Admin Controls")
st.sidebar.caption("Clear cached data and reload the latest source files after an update.")
if st.sidebar.button("Clear Cache & Reload Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

# Load only the active data scope; Guide renders without loading the academic
# corpus at all. This avoids paying the combined-data memory cost for every
# visitor and prevents unused SHS/JHS working copies from coexisting.
df = pd.DataFrame()
all_df = pd.DataFrame()
shs_df = pd.DataFrame()
jhs_quarter_df = pd.DataFrame()
jhs_annual_df = pd.DataFrame()
if page not in ("Guide", TEST_PAGE):
    data_scope = {
        "General Analysis — SHS": "shs",
        "General Analysis — JHS": "jhs",
        "Student Profile": "combined",
    }[page]
    try:
        with st.spinner("Loading student data..."):
            df = get_data(data_scope)
        if df.empty:
            st.error("No data loaded. Check files."); st.stop()
    except Exception as e:
        st.error(f"Error: {e}"); st.stop()

    # Split only the active level. The Student Profile remains unified so
    # students who attended both levels are searchable in one place.
        importance = grade12_test.granular_feature_importance(
            granular_data, primary_model
        )
        if not importance.empty:
            st.subheader(f"Feature importance: {primary_model}")
            st.caption(
                "Explanatory full-cohort fit, not out-of-fold validation evidence. "
                "Tree importance and absolute coefficients show model association, not causation."
            )
            importance_plot = importance.head(20).sort_values('Importance')
            importance_fig = px.bar(
                importance_plot,
                x='Importance', y='Feature', orientation='h',
                title=f"Top predictors used by {primary_model}",
                hover_data=['Importance Type']
            )
            importance_fig.update_layout(dragmode=False, yaxis_title=None)
            st.plotly_chart(importance_fig, width='stretch', config=chart_config_global)
    all_df = df
    if page == "General Analysis — SHS":
        shs_df = df
    elif page == "General Analysis — JHS":
        jhs_quarter_df = df[
            df['school_level'].eq('JHS') & df['period_type'].eq('quarter')
        ].copy()
        jhs_annual_df = df[
            df['school_level'].eq('JHS') & df['period_type'].eq('annual')
        ].copy()
    else:
        shs_df = all_df[all_df['school_level'].eq('SHS')].copy()
        jhs_quarter_df = all_df[
            all_df['school_level'].eq('JHS') & all_df['period_type'].eq('quarter')
        ].copy()
        jhs_annual_df = all_df[
            all_df['school_level'].eq('JHS') & all_df['period_type'].eq('annual')
        ].copy()

if page == "Guide":
    st.title("Guide")
    st.caption(
        "Concise operating reference for faculty and administrators using the FEU High School academic dashboard."
    )
    st.info(
        "Recommended workflow: choose a page → set the relevant filters → read the KPI summary → inspect the charts → "
        "open detail tables only when the raw records are needed."
    )

    guide_start, guide_analysis, guide_profile, guide_methods = st.tabs([
        "Start Here", "General Analysis", "Student Profile", "Definitions & Operations"
    ])

    with guide_start:
        st.header("Purpose and scope")
        st.markdown(
            "- **Purpose:** support academic monitoring, cohort review, subject review, and early administrative prioritization.\n"
            "- **Users:** authorized faculty, coordinators, school leaders, and administrators.\n"
            "- **SHS:** semester-based records for Grade 11–12 and strand cohorts.\n"
            "- **JHS:** quarterly records for Grades 7–10; annual subject grade is the mean of Q1–Q4.\n"
            "- **Unified Student Profile:** searches one student across both school levels when records exist for both.\n"
            "- **Privacy:** treat names, student numbers, grades, predictions, and class standing as restricted information."
        )

        st.header("Navigation at a glance")
        st.markdown(
            "| Page | Use it for | Main output |\n"
            "|---|---|---|\n"
            "| **General Analysis — SHS** | SHS cohort and strand monitoring | Historical patterns and semester outlook |\n"
            "| **General Analysis — JHS** | JHS grade-level and quarter monitoring | Historical patterns and quarterly outlook |\n"
            "| **Student Profile** | Individual review | KPIs, trajectory, peer comparison, predictions, and transcripts |\n"
            "| **Guide** | Interpretation and operating reference | Definitions, workflow, and limitations |"
        )

        st.subheader("Visual location map")
        st.caption("Use the sidebar first; then choose the tab containing the feature you need.")
        st.markdown(
            """
            <div style="display:flex; flex-direction:column; gap:0.45rem; max-width:100%;">
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.5rem; padding:.65rem .8rem;">
                <strong>Sidebar Navigation</strong><br>
                Guide · General Analysis — SHS · General Analysis — JHS · Student Profile
              </div>
              <div style="text-align:center; font-size:1.15rem; line-height:1;">↓</div>
              <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.55rem;">
                <div style="border-left:4px solid #4C78A8; border-radius:.35rem; padding:.65rem .75rem; background:rgba(76,120,168,.10);">
                  <strong>General Analysis — SHS</strong><br>
                  Overview &amp; Trends · Subject Deep Dive · Correlations · Predictive Outlook
                </div>
                <div style="border-left:4px solid #F58518; border-radius:.35rem; padding:.65rem .75rem; background:rgba(245,133,24,.10);">
                  <strong>General Analysis — JHS</strong><br>
                  Overview &amp; Trends · Subject Deep Dive · Correlations · Predictive Outlook
                </div>
              </div>
              <div style="text-align:center; font-size:1.15rem; line-height:1;">↓</div>
              <div style="border-left:4px solid #54A24B; border-radius:.35rem; padding:.65rem .8rem; background:rgba(84,162,75,.10);">
                <strong>Student Profile</strong><br>
                Search by name/SIS ID → review the available JHS and SHS sections → expand detail tables only when needed
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Analysis-tab map")
        st.markdown(
            """
            <div style="display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.55rem; max-width:100%;">
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.6rem;">
                <strong>1 · Overview</strong><br>KPIs, distributions, density, subject extremes
              </div>
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.6rem;">
                <strong>2 · Deep Dive</strong><br>One subject and head-to-head comparison
              </div>
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.6rem;">
                <strong>3 · Correlations</strong><br>Heatmap and all/top/bottom scatter grids
              </div>
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.6rem;">
                <strong>4 · Predictive</strong><br>Model validation, forecast, risk, full table
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.header("Quick operating sequence")
        st.markdown(
            "1. Select **SHS** or **JHS** for cohort analysis, or **Student Profile** for an individual.\n"
            "2. Set the school year, grade, strand, reporting period, or prediction target shown on that page.\n"
            "3. Read descriptive results first; use predictive results for prioritization and planning.\n"
            "4. Use chart legends, hover details, and expanders for additional context.\n"
            "5. If source files were updated, use **Clear Cache & Reload Data** after confirming the app is not being used for live review."
        )

        st.subheader("What kind of result am I reading?")
        st.markdown(
            """
            <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.65rem;">
              <div style="border-left:4px solid #4C78A8; border-radius:.45rem; padding:.7rem .8rem; background:rgba(76,120,168,.10);">
                <strong>Descriptive · What happened?</strong><br>
                Actual grades → averages, passing rates, distributions, rankings, correlations
              </div>
              <div style="border-left:4px solid #F58518; border-radius:.45rem; padding:.7rem .8rem; background:rgba(245,133,24,.10);">
                <strong>Predictive · What may happen?</strong><br>
                Historical patterns → predicted grades, peer baselines, risk-priority flags
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with guide_analysis:
        st.header("General Analysis — SHS")
        st.caption("Semester-level cohort analysis organized around strands.")
        st.subheader("Static view of the analysis tabs")
        st.caption("The same four tab positions are used for SHS and JHS; the grouping and period labels differ.")
        st.markdown(
            """
            <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.7rem;">
              <div style="border:1px solid rgba(128,128,128,.4); border-radius:.5rem; padding:.7rem;">
                <strong>1 · Overview &amp; Trends</strong>
                <div style="display:flex; gap:.35rem; margin:.55rem 0 .45rem;">
                  <span style="border:1px solid rgba(128,128,128,.35); border-radius:.3rem; padding:.3rem; font-size:.72rem;">Students<br><b>1,240</b></span>
                  <span style="border:1px solid rgba(128,128,128,.35); border-radius:.3rem; padding:.3rem; font-size:.72rem;">Avg<br><b>88.4</b></span>
                  <span style="border:1px solid rgba(128,128,128,.35); border-radius:.3rem; padding:.3rem; font-size:.72rem;">Pass<br><b>91%</b></span>
                </div>
                <div style="height:28px; display:flex; align-items:end; gap:4px;">
                  <span style="height:55%; width:14%; background:#76A5D2;"></span><span style="height:78%; width:14%; background:#76A5D2;"></span><span style="height:65%; width:14%; background:#76A5D2;"></span><span style="height:90%; width:14%; background:#76A5D2;"></span><span style="height:72%; width:14%; background:#76A5D2;"></span>
                </div>
                <small>KPIs · ranked subjects · box plot · density</small>
              </div>
              <div style="border:1px solid rgba(128,128,128,.4); border-radius:.5rem; padding:.7rem;">
                <strong>2 · Subject Deep Dive</strong>
                <div style="display:flex; gap:.35rem; margin:.55rem 0 .45rem;">
                  <span style="border:1px solid rgba(128,128,128,.35); border-radius:.3rem; padding:.3rem; font-size:.72rem;">Avg<br><b>86.2</b></span>
                  <span style="border:1px solid rgba(128,128,128,.35); border-radius:.3rem; padding:.3rem; font-size:.72rem;">Pass<br><b>87%</b></span>
                  <span style="border:1px solid rgba(128,128,128,.35); border-radius:.3rem; padding:.3rem; font-size:.72rem;">SD<br><b>4.8</b></span>
                </div>
                <div style="height:28px; display:flex; align-items:center; gap:5px;"><span style="height:8px; width:34%; background:#B8D8BE; border-radius:4px;"></span><span style="height:8px; width:45%; background:#76A5D2; border-radius:4px;"></span><span style="height:8px; width:22%; background:#B8D8BE; border-radius:4px;"></span></div>
                <small>One subject · distribution · head-to-head</small>
              </div>
              <div style="border:1px solid rgba(128,128,128,.4); border-radius:.5rem; padding:.7rem;">
                <strong>3 · Correlations</strong>
                <div style="display:flex; gap:.8rem; align-items:center; margin:.55rem 0 .45rem;">
                  <div style="display:grid; grid-template-columns:repeat(4,12px); gap:2px;">
                    <i style="height:12px;background:#2166AC;"></i><i style="height:12px;background:#67A9CF;"></i><i style="height:12px;background:#D1E5F0;"></i><i style="height:12px;background:#F7F7F7;"></i>
                    <i style="height:12px;background:#67A9CF;"></i><i style="height:12px;background:#2166AC;"></i><i style="height:12px;background:#F4A582;"></i><i style="height:12px;background:#D6604D;"></i>
                    <i style="height:12px;background:#D1E5F0;"></i><i style="height:12px;background:#F4A582;"></i><i style="height:12px;background:#2166AC;"></i><i style="height:12px;background:#92C5DE;"></i>
                    <i style="height:12px;background:#F7F7F7;"></i><i style="height:12px;background:#D6604D;"></i><i style="height:12px;background:#92C5DE;"></i><i style="height:12px;background:#2166AC;"></i>
                  </div>
                  <div style="height:36px; width:64px; border-left:1px solid currentColor; border-bottom:1px solid currentColor; position:relative;"><span style="position:absolute; left:8px; bottom:8px; width:5px; height:5px; border-radius:50%; background:#4C78A8;"></span><span style="position:absolute; left:28px; bottom:20px; width:5px; height:5px; border-radius:50%; background:#4C78A8;"></span><span style="position:absolute; left:46px; bottom:28px; width:5px; height:5px; border-radius:50%; background:#4C78A8;"></span></div>
                </div>
                <small>Heatmap · scatter grid · top/bottom 20%</small>
              </div>
              <div style="border:1px solid rgba(128,128,128,.4); border-radius:.5rem; padding:.7rem;">
                <strong>4 · Predictive Outlook</strong>
                <div style="margin:.55rem 0 .45rem;">
                  <div style="display:flex; align-items:center; gap:5px; margin:3px 0;"><span style="width:42%; height:9px; background:#81C784;"></span><span style="font-size:.7rem;">On Track</span></div>
                  <div style="display:flex; align-items:center; gap:5px; margin:3px 0;"><span style="width:30%; height:9px; background:#FFD54F;"></span><span style="font-size:.7rem;">Moderate</span></div>
                  <div style="display:flex; align-items:center; gap:5px; margin:3px 0;"><span style="width:20%; height:9px; background:#E57373;"></span><span style="font-size:.7rem;">High Risk</span></div>
                </div>
                <small>Validation → prediction → risk summary → table</small>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "**Overview & Trends**\n"
            "- **School Year:** limits the overview and subject-extreme results to one academic year.\n"
            "- **Performance Overview:** reports total students, strand average, and strand passing rate.\n"
            "- **Subject Performance Extremes:** five lowest-mean and five highest-mean subjects within each strand.\n"
            "- **Rank labels:** A–E replace 1–5; A is the most difficult subject among the displayed subjects.\n"
            "- **Grade Distributions:** box plots show median, spread, and outliers across school years.\n"
            "- **Grade Density:** KDE-style distributions show where grades are concentrated.\n"
            "- Use the school-year legend to turn a year’s bar, whisker, or density trace on or off.\n\n"
            "**Subject Deep Dive**\n"
            "- Select one subject to view its historical distribution and summary metrics.\n"
            "- **Global Subject Average:** mean numeric grade in the available SHS records.\n"
            "- **Global Passing Rate:** share of grades at or above 75.\n"
            "- **Variability (SD):** spread of grades; larger values indicate less consistency.\n"
            "- **Subject Comparison:** compare two selected subjects head-to-head by achievement and variability.\n\n"
            "**Correlations**\n"
            "- Filter by school year, grade level, and strand.\n"
            "- **Heatmap:** compact matrix of pairwise subject correlations.\n"
            "- **Scatter Grid (All):** subject-pair scatter plots for the selected cohort.\n"
            "- **Top 20% / Bottom 20%:** repeat the scatter analysis for the highest- or lowest-performing subgroup and show its student report.\n"
            "- Correlation indicates co-movement, not cause-and-effect. Small cohorts should be interpreted cautiously.\n\n"
            "**Predictive Outlook — Next Semester Performance**\n"
            "- Select strand, grade level, and semester, then generate a cohort-level subject outlook.\n"
            "- **Predicted Mean:** Random Forest regression output.\n"
            "- **Peer Mean:** comparison baseline for the selected cohort.\n"
            "- **Risk status:** separate classifier-based administrative priority score.\n"
            "- The validation expander reports MAE, R², AUC-ROC, train/test samples, and feature importance.\n"
            "- The full prediction table is available in an expander."
        )

        st.header("General Analysis — JHS")
        st.caption("Quarter-aware grade-level analysis using JHS-specific data and models.")
        st.markdown(
            "- **School Year, Grade Level, Reporting Period:** filters are shown above the overview tabs and reused by the overview and subject analysis.\n"
            "- **Performance Overview:** reports total students, subject count, and average/passing results by grade level.\n"
            "- **Subject Performance Extremes:** three lowest-mean and three highest-mean subjects within each grade level.\n"
            "- **Rank labels:** A–E; A is the most difficult displayed subject and E is the least difficult.\n"
            "- **Grade Distributions and Density:** compare JHS grade levels across school years; toggle school years through the legend.\n"
            "- **Subject Deep Dive:** inspect one subject within the selected JHS view, including average, passing rate, SD, distribution, and head-to-head comparison.\n"
            "- **Correlations:** choose school year, grade, reporting period, and view type; heatmap and all/top/bottom-20% scatter grids are available.\n"
            "- **Predictive Outlook — Next Quarter:** choose grade and target quarter; the JHS Random Forest model returns subject predictions, peer means, risk counts, validation metrics, and a full table.\n"
            "- JHS annual subject grades require all four quarters; incomplete annual records remain blank."
        )

    with guide_profile:
        st.header("Student Profile workflow")
        st.markdown(
            "1. Search by student name or student number/SIS ID; at least three characters are required.\n"
            "2. Select the intended match when multiple students are returned.\n"
            "3. Review the unified identity header, then read the available **JHS** and/or **SHS** sections.\n"
            "4. Use predictive flags to prioritize follow-up; confirm decisions against actual records and teacher knowledge."
        )

        st.subheader("Visual profile map")
        st.markdown(
            """
            <div style="display:flex; flex-wrap:wrap; align-items:center; gap:.45rem; max-width:100%;">
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.55rem .7rem;"><strong>Search</strong><br>Name or SIS ID</div>
              <span style="font-size:1.15rem;">→</span>
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.55rem .7rem;"><strong>Profile Header</strong><br>Name + student number</div>
              <span style="font-size:1.15rem;">→</span>
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.55rem .7rem;"><strong>JHS / SHS KPIs</strong><br>Achievement snapshot</div>
              <span style="font-size:1.15rem;">→</span>
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.55rem .7rem;"><strong>Charts</strong><br>Trajectory + peers</div>
              <span style="font-size:1.15rem;">→</span>
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.55rem .7rem;"><strong>Predictions</strong><br>Current + future</div>
              <span style="font-size:1.15rem;">→</span>
              <div style="border:1px solid rgba(128,128,128,.45); border-radius:.45rem; padding:.55rem .7rem;"><strong>Expand details</strong><br>Transcripts + tables</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Static profile dashboard")
        st.caption("This mirrors the order of the actual student profile from top to bottom.")
        st.markdown(
            """
            <div style="border:1px solid rgba(128,128,128,.4); border-radius:.5rem; padding:.7rem;">
              <div style="display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.35rem; margin-bottom:.65rem;">
                <span style="border:1px solid rgba(128,128,128,.35); padding:.35rem; font-size:.72rem;">Student<br><b>H2021900910</b></span>
                <span style="border:1px solid rgba(128,128,128,.35); padding:.35rem; font-size:.72rem;">Grade / Strand<br><b>G11 · STEM</b></span>
                <span style="border:1px solid rgba(128,128,128,.35); padding:.35rem; font-size:.72rem;">GPA<br><b>90.4</b></span>
                <span style="border:1px solid rgba(128,128,128,.35); padding:.35rem; font-size:.72rem;">Class Standing<br><b>88.0th</b></span>
              </div>
              <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.55rem;">
                <div style="border:1px solid rgba(128,128,128,.3); padding:.5rem; min-height:75px;"><strong>Trajectory</strong><br><span style="color:#7B3F98; font-size:1.2rem;">●—●—●—●</span><br><small>SHS terms or latest JHS grade/year</small></div>
                <div style="border:1px solid rgba(128,128,128,.3); padding:.5rem; min-height:75px;"><strong>Radar</strong><br><span style="color:#7B3F98; letter-spacing:.25rem;">◆◆◆</span><br><small>Student versus peer subject profile</small></div>
                <div style="border:1px solid rgba(128,128,128,.3); padding:.5rem; min-height:75px;"><strong>Peer Comparison</strong><br><span style="color:#4C78A8;">—◆——●—</span><br><small>Subject grade versus peer mean</small></div>
                <div style="border:1px solid rgba(128,128,128,.3); padding:.5rem; min-height:75px;"><strong>Predictions</strong><br><span style="color:#81C784;">████</span> <span style="color:#FFD54F;">██</span><br><small>Current outlook and future forecast</small></div>
              </div>
              <div style="margin-top:.55rem; border:1px dashed rgba(128,128,128,.45); padding:.45rem; font-size:.78rem;">▸ View JHS Quarterly Transcript &nbsp;&nbsp; ▸ View Forecast Details &nbsp;&nbsp; ▸ View Full Academic Transcript</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.header("Profile components")
        st.markdown(
            "| Component | Meaning |\n"
            "|---|---|\n"
            "| **KPI summary** | Identity, current/latest level, GPA, subject count, highest/lowest grade, and class standing |\n"
            "| **Performance Visualization** | Trajectory and subject profile relative to peers |\n"
            "| **Academic Trajectory** | SHS: G11-S1, G11-S2, G12-S1, G12-S2; JHS: latest grade level in the student’s most recent attended year |\n"
            "| **Radar / Spider Chart** | Student subject grades compared with peer averages |\n"
            "| **Subject-Level Peer Comparison** | Student grade versus peer mean by subject |\n"
            "| **Predictive Grade Outlook** | Current-term/quarter subject predictions and administrative risk status |\n"
            "| **Future Term Forecast** | Untaken curriculum subjects with projected grades; planning output, not completed records |\n"
            "| **View JHS Quarterly Transcript** | Q1–Q4 grades plus completed annual grade; incomplete annual values remain blank |\n"
            "| **View Full Academic Transcript** | Collapsed SHS term-level raw records by subject, strand, and section |"
        )

        st.header("Reading profile predictions")
        st.markdown(
            "- **Predicted grade and peer mean are different quantities:** one is the model output; the other is a reference baseline.\n"
            "- A grade above the threshold can still be yellow or red because risk status uses the student’s broader feature pattern.\n"
            "- Risk scores are prioritization indicators, not calibrated probabilities or final judgments.\n"
            "- Open prediction tables only when subject-level raw details are needed."
        )

    with guide_methods:
        st.subheader("Static interpretation aids")
        st.markdown(
            """
            <div style="display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.65rem;">
              <div style="border:1px solid rgba(128,128,128,.4); border-radius:.45rem; padding:.65rem;"><strong>JHS annual grade</strong><br><span style="font-size:1.05rem;">(Q1 + Q2 + Q3 + Q4) ÷ 4</span><br><small>All four quarters required; incomplete annual result stays blank.</small></div>
              <div style="border:1px solid rgba(128,128,128,.4); border-radius:.45rem; padding:.65rem;"><strong>Risk bands</strong><br><span style="color:#E57373;">● ≥ .60</span> &nbsp; <span style="color:#F9C32E;">● .35–.59</span> &nbsp; <span style="color:#43A047;">● &lt; .35</span><br><small>Priority bands, not calibrated probabilities.</small></div>
              <div style="border:1px solid rgba(128,128,128,.4); border-radius:.45rem; padding:.65rem;"><strong>Correlation</strong><br><span style="font-size:1.1rem;">Subject A ↗ Subject B</span><br><small>Relationship pattern; not evidence that one subject causes another.</small></div>
              <div style="border:1px solid rgba(128,128,128,.4); border-radius:.45rem; padding:.65rem;"><strong>Refresh flow</strong><br><span style="font-size:1.05rem;">Source files → Cache → Models → Dashboard</span><br><small>Clear cache after approved data updates.</small></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.header("Metric definitions")
        st.markdown(
            "- **Numeric grade:** cleaned grade value used by analysis and models.\n"
            "- **Passing rate:** proportion of included grades ≥ 75.\n"
            "- **GPA / average:** arithmetic mean of available numeric grades within the stated scope.\n"
            "- **JHS annual subject grade:** arithmetic mean of Q1, Q2, Q3, and Q4; fewer than four quarters produces a blank annual value.\n"
            "- **Class standing:** percentage of the applicable latest cohort whose GPA is at or below the student’s GPA; cohort size limitations may return N/A.\n"
            "- **Subject difficulty:** lower mean grade indicates greater observed difficulty; A–E labels are relative ranks, not curriculum difficulty diagnoses.\n"
            "- **Peer mean:** average of the matched comparison cohort used by the relevant feature."
        )

        st.header("Predictive model interpretation")
        st.markdown(
            "- **Regression model:** estimates a grade/mean grade for the selected subject and period.\n"
            "- **At-risk classifier:** estimates a model score from the historical feature pattern.\n"
            "- **Historical threshold:** the dashboard uses 80 as the at-risk grade threshold for training labels and chart reference lines.\n"
            "- **Status bands:** High Risk ≥ 0.60; Moderate 0.35–<0.60; On Track < 0.35.\n"
            "- **Validation:** MAE is grade-point error; R² summarizes explained variation; AUC-ROC summarizes ranking separation.\n"
            "- Validation metrics describe the available historical corpus and should not be treated as guarantees for a future cohort."
        )

        st.header("Data refresh and troubleshooting")
        st.markdown(
            "- Add or replace approved source exports in the configured data locations.\n"
            "- Confirm column structure, school year, period labels, student numbers, and numeric grades before refreshing.\n"
            "- Use **Clear Cache & Reload Data** to load the latest files and retrain cached models.\n"
            "- If a chart is empty, first check the selected year, grade, strand, reporting period, cohort size, and data completeness.\n"
            "- If predictions are unavailable, review the model-validation message and confirm that enough historical records remain after filtering.\n"
            "- Keep the app restricted to authorized users; do not export or share raw student tables outside approved channels."
        )

        st.header("Important boundaries")
        st.markdown(
            "- Descriptive views report observed records; predictive views estimate future performance.\n"
            "- Correlation does not establish causation.\n"
            "- Risk labels support prioritization and require professional review.\n"
            "- Relative subject ranks do not replace curriculum review or teacher judgment.\n"
            "- The guide should be updated whenever the corpus, curriculum, model features, thresholds, or dashboard behavior changes."
        )

elif page == "General Analysis — SHS":
    df = shs_df
    st.title("FEU High School Academic Performance Dashboard")
    st.caption("Administrative view of SHS enrollment, achievement patterns, subject difficulty, relationships, and projected performance.")

    general_tabs = ["Overview & Trends", "Subject Deep Dive", "Correlations", "Predictive Outlook"]
    general_tab_default = st.session_state.get(
        '_general_tab_default',
        "Overview & Trends"
    )
    tab1, tab2, tab3, tab4 = st.tabs(
        general_tabs,
        default=general_tab_default
    )

    # ── Tab 1: Overview & Trends ───────────────────────────────────────────
    with tab1:
        render_location_strip(
            'SHS', 'Overview & Trends',
            'School Year → Performance Overview → Subject Extremes → Distributions'
        )
        # Tooltip on the heading explains the two KPI metrics to users who
        # may not be familiar with the distinction between strand average
        # (all subjects) and passing rate (grades >= 75 only).
        st.markdown("""
            <h2 title="A high-level snapshot of the school's population and academic performance. 
            • Strand Average: Mean grade of all enrolled students within the strand. This includes all subjects regardless of difficulty.
            • Passing Rate: Percentage of grades ≥75 on subjects taken within the strand. Calculated per grade.">
            Performance Overview - SHS
            </h2>
        """, unsafe_allow_html=True)
        st.caption("Summarizes the selected school year by strand using average grade and passing-rate indicators.")

        all_years = sorted(df['school_year'].dropna().unique())
        selected_year_overview = st.selectbox("Select School Year for Overview", all_years, index=len(all_years)-1)

        overview_df = df[df['school_year'] == selected_year_overview]

        metrics = engine.get_overview_metrics(overview_df)

        if metrics:
            c1, c2 = st.columns(2)
            c1.metric("Total Students Enrolled", metrics["Total Students"])

            st.write("### Performance by Strand")
            st.caption("Compare average achievement and the percentage of grades at or above the passing benchmark across strands.")
            strand_cols = st.columns(len(metrics["Strands"]))
            for i, (strand, data) in enumerate(metrics["Strands"].items()):
                with strand_cols[i]:
                    st.metric(f"{strand} Average", f"{data['Avg']:.2f}")
                    st.metric(f"{strand} Passing", f"{data['PassRate']:.1f}%")
        else:
            st.warning("No data for selected year.")

        st.divider()

        st.subheader("Subject Performance Extremes")
        st.markdown("Ranks the five lowest- and highest-mean subjects within each strand for the selected school year.")

        col_hardest, col_easiest = st.columns(2)

        fig_hard, fig_easy = engine.plot_subject_extremes_split(df, selected_year_overview)

        if fig_hard is not None and fig_easy is not None:
            fig_hard.update_layout(dragmode=False)
            fig_easy.update_layout(dragmode=False)
            with col_hardest:
                st.plotly_chart(fig_hard, width='stretch', config=chart_config_global)
            with col_easiest:
                st.plotly_chart(fig_easy, width='stretch', config=chart_config_global)
        else:
            st.warning("Data insufficient to plot subject extremes for the current filters.")

        st.divider()

        c_left, c_right = st.columns(2)

        with c_left:
            st.write("### Grade Distributions Across School Years")
            st.markdown("Shows the median, spread, and outlier grades for each strand across school years.")
            fig_dist = engine.plot_grade_distribution_interactive(df)
            fig_dist.update_layout(dragmode=False)
            st.plotly_chart(fig_dist, width='stretch', config=chart_config_global)

        with c_right:
            st.write("### Grade Density Across School Years")
            st.markdown("Shows where grades are concentrated and how the distribution changes across school years.")
            fig_dens = engine.plot_grade_density_interactive(df)
            fig_dens.update_layout(dragmode=False)
            st.plotly_chart(fig_dens, width='stretch', config=chart_config_global)

    # ── Tab 2: Subject Deep Dive ───────────────────────────────────────────
    with tab2:
        render_location_strip(
            'SHS', 'Subject Deep Dive',
            'Select Subject → Summary KPIs → Distribution → Head-to-Head'
        )
        st.subheader("Subject-Level Analysis")
        st.markdown("Inspect one subject in detail, then compare two subjects by average, passing rate, spread, and distribution.")
        all_subjects = sorted(df['course'].dropna().unique())
        selected_subj = st.selectbox("Select Subject", all_subjects)
        
        if selected_subj:
            # Calculate metrics specific to this subject (Aggregated across all years)
            subj_data = df[df['course'] == selected_subj]
            avg_grade = subj_data['numeric_grade'].mean()
            pass_rate = (subj_data['numeric_grade'] >= 75).mean() * 100
            std_dev = subj_data['numeric_grade'].std()
            
            # Display Subject KPIs
            k1, k2, k3 = st.columns(3)
            k1.metric("Global Subject Average", f"{avg_grade:.2f}")
            k2.metric("Global Passing Rate", f"{pass_rate:.1f}%")
            k3.metric("Variability (SD)", f"{std_dev:.2f}")
            st.divider()
            
            fig_deep = engine.plot_subject_deep_dive_interactive(df, selected_subj)
            fig_deep.update_layout(dragmode=False)
            st.plotly_chart(fig_deep, width='stretch', config=chart_config_global)

        st.divider()

        st.subheader("Subject Comparison (Head-to-Head)")
        st.markdown("Compare two subjects to identify differences in historical achievement and variability.")

        col1, col2 = st.columns(2)
        with col1:
            subject_a = st.selectbox("Select Subject 1", all_subjects, index=0, key='sub1')
        with col2:
            default_idx_2 = 1 if len(all_subjects) > 1 else 0
            subject_b = st.selectbox("Select Subject 2", all_subjects, index=default_idx_2, key='sub2')

        if subject_a and subject_b:
            fig_compare = engine.plot_subject_comparison_interactive(df, subject_a, subject_b)
            fig_compare.update_layout(dragmode=False)
            st.plotly_chart(fig_compare, width='stretch', config=chart_config_global)

    # ── Tab 3: Correlations ───────────────────────────────────────────────
    with tab3:
        render_location_strip(
            'SHS', 'Correlations',
            'Cohort Filters → View Type → Analyze Correlations → Charts / Report'
        )
        st.subheader("Subject Correlations")
        st.markdown("Examine whether performance in subjects moves together for a selected SHS cohort; correlation does not establish causation.")
        c1, c2, c3, c4 = st.columns(4)

        with c1: sel_year   = st.selectbox("School Year",  sorted(df['school_year'].dropna().unique()), key='cor_year')
        with c2: sel_grade  = st.selectbox("Grade Level",  sorted(df['grade_level'].dropna().unique()), key='cor_grade')
        with c3: sel_strand = st.selectbox("Strand",       sorted(df['strand'].dropna().unique()),      key='cor_strand')
        with c4: view_type  = st.selectbox("View Type", [
            "Correlation Heatmap (Interactive)",
            "Scatter Grid (All)",
            "Scatter Grid (Top 20%)",
            "Scatter Grid (Bottom 20%)"
        ])

        if st.button("Analyze Correlations"):

            if "Correlation Heatmap" in view_type:
                with st.spinner("Generating Interactive Heatmap..."):
                    fig, msg = engine.plot_correlation_heatmap_interactive(df, sel_year, sel_grade, sel_strand)
                    if fig:
                        fig.update_layout(dragmode=False)
                        st.plotly_chart(fig, width='stretch', config=chart_config_global)
                    else:
                        st.warning(msg)

            elif "Scatter Grid (All)" in view_type:
                with st.spinner("Generating Interactive Grid..."):
                    figs, msg = engine.plot_pairwise_correlations_interactive(df, sel_year, sel_grade, sel_strand)
                    if figs:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, f in enumerate(figs):
                            status_text.text(f"Rendering page {i+1} of {len(figs)}...")
                            f.update_layout(dragmode=False)
                            st.plotly_chart(f, width='stretch', config=chart_config_global)
                            progress_bar.progress((i + 1) / len(figs))
                        
                        progress_bar.empty()
                        status_text.empty()
                    else: 
                        st.warning(msg)

            # 3. SCATTER GRID (Top 20%)
            elif "Top 20%" in view_type:
                with st.spinner("Analyzing Top Performers..."):
                    stats_df, stud_df, raw_df, m = engine.get_subgroup_statistics(df, sel_year, sel_grade, sel_strand, 'top')
                    if stats_df is not None:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Count", m['count']); col2.metric("Avg GPA", f"{m['avg_gpa']:.2f}"); col3.metric("Threshold", f"{m['threshold']:.2f}")
                        with st.expander("View Student Report"): st.dataframe(stud_df, width='stretch')
                        
                        top_ids = raw_df['student sis'].unique()
                        figs, msg = engine.plot_pairwise_correlations_interactive(df, sel_year, sel_grade, sel_strand, top_students=top_ids)
                        if figs:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, f in enumerate(figs):
                                status_text.text(f"Rendering page {i+1} of {len(figs)}...")
                                st.plotly_chart(f, width='stretch', config=chart_config_global)
                                progress_bar.progress((i + 1) / len(figs))
                            
                            progress_bar.empty()
                            status_text.empty()
                        else: 
                            st.warning(msg)

            elif "Bottom 20%" in view_type:
                with st.spinner("Analyzing At-Risk..."):
                    stats_df, stud_df, raw_df, m = engine.get_subgroup_statistics(df, sel_year, sel_grade, sel_strand, 'bottom')
                    if stats_df is not None:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Count", m['count']); col2.metric("Avg GPA", f"{m['avg_gpa']:.2f}"); col3.metric("Threshold", f"{m['threshold']:.2f}")
                        with st.expander("View Student Report"): st.dataframe(stud_df, width='stretch')
                        
                        bot_ids = raw_df['student sis'].unique()
                        figs, msg = engine.plot_pairwise_correlations_interactive(df, sel_year, sel_grade, sel_strand, bottom_students=bot_ids)
                        if figs:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            for i, f in enumerate(figs):
                                status_text.text(f"Rendering page {i+1} of {len(figs)}...")
                                st.plotly_chart(f, width='stretch', config=chart_config_global)
                                progress_bar.progress((i + 1) / len(figs))
                            
                            progress_bar.empty()
                            status_text.empty()
                        else: 
                            st.warning(msg)

    # ── Tab 4: Predictive Outlook ─────────────────────────────────────────
    with tab4:
        render_location_strip(
            'SHS', 'Predictive Outlook',
            'Model Validation → Configure Prediction → Risk Summary → Details'
        )
        st.subheader("Predictive Outlook — Next Semester Performance")
        st.markdown(
            "Uses a **Random Forest** ensemble model trained on historical cohort data "
            "to forecast subject-level performance for the upcoming semester. "
            "Predictions are generated per strand and grade level. "
            f"The grade threshold used to define historical at-risk records is **{engine.AT_RISK_THRESHOLD}**. "
            "The colour status is produced by a separate at-risk classifier using the full feature pattern, "
            "so a subject can be above 80 yet still receive a moderate administrative-priority score. "
            "Risk scores are prioritization indicators, not calibrated probabilities."
        )

        with st.spinner("Loading predictive models... (first load only)"):
            macro_reg, macro_cls, macro_metrics, macro_fi = get_macro_models(df)

        if macro_reg is None:
            err = macro_metrics.get('error', 'Unknown error')
            st.warning(f"Predictive model could not be trained: {err}. "
                       "Ensure at least two complete school years of data are loaded.")
        else:
            with st.expander(
                "📊 Model Validation (Temporal Split — "
                "Train: all years except latest | Test: latest year)"
            ):
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric(
                    "MAE (grade pts)",
                    macro_metrics.get('MAE', 'N/A'),
                    help="Mean Absolute Error: average difference in grade points "
                         "between predicted and actual cohort means on the test set."
                )
                m2.metric(
                    "R²",
                    macro_metrics.get('R2', 'N/A'),
                    help="Proportion of variance in cohort mean grades explained by the model."
                )
                m3.metric(
                    "AUC-ROC (At-Risk)",
                    macro_metrics.get('AUC', 'N/A'),
                    help="At-risk classification performance. "
                         "1.0 = perfect discrimination, 0.5 = random."
                )
                m4.metric("Train Samples", macro_metrics.get('train_n', 'N/A'))
                m5.metric("Test Samples",  macro_metrics.get('test_n',  'N/A'))

                if not macro_fi.empty:
                    fi_fig = engine.plot_feature_importance(
                        macro_fi,
                        title="What Historical Factors Drive Cohort Grade Predictions?"
                    )
                    fi_fig.update_layout(dragmode=False)
                    st.plotly_chart(fi_fig, width='stretch',
                                    config=chart_config_global)

            st.divider()

            st.markdown("##### Configure Prediction")
            st.caption("Choose the SHS strand, grade level, and semester to project, then generate the subject-level outlook.")
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                pred_strand = st.selectbox(
                    "Strand", sorted(df['strand'].dropna().unique()),
                    key='pred_strand'
                )
            with pc2:
                pred_grade = st.selectbox(
                    "Grade Level",
                    sorted(df['grade_level'].dropna().unique()),
                    key='pred_grade'
                )
            with pc3:
                pred_sem = st.selectbox(
                    "Predict for Semester", ['S1', 'S2'],
                    key='pred_sem'
                )

            all_years = sorted(df['school_year'].dropna().unique())
            latest_yr = all_years[-1] if all_years else '2025-2026'

            if st.button("Generate Prediction", key='run_macro_pred'):
                with st.spinner("Generating outlook..."):
                    pred_df = engine.predict_macro_outlook(
                        macro_reg, macro_cls, df,
                        pred_strand, pred_grade, pred_sem, latest_yr
                    )

                st.session_state['macro_prediction'] = {
                    'filters': (pred_strand, pred_grade, pred_sem, latest_yr),
                    'data': pred_df
                }
                st.session_state['_general_tab_default'] = "Predictive Outlook"
                st.rerun()

            saved_prediction = st.session_state.get('macro_prediction')
            active_filters = (pred_strand, pred_grade, pred_sem, latest_yr)
            if saved_prediction and saved_prediction.get('filters') == active_filters:
                pred_df = saved_prediction['data']

                if pred_df.empty:
                    st.warning(
                        "No prediction data for the selected filters. "
                        "Check that historical data exists for this strand and grade."
                    )
                else:
                    # Summary KPI row
                    risk_counts = pred_df['risk_label'].value_counts()
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("🔴 High Risk Subjects",
                               risk_counts.get('🔴 High Risk', 0))
                    rc2.metric("🟡 Moderate Risk Subjects",
                               risk_counts.get('🟡 Moderate', 0))
                    rc3.metric("🟢 On Track Subjects",
                               risk_counts.get('🟢 On Track', 0))

                    # Prediction chart
                    fig_macro_pred = engine.plot_macro_prediction_chart(
                        pred_df, pred_strand, pred_grade
                    )
                    fig_macro_pred.update_layout(dragmode=False)
                    st.plotly_chart(fig_macro_pred, width='stretch',
                                    config=chart_config_global)

                    # Detailed table
                    with st.expander("View Full Prediction Table"):
                        tbl = pred_df[[
                            'course', 'prior_mean',
                            'predicted_mean', 'risk_probability', 'risk_label'
                        ]].copy()
                        tbl.columns = [
                            'Subject', 'Prior Actual Mean',
                            'Predicted Mean', 'Risk Score', 'Status'
                        ]
                        tbl['Risk Score'] = tbl['Risk Score'].apply(
                            lambda p: f"{p * 100:.1f}%"
                        )
                        st.dataframe(tbl, width='stretch', hide_index=True)

elif page == "General Analysis — JHS":
    st.title("FEU High School Academic Performance Dashboard")
    st.caption(
        "Administrative view of JHS achievement, subject patterns, relationships, and projected quarterly performance. "
        "Quarterly grades remain visible; the annual subject grade is the mean of Q1, Q2, Q3, and Q4."
    )

    jhs_years = sorted(jhs_quarter_df['school_year'].dropna().unique())
    jhs_grades = sorted(jhs_quarter_df['grade_level'].dropna().astype(str).unique(), key=int)
    # These filters define the overview/deep-dive data view.  Keep their
    # values in session state so the other tabs can reuse the selected view
    # without rendering the controls above every tab.
    jhs_year = st.session_state.get('jhs_year', jhs_years[-1])
    if jhs_year not in jhs_years:
        jhs_year = jhs_years[-1]
        st.session_state['jhs_year'] = jhs_year
    else:
        st.session_state.setdefault('jhs_year', jhs_year)

    jhs_grade = str(st.session_state.get('jhs_grade', 'All'))
    if jhs_grade != 'All' and jhs_grade not in jhs_grades:
        jhs_grade = 'All'
        st.session_state['jhs_grade'] = jhs_grade
    else:
        st.session_state.setdefault('jhs_grade', jhs_grade)

    reporting_periods = ['Annual Subject Grade', 'Q1', 'Q2', 'Q3', 'Q4']
    jhs_period = st.session_state.get('jhs_period', reporting_periods[0])
    if jhs_period not in reporting_periods:
        jhs_period = reporting_periods[0]
        st.session_state['jhs_period'] = jhs_period
    else:
        st.session_state.setdefault('jhs_period', jhs_period)

    if jhs_period == 'Annual Subject Grade':
        jhs_base = jhs_annual_df.copy()
    else:
        jhs_base = jhs_quarter_df[jhs_quarter_df['period_label'] == jhs_period].copy()
    jhs_view = jhs_base[jhs_base['school_year'] == jhs_year].copy()
    if jhs_grade != 'All':
        jhs_view = jhs_view[jhs_view['grade_level'].astype(str) == jhs_grade]

    jhs_tabs = st.tabs([
        'Overview & Trends', 'Subject Deep Dive', 'Correlations', 'Predictive Outlook'
    ])

    with jhs_tabs[0]:
        render_location_strip(
            'JHS', 'Overview & Trends',
            'School Year / Grade / Period → Overview → Extremes → Distributions'
        )
        filter_cols = st.columns(3)
        with filter_cols[0]:
            jhs_year = st.selectbox('School Year', jhs_years, key='jhs_year')
        with filter_cols[1]:
            jhs_grade = st.selectbox('Grade Level', ['All'] + jhs_grades, key='jhs_grade')
        with filter_cols[2]:
            jhs_period = st.selectbox(
                'Reporting Period', reporting_periods, key='jhs_period'
            )

        st.markdown("""
            <h2 title="A high-level snapshot of the JHS population and academic performance.
            • Mean Grade: Mean of the selected subject-grade records.
            • Passing Rate: Percentage of selected grades ≥75.">
            Performance Overview - JHS
            </h2>
        """, unsafe_allow_html=True)
        st.caption("Summarizes the selected JHS school year and reporting period by grade level using average grade and passing rate.")
        metrics = engine.get_jhs_overview_metrics(jhs_view)
        if metrics:
            k1, k2 = st.columns(2)
            k1.metric('Total Students Enrolled', metrics['Total Students'])
            k2.metric('Subjects', metrics['Subjects'])
            st.write('### Performance by Grade Level')
            st.caption("Compare average achievement and passing rates across Grade 7 through Grade 10.")
            grade_cols = st.columns(len(metrics['Grade Levels']))
            for index, (grade, grade_metrics) in enumerate(metrics['Grade Levels'].items()):
                with grade_cols[index]:
                    grade_label = f'G{str(grade).zfill(2)}'
                    st.metric(f'{grade_label} Average', f"{grade_metrics['Avg']:.2f}")
                    st.metric(f'{grade_label} Passing', f"{grade_metrics['PassRate']:.1f}%")
        else:
            st.info('No completed numeric grades are available for this selection.')

        st.divider()
        st.subheader("Subject Performance Extremes")
        st.markdown(
            "Ranks the three lowest- and highest-mean subjects within each grade level for the selected view."
        )
        hard, easy = engine.plot_jhs_subject_extremes_split(jhs_view, school_year=jhs_year, rank_count=3)
        if hard is not None and easy is not None:
            c1, c2 = st.columns(2)
            with c1:
                hard.update_layout(dragmode=False)
                st.plotly_chart(hard, width='stretch', config=chart_config_global)
            with c2:
                easy.update_layout(dragmode=False)
                st.plotly_chart(easy, width='stretch', config=chart_config_global)
        else:
            st.warning('Data insufficient to plot subject extremes for the current filters.')

        st.divider()
        jhs_trend = jhs_base.copy()
        if jhs_grade != 'All':
            jhs_trend = jhs_trend[jhs_trend['grade_level'].astype(str) == jhs_grade]
        c1, c2 = st.columns(2)
        with c1:
            st.write("### Grade Distributions Across School Years")
            st.markdown("Shows the median, spread, and outlier grades for each JHS grade level across school years.")
            fig_dist = engine.plot_jhs_grade_distribution_interactive(jhs_trend)
            fig_dist.update_layout(dragmode=False)
            st.plotly_chart(fig_dist, width='stretch', config=chart_config_global)
        with c2:
            st.write("### Grade Density Across School Years")
            st.markdown("Shows where JHS grades are concentrated and how the distribution changes across school years.")
            fig_dens = engine.plot_jhs_grade_density_interactive(jhs_trend)
            fig_dens.update_layout(dragmode=False)
            st.plotly_chart(fig_dens, width='stretch', config=chart_config_global)

    with jhs_tabs[1]:
        render_location_strip(
            'JHS', 'Subject Deep Dive',
            'Selected View → Subject KPIs → Distribution → Head-to-Head'
        )
        st.subheader('Subject-Level Analysis')
        st.markdown(
            "Inspect one JHS subject across academic years, then compare two subjects by achievement and variability."
        )
        deep_source = jhs_base.copy()
        if jhs_grade != 'All':
            deep_source = deep_source[deep_source['grade_level'].astype(str) == jhs_grade]
        subjects = sorted(deep_source['course'].dropna().unique())
        if not subjects:
            st.info('No subjects are available for this selection.')
        else:
            selected_subject = st.selectbox('Select Subject', subjects, key='jhs_subject')
            subject_rows = jhs_view[jhs_view['course'] == selected_subject]['numeric_grade'].dropna()
            s1, s2, s3 = st.columns(3)
            s1.metric('Global Subject Average', f"{subject_rows.mean():.2f}" if not subject_rows.empty else 'N/A')
            s2.metric('Global Passing Rate', f"{(subject_rows >= 75).mean() * 100:.1f}%" if not subject_rows.empty else 'N/A')
            s3.metric('Variability (SD)', f"{subject_rows.std():.2f}" if len(subject_rows) > 1 else 'N/A')
            st.divider()
            fig_deep = engine.plot_subject_deep_dive_interactive(deep_source, selected_subject)
            fig_deep.update_layout(dragmode=False)
            st.plotly_chart(
                fig_deep,
                width='stretch', config=chart_config_global
            )

            st.divider()
            st.subheader('Subject Comparison (Head-to-Head)')
            c1, c2 = st.columns(2)
            with c1:
                subject_a = st.selectbox('Select Subject 1', subjects, key='jhs_sub1')
            with c2:
                subject_b = st.selectbox(
                    'Select Subject 2', subjects,
                    index=1 if len(subjects) > 1 else 0, key='jhs_sub2'
                )
            if subject_a and subject_b:
                fig_compare = engine.plot_subject_comparison_interactive(
                    deep_source, subject_a, subject_b
                )
                fig_compare.update_layout(dragmode=False)
                st.plotly_chart(
                    fig_compare,
                    width='stretch', config=chart_config_global
                )

    with jhs_tabs[2]:
        render_location_strip(
            'JHS', 'Correlations',
            'Year / Grade / Period → View Type → Analyze → Charts / Report'
        )
        st.subheader('Subject Correlations')
        st.markdown("Examine whether JHS subject performance moves together for a selected grade, period, and school year; correlation does not establish causation.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cor_year = st.selectbox('School Year', jhs_years, key='jhs_cor_year')
        with c2:
            cor_grade = st.selectbox('Grade Level', jhs_grades, key='jhs_cor_grade')
        with c3:
            cor_period = st.selectbox(
                'Reporting Period', ['Annual Subject Grade', 'Q1', 'Q2', 'Q3', 'Q4'],
                key='jhs_cor_period'
            )
        with c4:
            view_type = st.selectbox(
                'View Type', [
                    'Correlation Heatmap (Interactive)', 'Scatter Grid (All)',
                    'Scatter Grid (Top 20%)', 'Scatter Grid (Bottom 20%)'
                ], key='jhs_correlation_view'
            )

        correlation_source = (
            jhs_annual_df if cor_period == 'Annual Subject Grade' else jhs_quarter_df
        )
        correlation_period_label = (
            'Annual' if cor_period == 'Annual Subject Grade' else cor_period
        )

        if st.button('Analyze Correlations', key='jhs_analyze_correlations'):
            if 'Correlation Heatmap' in view_type:
                fig, msg = engine.plot_correlation_heatmap_interactive(
                    correlation_source, cor_year, cor_grade, 'JHS',
                    school_level='JHS', period_label=correlation_period_label
                )
                if fig:
                    fig.update_layout(dragmode=False)
                    st.plotly_chart(fig, width='stretch', config=chart_config_global)
                else:
                    st.warning(msg)
            else:
                target_grade = cor_grade
                top_ids = bottom_ids = None
                student_report = None
                if 'Top 20%' in view_type or 'Bottom 20%' in view_type:
                    group_type = 'top' if 'Top 20%' in view_type else 'bottom'
                    _, student_report, raw, group_metrics = engine.get_jhs_subgroup_statistics(
                        correlation_source, cor_year, target_grade,
                        correlation_period_label, group_type
                    )
                    if student_report is None:
                        st.warning('No students are available for this subgroup.')
                    else:
                        c1, c2, c3 = st.columns(3)
                        c1.metric('Count', group_metrics['count'])
                        c2.metric('Average GPA', f"{group_metrics['avg_gpa']:.2f}")
                        c3.metric('Threshold', f"{group_metrics['threshold']:.2f}")
                        with st.expander('View Student Report'):
                            st.dataframe(student_report, width='stretch', hide_index=True)
                        if group_type == 'top':
                            top_ids = raw['student sis'].unique()
                        else:
                            bottom_ids = raw['student sis'].unique()
                if student_report is not None or ('Top 20%' not in view_type and 'Bottom 20%' not in view_type):
                    with st.spinner('Generating Interactive Grid...'):
                        figs, msg = engine.plot_pairwise_correlations_interactive(
                            correlation_source, cor_year, target_grade, 'JHS',
                            top_students=top_ids, bottom_students=bottom_ids,
                            school_level='JHS', period_label=correlation_period_label
                        )
                    if figs:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        for i, fig in enumerate(figs):
                            status_text.text(f'Rendering page {i + 1} of {len(figs)}...')
                            fig.update_layout(dragmode=False)
                            st.plotly_chart(fig, width='stretch', config=chart_config_global)
                            progress_bar.progress((i + 1) / len(figs))
                        progress_bar.empty()
                        status_text.empty()
                    else:
                        st.warning(msg)

    with jhs_tabs[3]:
        render_location_strip(
            'JHS', 'Predictive Outlook',
            'Model Validation → Grade / Quarter → Risk Summary → Details'
        )
        st.subheader('Predictive Outlook — Next Quarter Performance')
        st.markdown(
            'Uses a **Random Forest** ensemble trained on historical JHS quarter '
            'grades to forecast subject-level performance for the selected quarter. '
            f'The grade threshold used to define historical at-risk records is **{engine.AT_RISK_THRESHOLD}**. '
            'The colour status comes from a separate at-risk classifier using the full historical feature pattern, '
            'so a predicted grade above 80 can still receive a moderate administrative-priority status. '
            'Risk scores are prioritization indicators, not calibrated probabilities.'
        )
        with st.spinner('Loading JHS predictive models... (first load only)'):
            jhs_reg, jhs_cls, jhs_metrics, jhs_fi = get_jhs_models(jhs_quarter_df)
        if jhs_reg is None:
            st.warning(f"JHS predictive model unavailable: {jhs_metrics.get('error', 'Unknown error')}")
        else:
            with st.expander(
                '📊 Model Validation (Temporal Split — '
                'Train: all years except latest | Test: latest year)'
            ):
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric('MAE (grade pts)', jhs_metrics.get('MAE', 'N/A'))
                m2.metric('R²', jhs_metrics.get('R2', 'N/A'))
                m3.metric('AUC-ROC (At-Risk)', jhs_metrics.get('AUC', 'N/A'))
                m4.metric('Train Samples', jhs_metrics.get('train_n', 'N/A'))
                m5.metric('Test Samples', jhs_metrics.get('test_n', 'N/A'))
                if not jhs_fi.empty:
                    fi_fig = engine.plot_feature_importance(
                        jhs_fi, 'What Historical Factors Drive JHS Quarter Predictions?'
                    )
                    fi_fig.update_layout(dragmode=False)
                    st.plotly_chart(fi_fig, width='stretch', config=chart_config_global)

            st.divider()
            st.markdown('##### Configure Prediction')
            st.caption('Choose a grade level and target quarter, then generate the cohort-level subject outlook.')
            pc1, pc2 = st.columns(2)
            with pc1:
                target_grade = st.selectbox(
                    'Grade Level', sorted(jhs_quarter_df['grade_level'].dropna().unique()),
                    key='jhs_pred_grade'
                )
            with pc2:
                target_quarter = st.selectbox(
                    'Target Quarter', ['Q1', 'Q2', 'Q3', 'Q4'], key='jhs_pred_quarter'
                )
            if st.button('Generate JHS Prediction', key='run_jhs_pred'):
                with st.spinner('Generating outlook...'):
                    forecast = engine.predict_jhs_cohort_outlook(
                        jhs_reg, jhs_cls, jhs_quarter_df, target_grade, target_quarter
                    )
                st.session_state['jhs_prediction'] = {
                    'filters': (target_grade, target_quarter), 'data': forecast
                }
                st.rerun()

            saved_prediction = st.session_state.get('jhs_prediction')
            active_filters = (target_grade, target_quarter)
            if saved_prediction and saved_prediction.get('filters') == active_filters:
                forecast = saved_prediction['data']
                if forecast.empty:
                    st.warning('No JHS forecast is available for the selected filters.')
                else:
                    counts = forecast['risk_label'].value_counts()
                    c1, c2, c3 = st.columns(3)
                    c1.metric('🔴 High Risk Subjects', counts.get('🔴 High Risk', 0))
                    c2.metric('🟡 Moderate Risk Subjects', counts.get('🟡 Moderate', 0))
                    c3.metric('🟢 On Track Subjects', counts.get('🟢 On Track', 0))

                    # Aggregate individual predictions by subject so the JHS
                    # outlook uses the same cohort-level visual language as SHS.
                    chart_forecast = (
                        forecast.groupby('course', as_index=False)
                        .agg(
                            predicted_mean=('predicted_grade', 'mean'),
                            prior_mean=('peer_mean', 'mean'),
                            risk_probability=('risk_probability', 'mean')
                        )
                    )
                    chart_forecast['risk_label'] = chart_forecast['risk_probability'].apply(
                        lambda p: '🔴 High Risk' if p >= 0.6
                        else ('🟡 Moderate' if p >= 0.35 else '🟢 On Track')
                    )
                    fig_jhs_pred = engine.plot_macro_prediction_chart(
                        chart_forecast, 'JHS', f'{target_grade} {target_quarter}',
                        baseline_label='Peer Mean',
                        threshold_annotation_outside=True
                    )
                    fig_jhs_pred.update_layout(
                        title=f'Predicted JHS Subject Performance — Grade {target_grade} {target_quarter}',
                        dragmode=False
                    )
                    st.plotly_chart(fig_jhs_pred, width='stretch', config=chart_config_global)

                    with st.expander('View Full Prediction Table'):
                        tbl = forecast.rename(columns={
                            'course': 'Subject', 'predicted_grade': 'Predicted Grade',
                            'peer_mean': 'Peer Mean', 'risk_probability': 'Risk Score',
                            'risk_label': 'Status', 'target_period': 'Target Quarter'
                        }).copy()
                        tbl['Risk Score'] = tbl['Risk Score'].apply(lambda p: f'{p * 100:.1f}%')
                        st.dataframe(tbl, width='stretch', hide_index=True)

elif page == "Student Profile":
    df = all_df
    st.title("Individual Student Performance Profile")
    st.caption("Unified JHS and SHS view for an individual student, including achievement history, peer comparisons, predictions, and transcript records.")
    render_location_strip(
        'Student Profile', 'Individual Review',
        'Search → KPIs → Visualizations → Peer Comparison → Predictions → Details'
    )

    # Deduplicated student list used to resolve SIS ID from selected name.
    student_data_unique = (
        df[['student name', 'student sis']]
        .dropna(subset=['student name', 'student sis'])
        .drop_duplicates()
        .reset_index(drop=True)
    )

    search_query = st.text_input("Search Student by Name or ID:", "")

    selected_student_sis = None

    if len(search_query) >= 3:
        query_lower = search_query.strip().lower()

        filtered_students = student_data_unique[
            student_data_unique['student name'].astype(str).str.contains(
                query_lower, case=False, na=False, regex=False
            ) |
            student_data_unique['student sis'].astype(str).str.contains(
                query_lower, case=False, na=False, regex=False
            )
        ].sort_values('student name')

        if not filtered_students.empty:
            st.markdown("##### Select Matching Student:")
            st.caption("Select the intended student from the matching name or SIS ID results.")

            # Label is set to "_" (hidden) because the text_input above serves
            # as the visible label.  The selectbox is needed to handle multiple
            # name matches from the same search string.
            selected_name = st.selectbox(
                "_",
                filtered_students['student name'].tolist(),
                key='result_select',
                index=0
            )

            # Where two students share a name, SIS ID is used as the
            # tiebreaker; .iloc[0] takes the first match.
            selected_student_sis = filtered_students[
                filtered_students['student name'] == selected_name
            ].iloc[0]['student sis']

        else:
            st.warning("No students found matching your query.")

    elif len(search_query) > 0:
        st.info("Please type at least 3 characters to search.")

    if selected_student_sis:
        shs_kpis = engine.get_student_kpis(shs_df, selected_student_sis)
        jhs_kpis = engine.get_jhs_student_kpis(all_df, selected_student_sis)

        profile_name = (jhs_kpis or shs_kpis).get('Name', 'N/A')
        st.divider()
        st.subheader(profile_name)
        st.caption(f"Unified student profile · Student Number: {selected_student_sis}")

        if jhs_kpis:
            jhs_comparison = engine.get_jhs_subject_performance_vs_peer(
                all_df, selected_student_sis, period_label='Annual'
            )
            st.header('JHS')
            st.caption("JHS achievement summary based on the student’s quarterly and completed annual subject records.")
            j1, j2, j3, j4 = st.columns(4)
            j1.metric('Student Number', jhs_kpis.get('ID', selected_student_sis))
            j2.metric('JHS Grade', jhs_kpis.get('Grade', 'N/A'))
            j3.metric('Latest Annual GPA', jhs_kpis.get('Latest Annual GPA', 'N/A'))
            j4.metric('JHS GPA', jhs_kpis.get('JHS GPA', 'N/A'))
            j1, j2, j3, j4 = st.columns(4)
            j1.metric('Subjects Taken', jhs_kpis.get('Total Subjects Taken', 'N/A'))
            j2.metric('Highest Grade', jhs_kpis.get('Highest Grade', 'N/A'))
            j3.metric('Lowest Grade', jhs_kpis.get('Lowest Grade', 'N/A'))
            j4.metric(
                'Class Standing',
                engine.calculate_jhs_class_standing(all_df, selected_student_sis),
                help='Percentile rank within the student’s latest JHS grade-level cohort.',
            )

            st.divider()
            st.subheader('JHS Performance Visualization')
            st.caption("View the student’s quarterly trajectory and annual subject performance relative to peers.")
            c1, c2 = st.columns(2)
            with c1:
                st.write('#### Quarterly Academic Trajectory')
                st.caption('Tracks the student’s latest JHS grade level during the most recent school year.')
                st.plotly_chart(
                    engine.plot_jhs_growth_curve(all_df, selected_student_sis),
                    width='stretch', config=chart_config_global
                )
            with c2:
                st.write('#### Annual Subject Radar (Individual vs Peers)')
                st.caption('Compares completed annual subject grades with the corresponding peer averages.')
                st.plotly_chart(
                    engine.plot_spider_graph(jhs_comparison),
                    width='stretch', config=chart_config_global
                )

            st.subheader('JHS Subject-Level Peer Comparison')
            st.caption('Shows the subject-by-subject difference between the student’s annual grade and the peer mean.')
            st.plotly_chart(
                engine.plot_subject_comparison_dumbbell(jhs_comparison),
                width='stretch', config=chart_config_global
            )

            st.subheader('JHS Predictive Grade Outlook')
            st.caption(
                'Separate JHS RF model. The peer mean is a reference value; the '
                'predicted grade is the model output. The grade threshold is '
                f'{engine.AT_RISK_THRESHOLD}; risk status is a separate model '
                'score, so a grade above the threshold can still be flagged for '
                'moderate administrative review.'
            )
            with st.spinner('Running JHS student-level predictions...'):
                jhs_reg, jhs_cls, jhs_metrics, jhs_fi = get_jhs_models(jhs_quarter_df)
                jhs_prediction = engine.predict_jhs_outlook(
                    jhs_reg, jhs_cls, jhs_quarter_df, selected_student_sis
                )
            if jhs_reg is None:
                st.warning(f"JHS predictive model unavailable: {jhs_metrics.get('error', 'Unknown error')}")
            elif jhs_prediction.empty:
                st.info('No next-quarter JHS forecast is available; the student may have completed Q4.')
            else:
                counts = jhs_prediction['risk_label'].value_counts()
                c1, c2, c3 = st.columns(3)
                c1.metric('🔴 High Risk', counts.get('🔴 High Risk', 0))
                c2.metric('🟡 Moderate', counts.get('🟡 Moderate', 0))
                c3.metric('🟢 On Track', counts.get('🟢 On Track', 0))
                st.plotly_chart(
                    engine.plot_micro_prediction_chart(
                        jhs_prediction, f"JHS Forecast: {jhs_kpis.get('Name', 'Student')}"
                    ), width='stretch', config=chart_config_global
                )
                st.dataframe(
                    jhs_prediction.rename(columns={
                        'course': 'Subject', 'predicted_grade': 'Predicted Grade',
                        'peer_mean': 'Peer Mean', 'risk_probability': 'Risk Score',
                        'risk_label': 'Risk Status', 'target_period': 'Target Quarter'
                    }), width='stretch', hide_index=True
                )

            with st.expander('View JHS Quarterly Transcript'):
                st.caption('Lists quarterly grades and the completed annual subject grade; incomplete annual records remain blank.')
                jhs_transcript = jhs_annual_df[
                    jhs_annual_df['student sis'] == selected_student_sis
                ][[
                    'grade_level', 'school_year', 'course', 'Q1', 'Q2', 'Q3', 'Q4',
                    'numeric_grade', 'quarter_count'
                ]].rename(columns={
                    'grade_level': 'Grade', 'school_year': 'Academic Year',
                    'course': 'Subject', 'numeric_grade': 'Annual Subject Grade',
                    'quarter_count': 'Completeness'
                }).sort_values(['Academic Year', 'Grade', 'Subject'], ascending=[False, False, True])
                jhs_transcript['Completeness'] = jhs_transcript['Completeness'].apply(lambda n: f'{int(n)}/4')
                st.dataframe(jhs_transcript, width='stretch', hide_index=True)

        if not shs_kpis:
            st.info('This student has JHS records only. The SHS sections are unavailable for this profile.')

    if selected_student_sis and shs_kpis:
        kpis = shs_kpis
        comparison_df = engine.get_subject_performance_vs_peer(shs_df, selected_student_sis)

        if jhs_kpis:
            st.divider()
            st.divider()
        st.header('SHS')
        st.caption("SHS achievement summary based on the student’s enrolled subjects, academic terms, and strand history.")

        col_id, col_strand, col_section, col_gpa = st.columns(4)
        col_id.metric("Student Number", kpis.get('ID', 'N/A'))
        col_strand.metric("Most Recent Strand", f"{kpis.get('Strand', 'N/A')} ({kpis.get('Latest School Year', 'N/A')})")
        col_section.metric("Most Recent Section", kpis.get('Section', 'N/A'))
        col_gpa.metric("Cumulative GPA", kpis.get('Cumulative GPA', 'N/A'))

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Subjects Taken", kpis.get('Total Subjects Taken', 'N/A'))
        c2.metric("Highest Grade", kpis.get('Highest Grade', 'N/A'))
        c3.metric("Lowest Grade", kpis.get('Lowest Grade', 'N/A'))

        with c4:
            class_standing = engine.calculate_class_standing(shs_df, selected_student_sis)
            st.metric(
                "Class Standing",
                class_standing,
                help='Percentile rank within the student’s latest SHS strand/grade cohort.',
            )

        st.divider()

        # Growth Curve and Spider Chart displayed side-by-side.
        st.subheader("Performance Visualization")
        st.caption("View the student’s term-level trajectory and subject profile relative to SHS peers.")
        col_growth, col_spider = st.columns(2)

        with col_growth:
            st.write("#### Academic Trajectory (Term Average)")
            st.caption("Tracks the student’s average grade across SHS semesters.")
            fig_growth = engine.plot_growth_curve(shs_df, selected_student_sis)
            fig_growth.update_layout(dragmode=False)
            st.plotly_chart(fig_growth, width='stretch', config=chart_config_global)

        with col_spider:
            st.write("#### Performance Radar (Individual vs Peers)")
            st.caption("Compares the student’s subject grades with peer averages.")
            fig_spider = engine.plot_spider_graph(comparison_df)
            fig_spider.update_layout(dragmode=False)
            st.plotly_chart(fig_spider, width='stretch', config=chart_config_global)

        st.divider()

        # Dumbbell Plot: subject-by-subject gap between student and peer average.
        st.subheader("Subject-Level Peer Comparison")
        st.caption("Highlights where the student is performing above or below the SHS peer mean by subject.")
        fig_dumb = engine.plot_subject_comparison_dumbbell(comparison_df)
        fig_dumb.update_layout(dragmode=False)
        st.plotly_chart(fig_dumb, width='stretch', config=chart_config_global)

        st.divider()

        # ── Predictive Grade Outlook (current semester) ───────────────────────
        st.subheader("Predictive Grade Outlook")
        st.markdown(
            "Predicts this student's subject-level performance based on their "
            "academic trajectory, cumulative GWA trend, and subject difficulty baselines. "
            "Predictions reflect the student's **most recently enrolled semester**. "
            f"The grade threshold used to define historical at-risk records is **{engine.AT_RISK_THRESHOLD}**. "
            "The colour status comes from a separate at-risk classifier using the full feature pattern, "
            "so a predicted grade above 80 can still receive a moderate administrative-priority status. "
            "Risk scores are prioritization indicators, not calibrated probabilities."
        )

        with st.spinner("Running student-level predictions..."):
            micro_reg, micro_cls, micro_metrics, micro_fi = get_micro_models(shs_df)

        if micro_reg is None:
            err = micro_metrics.get('error', 'Unknown error')
            st.warning(f"Student-level predictive model unavailable: {err}.")
        else:
            with st.spinner("Generating student outlook..."):
                pred_student_df = engine.predict_student_outlook(
                    micro_reg, micro_cls, shs_df, selected_student_sis
                )

            if pred_student_df.empty:
                st.info(
                    "Not enough historical transcript data for this student "
                    "to generate reliable predictions."
                )
            else:
                risk_counts_s = pred_student_df['risk_label'].value_counts()
                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("🔴 High Risk",  risk_counts_s.get('🔴 High Risk', 0))
                sc2.metric("🟡 Moderate",   risk_counts_s.get('🟡 Moderate',  0))
                sc3.metric("🟢 On Track",   risk_counts_s.get('🟢 On Track',  0))

                fig_micro_pred = engine.plot_micro_prediction_chart(
                    pred_student_df, kpis.get('Name', 'Student')
                )
                fig_micro_pred.update_layout(dragmode=False)
                st.plotly_chart(fig_micro_pred, width='stretch',
                                config=chart_config_global)

                with st.expander("📊 Model Validation & Feature Importance"):
                    mm1, mm2, mm3 = st.columns(3)
                    mm1.metric(
                        "MAE (grade pts)", micro_metrics.get('MAE', 'N/A'),
                        help="Average prediction error in grade points on the test set."
                    )
                    mm2.metric(
                        "R²", micro_metrics.get('R2', 'N/A'),
                        help="Variance explained by the model."
                    )
                    mm3.metric(
                        "AUC-ROC", micro_metrics.get('AUC', 'N/A'),
                        help="At-risk classification performance."
                    )
                    if not micro_fi.empty:
                        fi_fig_m = engine.plot_feature_importance(
                            micro_fi,
                            title="What Predicts Individual Student Grade Outcomes?"
                        )
                        fi_fig_m.update_layout(dragmode=False)
                        st.plotly_chart(fi_fig_m, width='stretch',
                                        config=chart_config_global)

        st.divider()

        # ── Future Term Forecast (untaken curriculum subjects) ────────────────
        st.subheader("Future Term Forecast")
        st.markdown(
            "Identifies subjects in the student's strand curriculum that have not yet "
            "been taken and generates predicted grades using the student's current "
            "cumulative GWA as the performance baseline. This is a planning view, "
            "not a record of completed grades."
        )

        if micro_reg is None or micro_cls is None:
            future_preds = pd.DataFrame()
            st.warning(
                "Future forecasting is unavailable because the student-level "
                "models did not load. This is different from having no upcoming subjects."
            )
        else:
            with st.spinner("Generating future forecast..."):
                future_preds = engine.predict_future_performance(
                    micro_reg, micro_cls, shs_df, selected_student_sis
                )

        if not future_preds.empty:
            # plot_micro_prediction_chart suppresses the actual-grade trace
            # when numeric_grade is NaN (future subjects), making the
            # forward-looking nature of this section explicit.
            fig_future = engine.plot_micro_prediction_chart(
                future_preds,
                f"Future Forecast: {kpis.get('Name', 'Student')}"
            )
            fig_future.update_layout(dragmode=False)
            st.plotly_chart(fig_future, width='stretch', config=chart_config_global)

            with st.expander("View Forecast Details"):
                st.dataframe(
                    future_preds[['full_term', 'course', 'predicted_grade', 'risk_label']].rename(
                        columns={
                            'full_term': 'Term', 'course': 'Subject',
                            'predicted_grade': 'Predicted Grade',
                            'risk_label': 'Risk Status'
                        }
                    ),
                    width='stretch', hide_index=True
                )
        elif micro_reg is not None and micro_cls is not None:
            st.info("No upcoming subjects identified for this student's current track or they have completed all levels.")

        st.divider()

        # ── Full Academic Transcript ──────────────────────────────────────────
        # Keep the raw, term-level breakdown available without making it part
        # of the default profile view.
        with st.expander("View Full Academic Transcript"):
            st.caption("Lists the student’s SHS subject grades by academic term, strand, and section.")
            transcript_cols = ['course', 'numeric_grade', 'full_term', 'strand', 'section_name']
            transcript_df = shs_df[
                shs_df['student sis'] == selected_student_sis
            ][transcript_cols].sort_values('full_term', ascending=False)

            st.dataframe(
                transcript_df.rename(columns={
                    'course': 'Subject', 'numeric_grade': 'Final Grade',
                    'full_term': 'Academic Term', 'strand': 'Strand', 'section_name': 'Section'
                }),
                width='stretch',
                hide_index=True
            )

elif page == TEST_PAGE:
    st.title("TEST")
    st.warning(
        "Evaluation-only page. This analysis is experimental, uses a small linked cohort, "
        "and is scheduled for removal after review. It must not be used for student decisions."
    )
    st.caption(
        "Approximate cohort mapping: 2024-2025 = Grade 11 and 2025-2026 = Grade 12. "
        "Irregular section enrollment is retained in the data and is not separately identified."
    )
    with st.spinner("Loading the evaluation cohort..."):
        test_data = get_grade12_test_data()
    with st.spinner("Building letter-grade classification..."):
        letter_granular_data = grade12_test.build_granular_study_data('.')
        letter_data = grade12_test.build_letter_study_data(letter_granular_data)
        letter_metrics, letter_predictions, letter_error = grade12_test.evaluate_letter_models(letter_data)

    letter_admin, letter_technical = st.tabs([
        'Administrator View', 'Technical Analysis'
    ])
    with letter_admin:
        st.subheader('Predicted Grade 12 progression')
        st.caption(
            'This view reports only the probability-aggregated letter-grade classification. '
            'It does not display numeric-grade model metrics.'
        )
        if letter_error:
            st.error(letter_error)
        else:
            letter_counts = letter_predictions['Assessment'].value_counts().rename_axis('Assessment').reset_index(name='Students')
            letter_fig = px.bar(
                letter_counts, x='Assessment', y='Students', color='Assessment',
                title='Predicted progression assessment frequency', text='Students',
                color_discrete_map=TEST_STATUS_COLORS,
            )
            letter_fig.update_layout(dragmode=False, yaxis_title='Students')
            st.plotly_chart(letter_fig, width='stretch', config=chart_config_global)
            classifier_names = [
                'Ordinal Logistic Regression',
                'Regularized Multinomial Logistic',
                'Random Forest Classifier',
                'Extra Trees Classifier',
            ]
            clustered_rows = []
            for model_name in classifier_names:
                column = f'{model_name} - Predicted Status'
                if column in letter_predictions:
                    counts = letter_predictions[column].value_counts()
                    for status in ('Improved', 'Maintained', 'Declined'):
                        clustered_rows.append({
                            'Model': model_name,
                            'Outcome': status,
                            'Students': int(counts.get(status, 0)),
                        })
            consensus_counts = letter_predictions['Predicted Status'].value_counts()
            for status in ('Improved', 'Maintained', 'Declined'):
                clustered_rows.append({
                    'Model': 'Probability Consensus',
                    'Outcome': status,
                    'Students': int(consensus_counts.get(status, 0)),
                })
            clustered_fig = px.bar(
                pd.DataFrame(clustered_rows), x='Outcome', y='Students', color='Model',
                barmode='group', title='Model classifications versus probability consensus',
                text='Students', color_discrete_map=TEST_MODEL_COLORS,
            )
            clustered_fig.update_layout(dragmode=False, yaxis_title='Students')
            st.plotly_chart(clustered_fig, width='stretch', config=chart_config_global)
            st.caption(
                'Assessment is based on the average class probabilities from the letter-grade '
                'classifiers. Results below 60% confidence are Uncertain.'
            )
            selected_letter_id = st.selectbox(
                'Select student', sorted(letter_predictions['student sis'].astype(str)),
                key='letter_admin_student'
            )
            selected_letter_row = letter_predictions[
                letter_predictions['student sis'].eq(selected_letter_id)
            ].iloc[0]
            c1, c2 = st.columns(2)
            with c1:
                st.metric('Assessment', selected_letter_row['Assessment'])
                st.metric('Confidence', f"{selected_letter_row['Confidence']:.1%}")
            with c2:
                st.metric('P(Improved)', f"{selected_letter_row['Consensus P(Improved)']:.1%}")
                st.metric('P(Declined)', f"{selected_letter_row['Consensus P(Declined)']:.1%}")
            st.dataframe(pd.DataFrame({
                'Outcome': ['Improved', 'Maintained', 'Declined'],
                'Consensus probability': [
                    selected_letter_row['Consensus P(Improved)'],
                    selected_letter_row['Consensus P(Maintained)'],
                    selected_letter_row['Consensus P(Declined)'],
                ],
            }).style.format({'Consensus probability': '{:.1%}'}), width='stretch', hide_index=True)
    with letter_technical:
        st.subheader('Technical analysis')
        if letter_error:
            st.error(letter_error)
        else:
            model_feature_inventory = grade12_test.letter_feature_inventory(letter_data)
            feature_group_counts = (
                model_feature_inventory.groupby('Feature group', as_index=False)
                .size().rename(columns={'size': 'Count'})
            )
            st.subheader('Active classifier feature inventory')
            st.caption(
                f"The active classifiers use {len(model_feature_inventory)} model features (predictors). "
                'The table below is generated from the exact columns passed to model training, not from a separate summary.'
            )
            st.dataframe(feature_group_counts, width='stretch', hide_index=True)
            st.dataframe(model_feature_inventory, width='stretch', hide_index=True)
            st.subheader('Why ten model features?')
            st.markdown(
                '- **Four external-JHS features:** Math, Science, English, and the mean of those three ordinal grades.\n'
                '- **Six Grade 11 SHS summary features:** overall mean, Semester 1 mean, Semester 2 mean, '
                'semester change, course-grade standard deviation, and retained course count.\n\n'
                'This reduced set keeps the machine-learning comparison focused on prior external performance and '
                'compact Grade 11 context. It removes external family aliases, individual subject columns, strand '
                'and section proxies, and missingness indicators from the active model to reduce redundancy and '
                'the feature-to-sample ratio.'
            )
            st.subheader('Terminology')
            st.markdown(
                '**Model feature** is the preferred technical term for any input column supplied to a classifier. '
                '**Predictor** describes the same input by its role: it is used to predict the target outcome. '
                'Therefore, “model feature (predictor)” is used here to make both meanings explicit. Grade 12 '
                'outcomes and progression status are targets or evaluation fields, not model features.'
            )
            st.subheader('Excluded outcomes and target fields')
            st.markdown(
                '- **Excluded from predictors:** Grade 12 subject grades, Grade 12 family grades, '
                'Grade 12 overall grade, Grade 12 letter grade, and progression status.\n'
                '- **Target:** progression status derived by comparing the external baseline letter '
                'with the observed Grade 12 overall letter. A movement of at least +2 ordinal steps is '
                '`Improved`, at most -2 is `Declined`, and movement between those limits is `Maintained`.\n'
                '- **Validation:** each held-out student is predicted using training-fold median imputation; '
                'the ordinal model additionally scales, removes constant inputs, and uses fold-local PCA when needed.'
            )
            st.subheader('Provisional numeric-to-letter conversion')
            st.caption(
                'This conversion is used by the letter-grade classifier. It is shown here for technical review, '
                'not as an administrator-facing result.'
            )
            st.dataframe(grade12_test.letter_grade_scale(), width='stretch', hide_index=True)
            st.dataframe(letter_metrics.style.format({
                'Accuracy': '{:.3f}', 'Balanced Accuracy': '{:.3f}', 'Macro F1': '{:.3f}'
            }), width='stretch', hide_index=True)
            recall_metrics, confusion_metrics = grade12_test.letter_classification_diagnostics(
                letter_predictions
            )
            st.subheader('Per-class recall')
            st.dataframe(
                recall_metrics.style.format({'Recall': '{:.3f}'}),
                width='stretch', hide_index=True
            )
            st.subheader('Confusion matrix records')
            selected_diagnostic_model = st.selectbox(
                'Model for confusion matrix',
                recall_metrics['Model'].drop_duplicates().tolist(),
                key='letter_confusion_model'
            )
            selected_confusion = confusion_metrics[
                confusion_metrics['Model'].eq(selected_diagnostic_model)
            ].pivot(index='Actual', columns='Predicted', values='Students').reindex(
                index=['Declined', 'Maintained', 'Improved'],
                columns=['Declined', 'Maintained', 'Improved'],
                fill_value=0,
            )
            st.dataframe(selected_confusion, width='stretch')
            st.subheader('Per-student technical audit')
            technical_student_id = st.selectbox(
                'Select student for model comparison',
                sorted(letter_predictions['student sis'].astype(str)),
                key='letter_technical_student'
            )
            technical_row = letter_predictions[
                letter_predictions['student sis'].eq(technical_student_id)
            ].iloc[0]
            model_status_rows = [
                {
                    'Model': model_name,
                    'Predicted status': technical_row[
                        f'{model_name} - Predicted Status'
                    ],
                }
                for model_name in [
                    'Ordinal Logistic Regression',
                    'Regularized Multinomial Logistic',
                    'Random Forest Classifier',
                    'Extra Trees Classifier',
                ]
            ]
            model_status_rows.append({
                'Model': 'Probability Consensus',
                'Predicted status': technical_row['Predicted Status'],
            })
            audit_table = pd.DataFrame(model_status_rows)
            audit_table['Actual final Grade 12 letter'] = technical_row['g12_overall_letter']
            audit_table['Actual progression status'] = technical_row['Actual Status']
            st.dataframe(audit_table, width='stretch', hide_index=True)
            st.markdown(
                '**Models:** Ordinal Logistic Regression, Regularized Multinomial Logistic Regression, '
                'Random Forest Classifier, and Extra Trees Classifier. Each model predicts class probabilities; the displayed '
                'assessment averages those probabilities. Letter-grade ordinal inputs are used for '
                'the classifier features and progression target.'
            )
            st.markdown(
                '**What this analysis is:** this is a three-class machine-learning **classification** '
                'problem, not a numeric Grade 12 regression. Each classifier estimates probabilities '
                'for Improved, Maintained, and Declined. The Probability Consensus averages those '
                'class probabilities; it is a reporting ensemble, not a fifth independently trained model.\n\n'
                '**Feature construction:** external Math, Science, and English grades are converted '
                'to ordered letter codes. Retained Grade 11 subjects are converted individually; '
                'Mathematics, Science, and English/Language family means are also converted. Semester '
                'means, semester change, variability, course count, section proxies, strand indicators, '
                'and feature-missingness indicators preserve progression and record-context information. '
                'Missing numeric inputs are imputed using training-fold medians only.\n\n'
                '**Target construction:** the observed target compares the external baseline letter '
                'with the Grade 12 overall letter: at least two ordered steps higher = Improved, '
                'at most two steps lower = Declined, and intermediate movement = Maintained. '
                'This is a study classification rule, not a causal claim.\n\n'
                '**Validation:** leave-one-student-out validation trains on all other students and predicts '
                'the held-out student. Accuracy, balanced accuracy, Macro F1, per-model probabilities, '
                'consensus confidence, and Uncertain results are reported. With sparse Improved cases, '
                'uncertainty is expected; increasing the linked cohort is the main way '
                'to improve stability and interpretability.'
            )
            st.info(
                'Uncertainty is expected with the current small cohort and uneven letter-status classes. '
                'A low-confidence consensus is shown as Uncertain rather than forced into Improved, '
                'Maintained, or Declined. Feature importance or class probability does not establish causation.'
            )
    st.stop()

    descriptive = grade12_test.descriptive_summary(test_data)
    correlations = grade12_test.correlation_summary(test_data)
    metrics, predictions, evaluation_error = grade12_test.evaluate_models(test_data)

    st.header("Cohort and target summary")
    complete_cases = int(test_data[grade12_test.FEATURE_COLUMNS + ['overall_mean_g12']].dropna().shape[0])
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Linked students", int(test_data['student sis'].notna().sum()))
    s2.metric("Complete model cases", complete_cases)
    s3.metric("G11 course mean", f"{test_data['overall_mean_g11'].mean():.2f}")
    s4.metric("G12 course mean", f"{test_data['overall_mean_g12'].mean():.2f}")

    st.dataframe(
        descriptive.style.format({
            'Mean': '{:.2f}', 'SD': '{:.2f}', 'Minimum': '{:.2f}',
            'Median': '{:.2f}', 'Maximum': '{:.2f}'
        }),
        width='stretch', hide_index=True
    )

    complete_data = test_data.dropna(
        subset=['external_mean', 'overall_mean_g11', 'overall_mean_g12']
    ).copy()
    st.subheader("Performance relationships")
    st.caption(
        "Each point represents one anonymous student. These plots show association in this cohort; "
        "they are not fitted validation predictions."
    )
    relationship_left, relationship_right = st.columns(2)
    with relationship_left:
        external_fig = px.scatter(
            complete_data,
            x='external_mean', y='overall_mean_g12',
            labels={
                'external_mean': 'External Grade Mean',
                'overall_mean_g12': 'Grade 12 Overall Mean'
            },
            title='External Grades vs Grade 12'
        )
        external_fig.update_layout(dragmode=False)
        st.plotly_chart(external_fig, width='stretch', config=chart_config_global)
    with relationship_right:
        progression_fig = px.scatter(
            complete_data,
            x='overall_mean_g11', y='overall_mean_g12',
            labels={
                'overall_mean_g11': 'Grade 11 Overall Mean',
                'overall_mean_g12': 'Grade 12 Overall Mean'
            },
            title='Grade 11 vs Grade 12'
        )
        progression_fig.update_layout(dragmode=False)
        st.plotly_chart(progression_fig, width='stretch', config=chart_config_global)

    st.header("Descriptive associations")
    st.caption(
        "Pearson and Spearman associations are descriptive only. They do not establish causation "
        "and are sensitive to the small cohort size."
    )
    st.dataframe(
        correlations.style.format({'Pearson r': '{:.3f}', 'Spearman rho': '{:.3f}'}),
        width='stretch', hide_index=True
    )

    st.header("Machine-learning comparison")
    st.markdown(
        "The target is the student's Grade 12 overall mean across course-level averages. "
        "Predictors are limited to the external Math, Science, English, and external-mean features. "
        "All models are evaluated with leave-one-student-out validation; preprocessing is fitted "
        "inside each training fold."
    )
    st.info(
        "Model selection is grounded in educational data-mining practice: an interpretable mean "
        "baseline, regularized linear models, and tree ensembles are compared before considering "
        "a voting ensemble. References: Baker & Inventado (2014); Romero & Ventura (2020)."
    )

    if evaluation_error:
        st.error(evaluation_error)
    else:
        st.dataframe(
            metrics.style.format({
                'MAE': '{:.3f}', 'RMSE': '{:.3f}',
                'R2': '{:.3f}', 'Spearman rho': '{:.3f}'
            }),
            width='stretch', hide_index=True
        )

        metric_chart_data = metrics.melt(
            id_vars='Model', value_vars=['MAE', 'RMSE'],
            var_name='Metric', value_name='Grade Points'
        )
        metric_fig = px.bar(
            metric_chart_data, x='Model', y='Grade Points', color='Metric',
            barmode='group', title='Cross-validated Error by Model'
        )
        metric_fig.update_layout(xaxis_tickangle=-30, dragmode=False)
        st.plotly_chart(metric_fig, width='stretch', config=chart_config_global)

        selected_model = st.selectbox(
            "Inspect held-out predictions", metrics['Model'].tolist(),
            key='grade12_test_model'
        )
        chart_data = predictions[['Actual G12', selected_model]].copy()
        chart_data['Case'] = np.arange(1, len(chart_data) + 1)
        fig = px.scatter(
            chart_data, x='Actual G12', y=selected_model, hover_data=['Case'],
            title=f"Leave-one-student-out predictions: {selected_model}"
        )
        fig.add_shape(
            type='line', x0=chart_data['Actual G12'].min(),
            y0=chart_data['Actual G12'].min(),
            x1=chart_data['Actual G12'].max(),
            y1=chart_data['Actual G12'].max(),
            line={'dash': 'dash', 'color': 'gray'}
        )
        fig.update_layout(xaxis_title='Actual Grade 12 overall mean', yaxis_title='Predicted Grade 12 overall mean')
        st.plotly_chart(fig, width='stretch', config=chart_config_global)

        with st.expander("View anonymous held-out prediction table"):
            display_predictions = chart_data.rename(columns={
                'Case': 'Anonymous Case',
                'Actual G12': 'Actual Grade 12 Mean',
                selected_model: 'Predicted Grade 12 Mean',
            })
            st.dataframe(
                display_predictions.style.format({
                    'Actual Grade 12 Mean': '{:.2f}',
                    'Predicted Grade 12 Mean': '{:.2f}'
                }),
                width='stretch', hide_index=True
            )

        per_student_tab, methods_tab = st.tabs([
            "Per-Student Predictions", "How Results Were Obtained"
        ])
        with per_student_tab:
            st.subheader("Per-student held-out prediction")
            st.caption(
                "Each student is predicted by a model trained on the other complete students. "
                "SIS is shown because this is a restricted evaluation page; do not export this table."
            )
            student_model = st.selectbox(
                "Model", metrics['Model'].tolist(), key='grade12_student_model'
            )
            student_predictions = predictions[
                ['student sis', 'external_math', 'external_science',
                 'external_english', 'external_mean', 'overall_mean_g11',
                 'Actual G12', student_model]
            ].copy()
            student_predictions['Residual'] = (
                student_predictions[student_model] - student_predictions['Actual G12']
            )
            student_predictions['Absolute Error'] = student_predictions['Residual'].abs()
            student_predictions = student_predictions.rename(columns={
                'student sis': 'Student SIS',
                'external_math': 'External Math',
                'external_science': 'External Science',
                'external_english': 'External English',
                'external_mean': 'External Mean',
                'overall_mean_g11': 'Grade 11 Mean',
                'Actual G12': 'Actual Grade 12 Mean',
                student_model: 'Predicted Grade 12 Mean',
            })
            st.dataframe(
                student_predictions.style.format({
                    column: '{:.2f}' for column in student_predictions.columns
                    if column != 'Student SIS'
                }),
                width='stretch', hide_index=True
            )

        with methods_tab:
            st.subheader("How the results are calculated")
            st.markdown(
                "**Input and target**\n"
                "- The workbook Student Number is converted to `H` plus the numeric value and matched to FEUHS SIS.\n"
                "- 2024-2025 records are treated as approximate Grade 11; 2025-2026 records as approximate Grade 12.\n"
                "- Repeated rows for the same student, course, and year are averaged first. The student’s overall result is then the mean of course means.\n"
                "- The target is the Grade 12 overall mean. Predictors are external Math, Science, English, and their row-wise mean.\n\n"
                "**Descriptive statistics**\n"
                "- Mean: arithmetic average.\n"
                "- SD: spread of student values around the mean.\n"
                "- Median: middle ordered value, less affected by extremes.\n"
                "- Pearson `r`: linear association from -1 to 1.\n"
                "- Spearman `rho`: association of student rankings from -1 to 1.\n\n"
                "**Validation metrics**\n"
                "- MAE: average absolute prediction error in grade points; lower is better.\n"
                "- RMSE: error metric that penalizes larger misses more strongly; lower is better.\n"
                "- R²: improvement relative to predicting the mean target; 1 is perfect, 0 is no improvement over that baseline, and negative values are worse than the baseline.\n"
                "- Spearman `rho`: whether students are ranked in roughly the correct order by the predictions.\n\n"
                "**Validation procedure**\n"
                "Leave-one-student-out validation holds out one student, trains on the remaining students, "
                "and predicts the held-out student. This repeats for every complete student. The model never "
                "sees the held-out Grade 12 outcome during that student’s prediction."
            )
            st.markdown(
                "**Models**\n"
                "- Mean Baseline: predicts the training students’ average Grade 12 mean.\n"
                "- Ridge: standardized linear regression with L2 regularization, useful when predictors are correlated.\n"
                "- Elastic Net: standardized linear model combining L1 and L2 regularization.\n"
                "- Random Forest: averages many randomized decision trees.\n"
                "- Extra Trees: uses more randomized tree splits to reduce variance.\n"
                "- Histogram Gradient Boosting: builds trees sequentially to correct earlier errors.\n"
                "- Voting Ensemble: averages Ridge, Random Forest, and Extra Trees predictions."
            )

    st.header("Granular feature and target study")
    st.caption(
        "This section keeps one row per student, expands Grade 11 into subject-level features, "
        "and evaluates Grade 12 overall and subject-level targets."
    )
    with st.spinner("Building granular features and subject targets..."):
        granular_data = grade12_test.build_granular_study_data('.')
        granular_metrics, granular_predictions, subject_metrics, granular_error = (
            grade12_test.evaluate_granular_models(granular_data)
        )
        letter_data = grade12_test.build_letter_study_data(granular_data)
        letter_metrics, letter_predictions, letter_error = grade12_test.evaluate_letter_models(letter_data)

    st.header("Letter-grade progression classification")
    st.caption(
        "External and Grade 11 grades are converted to ordered letter-grade codes. "
        "Each classifier predicts Improved, Maintained, or Declined; the consensus averages "
        "their class probabilities."
    )
    letter_admin_tab, letter_metrics_tab = st.tabs([
        'Administrator View', 'Technical ML Metrics'
    ])
    if letter_error:
        st.warning(letter_error)
    else:
        with letter_metrics_tab:
            st.subheader('Letter-classification model comparison')
            st.dataframe(
                letter_metrics.style.format({
                    'Accuracy': '{:.3f}', 'Balanced Accuracy': '{:.3f}', 'Macro F1': '{:.3f}'
                }),
                width='stretch', hide_index=True
            )
        with letter_admin_tab:
            st.subheader('Predicted progression by letter-grade models')
            assessment_counts = letter_predictions['Assessment'].value_counts().rename_axis('Assessment').reset_index(name='Students')
            assessment_fig = px.bar(
                assessment_counts, x='Assessment', y='Students', color='Assessment',
                title='Frequency of predicted progression assessments', text='Students'
            )
            assessment_fig.update_layout(dragmode=False, yaxis_title='Students')
            st.plotly_chart(assessment_fig, width='stretch', config=chart_config_global)
        letter_student_ids = sorted(letter_predictions['student sis'].astype(str))
        selected_letter_student = st.selectbox(
            "Inspect letter-grade progression prediction",
            letter_student_ids, key='letter_student_prediction'
        )
        letter_row = letter_predictions[
            letter_predictions['student sis'].eq(selected_letter_student)
        ].iloc[0]
        letter_left, letter_right = st.columns(2)
        with letter_left:
            st.metric('Predicted assessment', letter_row['Assessment'])
            st.metric('Confidence', f"{letter_row['Confidence']:.1%}")
        with letter_right:
            st.metric('P(Improved)', f"{letter_row['Consensus P(Improved)']:.1%}")
            st.metric('P(Declined)', f"{letter_row['Consensus P(Declined)']:.1%}")
        st.caption(
            f"Model confidence: {letter_row['Confidence']:.1%}. "
            "A confidence below 60% is shown as Uncertain."
        )

    feature_groups = pd.DataFrame([
        {'Feature group': 'External predictors', 'Count': len([c for c in granular_data if c.startswith('external_')])},
        {'Feature group': 'Grade 11 subject features', 'Count': len([c for c in granular_data if c.startswith('feature_g11_subject_')])},
        {'Feature group': 'Grade 11 family features', 'Count': len([c for c in granular_data if c.startswith('feature_g11_family_')])},
        {'Feature group': 'Progression and variability', 'Count': len([c for c in granular_data if c.startswith('feature_g11_') and ('semester' in c or any(token in c for token in ('_sd', '_min', '_max', '_range', 'course_count')))] )},
        {'Feature group': 'Strand and section proxies', 'Count': len([c for c in granular_data if c.startswith('feature_g11_') and ('strand_' in c or 'section_' in c)])},
        {'Feature group': 'Grade 12 subject targets', 'Count': len([c for c in granular_data if c.startswith('target_g12_subject_')])},
    ])
    st.dataframe(feature_groups, width='stretch', hide_index=True)

    with st.expander("Feature construction details"):
        st.markdown(
            "- **Grade 11 subject features:** one column per retained Grade 11 course; missing courses remain missing and are imputed inside each validation fold.\n"
            "- **Subject-family alignment:** retained course names are mapped only to Science, Mathematics, or English/Language. Grade 11 family means provide aligned prior evidence for Grade 12 subjects such as Earth Science or Physics.\n"
            "- **Variability:** Grade 11 standard deviation, minimum, maximum, range, and course count.\n"
            "- **Semester progression:** Grade 11 Semester 1 mean, Semester 2 mean, and S2 minus S1.\n"
            "- **Strand:** modal strand from recognized section patterns, with unknown retained when parsing is unavailable.\n"
            "- **Section context:** number of distinct Grade 11 sections and section-linked rows. These are proxies only; they do not prove regular or irregular enrollment.\n"
            "- **Targets:** one Grade 12 target column per normalized course plus the primary Grade 12 overall mean."
        )

    st.subheader("Student progression outcome")
    st.markdown(
        "The primary progression quantity is **Grade 12 overall mean minus external overall mean**. "
        "This estimates whether the student improved, maintained, or declined relative to the external baseline."
    )
    tolerance = st.number_input(
        "Maintained tolerance in grade points",
        min_value=0.0, max_value=5.0, value=1.0, step=0.5,
        help="A gain within +/- this value is labelled Maintained."
    )
    st.info(
        f"Tolerance = ±{tolerance:.1f} grade points. A predicted or actual gain "
        f"of at least +{tolerance:.1f} is **Improved**; a gain between "
        f"-{tolerance:.1f} and +{tolerance:.1f} is **Maintained**; and a gain "
        f"of at most -{tolerance:.1f} is **Declined**. Larger tolerance values "
        "create a wider Maintained band and fewer Improved/Declined labels."
    )
    progression_metrics, progression_predictions, status_metrics, progression_error = (
        grade12_test.evaluate_progression_models(granular_data, tolerance=tolerance)
    )
    if progression_error or granular_error:
        st.error(progression_error or granular_error)
    else:
        primary_model = granular_metrics.loc[
            granular_metrics['Selected Primary'], 'Model'
        ].iloc[0]
        st.info(
            f"Primary per-student model: **{primary_model}**. It was selected from the direct "
            "Grade 12 overall models using status Macro F1 first, then balanced accuracy, "
            "gain MAE, overall MAE, and RMSE."
        )
        st.caption(
            "The table below shows the four direct models individually. Consensus is their mean; "
            "spread is the highest prediction minus the lowest. A Likely status requires model "
            "agreement within the selected tolerance; otherwise the assessment is Uncertain."
        )
        st.caption("The gain-model table below is a secondary sensitivity comparison, not the source of the primary per-student predictions.")
        st.dataframe(
            progression_metrics.style.format({
                'Gain MAE': '{:.3f}', 'Gain RMSE': '{:.3f}',
                'Gain R2': '{:.3f}', 'Gain Spearman rho': '{:.3f}'
            }),
            width='stretch', hide_index=True
        )
        st.dataframe(
            status_metrics.style.format({
                'Tolerance': '{:.1f}', 'Accuracy': '{:.3f}', 'Macro F1': '{:.3f}'
            }),
            width='stretch', hide_index=True
        )
        direct_model_names = [
            name for name in ('Ridge', 'Elastic Net', 'PLS', 'Extra Trees')
            if name in granular_predictions
        ]
        student_progression = grade12_test.summarize_model_consensus(
            granular_predictions, direct_model_names, tolerance=tolerance
        )
        student_progression['Actual G12'] = granular_predictions['Actual G12']
        student_progression['Actual Gain'] = (
            student_progression['Actual G12'] - student_progression['external_mean']
        )
        student_progression['Actual Status'] = np.where(
            student_progression['Actual Gain'] >= tolerance, 'Improved',
            np.where(student_progression['Actual Gain'] <= -tolerance, 'Declined', 'Maintained')
        )
        student_progression = student_progression.rename(columns={
            'student sis': 'Student SIS',
            'external_mean': 'External Overall Mean',
            'Consensus G12': 'Consensus Predicted G12',
            'Prediction Spread': 'Model Spread',
            'Actual G12': 'Actual Grade 12 Mean',
        })
        aggregate = grade12_test.aggregate_progression_results(
            student_progression.rename(columns={
                'Student SIS': 'student sis',
                'External Overall Mean': 'external_mean',
                'Consensus Predicted G12': 'Consensus G12',
                'Model Spread': 'Prediction Spread',
            }),
            actual_g12=student_progression['Actual Grade 12 Mean'],
            tolerance=tolerance,
        )
        aggregate_tab, summary_tab, details_tab = st.tabs([
            'Aggregate Outcomes', 'Student Summary', 'Model Details'
        ])
        with aggregate_tab:
            st.subheader('Aggregate progression outcomes')
            st.caption(
                f"Complete linked cases: {aggregate['complete_cases']}. Percentages use this denominator. "
                f"Tolerance: ±{tolerance:.1f} grade points."
            )
            aggregate_left, aggregate_right = st.columns(2)
            with aggregate_left:
                st.markdown('**Model consensus assessment**')
                st.dataframe(
                    aggregate['assessment'].style.format({'Percent': '{:.1f}%'}),
                    width='stretch', hide_index=True
                )
            with aggregate_right:
                st.markdown('**Observed outcome status**')
                st.dataframe(
                    aggregate['observed'].style.format({'Percent': '{:.1f}%'}),
                    width='stretch', hide_index=True
                )
            assessment_chart = aggregate['assessment'].assign(Source='Model assessment')
            observed_chart = aggregate['observed'].assign(Source='Observed status')
            chart_frame = pd.concat([assessment_chart, observed_chart], ignore_index=True)
            chart_frame = chart_frame[chart_frame['Status'].isin([
                'Likely Improved', 'Likely Maintained', 'Likely Declined',
                'Uncertain', 'Improved', 'Maintained', 'Declined'
            ])]
            aggregate_fig = px.bar(
                chart_frame, x='Status', y='Count', color='Source', barmode='group',
                title='Modeled assessment versus observed status',
                text='Count'
            )
            aggregate_fig.update_layout(dragmode=False, yaxis_title='Students')
            st.plotly_chart(aggregate_fig, width='stretch', config=chart_config_global)
            st.metric(
                'Model agreement rate',
                f"{aggregate['agreement_rate']:.1f}%",
                help='Percentage of complete cases whose model spread is within the selected tolerance.'
            )
            st.markdown(
                "**How to read this:** the model assessment is an out-of-fold estimate based on the "
                "external baseline and four-model consensus. Observed status is calculated afterward "
                "from actual Grade 12 results. These counts are not evidence that the model caused "
                "improvement or decline, and `Uncertain` cases should not be forced into a status."
            )
            st.markdown('**Tolerance sensitivity**')
            st.dataframe(aggregate['sensitivity'], width='stretch', hide_index=True)
        with summary_tab:
            st.subheader('Administrator summary')
            summary_ids = sorted(student_progression['Student SIS'].astype(str))
            selected_summary_id = st.selectbox(
                'Select student', summary_ids, key='admin_student_summary'
            )
            summary_source = student_progression.rename(columns={
                'Student SIS': 'student sis',
                'External Overall Mean': 'external_mean',
                'Consensus Predicted G12': 'Consensus G12',
                'Model Spread': 'Prediction Spread',
            })
            summary = grade12_test.build_student_summary(
                granular_data, summary_source,
                selected_summary_id
            )
            if summary:
                left, right = st.columns(2)
                with left:
                    st.metric('External baseline', f"{summary['External baseline']:.2f}")
                    st.metric('Predicted Grade 12 mean', f"{summary['Consensus predicted G12']:.2f}")
                    st.metric('Predicted gain', f"{summary['Predicted gain']:+.2f}")
                with right:
                    st.metric('Assessment', summary['Assessment'])
                    st.metric('Model agreement spread', f"{summary['Model spread']:.2f}")
                    st.metric(
                        'Strongest family',
                        f"{summary['Strongest family']} ({summary['Strongest family grade']:.2f})"
                    )
                st.markdown(
                    f"**Strongest subject:** {summary['Strongest subject']} ({summary['Strongest subject grade']:.2f})  \n"
                    f"**Weakest subject:** {summary['Weakest subject']} ({summary['Weakest subject grade']:.2f})"
                )
                st.info(
                    'This is a concise exploratory progression signal. It does not establish causation '
                    'or replace professional review.'
                )
        with details_tab:
            st.subheader('Detailed model outputs')
            st.dataframe(
                student_progression[[
                    'Student SIS', 'External Overall Mean',
                    *direct_model_names, 'Consensus Predicted G12', 'Predicted Gain',
                    'Model Spread', 'Assessment', 'Actual Grade 12 Mean',
                    'Actual Gain', 'Actual Status'
                ]].style.format({
                    'External Overall Mean': '{:.2f}',
                    'Actual Grade 12 Mean': '{:.2f}',
                    'Consensus Predicted G12': '{:.2f}',
                    'Actual Gain': '{:.2f}',
                    'Predicted Gain': '{:.2f}',
                    'Model Spread': '{:.2f}',
                    **{name: '{:.2f}' for name in direct_model_names},
                }),
                width='stretch', hide_index=True
            )

    if granular_error:
        st.error(granular_error)
    else:
        st.subheader("Granular Grade 12 overall prediction")
        st.caption(
            "Ridge and PLS are primary low-sample comparisons; Extra Trees is a nonlinear sensitivity model. "
            "Metrics use leave-one-student-out validation."
        )
        st.dataframe(
            granular_metrics.style.format({
                'MAE': '{:.3f}', 'RMSE': '{:.3f}',
                'R2': '{:.3f}', 'Spearman rho': '{:.3f}'
            }),
            width='stretch', hide_index=True
        )
        if not subject_metrics.empty:
            st.subheader("Separate Grade 12 subject and family targets")
            st.caption(
                "These are exploratory Ridge results. Subject and family sample sizes and course availability "
                "must be reviewed before interpreting them as reliable target-specific predictors."
            )
            st.dataframe(
                subject_metrics.style.format({
                    'MAE': '{:.3f}', 'RMSE': '{:.3f}', 'R2': '{:.3f}'
                }),
                width='stretch', hide_index=True
            )

    st.subheader("Interpretation boundaries")
    st.markdown(
        "- This is an exploratory cohort study, not a validated production predictor.\n"
        "- The 2024-2025 and 2025-2026 year mapping is approximate.\n"
        "- Irregular enrollment is not separately modeled because no reliable indicator is available.\n"
        "- The all-zero external record is treated as missing.\n"
        "- Leave-one-student-out results are unstable with only 39 complete cases.\n"
        "- Predictions describe association and should not be used for high-stakes decisions."
    )
