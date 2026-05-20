"""
Always Healthy Hospital — Customer Experience Dashboard
========================================================
Dataset  : Hospital customer satisfaction survey (3,600 respondents, 15 branches)
Stack    : Streamlit · Plotly · Pandas · WordCloud · Gemini API
"""

import warnings
warnings.filterwarnings("ignore")

import os
import textwrap
from datetime import datetime

import google.generativeai as genai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

# ──────────────────────────────────────────────
# PAGE CONFIG & THEME
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CX Dashboard | Always Healthy Hospital",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Minimalist custom CSS just to adjust top padding and hide standard header
st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1, h2, h3 { color: #1E293B; }
    .st-emotion-cache-1wivap2 { color: #334155; }
</style>
""", unsafe_allow_html=True)

# Corporate Color Palette
COLOR_PRIMARY = "#0F766E"  # Dark Teal
COLOR_POS     = "#15803D"  # Green
COLOR_WARN    = "#B45309"  # Amber/Orange
COLOR_NEG     = "#B91C1C"  # Red
COLOR_NEUTRAL = "#64748B"  # Slate

TOUCHPOINTS = [
    "Registration", "Doctor Consultation", "Nurse Service",
    "Pharmacy Service", "Laboratory", "Emergency Response",
    "Billing Process", "Facility Cleanliness", "Staff Friendliness",
    "Waiting Time",
]

MONTH_LABELS = {
    "2025-01": "Jan", "2025-02": "Feb", "2025-03": "Mar",
    "2025-04": "Apr", "2025-05": "May", "2025-06": "Jun",
    "2025-07": "Jul", "2025-08": "Aug", "2025-09": "Sep",
    "2025-10": "Oct", "2025-11": "Nov", "2025-12": "Dec",
}

# ──────────────────────────────────────────────
# DATA LOADING & PREP
# ──────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data.csv", sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d/%m/%Y %H.%M")
    df["Month"]    = df["Datetime"].dt.to_period("M").astype(str)
    df["Quarter"]  = df["Datetime"].dt.to_period("Q").astype(str)
    df["City"]     = df["Branch"].str.replace("Always Healthy Hospital ", "", regex=False)
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[20, 30, 40, 50, 60, 70],
        labels=["21–30", "31–40", "41–50", "51–60", "61–65"],
    )
    df["NPS_Segment"] = pd.cut(
        df["NPS"],
        bins=[-1, 6, 8, 10],
        labels=["Detractor (0-6)", "Passive (7-8)", "Promoter (9-10)"],
    )
    return df

df_full = load_data()

def calc_nps(series: pd.Series) -> float:
    if len(series) == 0: return 0.0
    n = len(series)
    return round(((series >= 9).sum() - (series <= 6).sum()) / n * 100, 1)

def score_color(val: float, vmin=1.0, vmax=5.0) -> str:
    ratio = (val - vmin) / (vmax - vmin)
    if ratio >= .75: return COLOR_POS
    if ratio >= .40: return COLOR_WARN
    return COLOR_NEG

# ──────────────────────────────────────────────
# SIDEBAR CONTROLS
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("Filters")
    st.markdown("---")

    all_months = sorted(df_full["Month"].unique())
    month_options = [f"{MONTH_LABELS.get(m, m)} {m[:4]}" for m in all_months]
    start_idx, end_idx = st.select_slider(
        "Time Period",
        options=list(range(len(all_months))),
        value=(0, len(all_months) - 1),
        format_func=lambda i: month_options[i],
    )
    selected_months = all_months[start_idx : end_idx + 1]

    branch_select = st.multiselect(
        "Branch",
        options=sorted(df_full["City"].unique()),
        default=sorted(df_full["City"].unique())
    )

    gender_select = st.multiselect("Gender", options=["Male", "Female"], default=["Male", "Female"])
    
    tp_select = st.multiselect("Touchpoints", options=TOUCHPOINTS, default=TOUCHPOINTS)

    st.markdown("---")
    st.markdown("### AI Configuration")
    api_key_env = os.environ.get("GEMINI_API_KEY", "")
    try:
        api_key_env = api_key_env or st.secrets.get("GEMINI_API_KEY", "")
    except Exception: pass
    
    ai_enabled = False
    if api_key_env:
        ai_enabled = st.toggle("Enable AI Insights", value=True)
    else:
        st.info("Set GEMINI_API_KEY in environment/secrets to enable AI.")

# Filter Data
df = df_full[
    df_full["Month"].isin(selected_months) &
    df_full["City"].isin(branch_select if branch_select else sorted(df_full["City"].unique())) &
    df_full["Gender"].isin(gender_select if gender_select else ["Male", "Female"])
].copy()

active_touchpoints = tp_select if tp_select else TOUCHPOINTS

if df.empty:
    st.error("No data matches the selected filters.")
    st.stop()

# ──────────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────────
st.markdown(f"## Customer Experience Dashboard")
st.markdown(f"**{MONTH_LABELS.get(selected_months[0], selected_months[0])} - {MONTH_LABELS.get(selected_months[-1], selected_months[-1])} 2025** | {len(df):,} Respondents | {df['City'].nunique()} Branches")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Executive Summary", "Performance Deep Dive", "Voice of Customer & AI"])

# ═══════════════════════════════════════════════
# TAB 1 — EXECUTIVE SUMMARY
# ═══════════════════════════════════════════════
with tab1:
    # ── High-level Metrics ──
    nps_val = calc_nps(df["NPS"])
    nps_overall = calc_nps(df_full["NPS"])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Promoter Score", f"{nps_val:+.1f}", f"{nps_val - nps_overall:+.1f} vs Avg")
    c2.metric("Customer Satisfaction", f"{df['CSI'].mean():.2f}", f"{df['CSI'].mean() - df_full['CSI'].mean():+.2f} vs Avg")
    c3.metric("Customer Loyalty", f"{df['Loyalty'].mean():.2f}", f"{df['Loyalty'].mean() - df_full['Loyalty'].mean():+.2f} vs Avg")
    c4.metric("Customer Effort", f"{df['CES'].mean():.2f}", f"{-(df['CES'].mean() - df_full['CES'].mean()):+.2f} vs Avg (Inverted)")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])

    # ── Time Series Trend ──
    with col_l:
        st.markdown("#### NPS & Satisfaction Trend")
        ts_data = df.groupby("Month").agg(NPS=("NPS", calc_nps), CSI=("CSI", "mean")).reset_index()
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts_data["Month"], y=ts_data["NPS"], mode="lines+markers", name="NPS", line=dict(color=COLOR_PRIMARY, width=3), yaxis="y1"))
        fig_ts.add_trace(go.Scatter(x=ts_data["Month"], y=ts_data["CSI"], mode="lines+markers", name="CSI", line=dict(color=COLOR_NEUTRAL, width=2, dash="dash"), yaxis="y2"))
        
        fig_ts.update_layout(
            height=350, plot_bgcolor="white",
            margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="NPS Score", side="left", showgrid=True, gridcolor="#F1F5F9"),
            yaxis2=dict(title="CSI Score (1-5)", side="right", overlaying="y", range=[1, 5]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(fig_ts, use_container_width=True)

    # ── Demographics Breakdown ──
    with col_r:
        st.markdown("#### Respondent Profile")
        fig_dem = px.histogram(df, x="Age_Group", color="Gender", barmode="group", color_discrete_sequence=[COLOR_PRIMARY, COLOR_NEUTRAL])
        fig_dem.update_layout(height=350, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Count", xaxis_title="Age Group", legend=dict(title=None, orientation="h", y=-0.2))
        st.plotly_chart(fig_dem, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 2 — PERFORMANCE DEEP DIVE
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("#### Branch Performance Ranking")
    
    branch_kpi = df.groupby("City").agg(
        Respondents=("NPS", "count"),
        NPS_Score=("NPS", calc_nps),
        CSI_Mean=("CSI", "mean"),
        CLI_Mean=("Loyalty", "mean"),
        CES_Mean=("CES", "mean")
    ).reset_index().rename(columns={"City": "Branch"}).sort_values("NPS_Score", ascending=False)
    
    # Accessible styling for dataframe
    def style_nps(val):
        if val >= 50: return f"color: {COLOR_POS}; font-weight: bold"
        if val >= 0: return f"color: {COLOR_WARN}; font-weight: bold"
        return f"color: {COLOR_NEG}; font-weight: bold"

    st.dataframe(
        branch_kpi.style
        .map(style_nps, subset=["NPS_Score"])
        .format({"NPS_Score": "{:+.1f}", "CSI_Mean": "{:.2f}", "CLI_Mean": "{:.2f}", "CES_Mean": "{:.2f}"}),
        use_container_width=True, height=250
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Touchpoint Average Scores")
        tp_means = df[active_touchpoints].mean().sort_values()
        fig_tp = go.Figure(go.Bar(
            x=tp_means.values, y=tp_means.index, orientation="h",
            marker_color=[score_color(v) for v in tp_means.values],
            text=tp_means.values.round(2), textposition="outside"
        ))
        fig_tp.update_layout(height=400, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[1, 5.2], gridcolor="#F1F5F9"))
        st.plotly_chart(fig_tp, use_container_width=True)

    with c2:
        st.markdown("#### Touchpoint Correlation with NPS")
        corr_vals = df[active_touchpoints + ["NPS"]].corr()["NPS"].drop("NPS").sort_values()
        fig_corr = go.Figure(go.Bar(
            x=corr_vals.values, y=corr_vals.index, orientation="h",
            marker_color=COLOR_PRIMARY,
            text=corr_vals.values.round(3), textposition="outside"
        ))
        fig_corr.update_layout(height=400, plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(gridcolor="#F1F5F9"))
        st.plotly_chart(fig_corr, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 3 — VOICE OF CUSTOMER & AI
# ═══════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.markdown("#### Top Complaint Themes")
        complaints = df[df["Improvement_Feedback"].str.startswith("improve", na=False)]
        complaint_counts = complaints["Improvement_Feedback"].str.replace("improve ", "").str.split(", ").explode().value_counts().head(8)
        
        if not complaint_counts.empty:
            fig_comp = px.bar(y=complaint_counts.index, x=complaint_counts.values, orientation="h", color_discrete_sequence=[COLOR_NEUTRAL])
            fig_comp.update_layout(height=350, plot_bgcolor="white", yaxis_title=None, xaxis_title="Frequency", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.info("Insufficient feedback data.")

    with c2:
        st.markdown("#### AI Executive Synthesis")
        if ai_enabled:
            if st.button("Generate Insights via Gemini", type="primary"):
                with st.spinner("Analyzing data..."):
                    try:
                        summary = textwrap.dedent(f"""
                        NPS: {nps_val:+.1f} | Avg CSI: {df['CSI'].mean():.2f}
                        Top branch: {branch_kpi.iloc[0]['Branch']} (NPS {branch_kpi.iloc[0]['NPS_Score']})
                        Worst branch: {branch_kpi.iloc[-1]['Branch']} (NPS {branch_kpi.iloc[-1]['NPS_Score']})
                        Top complaint themes: {', '.join(complaint_counts.index[:3]) if not complaint_counts.empty else 'N/A'}
                        """)
                        genai.configure(api_key=api_key_env)
                        model = genai.GenerativeModel("gemini-pro")
                        response = model.generate_content(
                            f"Act as a CX Director. Based on this brief data: {summary}. Write a concise 3-bullet point executive action plan focusing strictly on operations and process improvement. No fluff, professional tone."
                        )
                        st.session_state["ai_insight_clean"] = response.text
                    except Exception as e:
                        st.error(f"API Error: {e}")
            
            if "ai_insight_clean" in st.session_state:
                st.markdown(st.session_state["ai_insight_clean"])
        else:
            st.warning("AI integration disabled. Configure API key to activate.")
