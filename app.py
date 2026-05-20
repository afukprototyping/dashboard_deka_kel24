"""
Always Healthy Hospital — Customer Experience Dashboard
========================================================
Dataset  : Hospital customer satisfaction survey (3,600 respondents, 15 branches, Jan–Dec 2025)
Author   : Afiq Dzakwan Anasti
Stack    : Streamlit · Plotly · Pandas · WordCloud · Gemini API
"""

import warnings
warnings.filterwarnings("ignore")

import io
import os
import re
import textwrap
from datetime import datetime

import google.generativeai as genai
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud, STOPWORDS

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Always Healthy Hospital — CX Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# THEME / CSS
# ──────────────────────────────────────────────
BLUE_MAIN   = "#1A8C4E"   # primary green
BLUE_LIGHT  = "#52C27A"   # light green
BLUE_PALE   = "#E8F7EE"   # pale green bg
BLUE_DARK   = "#0D5C32"   # dark forest green
TEAL        = "#00A896"   # teal-green accent
RED_ALERT   = "#E63946"
GREEN_OK    = "#27AE60"
AMBER_WARN  = "#F4A261"
GREY_BG     = "#F4FAF6"   # very light green-white
GREY_TEXT   = "#2D4A36"   # dark green-grey text

PLOTLY_COLORS = [BLUE_MAIN, TEAL, BLUE_LIGHT, "#3A86FF", AMBER_WARN,
                 "#8BC34A", "#FF6B6B", "#06D6A0", "#0D5C32", "#FFD166"]

st.markdown(f"""
<style>
/* ── Global ── */
html, body, [class*="css"] {{
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: {GREY_TEXT};
}}
.stApp {{ background: {GREY_BG}; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: {BLUE_DARK};
    color: white;
}}
[data-testid="stSidebar"] * {{ color: white !important; }}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stDateInput label {{ color: #CBD5E0 !important; }}
[data-testid="stSidebar"] .stMarkdown h2 {{
    color: {TEAL} !important;
    font-size: 1rem;
    letter-spacing: .06em;
    text-transform: uppercase;
}}

/* ── KPI Cards ── */
.kpi-card {{
    background: white;
    border-radius: 14px;
    padding: 1.4rem 1rem 1rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(30,95,173,.10);
    border-top: 4px solid {BLUE_MAIN};
    height: 100%;
}}
.kpi-card.nps  {{ border-top-color: {BLUE_MAIN}; }}
.kpi-card.csi  {{ border-top-color: {TEAL}; }}
.kpi-card.cli  {{ border-top-color: {GREEN_OK}; }}
.kpi-card.ces  {{ border-top-color: {AMBER_WARN}; }}
.kpi-value {{ font-size: 2.4rem; font-weight: 800; margin: .2rem 0; }}
.kpi-label {{ font-size: .82rem; color: #718096; text-transform: uppercase; letter-spacing: .07em; }}
.kpi-delta {{ font-size: .88rem; font-weight: 600; margin-top: .3rem; }}
.delta-pos {{ color: {GREEN_OK}; }}
.delta-neg {{ color: {RED_ALERT}; }}
.delta-neu {{ color: #A0AEC0; }}

/* ── Section header ── */
.section-header {{
    font-size: 1.15rem; font-weight: 700;
    color: {BLUE_DARK};
    border-left: 4px solid {BLUE_MAIN};
    padding-left: .75rem;
    margin: 1.4rem 0 .8rem;
}}

/* ── Insight box ── */
.insight-box {{
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    border-left: 5px solid {BLUE_MAIN};
    box-shadow: 0 2px 8px rgba(30,95,173,.08);
    margin-bottom: .8rem;
    font-size: .93rem;
    line-height: 1.65;
}}
.insight-box.warning {{ border-left-color: {RED_ALERT}; }}
.insight-box.ok      {{ border-left-color: {GREEN_OK}; }}

/* ── Tab styling ── */
button[data-baseweb="tab"] {{
    font-weight: 600;
    font-size: .88rem;
    color: {GREY_TEXT} !important;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {BLUE_MAIN} !important;
    border-bottom: 3px solid {BLUE_MAIN} !important;
}}

/* ── Scrollable table ── */
.dataframe-container {{ max-height: 420px; overflow-y: auto; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
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
# DATA LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data.csv", sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d/%m/%Y %H.%M")
    df["Month"]    = df["Datetime"].dt.to_period("M").astype(str)
    df["Quarter"]  = df["Datetime"].dt.to_period("Q").astype(str)
    df["Year"]     = df["Datetime"].dt.year
    df["City"]     = df["Branch"].str.replace("Always Healthy Hospital ", "", regex=False)
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[20, 30, 40, 50, 60, 70],
        labels=["21–30", "31–40", "41–50", "51–60", "61–65"],
    )
    df["NPS_Segment"] = pd.cut(
        df["NPS"],
        bins=[-1, 6, 8, 10],
        labels=["Detractor (0–6)", "Passive (7–8)", "Promoter (9–10)"],
    )
    return df

df_full = load_data()

# ──────────────────────────────────────────────
# KPI HELPERS
# ──────────────────────────────────────────────
def calc_nps(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    n = len(series)
    promoters  = (series >= 9).sum()
    detractors = (series <= 6).sum()
    return round((promoters - detractors) / n * 100, 1)

def nps_label(score: float) -> str:
    if score >= 70:  return "🌟 World Class"
    if score >= 50:  return "✅ Excellent"
    if score >= 30:  return "👍 Good"
    if score >= 0:   return "⚠️ Needs Attention"
    return "🚨 Critical"

def ces_label(score: float) -> str:
    """Lower CES is better (less effort required)."""
    if score <= 2.0: return "😊 Very Easy"
    if score <= 2.5: return "😐 Easy"
    if score <= 3.0: return "😶 Moderate"
    return "😓 High Effort"

def score_color(val: float, vmin=1.0, vmax=5.0) -> str:
    ratio = (val - vmin) / (vmax - vmin)
    if ratio >= .75: return GREEN_OK
    if ratio >= .50: return BLUE_MAIN
    if ratio >= .30: return AMBER_WARN
    return RED_ALERT

# ──────────────────────────────────────────────
# ════════════════  SIDEBAR  ═════════════════
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Always Healthy")
    st.markdown("**Customer Experience Dashboard**")
    st.markdown("---")

    # ── Date range ──
    st.markdown("## 📅 Time Range")
    all_months = sorted(df_full["Month"].unique())
    month_options = [f"{MONTH_LABELS.get(m, m)} {m[:4]}" for m in all_months]
    start_idx, end_idx = st.select_slider(
        "Select period",
        options=list(range(len(all_months))),
        value=(0, len(all_months) - 1),
        format_func=lambda i: month_options[i],
    )
    selected_months = all_months[start_idx : end_idx + 1]

    # ── Branch ──
    st.markdown("## 🏢 Branch / Area")
    all_branches = sorted(df_full["City"].unique())
    branch_select = st.multiselect(
        "Select branches",
        options=all_branches,
        default=all_branches,
        placeholder="All branches",
    )

    # ── Demographics ──
    st.markdown("## 👥 Demographics")
    gender_select = st.multiselect(
        "Gender",
        options=["Male", "Female"],
        default=["Male", "Female"],
    )
    age_groups = ["21–30", "31–40", "41–50", "51–60", "61–65"]
    age_select = st.multiselect(
        "Age Group",
        options=age_groups,
        default=age_groups,
    )

    # ── Touchpoint filter ──
    st.markdown("## 🔍 Touchpoint Focus")
    tp_select = st.multiselect(
        "Touchpoints to analyse",
        options=TOUCHPOINTS,
        default=TOUCHPOINTS,
    )

    # ── AI settings ──
    st.markdown("## 🤖 AI Insights")
    ai_enabled = False
    api_key_env = os.environ.get("GEMINI_API_KEY", "")
    # Also try st.secrets
    try:
        api_key_env = api_key_env or st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass
    if api_key_env:
        ai_enabled = st.toggle("Enable AI Insights", value=True)
        st.success("API key loaded ✓", icon="🔑")
    else:
        st.info("Set GEMINI_API_KEY in environment / st.secrets to enable AI insights.")
        ai_enabled = False

    st.markdown("---")
    st.markdown(f"<small style='opacity:.5'>Data: 3,600 respondents · 15 branches<br>Period: Jan–Dec 2025</small>", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FILTERED DATA
# ──────────────────────────────────────────────
df = df_full[
    df_full["Month"].isin(selected_months) &
    df_full["City"].isin(branch_select if branch_select else all_branches) &
    df_full["Gender"].isin(gender_select if gender_select else ["Male", "Female"]) &
    df_full["Age_Group"].isin(age_select if age_select else age_groups)
].copy()

active_touchpoints = tp_select if tp_select else TOUCHPOINTS

if df.empty:
    st.warning("No data matches the current filters. Please adjust your selection.")
    st.stop()

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown(f"""
<div style="background:linear-gradient(135deg,{BLUE_DARK} 0%,{BLUE_MAIN} 55%,{TEAL} 100%);
            border-radius:16px;padding:1.6rem 2rem;margin-bottom:1.4rem;color:white;">
  <h1 style="margin:0;font-size:1.8rem;font-weight:800;">🏥 Always Healthy Hospital</h1>
  <p style="margin:.4rem 0 0;opacity:.85;font-size:1rem;">
      Customer Experience Dashboard &nbsp;·&nbsp;
      {MONTH_LABELS.get(selected_months[0], selected_months[0])} –
      {MONTH_LABELS.get(selected_months[-1], selected_months[-1])} 2025 &nbsp;·&nbsp;
      {len(df):,} respondents &nbsp;·&nbsp; {df['City'].nunique()} branches
  </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "🗺️ Area Analysis",
    "👥 Demographics",
    "📈 Time Series",
    "🔍 Touchpoints",
    "🤖 AI Insights",
])

# ═══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ═══════════════════════════════════════════════
with tabs[0]:

    # ── KPI Cards ──────────────────────────────
    nps_score = calc_nps(df["NPS"])
    csi_score = round(df["CSI"].mean(), 2)
    cli_score = round(df["Loyalty"].mean(), 2)
    ces_score = round(df["CES"].mean(), 2)

    # Full-period benchmarks for delta
    nps_all = calc_nps(df_full["NPS"])
    csi_all = round(df_full["CSI"].mean(), 2)
    cli_all = round(df_full["Loyalty"].mean(), 2)
    ces_all = round(df_full["CES"].mean(), 2)

    def delta_html(val, ref, invert=False):
        d = val - ref
        if invert:
            d = -d
        if abs(d) < 0.05:
            return f'<div class="kpi-delta delta-neu">— same as avg</div>'
        cls  = "delta-pos" if d > 0 else "delta-neg"
        icon = "▲" if d > 0 else "▼"
        return f'<div class="kpi-delta {cls}">{icon} {abs(d):.1f} vs overall avg</div>'

    kpi_data = [
        ("nps",  "Net Promoter Score",         f"{nps_score:+.1f}",  nps_label(nps_score),  delta_html(nps_score, nps_all)),
        ("csi",  "Customer Satisfaction Index", f"{csi_score:.2f}/5", f"{'★'*round(csi_score)}{'☆'*(5-round(csi_score))}", delta_html(csi_score, csi_all)),
        ("cli",  "Customer Loyalty Index",      f"{cli_score:.2f}/5", f"{'★'*round(cli_score)}{'☆'*(5-round(cli_score))}", delta_html(cli_score, cli_all)),
        ("ces",  "Customer Effort Score",       f"{ces_score:.2f}/5", ces_label(ces_score),  delta_html(ces_score, ces_all, invert=True)),
    ]

    c1, c2, c3, c4 = st.columns(4)
    for col, (cls, label, val, tag, delta) in zip([c1, c2, c3, c4], kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{BLUE_MAIN if cls != 'nps' else BLUE_DARK}">{val}</div>
              <div style="font-size:.8rem;color:#718096;margin:.1rem 0">{tag}</div>
              {delta}
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── NPS Breakdown + Segment donut ──────────
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown('<div class="section-header">NPS Segment Distribution</div>', unsafe_allow_html=True)
        seg_counts = df["NPS_Segment"].value_counts().reindex(
            ["Promoter (9–10)", "Passive (7–8)", "Detractor (0–6)"]
        ).fillna(0)
        fig_donut = go.Figure(go.Pie(
            labels=seg_counts.index,
            values=seg_counts.values,
            hole=.55,
            marker_colors=[GREEN_OK, AMBER_WARN, RED_ALERT],
            textinfo="label+percent",
            textfont_size=12,
        ))
        fig_donut.add_annotation(
            text=f"<b>{nps_score:+.0f}</b><br><span style='font-size:11px'>NPS</span>",
            x=0.5, y=0.5, showarrow=False, font_size=20, align="center"
        )
        fig_donut.update_layout(
            showlegend=True, height=320,
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=.5),
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">Score Distribution (CSI · Loyalty · CES)</div>', unsafe_allow_html=True)
        fig_box = go.Figure()
        for metric, color in [("CSI", TEAL), ("Loyalty", GREEN_OK), ("CES", AMBER_WARN)]:
            fig_box.add_trace(go.Box(
                y=df[metric], name=metric,
                marker_color=color,
                boxmean=True,
                line_width=1.5,
            ))
        fig_box.update_layout(
            height=320, yaxis=dict(range=[0.5, 5.5], title="Score (1–5)"),
            margin=dict(t=10, b=10, l=40, r=10),
            showlegend=False,
            plot_bgcolor="white",
            yaxis_gridcolor="#EDF2F7",
        )
        st.plotly_chart(fig_box, use_container_width=True)

    # ── Quick Summary Stats ──────────────────────
    st.markdown('<div class="section-header">Response Quality Overview</div>', unsafe_allow_html=True)
    total       = len(df)
    promoters   = (df["NPS"] >= 9).sum()
    detractors  = (df["NPS"] <= 6).sum()
    passives    = total - promoters - detractors
    complaints  = df["Improvement_Feedback"].str.startswith("improve").sum()

    c1, c2, c3, c4, c5 = st.columns(5)
    stat_items = [
        ("Total Respondents",  f"{total:,}",          BLUE_MAIN),
        ("Promoters",          f"{promoters:,} ({promoters/total*100:.0f}%)", GREEN_OK),
        ("Passives",           f"{passives:,} ({passives/total*100:.0f}%)",   AMBER_WARN),
        ("Detractors",         f"{detractors:,} ({detractors/total*100:.0f}%)", RED_ALERT),
        ("Complaints (feedback)", f"{complaints:,} ({complaints/total*100:.0f}%)", "#7B2FBE"),
    ]
    for col, (lbl, val, clr) in zip([c1, c2, c3, c4, c5], stat_items):
        col.markdown(f"""
        <div style="background:white;border-radius:10px;padding:.9rem;text-align:center;
                    box-shadow:0 1px 6px rgba(0,0,0,.07);border-top:3px solid {clr}">
          <div style="font-size:1.3rem;font-weight:800;color:{clr}">{val}</div>
          <div style="font-size:.75rem;color:#718096;margin-top:.2rem">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # ── Export button ─────────────────────────
    st.markdown('<div class="section-header">📥 Export Filtered Data</div>', unsafe_allow_html=True)
    csv_export = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download filtered data as CSV",
        data=csv_export,
        file_name=f"AHH_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

# ═══════════════════════════════════════════════
# TAB 2 — AREA ANALYSIS
# ═══════════════════════════════════════════════
with tabs[1]:

    st.markdown('<div class="section-header">Branch Ranking — All KPIs</div>', unsafe_allow_html=True)

    # Per-branch KPI table
    branch_kpi = (
        df.groupby("City")
        .agg(
            Respondents=("NPS", "count"),
            NPS_Score=("NPS", calc_nps),
            CSI_Mean=("CSI", "mean"),
            CLI_Mean=("Loyalty", "mean"),
            CES_Mean=("CES", "mean"),
        )
        .reset_index()
        .rename(columns={"City": "Branch"})
    )
    for col in ["NPS_Score", "CSI_Mean", "CLI_Mean", "CES_Mean"]:
        branch_kpi[col] = branch_kpi[col].round(2)
    branch_kpi["Overall_Score"] = (
        (branch_kpi["NPS_Score"].clip(-100, 100) + 100) / 200 * 40 +   # NPS weight 40%
        (branch_kpi["CSI_Mean"] - 1) / 4 * 25 +                        # CSI weight 25%
        (branch_kpi["CLI_Mean"] - 1) / 4 * 20 +                        # CLI weight 20%
        (1 - (branch_kpi["CES_Mean"] - 1) / 4) * 15                    # CES weight 15% (inverted)
    ).round(3)
    branch_kpi = branch_kpi.sort_values("Overall_Score", ascending=False).reset_index(drop=True)
    branch_kpi.index += 1

    # Color-coded table
    def color_nps(v):
        if v >= 50:   return f"background-color: {GREEN_OK}22; color: {GREEN_OK}"
        if v >= 0:    return f"background-color: {AMBER_WARN}22; color: #B7791F"
        return             f"background-color: {RED_ALERT}22; color: {RED_ALERT}"
    def color_score(v, col):
        c = score_color(v)
        return f"color: {c}; font-weight: 600"

    st.dataframe(
        branch_kpi.style
            .applymap(color_nps, subset=["NPS_Score"])
            .applymap(lambda v: color_score(v, "CSI"), subset=["CSI_Mean", "CLI_Mean"])
            .format({"NPS_Score": "{:+.1f}", "CSI_Mean": "{:.2f}", "CLI_Mean": "{:.2f}",
                     "CES_Mean": "{:.2f}", "Overall_Score": "{:.3f}"}),
        use_container_width=True, height=420
    )

    # ── NPS per branch bar chart ──────────────
    st.markdown('<div class="section-header">NPS Score by Branch</div>', unsafe_allow_html=True)
    bkpi_sorted = branch_kpi.sort_values("NPS_Score")
    bar_colors = [GREEN_OK if v >= 30 else (AMBER_WARN if v >= 0 else RED_ALERT)
                  for v in bkpi_sorted["NPS_Score"]]
    fig_bar = go.Figure(go.Bar(
        x=bkpi_sorted["NPS_Score"],
        y=bkpi_sorted["Branch"],
        orientation="h",
        marker_color=bar_colors,
        text=bkpi_sorted["NPS_Score"].apply(lambda v: f"{v:+.1f}"),
        textposition="outside",
    ))
    fig_bar.add_vline(x=0, line_dash="dot", line_color=GREY_TEXT, line_width=1)
    fig_bar.update_layout(
        height=420, xaxis_title="NPS Score",
        plot_bgcolor="white", xaxis_gridcolor="#EDF2F7",
        margin=dict(l=10, r=60, t=10, b=10),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Touchpoint heatmap per branch ─────────
    st.markdown('<div class="section-header">Touchpoint Performance Heatmap by Branch</div>', unsafe_allow_html=True)
    tp_heat = df.groupby("City")[active_touchpoints].mean().round(2)
    tp_heat = tp_heat.loc[branch_kpi["Branch"].values]  # keep ranking order

    fig_heat = go.Figure(go.Heatmap(
        z=tp_heat.values,
        x=[t.replace(" ", "<br>") for t in tp_heat.columns],
        y=tp_heat.index,
        colorscale=[[0, RED_ALERT], [0.5, AMBER_WARN], [1, GREEN_OK]],
        zmin=1, zmax=5,
        text=tp_heat.values.round(2),
        texttemplate="%{text}",
        textfont_size=10,
        colorbar=dict(title="Score", tickvals=[1, 2, 3, 4, 5]),
    ))
    fig_heat.update_layout(
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(side="top"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Worst touchpoints per branch ──────────
    st.markdown('<div class="section-header">⚠️ Priority Improvement Areas by Branch</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    for i, (_, row) in enumerate(tp_heat.iterrows()):
        worst = row.nsmallest(3)
        icons = ["🔴", "🟠", "🟡"]
        html  = f"<b>{row.name}</b><br>" + " &nbsp; ".join(
            f'{icons[j]} {tp} <span style="color:{RED_ALERT if s<3 else AMBER_WARN}">{s:.2f}</span>'
            for j, (tp, s) in enumerate(worst.items())
        )
        (col_a if i % 2 == 0 else col_b).markdown(
            f'<div class="insight-box {"warning" if worst.min()<3 else ""}">{html}</div>',
            unsafe_allow_html=True
        )

# ═══════════════════════════════════════════════
# TAB 3 — DEMOGRAPHICS
# ═══════════════════════════════════════════════
with tabs[2]:

    col_l, col_r = st.columns(2)

    # ── Gender ────────────────────────────────
    with col_l:
        st.markdown('<div class="section-header">KPIs by Gender</div>', unsafe_allow_html=True)
        gen_kpi = df.groupby("Gender").agg(
            Count=("NPS", "count"),
            NPS=("NPS", calc_nps),
            CSI=("CSI", "mean"),
            CLI=("Loyalty", "mean"),
            CES=("CES", "mean"),
        ).reset_index().round(2)
        gen_kpi["NPS"] = gen_kpi["NPS"].round(1)

        fig_gen = go.Figure()
        metrics_g = ["NPS", "CSI", "CLI", "CES"]
        for gender, color in [("Male", BLUE_MAIN), ("Female", TEAL)]:
            row = gen_kpi[gen_kpi["Gender"] == gender]
            if row.empty:
                continue
            # Normalise NPS to 0–5 scale for visual
            vals = [
                (row["NPS"].values[0] + 100) / 200 * 5,
                row["CSI"].values[0],
                row["CLI"].values[0],
                row["CES"].values[0],
            ]
            fig_gen.add_trace(go.Bar(
                name=gender, x=metrics_g, y=vals,
                marker_color=color,
                text=[f"{row['NPS'].values[0]:+.1f}", f"{row['CSI'].values[0]:.2f}",
                      f"{row['CLI'].values[0]:.2f}", f"{row['CES'].values[0]:.2f}"],
                textposition="outside",
            ))
        fig_gen.update_layout(
            barmode="group", height=300, yaxis=dict(range=[0, 5.8]),
            plot_bgcolor="white", yaxis_gridcolor="#EDF2F7",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig_gen, use_container_width=True)

    # ── Age Group ─────────────────────────────
    with col_r:
        st.markdown('<div class="section-header">KPIs by Age Group</div>', unsafe_allow_html=True)
        age_kpi = df.groupby("Age_Group", observed=True).agg(
            Count=("NPS", "count"),
            NPS=("NPS", calc_nps),
            CSI=("CSI", "mean"),
            CLI=("Loyalty", "mean"),
            CES=("CES", "mean"),
        ).reset_index().round(2)
        age_kpi["NPS"] = age_kpi["NPS"].round(1)

        fig_age = go.Figure()
        for metric, color in [("NPS", BLUE_MAIN), ("CSI", TEAL), ("CLI", GREEN_OK), ("CES", AMBER_WARN)]:
            y_vals = [(age_kpi.loc[i, "NPS"] + 100) / 200 * 5 if metric == "NPS"
                      else age_kpi.loc[i, metric]
                      for i in age_kpi.index]
            fig_age.add_trace(go.Scatter(
                x=age_kpi["Age_Group"].astype(str),
                y=y_vals,
                mode="lines+markers",
                name=metric,
                line=dict(color=color, width=2.5),
                marker=dict(size=8),
            ))
        fig_age.update_layout(
            height=300, yaxis=dict(range=[0, 5.8]),
            plot_bgcolor="white", yaxis_gridcolor="#EDF2F7",
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        st.plotly_chart(fig_age, use_container_width=True)

    # ── NPS segment by age & gender ───────────
    st.markdown('<div class="section-header">NPS Segment by Age Group & Gender</div>', unsafe_allow_html=True)
    seg_age = df.groupby(["Age_Group", "Gender", "NPS_Segment"], observed=True).size().reset_index(name="n")
    seg_age["pct"] = seg_age.groupby(["Age_Group", "Gender"])["n"].transform(lambda x: x / x.sum() * 100)

    fig_seg = px.bar(
        seg_age[seg_age["NPS_Segment"] == "Promoter (9–10)"],
        x="Age_Group", y="pct", color="Gender", barmode="group",
        color_discrete_map={"Male": BLUE_MAIN, "Female": TEAL},
        labels={"pct": "Promoter %", "Age_Group": "Age Group"},
    )
    fig_seg.update_layout(
        height=300, yaxis=dict(range=[0, 100], title="Promoter %"),
        plot_bgcolor="white", yaxis_gridcolor="#EDF2F7",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    st.plotly_chart(fig_seg, use_container_width=True)

    # ── Age distribution ─────────────────────
    st.markdown('<div class="section-header">Respondent Age Distribution</div>', unsafe_allow_html=True)
    fig_hist = px.histogram(
        df, x="Age", color="Gender", nbins=30, barmode="overlay",
        color_discrete_map={"Male": BLUE_MAIN, "Female": TEAL},
        opacity=0.75,
    )
    fig_hist.update_layout(
        height=260, plot_bgcolor="white",
        yaxis_gridcolor="#EDF2F7",
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 4 — TIME SERIES
# ═══════════════════════════════════════════════
with tabs[3]:

    granularity = st.radio(
        "Granularity", ["Monthly", "Quarterly"], horizontal=True, key="ts_gran"
    )
    grp_col = "Month" if granularity == "Monthly" else "Quarter"

    ts_data = df.groupby(grp_col).agg(
        NPS_raw=("NPS", calc_nps),
        CSI=("CSI", "mean"),
        CLI=("Loyalty", "mean"),
        CES=("CES", "mean"),
        Count=("NPS", "count"),
    ).reset_index().rename(columns={"NPS_raw": "NPS"})

    # ── NPS Trend ────────────────────────────
    st.markdown('<div class="section-header">NPS Trend Over Time</div>', unsafe_allow_html=True)
    fig_ts_nps = go.Figure()
    fig_ts_nps.add_trace(go.Scatter(
        x=ts_data[grp_col], y=ts_data["NPS"],
        mode="lines+markers+text",
        name="NPS",
        line=dict(color=BLUE_MAIN, width=3),
        marker=dict(size=10, color=BLUE_MAIN),
        text=ts_data["NPS"].apply(lambda v: f"{v:+.1f}"),
        textposition="top center",
        fill="tozeroy",
        fillcolor=f"rgba(30,95,173,.08)",
    ))
    fig_ts_nps.add_hline(y=0, line_dash="dash", line_color=RED_ALERT, line_width=1.2,
                         annotation_text="Zero NPS", annotation_position="left")
    fig_ts_nps.add_hline(y=30, line_dash="dot", line_color=AMBER_WARN, line_width=1,
                         annotation_text="Good (30)", annotation_position="left")
    fig_ts_nps.add_hline(y=50, line_dash="dot", line_color=GREEN_OK, line_width=1,
                         annotation_text="Excellent (50)", annotation_position="left")
    fig_ts_nps.update_layout(
        height=320, plot_bgcolor="white",
        yaxis=dict(title="NPS Score", gridcolor="#EDF2F7"),
        xaxis=dict(title="", tickangle=-30),
        margin=dict(l=60, r=20, t=10, b=50),
    )
    st.plotly_chart(fig_ts_nps, use_container_width=True)

    # ── CSI / CLI / CES Trend ────────────────
    st.markdown('<div class="section-header">CSI · CLI · CES Trend</div>', unsafe_allow_html=True)
    fig_ts_3 = go.Figure()
    for metric, color, dash in [("CSI", TEAL, "solid"), ("CLI", GREEN_OK, "dash"), ("CES", AMBER_WARN, "dot")]:
        fig_ts_3.add_trace(go.Scatter(
            x=ts_data[grp_col], y=ts_data[metric],
            mode="lines+markers",
            name=metric,
            line=dict(color=color, width=2.5, dash=dash),
            marker=dict(size=8),
        ))
    fig_ts_3.update_layout(
        height=280, plot_bgcolor="white",
        yaxis=dict(range=[1, 5.2], title="Score (1–5)", gridcolor="#EDF2F7"),
        xaxis=dict(tickangle=-30),
        margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    st.plotly_chart(fig_ts_3, use_container_width=True)

    # ── NPS by Branch over time ──────────────
    st.markdown('<div class="section-header">NPS Trend — Top 5 vs Bottom 5 Branches</div>', unsafe_allow_html=True)
    nps_branch_ts = (
        df.groupby(["City", grp_col])["NPS"]
        .apply(calc_nps).reset_index(name="NPS")
    )
    # pick top/bottom 5 by overall NPS
    branch_order = branch_kpi["Branch"].tolist()
    top5    = branch_order[:5]
    bottom5 = branch_order[-5:]
    fig_bts = go.Figure()
    for city in top5 + bottom5:
        sub = nps_branch_ts[nps_branch_ts["City"] == city]
        is_top = city in top5
        fig_bts.add_trace(go.Scatter(
            x=sub[grp_col], y=sub["NPS"],
            mode="lines+markers",
            name=city,
            line=dict(
                color=GREEN_OK if is_top else RED_ALERT,
                width=2 if is_top else 1.5,
                dash="solid" if is_top else "dot",
            ),
            marker=dict(size=6),
            opacity=0.85 if is_top else 0.6,
        ))
    fig_bts.add_hline(y=0, line_dash="dash", line_color=GREY_TEXT, line_width=1)
    fig_bts.update_layout(
        height=360, plot_bgcolor="white",
        yaxis=dict(title="NPS Score", gridcolor="#EDF2F7"),
        xaxis=dict(tickangle=-30),
        margin=dict(l=50, r=20, t=10, b=50),
        legend=dict(font_size=10, x=1.01),
    )
    st.plotly_chart(fig_bts, use_container_width=True)

    # ── Response volume ──────────────────────
    st.markdown('<div class="section-header">Monthly Response Volume</div>', unsafe_allow_html=True)
    fig_vol = go.Figure(go.Bar(
        x=ts_data[grp_col], y=ts_data["Count"],
        marker_color=BLUE_LIGHT, text=ts_data["Count"],
        textposition="outside",
    ))
    fig_vol.update_layout(
        height=230, plot_bgcolor="white",
        yaxis=dict(title="Responses", gridcolor="#EDF2F7"),
        margin=dict(l=40, r=20, t=10, b=50),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 5 — TOUCHPOINTS
# ═══════════════════════════════════════════════
with tabs[4]:

    col_l, col_r = st.columns([1, 1])

    # ── Radar chart ───────────────────────────
    with col_l:
        st.markdown('<div class="section-header">Touchpoint Radar — Overall</div>', unsafe_allow_html=True)
        tp_means = df[active_touchpoints].mean().round(2)
        fig_radar = go.Figure(go.Scatterpolar(
            r=list(tp_means.values) + [tp_means.values[0]],
            theta=list(tp_means.index) + [tp_means.index[0]],
            fill="toself",
            fillcolor=f"rgba(30,95,173,.18)",
            line=dict(color=BLUE_MAIN, width=2.5),
            text=tp_means.values.round(2),
            hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[1, 5], gridcolor="#EDF2F7", tickfont_size=9),
                angularaxis=dict(tickfont_size=10),
            ),
            height=380,
            margin=dict(l=30, r=30, t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Bar chart touchpoint ──────────────────
    with col_r:
        st.markdown('<div class="section-header">Mean Score per Touchpoint</div>', unsafe_allow_html=True)
        tp_sorted = tp_means.sort_values()
        bar_clr = [score_color(v) for v in tp_sorted.values]
        fig_tp_bar = go.Figure(go.Bar(
            x=tp_sorted.values,
            y=tp_sorted.index,
            orientation="h",
            marker_color=bar_clr,
            text=tp_sorted.values.round(2),
            textposition="outside",
        ))
        fig_tp_bar.add_vline(x=tp_means.mean(), line_dash="dot",
                             line_color=GREY_TEXT, line_width=1,
                             annotation_text=f"Avg {tp_means.mean():.2f}",
                             annotation_position="top right")
        fig_tp_bar.update_layout(
            height=380, xaxis=dict(range=[1, 5.5], title="Mean Score"),
            plot_bgcolor="white", xaxis_gridcolor="#EDF2F7",
            margin=dict(l=10, r=60, t=10, b=10),
        )
        st.plotly_chart(fig_tp_bar, use_container_width=True)

    # ── Correlation with NPS ──────────────────
    st.markdown('<div class="section-header">Touchpoint Correlation with NPS</div>', unsafe_allow_html=True)
    corr_vals = df[active_touchpoints + ["NPS"]].corr()["NPS"].drop("NPS").sort_values(ascending=False)
    fig_corr = go.Figure(go.Bar(
        x=corr_vals.index,
        y=corr_vals.values,
        marker_color=[score_color(v * 2 + 3) for v in corr_vals.values],
        text=corr_vals.values.round(3),
        textposition="outside",
    ))
    fig_corr.update_layout(
        height=280, xaxis_title="",
        yaxis=dict(title="Pearson r with NPS", range=[-0.05, 0.55]),
        plot_bgcolor="white", yaxis_gridcolor="#EDF2F7",
        margin=dict(l=40, r=10, t=10, b=80),
        xaxis=dict(tickangle=-30),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ── Feedback Word Cloud ───────────────────
    st.markdown('<div class="section-header">Improvement Feedback Word Cloud</div>', unsafe_allow_html=True)
    feedback_text = " ".join(
        df["Improvement_Feedback"]
        .str.replace("service good", "", regex=False)
        .str.replace("improve", "", regex=False)
        .dropna()
        .tolist()
    )
    if feedback_text.strip():
        stopwords_extra = set(STOPWORDS) | {"improve", "service", "good", "nan"}
        wc = WordCloud(
            width=1200, height=400,
            background_color="white",
            colormap="Greens",
            stopwords=stopwords_extra,
            max_words=60,
            min_font_size=12,
        ).generate(feedback_text)
        fig_wc, ax = plt.subplots(figsize=(14, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig_wc)
    else:
        st.info("No feedback text available for word cloud with current filters.")

    # ── Complaint frequency ───────────────────
    st.markdown('<div class="section-header">Top Complaint Areas</div>', unsafe_allow_html=True)
    complaint_df = (
        df[df["Improvement_Feedback"].str.startswith("improve", na=False)]
        ["Improvement_Feedback"]
        .str.replace("improve ", "", regex=False)
        .str.split(", ")
        .explode()
        .str.strip()
        .value_counts()
        .reset_index()
        .rename(columns={"index": "Area", "Improvement_Feedback": "Count",
                         "count": "Count"})
        .head(10)
    )
    # Rename columns properly
    complaint_df.columns = ["Area", "Count"]
    complaint_df = complaint_df[complaint_df["Area"] != ""]
    fig_comp = go.Figure(go.Bar(
        x=complaint_df["Area"], y=complaint_df["Count"],
        marker_color=[RED_ALERT if v == complaint_df["Count"].max() else BLUE_LIGHT
                      for v in complaint_df["Count"]],
        text=complaint_df["Count"], textposition="outside",
    ))
    fig_comp.update_layout(
        height=280, plot_bgcolor="white", yaxis_gridcolor="#EDF2F7",
        margin=dict(l=10, r=10, t=10, b=80),
        xaxis=dict(tickangle=-30, title=""),
        yaxis_title="Complaint Count",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 6 — AI INSIGHTS
# ═══════════════════════════════════════════════
with tabs[5]:

    st.markdown('<div class="section-header">🤖 AI-Powered Insights & Recommendations</div>',
                unsafe_allow_html=True)

    if not ai_enabled:
        st.info(
            "AI insights are disabled. Set the `GEMINI_API_KEY` environment variable "
            "or add it to your `.streamlit/secrets.toml` file, then toggle **Enable AI Insights** "
            "in the sidebar.",
            icon="💡",
        )
        st.markdown("""
        **How to enable:**
        ```toml
        # .streamlit/secrets.toml
        GEMINI_API_KEY = "AIzaSy..."
        ```
        Or set the environment variable before running:
        ```bash
        GEMINI_API_KEY=AIzaSy... streamlit run app.py
