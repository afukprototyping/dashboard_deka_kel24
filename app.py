# ============================================================
# Always Healthy Hospital — Customer Experience Dashboard
# Team 24 × Deka Insight | 2025
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="AHH CX Dashboard | Deka Insight",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Brand Colors ──────────────────────────────────────────────
YELLOW      = "#FAA61A"
BLUE        = "#1A2D5A"
YELLOW_LIGHT = "#FFF8EC"
YELLOW_MID  = "#FDCB6E"
BLUE_LIGHT  = "#E8EDF5"
BLUE_MID    = "#2D4F8E"
BLUE_PALE   = "#9BAED0"
GREEN       = "#22C55E"
RED         = "#EF4444"
PURPLE      = "#8B5CF6"
WHITE       = "#FFFFFF"
GRAY_LIGHT  = "#F5F6FA"
GRAY_MID    = "#9EA8BA"

# ── CSS Styling ───────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
.main {{ background-color: {GRAY_LIGHT}; }}
.block-container {{ padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {BLUE} 0%, #0F1E3D 100%);
}}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {{
    color: rgba(255,255,255,0.85) !important;
    font-size: 13px;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: {YELLOW} !important;
}}
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    color: white;
}}
[data-testid="stSidebar"] .stTextInput > div > div > input {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
    border-radius: 8px;
}}
[data-testid="stSidebar"] .stMultiSelect span[data-baseweb="tag"] {{
    background: rgba(250,166,26,0.25);
    color: {YELLOW};
}}

/* ── KPI Card ── */
.kpi-card {{
    background: white;
    border-radius: 16px;
    padding: 20px 22px 16px 22px;
    box-shadow: 0 2px 16px rgba(26,45,90,0.09);
    border-top: 4px solid {YELLOW};
    position: relative;
    overflow: hidden;
    min-height: 150px;
}}
.kpi-card-blue {{ border-top-color: {BLUE}; }}
.kpi-card-green {{ border-top-color: {GREEN}; }}
.kpi-card-purple {{ border-top-color: {PURPLE}; }}
.kpi-bg-circle {{
    position: absolute;
    bottom: -20px; right: -20px;
    width: 90px; height: 90px;
    border-radius: 50%;
    opacity: 0.06;
}}
.kpi-label {{
    font-size: 11px;
    font-weight: 700;
    color: {GRAY_MID};
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
}}
.kpi-value {{
    font-size: 38px;
    font-weight: 800;
    color: {BLUE};
    line-height: 1;
    margin-bottom: 6px;
}}
.kpi-sub {{
    font-size: 11px;
    color: {GRAY_MID};
    margin-bottom: 8px;
}}
.kpi-delta-pos {{ color: {GREEN}; font-size: 12px; font-weight: 600; }}
.kpi-delta-neg {{ color: {RED}; font-size: 12px; font-weight: 600; }}
.kpi-delta-neu {{ color: {GRAY_MID}; font-size: 12px; font-weight: 600; }}
.kpi-icon {{
    position: absolute;
    top: 18px; right: 20px;
    font-size: 30px; opacity: 0.15;
}}

/* ── Section Header ── */
.section-hdr {{
    font-size: 15px;
    font-weight: 700;
    color: {BLUE};
    border-left: 4px solid {YELLOW};
    padding-left: 12px;
    margin: 28px 0 14px 0;
    letter-spacing: 0.3px;
}}

/* ── Insight Box ── */
.insight-box {{
    background: linear-gradient(135deg, {YELLOW_LIGHT} 0%, #FFF0D0 100%);
    border: 1px solid rgba(250,166,26,0.3);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
}}
.insight-title {{
    color: {BLUE};
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 6px;
}}
.insight-text {{
    color: #3A3F52;
    font-size: 13px;
    line-height: 1.65;
}}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(26,45,90,0.05);
    border-radius: 10px;
    padding: 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border-radius: 8px;
    color: {BLUE};
    font-weight: 500;
    font-size: 13px;
    padding: 6px 16px;
}}
.stTabs [aria-selected="true"] {{
    background: {YELLOW} !important;
    color: {BLUE} !important;
    font-weight: 700;
}}

/* ── Metric chip ── */
.chip {{
    display: inline-block;
    background: {BLUE_LIGHT};
    color: {BLUE};
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
}}

/* ── Hide streamlit branding ── */
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONSTANTS
# ============================================================
REGION_MAP = {
    'Jakarta': 'Java', 'Bogor': 'Java', 'Depok': 'Java', 'Tangerang': 'Java',
    'Bekasi': 'Java', 'Bandung': 'Java', 'Semarang': 'Java', 'Yogyakarta': 'Java',
    'Surabaya': 'Java', 'Malang': 'Java',
    'Padang': 'Sumatera', 'Batam': 'Sumatera', 'Medan': 'Sumatera',
    'Palembang': 'Sumatera', 'Pekanbaru': 'Sumatera',
    'Balikpapan': 'Kalimantan', 'Banjarmasin': 'Kalimantan',
    'Makassar': 'Sulawesi', 'Manado': 'Sulawesi',
    'Denpasar': 'Bali'
}

REGION_ORDER = ['Java', 'Sumatera', 'Kalimantan', 'Sulawesi', 'Bali']

TOUCHPOINTS = [
    'Registration', 'Doctor Consultation', 'Nurse Service',
    'Pharmacy Service', 'Laboratory', 'Emergency Response',
    'Billing Process', 'Facility Cleanliness', 'Staff Friendliness', 'Waiting Time'
]

TOUCHPOINT_ICONS = {
    'Registration': '📋', 'Doctor Consultation': '👨‍⚕️', 'Nurse Service': '💉',
    'Pharmacy Service': '💊', 'Laboratory': '🔬', 'Emergency Response': '🚨',
    'Billing Process': '🧾', 'Facility Cleanliness': '🧹',
    'Staff Friendliness': '😊', 'Waiting Time': '⏱️'
}

MONTHS_SHORT = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'
}

PLOTLY_BASE = dict(
    paper_bgcolor='white', plot_bgcolor='white',
    font=dict(family='Inter', color=BLUE),
    margin=dict(t=44, b=24, l=10, r=10),
    title_font=dict(size=14, color=BLUE, family='Inter'),
    title_x=0,
)


# ============================================================
# DATA LOADING & PROCESSING
# ============================================================
@st.cache_data
def load_data():
    # Mengamankan path baca agar support cloud environment
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'data.csv')
    
    df = pd.read_csv(file_path, sep=None, engine='python')
    df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d/%m/%Y %H.%M')
    df['Month']       = df['Datetime'].dt.month
    df['Month_Name']  = df['Datetime'].dt.strftime('%b')
    df['Quarter']     = 'Q' + df['Datetime'].dt.quarter.astype(str)
    
    # NPS segmentation (standard method)
    conditions = [df['NPS'] >= 9, df['NPS'] >= 7]
    choices    = ['Promoter', 'Passive']
    df['NPS_Group'] = np.select(conditions, choices, default='Detractor')

    # Age groups
    df['Age_Group'] = pd.cut(
        df['Age'], bins=[20, 30, 40, 50, 65],
        labels=['21–30', '31–40', '41–50', '51–65']
    )

    # City & Region
    df['City']   = df['Branch'].str.replace('Always Healthy Hospital ', '', regex=False)
    df['Region'] = df['City'].map(REGION_MAP)

    # Feedback category
    df['Feedback_Type'] = np.where(
        df['Improvement_Feedback'] == 'service good', 'Positive', 'Needs Improvement'
    )
    df['Feedback_Area'] = df['Improvement_Feedback'].str.replace(
        'improve ', '', regex=False
    ).str.strip()
    df.loc[df['Feedback_Type'] == 'Positive', 'Feedback_Area'] = 'Positive'

    return df

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def calc_nps(df):
    if len(df) == 0: return 0.0
    p = (df['NPS'] >= 9).sum()
    d = (df['NPS'] <= 6).sum()
    return round((p / len(df) - d / len(df)) * 100, 1)

def calc_kpis(df):
    if len(df) == 0: return 0.0, 0.0, 0.0, 0.0
    return (
        calc_nps(df),
        round(df['CSI'].mean(),     2),
        round(df['Loyalty'].mean(), 2),
        round(df['CES'].mean(),     2)
    )

def nps_color(v):
    if v >= 50: return GREEN
    if v >= 0:  return YELLOW
    return RED

def score_color(v, scale=5):
    r = v / scale
    if r >= 0.80: return GREEN
    if r >= 0.65: return YELLOW
    return RED

def delta_html(val, label="vs H1", reverse=False):
    if val is None: return f'<span class="kpi-delta-neu">— no prior period</span>'
    good = (val > 0) if not reverse else (val < 0)
    arrow = "▲" if val > 0 else "▼"
    cls = "kpi-delta-pos" if good else "kpi-delta-neg"
    return f'<span class="{cls}">{arrow} {abs(val):.2f} {label}</span>'

def fmt_score(v): return f"{v:.2f}"
def fmt_nps(v):   return f"{v:+.1f}"

def plotly_chart(fig, height=None, **kwargs):
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False}, **kwargs)


# ============================================================
# LOAD DATA
# ============================================================
try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ `data.csv` tidak ditemukan di direktori saat ini. Pastikan file diupload bersama app.py.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 8px 0 18px 0;'>
        <div style='font-size:28px;'>🏥</div>
        <div style='color:{YELLOW}; font-weight:800; font-size:15px; margin-top:6px;'>
            Always Healthy Hospital
        </div>
        <div style='color:rgba(255,255,255,0.6); font-size:11px; margin-top:4px;'>
            CX Analytics · Team 24 × Deka Insight
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"### 📅 Time Period")
    quarter_opts = ['All Quarters', 'Q1', 'Q2', 'Q3', 'Q4']
    selected_q = st.selectbox("Quarter", quarter_opts, index=0)

    month_map = {MONTHS_SHORT[m]: m for m in sorted(df['Month'].unique())}
    if selected_q == 'All Quarters':
        default_months = list(month_map.keys())
    else:
        q_months = {'Q1': [1,2,3], 'Q2': [4,5,6], 'Q3': [7,8,9], 'Q4': [10,11,12]}
        default_months = [k for k, v in month_map.items() if v in q_months[selected_q]]

    selected_months = st.multiselect("Month(s)", list(month_map.keys()), default=default_months)
    if not selected_months:
        selected_months = list(month_map.keys())
    month_nums = [month_map[m] for m in selected_months]

    st.markdown(f"### 🗺️ Area & Branch")
    region_opts = ['All Regions'] + REGION_ORDER
    selected_region = st.selectbox("Region", region_opts)

    if selected_region == 'All Regions':
        city_pool = sorted(df['City'].unique())
    else:
        city_pool = sorted(df[df['Region'] == selected_region]['City'].unique())

    selected_cities = st.multiselect("Branch/City", city_pool, default=city_pool)
    if not selected_cities:
        selected_cities = city_pool

    st.markdown(f"### 🎯 Touchpoint Focus")
    tp_opts = ['All Touchpoints'] + TOUCHPOINTS
    selected_tp = st.selectbox("Touchpoint", tp_opts)

    st.markdown(f"### 👤 Demographics")
    gender_opts = ['All'] + sorted(df['Gender'].unique().tolist())
    selected_gender = st.selectbox("Gender", gender_opts)

    age_group_opts = ['21–30', '31–40', '41–50', '51–65']
    selected_ages = st.multiselect("Age Group", age_group_opts, default=age_group_opts)
    if not selected_ages:
        selected_ages = age_group_opts

    st.markdown("---")
    st.markdown(f"### 🤖 AI Insights (Groq)")
    groq_key = ""
    try:
        # Penyesuaian st.secrets untuk GCP/Streamlit Cloud
        groq_key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        pass
        
    if not groq_key:
        # Fallback to Environment Variable for GCP (Cloud Run)
        groq_key = os.environ.get("GROQ_API_KEY", "")
        
    if not groq_key:
        groq_key = st.text_input(
            "Groq API Key", type="password",
            placeholder="gsk_xxxx…  (free at groq.com)",
            help="Get a free key at console.groq.com. No credit card required."
        )


# ============================================================
# APPLY FILTERS
# ============================================================
dff = df[
    df['Month'].isin(month_nums) &
    df['City'].isin(selected_cities) &
    df['Age_Group'].isin(selected_ages)
].copy()

if selected_gender != 'All':
    dff = dff[dff['Gender'] == selected_gender]

n_resp = len(dff)

# ============================================================
# DASHBOARD HEADER
# ============================================================
c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
with c1:
    st.markdown(f"""
    <div style='padding: 6px 0 4px;'>
        <span style='font-size:23px; font-weight:800; color:{BLUE};'>🏥 Always Healthy Hospital</span>
        <span style='font-size:13px; color:{GRAY_MID}; margin-left:12px;'>
            Customer Experience Dashboard · 2025
        </span>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div style='background:{BLUE}; border-radius:12px; padding:10px 14px; text-align:center;'>
        <div style='color:{YELLOW}; font-size:10px; font-weight:700; letter-spacing:1px;'>RESPONSES</div>
        <div style='color:white; font-size:22px; font-weight:800;'>{n_resp:,}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:4px; background:linear-gradient(90deg,{0} 0%,{1} 50%,{0} 100%); border-radius:4px; margin:14px 0 4px;'></div>".format(YELLOW, BLUE), unsafe_allow_html=True)

# ============================================================
# KPI CARDS
# ============================================================
st.markdown('<div class="section-hdr">Key Performance Indicators</div>', unsafe_allow_html=True)
nps_val, csi_val, loy_val, ces_val = calc_kpis(dff)

col1, col2, col3, col4 = st.columns(4)
kpi_defs = [
    (col1, "Net Promoter Score", fmt_nps(nps_val), nps_color(nps_val), "kpi-card", None, "Scale: −100 to +100", "📊"),
    (col2, "Customer Satisfaction", fmt_score(csi_val), score_color(csi_val), "kpi-card kpi-card-blue", None, "Scale: 1–5", "⭐"),
    (col3, "Customer Loyalty Index", fmt_score(loy_val), score_color(loy_val), "kpi-card kpi-card-green", None, "Scale: 1–5", "🤝"),
    (col4, "Customer Effort Score", fmt_score(ces_val), score_color(ces_val), "kpi-card kpi-card-purple", None, "Scale: 1–5 (↑ easier)", "⚡"),
]

for col, label, value, vcolor, card_cls, delta, subscale, icon in kpi_defs:
    with col:
        st.markdown(f"""
        <div class="{card_cls}">
            <div class="kpi-icon">{icon}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value" style="color:{vcolor};">{value}</div>
            <div class="kpi-sub">{subscale}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# DEMOGRAPHIC ANALYSIS (WITH FIX)
# ============================================================
st.markdown('<div class="section-hdr">Demographic Analysis</div>', unsafe_allow_html=True)
tab_d1, tab_d2 = st.tabs(["👥 Gender Breakdown", "🎂 Age Group Breakdown"])

with tab_d1:
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        gender_summary = (
            dff.groupby('Gender')
            .apply(lambda x: pd.Series({
                'NPS': calc_nps(x), 'CSI': x['CSI'].mean(),
                'Loyalty': x['Loyalty'].mean(), 'CES': x['CES'].mean(),
                'Count': len(x)
            }))
            .reset_index()
        )
        fig_gen_kpi = go.Figure()
        for metric, color in [('CSI', YELLOW), ('Loyalty', GREEN), ('CES', PURPLE)]:
            fig_gen_kpi.add_trace(go.Bar(
                name=metric, x=gender_summary['Gender'], y=gender_summary[metric],
                marker_color=color, text=gender_summary[metric].round(2),
                textposition='auto', textfont_size=12
            ))
        fig_gen_kpi.update_layout(**PLOTLY_BASE, title="CSI, Loyalty & CES by Gender", barmode='group', height=300)
        plotly_chart(fig_gen_kpi)

with tab_d2:
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        # FIX: Ditambahkan 'CES': x['CES'].mean() agar tidak terjadi KeyError lagi.
        age_summary = (
            dff.groupby('Age_Group', observed=True)
            .apply(lambda x: pd.Series({
                'NPS': calc_nps(x), 
                'CSI': x['CSI'].mean(),
                'Loyalty': x['Loyalty'].mean(), 
                'CES': x['CES'].mean(), 
                'Count': len(x)
            }))
            .reset_index()
        )
        fig_age = go.Figure()
        fig_age.add_trace(go.Bar(
            x=age_summary['Age_Group'].astype(str), y=age_summary['Count'],
            name='Responses', marker_color=BLUE_LIGHT, yaxis='y2', opacity=0.5
        ))
        fig_age.add_trace(go.Scatter(
            x=age_summary['Age_Group'].astype(str), y=age_summary['NPS'],
            name='NPS', mode='lines+markers', line=dict(color=YELLOW, width=3),
            yaxis='y'
        ))
        fig_age.update_layout(**PLOTLY_BASE, title="NPS & Response Volume by Age", height=300, yaxis2=dict(overlaying='y', side='right'))
        plotly_chart(fig_age)

    with col_a2:
        fig_age_kpi = go.Figure()
        for metric, color in [('CSI', YELLOW), ('Loyalty', GREEN), ('CES', PURPLE)]:
            fig_age_kpi.add_trace(go.Bar(
                name=metric,
                x=age_summary['Age_Group'].astype(str),
                y=age_summary[metric],
                marker_color=color,
                text=age_summary[metric].round(2),
                textposition='auto', textfont_size=11
            ))
        fig_age_kpi.update_layout(**PLOTLY_BASE, title="KPI Scores by Age Group", height=300, barmode='group')
        plotly_chart(fig_age_kpi)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; color:{GRAY_MID}; font-size:12px; padding:16px; border-top: 1px solid #E8EDF5;'>
    <strong style='color:{BLUE};'>Always Healthy Hospital</strong> · Customer Experience Analytics Dashboard
</div>
""", unsafe_allow_html=True)