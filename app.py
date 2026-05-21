"""
Always Healthy Hospital — Customer Experience Dashboard
========================================================
Dataset  : Survei kepuasan pelanggan (3.600 responden, 15 cabang)
Stack    : Streamlit · Plotly · Pandas · Gemini API (Chatbot)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import textwrap
from datetime import datetime

import google.generativeai as genai
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────
# PENGATURAN HALAMAN & TEMA GELAP
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Always Healthy Hospital KPI Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palet Warna Kontras Tinggi untuk Background Gelap
COLOR_BG = "#121212"
COLOR_TEXT = "#E0E0E0"
COLOR_PRIMARY = "#00E5FF"  # Cyan/Neon Blue
COLOR_POS = "#00E676"      # Bright Lime Green
COLOR_WARN = "#FFC400"     # Amber/Yellow
COLOR_NEG = "#FF1744"      # Bright Red
COLOR_NEUTRAL = "#9E9E9E"  # Gray

# Injeksi CSS untuk memaksa warna gelap dan menyesuaikan teks
st.markdown(f"""
<style>
    .stApp {{ background-color: {COLOR_BG}; color: {COLOR_TEXT}; }}
    .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
    h1, h2, h3, h4, p, span, label {{ color: {COLOR_TEXT} !important; }}
    [data-testid="stMetricValue"] {{ color: {COLOR_PRIMARY} !important; }}
    [data-testid="stMetricDelta"] svg {{ fill: currentColor; }}
    hr {{ border-color: #333333; }}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# PEMETAAN BAHASA & DATA
# ──────────────────────────────────────────────
# Memetakan nama titik interaksi UI (Indonesia) ke Data CSV (Inggris)
TOUCHPOINT_MAP = {
    "Registrasi": "Registration",
    "Konsultasi Dokter": "Doctor Consultation",
    "Layanan Perawat": "Nurse Service",
    "Layanan Farmasi": "Pharmacy Service",
    "Laboratorium": "Laboratory",
    "Respon Gawat Darurat": "Emergency Response",
    "Proses Tagihan": "Billing Process",
    "Kebersihan Fasilitas": "Facility Cleanliness",
    "Keramahan Staf": "Staff Friendliness",
    "Waktu Tunggu": "Waiting Time",
}

MONTH_LABELS = {
    "2025-01": "Jan", "2025-02": "Feb", "2025-03": "Mar",
    "2025-04": "Apr", "2025-05": "Mei", "2025-06": "Jun",
    "2025-07": "Jul", "2025-08": "Agu", "2025-09": "Sep",
    "2025-10": "Okt", "2025-11": "Nov", "2025-12": "Des",
}

# ──────────────────────────────────────────────
# FUNGSI PEMROSESAN DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("data.csv", sep=";")
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d/%m/%Y %H.%M")
    df["Month"]    = df["Datetime"].dt.to_period("M").astype(str)
    df["City"]     = df["Branch"].str.replace("Always Healthy Hospital ", "", regex=False)
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[20, 30, 40, 50, 60, 70],
        labels=["21-30", "31-40", "41-50", "51-60", "61-65"],
    )
    return df

df_full = load_data()

def calc_nps(series: pd.Series) -> float:
    if len(series) == 0: return 0.0
    return round(((series >= 9).sum() - (series <= 6).sum()) / len(series) * 100, 1)

def score_color(val: float, vmin=1.0, vmax=5.0) -> str:
    ratio = (val - vmin) / (vmax - vmin)
    if ratio >= .75: return COLOR_POS
    if ratio >= .40: return COLOR_WARN
    return COLOR_NEG

def apply_dark_theme(fig):
    """Menerapkan tema gelap dan kontras tinggi ke semua grafik Plotly."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLOR_TEXT),
        xaxis=dict(gridcolor="#333333"),
        yaxis=dict(gridcolor="#333333")
    )
    return fig

# ──────────────────────────────────────────────
# PENGATURAN SIDEBAR & FILTER
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Filter Data")
    
    all_months = sorted(df_full["Month"].unique())
    month_options = [f"{MONTH_LABELS.get(m, m)} {m[:4]}" for m in all_months]
    start_idx, end_idx = st.select_slider(
        "Periode Waktu",
        options=list(range(len(all_months))),
        value=(0, len(all_months) - 1),
        format_func=lambda i: month_options[i],
    )
    selected_months = all_months[start_idx : end_idx + 1]

    branch_select = st.multiselect(
        "Cabang Rumah Sakit",
        options=sorted(df_full["City"].unique()),
        default=sorted(df_full["City"].unique())
    )

    gender_select = st.multiselect("Jenis Kelamin", options=["Male", "Female"], default=["Male", "Female"])
    
    ui_tp_select = st.multiselect("Titik Interaksi", options=list(TOUCHPOINT_MAP.keys()), default=list(TOUCHPOINT_MAP.keys()))
    active_touchpoints = [TOUCHPOINT_MAP[tp] for tp in ui_tp_select]

    st.markdown("---")
    st.markdown("### Integrasi AI")
    api_key_env = os.environ.get("GEMINI_API_KEY", "")
    try:
        api_key_env = api_key_env or st.secrets.get("GEMINI_API_KEY", "")
    except Exception: pass
    
    ai_enabled = False
    if api_key_env:
        ai_enabled = st.toggle("Aktifkan Asisten AI", value=True)
    else:
        st.info("Konfigurasikan GEMINI_API_KEY di Streamlit Secrets untuk fitur AI.")

# Filter Dataset
df = df_full[
    df_full["Month"].isin(selected_months) &
    df_full["City"].isin(branch_select if branch_select else sorted(df_full["City"].unique())) &
    df_full["Gender"].isin(gender_select if gender_select else ["Male", "Female"])
].copy()

if df.empty:
    st.error("Tidak ada data yang cocok dengan filter saat ini.")
    st.stop()

# ──────────────────────────────────────────────
# TATA LETAK UTAMA (MAIN LAYOUT)
# ──────────────────────────────────────────────
st.markdown("## Customer Experience Dashboard")
st.markdown(f"**Periode: {MONTH_LABELS.get(selected_months[0], selected_months[0])} - {MONTH_LABELS.get(selected_months[-1], selected_months[-1])} 2025** | Total Responden: {len(df):,} | Jumlah Cabang: {df['City'].nunique()}")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Executive Summary", "Analisis Performa", "Asisten AI & Feedback Pelanggan"])

# ═══════════════════════════════════════════════
# TAB 1 — RINGKASAN EKSEKUTIF
# ═══════════════════════════════════════════════
with tab1:
    nps_val = calc_nps(df["NPS"])
    nps_overall = calc_nps(df_full["NPS"])
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Promoter Score (NPS)", f"{nps_val:+.1f}", f"{nps_val - nps_overall:+.1f} vs Rata-rata")
    c2.metric("Kepuasan Pelanggan (CSI)", f"{df['CSI'].mean():.2f}", f"{df['CSI'].mean() - df_full['CSI'].mean():+.2f} vs Rata-rata")
    c3.metric("Loyalitas (CLI)", f"{df['Loyalty'].mean():.2f}", f"{df['Loyalty'].mean() - df_full['Loyalty'].mean():+.2f} vs Rata-rata")
    c4.metric("Usaha Pelanggan (CES)", f"{df['CES'].mean():.2f}", f"{-(df['CES'].mean() - df_full['CES'].mean()):+.2f} (Skor Rendah = Baik)")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([2, 1])

    with col_l:
        st.markdown("#### Tren NPS & Kepuasan Bulanan")
        ts_data = df.groupby("Month").agg(NPS=("NPS", calc_nps), CSI=("CSI", "mean")).reset_index()
        
        fig_ts = go.Figure()
        fig_ts.add_trace(go.Scatter(x=ts_data["Month"], y=ts_data["NPS"], mode="lines+markers", name="NPS", line=dict(color=COLOR_PRIMARY, width=3), yaxis="y1"))
        fig_ts.add_trace(go.Scatter(x=ts_data["Month"], y=ts_data["CSI"], mode="lines+markers", name="CSI", line=dict(color=COLOR_NEUTRAL, width=2, dash="dash"), yaxis="y2"))
        
        fig_ts.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            yaxis=dict(title="Skor NPS", side="left"),
            yaxis2=dict(title="Skor CSI (1-5)", side="right", overlaying="y", range=[1, 5]),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2)
        )
        st.plotly_chart(apply_dark_theme(fig_ts), use_container_width=True)

    with col_r:
        st.markdown("#### Distribusi Demografi")
        fig_dem = px.histogram(df, x="Age_Group", color="Gender", barmode="group", color_discrete_sequence=[COLOR_PRIMARY, COLOR_NEUTRAL])
        fig_dem.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Jumlah", xaxis_title="Kelompok Umur", legend=dict(title=None, orientation="h", y=-0.2))
        st.plotly_chart(apply_dark_theme(fig_dem), use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 2 — ANALISIS PERFORMA
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("#### Peringkat Performa Cabang")
    
    branch_kpi = df.groupby("City").agg(
        Responden=("NPS", "count"),
        NPS=("NPS", calc_nps),
        CSI=("CSI", "mean"),
        CLI=("Loyalty", "mean"),
        CES=("CES", "mean")
    ).reset_index().rename(columns={"City": "Cabang"}).sort_values("NPS", ascending=False)
    
    def style_nps(val):
        if val >= 50: return f"color: {COLOR_POS}; font-weight: bold"
        if val >= 0: return f"color: {COLOR_WARN}; font-weight: bold"
        return f"color: {COLOR_NEG}; font-weight: bold"

    st.dataframe(
        branch_kpi.style
        .map(style_nps, subset=["NPS"])
        .format({"NPS": "{:+.1f}", "CSI": "{:.2f}", "CLI": "{:.2f}", "CES": "{:.2f}"}),
        use_container_width=True, height=250
    )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Rata-rata Skor per Titik Interaksi")
        tp_means = df[active_touchpoints].mean().sort_values()
        
        # Terjemahkan index untuk chart
        ui_tp_names = [list(TOUCHPOINT_MAP.keys())[list(TOUCHPOINT_MAP.values()).index(tp)] for tp in tp_means.index]
        
        fig_tp = go.Figure(go.Bar(
            x=tp_means.values, y=ui_tp_names, orientation="h",
            marker_color=[score_color(v) for v in tp_means.values],
            text=tp_means.values.round(2), textposition="outside"
        ))
        fig_tp.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0), xaxis=dict(range=[1, 5.2]))
        st.plotly_chart(apply_dark_theme(fig_tp), use_container_width=True)

    with c2:
        st.markdown("#### Korelasi Layanan terhadap NPS")
        corr_vals = df[active_touchpoints + ["NPS"]].corr()["NPS"].drop("NPS").sort_values()
        ui_corr_names = [list(TOUCHPOINT_MAP.keys())[list(TOUCHPOINT_MAP.values()).index(tp)] for tp in corr_vals.index]
        
        fig_corr = go.Figure(go.Bar(
            x=corr_vals.values, y=ui_corr_names, orientation="h",
            marker_color=COLOR_PRIMARY,
            text=corr_vals.values.round(3), textposition="outside"
        ))
        fig_corr.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(apply_dark_theme(fig_corr), use_container_width=True)

# ═══════════════════════════════════════════════
# TAB 3 — ASISTEN AI & UMPAN BALIK
# ═══════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        st.markdown("#### Topik Keluhan Tertinggi")
        complaints = df[df["Improvement_Feedback"].str.startswith("improve", na=False)]
        complaint_counts = complaints["Improvement_Feedback"].str.replace("improve ", "").str.split(", ").explode().value_counts().head(8)
        
        if not complaint_counts.empty:
            fig_comp = px.bar(y=complaint_counts.index, x=complaint_counts.values, orientation="h", color_discrete_sequence=[COLOR_NEG])
            fig_comp.update_layout(height=350, yaxis_title=None, xaxis_title="Frekuensi", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(apply_dark_theme(fig_comp), use_container_width=True)
        else:
            st.info("Data umpan balik tidak mencukupi untuk rentang waktu ini.")

    with c2:
        st.markdown("#### Asisten Data Analis AI")
        
        if not ai_enabled:
            st.warning("Integrasi AI dinonaktifkan. Harap masukkan GEMINI_API_KEY untuk berinteraksi dengan data.")
        else:
            # Mempersiapkan konteks sistem dari data aktual
            context_summary = textwrap.dedent(f"""
            Ringkasan Data Saat Ini:
            - Skor NPS: {nps_val:+.1f}
            - Skor Kepuasan (CSI): {df['CSI'].mean():.2f}
            - Cabang Terbaik: {branch_kpi.iloc[0]['Cabang']} (NPS: {branch_kpi.iloc[0]['NPS']})
            - Cabang Terburuk: {branch_kpi.iloc[-1]['Cabang']} (NPS: {branch_kpi.iloc[-1]['NPS']})
            - Keluhan terbanyak: {', '.join(complaint_counts.index[:5]) if not complaint_counts.empty else 'Tidak ada data spesifik'}
            """)

            sys_prompt = (
                "Anda adalah analis CX senior di Always Healthy Hospital. "
                "Jawab pertanyaan pengguna dalam bahasa Indonesia yang profesional dan lugas. "
                "Gunakan konteks data berikut untuk memberikan wawasan operasional yang dapat ditindaklanjuti:\n"
                f"{context_summary}"
            )

            # Inisialisasi riwayat obrolan di session state
            if "messages" not in st.session_state:
                st.session_state.messages = [
                    {"role": "assistant", "content": "Halo! Saya Asisten AI Always Healthy Hospital. Data performa bulan ini sudah saya pelajari. Ada wawasan spesifik yang ingin Anda gali?"}
                ]

            # Area obrolan (chat container)
            chat_container = st.container(height=350)
            
            with chat_container:
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            # Input obrolan pengguna
            if prompt := st.chat_input("Tanyakan sesuatu terkait performa data..."):
                # Simpan dan tampilkan pesan user
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Panggil model Gemini
                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        try:
                            genai.configure(api_key=api_key_env)
                            model = genai.GenerativeModel(
                                model_name="gemini-2.5-flash",
                                system_instruction=sys_prompt
                            )
                            
                            # Konversi format histori Streamlit ke format Gemini API
                            gemini_history = []
                            for msg in st.session_state.messages[:-1]:
                                role = "user" if msg["role"] == "user" else "model"
                                gemini_history.append({"role": role, "parts": [msg["content"]]})
                                
                            chat = model.start_chat(history=gemini_history)
                            response = chat.send_message(prompt)
                            
                            message_placeholder.markdown(response.text)
                            st.session_state.messages.append({"role": "assistant", "content": response.text})
                        except Exception as e:
                            message_placeholder.error(f"Terjadi kesalahan saat memuat AI: {e}")
