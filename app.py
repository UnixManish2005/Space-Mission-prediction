"""
app.py - Space Mission Success Prediction Dashboard
=====================================================
A modern, space-themed Streamlit application for predicting
space mission outcomes using machine learning.

Run:
    streamlit run app.py
"""

import os
import sys
import warnings
import json
import io
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    load_data, clean_data, create_target, feature_engineering,
    build_input_row, compute_metrics, FEATURE_COLUMNS
)

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'data', 'Space_Corrected.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.pkl')
META_PATH  = os.path.join(BASE_DIR, 'models', 'model_meta.json')


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🚀 Space Mission Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── Import Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

/* ─── Global ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

/* ─── Background ─────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #020918 0%, #050d2a 40%, #0a1628 70%, #060e1e 100%);
}

/* ─── Stars overlay (pure CSS) ──────────────────────── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        radial-gradient(1px 1px at 15% 10%, rgba(255,255,255,0.8) 0%, transparent 100%),
        radial-gradient(1px 1px at 35% 25%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 55% 8%,  rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 75% 18%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 90% 5%,  rgba(255,255,255,0.9) 0%, transparent 100%),
        radial-gradient(1px 1px at 5%  40%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 25% 55%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 65% 42%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 85% 60%, rgba(255,255,255,0.7) 0%, transparent 100%),
        radial-gradient(1px 1px at 45% 75%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 10% 80%, rgba(255,255,255,0.6) 0%, transparent 100%),
        radial-gradient(1px 1px at 70% 85%, rgba(255,255,255,0.5) 0%, transparent 100%);
    pointer-events: none;
    z-index: 0;
}

/* ─── Header ─────────────────────────────────────────── */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 50%, #ff6b35 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: 2px;
}
.hero-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.2rem;
    color: #7ec8e3;
    text-align: center;
    letter-spacing: 4px;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ─── Cards ──────────────────────────────────────────── */
.metric-card {
    background: rgba(0, 212, 255, 0.05);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: rgba(0, 212, 255, 0.5);
    background: rgba(0, 212, 255, 0.1);
    transform: translateY(-2px);
}
.metric-value {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #00d4ff;
}
.metric-label {
    font-size: 0.9rem;
    color: #7ec8e3;
    text-transform: uppercase;
    letter-spacing: 2px;
}

/* ─── Success/Failure Banners ───────────────────────── */
.success-banner {
    background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,212,255,0.1));
    border: 2px solid #00ff88;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.failure-banner {
    background: linear-gradient(135deg, rgba(255,65,65,0.15), rgba(255,107,53,0.1));
    border: 2px solid #ff4141;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.banner-emoji { font-size: 3rem; }
.banner-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
}
.success-banner .banner-title { color: #00ff88; }
.failure-banner  .banner-title { color: #ff4141; }

/* ─── Section Headers ────────────────────────────────── */
.section-header {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    color: #00d4ff;
    text-transform: uppercase;
    letter-spacing: 3px;
    border-bottom: 1px solid rgba(0,212,255,0.3);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}

/* ─── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(2, 9, 24, 0.95);
    border-right: 1px solid rgba(0, 212, 255, 0.15);
}
section[data-testid="stSidebar"] .stRadio > label {
    color: #7ec8e3;
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
}

/* ─── Widgets ────────────────────────────────────────── */
.stSelectbox > label, .stSlider > label,
.stNumberInput > label, .stTextInput > label {
    color: #7ec8e3 !important;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 1px;
}
.stButton > button {
    background: linear-gradient(135deg, #7b2ff7, #00d4ff);
    color: white;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    border: none;
    border-radius: 8px;
    padding: 0.8rem 2rem;
    font-size: 1rem;
    letter-spacing: 2px;
    width: 100%;
    transition: all 0.3s ease;
    box-shadow: 0 0 20px rgba(123, 47, 247, 0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
}

/* ─── Tabs ───────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0, 212, 255, 0.05);
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] {
    color: #7ec8e3;
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff;
    background: rgba(0, 212, 255, 0.1);
}

/* ─── DataFrames ─────────────────────────────────────── */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* ─── Expander ───────────────────────────────────────── */
.streamlit-expanderHeader {
    font-family: 'Rajdhani', sans-serif;
    color: #7ec8e3;
}

/* ─── Divider ────────────────────────────────────────── */
hr { border-color: rgba(0,212,255,0.15); }

/* ─── Plotly charts transparent bg ──────────────────── */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_process():
    """Load, clean and feature-engineer the dataset (cached)."""
    df_raw  = load_data(DATA_PATH)
    df_clean = clean_data(df_raw)
    df_clean = create_target(df_clean)
    df_feat  = feature_engineering(df_clean)
    return df_raw, df_feat


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained model artefact (cached)."""
    if not os.path.exists(MODEL_PATH):
        st.error("⚠️ Model not found. Run `python train_model.py` first.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_meta():
    if os.path.exists(META_PATH):
        with open(META_PATH) as f:
            return json.load(f)
    return {}


# ── load ──────────────────────────────────────────────────────────────────────
with st.spinner("🚀 Initialising mission control …"):
    df_raw, df = load_and_process()
    artefact   = load_model()
    meta       = load_meta()

model        = artefact['model']
encodings    = artefact['encodings']
feature_cols = artefact['feature_cols']
all_results  = artefact.get('all_results', {})
feat_imp     = artefact.get('feat_importance', {})
lists        = artefact.get('lists', {})

COMPANIES = lists.get('companies', sorted(df['Company Name'].dropna().unique().tolist()) if 'Company Name' in df.columns else ['SpaceX'])
LOCATIONS  = lists.get('locations',  sorted(df['Location'].dropna().unique().tolist())   if 'Location'     in df.columns else ['Cape Canaveral'])
ROCKETS    = lists.get('rockets',    sorted(df['Detail'].dropna().unique().tolist())     if 'Detail'       in df.columns else ['Falcon 9'])

# Prediction history stored in session state
if 'pred_history' not in st.session_state:
    st.session_state.pred_history = []


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTLY LAYOUT DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════

PLOT_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor ='rgba(0,0,0,0)',
    font         =dict(color='#7ec8e3', family='Rajdhani'),
    title_font   =dict(color='#00d4ff', family='Orbitron', size=14),
    legend       =dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#7ec8e3')),
    margin       =dict(l=30, r=30, t=50, b=30),
    xaxis        =dict(gridcolor='rgba(0,212,255,0.1)', linecolor='rgba(0,212,255,0.2)'),
    yaxis        =dict(gridcolor='rgba(0,212,255,0.1)', linecolor='rgba(0,212,255,0.2)'),
)

COLORS = ['#00d4ff', '#7b2ff7', '#ff6b35', '#00ff88', '#ffd700', '#ff4141']


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-family:Orbitron; font-size:1.4rem; font-weight:900;
                    background:linear-gradient(90deg,#00d4ff,#7b2ff7);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            🚀 MISSION CTRL
        </div>
        <div style='color:#7ec8e3; font-size:0.75rem; letter-spacing:3px;'>SPACE PREDICTOR v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()



    page = st.radio(   
        "Navigation",
        ["🎯 Predict Mission", "📊 Data Explorer", "🤖 Model Analysis", "📋 Prediction History"],
        label_visibility="collapsed",
    )

    st.divider()

    # ── Quick model stats ──────────────────────────────
    st.markdown("### 📈 Model Stats")
    m = artefact.get('metrics', {})
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Accuracy",  f"{m.get('accuracy',  0)*100:.1f}%")
        st.metric("Precision", f"{m.get('precision', 0)*100:.1f}%")
    with c2:
        st.metric("F1 Score",  f"{m.get('f1',        0)*100:.1f}%")
        st.metric("ROC-AUC",   f"{m.get('roc_auc',   0)*100:.1f}%")

    st.divider()

    # ── Dataset overview ───────────────────────────────
    st.markdown("### 📂 Dataset Overview")
    total   = len(df)
    success = int(df['target'].sum())
    st.write(f"**Total Missions:** {total:,}")
    st.write(f"**Successful:** {success:,}  ({success/total*100:.1f}%)")
    st.write(f"**Failed:** {total-success:,}  ({(total-success)/total*100:.1f}%)")
    st.write(f"**Best Model:** {artefact.get('best_model_name','—')}")
    st.write(f"**Features:** {len(feature_cols)}")

    st.divider()
    st.caption("Built with ❤️  by AI | Powered by scikit-learn")


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown('<div class="hero-title">Manishankar Space Mission Prediction APP</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered Mission Success Intelligence</div>', unsafe_allow_html=True)

# Quick KPI bar
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(df):,}</div>
        <div class="metric-label">Total Missions</div>
    </div>""", unsafe_allow_html=True)
with k2:
    sr = int(df['target'].sum())
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{sr/len(df)*100:.0f}%</div>
        <div class="metric-label">Success Rate</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{m.get('accuracy',0)*100:.1f}%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{len(COMPANIES)}</div>
        <div class="metric-label">Space Agencies</div>
    </div>""", unsafe_allow_html=True)

st.write("")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT MISSION
# ═══════════════════════════════════════════════════════════════════════════════

if "🎯 Predict Mission" in page:

    st.markdown('<div class="section-header">🎯 Mission Configuration</div>', unsafe_allow_html=True)

    col_inp, col_out = st.columns([1, 1], gap="large")

    with col_inp:
        with st.container():
            company      = st.selectbox("🏢 Company / Agency",   COMPANIES)
            location     = st.selectbox("📍 Launch Location",    LOCATIONS)
            rocket_name  = st.selectbox("🚀 Rocket / Vehicle",   ROCKETS)
            rocket_status = st.radio("🔧 Rocket Status", ["StatusActive", "StatusRetired"], horizontal=True)
            rocket_cost  = st.slider("💰 Rocket Cost (Million USD)", 0.0, 1000.0, 100.0, step=5.0)
            launch_year  = st.slider("📅 Launch Year", 1957, 2030, 2020)
            launch_month = st.slider("📆 Launch Month", 1, 12, 6)

        predict_btn = st.button("🚀 PREDICT MISSION SUCCESS")

    with col_out:
        if predict_btn:
            # Build input row
            row = build_input_row(
                company, location, rocket_name, rocket_status,
                rocket_cost, launch_year, launch_month, encodings
            )

            # Align columns
            for c in feature_cols:
                if c not in row.columns:
                    row[c] = 0
            row = row[feature_cols]

            # Predict
            pred      = model.predict(row)[0]
            proba     = model.predict_proba(row)[0]
            success_p = proba[1]
            failure_p = proba[0]

            # Display result
            if pred == 1:
                st.markdown(f"""<div class="success-banner">
                    <div class="banner-emoji">✅</div>
                    <div class="banner-title">MISSION SUCCESS</div>
                    <div style="color:#7ec8e3; font-size:1rem; margin-top:0.5rem;">
                        High probability of successful launch
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="failure-banner">
                    <div class="banner-emoji">❌</div>
                    <div class="banner-title">MISSION FAILURE</div>
                    <div style="color:#7ec8e3; font-size:1rem; margin-top:0.5rem;">
                        Risk factors detected — review parameters
                    </div>
                </div>""", unsafe_allow_html=True)

            st.write("")

            # Probability bars
            st.markdown("**Mission Probability Analysis**")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("✅ Success Probability", f"{success_p*100:.1f}%")
                st.progress(float(success_p))
            with c2:
                st.metric("❌ Failure Probability", f"{failure_p*100:.1f}%")
                st.progress(float(failure_p))

            # Confidence gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=success_p * 100,
                delta={'reference': 60, 'valueformat': '.1f'},
                title={'text': "Success Confidence", 'font': {'color': '#00d4ff', 'family': 'Orbitron', 'size': 14}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#7ec8e3'},
                    'bar': {'color': '#00d4ff'},
                    'bgcolor': 'rgba(0,0,0,0)',
                    'bordercolor': 'rgba(0,212,255,0.3)',
                    'steps': [
                        {'range': [0, 40],   'color': 'rgba(255,65,65,0.3)'},
                        {'range': [40, 70],  'color': 'rgba(255,215,0,0.3)'},
                        {'range': [70, 100], 'color': 'rgba(0,255,136,0.3)'},
                    ],
                    'threshold': {'line': {'color': '#ff6b35', 'width': 4}, 'thickness': 0.75, 'value': 60},
                },
                number={'font': {'color': '#00d4ff', 'family': 'Orbitron'}, 'suffix': '%'},
            ))
            fig_gauge.update_layout(**PLOT_LAYOUT, height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Save to history
            st.session_state.pred_history.append({
                'timestamp':     datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'company':       company,
                'location':      location,
                'rocket':        rocket_name,
                'rocket_status': rocket_status,
                'cost_m':        rocket_cost,
                'year':          launch_year,
                'month':         launch_month,
                'prediction':    'Success' if pred == 1 else 'Failure',
                'success_prob':  round(float(success_p), 4),
                'failure_prob':  round(float(failure_p), 4),
            })

        else:
            # Placeholder info
            st.info("👈 Configure mission parameters on the left and click **PREDICT MISSION SUCCESS**.")
            st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/NASA_logo.svg/1224px-NASA_logo.svg.png",
                     width=120, caption="Space Mission Predictor")

    # ── Bulk CSV prediction ────────────────────────────────────────────────────
    st.divider()
    st.markdown('<div class="section-header">📁 Bulk Prediction — Upload CSV</div>', unsafe_allow_html=True)

    with st.expander("⬆️ Upload a CSV for batch predictions"):
        st.write("Upload a CSV with columns: `Company Name`, `Location`, `Detail`, `Status Rocket`, `Rocket` (cost), `Datum` (date)")
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
        if uploaded:
            try:
                from utils import load_data as _ld, clean_data as _cd, feature_engineering as _fe, create_target as _ct
                bulk_df = pd.read_csv(uploaded)
                bulk_df.columns = bulk_df.columns.str.strip()
                bulk_clean = _cd(bulk_df)

                # Add dummy target for FE (unknown)
                if 'Status Mission' not in bulk_clean.columns:
                    bulk_clean['Status Mission'] = 'Unknown'
                bulk_clean = _ct(bulk_clean)
                bulk_feat  = _fe(bulk_clean)

                bulk_X = bulk_feat[[c for c in feature_cols if c in bulk_feat.columns]]
                for c in feature_cols:
                    if c not in bulk_X.columns:
                        bulk_X[c] = 0
                bulk_X = bulk_X[feature_cols]

                preds = model.predict(bulk_X)
                probs = model.predict_proba(bulk_X)

                bulk_df['Predicted']       = ['Success' if p == 1 else 'Failure' for p in preds]
                bulk_df['Success_Prob_%']  = (probs[:, 1] * 100).round(1)
                bulk_df['Failure_Prob_%']  = (probs[:, 0] * 100).round(1)

                st.success(f"✅ Predictions done for {len(bulk_df)} rows!")
                st.dataframe(bulk_df.head(50))

                csv_bytes = bulk_df.to_csv(index=False).encode()
                st.download_button("⬇️ Download Predictions CSV", csv_bytes,
                                   "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════

elif "📊 Data Explorer" in page:

    st.markdown('<div class="section-header">📊 Space Mission Data Explorer</div>', unsafe_allow_html=True)

    tabs = st.tabs(["Overview", "Success vs Failure", "Company Analysis", "Year Trends", "Cost Analysis"])

    # ── Tab 1 – Overview ──────────────────────────────────────────────────────
    with tabs[0]:
        st.dataframe(df_raw.head(100), use_container_width=True, height=300)
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Shape:**", df_raw.shape)
            st.write("**Columns:**", list(df_raw.columns))
        with c2:
            st.write("**Missing values per column:**")
            mv = df_raw.isnull().sum().reset_index()
            mv.columns = ['Column', 'Missing']
            st.dataframe(mv[mv['Missing'] > 0], use_container_width=True, hide_index=True)

    # ── Tab 2 – Success vs Failure ────────────────────────────────────────────
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            vc = df['target'].value_counts().reset_index()
            vc['Status'] = vc['target'].map({1: 'Success', 0: 'Failure'})
            fig_pie = px.pie(vc, names='Status', values='count',
                             color='Status',
                             color_discrete_map={'Success':'#00ff88','Failure':'#ff4141'},
                             title='Mission Outcome Distribution')
            fig_pie.update_layout(**PLOT_LAYOUT, height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            if 'Status Mission' in df.columns:
                sm_vc = df['Status Mission'].value_counts().reset_index()
                sm_vc.columns = ['Status', 'Count']
                fig_bar = px.bar(sm_vc, x='Status', y='Count',
                                 color='Status', color_discrete_sequence=COLORS,
                                 title='Mission Status Breakdown')
                fig_bar.update_layout(**PLOT_LAYOUT, height=350, showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

    # ── Tab 3 – Company Analysis ──────────────────────────────────────────────
    with tabs[2]:
        if 'Company Name' in df.columns:
            top_n = st.slider("Top N Companies", 5, 20, 10)
            comp_df = df.groupby('Company Name').agg(
                total   =('target','count'),
                success =('target','sum')
            ).reset_index()
            comp_df['success_rate'] = (comp_df['success'] / comp_df['total'] * 100).round(1)
            comp_df = comp_df.nlargest(top_n, 'total')

            fig_comp = px.bar(comp_df, x='Company Name', y='total',
                              color='success_rate', color_continuous_scale='Blues',
                              title=f'Top {top_n} Companies — Total Launches vs Success Rate',
                              labels={'total':'Total Launches','success_rate':'Success Rate %'})
            fig_comp.update_layout(**PLOT_LAYOUT, height=400)
            st.plotly_chart(fig_comp, use_container_width=True)

            fig_scatter = px.scatter(comp_df, x='total', y='success_rate',
                                     size='total', text='Company Name',
                                     color_discrete_sequence=['#00d4ff'],
                                     title='Launches vs Success Rate (bubble = total launches)')
            fig_scatter.update_layout(**PLOT_LAYOUT, height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Tab 4 – Year Trends ───────────────────────────────────────────────────
    with tabs[3]:
        if 'launch_year' in df.columns:
            yr_df = df.groupby('launch_year').agg(
                total  =('target','count'),
                success=('target','sum')
            ).reset_index()
            yr_df['success_rate'] = (yr_df['success'] / yr_df['total'] * 100).round(1)

            fig_yr = make_subplots(specs=[[{"secondary_y": True}]])
            fig_yr.add_trace(go.Bar(x=yr_df['launch_year'], y=yr_df['total'],
                                    name='Total Launches', marker_color='rgba(0,212,255,0.4)'), secondary_y=False)
            fig_yr.add_trace(go.Scatter(x=yr_df['launch_year'], y=yr_df['success_rate'],
                                        name='Success Rate %', line=dict(color='#00ff88', width=2),
                                        mode='lines+markers'), secondary_y=True)
            fig_yr.update_layout(**PLOT_LAYOUT, title='Launches & Success Rate by Year', height=400)
            fig_yr.update_yaxes(title_text="Total Launches",   secondary_y=False, gridcolor='rgba(0,212,255,0.1)')
            fig_yr.update_yaxes(title_text="Success Rate (%)", secondary_y=True,  gridcolor='rgba(0,212,255,0.1)')
            st.plotly_chart(fig_yr, use_container_width=True)

    # ── Tab 5 – Cost Analysis ─────────────────────────────────────────────────
    with tabs[4]:
        if 'rocket_cost' in df.columns:
            cost_df = df[df['rocket_cost'].notna() & (df['rocket_cost'] > 0)].copy()
            cost_df['Outcome'] = cost_df['target'].map({1:'Success',0:'Failure'})
            fig_box = px.box(cost_df, x='Outcome', y='rocket_cost',
                             color='Outcome',
                             color_discrete_map={'Success':'#00ff88','Failure':'#ff4141'},
                             title='Rocket Cost by Mission Outcome (Million USD)')
            fig_box.update_layout(**PLOT_LAYOUT, height=400)
            st.plotly_chart(fig_box, use_container_width=True)

            fig_hist = px.histogram(cost_df, x='rocket_cost', color='Outcome',
                                    color_discrete_map={'Success':'#00ff88','Failure':'#ff4141'},
                                    nbins=50, barmode='overlay', opacity=0.7,
                                    title='Rocket Cost Distribution')
            fig_hist.update_layout(**PLOT_LAYOUT, height=350)
            st.plotly_chart(fig_hist, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

elif "🤖 Model Analysis" in page:

    st.markdown('<div class="section-header">🤖 Model Performance Analysis</div>', unsafe_allow_html=True)

    # ── Model comparison table ────────────────────────────────────────────────
    st.markdown("### 📋 Model Comparison")
    rows = []
    for name, met in all_results.items():
        rows.append({
            'Model':     name,
            'Accuracy':  f"{met['accuracy']*100:.1f}%",
            'Precision': f"{met['precision']*100:.1f}%",
            'Recall':    f"{met['recall']*100:.1f}%",
            'F1 Score':  f"{met['f1']*100:.1f}%",
            'ROC-AUC':   f"{met.get('roc_auc',0)*100:.1f}%",
        })
    cmp_df = pd.DataFrame(rows)
    st.dataframe(cmp_df, use_container_width=True, hide_index=True)

    # ── Bar comparison ────────────────────────────────────────────────────────
    metrics_list = ['accuracy','precision','recall','f1']
    fig_cmp = go.Figure()
    for i, met_name in enumerate(metrics_list):
        vals = [all_results[n].get(met_name, 0)*100 for n in all_results]
        fig_cmp.add_trace(go.Bar(
            name=met_name.capitalize(),
            x=list(all_results.keys()),
            y=vals,
            marker_color=COLORS[i],
            opacity=0.85,
        ))
    fig_cmp.update_layout(**PLOT_LAYOUT, barmode='group',
                          title='Model Comparison — All Metrics', height=400)
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    if feat_imp:
        st.markdown("### 🔍 Feature Importance")
        fi_df = pd.DataFrame(list(feat_imp.items()), columns=['Feature','Importance'])
        fi_df = fi_df.sort_values('Importance', ascending=True)
        fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                        color='Importance', color_continuous_scale='Blues',
                        title='Feature Importance')
        fig_fi.update_layout(**PLOT_LAYOUT, height=400)
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    st.markdown("### 🧮 Confusion Matrix (Best Model)")
    cm = artefact.get('metrics', {}).get('confusion_matrix')
    if cm:
        cm = np.array(cm)
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Failure','Success'], y=['Failure','Success'],
            color_continuous_scale='Blues', text_auto=True,
            title='Confusion Matrix',
        )
        fig_cm.update_layout(**PLOT_LAYOUT, height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── SHAP (optional / best-effort) ─────────────────────────────────────────
    with st.expander("🔬 SHAP Explainability (experimental)"):
        try:
            import shap
            from sklearn.pipeline import Pipeline as SKPipeline

            st.info("Computing SHAP values … this may take a moment.")
            raw_model = model
            if isinstance(raw_model, SKPipeline):
                raw_model = raw_model.named_steps.get('clf', raw_model)

            from utils import load_data as _ld, clean_data as _cd, create_target as _ct, feature_engineering as _fe, get_features_target as _gft
            _X, _y, _ = __import__('utils').preprocess_pipeline(DATA_PATH)
            sample = _X.sample(min(200, len(_X)), random_state=42)

            if hasattr(raw_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(raw_model)
                shap_values = explainer.shap_values(sample)
                if isinstance(shap_values, list):
                    sv = shap_values[1]
                else:
                    sv = shap_values
                shap_df = pd.DataFrame({
                    'Feature': sample.columns,
                    'Mean |SHAP|': np.abs(sv).mean(axis=0)
                }).sort_values('Mean |SHAP|', ascending=True)
                fig_shap = px.bar(shap_df, x='Mean |SHAP|', y='Feature', orientation='h',
                                  color='Mean |SHAP|', color_continuous_scale='Purples',
                                  title='SHAP Feature Importance')
                fig_shap.update_layout(**PLOT_LAYOUT, height=400)
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.warning("SHAP TreeExplainer requires a tree-based model.")
        except Exception as e:
            st.warning(f"SHAP not available for this model type: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

elif "📋 Prediction History" in page:

    st.markdown('<div class="section-header">📋 Prediction History</div>', unsafe_allow_html=True)

    if not st.session_state.pred_history:
        st.info("No predictions made yet. Go to 🎯 Predict Mission to get started.")
    else:
        hist_df = pd.DataFrame(st.session_state.pred_history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        # Download
        csv_bytes = hist_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download History CSV", csv_bytes, "prediction_history.csv", "text/csv")

        # Success/Failure distribution of user predictions
        if len(hist_df) >= 2:
            vc = hist_df['prediction'].value_counts().reset_index()
            vc.columns = ['Prediction', 'Count']
            fig_h = px.pie(vc, names='Prediction', values='Count',
                           color='Prediction',
                           color_discrete_map={'Success':'#00ff88','Failure':'#ff4141'},
                           title='Your Prediction Summary')
            fig_h.update_layout(**PLOT_LAYOUT, height=300)
            st.plotly_chart(fig_h, use_container_width=True)

        if st.button("🗑️ Clear History"):
            st.session_state.pred_history = []
            st.rerun()
