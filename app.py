import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# 1. Page Configuration & Custom Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="EcoBio Brick AI Predictor",
    page_icon="🧱",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .stApp {
        background: radial-gradient(ellipse at top left, #172554 0%, #0B1120 42%, #080C16 100%);
        color: #E2E8F0;
    }
    [data-testid="stHeader"] {
        background-color: rgba(8, 12, 22, 0.75);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #151D2D 0%, #0D1422 100%);
        border-right: 1px solid #263449;
    }
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: #A8B5C9;
    }
    .main-header {
        font-size: clamp(1.6rem, 3vw, 2.2rem);
        font-weight: 700;
        color: #F1F5F9;
        line-height: 1.25;
        overflow-wrap: anywhere;
        margin-bottom: 1rem;
        text-shadow: 0 0 24px rgba(56, 189, 248, 0.22);
    }
    h1, h2, h3 {
        color: #F1F5F9 !important;
    }
    [data-testid="stMetric"] {
        box-sizing: border-box;
        width: 100%;
        min-width: 0;
        background: linear-gradient(145deg, rgba(24, 37, 58, 0.96), rgba(15, 25, 42, 0.96));
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        border: 1px solid #2C405A;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.24);
        margin: 0.75rem 0;
        transition: transform 160ms ease, border-color 160ms ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #38BDF8;
    }
    [data-testid="stMetricLabel"] {
        color: #A8B5C9 !important;
        font-size: 1rem;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-size: clamp(1.5rem, 3vw, 2rem);
        font-weight: 700;
        overflow-wrap: anywhere;
    }
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input {
        background-color: #0B1220;
        color: #F1F5F9;
        border: 1px solid #33445C;
        border-radius: 8px;
    }
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus {
        border-color: #38BDF8;
        box-shadow: 0 0 0 1px #38BDF8;
    }
    [data-testid="stNumberInput"] button {
        color: #CBD5E1;
    }
    [data-testid="stAlert"] {
        border-radius: 10px;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {
        color: #E2E8F0 !important;
    }
    @media (max-width: 600px) {
        [data-testid="stMetric"] {
            padding: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-header'>🧱 EcoBio Brick AI — Mix Design Optimizer</div>", unsafe_allow_html=True)
st.caption("Adjust the mix proportions in the sidebar to see updated predictions.")

# ---------------------------------------------------------
# 2. Load Model Artifacts
# ---------------------------------------------------------
@st.cache_resource
def load_models():
    strength_model = joblib.load("model_compressive_strength.pkl")
    absorption_model = joblib.load("model_water_absorption.pkl")
    return strength_model, absorption_model

try:
    model_strength, model_absorption = load_models()
except Exception as e:
    st.error(f"⚠️ Could not load pickle files: {e}")
    st.info("Ensure 'model_compressive_strength.pkl' and 'model_water_absorption.pkl' are in the same folder as app.py.")
    st.stop()

# Feature Columns expected by the models
FEATURE_COLS = [
    "GGBS (%)", "Fly Ash (%)", "Crush Sand (%)", "Marble Dust (%)",
    "Rice Husk (%)", "Coconut Fibre (%)", "Glass Fibre (%)",
    "Lime (%)", "Jaggery (%)", "W/B Ratio"
]

# ---------------------------------------------------------
# 3. Sidebar Controls (Direct Numeric Inputs)
# ---------------------------------------------------------
st.sidebar.header("🎛️ Input Mix Proportions")
st.sidebar.caption("Type in values or adjust using step buttons:")

# Users can type exact values into these fields
ggbs = st.sidebar.number_input("GGBS (%)", min_value=0.0, max_value=50.0, value=20.0, step=0.1, format="%.2f")
fly_ash = st.sidebar.number_input("Fly Ash (%)", min_value=0.0, max_value=50.0, value=15.0, step=0.1, format="%.2f")
crush_sand = st.sidebar.number_input("Crush Sand (%)", min_value=0.0, max_value=60.0, value=35.0, step=0.1, format="%.2f")
marble_dust = st.sidebar.number_input("Marble Dust (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.1, format="%.2f")
rice_husk = st.sidebar.number_input("Rice Husk (%)", min_value=0.0, max_value=15.0, value=2.0, step=0.1, format="%.2f")
coconut_fibre = st.sidebar.number_input("Coconut Fibre (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.05, format="%.2f")
glass_fibre = st.sidebar.number_input("Glass Fibre (%)", min_value=0.0, max_value=5.0, value=0.5, step=0.05, format="%.2f")
lime = st.sidebar.number_input("Lime (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1, format="%.2f")
jaggery = st.sidebar.number_input("Jaggery (%)", min_value=0.0, max_value=5.0, value=0.2, step=0.05, format="%.2f")
wb_ratio = st.sidebar.number_input("Water / Binder (W/B) Ratio", min_value=0.20, max_value=0.60, value=0.38, step=0.01, format="%.2f")

# Monitor Mix Balance
total_solid_mix = ggbs + fly_ash + crush_sand + marble_dust + rice_husk + coconut_fibre + glass_fibre + lime + jaggery

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Total Solids Mix:** `{total_solid_mix:.2f}%`")
if not (98.0 <= total_solid_mix <= 102.0):
    st.sidebar.warning("⚠️ Total solids percentage usually sums close to 100%.")

# ---------------------------------------------------------
# 4. Model Inference
# ---------------------------------------------------------
input_df = pd.DataFrame([[
    ggbs, fly_ash, crush_sand, marble_dust, rice_husk,
    coconut_fibre, glass_fibre, lime, jaggery, wb_ratio
]], columns=FEATURE_COLS)

pred_strength = model_strength.predict(input_df)[0]
pred_absorption = model_absorption.predict(input_df)[0]

# Good-range limits previously used for the app's strength and absorption grades.
GOOD_STRENGTH_MIN = 3.8
GOOD_ABSORPTION_MAX = 12.0

# ---------------------------------------------------------
# 5. Live Predictions Display
# ---------------------------------------------------------
st.subheader("🎯 Live Predictions")
st.metric(
    label="Predicted Compressive Strength",
    value=f"{pred_strength:.3f} N/mm²"
)
st.caption(f"Good range: ≥ {GOOD_STRENGTH_MIN:.1f} N/mm²")
if pred_strength >= GOOD_STRENGTH_MIN:
    st.success("✅ Compressive strength is in the good range.")
else:
    st.warning("⚠️ Compressive strength is below the good range.")

st.metric(
    label="Predicted Water Absorption",
    value=f"{pred_absorption:.3f} %"
)
st.caption(f"Good range: ≤ {GOOD_ABSORPTION_MAX:.1f}%")
if pred_absorption <= GOOD_ABSORPTION_MAX:
    st.success("✅ Water absorption is in the good range.")
else:
    st.warning("⚠️ Water absorption is above the good range.")