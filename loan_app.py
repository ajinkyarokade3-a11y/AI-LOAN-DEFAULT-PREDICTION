import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Loan Default Risk Analyzer",
    page_icon="💳",
    layout="wide"
)

# --- UI Theme Styling ---
st.markdown(
    """
    <style>
    :root {
        --bg: #0d1117;
        --panel: #111827;
        --panel-2: #0f172a;
        --text: #e5e7eb;
        --muted: #9ca3af;
        --accent: #22c55e;
        --accent-2: #16a34a;
        --danger: #ef4444;
        --warning: #f59e0b;
    }

    .stApp {
        background: radial-gradient(1200px 800px at 15% 5%, #101827 0%, #0b101a 35%, #0a0f18 100%);
        color: var(--text);
    }

    div[data-testid="stVerticalBlock"] > div:has(div.stButton) {
        padding-top: 0.25rem;
    }

    .section-card-marker {
        display: none;
    }

    div[data-testid="stVerticalBlock"] > div:has(> .section-card-marker) {
        background: linear-gradient(145deg, rgba(17,24,39,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 1.25rem 1.25rem 0.75rem 1.25rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }

    .result-card {
        border-radius: 16px;
        padding: 1.1rem 1.25rem;
        border: 1px solid rgba(148, 163, 184, 0.2);
        box-shadow: 0 12px 28px rgba(0,0,0,0.3);
        font-weight: 600;
        font-size: 1.05rem;
    }

    .result-low {
        background: rgba(34, 197, 94, 0.15);
        border-color: rgba(34, 197, 94, 0.5);
        color: #d1fae5;
    }

    .result-high {
        background: rgba(239, 68, 68, 0.15);
        border-color: rgba(239, 68, 68, 0.5);
        color: #fee2e2;
    }

    .caption {
        color: var(--muted);
        font-size: 0.9rem;
    }

    button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent), var(--accent-2)) !important;
        border: none !important;
        color: #051b10 !important;
        border-radius: 999px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.6rem !important;
    }

    button[kind="secondary"] {
        border-radius: 999px !important;
    }

    input, textarea, select, div[data-baseweb="select"] > div {
        border-radius: 12px !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 0.75rem 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- 1. Load Trained Pipeline and Supporting Artifacts ---
# Prefer the full pipeline for consistent training and inference.
@st.cache_resource
def load_artifacts():
    try:
        pipeline = joblib.load(MODELS_DIR / "model_pipeline.pkl")
        return {"pipeline": pipeline, "legacy": None}
    except FileNotFoundError:
        # Fallback for backward compatibility if the new pipeline is not present.
        model = joblib.load(MODELS_DIR / "xgboost_model.pkl")
        features = joblib.load(MODELS_DIR / "model_features.pkl")
        scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        return {"pipeline": None, "legacy": (model, features, scaler)}
    except Exception as e:
        st.error(f"Error loading model or supporting files: {e}")
        return {"pipeline": None, "legacy": None}

artifacts = load_artifacts() or {"pipeline": None, "legacy": None}
pipeline = artifacts.get("pipeline")
legacy_artifacts = artifacts.get("legacy")

# --- 2. Streamlit Page Layout Setup ---
st.markdown("## Loan Default Risk Analyzer")
st.markdown('<div class="caption">AI-powered credit risk prediction system</div>', unsafe_allow_html=True)

st.write("")

# --- Session State Defaults ---
if "Credit_Score" not in st.session_state:
    st.session_state.Credit_Score = 700
if "income" not in st.session_state:
    st.session_state.income = 50000.0
if "loan_amount" not in st.session_state:
    st.session_state.loan_amount = 200000.0
if "term" not in st.session_state:
    st.session_state.term = 360
if "dtir1" not in st.session_state:
    st.session_state.dtir1 = 35.0
if "loan_purpose_label" not in st.session_state:
    st.session_state.loan_purpose_label = "Home Loan"
if "loan_limit_label" not in st.session_state:
    st.session_state.loan_limit_label = "Conforming Loan"


def apply_example():
    st.session_state.Credit_Score = 745
    st.session_state.income = 82000.0
    st.session_state.loan_amount = 250000.0
    st.session_state.term = 360
    st.session_state.dtir1 = 28.0
    st.session_state.loan_purpose_label = "Home Loan"
    st.session_state.loan_limit_label = "Conforming Loan"


# --- 3. Borrower Information Input Section ---
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="section-card-marker"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Financial Information")
    Credit_Score = st.slider(
        "Credit Score",
        min_value=300,
        max_value=850,
        help="Typical ranges: 300-579 (poor), 580-669 (fair), 670-739 (good), 740+ (excellent).",
        key="Credit_Score",
    )
    income = st.number_input(
        "Annual Income",
        min_value=0.0,
        step=1000.0,
        help="Enter the borrower's gross annual income.",
        key="income",
    )

    st.write("")
    st.markdown("### 🧾 Loan Details")
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=0.0,
        step=1000.0,
        help="Total principal requested.",
        key="loan_amount",
    )
    term = st.selectbox(
        "Term (months)",
        options=[120, 180, 240, 360],
        help="Choose the repayment duration in months.",
        key="term",
    )
    dtir1 = st.slider(
        "Debt-to-Income Ratio (DTI %)",
        min_value=0.0,
        max_value=100.0,
        help="Monthly debt payments divided by monthly income.",
        key="dtir1",
    )

    loan_purpose_map = {
        "Home Loan": "p1",
        "Education Loan": "p2",
        "Business Loan": "p3",
        "Personal Loan": "p4",
    }
    loan_purpose_label = st.selectbox(
        "Loan Purpose",
        options=list(loan_purpose_map.keys()),
        key="loan_purpose_label",
    )

    loan_limit_map = {
        "Conforming Loan": "cf",
        "Non-Conforming Loan": "ncf",
    }
    loan_limit_label = st.selectbox(
        "Loan Type",
        options=list(loan_limit_map.keys()),
        key="loan_limit_label",
    )

    st.write("")
    if st.button("Try Example", use_container_width=True):
        apply_example()

with right:
    st.markdown('<div class="section-card-marker"></div>', unsafe_allow_html=True)
    st.markdown("### 🧠 Prediction Results")
    st.write("")
    predict_clicked = st.button("Predict Risk", type="primary", use_container_width=True)

    if predict_clicked:
        if pipeline is None and legacy_artifacts is None:
            st.error("Model artifacts are not loaded properly. Cannot proceed.")
        else:
            with st.spinner("Analyzing risk..."):
                loan_purpose = loan_purpose_map[st.session_state.loan_purpose_label]
                loan_limit = loan_limit_map[st.session_state.loan_limit_label]

                # Build dictionary of raw user inputs
                input_dict = {
                    "Credit_Score": st.session_state.Credit_Score,
                    "income": st.session_state.income,
                    "loan_amount": st.session_state.loan_amount,
                    "term": st.session_state.term,
                    "dtir1": st.session_state.dtir1,
                    "loan_purpose": loan_purpose,
                    "loan_limit": loan_limit,
                }

                input_df = pd.DataFrame([input_dict])

                income = float(st.session_state.income)
                loan_amount = float(st.session_state.loan_amount)
                term_value = float(st.session_state.term)

                loan_to_income = loan_amount / income if income > 0 else np.inf
                emi = loan_amount / term_value if term_value > 0 else np.nan
                emi_to_income = emi / income if income > 0 else np.nan

                guardrail_triggered = loan_to_income > 5

                if guardrail_triggered:
                    prediction = 1
                    prediction_proba = 1.0
                elif pipeline is not None:
                    prediction = pipeline.predict(input_df)[0]
                    prediction_proba = pipeline.predict_proba(input_df)[0][1]
                else:
                    model, model_features, scaler = legacy_artifacts

                    # Convert categorical inputs to the same one-hot encoded format used during training
                    legacy_dict = {
                        "Credit_Score": st.session_state.Credit_Score,
                        "income": st.session_state.income,
                        "loan_amount": st.session_state.loan_amount,
                        "term": st.session_state.term,
                        "dtir1": st.session_state.dtir1,
                    }

                    if loan_purpose in ["p2", "p3", "p4"]:
                        legacy_dict[f"loan_purpose_{loan_purpose}"] = 1

                    if loan_limit == "ncf":
                        legacy_dict["loan_limit_ncf"] = 1

                    legacy_df = pd.DataFrame([legacy_dict])
                    legacy_df = legacy_df.reindex(columns=model_features, fill_value=0)

                    numeric_cols = ["Credit_Score", "income", "loan_amount", "term", "dtir1"]
                    legacy_df[numeric_cols] = scaler.transform(legacy_df[numeric_cols])

                    prediction = model.predict(legacy_df)[0]
                    prediction_proba = model.predict_proba(legacy_df)[0][1]

            risk_percentage = prediction_proba * 100
            st.metric(
                label="Default Probability",
                value=f"{risk_percentage:.2f}%"
            )

            if guardrail_triggered:
                st.warning("Loan amount is significantly higher than income.")

            if prediction == 1:
                st.markdown(
                    '<div class="result-card result-high">🚨 High Risk: Applicant is likely to default.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-card result-low">✅ Low Risk: Applicant is unlikely to default.</div>',
                    unsafe_allow_html=True,
                )

            st.write("")
            st.markdown("#### 🔎 Risk Insight")
            st.progress(int(risk_percentage))

            if risk_percentage < 30:
                st.success("Low Risk Borrower")
            elif risk_percentage <= 60:
                st.warning("Moderate Risk Borrower")
            else:
                st.error("High Risk Borrower")

            st.write("")
            metrics_left, metrics_right = st.columns(2)
            metrics_left.metric("Loan-to-Income Ratio", f"{loan_to_income:.2f}")
            metrics_right.metric("EMI-to-Income Ratio", f"{emi_to_income:.4f}")

            st.write("")
            st.markdown("#### 🧾 Risk Explanation")
            if guardrail_triggered:
                st.write(
                    "Rule-based override: loan-to-income ratio exceeds 5, which indicates elevated risk."
                )
            elif loan_to_income > 3:
                st.write("Higher loan-to-income ratio increases default risk.")
            elif st.session_state.dtir1 > 45:
                st.write("Debt-to-income ratio is relatively high, which raises risk.")
            else:
                st.write("Risk appears within typical ranges for the provided inputs.")

            with st.expander("Debug Info"):
                st.write(f"Loan-to-Income Ratio: {loan_to_income:.4f}")
                st.write(f"EMI: {emi:.4f}")
                st.write(f"EMI-to-Income Ratio: {emi_to_income:.6f}")
                if pipeline is not None:
                    engineered_df = pipeline.named_steps["feature_engineering"].transform(input_df)
                    preprocessed = pipeline.named_steps["preprocess"].transform(engineered_df)
                    if hasattr(preprocessed, "toarray"):
                        preprocessed = preprocessed.toarray()
                    preprocess_step = pipeline.named_steps["preprocess"]
                    if hasattr(preprocess_step, "get_feature_names_out"):
                        feature_names = preprocess_step.get_feature_names_out()
                    else:
                        feature_names = [f"f{i}" for i in range(preprocessed.shape[1])]
                    debug_df = pd.DataFrame(preprocessed, columns=feature_names)
                    st.write("Final Feature Vector:")
                    st.dataframe(debug_df, use_container_width=True)
    else:
        st.info("Run a prediction to view risk insights and probability.")
