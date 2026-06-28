import streamlit as st
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import shap
import joblib
from pathlib import Path

#Page config
st.set_page_config(
    page_title = "Stroke Risk Predictor",
    page_icon  = "🧠",
    layout     = "wide"
)

#Constants
API_URL    = "http://localhost:8000/predict"
HEALTH_URL = "http://localhost:8000/health"

#Custom CSS 
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high   { background: #fff0f0; border: 2px solid #e74c3c; }
    .risk-medium { background: #fff8e1; border: 2px solid #f39c12; }
    .risk-low    { background: #f0fff4; border: 2px solid #27ae60; }
    .risk-value  { font-size: 3rem; font-weight: 700; }
    .risk-high   .risk-value { color: #e74c3c; }
    .risk-medium .risk-value { color: #f39c12; }
    .risk-low    .risk-value { color: #27ae60; }
    .risk-label  { font-size: 1.4rem; font-weight: 600; margin-top: 0.3rem; }
    .info-box {
        background: #f8f9fa;
        border-left: 4px solid #4C9BE8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.9rem;
        color: #444;
    }
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .stButton > button {
        width: 100%;
        background: #4C9BE8;
        color: white;
        border: none;
        padding: 0.7rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: #2980b9;
    }
</style>
""", unsafe_allow_html=True)


#Header
st.markdown('<p class="main-title">🧠 Stroke Risk Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter patient clinical data to assess stroke risk using XGBoost + SHAP explainability</p>', unsafe_allow_html=True)

#API Health check 
try:
    health = requests.get(HEALTH_URL, timeout=3).json()
    st.success(f"✅ Model online — {health['model']} | ROC-AUC: {health['roc_auc']:.4f} | Threshold: {health['threshold']:.4f}")
except:
    st.error("❌ API is offline. Please start the FastAPI backend: `uvicorn main:app --reload --port 8000`")
    st.stop()

st.divider()

#Sidebar-Input form
with st.sidebar:
    st.header("🩺 Patient Information")
    st.caption("Fill in all fields and click Predict")

    st.subheader("Demographics")
    age    = st.slider("Age", min_value=1, max_value=100, value=55, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married    = st.selectbox("Ever Married", ["Yes", "No"])
    work_type       = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Never_worked", "children"])
    Residence_type  = st.selectbox("Residence Type", ["Urban", "Rural"])

    st.subheader("Clinical Indicators")
    avg_glucose_level = st.slider("Average Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=0.5)
    bmi               = st.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    hypertension  = st.selectbox("Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

    st.subheader("Lifestyle")
    smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.divider()
    predict_btn = st.button("🔍 Predict Stroke Risk", type="primary")


col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("📋 Patient Summary")
    st.markdown(f"""
    <div class="info-box">👤 <b>Age:</b> {age} years | <b>Gender:</b> {gender}</div>
    <div class="info-box">💍 <b>Married:</b> {ever_married} | <b>Work:</b> {work_type}</div>
    <div class="info-box">🏠 <b>Residence:</b> {Residence_type} | <b>Smoking:</b> {smoking_status}</div>
    <div class="info-box">🩸 <b>Glucose:</b> {avg_glucose_level} mg/dL | <b>BMI:</b> {bmi}</div>
    <div class="info-box">❤️ <b>Hypertension:</b> {hypertension} | <b>Heart Disease:</b> {heart_disease}</div>
    """, unsafe_allow_html=True)

    # Clinical reference ranges
    st.subheader("📊 Reference Ranges")
    glucose_status = "⚠️ High" if avg_glucose_level > 140 else ("✅ Normal" if avg_glucose_level < 100 else "⚡ Pre-diabetic")
    bmi_status     = "⚠️ Obese" if bmi > 30 else ("✅ Normal" if 18.5 <= bmi <= 25 else "⚡ Overweight")

    st.metric("Glucose Level", f"{avg_glucose_level} mg/dL", glucose_status)
    st.metric("BMI", f"{bmi}", bmi_status)

with col2:
    if predict_btn:
        # Build payload
        payload = {
            "age"               : float(age),
            "hypertension"      : 1 if hypertension == "Yes" else 0,
            "heart_disease"     : 1 if heart_disease == "Yes" else 0,
            "ever_married"      : 1 if ever_married == "Yes" else 0,
            "avg_glucose_level" : float(avg_glucose_level),
            "bmi"               : float(bmi),
            "gender"            : gender,
            "work_type"         : work_type,
            "Residence_type"    : Residence_type,
            "smoking_status"    : smoking_status,
        }

        with st.spinner("Analysing patient data..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                data   = response.json()
                result = data["result"]

                risk_score   = result["risk_score"]
                risk_percent = result["risk_percent"]
                risk_label   = result["risk_label"]
                shap_values  = result["shap_values"]
                base_value   = result["base_value"]

                #Risk card
                css_class = {
                    "High"  : "risk-high",
                    "Medium": "risk-medium",
                    "Low"   : "risk-low"
                }[risk_label]

                emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[risk_label]

                st.markdown(f"""
                <div class="risk-card {css_class}">
                    <div class="risk-value">{risk_percent}%</div>
                    <div class="risk-label">{emoji} {risk_label} Risk</div>
                    <div style="color:#888; font-size:0.85rem; margin-top:0.5rem;">
                        Threshold: {result['threshold_used']} | Score: {risk_score}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if risk_label == "High":
                    st.error("⚠️ **High stroke risk detected.** Immediate clinical evaluation recommended.")
                elif risk_label == "Medium":
                    st.warning("⚡ **Moderate stroke risk.** Lifestyle modifications and regular monitoring advised.")
                else:
                    st.success("✅ **Low stroke risk.** Continue healthy lifestyle practices.")

                st.divider()

                #SHAP waterfall chart
                st.subheader("🔍 SHAP Feature Explanation")
                st.caption("Red bars increase stroke risk · Blue bars decrease stroke risk")

                features = list(shap_values.keys())
                values   = list(shap_values.values())

                # Sort by absolute value
                sorted_pairs = sorted(zip(features, values), key=lambda x: abs(x[1]))
                features_sorted = [p[0] for p in sorted_pairs]
                values_sorted   = [p[1] for p in sorted_pairs]

                colors = ["#DC4545" if v > 0 else "#4C9BE8" for v in values_sorted]

                fig, ax = plt.subplots(figsize=(7, 5))
                bars = ax.barh(features_sorted, values_sorted, color=colors, edgecolor='white', height=0.6)
                ax.axvline(0, color='black', linewidth=0.8, linestyle='-')
                ax.set_xlabel("SHAP Value (impact on stroke risk)", fontsize=10)
                ax.set_title(f"Top feature contributions\nBase value: {base_value:.3f}", fontsize=11)
                ax.tick_params(axis='y', labelsize=9)

                # Value labels 
                for bar, val in zip(bars, values_sorted):
                    ax.text(
                        val + (0.01 if val >= 0 else -0.01),
                        bar.get_y() + bar.get_height() / 2,
                        f"{val:+.3f}",
                        va='center',
                        ha='left' if val >= 0 else 'right',
                        fontsize=8,
                        color='#333'
                    )

                red_patch  = mpatches.Patch(color='#E85D4C', label='Increases risk')
                blue_patch = mpatches.Patch(color='#4C9BE8', label='Decreases risk')
                ax.legend(handles=[red_patch, blue_patch], fontsize=9, loc='lower right')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    else:
        # Placeholder
        st.info("👈 Fill in patient details in the sidebar and click **Predict Stroke Risk** to see results.")

        st.subheader("ℹ️ About this tool")
        st.markdown("""
        This tool uses a machine learning model trained on **5,110 patient records** to predict stroke risk.

        **Model details:**
        - Algorithm: XGBoost (Optuna hyperparameter tuned)
        - Resampling: SMOTE (handles 95/5 class imbalance)
        - Threshold: 0.41 (optimised for healthcare recall)
        - Stroke recall: **80%** (catches 40 out of 50 real stroke cases)

        **Key risk factors identified:**
        - 🔴 Age (strongest predictor)
        - 🔴 High glucose levels
        - 🔴 Hypertension
        - 🔴 Heart disease history

        **⚠️ Disclaimer:** This tool is for educational purposes only and should not replace professional medical diagnosis.
        """)

#Footer 
st.divider()
st.caption("Built with XGBoost · FastAPI · Streamlit · SHAP | Stroke Prediction ML Project")