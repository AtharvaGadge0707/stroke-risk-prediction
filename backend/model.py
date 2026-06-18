import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from schema import PatientInput

# ── Load model bundle once at startup ──────────────────────────────────────
BASE_DIR    = Path(__file__).parent
BUNDLE_PATH = BASE_DIR / "model.pkl"

bundle          = joblib.load(BUNDLE_PATH)
model           = bundle["model"]
scaler          = bundle["scaler"]
feature_columns = bundle["feature_columns"]
THRESHOLD       = bundle["threshold"]

# SHAP explainer — initialised once
explainer = shap.Explainer(model)

print(f"Model loaded: {bundle['model_name']}")
print(f"Threshold   : {THRESHOLD:.4f}")
print(f"Features    : {len(feature_columns)}")


# ── Preprocessing ───────────────────────────────────────────────────────────
def preprocess(patient: PatientInput) -> pd.DataFrame:
    """Convert PatientInput → scaled DataFrame matching training feature order."""

    raw = {
        "age"               : patient.age,
        "hypertension"      : patient.hypertension,
        "heart_disease"     : patient.heart_disease,
        "ever_married"      : patient.ever_married,
        "avg_glucose_level" : patient.avg_glucose_level,
        "bmi"               : patient.bmi,
    }

    # ── Engineered features ────────────────────────────────────────────────
    raw["age_glucose"]           = patient.age * patient.avg_glucose_level
    raw["high_risk_comorbidity"] = int(patient.hypertension == 1 and
                                       patient.heart_disease == 1)

    # ── Age group bins ─────────────────────────────────────────────────────
    age = patient.age
    if   age <= 18:  age_group = "child"
    elif age <= 40:  age_group = "young_adult"
    elif age <= 60:  age_group = "middle_aged"
    elif age <= 80:  age_group = "senior"
    else:            age_group = "elderly"

    # ── One-hot: gender ────────────────────────────────────────────────────
    raw["gender_Female"] = int(patient.gender == "Female")
    raw["gender_Male"]   = int(patient.gender == "Male")

    # ── One-hot: work_type ─────────────────────────────────────────────────
    for wt in ["Govt_job", "Never_worked", "Private", "Self-employed", "children"]:
        raw[f"work_type_{wt}"] = int(patient.work_type == wt)

    # ── One-hot: Residence_type ────────────────────────────────────────────
    raw["Residence_type_Rural"] = int(patient.Residence_type == "Rural")
    raw["Residence_type_Urban"] = int(patient.Residence_type == "Urban")

    # ── One-hot: smoking_status ────────────────────────────────────────────
    for ss in ["Unknown", "formerly smoked", "never smoked", "smokes"]:
        raw[f"smoking_status_{ss}"] = int(patient.smoking_status == ss)

    # ── One-hot: age_group ─────────────────────────────────────────────────
    for ag in ["child", "elderly", "middle_aged", "senior", "young_adult"]:
        raw[f"age_group_{ag}"] = int(age_group == ag)

    # ── Build DataFrame in exact training column order ─────────────────────
    df = pd.DataFrame([raw])[feature_columns]

    # ── Scale numeric features ─────────────────────────────────────────────
    # FIXED - only scale what the scaler was fitted on
    num_features = ["age", "avg_glucose_level", "bmi"]
    df[num_features] = scaler.transform(df[num_features])

    return df


# ── Prediction ───────────────────────────────────────────────────────────────
def predict(patient: PatientInput) -> dict:
    """Run prediction and return risk score, label, and SHAP values."""

    df       = preprocess(patient)
    prob     = float(model.predict_proba(df)[0][1])
    label    = "High" if prob >= THRESHOLD else ("Medium" if prob >= 0.20 else "Low")

    # ── SHAP explanation ───────────────────────────────────────────────────
    # NEW - explicitly convert all numpy types to native Python
    shap_vals = explainer(df)
    shap_dict = {
        col: round(float(shap_vals.values[0][i]), 4)
        for i, col in enumerate(feature_columns)
    }

    top_shap = dict(
        sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    )

    # Force all values to native Python types for JSON serialization
    return {
        "risk_score"    : float(round(prob, 4)),
        "risk_percent"  : float(round(prob * 100, 2)),
        "risk_label"    : str(label),
        "threshold_used": float(round(THRESHOLD, 4)),
        "shap_values"   : {k: float(v) for k, v in top_shap.items()},
        "base_value"    : float(round(float(shap_vals.base_values[0]), 4)),
    }