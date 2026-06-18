# 🧠 Stroke Risk Prediction System

An end-to-end machine learning system that predicts stroke risk from patient clinical data, with real-time SHAP explainability — built as part of an ML Internship project.

---

## 📋 Overview

This project predicts a patient's stroke risk using clinical and lifestyle data such as age, glucose level, BMI, hypertension, and smoking status. It addresses a real-world healthcare ML challenge: **severe class imbalance** (only ~5% of patients in the dataset had a stroke), where naive models achieve high accuracy by simply ignoring the minority class.

The system is built as a two-part application:
- A **FastAPI** backend serving predictions and SHAP explanations via REST endpoints
- A **Streamlit** frontend providing an interactive clinical interface for entering patient data and visualising risk

---

## 🎯 Problem Statement

Healthcare datasets are often heavily imbalanced — stroke occurs in roughly 5% of patients. A model that always predicts "no stroke" would score 95% accuracy while being clinically useless, since it would miss every real stroke case. This project specifically addresses that challenge through:

- SMOTE-based class balancing
- Threshold optimisation tuned for recall (not accuracy)
- Scientific comparison of multiple resampling strategies

---

## 📊 Dataset

- **Source:** [Stroke Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Size:** 5,110 patient records, 11 original features
- **Target:** Binary stroke occurrence (0 = No, 1 = Yes)
- **Class distribution:** 95.1% No Stroke / 4.9% Stroke

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Imputed 201 missing BMI values using median imputation
- Fixed rare `gender = 'Other'` category (1 row) by replacing with mode
- One-hot encoded all categorical features
- Preserved `smoking_status = 'Unknown'` as its own valid category

### 2. Feature Engineering
Three additional features were engineered beyond the original 11:

| Feature | Description |
|---|---|
| `age_glucose` | Interaction term — captures elderly + diabetic high-risk group |
| `age_group` | Clinical age brackets (child, young_adult, middle_aged, senior, elderly) |
| `high_risk_comorbidity` | Binary flag for patients with both hypertension AND heart disease |

Final feature count after encoding: **26**

### 3. Handling Class Imbalance
Three resampling strategies were tested and compared:

| Strategy | ROC-AUC | Recall | Strokes Caught | False Alarms |
|---|---|---|---|---|
| **SMOTE** ✅ | 0.8201 | 0.80 | 40/50 | 234 |
| SMOTETomek | 0.8260 | 0.78 | 39/50 | 243 |
| ADASYN | 0.8130 | 0.76 | 38/50 | 232 |

**SMOTE was selected** — despite SMOTETomek's marginally higher AUC, SMOTE caught more real stroke cases, which matters more clinically than a 0.006 AUC difference.

### 4. Model Selection & Tuning

| Model | ROC-AUC | Stroke Recall | Strokes Caught (/50) |
|---|---|---|---|
| Logistic Regression | 0.840 | 0.04 | 2 |
| Random Forest | 0.757 | 0.04 | 2 |
| **XGBoost (Optuna tuned)** | **0.820** | **0.80** | **40** |

XGBoost was tuned using **Optuna** across 50 trials, optimising `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, and `scale_pos_weight`.

### 5. Threshold Optimisation
The default 0.5 classification threshold is inappropriate for imbalanced healthcare data. The decision threshold was tuned to **0.41**, prioritising recall over precision — since missing a real stroke is far more costly than a false alarm.

| Threshold | Stroke Recall | Strokes Caught |
|---|---|---|
| 0.50 (default) | 0.68 | 34/50 |
| **0.41 (optimised)** | **0.80** | **40/50** |

### 6. Explainability
SHAP (SHapley Additive exPlanations) was used to interpret individual predictions:
- **Global importance:** `age` is the dominant predictor, followed by smoking status, work type, and the engineered `age_glucose` feature
- **Local explanations:** Each prediction includes a SHAP waterfall breakdown showing exactly which factors increased or decreased that patient's risk

---

## 🏗️ Architecture

```
┌─────────────┐      HTTP POST       ┌──────────────┐      Loads       ┌─────────────┐
│  Streamlit  │ ───────────────────► │   FastAPI    │ ───────────────► │  model.pkl  │
│  Frontend   │ ◄─────────────────── │   Backend    │                  │ (XGBoost +  │
│  (port 8501)│   JSON + SHAP values │  (port 8000) │                  │  scaler)    │
└─────────────┘                      └──────────────┘                  └─────────────┘
```

---

## 📁 Project Structure

```
stroke-prediction/
├── data/                          # Datasets, plots, preprocessing artefacts
├── notebooks/
│   └── stroke_eda_model.ipynb     # Full EDA, training, and evaluation notebook
├── backend/
│   ├── main.py                    # FastAPI app & routes
│   ├── model.py                   # Preprocessing, prediction, SHAP logic
│   ├── schema.py                  # Pydantic request validation
│   └── model.pkl                  # Trained model bundle
├── frontend/
│   └── app.py                     # Streamlit UI
├── requirements.txt
└── README.md
```

---

## 🚀 How to Run

### 1. Setup
```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

pip install -r requirements.txt
```

### 2. Start the backend (Terminal 1)
```bash
cd backend
uvicorn main:app --reload --port 8000
```
API docs available at: `http://localhost:8000/docs`

### 3. Start the frontend (Terminal 2)
```bash
cd frontend
streamlit run app.py
```
App opens at: `http://localhost:8501`

---

## 🔌 API Reference

### `GET /health`
Returns model status and metadata.
```json
{
  "status": "ok",
  "model": "XGBoost (Optuna tuned)",
  "threshold": 0.4118,
  "roc_auc": 0.8201,
  "features": 26
}
```

### `POST /predict`
Accepts patient data, returns risk score and SHAP explanation.

**Request:**
```json
{
  "age": 67,
  "hypertension": 0,
  "heart_disease": 1,
  "ever_married": 1,
  "avg_glucose_level": 228.69,
  "bmi": 36.6,
  "gender": "Male",
  "work_type": "Private",
  "Residence_type": "Urban",
  "smoking_status": "formerly smoked"
}
```

**Response:**
```json
{
  "status": "success",
  "result": {
    "risk_score": 0.5292,
    "risk_percent": 52.92,
    "risk_label": "High",
    "threshold_used": 0.4118,
    "shap_values": { "...": "top 8 contributing features" },
    "base_value": 0.43
  }
}
```

---

## 📈 Results Summary

| Metric | Value |
|---|---|
| Model | XGBoost (Optuna tuned, 50 trials) |
| Resampling | SMOTE |
| Decision Threshold | 0.41 |
| ROC-AUC | 0.82 |
| Stroke Recall | 80% (40/50 patients caught) |
| Baseline Recall (Logistic Regression / Random Forest) | 4% (2/50 patients caught) |

---

## ⚠️ Disclaimer

This project is built for educational and internship purposes only. It is **not a certified medical diagnostic tool** and should not be used for actual clinical decision-making without further validation, regulatory approval, and supervision by qualified healthcare professionals.

---

## 🛠️ Tech Stack

`Python` · `XGBoost` · `scikit-learn` · `imbalanced-learn` · `Optuna` · `SHAP` · `FastAPI` · `Streamlit` · `Pandas` · `NumPy`

---

## 👤 Author

Atharva — B.Tech Information Technology, VESIT
ML Internship Project, 2026