from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from schema import PatientInput
import model as ml

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Stroke Prediction API",
    description = "Predicts stroke risk from patient clinical data using XGBoost + SHAP",
    version     = "1.0.0",
)

# ── CORS — allows Streamlit frontend to call this API ────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
def root():
    return {
        "message" : "Stroke Prediction API is running",
        "docs"    : "/docs",
        "health"  : "/health",
        "predict" : "/predict"
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status"    : "ok",
        "model"     : str(ml.bundle["model_name"]),
        "threshold" : float(round(ml.THRESHOLD, 4)),
        "roc_auc"   : float(ml.bundle["best_roc_auc"]),
        "features"  : int(len(ml.feature_columns)),
    }

@app.post("/predict", tags=["Prediction"])
def predict(patient: PatientInput):
    try:
        result = ml.predict(patient)
        return {
            "status"  : "success",
            "input"   : patient.model_dump(),
            "result"  : result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))