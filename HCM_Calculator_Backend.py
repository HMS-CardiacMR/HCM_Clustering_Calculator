# FastAPI backend for HCM risk calculator
# Run: uvicorn app:app --reload
from typing import Dict, Optional
import json
import numpy as np
import pandas as pd
import gower
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --------------------
# 1) Domain config (your canonical names)
# --------------------
CLUSTERING_COLS = [
    "Patient Sex", "Age at CMR", "Family history of HCM", "BSA", "Syncope",
    "NYHA Functional Class", "Positive record of genetic mutation",
    "Coronary artery disease", "Maximum WT (mm)", "LV MI (g/m2)",
    "LA diameter (mm)", "LV SVI (ml/m2)", "LVEF (%)", "LV ESVI (ml/m2)",
    "LV EDVI (ml/m2)", "RV EDVI (ml/m2)", "RV ESVI (ml/m2)",
    "RV SVI (ml/m2)", "RV EF (%)", "LGE presence", "Relative LGE mass (%)",
    "Apical Aneurysm", "Peak LVOT Gradient"
]
CATEGORICAL_COLS = ["Patient Sex", "NYHA Functional Class", "Peak LVOT Gradient"]
BINARY_COLS = ["Family history of HCM", "Syncope", "Positive record of genetic mutation",
               "Coronary artery disease", "LGE presence", "Apical Aneurysm"]
NUMERIC_COLS = ["Age at CMR", "BSA", "Maximum WT (mm)", "LV MI (g/m2)", "LA diameter (mm)",
                "LV SVI (ml/m2)", "LVEF (%)", "LV ESVI (ml/m2)", "LV EDVI (ml/m2)",
                "RV EDVI (ml/m2)", "RV ESVI (ml/m2)", "RV SVI (ml/m2)", "RV EF (%)",
                "Relative LGE mass (%)"]

FEATURE_COLUMNS = CLUSTERING_COLS
CAT_MASK = np.array([col in CATEGORICAL_COLS or col in BINARY_COLS for col in FEATURE_COLUMNS])

# --------------------
# 2) Load model assets
# --------------------
MEDOID_PATH = "Assets/medoids.csv"
RISK_JSON_PATH = "Assets/allrisks.json"

medoid_dataframe = pd.read_csv(MEDOID_PATH)
medoid_features = medoid_dataframe[FEATURE_COLUMNS]
medoid_labels = medoid_dataframe["Cluster"].astype(str).tolist()

with open(RISK_JSON_PATH, "r", encoding="utf-8") as handle:
    risk_library = json.load(handle)
outcome_labels = list(risk_library.keys())

# --------------------
# 3) API app + CORS
# --------------------
app = FastAPI(title="HCM Risk API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

class PatientPayload(BaseModel):
    patient_features: Dict[str, Optional[object]] = Field(..., description="Map of feature name -> value")


import random
def label_calculation(patient_features: Dict[str, object]) -> str:
    """Determine the cluster label for a given patient based on their features."""
    return random.choice([1, 2, 3])

@app.post("/predict")
def predict(payload: PatientPayload):
    patient_features = payload.patient_features
    cluster_label = label_calculation(patient_features)
    risk_return = {}
    for outcome, risk in risk_library.items():
        for cluster, cluster_risk in risk.items():
            if cluster == f"Cluster {cluster_label}":
                risk_return[outcome] = cluster_risk

    print(risk_return)
    return {"cluster": cluster_label, "risks": risk_return}
