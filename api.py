from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import glob

# ============================================================
# 1. CHARGER LES MODELES (MLOps)
# ============================================================

app = FastAPI(title="Battery Health API")
rul_model = joblib.load("model/rul_model.pkl")
model      = joblib.load("model/battery_model.pkl")
classifier = joblib.load("model/battery_classifier.pkl")
# Charger le scaler
last_scaler = max(glob.glob("model/scaler_*.pkl"))  
scaler      = joblib.load(last_scaler)              
print(f"Scaler chargé : {last_scaler}")  

# ============================================================
# 2. SCHEMA D'ENTREE (les données envoyées par l'utilisateur)
# ============================================================

class BatteryInput(BaseModel):
    cycle : float
    chI   : float
    chV   : float
    chT   : float
    disI  : float
    disV  : float
    disT  : float
    BCt   : float

# ============================================================
# 3. ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {"message": "Battery Health API is running !"}


@app.post("/predict/soh")
def predict_soh(data: BatteryInput):
    """Régression → prédit le SOH en %"""
    df = pd.DataFrame([[
        data.cycle, data.chI, data.chV, data.chT,
        data.disI,  data.disV, data.disT, data.BCt
    ]], columns=["cycle","chI","chV","chT","disI","disV","disT","BCt"])
    df = scaler.transform(df) 
    soh = model.predict(df)[0]

    return {
        "SOH_predit"  : round(float(soh), 2),
        "unite"       : "%",
        "interpretation" : "Bonne" if soh >= 90 else "Acceptable" if soh >= 80 else "Dégradée"
    }

@app.post("/predict/rul")
def predict_rul(data: BatteryInput):

    df = pd.DataFrame([[
        data.cycle, data.chI, data.chV, data.chT,
        data.disI, data.disV, data.disT, data.BCt
    ]], columns=["cycle","chI","chV","chT","disI","disV","disT","BCt"])
    df = scaler.transform(df) 
    rul = rul_model.predict(df)[0]

    return {
        "RUL_predite": round(float(rul), 2),
        "unite": "cycles",
        "message": f"Durée de vie restante estimée : {int(rul)} cycles"
    }
    
@app.post("/predict/label")
def predict_label(data: BatteryInput):
    """Classification → prédit Bonne / Acceptable / Dégradée"""
    df = pd.DataFrame([[
        data.cycle, data.chI, data.chV, data.chT,
        data.disI,  data.disV, data.disT, data.BCt
    ]], columns=["cycle","chI","chV","chT","disI","disV","disT","BCt"])
    df = scaler.transform(df) 
    label = classifier.predict(df)[0]

    return {
        "label_predit" : label,
        "message"      : f"La batterie est : {label}"
    }