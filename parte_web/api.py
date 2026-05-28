"""
PARTE WEB — Dashboard con video + winrate en tiempo real por puntos
====================================================================

Qué hace:
  - Carga frontend_data.json generado por la Parte 5.
  - Permite subir un video del partido.
  - Sincroniza el avance del video con los puntos usando time_start_s/time_end_s.
  - Muestra prob_e1/prob_e2, marcador, favorito y contexto del punto.
  - Incluye endpoint /api/live/predict-point para predecir un punto recibido como JSON.

Ejecutar:
  pip install fastapi uvicorn python-multipart pandas numpy scikit-learn xgboost joblib
  python parte_5_frontend_api_corregido.py
  uvicorn parte_web.api:app --reload

Abrir:
  http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = WEB_DIR / "uploads"
TEMPLATES_DIR = WEB_DIR / "templates"
FRONTEND_JSON = BASE_DIR / "frontend_data.json"
MODEL_PATH = BASE_DIR / "modelo_final.pkl"

UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Pádel Winrate Live Dashboard", version="1.0")
app.mount("/videos", StaticFiles(directory=str(UPLOADS_DIR)), name="videos")


def load_frontend_data() -> Dict[str, Any]:
    if not FRONTEND_JSON.exists():
        raise HTTPException(
            status_code=404,
            detail="No existe frontend_data.json. Ejecuta primero parte_5_frontend_api_corregido.py",
        )
    with FRONTEND_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_partido(data: Dict[str, Any], partido_id: int) -> Dict[str, Any]:
    for partido in data.get("partidos", []):
        if int(partido.get("partido_id")) == int(partido_id):
            return partido
    raise HTTPException(status_code=404, detail=f"No existe partido_id={partido_id}")


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = TEMPLATES_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok", "frontend_data_exists": FRONTEND_JSON.exists(), "model_exists": MODEL_PATH.exists()}


@app.get("/api/partidos")
def list_partidos():
    data = load_frontend_data()
    return {
        "meta": data.get("meta", {}),
        "partidos": [
            {
                "partido_id": p["partido_id"],
                "cancha": p.get("cancha"),
                "total_puntos": p.get("resumen", {}).get("total_puntos"),
                "marcador_final_e1": p.get("resumen", {}).get("marcador_final_e1"),
                "marcador_final_e2": p.get("resumen", {}).get("marcador_final_e2"),
                "prediccion_final": p.get("resumen", {}).get("prediccion_final"),
            }
            for p in data.get("partidos", [])
        ],
    }


@app.get("/api/partidos/{partido_id}")
def get_partido(partido_id: int):
    data = load_frontend_data()
    return find_partido(data, partido_id)


@app.post("/api/upload-video")
def upload_video(file: UploadFile = File(...)):
    allowed = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    suffix = Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Formato no permitido. Usa mp4, mov, avi, mkv o webm.")

    safe_name = Path(file.filename or f"video{suffix}").name.replace(" ", "_")
    destination = UPLOADS_DIR / safe_name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": safe_name, "url": f"/videos/{safe_name}"}


@app.post("/api/live/predict-point")
def predict_point(payload: Dict[str, Any]):
    """
    Endpoint para integración en vivo.

    Recibe un JSON con features de UNA fila en perspectiva dual y devuelve la probabilidad
    de que ese equipo focal gane el partido. Para generar prob_e1/prob_e2 en vivo, llama
    dos veces: una con perspectiva_equipo=1 y otra con perspectiva_equipo=2.
    """
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=404, detail="No existe modelo_final.pkl. Ejecuta primero la Parte 4.")

    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact
    feature_cols = artifact.get("feature_cols", []) if isinstance(artifact, dict) else []
    if not feature_cols:
        raise HTTPException(status_code=400, detail="El artefacto del modelo no contiene feature_cols.")

    row = pd.DataFrame([payload])
    for col in feature_cols:
        if col not in row.columns:
            row[col] = None
    proba = float(model.predict_proba(row[feature_cols])[:, 1][0])
    return {"prob_gana_partido_equipo_focal": round(proba, 4)}
