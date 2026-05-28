"""
PARTE WEB — Dashboard con clips por punto + winrate
====================================================

Esta version esta pensada para trabajar con UN VIDEO CORTO POR CADA PUNTO.
Flujo recomendado:
  1. Ejecutar parte_5_frontend_api.py para generar frontend_data.json.
  2. Ejecutar la web.
  3. Seleccionar un partido.
  4. Cargar los clips de sus puntos. Ejemplo de nombres:
       partido_6_punto_1.mp4
       partido_6_punto_2.mp4
       p6_p3.mp4
       punto_4.mp4
  5. La web muestra el clip de cada punto y actualiza el winrate al cambiar de punto.

Ejecutar:
  python -m uvicorn parte_web.api:app --reload
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = WEB_DIR / "uploads"
TEMPLATES_DIR = WEB_DIR / "templates"
FRONTEND_JSON = BASE_DIR / "frontend_data.json"
MODEL_PATH = BASE_DIR / "modelo_final.pkl"

UPLOADS_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Pádel Winrate por Punto", version="2.0")
app.mount("/videos", StaticFiles(directory=str(UPLOADS_DIR)), name="videos")

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def load_frontend_data() -> Dict[str, Any]:
    if not FRONTEND_JSON.exists():
        raise HTTPException(
            status_code=404,
            detail="No existe frontend_data.json. Ejecuta primero: python parte_5_frontend_api.py",
        )
    with FRONTEND_JSON.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_partido(data: Dict[str, Any], partido_id: int) -> Dict[str, Any]:
    for partido in data.get("partidos", []):
        if int(partido.get("partido_id")) == int(partido_id):
            return partido
    raise HTTPException(status_code=404, detail=f"No existe partido_id={partido_id}")


def safe_video_filename(filename: Optional[str]) -> str:
    original = Path(filename or "video.mp4").name
    clean = re.sub(r"[^A-Za-z0-9_.\- ]+", "_", original).strip().replace(" ", "_")
    return clean or "video.mp4"


def validate_video_file(filename: Optional[str]) -> str:
    suffix = Path(filename or "video.mp4").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Formato no permitido. Usa mp4, mov, avi, mkv o webm.",
        )
    return suffix


def infer_point_from_filename(filename: str) -> Optional[int]:
    """
    Intenta extraer el numero del punto desde nombres como:
      partido_6_punto_1.mp4
      punto-1.mp4
      p1.mp4
      p6_p3.mp4
      clip_7.mp4
    """
    name = Path(filename).stem.lower()
    patterns = [
        r"punto[_\-\s]*(\d+)",
        r"point[_\-\s]*(\d+)",
        r"(?:^|[_\-\s])p[_\-\s]*(\d+)(?:$|[_\-\s])",
        r"(?:^|[_\-\s])(\d+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))

    numbers = re.findall(r"\d+", name)
    if numbers:
        # Usamos el ultimo numero porque en partido_6_punto_1 el punto suele ser el ultimo.
        return int(numbers[-1])
    return None


def partido_upload_dir(partido_id: int) -> Path:
    path = UPLOADS_DIR / f"partido_{int(partido_id)}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def scan_point_videos(partido_id: int) -> Dict[str, Any]:
    folder = partido_upload_dir(partido_id)
    clips: Dict[str, Any] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in ALLOWED_VIDEO_EXTENSIONS:
            continue
        punto = infer_point_from_filename(path.name)
        if punto is None:
            continue
        clips[str(punto)] = {
            "punto": punto,
            "filename": path.name,
            "url": f"/videos/partido_{int(partido_id)}/{path.name}",
        }
    return clips


@app.get("/", response_class=HTMLResponse)
def home():
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="No existe parte_web/templates/index.html")
    return index_path.read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "frontend_data_exists": FRONTEND_JSON.exists(),
        "model_exists": MODEL_PATH.exists(),
        "mode": "clips_por_punto",
    }


@app.get("/api/partidos")
def list_partidos():
    data = load_frontend_data()
    partidos = []
    for p in data.get("partidos", []):
        clips = scan_point_videos(int(p["partido_id"]))
        partidos.append(
            {
                "partido_id": p["partido_id"],
                "cancha": p.get("cancha"),
                "total_puntos": p.get("resumen", {}).get("total_puntos"),
                "marcador_final_e1": p.get("resumen", {}).get("marcador_final_e1"),
                "marcador_final_e2": p.get("resumen", {}).get("marcador_final_e2"),
                "prediccion_final": p.get("resumen", {}).get("prediccion_final"),
                "clips_cargados": len(clips),
            }
        )
    return {"meta": data.get("meta", {}), "partidos": partidos}


@app.get("/api/partidos/{partido_id}")
def get_partido(partido_id: int):
    data = load_frontend_data()
    partido = find_partido(data, partido_id)
    partido = dict(partido)
    partido["point_videos"] = scan_point_videos(partido_id)
    return partido


@app.get("/api/point-videos/{partido_id}")
def list_point_videos(partido_id: int):
    return {"partido_id": partido_id, "clips": scan_point_videos(partido_id)}


@app.post("/api/upload-point-video/{partido_id}/{punto}")
def upload_point_video(partido_id: int, punto: int, file: UploadFile = File(...)):
    suffix = validate_video_file(file.filename)
    folder = partido_upload_dir(partido_id)
    destination = folder / f"punto_{int(punto)}{suffix}"

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "ok": True,
        "partido_id": partido_id,
        "punto": punto,
        "filename": destination.name,
        "url": f"/videos/partido_{int(partido_id)}/{destination.name}",
    }


@app.post("/api/upload-point-videos/{partido_id}")
def upload_point_videos(partido_id: int, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No se recibieron archivos.")

    uploaded = []
    rejected = []
    folder = partido_upload_dir(partido_id)

    for file in files:
        try:
            suffix = validate_video_file(file.filename)
            safe_name = safe_video_filename(file.filename)
            punto = infer_point_from_filename(safe_name)
            if punto is None:
                rejected.append({"filename": file.filename, "reason": "No se pudo inferir el numero del punto."})
                continue

            destination = folder / f"punto_{int(punto)}{suffix}"
            with destination.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            uploaded.append(
                {
                    "punto": int(punto),
                    "original_filename": file.filename,
                    "filename": destination.name,
                    "url": f"/videos/partido_{int(partido_id)}/{destination.name}",
                }
            )
        except HTTPException as exc:
            rejected.append({"filename": file.filename, "reason": exc.detail})

    return {
        "ok": True,
        "partido_id": partido_id,
        "uploaded": sorted(uploaded, key=lambda x: x["punto"]),
        "rejected": rejected,
        "clips": scan_point_videos(partido_id),
    }


@app.post("/api/live/predict-point")
def predict_point(payload: Dict[str, Any]):
    """
    Endpoint opcional para prediccion en vivo si en el futuro se reciben features
    calculadas automaticamente desde computer vision.
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
