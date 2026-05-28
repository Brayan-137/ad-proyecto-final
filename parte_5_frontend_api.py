"""
PARTE 5 - Generacion de JSON para dashboard/web
================================================

Esta parte NO es notebook: es un script de integracion.

Objetivo:
    Cargar el modelo final de la Parte 4 y generar frontend_data.json con:
      - prob_e1 y prob_e2 por punto
      - marcador por punto
      - ganador del punto
      - favorito y confianza
      - contexto textual
      - tiempos de sincronizacion con video

Ejecucion:
    python parte_5_frontend_api.py

Requisitos previos:
    python parte_4_modelos_mejorados.py
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATA_PKL = BASE_DIR / "dataframes" / "data_con_jugadores.pkl"
DATA_CSV = BASE_DIR / "dataframes" / "data_con_jugadores.csv"
MODEL_PATH = BASE_DIR / "modelo_final.pkl"
FALLBACK_MODEL_PATH = BASE_DIR / "xgboost_final.pkl"
OUTPUT_JSON = BASE_DIR / "frontend_data.json"

TARGET_COL = "ganador_partido"


def load_dataframe() -> pd.DataFrame:
    # Primero intenta pickle. Si falla por incompatibilidad de version de pandas,
    # cae automaticamente al CSV.
    if DATA_PKL.exists():
        try:
            return pd.read_pickle(DATA_PKL)
        except Exception as e1:
            try:
                with DATA_PKL.open("rb") as f:
                    return pickle.load(f)
            except Exception as e2:
                if DATA_CSV.exists():
                    print(f"Aviso: no se pudo leer el .pkl ({type(e2).__name__}). Se usara el CSV.")
                    return pd.read_csv(DATA_CSV)
                raise e2

    if DATA_CSV.exists():
        return pd.read_csv(DATA_CSV)

    raise FileNotFoundError("No existe dataframes/data_con_jugadores.pkl ni .csv")


def load_model_artifact() -> Dict[str, Any]:
    path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(
            "No existe modelo_final.pkl ni xgboost_final.pkl. Ejecuta primero parte_4_modelos_mejorados.py"
        )

    artifact = joblib.load(path)

    # Compatibilidad con modelos viejos guardados directamente como pipeline.
    if not isinstance(artifact, dict):
        artifact = {
            "model": artifact,
            "feature_cols": None,
            "target_col": TARGET_COL,
            "notes": "Artefacto antiguo sin metadata. Se inferiran columnas.",
        }

    if "model" not in artifact:
        raise ValueError("El artefacto no contiene la clave 'model'.")

    return artifact


def convert_duration_to_seconds(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric

    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "duracion_punto" in df.columns:
        df["duracion_punto"] = convert_duration_to_seconds(df["duracion_punto"])

    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})

    required = ["partido", "punto"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")

    return df


def infer_feature_cols(df: pd.DataFrame) -> List[str]:
    exclude = {
        TARGET_COL,
        "partido",
        "cancha",
        "perspectiva_equipo",
        "jugador_1_equipo",
        "jugador_2_equipo",
        "jugador_1_rival",
        "jugador_2_rival",
    }

    for col in df.columns:
        col_l = col.lower()
        if col_l.startswith("jugador") or "nombre" in col_l:
            exclude.add(col)
        if any(k in col_l for k in ["ganador", "winner", "target", "label", "resultado_final", "marcador_final"]):
            if col != TARGET_COL:
                exclude.add(col)

    return [c for c in df.columns if c not in exclude]


def prepare_features(df: pd.DataFrame, feature_cols: Optional[List[str]]) -> pd.DataFrame:
    if not feature_cols:
        feature_cols = infer_feature_cols(df)

    X = df.copy()

    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan

    return X[feature_cols]


def predict_probabilities(df: pd.DataFrame, artifact: Dict[str, Any]) -> pd.DataFrame:
    model = artifact["model"]
    feature_cols = artifact.get("feature_cols")

    X = prepare_features(df, feature_cols)

    probs = model.predict_proba(X)[:, 1]

    out = df.copy()
    out["prob_focal"] = np.clip(probs, 0, 1)
    out["pred_focal"] = (out["prob_focal"] >= 0.5).astype(int)

    return out


def safe_int(value, default=0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except Exception:
        return default


def safe_float(value, default=None):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_row_by_perspective(group: pd.DataFrame, perspective: int) -> Optional[pd.Series]:
    if "perspectiva_equipo" not in group.columns:
        return None

    subset = group[group["perspectiva_equipo"].astype(str) == str(perspective)]
    if subset.empty:
        return None
    return subset.iloc[0]


def extract_players(row_e1: pd.Series, row_e2: Optional[pd.Series] = None) -> Dict[str, List[str]]:
    if row_e1 is not None:
        return {
            "equipo_1": [
                str(row_e1.get("jugador_1_equipo", "Jugador 1 E1")),
                str(row_e1.get("jugador_2_equipo", "Jugador 2 E1")),
            ],
            "equipo_2": [
                str(row_e1.get("jugador_1_rival", "Jugador 1 E2")),
                str(row_e1.get("jugador_2_rival", "Jugador 2 E2")),
            ],
        }

    if row_e2 is not None:
        return {
            "equipo_1": [
                str(row_e2.get("jugador_1_rival", "Jugador 1 E1")),
                str(row_e2.get("jugador_2_rival", "Jugador 2 E1")),
            ],
            "equipo_2": [
                str(row_e2.get("jugador_1_equipo", "Jugador 1 E2")),
                str(row_e2.get("jugador_2_equipo", "Jugador 2 E2")),
            ],
        }

    return {"equipo_1": ["Equipo 1"], "equipo_2": ["Equipo 2"]}


def build_context(score_e1: int, score_e2: int, winner_point: str, prob_e1: float, prob_e2: float) -> str:
    diff = score_e1 - score_e2

    if diff > 0:
        marcador_txt = f"Equipo 1 lidera por {abs(diff)} punto(s)"
    elif diff < 0:
        marcador_txt = f"Equipo 2 lidera por {abs(diff)} punto(s)"
    else:
        marcador_txt = "Partido igualado"

    favorito = "Equipo 1" if prob_e1 >= prob_e2 else "Equipo 2"
    return (
        f"Marcador {score_e1}-{score_e2}. "
        f"{marcador_txt}. "
        f"Ganador del punto: {winner_point}. "
        f"Favorito actual: {favorito}."
    )


def point_duration_from_row(row: pd.Series) -> float:
    if row is None:
        return 0.0

    for col in ["duracion_punto_s", "duration_s", "duration", "duracion_punto"]:
        if col in row.index:
            val = safe_float(row[col], None)
            if val is not None:
                return max(float(val), 0.0)

    return 0.0


def build_point_record(group: pd.DataFrame, time_start_s: float) -> Tuple[Dict[str, Any], float]:
    row_e1 = get_row_by_perspective(group, 1)
    row_e2 = get_row_by_perspective(group, 2)

    # Si no hay doble perspectiva, se usa la primera fila como equipo 1.
    if row_e1 is None:
        row_e1 = group.iloc[0]

    if row_e2 is None and len(group) > 1:
        row_e2 = group.iloc[1]

    partido_id = safe_int(row_e1.get("partido"))
    punto = safe_int(row_e1.get("punto"))

    prob_e1 = safe_float(row_e1.get("prob_focal"), 0.5)
    if row_e2 is not None and "prob_focal" in row_e2.index:
        prob_e2 = safe_float(row_e2.get("prob_focal"), 1.0 - prob_e1)
    else:
        prob_e2 = 1.0 - prob_e1

    prob_e1 = round(float(np.clip(prob_e1, 0, 1)), 4)
    prob_e2 = round(float(np.clip(prob_e2, 0, 1)), 4)

    score_e1 = safe_int(row_e1.get("marcador_equipo", 0))
    score_e2 = safe_int(row_e1.get("marcador_rival", 0))

    gano_punto_e1 = safe_int(row_e1.get("gano_punto", 0))
    winner_point = "Equipo 1" if gano_punto_e1 == 1 else "Equipo 2"

    real_winner_e1_perspective = safe_int(row_e1.get(TARGET_COL, 0))
    ganador_real = "Equipo 1" if real_winner_e1_perspective == 1 else "Equipo 2"

    favorito = "Equipo 1" if prob_e1 >= prob_e2 else "Equipo 2"
    confianza = round(max(prob_e1, prob_e2), 4)

    duration_s = point_duration_from_row(row_e1)
    # Si no hay duracion real, asigna una duracion demo para que el video avance por puntos.
    if duration_s <= 0:
        duration_s = 6.0

    time_end_s = time_start_s + duration_s

    tracking = {
        "duracion_punto_s": round(duration_s, 3),
        "hits_e1": safe_int(row_e1.get("hits_equipo", 0)),
        "hits_e2": safe_int(row_e1.get("hits_rival", 0)),
        "velocidad_prom_e1": safe_float(row_e1.get("velocidad_prom_equipo"), None),
        "velocidad_prom_e2": safe_float(row_e1.get("velocidad_prom_rival"), None),
        "velocidad_max_e1": safe_float(row_e1.get("velocidad_max_equipo"), None),
        "velocidad_max_e2": safe_float(row_e1.get("velocidad_max_rival"), None),
        "velocidad_prom_pelota": safe_float(row_e1.get("velocidad_prom_pelota"), None),
        "velocidad_max_pelota": safe_float(row_e1.get("velocidad_max_pelota"), None),
        "desplazamiento_e1": safe_float(row_e1.get("desplazamiento_equipo"), None),
        "desplazamiento_e2": safe_float(row_e1.get("desplazamiento_rival"), None),
    }

    context = build_context(score_e1, score_e2, winner_point, prob_e1, prob_e2)

    point_record = {
        "partido_id": partido_id,
        "punto": punto,
        "time_start_s": round(time_start_s, 3),
        "time_end_s": round(time_end_s, 3),
        "marcador_e1": score_e1,
        "marcador_e2": score_e2,
        "gano_punto": winner_point,
        "ganador_real_partido": ganador_real,
        "prob_e1": prob_e1,
        "prob_e2": prob_e2,
        "favorito": favorito,
        "confianza": confianza,
        "contexto": context,
        "tracking": tracking,
    }

    return point_record, time_end_s


def prepare_frontend_data(predicted_df: pd.DataFrame, artifact: Dict[str, Any]) -> Dict[str, Any]:
    partidos_json = []

    predicted_df = predicted_df.sort_values(["partido", "punto"]).copy()

    for partido_id, partido_df in predicted_df.groupby("partido", sort=True):
        partido_df = partido_df.sort_values(["punto"]).copy()
        puntos = []
        time_cursor = 0.0

        first_group = None
        for _, group in partido_df.groupby("punto", sort=True):
            first_group = group
            break

        row_e1_first = get_row_by_perspective(first_group, 1) if first_group is not None else None
        if row_e1_first is None and first_group is not None:
            row_e1_first = first_group.iloc[0]

        row_e2_first = get_row_by_perspective(first_group, 2) if first_group is not None else None
        jugadores = extract_players(row_e1_first, row_e2_first)

        for _, point_group in partido_df.groupby("punto", sort=True):
            record, time_cursor = build_point_record(point_group, time_cursor)
            puntos.append(record)

        if puntos:
            last = puntos[-1]
            marcador_final_e1 = last["marcador_e1"]
            marcador_final_e2 = last["marcador_e2"]
            ganador_real = last["ganador_real_partido"]
            prediccion_final = last["favorito"]
            acierto_final = prediccion_final == ganador_real
            duracion_total_s = last["time_end_s"]
        else:
            marcador_final_e1 = marcador_final_e2 = 0
            ganador_real = prediccion_final = None
            acierto_final = False
            duracion_total_s = 0.0

        partidos_json.append({
            "partido_id": safe_int(partido_id),
            "cancha": safe_int(partido_df["cancha"].iloc[0]) if "cancha" in partido_df.columns else None,
            "jugadores": jugadores,
            "resumen": {
                "total_puntos": len(puntos),
                "marcador_final_e1": marcador_final_e1,
                "marcador_final_e2": marcador_final_e2,
                "ganador_real": ganador_real,
                "prediccion_final": prediccion_final,
                "acierto_prediccion_final": bool(acierto_final),
                "duracion_total_s": round(duracion_total_s, 3),
            },
            "puntos": puntos,
        })

    total_puntos = sum(len(p["puntos"]) for p in partidos_json)
    aciertos = [
        p["resumen"]["acierto_prediccion_final"]
        for p in partidos_json
        if p["resumen"]["ganador_real"] is not None
    ]

    return {
        "meta": {
            "descripcion": "JSON para dashboard de winrate en tiempo real por punto.",
            "modelo": artifact.get("best_model_name", "modelo_final"),
            "target": artifact.get("target_col", TARGET_COL),
            "total_partidos": len(partidos_json),
            "total_puntos": total_puntos,
            "accuracy_final_por_partido": round(float(np.mean(aciertos)), 4) if aciertos else None,
            "nota": (
                "prob_e1/prob_e2 representan la probabilidad de ganar el partido "
                "despues de cada punto, no la probabilidad de ganar el siguiente punto."
            ),
        },
        "partidos": partidos_json,
    }


def main():
    print("=" * 70)
    print("PARTE 5 - GENERACION FRONTEND_DATA.JSON")
    print("=" * 70)

    print("[1/4] Cargando dataframe...")
    df = clean_dataframe(load_dataframe())
    print(f"Datos: {df.shape[0]} filas | {df['partido'].nunique()} partidos")

    print("[2/4] Cargando modelo...")
    artifact = load_model_artifact()
    print(f"Modelo: {artifact.get('best_model_name', 'modelo_final')}")

    print("[3/4] Prediciendo probabilidades por perspectiva...")
    predicted = predict_probabilities(df, artifact)

    print("[4/4] Construyendo JSON por punto real...")
    frontend_data = prepare_frontend_data(predicted, artifact)

    with OUTPUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(frontend_data, f, ensure_ascii=False, indent=2)

    print(f"\nArchivo generado: {OUTPUT_JSON.name}")
    print(f"Partidos: {frontend_data['meta']['total_partidos']}")
    print(f"Puntos reales: {frontend_data['meta']['total_puntos']}")
    print(f"Accuracy final por partido: {frontend_data['meta']['accuracy_final_por_partido']}")


if __name__ == "__main__":
    main()
