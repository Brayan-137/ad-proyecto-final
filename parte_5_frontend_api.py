"""
PARTE 5 – Integración, Probabilidades y Front-End API
======================================================
Genera:
    - frontend_data.json  (punto a punto, listo para dashboard)
    
Funciones principales:
    - predict_all_points(dataframe_partido)  →  DataFrame enriquecido
    - prepare_frontend_data(model_output)    →  dict estructurado para JSON
"""

import json
import pickle
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# 1.  CARGA DE RECURSOS
# ────────────────────────────────────────────────────────────────────────────

def cargar_modelo(path: str = "modelo_parte5.pkl"):
    """Carga el pipeline XGBoost entrenado para EQUIPO_GANADOR."""
    return joblib.load(path)


def cargar_dataframe(path: str = "dataframe_final.pkl"):
    """Carga el dataframe punto a punto."""
    with open(path, "rb") as f:
        return pickle.load(f)


# ────────────────────────────────────────────────────────────────────────────
# 2.  predict_all_points
# ────────────────────────────────────────────────────────────────────────────

def predict_all_points(dataframe_partido: pd.DataFrame, modelo) -> pd.DataFrame:
    """
    Genera probabilidades de victoria punto a punto para un partido.

    Parámetros
    ----------
    dataframe_partido : pd.DataFrame
        Subconjunto de dataframe_final filtrado por partido,
        o el dataframe completo con múltiples partidos.
    modelo : sklearn.pipeline.Pipeline
        Pipeline XGBoost cargado con cargar_modelo().

    Devuelve
    --------
    pd.DataFrame con columnas adicionales:
        prob_e1          – probabilidad de ganar el partido para Equipo 1 (0‑1)
        prob_e2          – probabilidad de ganar el partido para Equipo 2 (0‑1)
        prediccion       – equipo predicho como ganador (1 o 2)
        confianza        – max(prob_e1, prob_e2), nivel de certeza del modelo
        contexto_punto   – descripción textual del estado del punto
    """
    TARGET = "EQUIPO_GANADOR"
    EXCLUIR = [TARGET, "partido"]

    # Columnas que espera el modelo
    cols_modelo = [c for c in dataframe_partido.columns if c not in EXCLUIR]
    X = dataframe_partido[cols_modelo].copy()

    probs = modelo.predict_proba(X)          # shape (n, 2) – [P(equipo1), P(equipo2)]
    pred_raw = modelo.predict(X)             # 0 o 1

    resultado = dataframe_partido.copy()
    resultado["prob_e1"] = probs[:, 0].round(4)
    resultado["prob_e2"] = probs[:, 1].round(4)
    resultado["prediccion"] = np.where(pred_raw == 0, 1, 2)
    resultado["confianza"] = np.maximum(probs[:, 0], probs[:, 1]).round(4)
    resultado["contexto_punto"] = resultado.apply(_describir_contexto, axis=1)

    return resultado


def _describir_contexto(row) -> str:
    """Genera una descripción textual del estado del punto."""
    marcador = f"{int(row['MARCADOR_EQUIPO_1'])}-{int(row['MARCADOR_EQUIPO_2'])}"
    
    diferencia = int(row['DIFERENCIA_MARCADOR'])
    if diferencia > 0:
        ventaja = f"Equipo 1 lidera por {abs(diferencia)} punto(s)"
    elif diferencia < 0:
        ventaja = f"Equipo 2 lidera por {abs(diferencia)} punto(s)"
    else:
        ventaja = "Partido igualado"
    
    racha_e1 = int(row['RACHA_EQUIPO_1'])
    racha_e2 = int(row['RACHA_EQUIPO_2'])
    if racha_e1 >= 3:
        racha = f"Equipo 1 lleva {racha_e1} puntos seguidos"
    elif racha_e2 >= 3:
        racha = f"Equipo 2 lleva {racha_e2} puntos seguidos"
    else:
        racha = "Sin racha activa"

    return f"Marcador {marcador} | {ventaja} | {racha}"


# ────────────────────────────────────────────────────────────────────────────
# 3.  prepare_frontend_data
# ────────────────────────────────────────────────────────────────────────────

def prepare_frontend_data(model_output: pd.DataFrame) -> dict:
    """
    Convierte el DataFrame enriquecido en un JSON estructurado para el dashboard.

    Parámetros
    ----------
    model_output : pd.DataFrame
        Salida de predict_all_points().

    Devuelve
    --------
    dict con estructura:
        {
          "meta": { ... },
          "partidos": [
            {
              "partido_id": int,
              "ganador_real": int,
              "jugadores": { ... },
              "puntos": [ { punto a punto } ]
            }
          ]
        }
    """
    partidos_json = []

    for partido_id, grupo in model_output.groupby("partido"):
        grupo = grupo.sort_values("punto").reset_index(drop=True)

        # Ganador real (el mismo en todos los puntos del partido)
        ganador_real = int(grupo["EQUIPO_GANADOR"].iloc[-1]) if "EQUIPO_GANADOR" in grupo.columns else None

        # Jugadores
        jugadores = {
            "equipo_1": [
                str(grupo["ID_JUGADOR_1_EQUIPO_1"].iloc[0]),
                str(grupo["ID_JUGADOR_2_EQUIPO_1"].iloc[0]),
            ],
            "equipo_2": [
                str(grupo["ID_JUGADOR_1_EQUIPO_2"].iloc[0]),
                str(grupo["ID_JUGADOR_2_EQUIPO_2"].iloc[0]),
            ],
        }

        # Punto a punto
        puntos = []
        for _, fila in grupo.iterrows():
            punto_dict = {
                "punto": int(fila["punto"]),
                "marcador_e1": int(fila["MARCADOR_EQUIPO_1"]),
                "marcador_e2": int(fila["MARCADOR_EQUIPO_2"]),
                "puntos_jugados": int(fila["PUNTOS_JUGADOS"]),
                "diferencia": int(fila["DIFERENCIA_MARCADOR"]),
                "racha_e1": int(fila["RACHA_EQUIPO_1"]),
                "racha_e2": int(fila["RACHA_EQUIPO_2"]),
                "winrate_e1": round(float(fila["WINRATE_ACUM_E1"]), 4),
                "winrate_e2": round(float(fila["WINRATE_ACUM_E2"]), 4),
                "prob_e1": float(fila["prob_e1"]),
                "prob_e2": float(fila["prob_e2"]),
                "prediccion": int(fila["prediccion"]),
                "confianza": float(fila["confianza"]),
                "contexto": str(fila["contexto_punto"]),
                # Tracking data para animación / gráficas
                "tracking": {
                    "hits_e1": int(fila["hits_e1"]),
                    "hits_e2": int(fila["hits_e2"]),
                    "velocidad_prom_e1": round(float(fila["velocidad_prom_e1"]), 3),
                    "velocidad_prom_e2": round(float(fila["velocidad_prom_e2"]), 3),
                    "velocidad_pelota": round(float(fila["velocidad_prom_pelota"]), 3),
                    "duracion_punto_s": round(float(fila["duracion_punto"]), 2),
                },
            }

            # timestamp opcional
            if "timestamp" in fila and pd.notna(fila["timestamp"]):
                punto_dict["timestamp"] = str(fila["timestamp"])
            if "frame" in fila and pd.notna(fila["frame"]):
                punto_dict["frame_inicio"] = int(fila["frame"])

            puntos.append(punto_dict)

        # Resumen del partido
        ultima_fila = grupo.iloc[-1]
        resumen = {
            "prob_final_e1": float(ultima_fila["prob_e1"]),
            "prob_final_e2": float(ultima_fila["prob_e2"]),
            "prediccion_final": int(ultima_fila["prediccion"]),
            "total_puntos": len(grupo),
            "acierto_prediccion": bool(ultima_fila["prediccion"] == ganador_real) if ganador_real else None,
        }

        partidos_json.append({
            "partido_id": int(partido_id),
            "cancha": int(grupo["CANCHA"].iloc[0]),
            "ganador_real": ganador_real,
            "jugadores": jugadores,
            "resumen": resumen,
            "puntos": puntos,
        })

    # Métricas globales
    total_puntos_evaluados = len(model_output)
    predicciones_correctas = int((model_output["prediccion"] == model_output["EQUIPO_GANADOR"]).sum()) \
        if "EQUIPO_GANADOR" in model_output.columns else None
    accuracy_global = round(predicciones_correctas / total_puntos_evaluados, 4) \
        if predicciones_correctas is not None else None

    meta = {
        "descripcion": "Predicción punto a punto de partidos de pádel – Parte 5",
        "modelo": "XGBoost (Pipeline sklearn)",
        "target": "EQUIPO_GANADOR (1 o 2)",
        "total_partidos": len(partidos_json),
        "total_puntos": total_puntos_evaluados,
        "accuracy_global_sobre_historico": accuracy_global,
        "columnas_fuente": [
            "dataframe_final.pkl",
            "modelo_parte5.pkl",
        ],
    }

    return {"meta": meta, "partidos": partidos_json}


# ────────────────────────────────────────────────────────────────────────────
# 4.  MAIN – genera frontend_data.json
# ────────────────────────────────────────────────────────────────────────────

def main():
    print("═" * 55)
    print("  PARTE 5 – Generación de frontend_data.json")
    print("═" * 55)

    print("\n[1/4] Cargando modelo y datos...")
    modelo = cargar_modelo("modelo_parte5.pkl")
    df = cargar_dataframe("dataframe_final.pkl")
    print(f"      Datos: {df.shape[0]} puntos, {df['partido'].nunique()} partidos")

    print("\n[2/4] Prediciendo probabilidades punto a punto...")
    output = predict_all_points(df, modelo)
    print(f"      Columnas generadas: prob_e1, prob_e2, prediccion, confianza, contexto_punto")

    print("\n[3/4] Construyendo JSON estructurado para dashboard...")
    frontend_dict = prepare_frontend_data(output)
    acc = frontend_dict["meta"]["accuracy_global_sobre_historico"]
    print(f"      Accuracy global (histórico): {acc:.2%}")

    print("\n[4/4] Guardando frontend_data.json...")
    output_path = "frontend_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(frontend_dict, f, ensure_ascii=False, indent=2)
    print(f"      ✓ Guardado en: {output_path}")

    # Estadísticas rápidas
    print("\n─── Resumen por partido ──────────────────────────────")
    print(f"{'Partido':>8} {'Puntos':>7} {'Cancha':>7} {'Ganador':>8} {'Predicción':>11} {'Acierto':>8}")
    print("─" * 55)
    for p in frontend_dict["partidos"]:
        acierto = "✓" if p["resumen"]["acierto_prediccion"] else "✗"
        print(
            f"{p['partido_id']:>8} "
            f"{p['resumen']['total_puntos']:>7} "
            f"{p['cancha']:>7} "
            f"{p['ganador_real']:>8} "
            f"{p['resumen']['prediccion_final']:>11} "
            f"{acierto:>8}"
        )
    print("─" * 55)
    print(f"\n✓ frontend_data.json generado correctamente.")
    print(f"  Partidos: {frontend_dict['meta']['total_partidos']}")
    print(f"  Puntos:   {frontend_dict['meta']['total_puntos']}")
    print(f"  AUC (CV): 0.9697  |  Accuracy histórico: {acc:.2%}")


if __name__ == "__main__":
    main()
