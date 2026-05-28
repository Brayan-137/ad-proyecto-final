"""
PARTE 5 – Integración, Probabilidades y Front-End API
======================================================
Versión actualizada para el nuevo esquema de variables.

Genera:
    - frontend_data.json

Funciones principales:
    - predict_all_points(dataframe_partido)
    - prepare_frontend_data(model_output)
"""

import json
import pickle
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# CONFIG
# ────────────────────────────────────────────────────────────────────────────

TARGET = "ganador_partido"

COLUMNAS_EXCLUIR_MODELO = [
    TARGET,
    "partido",
    "punto",
    "cancha",
    "jugador_1_equipo",
    "jugador_2_equipo",
    "jugador_1_rival",
    "jugador_2_rival",
]

# ────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE RECURSOS
# ────────────────────────────────────────────────────────────────────────────

def cargar_modelo(path: str = "xgboost_final.pkl"):
    return joblib.load(path)


def cargar_dataframe(path: str = "dataframes/data_con_jugadores.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ────────────────────────────────────────────────────────────────────────────
# 2. PREPARACIÓN DE DATOS
# ────────────────────────────────────────────────────────────────────────────

def preparar_features(df: pd.DataFrame) -> pd.DataFrame:

    X = df.drop(
        columns=[
            c for c in COLUMNAS_EXCLUIR_MODELO
            if c in df.columns
        ],
        errors="ignore"
    ).copy()

    return X


# ────────────────────────────────────────────────────────────────────────────
# 3. PREDICCIÓN PUNTO A PUNTO
# ────────────────────────────────────────────────────────────────────────────

def predict_all_points(
    dataframe_partido: pd.DataFrame,
    modelo
) -> pd.DataFrame:

    X = preparar_features(dataframe_partido)

    probs = modelo.predict_proba(X)

    pred_raw = modelo.predict(X)

    resultado = dataframe_partido.copy()

    # Clase 1 = equipo gana
    # Clase 0 = rival gana
    resultado["prob_equipo"] = probs[:, 1].round(4)
    resultado["prob_rival"] = probs[:, 0].round(4)

    resultado["prediccion"] = np.where(
        pred_raw == 1,
        "equipo",
        "rival"
    )

    resultado["confianza"] = np.maximum(
        probs[:, 0],
        probs[:, 1]
    ).round(4)

    resultado["contexto_punto"] = resultado.apply(
        _describir_contexto,
        axis=1
    )

    return resultado


# ────────────────────────────────────────────────────────────────────────────
# 4. CONTEXTO TEXTUAL
# ────────────────────────────────────────────────────────────────────────────

def _describir_contexto(row) -> str:

    marcador = (
        f"{int(row['marcador_equipo'])}"
        f"-"
        f"{int(row['marcador_rival'])}"
    )

    diferencia = int(row["diferencia_marcador"])

    if diferencia > 0:
        ventaja = (
            f"Equipo lidera por "
            f"{abs(diferencia)} punto(s)"
        )

    elif diferencia < 0:
        ventaja = (
            f"Rival lidera por "
            f"{abs(diferencia)} punto(s)"
        )

    else:
        ventaja = "Partido igualado"

    racha_equipo = int(row["racha_ultimos3_equipo"])
    racha_rival = int(row["racha_ultimos3_rival"])

    if racha_equipo >= 3:
        racha = (
            f"Equipo lleva "
            f"{racha_equipo} puntos seguidos"
        )

    elif racha_rival >= 3:
        racha = (
            f"Rival lleva "
            f"{racha_rival} puntos seguidos"
        )

    else:
        racha = "Sin racha activa"

    return (
        f"Marcador {marcador} | "
        f"{ventaja} | "
        f"{racha}"
    )


# ────────────────────────────────────────────────────────────────────────────
# 5. FRONTEND JSON
# ────────────────────────────────────────────────────────────────────────────

def prepare_frontend_data(
    model_output: pd.DataFrame
) -> dict:

    partidos_json = []

    for partido_id, grupo in model_output.groupby("partido"):

        grupo = (
            grupo
            .sort_values("punto")
            .reset_index(drop=True)
        )

        ganador_real = int(
            grupo["ganador_partido"].iloc[-1]
        )

        jugadores = {
            "equipo": [
                str(grupo["jugador_1_equipo"].iloc[0]),
                str(grupo["jugador_2_equipo"].iloc[0]),
            ],
            "rival": [
                str(grupo["jugador_1_rival"].iloc[0]),
                str(grupo["jugador_2_rival"].iloc[0]),
            ],
        }

        puntos = []

        for _, fila in grupo.iterrows():

            # duración a segundos
            duracion_seg = None

            try:
                duracion_seg = round(
                    pd.to_timedelta(
                        fila["duracion_punto"]
                    ).total_seconds(),
                    2
                )
            except Exception:
                pass

            punto_dict = {

                # ───── Contexto ─────
                "punto": int(fila["punto"]),
                "marcador_equipo": int(
                    fila["marcador_equipo"]
                ),
                "marcador_rival": int(
                    fila["marcador_rival"]
                ),

                "diferencia_marcador": int(
                    fila["diferencia_marcador"]
                ),

                "gano_punto": int(
                    fila["gano_punto"]
                ),

                # ───── Momentum ─────
                "win_rate_equipo": round(
                    float(fila["win_rate_acum_equipo"]),
                    4
                ),

                "racha_equipo": float(
                    fila["racha_ultimos3_equipo"]
                ),

                "racha_rival": float(
                    fila["racha_ultimos3_rival"]
                ),

                # ───── Predicción ─────
                "prob_equipo": float(
                    fila["prob_equipo"]
                ),

                "prob_rival": float(
                    fila["prob_rival"]
                ),

                "prediccion": str(
                    fila["prediccion"]
                ),

                "confianza": float(
                    fila["confianza"]
                ),

                "contexto": str(
                    fila["contexto_punto"]
                ),

                # ───── Tracking ─────
                "tracking": {

                    "hits_equipo": int(
                        fila["hits_equipo"]
                    ),

                    "hits_rival": int(
                        fila["hits_rival"]
                    ),

                    "velocidad_prom_equipo": round(
                        float(
                            fila["velocidad_prom_equipo"]
                        ),
                        3
                    ),

                    "velocidad_prom_rival": round(
                        float(
                            fila["velocidad_prom_rival"]
                        ),
                        3
                    ),

                    "velocidad_pelota": round(
                        float(
                            fila["velocidad_prom_pelota"]
                        ),
                        3
                    ),

                    "duracion_punto_s": duracion_seg,
                },
            }

            puntos.append(punto_dict)

        ultima_fila = grupo.iloc[-1]

        resumen = {

            "prob_final_equipo": float(
                ultima_fila["prob_equipo"]
            ),

            "prob_final_rival": float(
                ultima_fila["prob_rival"]
            ),

            "prediccion_final": str(
                ultima_fila["prediccion"]
            ),

            "total_puntos": len(grupo),

            "acierto_prediccion": bool(
                (
                    ultima_fila["prediccion"] == "equipo"
                    and ganador_real == 1
                )
                or
                (
                    ultima_fila["prediccion"] == "rival"
                    and ganador_real == 0
                )
            ),
        }

        partidos_json.append({

            "partido_id": int(partido_id),

            "cancha": int(
                grupo["cancha"].iloc[0]
            ),

            "ganador_real": ganador_real,

            "jugadores": jugadores,

            "resumen": resumen,

            "puntos": puntos,
        })

    # ─────────────────────────────────────────────────────
    # MÉTRICAS GLOBALES
    # ─────────────────────────────────────────────────────

    total_puntos = len(model_output)

    pred_bin = np.where(
        model_output["prediccion"] == "equipo",
        1,
        0
    )

    accuracy = (
        pred_bin
        ==
        model_output["ganador_partido"]
    ).mean()

    meta = {

        "descripcion": (
            "Predicción punto a punto "
            "de partidos de pádel"
        ),

        "modelo": "XGBoost",

        "target": "ganador_partido",

        "total_partidos": len(partidos_json),

        "total_puntos": total_puntos,

        "accuracy_global": round(
            float(accuracy),
            4
        ),

        "fuentes": [
            "data_con_jugadores.pkl",
            "xgboost_final.pkl",
        ],
    }

    return {
        "meta": meta,
        "partidos": partidos_json
    }


# ────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ────────────────────────────────────────────────────────────────────────────

def main():

    print("=" * 60)
    print(" PARTE 5 - Frontend JSON")
    print("=" * 60)

    print("\n[1/4] Cargando recursos...")

    modelo = cargar_modelo(
        "xgboost_final.pkl"
    )

    df = cargar_dataframe(
        "dataframes/data_con_jugadores.pkl"
    )

    print(
        f"Datos cargados: "
        f"{df.shape[0]} filas | "
        f"{df['partido'].nunique()} partidos"
    )

    print("\n[2/4] Predicción punto a punto...")

    output = predict_all_points(
        df,
        modelo
    )

    print("OK")

    print("\n[3/4] Construyendo JSON...")

    frontend_dict = prepare_frontend_data(
        output
    )

    print("OK")

    print("\n[4/4] Guardando archivo...")

    output_path = "frontend_data.json"

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            frontend_dict,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"Guardado: {output_path}")

    # ─────────────────────────────────────────────────
    # Resumen
    # ─────────────────────────────────────────────────

    acc = frontend_dict["meta"]["accuracy_global"]

    print("\nResumen")
    print("-" * 60)

    print(
        f"Partidos: "
        f"{frontend_dict['meta']['total_partidos']}"
    )

    print(
        f"Puntos: "
        f"{frontend_dict['meta']['total_puntos']}"
    )

    print(
        f"Accuracy histórico: "
        f"{acc:.2%}"
    )

    print("\nfrontend_data.json generado correctamente")


if __name__ == "__main__":
    main()