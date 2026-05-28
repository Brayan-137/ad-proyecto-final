"""
PARTE 4 - Modelos mejorados con control de leakage
===================================================

Objetivo:
    Entrenar modelos para estimar, DESPUES de cada punto, la probabilidad de que
    el equipo focal gane el partido de padel.

Por que este script reemplaza el entrenamiento anterior:
    El AUC cercano a 1.0 venia de dividir filas al azar. En este dataset cada partido
    aparece con muchos puntos y en doble perspectiva. Si puntos del mismo partido
    quedan en train y test, el modelo memoriza el partido.

Este script:
    1. Carga dataframes/data_con_jugadores.pkl o .csv.
    2. Elimina columnas con leakage o identificadores.
    3. Valida por PARTIDO usando GroupKFold / GroupShuffleSplit.
    4. Compara varios modelos.
    5. Entrena XGBoost con tuning.
    6. Guarda el artefacto final con columnas esperadas para la Parte 5 y la web.

Salidas:
    - modelo_final.pkl
    - xgboost_final.pkl
    - comparacion_modelos_mejorados.csv
    - importancia_variables_xgboost.csv
    - diagnostico_modelos.txt
    - reporte_xgboost.pdf
"""

from __future__ import annotations

import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_OK = True
except Exception:
    XGBOOST_OK = False
    XGBClassifier = None


warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PKL = BASE_DIR / "dataframes" / "data_con_jugadores.pkl"
DATA_CSV = BASE_DIR / "dataframes" / "data_con_jugadores.csv"

TARGET_COL = "ganador_partido"
GROUP_COL = "partido"

# Columnas que NO deben entrar al modelo.
# Nota: "punto" SI se deja porque en vivo se sabe en que punto va el partido.
BASE_EXCLUDE_COLS = {
    TARGET_COL,
    GROUP_COL,
    "cancha",
    "perspectiva_equipo",

    # Nombres/identificadores de jugadores: memorizan partidos con pocos datos.
    "jugador_1_equipo",
    "jugador_2_equipo",
    "jugador_1_rival",
    "jugador_2_rival",

    # Posibles IDs/tiempos brutos que pueden memorizar videos o partidos.
    "video_id",
    "match_id",
    "partido_id",
    "punto_id",
    "frame",
    "timestamp",
    "time",
    "fecha",
}

LEAKAGE_KEYWORDS = [
    "ganador",
    "winner",
    "resultado_final",
    "marcador_final",
    "target",
    "label",
]


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

    raise FileNotFoundError(
        "No encontre dataframes/data_con_jugadores.pkl ni dataframes/data_con_jugadores.csv"
    )


def convert_duration_to_seconds(series: pd.Series) -> pd.Series:
    """Convierte duracion_punto sin dañar valores que ya son numericos."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric

    return pd.to_timedelta(series, errors="coerce").dt.total_seconds()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if TARGET_COL not in df.columns:
        raise ValueError(f"No existe la columna objetivo '{TARGET_COL}'.")

    if GROUP_COL not in df.columns:
        raise ValueError(f"No existe la columna de grupo '{GROUP_COL}'.")

    if "duracion_punto" in df.columns:
        df["duracion_punto"] = convert_duration_to_seconds(df["duracion_punto"])

    df = df.replace([np.inf, -np.inf], np.nan)
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")
    df = df.dropna(subset=[TARGET_COL, GROUP_COL])
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # Normaliza strings vacios
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).replace({"nan": np.nan, "None": np.nan, "": np.nan})

    return df


def detect_excluded_cols(df: pd.DataFrame) -> List[str]:
    exclude = set(c for c in BASE_EXCLUDE_COLS if c in df.columns)

    for col in df.columns:
        col_l = col.lower()
        for kw in LEAKAGE_KEYWORDS:
            if kw in col_l and col != TARGET_COL:
                exclude.add(col)

    # Nombres de personas o columnas textuales libres que pueden memorizar.
    for col in df.columns:
        col_l = col.lower()
        if col_l.startswith("jugador") or "nombre" in col_l:
            exclude.add(col)

    return sorted(exclude)


def build_xy(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    excluded_cols = detect_excluded_cols(df)
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)
    groups = df[GROUP_COL].astype(str)

    return X, y, groups, excluded_cols


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipe = Pipeline(numeric_steps)

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_models(X: pd.DataFrame) -> Dict[str, Pipeline]:
    models = {}

    models["logistic_regression"] = Pipeline([
        ("prep", make_preprocessor(X, scale_numeric=True)),
        ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE)),
    ])

    models["random_forest"] = Pipeline([
        ("prep", make_preprocessor(X, scale_numeric=False)),
        ("model", RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
            class_weight="balanced_subsample",
        )),
    ])

    # HistGradientBoosting se omite por velocidad en este dataset pequeño.

    if XGBOOST_OK:
        models["xgboost"] = Pipeline([
            ("prep", make_preprocessor(X, scale_numeric=False)),
            ("model", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1,
                tree_method="hist",
                n_estimators=80,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.1,
                reg_lambda=3.0,
            )),
        ])

    return models


def evaluate_group_cv(models: Dict[str, Pipeline], X, y, groups) -> pd.DataFrame:
    n_groups = groups.nunique()
    n_splits = min(3, n_groups)

    cv = GroupKFold(n_splits=n_splits)
    scoring = {
        "auc": "roc_auc",
        "neg_log_loss": "neg_log_loss",
        "neg_brier": "neg_brier_score",
        "accuracy": "accuracy",
    }

    rows = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            groups=groups,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )
        rows.append({
            "modelo": name,
            "auc_cv_group_mean": scores["test_auc"].mean(),
            "auc_cv_group_std": scores["test_auc"].std(),
            "logloss_cv_group_mean": -scores["test_neg_log_loss"].mean(),
            "brier_cv_group_mean": -scores["test_neg_brier"].mean(),
            "accuracy_cv_group_mean": scores["test_accuracy"].mean(),
        })

    return pd.DataFrame(rows).sort_values("auc_cv_group_mean", ascending=False)


def diagnostic_random_vs_group(model: Pipeline, X, y, groups) -> Dict[str, float]:
    """Demuestra si el AUC sube artificialmente con split aleatorio."""
    group_cv = GroupKFold(n_splits=min(3, groups.nunique()))
    random_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    scoring = {"auc": "roc_auc"}

    s_group = cross_validate(model, X, y, groups=groups, cv=group_cv, scoring=scoring, n_jobs=1)
    s_random = cross_validate(model, X, y, cv=random_cv, scoring=scoring, n_jobs=1)

    return {
        "auc_random_cv_mean": float(s_random["test_auc"].mean()),
        "auc_group_cv_mean": float(s_group["test_auc"].mean()),
        "diferencia_random_menos_group": float(s_random["test_auc"].mean() - s_group["test_auc"].mean()),
    }


def tune_xgboost(X_train, y_train, groups_train) -> Pipeline:
    if not XGBOOST_OK:
        raise RuntimeError("xgboost no esta instalado. Instala con: pip install xgboost")

    pipe = Pipeline([
        ("prep", make_preprocessor(X_train, scale_numeric=False)),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
            tree_method="hist",
        )),
    ])

    param_dist = {
        "model__n_estimators": [50, 80, 120],
        "model__max_depth": [2, 3, 4],
        "model__learning_rate": [0.015, 0.03, 0.05, 0.08],
        "model__subsample": [0.65, 0.75, 0.85, 1.0],
        "model__colsample_bytree": [0.6, 0.75, 0.9, 1.0],
        "model__min_child_weight": [3, 5, 8, 12],
        "model__gamma": [0, 0.1, 0.3, 0.5],
        "model__reg_alpha": [0, 0.05, 0.1, 0.5],
        "model__reg_lambda": [1.5, 2.5, 4.0, 6.0],
    }

    cv = GroupKFold(n_splits=min(3, pd.Series(groups_train).nunique()))

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=6,
        scoring="roc_auc",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=1,
        refit=True,
        error_score="raise",
    )

    search.fit(X_train, y_train, groups=groups_train)
    print("\nMejores parametros XGBoost:")
    print(search.best_params_)
    print(f"Mejor AUC CV por grupos: {search.best_score_:.4f}")

    return search.best_estimator_


def evaluate_holdout(model: Pipeline, X_test, y_test) -> Dict[str, float]:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    return {
        "auc_holdout_group": float(roc_auc_score(y_test, proba)),
        "logloss_holdout_group": float(log_loss(y_test, proba, labels=[0, 1])),
        "brier_holdout_group": float(brier_score_loss(y_test, proba)),
        "accuracy_holdout_group": float(accuracy_score(y_test, pred)),
    }


def get_feature_importance(model: Pipeline) -> pd.DataFrame:
    prep = model.named_steps["prep"]
    estimator = model.named_steps["model"]

    feature_names = prep.get_feature_names_out()

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def save_report_pdf(
    path: Path,
    metrics_table: pd.DataFrame,
    holdout_metrics: Dict[str, float],
    y_test: pd.Series,
    proba_test: np.ndarray,
    feature_importance: pd.DataFrame,
    diagnostic: Dict[str, float],
):
    with PdfPages(path) as pdf:
        fig, ax = plt.subplots(figsize=(11, 7))
        ax.axis("off")
        text = (
            "PARTE 4 - MODELOS MEJORADOS PARA WINRATE DE PADEL\n\n"
            "Validacion correcta: por partido (GroupKFold / GroupShuffleSplit)\n"
            "Motivo: evitar que puntos del mismo partido queden en train y test.\n\n"
            "Diagnostico AUC alto:\n"
            f"  AUC con CV aleatoria: {diagnostic['auc_random_cv_mean']:.4f}\n"
            f"  AUC con CV por partido: {diagnostic['auc_group_cv_mean']:.4f}\n"
            f"  Diferencia: {diagnostic['diferencia_random_menos_group']:.4f}\n\n"
            "Metricas holdout por partido:\n"
            f"  AUC: {holdout_metrics['auc_holdout_group']:.4f}\n"
            f"  Log Loss: {holdout_metrics['logloss_holdout_group']:.4f}\n"
            f"  Brier: {holdout_metrics['brier_holdout_group']:.4f}\n"
            f"  Accuracy: {holdout_metrics['accuracy_holdout_group']:.4f}\n\n"
            "Conclusion:\n"
            "Un AUC cercano a 1.0 solo es aceptable si se mantiene separando partidos completos.\n"
            "Si aparece solo con split aleatorio, es fuga de informacion/memorizacion."
        )
        ax.text(0.02, 0.98, text, va="top", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 4))
        ax.axis("off")
        table_data = metrics_table.round(4).astype(str).values
        tbl = ax.table(
            cellText=table_data,
            colLabels=metrics_table.columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.5)
        ax.set_title("Comparacion de modelos - validacion por partido", pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        RocCurveDisplay.from_predictions(y_test, proba_test, ax=ax)
        ax.set_title("Curva ROC - holdout por partidos")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7, 6))
        pred_test = (proba_test >= 0.5).astype(int)
        ConfusionMatrixDisplay(confusion_matrix(y_test, pred_test)).plot(ax=ax)
        ax.set_title("Matriz de confusion - holdout por partidos")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        if not feature_importance.empty:
            top = feature_importance.head(20).iloc[::-1]
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(top["feature"], top["importance"])
            ax.set_title("Top 20 variables mas importantes")
            ax.set_xlabel("Importancia")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main():
    print("=" * 70)
    print("PARTE 4 - MODELOS MEJORADOS CON CONTROL DE LEAKAGE")
    print("=" * 70)

    df = clean_dataframe(load_dataframe())
    print(f"Dataset: {df.shape[0]} filas | {df[GROUP_COL].nunique()} partidos | {df.shape[1]} columnas")
    print(f"Distribucion target:\n{df[TARGET_COL].value_counts(normalize=True).round(4)}")

    X, y, groups, excluded_cols = build_xy(df)
    print(f"\nFeatures usadas: {X.shape[1]}")
    print(f"Columnas excluidas: {len(excluded_cols)}")
    print(", ".join(excluded_cols))

    models = make_models(X)

    print("\n[1/5] Diagnostico de AUC alto...")
    diagnostic_model = models["logistic_regression"]
    diagnostic = diagnostic_random_vs_group(diagnostic_model, X, y, groups)
    print(diagnostic)

    print("\n[2/5] Comparando modelos con GroupKFold...")
    cv_results = evaluate_group_cv(models, X, y, groups)
    print(cv_results.round(4).to_string(index=False))
    cv_results.to_csv(BASE_DIR / "comparacion_modelos_mejorados.csv", index=False)

    print("\n[3/5] Separando holdout por partidos completos...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    test_partidos = sorted(groups.iloc[test_idx].unique().tolist())

    print(f"Partidos en test: {test_partidos}")

    print("\n[4/5] Tuneando XGBoost por grupos...")
    if XGBOOST_OK:
        best_xgb = tune_xgboost(X_train, y_train, groups_train)
    else:
        print("xgboost no esta instalado; se usara Random Forest.")
        best_xgb = models["random_forest"].fit(X_train, y_train)

    holdout_metrics = evaluate_holdout(best_xgb, X_test, y_test)
    print("\nMetricas holdout por partido:")
    for k, v in holdout_metrics.items():
        print(f"{k}: {v:.4f}")

    proba_test = best_xgb.predict_proba(X_test)[:, 1]
    pred_test = (proba_test >= 0.5).astype(int)

    print("\nReporte clasificacion holdout:")
    print(classification_report(y_test, pred_test))

    feature_importance = get_feature_importance(best_xgb)
    feature_importance.to_csv(BASE_DIR / "importancia_variables_xgboost.csv", index=False)

    print("\n[5/5] Reentrenando modelo final con TODOS los partidos...")
    final_model = best_xgb
    final_model.fit(X, y)

    artifact = {
        "model": final_model,
        "feature_cols": X.columns.tolist(),
        "target_col": TARGET_COL,
        "group_col": GROUP_COL,
        "excluded_cols": excluded_cols,
        "best_model_name": "xgboost" if XGBOOST_OK else "random_forest",
        "metrics_group_cv": cv_results.to_dict(orient="records"),
        "metrics_holdout_group": holdout_metrics,
        "diagnostic_random_vs_group": diagnostic,
        "test_partidos": test_partidos,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "notes": (
            "Modelo entrenado con control de leakage. Evaluar siempre por partido, "
            "no con split aleatorio por filas."
        ),
    }

    joblib.dump(artifact, BASE_DIR / "modelo_final.pkl")
    joblib.dump(artifact, BASE_DIR / "xgboost_final.pkl")

    with (BASE_DIR / "diagnostico_modelos.txt").open("w", encoding="utf-8") as f:
        f.write("DIAGNOSTICO MODELOS PADEL\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {df.shape}\n")
        f.write(f"Partidos: {df[GROUP_COL].nunique()}\n")
        f.write(f"Features usadas: {X.shape[1]}\n\n")
        f.write("Columnas excluidas:\n")
        f.write("\n".join(excluded_cols) + "\n\n")
        f.write("Diagnostico AUC alto:\n")
        f.write(json.dumps(diagnostic, indent=2, ensure_ascii=False) + "\n\n")
        f.write("Comparacion modelos GroupKFold:\n")
        f.write(cv_results.round(4).to_string(index=False) + "\n\n")
        f.write("Holdout por partidos:\n")
        f.write(json.dumps(holdout_metrics, indent=2, ensure_ascii=False) + "\n\n")
        f.write("Classification report holdout:\n")
        f.write(classification_report(y_test, pred_test))

    save_report_pdf(
        BASE_DIR / "reporte_xgboost.pdf",
        cv_results,
        holdout_metrics,
        y_test,
        proba_test,
        feature_importance,
        diagnostic,
    )

    print("\nArchivos generados:")
    print("- modelo_final.pkl")
    print("- xgboost_final.pkl")
    print("- comparacion_modelos_mejorados.csv")
    print("- importancia_variables_xgboost.csv")
    print("- diagnostico_modelos.txt")
    print("- reporte_xgboost.pdf")
    print("\nIMPORTANTE: Si el AUC aleatorio queda muy alto y el AUC por partido baja, eso no es error.")
    print("Es la senal de que el entrenamiento anterior estaba sobreestimado por fuga de informacion.")


if __name__ == "__main__":
    main()
