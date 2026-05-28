# Parte Web — Pádel Winrate Live

Esta parte agrega una interfaz web para cargar un video del partido y visualizar el winrate en tiempo real por puntos.

## Flujo

1. Ejecutar la Parte 4 corregida para crear `modelo_final.pkl`.
2. Ejecutar la Parte 5 corregida para crear `frontend_data.json`.
3. Levantar la web con FastAPI.
4. Cargar un video desde el navegador.
5. Seleccionar el partido.
6. El dashboard actualiza marcador, probabilidad y favorito según el tiempo del video.

## Comandos

```bash
pip install fastapi uvicorn python-multipart pandas numpy scikit-learn xgboost joblib
python parte_4_xgboost_final_corregido.py
python parte_5_frontend_api_corregido.py
uvicorn parte_web.api:app --reload
```

Abrir:

```text
http://127.0.0.1:8000
```

## Nota importante sobre “en vivo”

Este módulo actualiza el winrate después de cada punto. Para análisis 100% automático desde video crudo se necesitaría una etapa adicional de computer vision que detecte fin de punto, marcador y ganador del punto desde la imagen. En este proyecto, la sincronización se hace con las columnas `timestamp`, `frame` o con la duración acumulada de los puntos.
