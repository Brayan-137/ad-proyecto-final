# Parte Web — modo recomendado: clips por punto

Esta version de la web esta pensada para trabajar con un video corto por cada punto.
Es la opcion recomendada cuando cada partido tiene pocos puntos registrados en el dataset.

## Flujo correcto

Desde la raiz del proyecto:

```bash
python parte_5_frontend_api.py
python -m uvicorn parte_web.api:app --reload
```

Luego abrir:

```text
http://127.0.0.1:8000
```

## Como nombrar los clips

Para que el sistema los asocie automaticamente con cada punto, usa nombres como:

```text
partido_6_punto_1.mp4
partido_6_punto_2.mp4
partido_6_punto_3.mp4
```

Tambien acepta nombres como:

```text
punto_1.mp4
p1.mp4
p6_p1.mp4
clip_1.mp4
```

Lo mas seguro es usar:

```text
partido_ID_punto_NUMERO.mp4
```

Ejemplo para el partido 6:

```text
partido_6_punto_1.mp4
partido_6_punto_2.mp4
partido_6_punto_3.mp4
partido_6_punto_4.mp4
partido_6_punto_5.mp4
partido_6_punto_6.mp4
partido_6_punto_7.mp4
```

## Como funciona

- El JSON trae el marcador, probabilidades y variables de tracking por punto.
- La web no intenta sincronizar un video largo.
- Al seleccionar un punto, carga el clip correspondiente.
- Al terminar un clip, puede avanzar automaticamente al siguiente punto.
- El winrate se actualiza punto a punto.

## Importante

La probabilidad mostrada no es la probabilidad de ganar el siguiente punto. Es la probabilidad de ganar el partido despues de observar el estado del partido en ese punto.
