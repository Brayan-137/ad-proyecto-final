# Objetivo/Requerimiento

Crear el mejor modelo posible para predecir el equipo ganador de un partido tras cada punto marcado por alguno de los dos equipos durante el partido

> ./databases
> Los archivos csv (Base_Videos_Final_202502 y Base_Videos_Final_202601)se exluyeron del repositorio en main por su tamaño

# Resumen de Progreso — Proyecto Analítica de Datos Pádel (2026-1)

## Objetivo del proyecto

Predecir la probabilidad de ganar un partido de pádel tras cada punto anotado, usando datos de tracking de video (computer vision) y registros manuales de partidos jugados por estudiantes universitarios.

---

## Bases de datos

Se trabaja con tres fuentes independientes para el periodo 2026-1:

- **`data_videos`** (`Base_Videos_Final_202601.csv`): datos frame a frame extraídos por computer vision. Cada fila es un frame de un punto de un partido, con 30 variables de posición y movimiento de jugadores y pelota. Shape final: **(431,122 × 30)**.
- **`data_partidos`** (`Traza_Datos_Partidos_202601.xlsx`): registro punto a punto de cada partido. Tiene dos hojas (CANCHA_1 y CANCHA_2), ambas cargadas y concatenadas.
- **`data_jugadores`** (`Data_Jugadores_202601.xlsx`): perfil de cada jugador con variables demográficas, deportivas y de rendimiento (`Win_Rate`, `Puntos_Ganados`, `Puntos_Jugados`).

---

## Lo que se ha hecho

### 1. Limpieza de `data_videos`

**Valores nulos:**

- Frame 0 de cada punto eliminado (nulos estructurales en variables de movimiento).
- Frames residuales con `prev_speed` y `player_acceleration_mps2` nulos eliminados.
- `time_since_last_hit` imputado con `-1` (indica que aún no ha ocurrido ningún golpe en el punto).
- `team` imputado con moda por punto → partido → jugador.
- `distance_player_to_teammate_m` recalculada con distancia euclidiana usando `transform`.
- Variables de pelota (`ball_position_x_prev`, etc.) recalculadas con `shift`; nulos irrecuperables eliminados (< 0.2%).
- `team` normalizado a mayúsculas.

**Outliers:**

- Método de límites físicos con clipeo (no eliminación), dado que IQR colapsaba por concentración de ceros.

```python
limites_fisicos = {
    'player_speed_mps':              (0, 7),
    'player_acceleration_mps2':      (-10, 10),
    'ball_speed_mps':                (0, 50),
    'distance_player_to_ball_m':     (0, 20),
    'distance_player_to_net_m':      (0, 10),
    'distance_player_to_teammate_m': (0, 10),
    'player_displacement':           (0, 2),
}
```

### 2. Limpieza de `data_partidos`

- Nombres de jugadores normalizados a título (mayúscula inicial) en las columnas de jugadores.
- Construcción del marcador acumulado por partido (`MARCADOR_EQUIPO_1`, `MARCADOR_EQUIPO_2`).
- Nota: el partido ID 23 terminó 7-6 (diferencia de 1) — tratado como error de registro.

### 3. Normalización de nombres entre bases

Se detectó que los nombres de jugadores tenían inconsistencias de capitalización y tildes entre `data_videos` y `data_partidos`. Se aplicó una función de normalización en ambas bases:

- Conversión a minúsculas → título
- Eliminación de tildes con `unicodedata`
- Strip de espacios

### 4. Verificación y corrección de posiciones TOP/BOTTOM

Se construyó un procedimiento de validación cruzando la columna `team` de `data_videos` contra las columnas `EQUIPO_1_POSICION` / `EQUIPO_2_POSICION` de `data_partidos`, por partido y punto.

**Hallazgos:**

- 209 casos de jugadores que aparecían asignados a ambos equipos en el mismo punto (ruido del tracking).
- Varios partidos con inversión sistemática de TOP/BOTTOM en `data_videos`.
- Partidos con jugadores no registrados en `data_partidos` (Player_X sin nombre real).

**Solución aplicada:**
Se usó `data_partidos` como fuente de verdad. Se construyó un lookup `partido → jugador → team_correcto` y se reasignó `team` en `data_videos` para todos los jugadores con nombre identificado:

- **96.7% de frames corregidos** (416,907 frames)
- **3.3% sin match** (14,215 frames correspondientes a Player_X)

### 5. Agregación de `data_videos` a nivel de punto

Se construyó `data_videos_agg` agrupando por `partido` y `punto`, calculando estadísticas separadas por equipo (BOTTOM = equipo 1 según `data_partidos`, TOP = equipo 2). Shape: **(287 × 19)**.

Variables agregadas:

- `duracion_punto`
- `velocidad_prom_e1/e2`, `velocidad_max_e1/e2`
- `desplazamiento_total_e1/e2`
- `dist_red_prom_e1/e2`
- `dist_companero_prom_e1/e2`
- `hits_e1/e2`
- `velocidad_prom_pelota`, `velocidad_max_pelota`
- `aceleracion_prom_e1/e2`

### 6. Join entre `data_videos_agg` y `data_partidos`

Se realizó un join `inner` por `partido` y `punto`, generando el dataframe consolidado **`data`**.

---

## Pendientes

1. **Construir la variable objetivo** (`EQUIPO_GANADOR`: 1 si gana equipo 1, 2 si gana equipo 2) y las variables de contexto por punto:
   - `DIFERENCIA_MARCADOR`
   - `PUNTOS_JUGADOS`
   - `WINRATE_ACUM_E1`, `WINRATE_ACUM_E2`
   - `RACHA_EQUIPO_1`, `RACHA_EQUIPO_2`
   - `PUNTOS_ULTIMOS_5_E1`, `PUNTOS_ULTIMOS_5_E2`

2. **Integrar `data_jugadores`** — enriquecer `data` con el perfil de cada jugador (experiencia, nivel, condición física, etc.).

3. **Análisis exploratorio (EDA)** sobre `data` consolidado — distribuciones, correlaciones, comportamiento de variables por equipo ganador.
