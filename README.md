## Resumen del proyecto — Analítica de Pádel 2026-1 (18 de mayo)

### Objetivo

Construir un modelo que prediga el equipo ganador de un partido de pádel tras cada punto jugado, usando datos de tracking de video (computer vision) y registros manuales de partidos.

### Contexto del deporte

Los partidos son jugados por estudiantes universitarios en formato de puntos directos: gana el primer equipo en llegar a 7 puntos con al menos 2 de diferencia. Cada partido enfrenta dos equipos de dos jugadores.

---

### Fuentes de datos

Se trabaja con tres bases independientes del periodo 2026-1:

| Base             | Descripción                                                                                            | Shape original |
| ---------------- | ------------------------------------------------------------------------------------------------------ | -------------- |
| `data_videos`    | Datos frame a frame (60fps) extraídos por computer vision. Posición y movimiento de jugadores y pelota | 434,198 × 30   |
| `data_partidos`  | Registro punto a punto de cada partido (2 canchas)                                                     | 406 × 15       |
| `data_jugadores` | Perfil demográfico y deportivo de cada jugador                                                         | 31 × 30        |

---

### Lo que se ha hecho

**1. Limpieza de `data_videos`**

- Eliminación de primeros frames de cada punto (nulos estructurales en variables de movimiento).
- Imputación de `time_since_last_hit` con −1 (indica que aún no se ha golpeado la pelota en el punto).
- Imputación de `team` por moda a nivel partido y jugador, usando `data_partidos` como fuente de verdad. El 95.7% de los frames quedaron correctamente asignados.
- Jugadores no identificados (`Player_X`) corregidos por descarte cuando era posible (4,499 corregidos); los 9,716 irrecuperables fueron eliminados.
- Imputación de `zone` por moda de partido y jugador; valores `left_box`/`right_box` normalizados a `left`/`right`.
- `distance_player_to_teammate_m` recalculada con distancia euclidiana cuando faltaba.
- Variables de pelota (`ball_position_x_prev`, etc.) recalculadas con `shift`; nulos residuales eliminados.
- Outliers tratados con clipeo por límites físicos (no eliminación), dado que IQR colapsaba por concentración de ceros.
- Shape final: **421,406 × 30**

**2. Limpieza de `data_partidos`**

- Nombres de jugadores normalizados (título, sin tildes, sin espacios extra).
- Construcción del marcador acumulado punto a punto.
- Corrección del partido 8: `ID_PUNTO` estaba siempre en 1 por error de registro; se reasignó la secuencia real usando el marcador acumulado.

**3. Variables de contexto construidas desde `data_partidos`**

| Variable                 | Descripción                                            |
| ------------------------ | ------------------------------------------------------ |
| `MARCADOR_EQUIPO_1/2`    | Puntos acumulados por cada equipo al momento del punto |
| `EQUIPO_GANADOR`         | Variable objetivo: equipo que ganó el partido (1 o 2)  |
| `DIFERENCIA_MARCADOR`    | Diferencia de marcador entre equipos (E1 − E2)         |
| `PUNTOS_JUGADOS`         | Total de puntos disputados hasta ese momento           |
| `WINRATE_ACUM_E1/E2`     | Proporción de puntos ganados acumulada (0–1)           |
| `RACHA_EQUIPO_1/2`       | Puntos consecutivos ganados en ese momento             |
| `PUNTOS_ULTIMOS_5_E1/E2` | Puntos ganados en los últimos 5 puntos                 |

**4. Agregación de `data_videos` a nivel de punto**

Se colapsaron los 421,406 frames a una fila por punto, calculando estadísticas separadas por equipo (BOTTOM = equipo 1, TOP = equipo 2): velocidad promedio y máxima, desplazamiento total, distancia a la red, distancia al compañero, golpes, velocidad promedio y máxima de pelota, aceleración promedio. Shape: **287 × 19**

**5. Join `data_videos_agg` + `data_partidos`**

Join inner por partido y punto. 20 puntos sin match fueron analizados y descartados justificadamente (ruido de tracking, registros incompletos). Shape final: **283 × 35**

**6. Limpieza de `data_jugadores`**

Se eliminaron variables irrelevantes para el modelo (ciudad, programa académico, género musical, etc.) y se aplicó encoding ordinal a variables de experiencia y condición física. `Win_Rate`, `Puntos_Ganados` y `Puntos_Jugados` se eliminaron — se calcularán desde `data_partidos` que es más confiable. Shape final: **31 × 13**

---

### Estado actual

El dataframe `data` (283 × 35) está listo con todas las variables de tracking y contexto de partido. El siguiente paso es hacer el join con `data_jugadores` para enriquecer cada punto con el perfil de los cuatro jugadores participantes, y posteriormente realizar el EDA y el modelado predictivo.
