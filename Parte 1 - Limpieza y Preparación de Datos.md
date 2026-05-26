# Parte 1 — Limpieza y Preparación de Datos

### Proyecto de Analítica de Datos · Pádel Universitario

---

## ¿De qué trata este notebook?

Este notebook toma tres bases de datos crudas sobre partidos de pádel recreativo jugados por estudiantes universitarios y las transforma en un dataset limpio y listo para modelar. El objetivo final es predecir si un equipo va a ganar el punto que está por jugarse.

Los datos vienen de tres fuentes:

| Base                               | Qué contiene                                                            | Filas   |
| ---------------------------------- | ----------------------------------------------------------------------- | ------- |
| `Base_Videos_Final_202601.csv`     | Posiciones y métricas físicas de cada jugador, frame a frame (~60 fps)  | 434,198 |
| `Traza_Datos_Partidos_202601.xlsx` | Registro punto a punto de cada partido: quién jugó, quién ganó          | 406     |
| `Data_Jugadores_202601.xlsx`       | Perfil deportivo de cada jugador (nivel, experiencia, condición física) | 31      |

---

## ¿Qué se hizo y en qué orden?

### 1. Normalización de nombres

Antes de cualquier limpieza se estandarizaron los nombres de los jugadores en las tres bases: minúsculas, formato título y sin tildes. Esto es crítico porque los nombres son la única llave de unión entre las tres fuentes.

---

### 2. Limpieza de `data_videos`

**Valores nulos**

| Problema                                                          | Causa                                                       | Solución                                                                                                                                                                         |
| ----------------------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nulos en `prev_x`, `prev_y`, velocidades                          | Son el primer frame de cada punto, no tienen frame anterior | Eliminar esas filas                                                                                                                                                              |
| Nulos en `time_since_last_hit`                                    | No ha ocurrido ningún golpe todavía en el punto             | Imputar con -1                                                                                                                                                                   |
| Nulos en `team`                                                   | El tracker no detectó a qué equipo pertenece el jugador     | Imputar con la moda por partido y jugador                                                                                                                                        |
| Nombres `Player_X`                                                | El tracker no reconoció al jugador por nombre               | Identificar por descarte: si en un frame hay 3 jugadores conocidos y 1 desconocido, el desconocido es el que falta. Los que no se pudieron resolver (9,716 frames) se eliminaron |
| Nulos en `zone` con valores `left_box`, `right_box`, `back_court` | El tracker no estaba detectando bien al jugador             | Normalizar a `left`/`right` e imputar con la moda por partido y jugador                                                                                                          |
| Nulos en `distance_player_to_teammate_m`                          | Solo uno de los dos compañeros tenía la distancia calculada | Copiar del compañero o calcular con las posiciones disponibles                                                                                                                   |
| Nulos residuales en variables de pelota                           | Frames sin posición previa de pelota calculable             | Calcular con `shift(1)` o eliminar los 285 restantes                                                                                                                             |

**Outliers**

Se comparó el método IQR contra límites físicos reales (ej: velocidad máxima humana ≈ 7 m/s, velocidad máxima de pelota ≈ 50 m/s). El IQR no funcionó bien porque muchas variables tienen distribuciones muy sesgadas hacia cero. Se optó por **clipeo con límites físicos**.

**Shape tras limpieza:** 421,406 filas · 30 columnas

---

### 3. Limpieza de `data_partidos`

No se encontraron valores nulos. El único ajuste fue formatear algunos nombres que estaban en mayúsculas al mismo formato título que el resto.

---

### 4. Limpieza de `data_jugadores`

Se eliminaron columnas irrelevantes para el rendimiento deportivo (ciudad, programa académico, género musical, etc.) y se aplicó encoding a las variables que sí importan:

- **Ordinal:** `NIVEL_ACTUAL_PADEL`, `TIEMPO_JUGANDO_PADEL`, `ESTADO_FISICO`, `FRECUENCIA_DEPORTE`, `CLASES_PADEL`
- **Binario:** `EXPERIENCIA_PADEL`, `LESIONES_LY`, `PRACTICA_OTRO_DEPORTE_RAQUETA`

Las columnas `Puntos_Ganados`, `Puntos_Jugados` y `Win_Rate` se eliminaron porque se recalculan desde `data_partidos`, que es más confiable.

**Shape final:** 31 jugadores · 13 columnas

---

### 5. Construcción de variables de contexto del partido

A partir de `data_partidos` se construyeron variables que describen el estado del partido en el momento de cada punto:

| Variable                 | Qué captura                                               |
| ------------------------ | --------------------------------------------------------- |
| `MARCADOR_EQUIPO_1/2`    | Puntos acumulados por cada equipo hasta ese momento       |
| `DIFERENCIA_MARCADOR`    | Quién va ganando y por cuánto                             |
| `WINRATE_ACUM_E1/E2`     | Proporción de puntos ganados hasta ahora                  |
| `RACHA_EQUIPO_1/2`       | Puntos consecutivos ganados en ese momento                |
| `PUNTOS_ULTIMOS_5_E1/E2` | Momentum reciente (últimos 5 puntos)                      |
| `EQUIPO_GANADOR`         | Quién ganó el partido (variable objetivo a nivel partido) |

También se corrigió un error de registro en el **partido 8**, donde todos los puntos tenían `ID_PUNTO = 1`. Se reasignaron secuencialmente.

---

### 6. Validación de posiciones

Se cruzaron las posiciones (`team`, `zone`) de `data_videos` contra `data_partidos`, dando prioridad a esta última por ser el registro manual. El lookup se construyó a nivel de partido **y punto** para capturar los cambios de posición entre puntos. Se encontraron 421,578 discrepancias en `team` (95.66%) y 284,819 en `zone` (64.63%), lo que confirmó que la imputación por moda del paso anterior no era suficientemente precisa.

---

### 7. Agregación de `data_videos` por punto

Como el modelo opera a nivel de punto (no de frame), se agregaron los 421,406 frames en 287 puntos calculando por equipo:

- Velocidad promedio y máxima
- Desplazamiento total
- Distancia promedio a la red y al compañero
- Número de golpes
- Aceleración promedio
- Velocidad promedio y máxima de la pelota

---

### 8. Join de las bases de datos

Se unieron `data_videos_agg` + `data_partidos` por partido y punto (inner join), obteniendo 283 filas y 35 columnas.

---

### 9. Perspectiva dual

Para eliminar el sesgo de qué equipo ocupó TOP o BOTTOM en la cancha, **cada punto se duplicó**: una fila desde la perspectiva del equipo 1 y otra desde la del equipo 2. En cada fila las variables se renombraron como `_equipo` (equipo focal) y `_rival`. Esto también duplica el tamaño del dataset.

**Shape:** 566 filas · 28 columnas

---

### 10. Variables acumuladas

Para capturar el historial del partido sin fuga de información, se calcularon estadísticas acumuladas **hasta el punto anterior** (nunca incluyendo el punto actual):

- Promedios acumulados de velocidad, aceleración, distancias
- Golpes y desplazamiento acumulados
- Win rate acumulado del equipo focal
- Racha de los últimos 3 puntos

Los nulos del primer punto (sin historial previo) se imputaron con 0 para variables de conteo y con la mediana del partido para variables físicas. Se creó la variable `es_primer_punto` como indicador.

---

### 11. Tratamiento de nulos por fallas de tracking

Quedaron 123 nulos en variables del punto actual. Diagnóstico por partido:

| Categoría              | Partidos                             | Causa                                                   | Tratamiento                     |
| ---------------------- | ------------------------------------ | ------------------------------------------------------- | ------------------------------- |
| Sin tracking en origen | 2, 4, 12, 18, 20, 24, 28, 30, 32, 35 | El video nunca capturó a uno de los equipos             | Eliminar filas                  |
| Tracking intermitente  | 8, 34                                | Jugadores detectados en algunos puntos pero no en todos | Imputar con mediana del partido |

---

### 12. Eliminación de partidos con video incompleto

Los partidos 14, 31 y 33 tenían registro completo en `data_partidos` pero solo entre 3 y 6 puntos grabados en video (de 10-12 totales). Al confirmar que los puntos faltantes no existían en el CSV crudo, se eliminaron estos partidos para no comprometer las variables acumuladas.

---

## Dataset final

|                   | Valor                                            |
| ----------------- | ------------------------------------------------ |
| Filas             | 338                                              |
| Columnas          | 48                                               |
| Partidos          | 17                                               |
| Variable objetivo | `gano_punto` (1 = el equipo focal ganó el punto) |

---

## Dataframes exportados

Se generaron dos versiones del dataset para comparar el poder predictivo de las variables de perfil de jugador:

### `data_sin_jugadores`

El dataset base. Solo contiene métricas físicas del tracking y variables del marcador. Permite evaluar si el comportamiento en cancha es suficiente para predecir el resultado.

**Shape:** 338 filas · 48 columnas

---

### `data_con_jugadores`

Incorpora el perfil deportivo de los 4 jugadores del punto. Los 9 jugadores sin formulario se imputaron con mediana/moda del resto.

Las variables de perfil se agregan 4 veces con el sufijo del rol:
`_jugador_1_equipo`, `_jugador_2_equipo`, `_jugador_1_rival`, `_jugador_2_rival`

| Variable agregada                | Tipo        | Descripción                         |
| -------------------------------- | ----------- | ----------------------------------- |
| `edad_`                          | entero      | Edad del jugador                    |
| `genero_`                        | texto       | Género del jugador                  |
| `experiencia_padel_`             | binario     | Experiencia previa en pádel         |
| `tiempo_jugando_padel_`          | ordinal 0-2 | Tiempo practicando pádel            |
| `practica_otro_deporte_raqueta_` | binario     | Si practica tenis, squash, etc.     |
| `nivel_actual_padel_`            | ordinal 0-2 | Nivel autorreportado                |
| `estado_fisico_`                 | ordinal 0-3 | Condición física percibida          |
| `frecuencia_deporte_`            | ordinal 0-3 | Frecuencia semanal de entrenamiento |
| `estatura_`                      | decimal     | Estatura en cm                      |
| `talla_`                         | decimal     | Talla de ropa codificada            |
| `lesiones_ly_`                   | binario     | Lesiones en el último año           |
| `clases_padel_`                  | ordinal 0-2 | Si ha tomado clases formales        |

**Shape:** 338 filas · 144 columnas

---

## Archivos generados

```
dataframes/
├── data_sin_jugadores.pkl
├── data_sin_jugadores.csv
├── data_con_jugadores.pkl
└── data_con_jugadores.csv
```
