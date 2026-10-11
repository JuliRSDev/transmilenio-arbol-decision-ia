# Descripción de los datos

**Proyecto:** Predicción del nivel de ocupación de estaciones de TransMilenio con árboles de decisión
**Actividad 3:** Métodos de aprendizaje supervisado

---

## 1. Problema de aprendizaje supervisado

Se quiere **predecir el nivel de ocupación** de una estación de TransMilenio en una
hora determinada. El nivel puede ser **Bajo, Medio, Alto o Crítico**, y la
predicción se hace a partir de información que se conoce de antemano: la hora,
el tipo de día, la zona de la estación, si llueve y si hay un evento masivo.

Es **supervisado** porque cada registro histórico trae la respuesta correcta (la
etiqueta `nivel_ocupacion`), y el modelo aprende de esos ejemplos.

| Nivel | Ocupación de la estación |
|---|---|
| Bajo | menos del 40 % |
| Medio | del 40 % al 69,9 % |
| Alto | del 70 % al 89,9 % |
| Crítico | 90 % o más (hacinamiento) |

---

## 2. Fuentes de datos reales identificadas

| # | Fuente | Publicador | Qué aporta |
|---|---|---|---|
| 1 | **Consolidado de salidas del sistema troncal por franja horaria**: [datosabiertos.bogota.gov.co](https://datosabiertos.bogota.gov.co/dataset/consolidado-de-salidas-sistema-troncal-por-franja-horaria) | TransMilenio S.A., en Datos Abiertos Bogotá (CC BY 4.0) | Afluencia de usuarios por estación y franja horaria. Es la base para medir la ocupación. |
| 2 | **Estaciones troncales de TRANSMILENIO**: [datosabiertos-transmilenio.hub.arcgis.com](https://datosabiertos-transmilenio.hub.arcgis.com/datasets/estaciones-troncales-de-transmilenio/explore) | TransMilenio S.A. | Catálogo de estaciones, troncal y ubicación. |
| 3 | **Datos hidrometeorológicos (precipitación y temperatura)**: [datos.gov.co](https://www.datos.gov.co) | IDEAM | Lluvia y temperatura por estación meteorológica y hora. |
| 4 | **Calendario de festivos de Colombia** (Ley 51 de 1983) | Gobierno de Colombia | Identifica días hábiles, sábados y domingos o festivos. |
| 5 | **Agenda de eventos masivos** (estadio El Campín, Movistar Arena, parque Simón Bolívar) | IDRD y operadores de escenarios | Fechas y horas de conciertos, partidos y festivales. |

### ¿Por qué se usa una muestra?

Estas fuentes existen, pero están en portales distintos, con formatos distintos
(CSV mensuales, capas geográficas, series del IDEAM) y con volúmenes muy grandes
que habría que cruzar por estación, fecha y hora. Por eso, como lo permite la
actividad (*"en caso de no existir dichas fuentes de datos, desarrolle un dataset
con una muestra de dichos datos"*), se construyó una **muestra con la misma
estructura**. La genera `src/generar_dataset.py` con semilla fija (2026), así que
siempre produce los mismos datos.

> **Aviso:** los nombres de estaciones y troncales son **reales**. La ocupación,
> la lluvia, la temperatura y los eventos son **simulados** con fines académicos.
> No son cifras oficiales de TransMilenio S.A. ni del IDEAM.

---

## 3. Archivos de datos

### 3.1 `datos/estaciones.csv` (30 filas)

| Columna | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `estacion_id` | texto | Identificador único | `S01` |
| `nombre` | texto | Nombre real de la estación | `Portal Norte` |
| `troncal` | texto | Troncal a la que pertenece | `Autonorte` |
| `tipo_estacion` | categoría | `Portal`, `Intermedia`, `Intercambio` o `Cabecera` | `Portal` |
| `zona_uso` | categoría | Uso del suelo predominante: `Residencial`, `Empresarial`, `Universitaria`, `Intercambio` o `Recreativa` | `Residencial` |

Distribución por zona: 10 estaciones residenciales, 6 empresariales, 5
universitarias, 5 de intercambio y 4 recreativas (Movistar Arena, El Campín,
Coliseo y Simón Bolívar).

### 3.2 `datos/calendario.csv` (28 filas, del 1 al 28 de octubre de 2026)

| Columna | Tipo | Descripción |
|---|---|---|
| `fecha` | fecha | Día (AAAA-MM-DD) |
| `dia_semana` | texto | Lunes a domingo |
| `es_festivo` | 0/1 | 1 si es festivo (el lunes 12 de octubre, Día de la Raza) |
| `tipo_dia` | categoría | `Hábil` (19 días), `Sábado` (4) o `Domingo/Festivo` (5) |
| `lluvia_manana` | 0/1 | Llovió en la mañana (4 días) |
| `lluvia_tarde` | 0/1 | Llovió en la tarde (12 días; octubre es temporada de lluvias) |

### 3.3 `datos/eventos.csv` (10 filas)

| Columna | Descripción |
|---|---|
| `fecha` | Día del evento |
| `estacion_cercana` | Estación de TransMilenio más cercana al escenario |
| `tipo_evento` | Concierto, partido de fútbol o festival al parque |
| `hora_inicio`, `hora_fin` | Horas en que el evento afecta la estación |

### 3.4 `datos/ocupacion_horaria.csv` (15.135 filas)

Contiene una fila por **estación × fecha × hora**: 30 estaciones × 28 días × 18
horas (de 5:00 a 22:00), más 15 filas duplicadas.

| Columna | Tipo | Descripción | ¿Se usa en el modelo? |
|---|---|---|---|
| `fecha` | fecha | Día | Sirve para unir con el calendario |
| `estacion_id` | texto | Estación | Sirve para unir con estaciones |
| `hora` | entero 5 a 22 | Hora de inicio de la franja | **Entrada** |
| `lluvia` | 0/1 | Llovía en esa hora (60 valores vacíos) | **Entrada** |
| `temperatura_c` | decimal | Temperatura en °C | **Entrada** |
| `evento_masivo` | 0/1 | Había un evento cerca en esa hora (60 registros) | **Entrada** |
| `ocupacion_pct` | decimal | Ocupación de la estación en % | **No.** De aquí sale la etiqueta, así que usarla sería fuga de información |
| `nivel_ocupacion` | categoría | **Etiqueta**: Bajo / Medio / Alto / Crítico | **Objetivo (y)** |

**Estadísticas** (datos sin limpiar):

| | hora | temperatura_c | ocupacion_pct |
|---|---|---|---|
| Media | 13,5 | 14,1 | 34,2 |
| Desviación estándar | 5,2 | 3,1 | 23,8 |
| Mínimo | 5 | 4,7 | 2,0 |
| Mediana | 13 | 13,9 | 27,3 |
| Máximo | 22 | 22,2 | 125,0 |

**Distribución de la etiqueta** (después de limpiar): Bajo 10.037 (66 %), Medio
3.667 (24 %), Alto 864 (6 %) y Crítico 567 (4 %). Las clases están
**desbalanceadas**, como pasa en la realidad: la mayor parte del día las
estaciones no están llenas. Por eso el modelo se evalúa con **F1 macro** y no
solo con exactitud.

---

## 4. Cómo se simularon los datos

```
ocupación = curva_base(zona, hora) × capacidad_estación × factor_día
            × 1,12 si llueve
            + 65 puntos si hay evento masivo cerca
            + sesgo propio de la estación + ruido aleatorio
```

- **Curva base por zona:**
  - Residencial: pico a las 6 a. m.
  - Empresarial: pico a las 5 o 6 p. m.
  - Universitaria: pico al mediodía.
  - Intercambio: picos en la mañana y en la tarde.
  - Recreativa: baja, salvo cuando hay eventos.
- **Factor según el tipo de día:** hábil 1,0; sábado 0,55; domingo o festivo 0,35.
- **La temperatura NO afecta la ocupación.** Se incluyó a propósito como variable
  *distractora*, para ver si el árbol aprende relaciones que no existen (ver la
  prueba E4 en `docs/pruebas.md`).

### Problemas de calidad inyectados (para practicar la limpieza)

| Problema | Cantidad | Tratamiento |
|---|---|---|
| Registros duplicados | 15 | `drop_duplicates()` |
| Etiquetas mal digitadas (`"BAJO "`, `"MEDIO "`, `"ALTO "`) | 25 | `.str.strip().str.capitalize()` |
| Lluvia sin dato (falla del sensor) | 60 | Se completa con `calendario.csv`: lluvia de la mañana si la hora es antes de las 12, lluvia de la tarde si es después |

---

## 5. Variables de entrada del modelo

Después de unir las fuentes, se agrega la variable derivada `franja_horaria`
(Madrugada, Pico AM, Valle, Pico PM o Noche). Las variables categóricas se
convierten a números con **one-hot encoding**. El resultado son **21 columnas de
entrada**:

- **Numéricas:** `hora`, `lluvia`, `temperatura_c`, `evento_masivo`.
- **One-hot:** `franja_horaria_*` (5), `tipo_dia_*` (3), `tipo_estacion_*` (4) y
  `zona_uso_*` (5).
