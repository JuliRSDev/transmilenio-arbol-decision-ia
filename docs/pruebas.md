# Documento de pruebas

**Proyecto:** Predicción del nivel de ocupación de estaciones de TransMilenio con árboles de decisión
**Actividad 3:** Métodos de aprendizaje supervisado

Este documento tiene dos partes:

- **Parte A:** 15 pruebas automáticas, con `unittest`.
- **Parte B:** ejecución completa del modelo y 4 experimentos.

Las salidas de consola son copia literal de las ejecuciones.

**Entorno:** Python 3.10, pandas 2.3, scikit-learn 1.7 y matplotlib 3.10.

---

## Parte A: pruebas automáticas

```bash
python -m unittest discover -s tests -v
```
```
test_p01_fuentes_con_columnas_esperadas (test_proyecto.PruebasDatos) ... ok
test_p02_limpieza_sin_duplicados_ni_nulos (test_proyecto.PruebasDatos) ... ok
test_p03_etiquetas_validas (test_proyecto.PruebasDatos) ... ok
test_p04_etiqueta_coherente_con_ocupacion (test_proyecto.PruebasDatos) ... ok
test_p05_detecta_etiqueta_desconocida (test_proyecto.PruebasDatos) ... ok
test_p09_supera_linea_base (test_proyecto.PruebasModelo) ... ok
test_p10_poda_reduce_sobreajuste (test_proyecto.PruebasModelo) ... ok
test_p11_reproducible (test_proyecto.PruebasModelo) ... ok
test_p12_reglas_exportables (test_proyecto.PruebasModelo) ... ok
test_p13_escenario_hora_pico_portal_es_critico (test_proyecto.PruebasModelo) ... ok
test_p14_evento_masivo_sube_la_ocupacion (test_proyecto.PruebasModelo) ... ok
test_p15_domingo_mediodia_universidad_es_bajo (test_proyecto.PruebasModelo) ... ok
test_p06_sin_fuga_de_informacion (test_proyecto.PruebasPreparacion) ... ok
test_p07_X_solo_numerica (test_proyecto.PruebasPreparacion) ... ok
test_p08_division_estratificada (test_proyecto.PruebasPreparacion) ... ok

----------------------------------------------------------------------
Ran 15 tests

OK
```

| ID | Qué se prueba | Resultado esperado | Estado |
|---|---|---|---|
| P01 | Las fuentes tienen las columnas y el tamaño documentados | 8 columnas, 30 estaciones y 28 días | OK |
| P02 | La limpieza quita duplicados y vacíos | 15 duplicados eliminados; 0 nulos en lluvia | OK |
| P03 | Solo quedan las 4 etiquetas válidas | {Bajo, Medio, Alto, Crítico} | OK |
| P04 | La etiqueta es coherente con la ocupación | Bajo < 40 %; Crítico ≥ 90 % | OK |
| P05 | Se detecta una etiqueta desconocida | Lanza `ValueError` | OK |
| P06 | No hay fuga de información | `ocupacion_pct` no está en X | OK |
| P07 | X es totalmente numérica (one-hot) | Solo columnas int o float | OK |
| P08 | La división es estratificada | 25 % para prueba; misma proporción de clases (±1 %) | OK |
| P09 | El árbol supera la línea base | F1 macro mayor que la línea base + 0,4; exactitud > 0,85 | OK |
| P10 | La poda reduce el sobreajuste | Menor brecha entre entrenamiento y prueba; menos hojas | OK |
| P11 | Reproducibilidad | Mismas predicciones al reentrenar | OK |
| P12 | El árbol se puede exportar como reglas | Una regla por hoja | OK |
| P13 | Portal residencial, día hábil, 6 a. m., con lluvia | Crítico | OK |
| P14 | Un evento masivo sube el nivel predicho | Con evento > sin evento | OK |
| P15 | Universidad, domingo al mediodía | Bajo | OK |

---

## Parte B: ejecución completa y experimentos

### E1. Generar el dataset

```bash
python src/generar_dataset.py
```
```
estaciones.csv        -> 30 estaciones
calendario.csv        -> 28 dias (2026-10-01 a 2026-10-28)
eventos.csv           -> 10 eventos masivos
ocupacion_horaria.csv -> 15135 registros
```

### E2. Ejecución completa del modelo

```bash
python src/main.py
```

**Limpieza (paso 2):**
```
Filas iniciales:                15135
Duplicados eliminados:          15
Etiquetas mal escritas arregladas: 25
Lluvia faltante completada:     60 (desde calendario.csv)
Filas finales:                  15120
```

**División (paso 4):** 11.340 ejemplos de entrenamiento y 3.780 de prueba.

**Línea base y árbol sin podar (paso 5):**
```
Línea base (siempre dice 'Bajo'): exactitud 0.663 | F1 macro 0.199
Árbol sin podar: profundidad 26, 1573 hojas
   F1 macro en ENTRENAMIENTO: 0.930
   F1 macro en PRUEBA:        0.745   <- sobreajuste: memoriza
```
El árbol sin límites crece hasta 26 niveles y 1.573 hojas. Acierta mucho con
los datos que ya vio (0,930), pero bastante menos con datos nuevos (0,745).
Eso es **sobreajuste**.

**Búsqueda del mejor árbol, es decir, la poda (paso 6):**
```
Combinaciones evaluadas: 70 x 5 particiones
Mejores hiperparámetros: {'criterion': 'entropy', 'max_depth': 10, 'min_samples_leaf': 10}
F1 macro promedio en validación cruzada: 0.799
Árbol final: profundidad 10, 243 hojas
```
La validación cruzada eligió la **entropía**, es decir, la ganancia de
información de ID3/C4.5 (0,799 frente a 0,795 con Gini). También eligió una
profundidad máxima de 10 y un mínimo de 10 ejemplos por hoja.

**Evaluación final con los datos de prueba (paso 7):**
```
Exactitud: 0.904 | F1 macro: 0.781

              precision    recall  f1-score   support

        Bajo      0.949     0.975     0.962      2507
       Medio      0.864     0.820     0.841       916
        Alto      0.604     0.620     0.612       216
     Crítico      0.796     0.638     0.709       141

    accuracy                          0.904      3780
   macro avg      0.803     0.763     0.781      3780
weighted avg      0.903     0.904     0.903      3780

Matriz de confusión (filas = real, columnas = predicho):
         Bajo  Medio  Alto  Crítico
Bajo     2444     63     0        0
Medio     126    751    39        0
Alto        4     55   134       23
Crítico     2      0    49       90
```

| Modelo | Exactitud | F1 macro |
|---|---|---|
| Línea base (siempre "Bajo") | 0,663 | 0,199 |
| Árbol sin podar (profundidad 26) | 0,874 | 0,745 |
| **Árbol podado (profundidad 10)** | **0,904** | **0,781** |

**Interpretación:**

- El árbol podado es mejor que el árbol sin podar y con menos de la sexta parte
  de las hojas: generaliza mejor.
- Casi todos los errores son entre **niveles vecinos** (Alto ↔ Crítico, Medio ↔
  Alto). Casi nunca confunde Bajo con Crítico: solo 2 de 141 casos críticos se
  predijeron como Bajo.
- Las clases más difíciles son **Alto y Crítico**, porque tienen pocos ejemplos
  (6 % y 4 % de los datos).

**Importancia de variables (paso 8):**

| Variable | Importancia |
|---|---|
| hora | 14,7 % |
| franja_horaria_Pico PM | 11,7 % |
| tipo_dia_Hábil | 11,5 % |
| zona_uso_Intercambio | 9,6 % |
| zona_uso_Universitaria | 9,3 % |
| zona_uso_Recreativa | 7,3 % |
| franja_horaria_Pico AM | 7,3 % |
| zona_uso_Residencial | 6,6 % |
| … | … |
| temperatura_c | 3,6 % |
| evento_masivo | 2,1 % |
| lluvia | 1,4 % |

Lo que más pesa es **cuándo** (la hora y el tipo de día) y **dónde** (la zona de
uso), que es justo lo que se esperaba. El árbol completo se exportó como **243
reglas SI … ENTONCES** en `resultados/reglas_arbol.txt`.

**Escenarios nuevos (paso 9):**
```
- Portal (residencial), martes 6 a. m., lloviendo         -> Crítico  (confianza 100%)
- Calle 72 (empresarial), jueves 5 p. m., sin lluvia      -> Alto     (confianza 65%)
- Universidades, domingo 12 m., sin lluvia                -> Bajo     (confianza 100%)
- El Campín, sábado 7 p. m., CON partido de fútbol        -> Alto     (confianza 93%)
- El Campín, sábado 7 p. m., SIN evento                   -> Bajo     (confianza 60%)
```
Todas las predicciones tienen sentido. La más interesante es la de El Campín: la
misma estación, el mismo día y la misma hora pasa de **Bajo a Alto** solo porque
hay un partido.

### E3. Experimento: fuga de información

Se agregó `ocupacion_pct` como variable de entrada:

| Modelo | Exactitud | F1 macro | Profundidad |
|---|---|---|---|
| Con `ocupacion_pct` (fuga) | 1,000 | 1,000 | 3 |

**Resultado:** el modelo es "perfecto" con solo 3 niveles, porque se limita a
copiar los umbrales 40/70/90. No sirve para predecir, ya que en la vida real la
ocupación no se conoce antes de que ocurra. Esto confirma que excluir esa
columna era **obligatorio**.

### E4. Experimento: selección de variables

Se usaron los mismos hiperparámetros del árbol final, quitando variables:

| Variables | Exactitud | F1 macro |
|---|---|---|
| Todas (modelo final) | 0,904 | 0,781 |
| Sin `temperatura_c` | 0,907 | **0,790** |
| Sin `lluvia` ni `evento_masivo` | 0,897 | 0,754 |

**Resultado:**

- **Quitar la temperatura mejora el modelo.** En los datos simulados la
  temperatura no influye en la ocupación, pero el árbol la usó (3,6 % de
  importancia) porque está correlacionada con la hora y con la lluvia. Es un
  ejemplo de cómo un árbol puede apoyarse en una variable que no tiene relación
  real con el resultado. Recomendación: en una siguiente versión, retirarla.
- **Quitar la lluvia y los eventos empeora el modelo** (de 0,781 a 0,754): esas
  variables sí aportan información útil.

---

## Resumen

| Prueba | Resultado |
|---|---|
| 15 pruebas automáticas | 15/15 OK |
| Limpieza | 15 duplicados, 25 etiquetas corregidas, 60 valores de lluvia completados |
| Árbol final | Entropía, profundidad 10, 243 hojas |
| Exactitud en prueba | 0,904 (línea base 0,663) |
| F1 macro en prueba | 0,781 (línea base 0,199) |
| Poda | Pasa de 1.573 a 243 hojas y mejora el F1 de 0,745 a 0,781 |
| Escenarios nuevos | 5/5 coherentes |

**Limitaciones:**

- Los datos son una muestra simulada.
- Las clases Alto y Crítico tienen pocos ejemplos, y es donde el modelo más se
  equivoca.
- Con datos reales se podría mejorar con más meses de historia, ponderando las
  clases minoritarias o probando un bosque aleatorio (*random forest*).
