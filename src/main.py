"""
main.py
=======
Programa principal: entrena y evalua el modelo SUPERVISADO (arbol de
decision) que predice el nivel de ocupacion de una estacion de TransMilenio.

    PASO 1. Cargar las 4 fuentes de datos.
    PASO 2. Limpiar e integrar los datos.
    PASO 3. Preparar variables de entrada (X) y etiqueta (y).
    PASO 4. Dividir en entrenamiento (75%) y prueba (25%).
    PASO 5. Linea base y arbol sin podar (sobreajuste).
    PASO 6. Buscar el mejor arbol con validacion cruzada (poda).
    PASO 7. Evaluar el arbol final con los datos de prueba.
    PASO 8. Importancia de variables y reglas de decision.
    PASO 9. Predecir escenarios nuevos.
    PASO 10. Guardar resultados y graficas.

Uso (desde la carpeta del proyecto):
    python src/main.py
"""

import sys
from pathlib import Path

import pandas as pd

from preparacion_datos import (ORDEN_NIVELES, cargar_datos, limpiar, preparar_escenarios,
                               preparar_X_y)
from modelo_arbol import (arbol_sin_podar, buscar_mejor_arbol, curva_profundidad, dividir,
                          evaluar, importancia_variables, modelo_base, reglas_texto)
import visualizacion as viz

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CARPETA_RESULTADOS = Path(__file__).resolve().parent.parent / "resultados"

# Situaciones nuevas para probar el modelo (no son filas del dataset).
ESCENARIOS = [
    {"descripcion": "Portal (residencial), martes 6 a. m., lloviendo",
     "hora": 6, "tipo_dia": "Hábil", "tipo_estacion": "Portal", "zona_uso": "Residencial",
     "lluvia": 1, "temperatura_c": 10.5, "evento_masivo": 0},
    {"descripcion": "Calle 72 (empresarial), jueves 5 p. m., sin lluvia",
     "hora": 17, "tipo_dia": "Hábil", "tipo_estacion": "Intermedia", "zona_uso": "Empresarial",
     "lluvia": 0, "temperatura_c": 16.0, "evento_masivo": 0},
    {"descripcion": "Universidades, domingo 12 m., sin lluvia",
     "hora": 12, "tipo_dia": "Domingo/Festivo", "tipo_estacion": "Intermedia", "zona_uso": "Universitaria",
     "lluvia": 0, "temperatura_c": 19.0, "evento_masivo": 0},
    {"descripcion": "El Campín, sábado 7 p. m., CON partido de fútbol",
     "hora": 19, "tipo_dia": "Sábado", "tipo_estacion": "Intermedia", "zona_uso": "Recreativa",
     "lluvia": 0, "temperatura_c": 13.0, "evento_masivo": 1},
    {"descripcion": "El Campín, sábado 7 p. m., SIN evento",
     "hora": 19, "tipo_dia": "Sábado", "tipo_estacion": "Intermedia", "zona_uso": "Recreativa",
     "lluvia": 0, "temperatura_c": 13.0, "evento_masivo": 0},
]


def titulo(texto):
    print("\n" + "=" * 70)
    print(texto)
    print("=" * 70)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)

    # ------------------------------------------------------------------ 1
    titulo("PASO 1. Carga de las fuentes de datos")
    fuentes = cargar_datos()
    for nombre, df in fuentes.items():
        print(f"  {nombre:<11} {df.shape[0]:>6} filas x {df.shape[1]} columnas")
    print("\nPrimeras filas de ocupacion_horaria.csv:")
    print(fuentes["ocupacion"].head().to_string(index=False))

    # ------------------------------------------------------------------ 2
    titulo("PASO 2. Limpieza e integración de los datos")
    tabla, rep = limpiar(fuentes)
    print(f"Filas iniciales:                {rep['filas_iniciales']}")
    print(f"Duplicados eliminados:          {rep['duplicados_eliminados']}")
    print(f"Etiquetas mal escritas arregladas: {rep['etiquetas_corregidas']}")
    print(f"Lluvia faltante completada:     {rep['lluvia_faltante']} (desde calendario.csv)")
    print(f"Filas finales:                  {rep['filas_finales']}")

    # ------------------------------------------------------------------ 3
    titulo("PASO 3. Variables de entrada (X) y etiqueta (y)")
    X, y = preparar_X_y(tabla)
    print(f"X: {X.shape[0]} ejemplos x {X.shape[1]} variables (después de one-hot encoding)")
    print("Variables:", ", ".join(X.columns))
    print("\nDistribución de la etiqueta (y):")
    print(y.value_counts().reindex(ORDEN_NIVELES).to_string())

    # ------------------------------------------------------------------ 4
    titulo("PASO 4. División en entrenamiento y prueba (75% / 25%, estratificada)")
    X_train, X_test, y_train, y_test = dividir(X, y)
    print(f"Entrenamiento: {len(X_train)} ejemplos | Prueba: {len(X_test)} ejemplos")

    # ------------------------------------------------------------------ 5
    titulo("PASO 5. Línea base y árbol sin podar")
    base = modelo_base(X_train, y_train)
    ev_base = evaluar(base, X_test, y_test, ORDEN_NIVELES)
    print(f"Línea base (siempre dice 'Bajo'): exactitud {ev_base['exactitud']:.3f} | F1 macro {ev_base['f1_macro']:.3f}")
    libre = arbol_sin_podar(X_train, y_train)
    ev_libre_tr = evaluar(libre, X_train, y_train, ORDEN_NIVELES)
    ev_libre_te = evaluar(libre, X_test, y_test, ORDEN_NIVELES)
    print(f"Árbol sin podar: profundidad {libre.get_depth()}, {libre.get_n_leaves()} hojas")
    print(f"   F1 macro en ENTRENAMIENTO: {ev_libre_tr['f1_macro']:.3f}")
    print(f"   F1 macro en PRUEBA:        {ev_libre_te['f1_macro']:.3f}   <- sobreajuste: memoriza")

    # ------------------------------------------------------------------ 6
    titulo("PASO 6. Búsqueda del mejor árbol (validación cruzada de 5 particiones)")
    busqueda = buscar_mejor_arbol(X_train, y_train)
    arbol = busqueda.best_estimator_
    print(f"Combinaciones evaluadas: {len(busqueda.cv_results_['params'])} x 5 particiones")
    print(f"Mejores hiperparámetros: {busqueda.best_params_}")
    print(f"F1 macro promedio en validación cruzada: {busqueda.best_score_:.3f}")
    print(f"Árbol final: profundidad {arbol.get_depth()}, {arbol.get_n_leaves()} hojas")

    # ------------------------------------------------------------------ 7
    titulo("PASO 7. Evaluación final con los datos de PRUEBA")
    ev = evaluar(arbol, X_test, y_test, ORDEN_NIVELES)
    print(f"Exactitud: {ev['exactitud']:.3f} | F1 macro: {ev['f1_macro']:.3f}\n")
    print(ev["reporte"])
    print("Matriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(ev["matriz"], index=ORDEN_NIVELES, columns=ORDEN_NIVELES).to_string())

    resumen = pd.DataFrame([
        {"modelo": "Línea base (clase más frecuente)", "exactitud": ev_base["exactitud"], "f1_macro": ev_base["f1_macro"]},
        {"modelo": f"Árbol sin podar (prof. {libre.get_depth()})", "exactitud": ev_libre_te["exactitud"], "f1_macro": ev_libre_te["f1_macro"]},
        {"modelo": f"Árbol podado (prof. {arbol.get_depth()})", "exactitud": ev["exactitud"], "f1_macro": ev["f1_macro"]},
    ]).round(3)
    print("\nComparación en el conjunto de prueba:")
    print(resumen.to_string(index=False))

    # ------------------------------------------------------------------ 8
    titulo("PASO 8. Importancia de variables y reglas de decisión")
    imp = importancia_variables(arbol, X.columns)
    print((imp.head(10) * 100).round(1).astype(str).add(" %").to_string())
    reglas = reglas_texto(arbol, X.columns)
    print(f"\nEl árbol equivale a {arbol.get_n_leaves()} reglas SI ... ENTONCES.")
    print("Primeras líneas (el árbol completo está en resultados/reglas_arbol.txt):")
    print("\n".join(reglas.splitlines()[:14]))

    # ------------------------------------------------------------------ 9
    titulo("PASO 9. Predicción de escenarios nuevos")
    X_esc = preparar_escenarios(ESCENARIOS, X.columns)
    pred = arbol.predict(X_esc)
    prob = arbol.predict_proba(X_esc)
    clases = list(arbol.classes_)
    for esc, p, pr in zip(ESCENARIOS, pred, prob):
        print(f"- {esc['descripcion']:<55} -> {p:<8} (confianza {pr[clases.index(p)]:.0%})")

    # ------------------------------------------------------------------ 10
    titulo("PASO 10. Guardando resultados y gráficas")
    curva = curva_profundidad(X_train, y_train, X_test, y_test)
    escenarios_df = pd.DataFrame(ESCENARIOS).assign(prediccion=pred)
    archivos = {
        "comparacion_modelos.csv": lambda r: resumen.to_csv(r, index=False, encoding="utf-8"),
        "curva_profundidad.csv": lambda r: curva.round(4).to_csv(r, index=False),
        "importancia_variables.csv": lambda r: imp.round(4).rename("importancia").to_csv(r, encoding="utf-8"),
        "predicciones_escenarios.csv": lambda r: escenarios_df.to_csv(r, index=False, encoding="utf-8"),
        "reglas_arbol.txt": lambda r: Path(r).write_text(reglas, encoding="utf-8"),
        "01_distribucion_clases.png": lambda r: viz.grafica_distribucion(y, ORDEN_NIVELES, r),
        "02_sobreajuste_profundidad.png": lambda r: viz.grafica_profundidad(curva, arbol.get_depth(), r),
        "03_matriz_confusion.png": lambda r: viz.grafica_matriz_confusion(ev["matriz"], ORDEN_NIVELES, r),
        "04_importancia_variables.png": lambda r: viz.grafica_importancia(imp, r),
        "05_arbol_decision.png": lambda r: viz.grafica_arbol(arbol, X.columns, arbol.classes_, r),
    }
    for nombre, guardar in archivos.items():
        guardar(CARPETA_RESULTADOS / nombre)
        print(f"  guardado: resultados/{nombre}")
    print("\nProceso terminado.")


if __name__ == "__main__":
    main()
