"""
modelo_arbol.py
===============
Modelo de APRENDIZAJE SUPERVISADO: arbol de decision (clasificacion).

Un arbol de decision aprende, a partir de ejemplos etiquetados, una serie de
preguntas del tipo "¿hora <= 8.5?", "¿zona_uso = Residencial?" que van
dividiendo los datos hasta llegar a una hoja con la clase mas probable
(Palma Mendez, 2008, cap. 17 "Aprendizaje de arboles y reglas de decision").

En cada nodo el algoritmo elige la pregunta que MAS reduce la impureza:
    - Entropia / ganancia de informacion (como en ID3 y C4.5).
    - Indice de Gini (como en CART, el que usa scikit-learn).

Cada camino desde la raiz hasta una hoja equivale a una REGLA de decision:
    SI hora <= 8.5 Y zona = Residencial Y tipo_dia = Habil ENTONCES Critico

Un arbol muy profundo "memoriza" los datos de entrenamiento (sobreajuste) y
predice mal datos nuevos. Por eso se PODA, limitando la profundidad y el
numero minimo de ejemplos por hoja, y esos valores se eligen con validacion
cruzada.
"""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

SEMILLA = 42


def dividir(X, y, proporcion_prueba=0.25):
    """
    Divide los datos en ENTRENAMIENTO (75%) y PRUEBA (25%).
    Estratificado: cada parte conserva la misma proporcion de cada clase.
    El conjunto de prueba NO se toca hasta la evaluacion final.
    """
    return train_test_split(X, y, test_size=proporcion_prueba, stratify=y, random_state=SEMILLA)


def modelo_base(X_train, y_train):
    """Linea base: siempre predice la clase mas frecuente. El arbol debe superarla."""
    return DummyClassifier(strategy="most_frequent").fit(X_train, y_train)


def arbol_sin_podar(X_train, y_train):
    """Arbol que crece sin limites (para mostrar el sobreajuste)."""
    return DecisionTreeClassifier(random_state=SEMILLA).fit(X_train, y_train)


def buscar_mejor_arbol(X_train, y_train):
    """
    Busca los mejores hiperparametros con VALIDACION CRUZADA de 5 particiones
    sobre los datos de entrenamiento (GridSearchCV). Se optimiza el F1 macro
    (promedio del F1 de las 4 clases) porque las clases estan desbalanceadas:
    la exactitud sola premiaria predecir siempre "Bajo".
    """
    parametros = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 4, 5, 6, 8, 10, 12],
        "min_samples_leaf": [1, 5, 10, 20, 50],
    }
    busqueda = GridSearchCV(
        DecisionTreeClassifier(random_state=SEMILLA),
        parametros,
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA),
        n_jobs=-1,
    )
    busqueda.fit(X_train, y_train)
    return busqueda


def curva_profundidad(X_train, y_train, X_test, y_test, profundidades=range(1, 21)):
    """F1 macro en entrenamiento y en prueba para cada profundidad maxima."""
    filas = []
    for d in profundidades:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEMILLA).fit(X_train, y_train)
        filas.append({
            "profundidad": d,
            "f1_entrenamiento": f1_score(y_train, m.predict(X_train), average="macro"),
            "f1_prueba": f1_score(y_test, m.predict(X_test), average="macro"),
        })
    return pd.DataFrame(filas)


def evaluar(modelo, X, y, orden_clases):
    """Metricas de clasificacion sobre un conjunto de datos."""
    pred = modelo.predict(X)
    return {
        "exactitud": accuracy_score(y, pred),
        "f1_macro": f1_score(y, pred, average="macro"),
        "reporte": classification_report(y, pred, labels=orden_clases, digits=3, zero_division=0),
        "matriz": confusion_matrix(y, pred, labels=orden_clases),
    }


def importancia_variables(modelo, columnas):
    """Cuanto aporta cada variable a reducir la impureza del arbol."""
    return (pd.Series(modelo.feature_importances_, index=columnas)
            .sort_values(ascending=False))


def reglas_texto(modelo, columnas):
    """El arbol escrito como reglas SI ... ENTONCES (formato texto)."""
    return export_text(modelo, feature_names=list(columnas), decimals=1)
