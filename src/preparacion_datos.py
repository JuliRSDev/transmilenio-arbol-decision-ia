"""
preparacion_datos.py
====================
Carga, limpieza, integracion de fuentes y preparacion de variables para el
modelo supervisado.

En el aprendizaje SUPERVISADO cada ejemplo tiene:
    - Variables de entrada (X): lo que se conoce ANTES de que pase el hecho
      (hora y franja, tipo de dia, tipo y zona de la estacion, lluvia,
      temperatura, evento masivo).
    - Variable objetivo (y): la ETIQUETA que se quiere predecir
      (nivel_ocupacion: Bajo, Medio, Alto o Critico).

OJO: la columna "ocupacion_pct" NO se usa como entrada, porque la etiqueta se
calcula directamente a partir de ella. Usarla seria "hacer trampa" (fuga de
informacion o data leakage): el modelo acertaria siempre pero no serviria
para predecir nada en la vida real.
"""

from pathlib import Path

import pandas as pd

CARPETA_DATOS = Path(__file__).resolve().parent.parent / "datos"

ORDEN_NIVELES = ["Bajo", "Medio", "Alto", "Crítico"]
VARIABLES_CATEGORICAS = ["franja_horaria", "tipo_dia", "tipo_estacion", "zona_uso"]
VARIABLES_NUMERICAS = ["hora", "lluvia", "temperatura_c", "evento_masivo"]


def franja_horaria(hora):
    """Agrupa la hora en franjas con sentido para el transporte."""
    if hora <= 5:
        return "Madrugada"
    if hora <= 8:
        return "Pico AM"
    if hora <= 15:
        return "Valle"
    if hora <= 19:
        return "Pico PM"
    return "Noche"
OBJETIVO = "nivel_ocupacion"


def cargar_datos(carpeta=CARPETA_DATOS):
    """Lee las 4 fuentes de datos y las devuelve en un diccionario."""
    carpeta = Path(carpeta)
    return {
        "ocupacion": pd.read_csv(carpeta / "ocupacion_horaria.csv", encoding="utf-8"),
        "estaciones": pd.read_csv(carpeta / "estaciones.csv", encoding="utf-8"),
        "calendario": pd.read_csv(carpeta / "calendario.csv", encoding="utf-8"),
        "eventos": pd.read_csv(carpeta / "eventos.csv", encoding="utf-8"),
    }


def limpiar(fuentes):
    """
    Limpieza e integracion de los datos:
      1. Elimina registros duplicados.
      2. Corrige etiquetas mal digitadas ("ALTO " -> "Alto").
      3. Une la ocupacion con los datos de la estacion y del calendario.
      4. Completa la lluvia faltante usando el calendario (lluvia de la manana
         si la hora es antes de las 12, lluvia de la tarde si es despues).

    Devuelve (tabla_limpia, reporte).
    """
    df = fuentes["ocupacion"].copy()
    reporte = {"filas_iniciales": len(df)}

    reporte["duplicados_eliminados"] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)

    etiqueta_original = df[OBJETIVO].copy()
    df[OBJETIVO] = df[OBJETIVO].str.strip().str.capitalize()
    reporte["etiquetas_corregidas"] = int((etiqueta_original != df[OBJETIVO]).sum())
    invalidas = set(df[OBJETIVO]) - set(ORDEN_NIVELES)
    if invalidas:
        raise ValueError(f"Etiquetas desconocidas en los datos: {invalidas}")

    df = df.merge(fuentes["estaciones"], on="estacion_id", how="left")
    df = df.merge(fuentes["calendario"], on="fecha", how="left")

    reporte["lluvia_faltante"] = int(df["lluvia"].isna().sum())
    lluvia_calendario = df["lluvia_manana"].where(df["hora"] < 12, df["lluvia_tarde"])
    df["lluvia"] = df["lluvia"].fillna(lluvia_calendario).astype(int)

    # Variable derivada (ingenieria de caracteristicas).
    df["franja_horaria"] = df["hora"].apply(franja_horaria)

    reporte["filas_finales"] = len(df)
    return df, reporte


def preparar_X_y(tabla):
    """
    Construye la matriz de entrada X y el vector objetivo y.

    Los arboles de scikit-learn solo aceptan numeros, por eso las variables
    categoricas se transforman con "one-hot encoding": por ejemplo zona_uso
    se convierte en columnas zona_uso_Residencial, zona_uso_Empresarial, ...
    con valores 0 o 1.
    """
    X = pd.get_dummies(tabla[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS],
                       columns=VARIABLES_CATEGORICAS, dtype=int)
    y = tabla[OBJETIVO]
    return X, y


def preparar_escenarios(escenarios, columnas_modelo):
    """Convierte escenarios nuevos (lista de dicts) al mismo formato de X."""
    df = pd.DataFrame(escenarios)
    df["franja_horaria"] = df["hora"].apply(franja_horaria)
    X = pd.get_dummies(df[VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS],
                       columns=VARIABLES_CATEGORICAS, dtype=int)
    return X.reindex(columns=columnas_modelo, fill_value=0)
