"""
generar_dataset.py
==================
Genera el dataset de MUESTRA para el modelo de aprendizaje SUPERVISADO.

Problema que se quiere resolver
-------------------------------
Predecir el NIVEL DE OCUPACION de una estacion de TransMilenio en una hora
determinada (Bajo, Medio, Alto o Critico), a partir de informacion que se
conoce de antemano: la hora, el tipo de dia, el tipo y la zona de la estacion,
si esta lloviendo y si hay un evento masivo cerca.

Como es un problema SUPERVISADO, cada registro trae la respuesta correcta
(la columna "nivel_ocupacion"), que el modelo usa para aprender.

Por que un dataset de muestra
-----------------------------
TransMilenio S.A. y Datos Abiertos Bogota publican conteos de entradas/salidas
por franja horaria, y el IDEAM publica datos de precipitacion, pero son
archivos muy pesados y en formatos distintos que habria que cruzar entre si.
Siguiendo la indicacion de la actividad ("en caso de no existir dichas fuentes
de datos, desarrolle un dataset con una muestra de dichos datos"), este script
construye una MUESTRA SIMULADA con la estructura de esas fuentes.

IMPORTANTE: los nombres de estaciones y troncales son reales; la ocupacion,
la lluvia y los eventos son SIMULADOS con fines academicos. No son cifras
oficiales de TransMilenio S.A.

Archivos generados (carpeta datos/):
    estaciones.csv         catalogo de estaciones con su tipo y zona de uso.
    calendario.csv         dias del periodo: tipo de dia, festivos y lluvia.
    eventos.csv            eventos masivos (conciertos, partidos de futbol).
    ocupacion_horaria.csv  registro por estacion, fecha y hora con el clima
                           (lluvia, temperatura), si hay evento, la ocupacion (%)
                           y el nivel de ocupacion (ETIQUETA).

Uso:
    python src/generar_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 2026
CARPETA_DATOS = Path(__file__).resolve().parent.parent / "datos"
HORAS = list(range(5, 23))  # 5:00 a 22:59
FECHA_INICIO, FECHA_FIN = "2026-10-01", "2026-10-28"
FESTIVOS = ["2026-10-12"]  # Dia de la Raza (festivo en Colombia, lunes 12 de octubre de 2026)

# (nombre, troncal, tipo_estacion, zona_uso, capacidad_relativa)
ESTACIONES = [
    ("Portal Norte", "Autonorte", "Portal", "Residencial", 1.00),
    ("Portal Suba", "Suba", "Portal", "Residencial", 1.00),
    ("Portal Américas", "Américas", "Portal", "Residencial", 1.00),
    ("Portal Sur", "NQS Sur", "Portal", "Residencial", 1.00),
    ("Portal Usme", "Caracas Sur", "Portal", "Residencial", 1.00),
    ("Portal 80", "Calle 80", "Portal", "Residencial", 1.00),
    ("Banderas", "Américas", "Intermedia", "Residencial", 0.90),
    ("Molinos", "Caracas Sur", "Intermedia", "Residencial", 0.85),
    ("Toberín", "Autonorte", "Intermedia", "Residencial", 0.85),
    ("Suba - Av. Boyacá", "Suba", "Intermedia", "Residencial", 0.85),
    ("Calle 72", "Caracas", "Intermedia", "Empresarial", 0.95),
    ("Calle 100", "Autonorte", "Intermedia", "Empresarial", 0.95),
    ("Héroes", "Autonorte", "Intermedia", "Empresarial", 0.90),
    ("Virrey", "Autonorte", "Intermedia", "Empresarial", 0.85),
    ("CAN", "Calle 26", "Intermedia", "Empresarial", 0.90),
    ("Salitre El Greco", "Calle 26", "Intermedia", "Empresarial", 0.85),
    ("Universidades", "Eje Ambiental", "Intermedia", "Universitaria", 0.90),
    ("Calle 45", "Caracas", "Intermedia", "Universitaria", 0.90),
    ("Marly", "Caracas", "Intermedia", "Universitaria", 0.85),
    ("Ciudad Universitaria", "Calle 26", "Intermedia", "Universitaria", 0.90),
    ("Las Aguas", "Eje Ambiental", "Cabecera", "Universitaria", 0.85),
    ("Ricaurte", "NQS", "Intercambio", "Intercambio", 1.00),
    ("Av. Jiménez", "Caracas", "Intercambio", "Intercambio", 1.00),
    ("San Mateo", "NQS Sur", "Intercambio", "Intercambio", 0.95),
    ("Calle 26", "Caracas", "Intermedia", "Intercambio", 0.90),
    ("Paloquemao", "NQS", "Intermedia", "Intercambio", 0.85),
    ("Movistar Arena", "NQS", "Intermedia", "Recreativa", 0.80),
    ("El Campín", "NQS", "Intermedia", "Recreativa", 0.80),
    ("Coliseo", "NQS", "Intermedia", "Recreativa", 0.75),
    ("Simón Bolívar", "NQS", "Intermedia", "Recreativa", 0.75),
]


def _campana(h, centro, ancho, altura):
    return altura * np.exp(-((np.asarray(h, float) - centro) ** 2) / (2 * ancho ** 2))


def curva_base(zona, h):
    """Ocupacion base (%) en un dia habil, segun la zona de la estacion y la hora."""
    if zona == "Residencial":
        return 20 + _campana(h, 6.3, 1.2, 85) + _campana(h, 18.0, 1.6, 35)
    if zona == "Empresarial":
        return 18 + _campana(h, 7.3, 1.1, 50) + _campana(h, 17.6, 1.4, 82)
    if zona == "Universitaria":
        return 18 + _campana(h, 7.5, 1.1, 38) + _campana(h, 12.0, 2.2, 58) + _campana(h, 17.5, 1.5, 38)
    if zona == "Intercambio":
        return 28 + _campana(h, 7.0, 1.3, 68) + _campana(h, 13.0, 2.5, 18) + _campana(h, 17.8, 1.5, 70)
    if zona == "Recreativa":
        return 12 + _campana(h, 7.0, 1.0, 20) + _campana(h, 17.5, 1.5, 22)
    raise ValueError(zona)


# Cuanto se reduce la demanda segun el tipo de dia.
FACTOR_DIA = {"Hábil": 1.0, "Sábado": 0.55, "Domingo/Festivo": 0.35}


def nivel(ocupacion):
    """Convierte la ocupacion (%) en la ETIQUETA que el modelo debe predecir."""
    if ocupacion < 40:
        return "Bajo"
    if ocupacion < 70:
        return "Medio"
    if ocupacion < 90:
        return "Alto"
    return "Crítico"


def generar():
    rng = np.random.default_rng(SEMILLA)
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)

    # ------------------------------ estaciones.csv ------------------------------
    estaciones = pd.DataFrame(ESTACIONES, columns=["nombre", "troncal", "tipo_estacion", "zona_uso", "capacidad"])
    estaciones.insert(0, "estacion_id", [f"S{i:02d}" for i in range(1, len(estaciones) + 1)])
    estaciones[["estacion_id", "nombre", "troncal", "tipo_estacion", "zona_uso"]].to_csv(
        CARPETA_DATOS / "estaciones.csv", index=False, encoding="utf-8")

    # ------------------------------ calendario.csv ------------------------------
    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    dias_es = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    cal = pd.DataFrame({"fecha": fechas.strftime("%Y-%m-%d"),
                        "dia_semana": [dias_es[d.weekday()] for d in fechas]})
    cal["es_festivo"] = cal["fecha"].isin(FESTIVOS).astype(int)
    cal["tipo_dia"] = np.where((cal["dia_semana"] == "domingo") | (cal["es_festivo"] == 1), "Domingo/Festivo",
                               np.where(cal["dia_semana"] == "sábado", "Sábado", "Hábil"))
    # Octubre es temporada de lluvias en Bogota: llueve en la tarde ~45% de los dias
    # y en la manana ~20%.
    cal["lluvia_manana"] = (rng.random(len(cal)) < 0.20).astype(int)
    cal["lluvia_tarde"] = (rng.random(len(cal)) < 0.45).astype(int)
    cal.to_csv(CARPETA_DATOS / "calendario.csv", index=False, encoding="utf-8")

    # -------------------------------- eventos.csv -------------------------------
    eventos = pd.DataFrame([
        ("2026-10-03", "Movistar Arena", "Concierto", 18, 22),
        ("2026-10-04", "El Campín", "Partido de fútbol", 15, 19),
        ("2026-10-09", "Movistar Arena", "Concierto", 18, 22),
        ("2026-10-11", "El Campín", "Partido de fútbol", 16, 20),
        ("2026-10-15", "Coliseo", "Concierto", 18, 22),
        ("2026-10-17", "Simón Bolívar", "Festival al parque", 11, 20),
        ("2026-10-18", "Simón Bolívar", "Festival al parque", 11, 20),
        ("2026-10-21", "El Campín", "Partido de fútbol", 18, 22),
        ("2026-10-24", "Movistar Arena", "Concierto", 18, 22),
        ("2026-10-25", "El Campín", "Partido de fútbol", 15, 19),
    ], columns=["fecha", "estacion_cercana", "tipo_evento", "hora_inicio", "hora_fin"])
    eventos.to_csv(CARPETA_DATOS / "eventos.csv", index=False, encoding="utf-8")

    # ---------------------------- ocupacion_horaria.csv -------------------------
    filas = []
    for _, est in estaciones.iterrows():
        sesgo_estacion = rng.normal(0, 4)  # cada estacion es un poco distinta
        for _, dia in cal.iterrows():
            ev = eventos[(eventos["fecha"] == dia["fecha"]) & (eventos["estacion_cercana"] == est["nombre"])]
            for h in HORAS:
                llueve = dia["lluvia_manana"] if h < 12 else dia["lluvia_tarde"]
                hay_evento = int(any((ev["hora_inicio"] <= h) & (h <= ev["hora_fin"])))
                ocup = curva_base(est["zona_uso"], h) * est["capacidad"] * FACTOR_DIA[dia["tipo_dia"]]
                ocup *= 1.12 if llueve else 1.0          # con lluvia la gente deja la bici/caminata
                ocup += 65 * hay_evento                    # evento masivo: llenura repentina
                ocup += sesgo_estacion + rng.normal(0, 6)  # ruido
                ocup = round(float(np.clip(ocup, 2, 125)), 1)
                # Temperatura (IDEAM): mas calor al mediodia, mas frio si llueve.
                # NO influye en la ocupacion simulada: es una variable "distractora"
                # para comprobar si el arbol aprende cosas que no son reales.
                temperatura = 11 + _campana(h, 13, 3.5, 8) - 2.5 * llueve + rng.normal(0, 1.2)
                filas.append({
                    "fecha": dia["fecha"], "estacion_id": est["estacion_id"], "hora": h,
                    "lluvia": int(llueve), "temperatura_c": round(float(temperatura), 1),
                    "evento_masivo": hay_evento,
                    "ocupacion_pct": ocup, "nivel_ocupacion": nivel(ocup),
                })
    datos = pd.DataFrame(filas)

    # Problemas de calidad inyectados a proposito (para la etapa de limpieza):
    n = len(datos)
    idx = rng.choice(n, size=int(n * 0.004), replace=False)
    datos["lluvia"] = datos["lluvia"].astype("Int64")
    datos.loc[idx, "lluvia"] = pd.NA                         # sensor de lluvia sin dato
    idx2 = rng.choice(n, size=25, replace=False)
    datos.loc[idx2, "nivel_ocupacion"] = datos.loc[idx2, "nivel_ocupacion"].str.upper() + " "  # mal digitado
    duplicadas = datos.sample(15, random_state=SEMILLA)
    datos = pd.concat([datos, duplicadas], ignore_index=True)  # registros repetidos
    datos.to_csv(CARPETA_DATOS / "ocupacion_horaria.csv", index=False, encoding="utf-8")

    print(f"estaciones.csv        -> {len(estaciones)} estaciones")
    print(f"calendario.csv        -> {len(cal)} dias ({FECHA_INICIO} a {FECHA_FIN})")
    print(f"eventos.csv           -> {len(eventos)} eventos masivos")
    print(f"ocupacion_horaria.csv -> {len(datos)} registros")
    print("Distribucion de la etiqueta:")
    print(datos["nivel_ocupacion"].str.strip().str.capitalize().value_counts().to_string())


if __name__ == "__main__":
    generar()
