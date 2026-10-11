"""
test_proyecto.py
================
Pruebas automaticas del modelo supervisado (arbol de decision).
Usa unittest (incluido en Python).

Ejecutar desde la carpeta del proyecto:
    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from preparacion_datos import (ORDEN_NIVELES, cargar_datos, limpiar,  # noqa: E402
                               preparar_escenarios, preparar_X_y)
from modelo_arbol import (arbol_sin_podar, dividir, evaluar, modelo_base,  # noqa: E402
                          reglas_texto)
from sklearn.tree import DecisionTreeClassifier  # noqa: E402

PARAMS_FINALES = dict(criterion="entropy", max_depth=10, min_samples_leaf=10, random_state=42)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fuentes = cargar_datos(RAIZ / "datos")
        cls.tabla, cls.reporte = limpiar(cls.fuentes)
        cls.X, cls.y = preparar_X_y(cls.tabla)
        cls.X_tr, cls.X_te, cls.y_tr, cls.y_te = dividir(cls.X, cls.y)
        cls.arbol = DecisionTreeClassifier(**PARAMS_FINALES).fit(cls.X_tr, cls.y_tr)


class PruebasDatos(Base):
    def test_p01_fuentes_con_columnas_esperadas(self):
        self.assertEqual(list(self.fuentes["ocupacion"].columns),
                         ["fecha", "estacion_id", "hora", "lluvia", "temperatura_c",
                          "evento_masivo", "ocupacion_pct", "nivel_ocupacion"])
        self.assertEqual(len(self.fuentes["estaciones"]), 30)
        self.assertEqual(len(self.fuentes["calendario"]), 28)

    def test_p02_limpieza_sin_duplicados_ni_nulos(self):
        self.assertEqual(self.reporte["duplicados_eliminados"], 15)
        self.assertEqual(self.tabla.duplicated(subset=["fecha", "estacion_id", "hora"]).sum(), 0)
        self.assertEqual(self.tabla["lluvia"].isna().sum(), 0)

    def test_p03_etiquetas_validas(self):
        self.assertEqual(set(self.tabla["nivel_ocupacion"]), set(ORDEN_NIVELES))

    def test_p04_etiqueta_coherente_con_ocupacion(self):
        t = self.tabla
        self.assertTrue((t.loc[t["nivel_ocupacion"] == "Bajo", "ocupacion_pct"] < 40).all())
        self.assertTrue((t.loc[t["nivel_ocupacion"] == "Crítico", "ocupacion_pct"] >= 90).all())

    def test_p05_detecta_etiqueta_desconocida(self):
        malo = {k: v.copy() for k, v in self.fuentes.items()}
        malo["ocupacion"].loc[0, "nivel_ocupacion"] = "Lleno"
        with self.assertRaises(ValueError):
            limpiar(malo)


class PruebasPreparacion(Base):
    def test_p06_sin_fuga_de_informacion(self):
        self.assertNotIn("ocupacion_pct", self.X.columns)
        self.assertNotIn("nivel_ocupacion", self.X.columns)

    def test_p07_X_solo_numerica(self):
        self.assertTrue(all(str(t) in ("int64", "int32", "float64") for t in self.X.dtypes))

    def test_p08_division_estratificada(self):
        self.assertEqual(len(self.X_te), int(round(len(self.X) * 0.25)))
        p_tr = self.y_tr.value_counts(normalize=True)
        p_te = self.y_te.value_counts(normalize=True)
        for c in ORDEN_NIVELES:
            self.assertAlmostEqual(p_tr[c], p_te[c], delta=0.01)


class PruebasModelo(Base):
    def test_p09_supera_linea_base(self):
        base = evaluar(modelo_base(self.X_tr, self.y_tr), self.X_te, self.y_te, ORDEN_NIVELES)
        arbol = evaluar(self.arbol, self.X_te, self.y_te, ORDEN_NIVELES)
        self.assertGreater(arbol["f1_macro"], base["f1_macro"] + 0.4)
        self.assertGreater(arbol["exactitud"], 0.85)

    def test_p10_poda_reduce_sobreajuste(self):
        libre = arbol_sin_podar(self.X_tr, self.y_tr)
        brecha_libre = (evaluar(libre, self.X_tr, self.y_tr, ORDEN_NIVELES)["f1_macro"]
                        - evaluar(libre, self.X_te, self.y_te, ORDEN_NIVELES)["f1_macro"])
        brecha_podado = (evaluar(self.arbol, self.X_tr, self.y_tr, ORDEN_NIVELES)["f1_macro"]
                         - evaluar(self.arbol, self.X_te, self.y_te, ORDEN_NIVELES)["f1_macro"])
        self.assertLess(brecha_podado, brecha_libre)
        self.assertLess(self.arbol.get_n_leaves(), libre.get_n_leaves())

    def test_p11_reproducible(self):
        otro = DecisionTreeClassifier(**PARAMS_FINALES).fit(self.X_tr, self.y_tr)
        self.assertTrue((otro.predict(self.X_te) == self.arbol.predict(self.X_te)).all())

    def test_p12_reglas_exportables(self):
        reglas = reglas_texto(self.arbol, self.X.columns)
        self.assertEqual(reglas.count("class:"), self.arbol.get_n_leaves())

    def test_p13_escenario_hora_pico_portal_es_critico(self):
        esc = [{"hora": 6, "tipo_dia": "Hábil", "tipo_estacion": "Portal", "zona_uso": "Residencial",
                "lluvia": 1, "temperatura_c": 10.5, "evento_masivo": 0}]
        self.assertEqual(self.arbol.predict(preparar_escenarios(esc, self.X.columns))[0], "Crítico")

    def test_p14_evento_masivo_sube_la_ocupacion(self):
        base = {"hora": 19, "tipo_dia": "Sábado", "tipo_estacion": "Intermedia", "zona_uso": "Recreativa",
                "lluvia": 0, "temperatura_c": 13.0}
        sin = self.arbol.predict(preparar_escenarios([{**base, "evento_masivo": 0}], self.X.columns))[0]
        con = self.arbol.predict(preparar_escenarios([{**base, "evento_masivo": 1}], self.X.columns))[0]
        self.assertGreater(ORDEN_NIVELES.index(con), ORDEN_NIVELES.index(sin))

    def test_p15_domingo_mediodia_universidad_es_bajo(self):
        esc = [{"hora": 12, "tipo_dia": "Domingo/Festivo", "tipo_estacion": "Intermedia",
                "zona_uso": "Universitaria", "lluvia": 0, "temperatura_c": 19.0, "evento_masivo": 0}]
        self.assertEqual(self.arbol.predict(preparar_escenarios(esc, self.X.columns))[0], "Bajo")


if __name__ == "__main__":
    unittest.main(verbosity=2)
