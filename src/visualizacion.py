"""
visualizacion.py
================
Graficas del proyecto. Se guardan como PNG en resultados/ (backend "Agg":
no abre ventanas, funciona igual en Windows, Linux y Mac).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.tree import plot_tree  # noqa: E402

AZUL, NARANJA = "#2a78d6", "#eb6834"
# Escala ordinal (una sola familia de color, de claro a oscuro) para los niveles.
COLOR_NIVEL = {"Bajo": "#9ec5f4", "Medio": "#5598e7", "Alto": "#256abf", "Crítico": "#0d366b"}
TINTA, TINTA_SUAVE, REJILLA = "#2b2b2b", "#6b6b6b", "#e6e6e6"


def _estilo(ax, titulo, eje_x="", eje_y=""):
    ax.set_title(titulo, fontsize=12, color=TINTA, loc="left")
    ax.set_xlabel(eje_x, color=TINTA_SUAVE)
    ax.set_ylabel(eje_y, color=TINTA_SUAVE)
    ax.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ["top", "right"]:
        ax.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        ax.spines[lado].set_color("#bdbdbd")
    ax.tick_params(colors=TINTA_SUAVE)


def grafica_distribucion(y, orden, ruta):
    """Cuantos registros hay de cada nivel de ocupacion (balance de clases)."""
    conteo = y.value_counts().reindex(orden)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    barras = ax.bar(orden, conteo.values, color=[COLOR_NIVEL[c] for c in orden], width=0.6)
    for b, v in zip(barras, conteo.values):
        ax.annotate(f"{v:,}\n({v / conteo.sum():.0%})".replace(",", "."), (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9, color=TINTA)
    _estilo(ax, "Distribución de la etiqueta: nivel de ocupación", "", "Número de registros")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, conteo.max() * 1.18)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def grafica_profundidad(curva, profundidad_elegida, ruta):
    """F1 en entrenamiento vs. prueba segun la profundidad: muestra el sobreajuste."""
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.plot(curva["profundidad"], curva["f1_entrenamiento"], color=AZUL, linewidth=2, marker="o",
            markersize=5, label="Entrenamiento")
    ax.plot(curva["profundidad"], curva["f1_prueba"], color=NARANJA, linewidth=2, marker="s",
            markersize=5, label="Prueba (datos nuevos)")
    ax.axvline(profundidad_elegida, color=TINTA_SUAVE, linestyle="--", linewidth=1.2)
    ax.text(profundidad_elegida + 0.2, 0.25, f"profundidad elegida\npor validación cruzada = {profundidad_elegida}",
            color=TINTA, fontsize=9)
    _estilo(ax, "Sobreajuste: entre más profundo el árbol, más memoriza", "Profundidad máxima del árbol",
            "F1 macro")
    ax.set_xticks(list(curva["profundidad"]))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def grafica_matriz_confusion(matriz, orden, ruta):
    """Matriz de confusion: filas = clase real, columnas = clase predicha."""
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    porcentaje = matriz / matriz.sum(axis=1, keepdims=True)
    ax.imshow(porcentaje, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(orden)):
        for j in range(len(orden)):
            color = "white" if porcentaje[i, j] > 0.55 else TINTA
            ax.text(j, i, f"{matriz[i, j]}\n{porcentaje[i, j]:.0%}", ha="center", va="center",
                    color=color, fontsize=10)
    ax.set_xticks(range(len(orden)), orden)
    ax.set_yticks(range(len(orden)), orden)
    ax.set_xlabel("Predicción del árbol", color=TINTA_SUAVE)
    ax.set_ylabel("Valor real", color=TINTA_SUAVE)
    ax.set_title("Matriz de confusión (conjunto de prueba)", fontsize=12, color=TINTA, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def grafica_importancia(importancias, ruta, top=10):
    """Las variables que mas usa el arbol para decidir."""
    imp = importancias.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(imp.index, imp.values, color=AZUL, height=0.6)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=9, color=TINTA)
    _estilo(ax, "Importancia de las variables en el árbol", "Importancia (reducción de impureza)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=130)
    plt.close(fig)


def grafica_arbol(modelo, columnas, clases, ruta, profundidad=3):
    """Dibuja los primeros niveles del arbol (el arbol completo no cabe en una imagen)."""
    fig, ax = plt.subplots(figsize=(28, 9))
    plot_tree(modelo, feature_names=list(columnas), class_names=list(clases), max_depth=profundidad,
              filled=True, rounded=True, impurity=False, proportion=True, fontsize=8, ax=ax)
    ax.set_title(f"Primeros {profundidad} niveles del árbol de decisión  "
                 f"(value = proporción de cada clase en orden: {', '.join(clases)})",
                 fontsize=13, color=TINTA, loc="left")
    fig.tight_layout()
    fig.savefig(ruta, dpi=110)
    plt.close(fig)
