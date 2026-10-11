# Mapa conceptual: aprendizaje supervisado con árboles de decisión

```mermaid
flowchart TD
    A["<b>Aprendizaje supervisado</b><br/>aprende de ejemplos con respuesta conocida"] --> B["Datos etiquetados<br/>(X = entradas, y = etiqueta)"]
    A --> C["Tipos de problema"]
    C --> C1["Clasificación<br/>(predice una categoría)"]
    C --> C2["Regresión<br/>(predice un número)"]
    C1 --> D["<b>Árbol de decisión</b><br/>(Palma Méndez, cap. 17)"]

    D --> E["Nodos internos:<br/>preguntas sobre una variable"]
    D --> F["Hojas:<br/>clase predicha"]
    D --> G["Criterio de división"]
    G --> G1["Entropía y<br/>ganancia de información<br/>(ID3, C4.5)"]
    G --> G2["Índice de Gini<br/>(CART)"]
    D --> H["Reglas de decisión<br/>SI ... ENTONCES<br/>(un camino = una regla)"]
    D --> I["Sobreajuste:<br/>el árbol memoriza"]
    I --> J["Poda:<br/>max_depth, min_samples_leaf"]
    J --> K["Validación cruzada<br/>(elige los hiperparámetros)"]

    B --> P["<b>Proyecto TransMilenio</b>"]
    P --> P1["Fuentes: afluencia por franja,<br/>estaciones, IDEAM, festivos, eventos"]
    P --> P2["Limpieza: duplicados,<br/>etiquetas mal escritas, lluvia faltante"]
    P --> P3["Entradas: hora, franja, tipo de día,<br/>zona, lluvia, temperatura, evento"]
    P --> P4["Etiqueta: nivel de ocupación<br/>Bajo / Medio / Alto / Crítico"]
    P3 --> D
    P4 --> D
    D --> M["Evaluación"]
    M --> M1["Entrenamiento 75% / prueba 25%"]
    M --> M2["Exactitud 0,904<br/>F1 macro 0,781"]
    M --> M3["Matriz de confusión"]
    M --> M4["Importancia de variables"]
```

**Cómo leerlo:** el aprendizaje supervisado necesita **datos etiquetados**. Para
un problema de **clasificación** se usa un **árbol de decisión**, que se construye
eligiendo en cada nodo la pregunta que más reduce la impureza (entropía o Gini).
Cada camino del árbol es una **regla**. Si el árbol crece sin control, se
**sobreajusta**, y por eso se **poda** con hiperparámetros elegidos por
**validación cruzada**. En el proyecto, las fuentes de TransMilenio se limpian y
se convierten en entradas y etiqueta, el árbol aprende de ellas y se evalúa con
datos que no vio durante el entrenamiento.
