# CLAUDE.md — Instrucciones de trabajo para este repositorio

Este archivo es la fuente de verdad operativa para cualquier sesión de agente que trabaje
en este proyecto. Léelo completo antes de tocar cualquier archivo.

---

## 1. Qué es este repositorio

Entregable del **Miniproyecto 2** del curso de NLP (Maestría, Universidad Icesi).
El producto final es **un único Jupyter Notebook** ejecutable de principio a fin que
implementa un **Transformer encoder desde cero** (arquitectura original del paper
*Attention Is All You Need*) sobre un corpus propio de reseñas turísticas en español, y lo
compara contra un **MLP simple** y contra los modelos del Miniproyecto 1.

Los dos archivos que definen el éxito del trabajo están en la raíz y **no se modifican**:

- `consigna.txt` — qué pide el profesor.
- `rubrica.txt` — cómo se califica (7 puntos en 4 criterios).

Toda decisión de diseño debe poder justificarse contra uno de esos dos archivos.

### Relación con el Miniproyecto 1

Este trabajo **reutiliza el corpus y el EDA** del Miniproyecto 1 (`../miniproyecto 1/`).
El profesor autorizó explícitamente esa reutilización, con una condición: el notebook debe
**contener el mismo EDA**, para que sea autosuficiente y se pueda leer sin abrir la entrega
anterior. Eso ya está hecho: las secciones 1–3 del notebook son las del Miniproyecto 1, con
las lecturas reorientadas hacia las decisiones de esta entrega (ver `docs/DECISIONS.md`
§D-201).

El Miniproyecto 1 es además la **fuente de comparación** de la Sección 9: sus resultados
están en `../miniproyecto 1/docs/EXPERIMENTS.md` y se midieron sobre la misma submuestra,
el mismo Split A y las mismas métricas.

## 2. Documentos de especificación (leer en este orden)

| Archivo | Contiene |
|---|---|
| `docs/SPEC.md` | Especificación completa del notebook: secciones, modelos, métricas, criterios de aceptación. **Es el contrato.** |
| `docs/PLAN.md` | Plan de ejecución por fases, con estado y orden de implementación. |
| `docs/DATASET.md` | Ficha del corpus: esquema, estadísticas verificadas, sesgos conocidos, licencia. |
| `docs/DECISIONS.md` | Registro de decisiones de diseño (ADR). Toda decisión no trivial se documenta aquí. |
| `docs/EXPERIMENTS.md` | Bitácora de resultados. Se llena a medida que se ejecutan los experimentos. |

**Flujo de trabajo:** `SPEC.md` define *qué*, `PLAN.md` define *en qué orden*,
`DECISIONS.md` registra *por qué*, `EXPERIMENTS.md` registra *qué salió*.
Si implementas algo que no está en el SPEC, primero actualiza el SPEC.

## 3. Reglas duras (violarlas cuesta puntos de rúbrica)

### 3.1 Reproducibilidad — vale 2 de 7 puntos

- **Toda celda debe correr desde cero, en orden, sin intervención manual.**
  El criterio de aceptación es «Restart & Run All» limpio.
- **Semilla global `SEED = 42`** aplicada a `random`, `numpy`, `torch` y a todo split.
  En este proyecto pesa doble: la Sección 8 compara corridas que difieren en una sola pieza
  de la arquitectura. Sin semilla fija esa comparación no mide nada. **Fijar la semilla
  antes de cada corrida de ablación**, no solo una vez al principio del notebook.
- **Nada de rutas locales.** El corpus se descarga desde HuggingFace en la celda de carga.
- **El notebook debe detectar Colab vs. local y GPU vs. CPU**, y degradarse sin romperse
  (modelo más pequeño / menos épocas / submuestra menor en CPU), nunca fallar.
- **Presupuesto de tiempo: ~30 min end-to-end en GPU T4.** El riesgo principal es la
  Sección 8 (varias corridas de ablación): se entrena con configuración reducida y se dice
  explícitamente que sus filas comparan entre sí, no contra la tabla de la Sección 7.
- Las salidas de las celdas se guardan en el `.ipynb` versionado. El profesor debe poder
  leer los resultados sin ejecutar nada.

### 3.2 Narrativa — vale 1 de 7 puntos

- **Todo en español.** Código, comentarios, markdown, gráficas, ejes, títulos.
- **Ninguna celda de código sin una celda markdown antes que explique el porqué**, no el qué.
- **Toda gráfica y toda tabla va seguida de su lectura.** Un número sin interpretación no
  cuenta como análisis.
- Cada sección de modelo cierra con un párrafo de hallazgos; el notebook cierra con
  conclusiones globales y limitaciones honestas.

### 3.3 Transformer en su forma original + comparación con MLP — vale 2 de 7 puntos

Es el criterio central de esta entrega y el que la distingue del Miniproyecto 1.

- **«En su forma original» es literal.** Prohibido `nn.Transformer`,
  `nn.TransformerEncoderLayer` y `nn.MultiheadAttention`. El producto punto escalado, la
  separación en cabezas, la máscara de padding, las conexiones residuales y la
  normalización por capa se escriben a mano. De `transformers`/`tokenizers` solo se usa el
  tokenizador BPE (§4.1); ninguna capa del modelo sale de ahí.
- **La comparación con un MLP simple es obligatoria**, no opcional: la rúbrica la nombra
  explícitamente. Debe estar en una tabla única con las mismas métricas y el mismo split.
- **El notebook guía (`Sesion2/1-transformers-from-scratch.ipynb`) se desvía del paper en
  tres puntos** que aquí se corrigen y se documentan: faltan las conexiones residuales, el
  clasificador aplana la secuencia en vez de agregarla, y el `assert` de divisibilidad de
  cabezas usa `&` en vez de `%`. Ver `docs/DECISIONS.md` §D-202. **No replicar esos
  defectos**; explicarlos es evidencia de comprensión y suma en Innovación.

### 3.4 Innovación — vale 2 de 7 puntos

El diferencial de esta entrega son las **ablaciones de la Sección 8**: teniendo la
arquitectura escrita pieza por pieza, se quita una pieza a la vez y se mide qué se pierde.
Eso solo es posible porque el modelo está implementado a mano, y es el argumento de por qué
valía la pena implementarlo así.

A eso se suman la comparación entre entregas (Sección 9) y la interpretación de los pesos de
atención frente al léxico del EDA (Sección 10).

## 4. Convenciones técnicas

- **El notebook es autocontenido.** No se importa nada de un `src/` del repo: en Colab eso
  obligaría a clonar y es un punto de fallo de reproducibilidad.
- Funciones auxiliares se definen en el propio notebook, en la sección donde se usan por
  primera vez.
- Métrica principal: **macro-F1**. Nunca reportar accuracy sin ponerlo al lado del baseline
  de clase mayoritaria (65.6%). El notebook guía reporta solo accuracy; sobre este corpus
  esa métrica no distingue un modelo entrenado de uno que no aprendió nada.
- La función `evaluar(...)` es única y la usan **todos** los modelos, incluidos los del MLP
  y los de las ablaciones. Es lo que permite pegar en una sola tabla resultados de dos
  notebooks distintos.
- Gráficas: `matplotlib` + `seaborn`, paleta consistente, ejes etiquetados en español.
- Nombres de variables y funciones en español o inglés, pero **consistentes**.

## 5. Qué NO hacer

- No usar `nn.Transformer`, `nn.TransformerEncoderLayer` ni `nn.MultiheadAttention`
  (ver §3.3). Sí se pueden usar para *verificar* que la implementación propia da lo mismo,
  si se documenta como tal.
- No partir de un modelo preentrenado (BETO, BERT multilingüe). Eso es materia de una
  entrega posterior del curso; aquí el punto es entrenar desde cero.
- No entrenar el tokenizador BPE sobre el corpus completo incluyendo el test: es una fuga
  de información sutil que el notebook guía comete. Entrenarlo solo con el split de train.
- No añadir dependencias fuera de `requirements.txt` sin actualizarlo y sin justificar el
  peso de la descarga (Colab reinstala todo en cada sesión).
- No dejar celdas con `!pip install` dispersas; toda la instalación va en la sección 1.
- No dejar `TODO`, celdas vacías, ni comentarios de andamiaje (`<!-- REDACTAR -->`,
  `<!-- LEER -->`) en el entregable final. El esqueleto los trae a propósito; se van
  llenando y desapareciendo a medida que se implementa.
- No inventar resultados en `EXPERIMENTS.md`: solo se registra lo que efectivamente se ejecutó.
- No hacer `git push` ni crear releases sin pedirlo explícitamente al usuario.

## 6. Comandos útiles

```bash
# Entorno local (Windows, Git Bash)
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements.txt
python -m spacy download es_core_news_lg   # solo para el EDA §3.7

# Verificar que el notebook corre limpio de principio a fin
jupyter nbconvert --to notebook --execute notebooks/miniproyecto2_restmex.ipynb \
  --output ejecutado.ipynb --ExecutePreprocessor.timeout=3600
```
