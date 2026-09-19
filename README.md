# Miniproyecto 2 - NLP

**El Transformer, pieza por pieza: qué aporta la atención sobre reseñas turísticas en español, y a qué costo**

Maestría · Universidad Icesi · Curso de Procesamiento de Lenguaje Natural

Autores: Juan José Aguado · Juan David Cruz · Juan Diego Ramírez

---

## El problema

Dada una reseña turística escrita en español sobre un destino mexicano, predecir cuántas
estrellas (1 a 5) le puso su autor, usando solo el texto.

Es la misma tarea del [Miniproyecto 1](../miniproyecto%201/). Lo que cambia es la técnica, y
con ella la pregunta:

> **¿Qué aporta realmente la atención sobre esta tarea, y a qué costo?**

Esa pregunta admite una respuesta trivial («el Transformer es mejor porque es más moderno»)
y una informativa. Buscamos la segunda, y para eso hace falta una hipótesis falsable.

Sale del EDA. Un MLP sobre bolsa de palabras es **invariante al orden**: para él «no está
mal» y «está mal, no» son la misma entrada. Y el análisis exploratorio mostró que el rasgo
más distintivo de las reseñas de 1★ es precisamente la **negación** — 3,75 negaciones por
cada 100 palabras, frente a 1,11 en las de 5★. De ahí:

- **H1.** El Transformer superará al MLP sobre todo en las clases negativas, porque puede
  ligar el adverbio de negación con el término al que afecta.
- **H2 (control).** Si se le quita la señal posicional, debería caer hacia el nivel del MLP.
  Si no cae, su ventaja —si la hay— no viene de usar el orden.

## La propuesta

Un Transformer *encoder* implementado **a mano**, sin `nn.Transformer`,
`nn.TransformerEncoderLayer` ni `nn.MultiheadAttention`. El producto punto escalado, la
separación en cabezas, la máscara de padding, las conexiones residuales y la normalización
por capa se escriben pieza por pieza.

No es purismo. Es lo que habilita el resto del notebook: teniendo las piezas escritas se
pueden **quitar una a una** y medir qué se pierde. Con la capa empaquetada de PyTorch, la
mitad de las secciones no existirían.

| Sección | Qué hace |
|---|---|
| §5–§6 | MLP simple y Transformer desde cero, mismo split y mismo bucle de entrenamiento |
| §7 | La comparación que pide la rúbrica, con F1 por clase y matrices de confusión |
| §8 | **Ablaciones estructurales**: sin residuales, `Flatten`, sin máscara, sin posicional |
| §9 | **Más allá del paper**: Pre-LN, token `[CLS]`, *label smoothing* aislado, BPE vs. vocabulario por palabras |
| §10 | **Barrido de hiperparámetros**, incluida la tasa `2e-5` del notebook guía |
| §11 | **El costo cuadrático de la atención, medido** (tiempo y memoria frente a `MAX_LEN`) |
| §12 | **Tarea de control**: la misma arquitectura prediciendo `Type` (balanceado, no ordinal) |
| §13 | Comparación contra los modelos del Miniproyecto 1, en métricas y en tiempo |
| §14 | Pesos de atención frente al léxico distintivo del EDA |
| §15 | **Demo** con pruebas de estrés dirigidas (con y sin negación, orden invertido) |

## Qué corregimos del notebook guía

El trabajo parte de `Sesion2/1-transformers-from-scratch.ipynb`. La rúbrica pide el
Transformer **en su forma original**, y el guía se desvía del paper en tres puntos que aquí
se corrigen, se explican y —lo más importante— **se miden** en la Sección 8:

| Pieza | Notebook guía | Aquí |
|---|---|---|
| Conexiones residuales | Ausentes: `LayerNorm(Sublayer(x))` | `LayerNorm(x + Sublayer(x))`, como el paper |
| Agregación de la secuencia | `Flatten` + `Linear(2048×256, 512)` ≈ 10⁸ parámetros | *Pooling* promedio enmascarado |
| Divisibilidad de cabezas | `assert embed_size & num_heads == 0` (AND de bits) | `assert emb_dim % num_heads == 0` |

Y una cuarta, más sutil, que no es un error de código sino una trampa: el guía entrena con
`lr = 2e-5`, que es una tasa de *fine-tuning* de un modelo preentrenado. Desde cero, el
modelo apenas se mueve — pero como el guía reporta solo *accuracy*, y predecir siempre 5★ ya
da 65,6 % en este corpus, un modelo que no aprendió nada parecería aceptable. La Sección 10
lo deja medido. Detalle en [`docs/DECISIONS.md`](docs/DECISIONS.md) §D-202 y §D-204.

## El corpus

[`vg055/Rest-Mex2025`](https://huggingface.co/datasets/vg055/Rest-Mex2025) — 208.051 reseñas
turísticas en español del shared task Rest-Mex 2025 (IberLEF). CC-BY-4.0.

**Se reutiliza del Miniproyecto 1 con autorización explícita del profesor**, bajo la
condición de que este notebook contenga el mismo EDA y sea autosuficiente. Por eso las
Secciones 2 y 3 están **copiadas sin modificar** —mismo código, mismas gráficas, mismas
lecturas— y una verificación programática lo comprueba celda por celda.

Como consecuencia, esas lecturas hablan del «modelo 3» y de la LSTM, que aquí no existen. No
es un descuido: corregirlas habría significado no entregar el mismo EDA. Dos **celdas
puente**, fuera del bloque heredado, avisan de ello y retraducen cada hallazgo a la sección
de esta entrega donde se aplica.

La tokenización, en cambio, **se hereda tal cual** del Miniproyecto 1: mismo vocabulario por
palabras de 30.000 tokens. Una primera versión de este trabajo planeó usar BPE como base,
razonando que el EDA (§3.7) mostró que el 26 % de los *tipos* de palabra no tiene vector
preentrenado; pero esa cifra es cobertura por tipos, y la que de verdad importa —por
apariciones— ya estaba medida en el propio Miniproyecto 1 en 98,5 %. Usar BPE como base
además mezclaría dos variables (tokenización y arquitectura) en la comparación de la
Sección 13. La decisión se revirtió (`docs/DECISIONS.md` §D-214): BPE se conserva como una
variante de preprocesamiento *medida*, no como la base del notebook.

Ficha completa: [`docs/DATASET.md`](docs/DATASET.md).

## Resultados

Corrida de referencia completa (`Restart & Run All`, sin errores, GPU RTX 3050 Laptop —
la misma máquina que el Miniproyecto 1). Detalle completo con todas las métricas,
ablaciones, barridos e interpretabilidad en
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

| Modelo | Entrega | macro-F1 | Accuracy | MAE | QWK |
|---|---|---:|---:|---:|---:|
| Baseline (clase mayoritaria) | — | 0.159 | 0.657 | 0.549 | 0.000 |
| **BiLSTM + atención** | MP1 | **0.527** | 0.676 | 0.362 | 0.755 |
| TF-IDF + Regresión Logística | MP1 | 0.524 | 0.679 | 0.376 | 0.726 |
| BiLSTM + spaCy (afinada) | MP1 | 0.486 | 0.636 | 0.431 | 0.694 |
| Transformer desde cero | MP2 | 0.416 | 0.586 | 0.605 | 0.507 |
| MLP simple | MP2 | 0.354 | 0.527 | 0.886 | 0.350 |
| LSTM desde cero | MP1 | 0.295 | 0.576 | 0.651 | 0.355 |

Se confirmó la expectativa que se registró antes de medir: con 32.000 reseñas de
entrenamiento, el Transformer desde cero queda **cuarto de seis**, por debajo de los tres
modelos "clásicos" del Miniproyecto 1. Sí supera claramente al MLP propio en las seis
métricas (la comparación que exige la rúbrica), y en la tarea de control `Type` (balanceada,
no ordinal) casi iguala a la mejor BiLSTM de MP1 (0.920 vs. 0.949) — la brecha se abre
específicamente en la tarea difícil, no por incapacidad general del modelo.

Sobre las dos hipótesis de investigación (`docs/SPEC.md` §1.2):

- **H1** (el Transformer gana sobre todo en clases negativas, por la negación): **parcial**.
  La mejora sobre el MLP es proporcionalmente mayor en 1★ (+38%) y 2★ (+105%) que en el
  resto, y hay 362 reseñas con negación donde el MLP falla y el Transformer acierta. Pero la
  cuantificación de atención no encuentra que el modelo ligue la negación con su término más
  de lo que lo haría por azar — la ventaja parece venir de mezclar contexto en general, no de
  un mecanismo de negación tan preciso como se planteaba.
- **H2** (sin señal posicional, cae al nivel del MLP): **no se sostiene** en esta corrida. El
  Transformer sin posición mantiene casi todo su rendimiento — documentado junto con la
  discrepancia frente a una corrida anterior en `docs/DECISIONS.md` §D-219.

La conclusión no es "el Transformer pierde": es qué haría falta para que ganara. El barrido
de hiperparámetros y el costo cuadrático medido en §10-§11 del notebook dan pistas concretas
(una tasa de aprendizaje mayor ayudó más que más profundidad; el costo de la atención crece
con el cuadrado de la longitud mientras el macro-F1 satura mucho antes), y las conclusiones
del notebook (§17) apuntan a lo que viene después en el curso: partir de un modelo
preentrenado en español.

## Cómo ejecutarlo

### Google Colab (recomendado)

Abrir `notebooks/miniproyecto2_restmex.ipynb` en Colab, seleccionar entorno con GPU T4 y
ejecutar todo. El notebook detecta el entorno e instala sus dependencias solo.

### Local

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows Git Bash
# source .venv/bin/activate       # Linux / macOS

pip install -r requirements.txt
python -m spacy download es_core_news_lg   # solo para el EDA §3.7

jupyter lab notebooks/miniproyecto2_restmex.ipynb
```

Sin GPU el notebook funciona igual: reduce la submuestra, el tamaño del modelo y el número
de épocas, y lo indica explícitamente en su salida.

## Estructura del repositorio

```
.
├── CLAUDE.md              Instrucciones de trabajo para agentes
├── README.md              Este archivo
├── consigna.txt           Enunciado del profesor (no modificar)
├── rubrica.txt            Criterios de calificación (no modificar)
├── requirements.txt
├── docs/
│   ├── SPEC.md            Especificación del notebook: el contrato
│   ├── PLAN.md            Plan de ejecución por fases, con estado
│   ├── DATASET.md         Ficha del corpus y estadísticas verificadas
│   ├── DECISIONS.md       Registro de decisiones de diseño (ADR)
│   └── EXPERIMENTS.md     Bitácora de resultados
├── notebooks/
│   └── miniproyecto2_restmex.ipynb
└── results/
    ├── figures/
    └── metrics/
```

## Cómo está organizado el trabajo

Mismo flujo *spec-driven* que la entrega anterior: la especificación se escribió antes de la
primera línea de código. `docs/SPEC.md` define qué se construye, `docs/PLAN.md` en qué orden,
`docs/DECISIONS.md` por qué se decidió así, y `docs/EXPERIMENTS.md` qué resultó.

`docs/SPEC.md` §6 mapea cada criterio de la rúbrica y cada frase de la consigna a la sección
del notebook que la satisface.
