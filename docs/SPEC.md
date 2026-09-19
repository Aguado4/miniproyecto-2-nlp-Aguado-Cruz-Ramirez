# SPEC - Especificación del entregable

> **Estado:** aprobado · **Versión:** 1.0 · **Última actualización:** 2026-09-12
>
> Este documento es el contrato del entregable. Define qué debe contener el notebook,
> con qué criterios se acepta cada sección y cómo se mide el éxito.
> Si la implementación se desvía, se actualiza primero este archivo.

---

## 1. Problema

### 1.1 Enunciado

Dada una reseña turística escrita en español por un visitante de un destino mexicano,
predecir la **polaridad** que el autor asignó (1 a 5 estrellas) usando únicamente el texto.

Es la misma tarea del Miniproyecto 1. Lo que cambia es la técnica, y con ella la pregunta de
investigación:

> **¿Qué aporta realmente la atención sobre esta tarea, y a qué costo?**

### 1.2 Por qué esta pregunta y no otra

La rúbrica pide implementar el Transformer en su forma original y compararlo contra un
modelo simple de tipo MLP. Esa comparación admite dos lecturas: una trivial («el Transformer
es mejor porque es más moderno») y una informativa, que es la que perseguimos.

La lectura informativa se apoya en una propiedad concreta de los dos modelos. Un MLP sobre
bolsa de palabras es **invariante al orden**: «no está mal» y «está mal, no» son el mismo
vector. El EDA §3.5 del Miniproyecto 1 mostró que el rasgo más distintivo de las reseñas de
1★ es precisamente la **negación** (3,75 negaciones por cada 100 palabras, frente a 1,11 en
las de 5★). Hay por tanto una hipótesis falsable:

> **H1.** El Transformer superará al MLP principalmente en las clases negativas, porque
> puede ligar el adverbio de negación con el término al que afecta, y el MLP no.

Y una hipótesis de control que la Sección 8 pone a prueba directamente:

> **H2.** Si se elimina la señal posicional, el Transformer debería caer hacia el nivel del
> MLP. Si no cae, entonces su ventaja —si la hay— no viene de usar el orden.

### 1.3 Por qué la propuesta es adecuada

Tres razones:

1. **El corpus tiene la propiedad que la arquitectura promete explotar.** No es un corpus
   cualquiera: la negación y las reseñas mixtas («la comida excelente pero el servicio
   pésimo») son fenómenos de dependencia a distancia, que es justo lo que la atención
   modela. La tarea puede por tanto discriminar entre arquitecturas, en lugar de dar el
   mismo número a todas.
2. **Implementar a mano habilita el experimento.** Escribir cada pieza no es un ejercicio
   de estilo: es lo que permite la Sección 8, donde se quita una pieza a la vez y se mide
   qué se pierde. Con `nn.TransformerEncoderLayer` esa sección no existiría.
3. **Hay una línea base ya medida.** El Miniproyecto 1 dejó tres modelos evaluados sobre
   esta misma submuestra y este mismo split. El Transformer no se compara contra el vacío,
   sino contra una escalera de representaciones ya construida.

## 2. Datos

`vg055/Rest-Mex2025` en HuggingFace Hub. Detalle completo en [`DATASET.md`](DATASET.md).

Resumen: 208,051 reseñas, campos `Title`, `Review`, `Polarity` (1–5), `Town` (40),
`Region` (19), `Type` (3). Licencia CC-BY-4.0.

- **EDA:** sobre el corpus completo (208k), reproducido del Miniproyecto 1.
- **Modelado:** submuestra **estratificada por polaridad de 40,000 reseñas**, semilla 42,
  **idéntica a la del Miniproyecto 1** (ver `DECISIONS.md` §D-207).

## 3. Protocolo experimental

### 3.1 Particiones

**Split A — aleatorio estratificado**, 80/10/10, estratificado por `Polarity`, `SEED = 42`.
Es el mismo del Miniproyecto 1 y el único que se usa en esta entrega: todas las
comparaciones, internas y entre proyectos, se hacen sobre él.

(El Split B geográfico del Miniproyecto 1 no se reutiliza aquí. Su pregunta —generalización
a destinos nuevos— ya se respondió allá y no es la pregunta de esta entrega.)

### 3.2 Métricas

Las mismas seis del Miniproyecto 1, calculadas por la misma función para todos los modelos:

| Métrica | Rol | Por qué |
|---|---|---|
| **macro-F1** | **Principal** | Da el mismo peso a las 5 clases; es la única que penaliza abandonar 1★ y 2★ |
| Accuracy | Contexto | Solo se reporta junto al baseline mayoritario (65.6%) |
| **MAE** | Ordinal | Mide la magnitud del error en estrellas |
| **QWK** | Ordinal | Acuerdo corregido por azar y ponderado por distancia |
| F1 por clase | Diagnóstico | Muestra dónde mejora la atención: si en las minoritarias o solo en 5★ |
| Tiempo de entrenamiento | Costo | Es la mitad de la pregunta de §1.1 («¿y a qué costo?») |

Se añade una séptima columna: **número de parámetros**, desglosado por componente
(embeddings / bloques / clasificador). En un Transformer mal ensamblado la capacidad se va
al clasificador; hacerlo visible es parte del análisis (ver `DECISIONS.md` §D-205).

### 3.3 Baselines de referencia obligatorios

- **Clase mayoritaria** (predecir siempre 5★): accuracy ≈ 0.656, macro-F1 ≈ 0.158.
- **Azar estratificado**.

Deben reproducir los valores del Miniproyecto 1 (`EXPERIMENTS.md` §1 de aquella entrega).
Si no los reproducen, la submuestra o el split no son los mismos y la Sección 13 queda
invalidada: hay que detenerse y corregir antes de seguir.

## 4. Estructura del notebook

Archivo único: `notebooks/miniproyecto2_restmex.ipynb`

### Secciones 0–3 — Contexto heredado del Miniproyecto 1

Portada y problema (propias de esta entrega), entorno y reproducibilidad (adaptados: `CFG`
contiene los hiperparámetros del Transformer), y a partir de ahí:

> **Las celdas 12–53 del notebook del Miniproyecto 1 —Sección 2 «El corpus» y Sección 3
> «Análisis exploratorio» completa— se copian SIN MODIFICAR.** Mismo código, mismas
> gráficas, mismas lecturas, mismos números. Es la condición que puso el profesor al
> autorizar la reutilización del corpus.

Como consecuencia, las lecturas del EDA están escritas en los términos del Miniproyecto 1
(hablan del «modelo 3», de la LSTM, de secciones con otro número). Eso **no se corrige
dentro del EDA**. En su lugar hay dos **celdas puente**, fuera del bloque heredado:

- una **antes** de la Sección 2, que advierte al lector de qué está leyendo y por qué;
- una **después** de §3.8, que retraduce cada hallazgo del EDA a la sección de esta entrega
  donde se aplica.

**Aceptación:** una verificación programática confirma que las 42 celdas heredadas son
idénticas a las del Miniproyecto 1, y el notebook se puede leer de principio a fin sin abrir
la entrega anterior.

### Sección 4 — Preprocesamiento y protocolo

> **Nota (D-214, 2026-09-15):** esta sección se revirtió después de escrita esta tabla.
> El tokenizador base ya **no** es BPE: se heredó tal cual el vocabulario por palabras del
> Miniproyecto 1, para que la Sección 13 compare una sola variable (arquitectura) contra
> aquella entrega. BPE no desapareció — pasó a ser una variante de preprocesamiento medida
> en §8 («BPE vs. palabras», D-216), no la base de todo el notebook. La tabla original queda
> tachada por trazabilidad; la columna «Vigente» dice qué aplica hoy.

| # | Contenido (versión original, 2026-09-12) | Vigente |
|---|---|---|
| ~~4.1~~ | ~~Tokenizador BPE entrenado solo sobre train~~ | Tokenizador y vocabulario por palabras **heredados del Miniproyecto 1** (`most_common`, 30.000 tokens, solo train de Split A) |
| ~~4.2~~ | ~~`MAX_LEN` = P95 en tokens BPE~~ | `MAX_LEN = 150` **heredado sin recalcular** (D-206); BPE se mide aparte en §8 |
| 4.3 | Submuestra 40k y Split A idénticos al Miniproyecto 1 | Vigente sin cambios. Se imprime la distribución de clases de las tres particiones |
| 4.4 | `Dataset` y `DataLoader` con `input_ids`/`ids`, `attention_mask`/máscara, `y` | Vigente sin cambios |
| 4.5 | Función única `evaluar(...)` + baselines | Vigente sin cambios. Los baselines reproducen los del Miniproyecto 1 |

### Sección 5 — Modelo A: MLP simple

Bolsa de palabras (o embeddings promediados con máscara) → capas densas → 5 clases.
Mismo tokenizador, misma partición, misma ponderación de clases, mismo optimizador y mismo
criterio de parada que el Transformer: la **única** diferencia es la arquitectura.

**Aceptación:** supera ambos baselines en macro-F1 y se reporta su F1 por clase.

### Sección 6 — Modelo B: Transformer encoder desde cero

Se construye pieza por pieza, cada una con su prueba de formas y su explicación:

| # | Pieza | Requisito específico |
|---|---|---|
| 6.1 | Token embeddings + *positional encoding* sinusoidal | Escalado por √d_model antes de sumar la posición (paper §3.4); visualización de la matriz |
| 6.2 | Multi-head self-attention | Implementación propia; máscara de padding con `-inf` antes del softmax; dropout sobre los pesos; `return_attention` para la §10 |
| 6.3 | Bloque codificador | **Conexiones residuales** en ambos sublayers + LayerNorm; parámetro para desactivarlas en §8 |
| 6.4 | Clasificador | **Pooling promedio enmascarado**; desglose de parámetros por componente; parámetro para alternar a `flatten` en §8 |
| 6.5 | Entrenamiento | `lr = 3e-4` con warmup lineal; early stopping por **macro-F1 de validación**; pérdida ponderada por clase; cronometrado |

**Prohibido:** `nn.Transformer`, `nn.TransformerEncoderLayer`, `nn.MultiheadAttention`.

**Aceptación:** el bucle de entrenamiento es el mismo que usa el Modelo A, de modo que la
comparación de tiempos no mezcle diferencias de implementación con diferencias de
arquitectura.

### Sección 7 — Comparación MLP vs. Transformer

La comparación que la rúbrica nombra explícitamente. Tabla única con las siete columnas de
§3.2, F1 por clase para ambos, matrices de confusión lado a lado, y **ejemplos concretos que
el MLP falla y el Transformer acierta**, buscados específicamente entre reseñas con negación
para contrastar H1.

**Aceptación:** la lectura responde si la atención compensa su costo, con los números de la
tabla y no con la expectativa.

### Sección 8 — Ablaciones estructurales (aporte propio)

Cada fila es el mismo modelo con **una sola** diferencia, semilla fijada antes de cada
corrida, mismo split, misma configuración reducida (`CFG_ABL`):

| Ablación | Pregunta |
|---|---|
| Sin conexiones residuales (versión del guía) | ¿Cuánto cuesta omitir el atajo residual? |
| `Flatten` en vez de pooling enmascarado | ¿Compra algo esa capa densa gigantesca? |
| Pooling sin máscara | ¿Cuánto contamina promediar sobre padding? |
| Sin *label smoothing* | ¿Aporta sobre una escala ordinal de fronteras difusas? |
| **Sin señal posicional** | **Control de H2**: ¿cae hasta el nivel del MLP? |

**Aceptación:** se declara explícitamente que sus filas comparan entre sí, no contra la
tabla de §7. Toda ablación que no empeore (o que mejore) se reporta tal cual, con su
explicación.

### Sección 9 — Más allá del paper (aporte propio)

> **Nota (D-216, 2026-09-15):** RoPE se retiró de esta sección por presupuesto y se
> reemplazó por una pregunta que ataca el mismo tipo de duda a menor costo — si el
> vocabulario por palabras (elegido en D-214 para comparar limpio con el Miniproyecto 1) le
> pone el listón bajo al Transformer frente a su preprocesamiento natural. Quedan tres
> variantes propias más la comparación de tokenización.

| Variante | Qué cambia | Motivación |
|---|---|---|
| **Pre-LN** | `x + Sublayer(LayerNorm(x))` | Estabilidad de gradiente en capas altas; tolera tasas mayores (Xiong et al., 2020) |
| **Token `[CLS]`** | Token dedicado en vez de promediar | El promedio diluye una queja breve dentro de una reseña larga |
| ***Label smoothing*** (aislado) | Reparte masa fuera de la etiqueta | Está en el paper (§5.4) y el guía lo omite; debería notarse en MAE y QWK |
| **BPE vs. palabras** (D-216, reemplaza a RoPE) | Mismo Transformer, entrenado con el tokenizador BPE de §8 en vez del vocabulario por palabras | ¿El preprocesamiento heredado de MP1 perjudica al Transformer frente a su tokenización natural? |

**Aceptación:** cada variante se compara contra el modelo base de §6 con la misma
configuración reducida de §8. Se advierte que muchas de estas mejoras se diseñaron para
regímenes de datos y profundidad muy distintos del nuestro.

### Sección 10 — Barrido de hiperparámetros

Barrido **univariado** (una variable a la vez desde la configuración base), no rejilla:

| Hiperparámetro | Valores |
|---|---|
| Tasa de aprendizaje | 2e-5 (la del guía) · 1e-4 · 3e-4 · 1e-3 |
| Número de bloques | 1 · 2 · 4 |
| Número de cabezas | 1 · 4 · 8 |
| Dropout | 0,0 · 0,1 · 0,3 |

**Aceptación:** la fila de `lr = 2e-5` se comenta explícitamente. Es la verificación empírica
de `DECISIONS.md` §D-204: si con esa tasa el macro-F1 se queda cerca de 0,16, entonces un
notebook que la hubiera copiado y reportado solo accuracy habría mostrado un 65 % de aspecto
aceptable y concluido que el modelo funciona.

### Sección 11 — El costo cuadrático, medido

Mismo modelo con `MAX_LEN` ∈ {64, 128, 256, 512}, registrando segundos por época, memoria
pico de GPU y macro-F1.

**Aceptación:** una gráfica de doble eje muestra el tiempo creciendo de forma supralineal y
el macro-F1 saturando; se marcan el P95 elegido en §4.2 y el 2.048 del notebook guía. Es el
argumento cuantitativo de §4.2 y de `DECISIONS.md` §D-206.

### Sección 12 — Tarea de control: `Type`

La misma arquitectura, sin cambiar una línea salvo el número de clases, prediciendo `Type`
(3 clases, balanceadas, no ordinal).

**Aceptación:** se compara su macro-F1 contra el de polaridad y contra el 0,949 que obtuvo
la BiLSTM del Miniproyecto 1 en esta misma tarea. La diferencia entre ambos macro-F1 es la
medida de cuánta dificultad pertenece a la tarea y no al modelo.

### Sección 13 — Comparación contra el Miniproyecto 1

Tabla única con los modelos de las dos entregas: TF-IDF + LogReg, LSTM desde cero,
BiLSTM + spaCy, BiLSTM + atención (de `../miniproyecto 1/docs/EXPERIMENTS.md` §2), MLP y
Transformer. Gráfica de costo (tiempo, parámetros) contra beneficio (macro-F1).

**Aceptación:** el entorno de medición de cada fila queda explícito. Si los tiempos del
Miniproyecto 1 vienen de otro hardware, se marcan como tales (ver `DECISIONS.md` §D-209).

### Sección 14 — ¿En qué se fija el Transformer?

Mapa de calor token × token sobre una reseña con negación, texto coloreado por peso
agregado, comparación entre cabezas, y contraste agregado contra el léxico distintivo del
EDA §3.5.

**Aceptación:** se responde si H1 se sostiene, y se advierte explícitamente que los pesos de
atención son una pista de dónde mira el modelo, no una explicación causal de su decisión.

### Sección 15 — Demo (aporte propio)

Función `predecir_resena(texto)` que devuelve estrellas, distribución de probabilidad y
pesos de atención, con salida visual. Se ejecutan tres **pruebas de estrés dirigidas**:

| Par de prueba | Qué revela |
|---|---|
| «la comida estuvo buena» vs. «…no estuvo buena» | ¿Responde a la negación? El MLP, por construcción, apenas puede |
| «el hotel bien, la comida pésima» vs. orden invertido | ¿Importa el orden, o promedia sentimientos? |
| Reseña larga con la queja al final | ¿La ve, o el promedio la diluye? |

**Aceptación:** el widget interactivo es un extra; los tres pares de prueba se ejecutan con
texto fijo y sus salidas quedan guardadas, para que «Restart & Run All» siga siendo
reproducible sin intervención manual. Se declara que la evidencia es anecdótica.

### Sección 16 — Análisis de errores

Errores por distancia en estrellas, reseñas que fallan todos los modelos agrupadas por causa
(ironía, reseña mixta, texto muy corto, etiqueta incoherente), con ejemplos concretos. Para
los errores del Transformer se miran además sus pesos de atención.

### Sección 17 — Conclusiones y limitaciones

Respuesta explícita a la pregunta de §1.1, veredicto sobre H1 y H2, qué pieza de la
arquitectura resultó imprescindible y si alguna variante posterior la mejoró, qué dice la
tarea de control, limitaciones honestas y trabajo futuro.

## 5. Criterios de aceptación del entregable

- [x] «Restart & Run All» completo sin errores. **~61 min medidos** en GPU RTX 3050 Laptop
      local, no ≤35 min: se documenta como limitación honesta en vez de ajustar el número
      (`EXPERIMENTS.md` §13, notebook §17). La corrida final en Colab/T4 queda a cargo del
      usuario para confirmar el tiempo en esa plataforma.
- [x] El Transformer está implementado a mano: cero uso de `nn.Transformer`,
      `nn.TransformerEncoderLayer` o `nn.MultiheadAttention` (verificado por inspección).
- [x] Las conexiones residuales, la máscara de padding y el pooling enmascarado están
      presentes y explicados frente a lo que hace el notebook guía.
- [x] La comparación Transformer vs. MLP está en una tabla única con las mismas métricas
      (§7; macro-F1 0.416 vs. 0.354).
- [x] Las ablaciones de §8 y los barridos de §9–§11 fijan la semilla antes de cada corrida.
- [x] §9 implementa tres variantes/comparaciones posteriores al paper (D-216: Pre-LN,
      `[CLS]`, label smoothing aislado y BPE-vs-palabras; RoPE se retiró, ver nota de §9).
- [x] §12 (tarea de control `Type`) ejecutada y comparada contra polaridad (+0.504) y
      contra el 0.949 del Miniproyecto 1.
- [x] §15 (demo) ejecuta los tres pares de prueba con texto fijo y salida guardada.
- [x] La tabla de §13 incluye los modelos del Miniproyecto 1 con su entorno de medición.
- [x] **El bloque EDA es idéntico al del Miniproyecto 1**, verificado con
      `scripts/verificar_eda_mp1.py` (control de autoría en local, no una celda del
      notebook — ver `DECISIONS.md` §D-217; Colab no puede clonar el repo del MP1 sin
      credenciales). Resultado: 39/42 celdas idénticas byte a byte, 3 con diferencia
      cosmética de codificación.
- [x] El notebook se lee sin abrir la entrega anterior.
- [x] Ninguna celda de código sin markdown explicativo previo.
- [x] Ninguna gráfica ni tabla sin su lectura escrita.
- [x] Todo en español, incluidas las etiquetas de los ejes.
- [x] Cero `TODO` y cero comentarios de andamiaje (`<!-- REDACTAR -->`, `<!-- LEER -->`)
      fuera del bloque heredado de MP1, que se conserva íntegro por definición
      (`DECISIONS.md` §D-220).
- [x] `EXPERIMENTS.md` con los resultados reales de la corrida final.
- [x] Salidas de celdas guardadas en el `.ipynb` versionado.

## 6. Cobertura de la rúbrica

| Criterio | Puntos | Dónde se satisface |
|---|---:|---|
| Notebook completo | 1 | Estructura de §4, narrativa en toda celda markdown, conclusiones en §17 |
| Reproducibilidad | 2 | §1 (semilla, config adaptativa), `CFG_ABL`, semilla fijada por corrida, presupuesto del `PLAN.md` |
| Transformer original + comparación con MLP | 2 | §6 (implementación a mano, pieza por pieza), §5 (MLP), §7 (comparación), §8 (fidelidad al paper medida) |
| Innovación | 2 | §8 ablaciones · §9 variantes posteriores al paper · §10 barrido · §11 costo cuadrático medido · §12 tarea de control · §13 comparación entre entregas · §14 interpretabilidad · §15 demo |

Y frente a la consigna: *«prueba diferentes técnicas, arquitecturas, parámetros»* → §5-§6,
§8-§9, §10. *«técnicas más allá de las vistas en clase»* → §9 (Pre-LN, `[CLS]`, BPE vs.
palabras). *«hacer demos»* → §15. *«combinar con otro tipo de casos»* → §12. *«añadir plots,
otro tipo de análisis»* → §11 y §14.

## 7. Fuera de alcance

- Modelos preentrenados (BETO, BERT multilingüe): son materia de una entrega posterior.
- El decodificador del Transformer: la tarea es clasificación, solo hace falta el encoder.
- El Split B geográfico y las formulaciones ordinales de pérdida: ya respondidos en el
  Miniproyecto 1.
- Búsqueda exhaustiva de hiperparámetros. §10 es un barrido univariado para acotar la
  sensibilidad, no una optimización; se documenta la diferencia.
- Despliegue o API. La demo de §15 vive dentro del notebook.
