# DECISIONS - Registro de decisiones de diseño

Formato ADR abreviado. Cada entrada: contexto, decisión, alternativas descartadas y
consecuencias. Se añade una entrada cada vez que se toma una decisión no obvia.

Estados: `aceptada` · `propuesta` · `revisada` · `revertida`

> La numeración empieza en **D-201** para no colisionar con las del Miniproyecto 1
> (`D-001`–`D-009`), que se citan aquí como `MP1 §D-00N`.

---

## D-201 · Reutilizar el corpus y el EDA del Miniproyecto 1

**Estado:** aceptada · 2026-09-12

**Contexto.** La consigna prohíbe reutilizar los datos de los notebooks guía, pero no dice
nada sobre reutilizar los datos de la entrega anterior del propio estudiante. El profesor
confirmó que sí se puede, con una condición explícita: que el notebook **contenga el mismo
EDA**, de modo que sea autosuficiente y se pueda evaluar sin abrir el Miniproyecto 1.

**Decisión.** Se reutiliza `vg055/Rest-Mex2025` y se copian **sin modificar** las celdas
12–53 del notebook anterior: la Sección 2 («El corpus») y la Sección 3 («Análisis
exploratorio», §3.1–§3.8) completas. Mismo código, mismas gráficas, mismas lecturas, mismos
números. Una verificación programática compara celda por celda contra el original.

El problema evidente es que esas lecturas hablan del «modelo 3», de la LSTM y de secciones
con otra numeración. **No se corrigen dentro del EDA**: hacerlo sería no entregar el mismo
EDA, que es justo lo que el profesor pidió. La adaptación vive en dos **celdas puente**
colocadas fuera del bloque heredado:

- una **antes** de la Sección 2, que advierte al lector de qué está leyendo, por qué está
  escrito en los términos de la otra entrega, y le ofrece saltar a la Sección 4;
- una **después** de §3.8, con una tabla que retraduce cada hallazgo del EDA a la sección de
  este notebook donde se aplica.

**Alternativas descartadas.**
- *Reescribir las lecturas del EDA para esta entrega* (lo que se hizo en una primera versión
  del esqueleto): más legible, pero incumple la condición literal del profesor. Se revirtió.
- *Un corpus nuevo*: habría obligado a rehacer el EDA desde cero, gastando en exploración el
  presupuesto que esta entrega necesita para la arquitectura, y habría impedido la
  comparación directa de la Sección 13, que es uno de sus aportes.
- *Enlazar al EDA anterior sin copiarlo*: incumple la condición y deja el notebook
  dependiente de otro archivo.

**Consecuencias.** El notebook es largo y sus secciones 2–3 hablan de modelos que aquí no
existen; las celdas puente absorben esa fricción. A cambio, el entregable es autosuficiente y
ganamos una línea base ya medida sobre el mismo split. `requirements.txt` arrastra `spacy` y
su modelo de vectores de ~570 MB únicamente para reproducir §3.7; se documenta en el propio
notebook que es una dependencia del EDA y no del modelo.

---

## D-202 · No replicar las desviaciones del notebook guía respecto al paper

**Estado:** aceptada · 2026-09-12

**Contexto.** El trabajo se basa en `Sesion2/1-transformers-from-scratch.ipynb`. La rúbrica
exige el Transformer **en su forma original**. Al revisar el guía se encontraron tres
desviaciones respecto del paper y varias imprecisiones menores.

**Decisión.** Corregirlas y **documentar la corrección en el notebook**, porque explicar por
qué algo estaba mal es evidencia de comprensión y suma en el criterio de Innovación. Las
tres principales:

1. **Faltan las conexiones residuales.** El guía hace
   `attn_output = self.layer_norm1(attn_output)` y `return self.layer_norm2(ffn_out)`, es
   decir `LayerNorm(Sublayer(x))`. El paper especifica `LayerNorm(x + Sublayer(x))`. Sin el
   atajo residual el gradiente no llega bien a las capas bajas y apilar bloques deja de
   ayudar: es, literalmente, lo que hace *profundo* a un modelo profundo.
   → Se implementan en ambos sublayers, con un parámetro para desactivarlas en la ablación
   de §8 y medir el efecto en lugar de afirmarlo.
2. **El clasificador aplana la secuencia.** El guía usa
   `nn.Flatten()` + `nn.Linear(max_len * emb_dim, 512)`. Con sus valores (2.048 × 256) esa
   sola capa tiene del orden de 10⁸ parámetros —más que todo el resto del modelo— y trata
   cada posición absoluta como una variable distinta.
   → Se usa *pooling* promedio enmascarado. Ver §D-205.
3. **`assert embed_size & num_heads == 0`.** Es un AND de bits, no un módulo. Pasa por
   casualidad con 128 y 8 (ambas potencias de 2), pero no comprueba lo que dice comprobar.
   → `assert emb_dim % num_heads == 0`.

Imprecisiones menores que también se corrigen: en las celdas de prueba el guía pasa
`input_ids` como máscara en vez de `attention_mask`; `LearnablePE` lee `x.size(-1)` (la
dimensión del embedding) creyendo que es la longitud de la secuencia y no mueve el tensor de
posiciones al dispositivo; y el `random_split` se hace sin estratificar y sin semilla.

**Consecuencias.** El notebook no es una copia con datos nuevos: cada componente heredado
está justificado o mejorado. La Sección 8 convierte esas correcciones en resultados medidos
en vez de afirmaciones.

---

## D-203 · Tokenización BPE en lugar del vocabulario por palabras del Miniproyecto 1

**Estado:** revertida · 2026-09-12 → revertida 2026-09-15. Ver **D-214**.

**Contexto (original).** El Miniproyecto 1 usaba un vocabulario de 30.000 palabras completas
con corte por frecuencia. Su EDA §3.7 midió que el 26 % de las palabras **distintas** no
tiene vector preentrenado, y que lo que falta son sobre todo nombres propios (destinos,
hoteles) y variantes ortográficas con y sin tilde.

**Decisión (original, revertida).** Entrenar un tokenizador ByteLevel BPE sobre el corpus,
como hace el notebook de la Sesión 2, y usarlo como preprocesamiento **base** de todo el
notebook.

**Por qué se revirtió.** Dos razones, desarrolladas en D-214:

1. El 26 % es cobertura por **tipos** (palabras distintas). La cobertura que de verdad
   importa —por **apariciones**— ya estaba medida en el propio Miniproyecto 1 y es 98,5 %:
   el vocabulario de 30.000 palabras deja fuera solo el 1,5 % de las palabras que un modelo
   realmente lee en el test. El argumento de D-203 sobreestimaba el problema.
2. Usar BPE como base metía **dos variables a la vez** en la Sección 13 (comparación contra
   los modelos del Miniproyecto 1): tokenización y arquitectura. Eso es precisamente lo que
   el «riesgo abierto» de esta entrada ya advertía y no llegó a resolver.

Se conserva esta entrada por trazabilidad; no se borra una decisión, se corrige.

---

## D-204 · Tasa de aprendizaje 3e-4 con warmup, no 2e-5

**Estado:** aceptada · 2026-09-12

**Contexto.** El notebook guía entrena con `AdamW(lr=2e-5)`. Esa es una tasa de
*fine-tuning*: la apropiada para mover suavemente un modelo ya preentrenado sin destruir lo
que sabe. Aquí el modelo se entrena **desde cero**, con pesos aleatorios.

**Decisión.** `lr = 3e-4` con calentamiento lineal sobre el 10 % de los pasos, siguiendo el
esquema del paper (§5.3), que sube la tasa durante los primeros pasos y luego la baja.

**Por qué importa.** Con `2e-5` y el presupuesto de épocas que cabe en un notebook
reproducible, el modelo apenas se mueve. El riesgo real no es «entrenar un poco peor»: es
obtener un Transformer que no aprendió, concluir que «los transformers no funcionan en esta
tarea» y escribir esa conclusión en el notebook. Es una trampa silenciosa, porque el
entrenamiento no falla —simplemente no converge— y el accuracy que reporta el guía la
esconde: predecir siempre 5★ ya da 65,6 %.

**Consecuencias.** Hay que implementar el scheduler de warmup a mano (unas pocas líneas),
lo cual encaja con el espíritu de la entrega. Se registra la curva de la tasa de aprendizaje
junto a las de pérdida.

---

## D-205 · Pooling promedio enmascarado en lugar de `Flatten`

**Estado:** aceptada · 2026-09-12

**Contexto.** El codificador devuelve un vector por token; la tarea pide uno por reseña.
Hay que agregar. El guía aplana toda la secuencia y la pasa por una capa densa.

**Decisión.** Promedio sobre las posiciones reales, dividiendo por la suma de la máscara y
no por `MAX_LEN`.

**Alternativas descartadas.**
- *`Flatten` + `Linear`* (la del guía): concentra la mayor parte de los parámetros del
  modelo en una sola capa final y ata el modelo a una longitud fija. El mérito de la
  representación dejaría de ser del codificador.
- *Token `[CLS]`*: es lo que hace BERT, pero el paper original no lo define —es una
  aportación posterior— y la rúbrica pide la forma original. Se menciona en el notebook
  como la alternativa estándar hoy.

**Consecuencias.** El modelo es mucho más pequeño y no depende de `MAX_LEN` en el
clasificador. Ambas variantes quedan implementadas tras un parámetro, y §8 mide la
diferencia en lugar de darla por supuesta.

**Nota de continuidad.** Este es el mismo error que el Miniproyecto 1 corrigió en la LSTM
con `pack_padded_sequence` (MP1 §D-002.2), reaparecido en otra forma: dejar que las
posiciones de relleno contaminen la representación final de la secuencia.

---

## D-206 · `MAX_LEN` heredado del Miniproyecto 1, releído bajo un costo distinto

**Estado:** revisada · 2026-09-12 → revisada 2026-09-15 (ver D-214: ya no hay tokens BPE
que recalcular; `MAX_LEN` se hereda igual que el resto de §4).

**Contexto.** El Miniproyecto 1 fijó `MAX_LEN = 150` como percentil 95 en palabras
(MP1 §D-005), con un vocabulario que aquí también se hereda íntegro (D-214). El número no
cambia; lo que cambia es lo que cuesta.

**Decisión.** Se conserva `MAX_LEN = 150` sin recalcular. Lo que se añade en esta entrega no
es un nuevo percentil, sino **medir el costo real de esa longitud bajo atención** en la
Sección 11: en una LSTM el costo crece linealmente con la longitud; en la atención crece con
su **cuadrado**, porque cada token mira a todos los demás.

**Por qué importa.** Fijar 2.048 como hace el guía —sin derivarlo de nada— multiplicaría el
cómputo de la matriz de atención por dos órdenes de magnitud frente al P95 real, casi todo
gastado en padding. La Sección 11 hace visible ese costo con una curva de tiempo y memoria
frente a `MAX_LEN` ∈ {64, 128, 256, 512}, marcando dónde caen tanto el 150 heredado como el
2.048 del guía. `MAX_LEN` sigue siendo la perilla que más pesa en el presupuesto de tiempo;
lo nuevo es que ahora se mide en lugar de solo declararse.

---

## D-207 · Conservar la submuestra, el Split A, el vocabulario y las métricas del
Miniproyecto 1

**Estado:** revisada · 2026-09-12 → ampliada 2026-09-15 (D-214 suma el tokenizador y el
vocabulario a lo que se hereda).

**Contexto.** La Sección 13 compara los modelos de las dos entregas en una sola tabla. Eso
solo es legítimo si se midieron sobre datos comparables. Con la reversión de D-203, «datos
comparables» pasa de «mismo split» a «mismo split **y mismo vocabulario**».

**Decisión.** Se heredan del Miniproyecto 1, tal cual: la submuestra estratificada por
`Polarity` de 40.000 registros con `SEED = 42`, el Split A 80/10/10 estratificado, el
tokenizador por palabras y el vocabulario de 30.000 tokens construido **solo** sobre el
train de ese split, y la función `evaluar(...)` con las mismas seis métricas. Se verifica
imprimiendo la distribución de clases de las tres particiones y comprobando que los
baselines reproducen los valores del Miniproyecto 1 (accuracy 0.6565, macro-F1 0.1585).

**Lo que NO se puede decir es que «la Sección 13 compara una sola variable».** El optimizador
(AdamW con warmup y *label smoothing* en vez del `Adam` plano de MP1), el número de épocas y,
potencialmente, el hardware (D-209) siguen siendo distintos entre las dos entregas. Lo que
esta decisión logra es más modesto y más defendible: **datos, split, vocabulario y métrica
dejan de ser variables**; la receta de entrenamiento y el hardware siguen siéndolo, y se
declaran explícitamente en la tabla de §13 en vez de darse por iguales.

**Consecuencias.** Si la verificación de baselines falla, la Sección 13 queda invalidada y
hay que detenerse antes de seguir. Está escrito como criterio de aceptación en `SPEC.md` §3.3.

---

## D-208 · Implementación a mano, sin `nn.TransformerEncoderLayer`

**Estado:** aceptada · 2026-09-12

**Contexto.** PyTorch trae `nn.MultiheadAttention` y `nn.TransformerEncoderLayer`. Usarlas
daría un modelo mejor optimizado en menos líneas.

**Decisión.** Implementar a mano el producto punto escalado, la separación en cabezas, la
máscara, el bloque con residuales y la normalización. De `transformers`/`tokenizers` solo se
usa el tokenizador BPE; ninguna capa del modelo sale de ahí.

**Por qué.** Dos razones, y la segunda es la que de verdad manda:

1. La rúbrica pide el Transformer «en su forma original», y el sentido de la entrega es
   entender la arquitectura por dentro.
2. **Habilita la Sección 8.** Teniendo las piezas escritas se puede quitar una a la vez y
   medir qué se pierde. Con la capa empaquetada de PyTorch, las ablaciones de residuales,
   de máscara o de pooling no serían posibles sin reescribirla de todos modos.

**Consecuencias.** El modelo es más lento que la versión optimizada de PyTorch (que usa
kernels fusionados). Eso se declara al leer la tabla de tiempos de §13: parte de la
diferencia de costo frente a TF-IDF es de la implementación, no de la arquitectura.
Opcionalmente se puede verificar que la implementación propia coincide numéricamente con
`nn.MultiheadAttention` sobre un tensor de prueba, y documentarlo como verificación.

---

## D-209 · Los tiempos entre entregas solo se comparan si el hardware coincide

**Estado:** aceptada · 2026-09-12

**Contexto.** La Sección 13 incluye una columna de tiempo de entrenamiento para los seis
modelos. Los tres del Miniproyecto 1 se midieron en una corrida local con GPU RTX 3050
Laptop (`MP1 docs/EXPERIMENTS.md`). Esta entrega se ejecuta en una máquina local con GPU
RTX 4060 — es decir, **ya sabemos que el hardware difiere**, no es un riesgo hipotético.
Además, con D-207 ampliada, el optimizador tampoco coincide: MP1 usa `Adam` plano, MP2 usa
`AdamW` con warmup y *label smoothing*. Los tiempos de las dos entregas nunca van a ser
directamente comparables, aunque el hardware coincidiera.

**Decisión.** La tabla de §13 lleva una columna de **entorno de medición** por fila. Si los
entornos difieren, los tiempos heredados se marcan como provenientes de otro hardware y se
interpretan como orden de magnitud, no como medición comparable.

**Mitigación preferible.** Reentrenar en este notebook al menos el modelo más barato del
Miniproyecto 1 (TF-IDF + LogReg, ~27 s) para tener un punto de calibración del hardware.
Con eso la comparación de tiempos deja de ser cualitativa.

**Consecuencias.** Una columna más en la tabla y un párrafo de advertencia. Es preferible a
publicar una comparación de costos que no se sostiene; el análisis costo/beneficio es la
mitad de la pregunta de investigación de `SPEC.md` §1.1.

---

## D-210 · Estructura del repositorio y nombre

**Estado:** aceptada · 2026-09-12

**Decisión.** Repositorio `miniproyecto-2-nlp-Aguado-Cruz-Ramirez`, mismo patrón de nombre
que la entrega anterior cambiando solo el número, privado por defecto. Misma estructura de
carpetas y mismos documentos de especificación que el Miniproyecto 1, para que el curso
completo se lea como una serie coherente.

`consigna.txt` y `rubrica.txt` se versionan para que el entregable sea autoexplicativo. El
notebook con salidas ejecutadas también se versiona, para que el profesor pueda leer
resultados sin ejecutar nada.

---

## D-211 · Incluir variantes posteriores al paper (§9)

**Estado:** aceptada · 2026-09-12

**Contexto.** Las Secciones 6 y 8 se ciñen a la arquitectura de 2017, que es lo que exige el
criterio 3 de la rúbrica. Pero la consigna dice que «se valora significativamente abordar
casos que impliquen incluir técnicas más allá de las vistas en clase», y ese criterio
(Innovación) vale otros 2 puntos. Ciñéndonos solo al paper no habría nada más allá del guía.

**Decisión.** Añadir una sección con tres modificaciones posteriores, implementadas sobre
la misma base: **Pre-LN**, token **`[CLS]`** y ***label smoothing***. (RoPE se evaluó y se
descartó por presupuesto — ver D-216, que la reemplaza por algo que ataca H1 más
directamente y cuesta menos.)

**Por qué estas tres.** No son una lista de novedades: cada una ataca un problema concreto
de la implementación original que el propio notebook expone.

- *Pre-LN* responde a la inestabilidad de gradiente que hace casi obligatorio el warmup de
  §D-204.
- *`[CLS]`* responde a la limitación del promedio de §D-205: diluye una queja breve dentro
  de una reseña larga.
- *Label smoothing* está en el paper (§5.4) y el guía lo omite; sobre una escala ordinal con
  fronteras difusas entre 4★ y 5★ debería notarse en MAE y QWK.

**Alternativas descartadas.** *Atención dispersa / Longformer*: resuelven el costo cuadrático
de §11, pero con `MAX_LEN` ≈ 200 el problema no se manifiesta y la comparación no diría nada.
*Fine-tuning de BETO*: prohibido por alcance, es la entrega siguiente del curso.

**Consecuencias.** Tres corridas más de entrenamiento en vez de cuatro.

---

## D-212 · Recuperar la tarea de control `Type` (§12)

**Estado:** aceptada · 2026-09-12

**Contexto.** Si el macro-F1 en polaridad resulta modesto, queda una ambigüedad de fondo:
¿es culpa del modelo, del pipeline, o de la tarea? Sin resolverla, cualquier conclusión sobre
el Transformer es discutible.

**Decisión.** Reutilizar la tarea de control del Miniproyecto 1: la misma arquitectura, sin
cambiar una línea salvo el número de clases, prediciendo `Type` (3 clases, balanceadas, no
ordinal). Es deliberadamente lo contrario de la polaridad en las dos propiedades que la
hacen difícil.

**Por qué se recupera.** En el Miniproyecto 1 fue el argumento más contundente del notebook:
la misma BiLSTM pasaba de macro-F1 0,486 en polaridad a **0,949** en `Type`. Eso cerró la
discusión sobre si el pipeline estaba mal. Aquí cumple la misma función y además da un punto
de comparación directo con la entrega anterior sobre una tarea distinta.

**Consecuencias.** Una corrida más, barata (la arquitectura ya está entrenada y solo cambia
la cabeza). Cubre además el «combinar con otro tipo de casos» de la consigna.

---

## D-213 · Demo dentro del notebook, con pruebas de estrés dirigidas (§15)

**Estado:** aceptada · 2026-09-12

**Contexto.** La consigna valora «significativamente» hacer demos. El Miniproyecto 1 las
dejó explícitamente fuera de alcance. Pero una demo que solo clasifica texto arbitrario es
decorativa: no aporta evidencia sobre ninguna hipótesis.

**Decisión.** Una función `predecir_resena(texto)` que devuelve estrellas, distribución de
probabilidad y pesos de atención, usada para ejecutar **tres pares de prueba dirigidos**:
con y sin negación; el mismo contenido en distinto orden; y una queja breve al final de una
reseña larga. Cada par aísla una propiedad que el MLP, por construcción, no puede tener.

**Alternativas descartadas.** *Gradio o Streamlit*: añaden una dependencia pesada, no
funcionan bien dentro de un `.ipynb` versionado con salidas, y desplazan el foco hacia la
interfaz. *Solo el widget interactivo*: rompería «Restart & Run All» porque su salida depende
de que alguien escriba algo.

**Consecuencias.** Los tres pares se ejecutan con texto fijo y sus salidas quedan guardadas
en el `.ipynb`, de modo que el notebook sigue siendo reproducible sin intervención. El widget
de `ipywidgets` se ofrece como extra opcional. Se declara explícitamente que la evidencia de
estas pruebas es anecdótica y complementa —no sustituye— las métricas de §7.

---

## D-214 · Vocabulario por palabras heredado de MP1 como base; BPE como variante medida

**Estado:** aceptada · 2026-09-15

**Contexto.** D-203 (2026-09-12) decidió usar BPE como preprocesamiento base, siguiendo el
notebook de la Sesión 2. Una auditoría posterior contra la consigna, la rúbrica de MP2 y el
propio EDA del Miniproyecto 1 encontró dos problemas.

**Decisión.** Revertir D-203. La Sección 4 se **copia** del Miniproyecto 1 casi literal:
mismo tokenizador por palabras, mismo vocabulario de 30.000 tokens (`most_common`, construido
solo sobre el train de Split A), misma submuestra de 40k, mismo Split A, mismo `MAX_LEN=150`,
misma función `evaluar(...)`. Lo único que se adapta es el `Dataset` de PyTorch: en MP1
devuelve `(ids, longitudes, y)` para una LSTM/BiLSTM; en MP2 además construye la
`attention_mask` que la atención necesita, a partir de esas mismas longitudes.

BPE no desaparece: pasa a ser una **variante de preprocesamiento medida**, junto a las
ablaciones estructurales de §8, con una sola fila nueva («BPE vs. palabras») en vez de ser
la base de todo el notebook.

**Por qué se revirtió (las dos razones).**

1. **El argumento de cobertura de D-203 estaba mal calibrado.** Citaba que *"el 26 % de las
   palabras distintas no tiene vector preentrenado"* (EDA §3.7, cobertura por **tipos**).
   Pero el propio Miniproyecto 1 ya midió la cifra que de verdad importa para un modelo que
   lee texto: la cobertura por **apariciones**. Su notebook (celda 64, verificación de §4.2)
   imprime: *"tokens fuera del vocabulario… 1,5 %"* sobre el test. El vocabulario de 30.000
   palabras deja fuera solo el 1,5 % de lo que un modelo realmente procesa. El 26 % de tipos
   sin cubrir son sobre todo nombres propios de baja frecuencia (un hotel mencionado dos
   veces); el costo real de no tener BPE es marginal, no el problema serio que D-203 asumía.
2. **BPE como base introducía dos variables a la vez en la Sección 13.** Esa sección compara
   el Transformer contra los modelos del Miniproyecto 1. Con vocabulario propio y distinto,
   cualquier diferencia de macro-F1 podía deberse a la tokenización, a la arquitectura, o a
   una mezcla de ambas, sin forma de separarlas. El «riesgo abierto» que la propia D-203 dejó
   escrito —*"cambiar de tokenización a la vez que de arquitectura introduce dos
   variables"*— nunca llegó a resolverse; solo se mitigaba a medias. Revertir la decisión lo
   resuelve de raíz en vez de mitigarlo.

**Precisión importante (no sobrevender la limpieza del experimento).** Con esta reversión,
datos, split, vocabulario y métrica dejan de ser variables entre las dos entregas. El
optimizador (D-204: AdamW+warmup+label smoothing vs. el `Adam` plano de MP1), el número de
épocas y el hardware (D-209: RTX 4060 vs. RTX 3050) **siguen siendo distintos** y se declaran
como tales en la tabla de §13. No es correcto afirmar que «la Sección 13 compara una sola
variable»; sí lo es afirmar que ya no compara tokenizaciones distintas.

**Alternativas descartadas.**
- *Mantener BPE como base y solo declarar el riesgo en la tabla de §13* (lo que hacía la
  versión anterior de D-203): descartado porque declarar un problema no lo resuelve, y
  resolverlo es prácticamente gratis (reutilizar código ya escrito en el Miniproyecto 1).
- *Entrenar dos Transformers completos, uno con cada tokenizador, como comparación central*:
  descartado por presupuesto — duplicaría el costo de §6 a §12 para una pregunta que una sola
  fila adicional en §8 ya responde.

**Consecuencias.** Se elimina la Sección 4.1/4.2 originales (tokenizador BPE, `MAX_LEN`
recalculado) y se sustituyen por la §4 heredada de MP1 más la adaptación del `Dataset`
(ver D-206, D-207). Aparece una fila nueva en §8: «BPE vs. palabras», que convierte lo que
antes era una decisión asumida en una hipótesis medida — coherente con el resto del diseño
de ablaciones. `requirements.txt` conserva `transformers`/`tokenizers`, ahora solo para esa
fila y no para todo el notebook.

---

## D-215 · El MLP simple usa embeddings promediados con máscara, no TF-IDF

**Estado:** aceptada · 2026-09-15

**Contexto.** La celda de la §5 (Modelo A) quedó redactada de forma ambigua: *"conteos/TF-IDF
sobre el vocabulario… o embeddings promediados con máscara"*. Son dos modelos distintos con
implicaciones distintas, y `SPEC.md` §5 promete que *"la **única** diferencia [con el
Transformer] es la arquitectura"* — promesa que solo una de las dos opciones cumple.

**Decisión.** El MLP usa la **misma tabla de embeddings aprendibles** que alimentaría a un
Transformer sin atención (mismo vocabulario de MP1, mismo `emb_dim`), promediados con la
misma máscara de padding que evita que el relleno contamine el promedio (mismo mecanismo que
§6.4), seguidos de un par de capas densas.

**Por qué, y no TF-IDF.** TF-IDF es una representación dispersa de conteos ponderados por
frecuencia inversa de documento: cambia la *representación* del texto, no solo la
arquitectura del clasificador que la usa. Con TF-IDF, una diferencia de macro-F1 entre el
MLP y el Transformer se explicaría en parte por la representación (dispersa vs. densa
aprendida) y en parte por la arquitectura (sin atención vs. con atención), sin poder
separarlas — el mismo problema que motivó D-214, ahora dentro de la propia Sección 7.

Con embeddings promediados, MLP y Transformer comparten: vocabulario, tokenizador,
tabla de embeddings inicial, `MAX_LEN`, máscara, partición, ponderación de clases,
optimizador y bucle de entrenamiento (§6.5). La única diferencia estructural es que el
Transformer inserta bloques de atención entre los embeddings y el pooling, y el MLP no.
Eso es exactamente lo que H1 necesita para ser una comparación limpia: si el Transformer
gana, la explicación disponible es "la atención", no "una mejor representación de entrada".

**Alternativas descartadas.**
- *TF-IDF + regresión logística*: es el Modelo 1 del Miniproyecto 1 y ya está en la tabla de
  §13 con sus propios números. Repetirlo como "MLP simple" de esta entrega sería confundir
  dos comparaciones distintas bajo el mismo nombre.
- *Bolsa de palabras con conteos crudos + MLP*: mismo problema de fondo que TF-IDF — una
  representación distinta a la del Transformer — sin la ventaja de que TF-IDF al menos
  aporta algo evaluado en MP1.

**Consecuencias.** El MLP de §5 es deliberadamente el punto de comparación más estrecho
posible con el Transformer, no un segundo baseline independiente (ese papel ya lo cumplen
los baselines de clase mayoritaria y azar de §4.5, y los tres modelos de MP1 en §13).

---

## D-216 · El Transformer se entrena con ambos vocabularios; RoPE se descarta

**Estado:** aceptada · 2026-09-15

**Contexto.** D-214 eligió vocabulario por palabras como base para que §13 compare una sola
variable (arquitectura) contra los modelos de MP1. Pero eso puede penalizar al Transformer:
la tokenización subword suele convenirle más que a una LSTM, y si pierde contra MP1 no se
sabría si es la arquitectura o un preprocesamiento subóptimo.

**Decisión.** El Transformer (§6) se entrena **dos veces**: una con el vocabulario por
palabras heredado de MP1 (la que alimenta la tabla de §13, comparación limpia con MP1) y otra
con BPE (su preprocesamiento natural, en §8/§9). Ninguna de las dos tablas se sobrevende como
"la" respuesta; cada una responde una pregunta distinta y se leen juntas.

Para mantener el presupuesto, se retira **RoPE** de las variantes posteriores al paper
(D-211). RoPE era la más cara de implementar y medir; la doble tokenización cubre el mismo
tipo de pregunta ("¿el diseño original le está poniendo el listón bajo al Transformer?") a
menor costo, y sin ella siguen quedando tres variantes propias (Pre-LN, `[CLS]`, label
smoothing) más las ablaciones estructurales — de sobra para el criterio de Innovación.

**Consecuencias.** Una corrida de entrenamiento más para el Transformer (con BPE), a cambio
de una menos (RoPE). El presupuesto total no cambia significativamente.

---

## Plantilla para nuevas entradas

```markdown
## D-2NN · Título breve

**Estado:** propuesta · AAAA-MM-DD

**Contexto.** Qué situación obliga a decidir.

**Decisión.** Qué se hace.

**Alternativas descartadas.** Qué más se consideró y por qué no.

**Consecuencias.** Qué se gana, qué se paga, qué queda abierto.
```
