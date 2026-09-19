# EXPERIMENTS - Bitácora de resultados

> **Regla:** aquí solo se escriben números que efectivamente se ejecutaron.
> Nada de estimaciones ni de valores esperados. Si un experimento no se corrió,
> la fila se queda vacía.

Entorno de la corrida de referencia (Restart & Run All completo, sin errores):

| Campo | Valor |
|---|---|
| Fecha | 2026-09-19 |
| Plataforma | Local (no Colab) |
| GPU | NVIDIA GeForce RTX 3050 6GB Laptop GPU (misma que el Miniproyecto 1) |
| Python / torch / sklearn | 3.14.4 / 2.14.0+cu130 / 1.9.1 |
| Otras versiones | pandas 3.0.6 · spacy 3.8.16 · datasets 5.0.1 |
| Semilla | 42 |
| Tamaño de submuestra | 40,000 (Split A: 32,000 / 4,000 / 4,000), idéntica al Miniproyecto 1 |
| Tokenizador base | Vocabulario por palabras heredado de MP1 (D-214), 30,000 tokens |
| `MAX_LEN` | 150 tokens (heredado de MP1, D-206) |
| Tasa de `[UNK]` / truncamiento en test | 1.5% / 4.2% (reproduce MP1 exacto) |
| Duración total de la corrida | ~61 minutos (13:04-14:05), supera el presupuesto de ~35 min de `PLAN.md` |

---

## 1. Baselines - Split A, test

Reproducen los del Miniproyecto 1 exactamente.

| Baseline | Accuracy | macro-F1 | MAE | QWK |
|---|---:|---:|---:|---:|
| Clase mayoritaria (siempre 5★) | 0.6565 | 0.1585 | 0.549 | 0.0000 |
| Azar estratificado | 0.4798 | 0.1931 | 0.834 | 0.0045 |

Valores de referencia del Miniproyecto 1: accuracy 0.6565, macro-F1 0.1585, MAE 0.549,
QWK 0.0000. **Verificación exitosa**: coinciden hasta el cuarto decimal.

## 2. Modelos principales - Split A, test

| # | Modelo | macro-F1 | Accuracy | MAE | QWK | Params | Tiempo entren. (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| A | MLP simple (embeddings promediados) | 0.3545 | 0.5268 | 0.886 | 0.3501 | 3,857,157 | 21.1 |
| B | Transformer desde cero (vocab. por palabras) | **0.4160** | 0.5858 | 0.605 | 0.5068 | 4,105,605 | 196.6 |

### Desglose de parámetros del Transformer

| Componente | Parámetros | % del total |
|---|---:|---:|
| Embeddings (token + posicional) | 3,840,000 | 93.6% |
| Bloques codificadores | 264,960 | 6.5% |
| Clasificador | 645 | 0.02% |

El embedding domina el conteo de parámetros (vocabulario de 30,000 × 128 dimensiones);
es la misma razón por la que el MLP (que comparte la misma tabla de embeddings, D-215)
tiene un total comparable (3.86M) pese a no tener bloques de atención.

### F1 por clase

| Modelo | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---:|---:|---:|---:|---:|
| MLP simple | 0.252 | 0.121 | 0.314 | 0.376 | 0.710 |
| Transformer desde cero | 0.349 | 0.248 | 0.344 | 0.389 | 0.750 |

**Hipótesis H1** (`SPEC.md` §1.2): el Transformer superará al MLP principalmente en las
clases negativas, donde la negación es el rasgo dominante.
Resultado: mejora relativa mayor en 1★ (+38%) y 2★ (+105%) que en 3★ (+10%), 4★ (+3%) o
5★ (+6%) — la dirección de H1 se confirma en el resultado, aunque no en el mecanismo (ver
§9 más abajo). 362 reseñas con negación en las que el MLP falla y el Transformer acierta
(de 4,000 test). Reseñas de 5★ predichas como 1★/2★: MLP 476, Transformer 182.

## 3. Ablaciones estructurales (§8)

Todas con `CFG_ABL` (5 épocas, 2 bloques, emb_dim 128) y semilla fijada antes de cada
corrida. **Comparan entre sí, no contra la tabla de §2.**

| Variante | macro-F1 | Δ vs. completo | MAE | QWK | Tiempo (s) |
|---|---:|---:|---:|---:|---:|
| Modelo completo (referencia de esta tabla) | 0.3595 | — | 0.869 | 0.353 | 101.6 |
| Sin conexiones residuales (versión del guía) | 0.3873 | +0.0278 | 0.650 | 0.499 | 100.0 |
| `Flatten` en vez de pooling enmascarado | 0.2774 | -0.0821 | 1.355 | 0.145 | 103.0 |
| Pooling sin máscara | 0.3700 | +0.0105 | 0.850 | 0.362 | 102.8 |
| Sin *label smoothing* | 0.4302 | +0.0707 | 0.523 | 0.600 | 101.1 |
| **Sin señal posicional** | 0.3624 | +0.0029 | 0.876 | 0.346 | 102.8 |

**Hipótesis H2** (`SPEC.md` §1.2): sin señal posicional el Transformer debería caer hacia el
nivel del MLP. Si no cae, su ventaja no viene de usar el orden.
Resultado: **no se sostiene en esta corrida**. Sin posicional (0.3624) queda prácticamente
igual al modelo completo (0.3595) y por encima del MLP (0.3545, `A` en la tabla de §2). Una
versión anterior de esta ablación, corrida en otra máquina, sí había mostrado el colapso
esperado — se documenta la discrepancia en `docs/DECISIONS.md` §D-219 como evidencia de
varianza entre corridas, no se oculta ni se promedia.

## 4. Más allá del paper (§9)

Misma configuración reducida que §3 (`CFG_ABL`).

| Variante | macro-F1 | Δ vs. base | MAE | QWK | Tiempo (s) |
|---|---:|---:|---:|---:|---:|
| Base (forma original, Post-LN, pooling promedio = fila "completo" de §3) | 0.3595 | — | 0.869 | 0.353 | 101.6 |
| Pre-LN | 0.2786 | -0.0809 | 1.464 | 0.157 | 104.3 |
| Token `[CLS]` | **0.3963** | **+0.0368** | 0.734 | 0.430 | 105.4 |
| *Label smoothing* 0.1 aislado (ya corrido en §3, se lee aquí) | ver fila "Sin label smoothing" de §3 (0.4302 sin smoothing vs. 0.3595 con) | — | — | — | — |
| BPE vs. palabras (D-216, reemplaza a RoPE) | 0.3608 | +0.0013 | 0.790 | 0.394 | 126.3 |

RoPE se descartó por presupuesto (`D-216`); se reemplazó por la comparación de
tokenización (BPE vs. vocabulario por palabras), que ataca la misma pregunta ("¿el
preprocesamiento le pone el listón bajo al Transformer?") a menor costo.

Resultado: de las tres variantes propiamente arquitectónicas, solo el token `[CLS]` mejora
claramente sobre la forma original. Pre-LN sale peor (posible limitación de esta
implementación, que no añadió la normalización final que ese esquema suele requerir — se
declara como tal, no como veredicto sobre la técnica). BPE da un empate técnico en macro-F1
(0.361 vs. 0.360) pero tarda 24% más (126.3 s vs. 101.6 s) porque genera más unidades por
reseña; vocabulario BPE: 20,000 tokens, `MAX_LEN` (P95) = 178, 0/2,000 reseñas de muestra
con algún `[UNK]`.

## 5. Barrido de hiperparámetros (§10)

Barrido univariado desde la configuración base (`CFG_ABL`, 5 épocas).

| Hiperparámetro | Valor | macro-F1 | Accuracy | Tiempo (s) |
|---|---|---:|---:|---:|
| Tasa de aprendizaje | 2e-5 *(la del guía)* | 0.0681 | 0.0688 | 105.9 |
| | 1e-4 | 0.2645 | 0.3508 | 105.4 |
| | 3e-4 *(base)* | 0.3595 | 0.5018 | 105.3 |
| | 1e-3 | **0.4193** | 0.5785 | 106.2 |
| Nº de bloques | 1 | 0.2888 | 0.4202 | 59.2 |
| | 2 *(base)* | 0.3595 | 0.5018 | 106.1 |
| | 4 | 0.3976 | 0.5565 | 195.4 |
| Nº de cabezas | 1 | 0.3326 | 0.4810 | 47.8 |
| | 4 | 0.3577 | 0.4982 | 74.1 |
| | 8 *(base)* | 0.3595 | 0.5018 | 104.9 |
| Dropout | 0.0 | 0.3752 | 0.5192 | 89.9 |
| | 0.1 *(base)* | 0.3595 | 0.5018 | 106.6 |
| | 0.3 | 0.3573 | 0.4958 | 104.1 |

**Verificación de `DECISIONS.md` §D-204.** Con `lr = 2e-5` el macro-F1 (0.0681) queda por
debajo incluso del baseline mayoritario (0.1585) — peor que no aprender nada. El accuracy
que habría reportado un notebook que solo midiera esa métrica es **6.9%**, muy por debajo
del 65.6% del baseline mayoritario: en esta configuración (5 épocas, pérdida ponderada por
clase) el problema es visible en ambas métricas, no oculto tras un accuracy engañosamente
alto como plantea la advertencia original de D-204 en su forma más literal. Hallazgo
adicional no anticipado: `lr = 1e-3` supera a la base (`3e-4`) en este presupuesto reducido
de 5 épocas — es el mejor resultado de toda la sección.

## 6. Costo cuadrático de la atención (§11)

| `MAX_LEN` | Segundos/época | Memoria pico (MB) | macro-F1 | % truncado |
|---:|---:|---:|---:|---:|
| 64 | 6.9 | 287 | 0.3494 | — |
| 128 | 14.2 | 471 | 0.3586 | — |
| 256 | 40.3 | 1,175 | 0.3505 | — |
| 512 | 136.4 | 3,841 | 0.3595 | — |

Micro-benchmark aislando solo la capa de atención (forward, con calentamiento, batch=4):

| `MAX_LEN` | ms/forward (solo atención) |
|---:|---:|
| 64 | 0.155 |
| 128 | 0.230 |
| 256 | 0.732 |
| 512 | 2.528 |
| 1,024 | 9.540 |
| 2,048 | 36.994 |

P95 elegido en §4.2 (heredado de MP1): `MAX_LEN = 150`. **¿Crecimiento supralineal?** Sí:
pendiente log-log del micro-benchmark de atención = **1.64**; pendiente log-log local del
modelo completo (256→512) = **1.76** (ambas cerca del 2.0 teórico). **¿Dónde satura el
macro-F1?** Ya en `MAX_LEN=128` (0.359); las diferencias entre 128 y 512 (0.351-0.360) caben
en el ruido de una sola corrida. **Costo estimado del `MAX_LEN=2048` del guía:**
extrapolando con la pendiente local (1.76), ~1,562 s/época (~26 minutos/época) — más de 4
horas para las 10 épocas de la configuración completa, frente a los ~35 minutos de
presupuesto de todo el notebook.

## 7. Tarea de control: `Type` (§12)

| Modelo | Tarea | macro-F1 | Accuracy |
|---|---|---:|---:|
| Transformer desde cero | Polaridad (5 clases, ordinal, desbalanceada) | 0.4160 | 0.5858 |
| Transformer desde cero | `Type` (3 clases, balanceada) | **0.9200** | 0.9233 |
| BiLSTM + spaCy *(Miniproyecto 1)* | `Type` | 0.9489 | 0.9510 |

La diferencia entre ambos macro-F1 del Transformer mide cuánta dificultad pertenece a la
tarea y no al modelo. Resultado: **+0.5040**. Además, en `Type` el Transformer (0.920) casi
iguala a la BiLSTM+spaCy de MP1 (0.949, -0.029), mientras que en polaridad la brecha con esa
misma BiLSTM es mucho mayor (0.486 vs. 0.416, -0.070) y con TF-IDF también (0.524 vs. 0.416,
-0.108): la brecha del Transformer con el resto de modelos se abre específicamente en la
tarea difícil.

## 8. Comparación con el Miniproyecto 1 - Split A, test

Los cuatro primeros modelos provienen de `docs/EXPERIMENTS.md` §2 del Miniproyecto 1
(repositorio `miniproyecto-1-nlp-Aguado-Cruz-Ramirez`), medidos sobre esta misma submuestra
y este mismo split. **La columna de entorno es obligatoria** (ver `DECISIONS.md` §D-209).

| Modelo | Entrega | macro-F1 | Accuracy | MAE | QWK | Params | Tiempo (s) | Entorno |
|---|---|---:|---:|---:|---:|---:|---:|---|
| BiLSTM + atención (Ext. C) | MP1 | **0.5274** | 0.6758 | 0.362 | 0.7546 | 9,441,862 | 88.4 | Local · RTX 3050 Laptop |
| TF-IDF + LogReg | MP1 | 0.5235 | 0.6785 | 0.376 | 0.7261 | 60,000 (features) | 27.0 | Local · RTX 3050 Laptop |
| BiLSTM + spaCy (afinada) | MP1 | 0.4863 | 0.6358 | 0.431 | 0.6942 | 9,441,605 | 38.9 | Local · RTX 3050 Laptop |
| Transformer desde cero | MP2 | 0.4160 | 0.5858 | 0.605 | 0.5068 | 4,105,605 | 196.6 | Local · RTX 3050 Laptop |
| MLP simple | MP2 | 0.3545 | 0.5268 | 0.886 | 0.3501 | 3,857,157 | 21.1 | Local · RTX 3050 Laptop |
| LSTM desde cero | MP1 | 0.2951 | 0.5755 | 0.651 | 0.3552 | 3,972,741 | 50.1 | Local · RTX 3050 Laptop |

**Nota sobre el hardware (D-209, corregida):** esta corrida se ejecutó en la misma GPU que
el Miniproyecto 1 (RTX 3050 Laptop), no en la RTX 4060 planeada originalmente. El
optimizador sigue siendo distinto y declarado (`Adam` plano en MP1 vs. `AdamW` con *warmup*
y *label smoothing* en MP2).

Calibración de hardware (TF-IDF reentrenado en este notebook): **macro-F1 0.5123 en 17.2 s**
(original de MP1: macro-F1 0.5235 en 27.0 s). El macro-F1 es consistente (diferencia de
0.011, plausible por versión de scikit-learn); el tiempo **no** es directamente comparable
ni con la misma GPU, porque TF-IDF + regresión logística corre en CPU, no en GPU — la
calibración confirma coherencia de datos/split, no comparabilidad de tiempos de GPU.

## 9. Interpretabilidad - pesos de atención (§14)

Contraste contra el léxico distintivo del EDA §3.5, sobre una muestra de 2,000 reseñas de
test (top-1 atendido por reseña, último bloque, promedio de cabezas):

| Pregunta | Resultado |
|---|---|
| ¿Los tokens más atendidos coinciden con el léxico distintivo del EDA? (en el MP1, 8 de 20) | **7 de 20** con `LEXICO[5]`; **0 de 20** con `LEXICO[1]` (dominado por el desbalance hacia 5★) |
| ¿El modelo liga los adverbios de negación con el término que modifican? | **No, según este proxy**: atención de la negación a los 3 tokens siguientes = 0.0428, frente a 0.0423 en tokens cualesquiera (razón 1.01×, sin diferencia real) |
| ¿Se especializan las cabezas en cosas distintas? | Entropía media entre 2.85 (cabeza 6, más enfocada) y 3.78 (cabeza 4, más difusa); desplazamiento posicional medio entre 20.7 y 21.4 en las 8 cabezas (todas atienden lejos de la posición propia, ninguna es claramente "local") |

## 10. Demo - pruebas de estrés dirigidas (§15)

| Par de prueba | Predicción A (Transformer) | Predicción B (Transformer) | ¿Cambia? |
|---|---|---|---|
| «la comida estuvo buena» / «la comida no estuvo buena» | 4★ (0.56) | 2★ (0.78) | **Sí, 2 estrellas** |
| «el hotel bien, la comida pésima» / orden invertido | 3★ (0.51 en 3★, 0.34 en 4★) | 3★ (0.43 en 3★, 0.39 en 4★) | No cambia el argmax; sí la distribución |
| Reseña larga con la queja al final / sin ella | 5★ (0.39) | 5★ (0.34, sube 4★ a 0.14) | No cambia el argmax; sí la distribución |

Mismo ejercicio con el MLP, para contrastar: par 1 también cambia (3★→2★, un salto de una
estrella frente a dos del Transformer); par 2 da la **misma** distribución exacta en ambos
órdenes (3★, 67%) — invariante al orden por construcción, como se esperaba.

Evidencia anecdótica; complementa las métricas de §2 y §9, no las sustituye.

## 11. Análisis de errores (§16)

| Métrica | Valor |
|---|---|
| Reseñas donde fallan MLP y Transformer a la vez | 1,300 de 4,000 (32.5%) |
| % de errores a distancia ≥ 2 estrellas (Transformer) | 11.9% |
| % de errores a distancia ≥ 2 estrellas (MLP) | 20.9% |

Causas contables (sobre las 1,300 reseñas donde fallan ambos modelos; no son excluyentes):

| Causa | n | % de los fallos compartidos |
|---|---:|---:|
| Truncamiento (texto real > `MAX_LEN` tokens) | 90 | 6.9% |
| Texto muy corto (≤ 8 tokens) | 0 | 0.0% |
| Mezcla de polaridades (léxico de 1★ y 5★ a la vez) — *ver limitación abajo* | 1,166 | 89.7% |
| Error grave (≥ 2 estrellas en algún modelo) | 632 | 48.6% |

**Limitación conocida:** `LEXICO[1]` y `LEXICO[5]` incluyen palabras funcionales de alta
frecuencia (*que, se, le, por* / *es, y, un, muy*), no solo vocabulario claramente polar; el
89.7% de "mezcla de polaridades" probablemente sobreestima la mezcla genuina de elogio y
queja en el mismo texto.

## 12. Veredicto sobre las hipótesis

| Hipótesis | Evidencia | Veredicto |
|---|---|---|
| **H1** — el Transformer gana sobre todo en clases negativas, por la negación | §2 (F1 por clase: +38% en 1★, +105% en 2★), §9 (362 ejemplos con negación), §10 (demo responde más a la negación que el MLP) | **Parcial**: se confirma en el resultado, no en el mecanismo (§9 no encuentra que la negación atienda especialmente hacia adelante) |
| **H2** — sin señal posicional cae hacia el nivel del MLP | §3 (última fila: 0.3624 sin posicional vs. 0.3595 completo vs. 0.3545 MLP) | **No se sostiene** en esta corrida (ver `DECISIONS.md` §D-219 sobre varianza entre corridas) |

## 13. Notas de la corrida

- Corrida completa ("Restart & Run All") sin errores, notebook `.ipynb` versionado con
  todas las salidas. Se detectaron y corrigieron dos bugs bloqueantes durante esta sesión:
  `pesos_de_clase()` ignoraba `n_clases` (rompía §12, `DECISIONS.md` §D-218), y
  `ClasificadorMLP.forward()` no aceptaba `devolver_atencion` (rompía la demo de §15 al
  contrastar con el MLP).
- La duración real de la corrida completa (~61 minutos) superó el presupuesto de ~35
  minutos estimado en `docs/PLAN.md`; el costo mayor está en las Secciones 8-11 (múltiples
  corridas de ablación, barrido y costo cuadrático), tal como ese documento anticipaba como
  riesgo principal.
- La lectura de la ablación sin señal posicional (§8, control de H2) se reescribió durante
  esta sesión: una versión anterior, basada en una corrida en otra máquina, reportaba el
  veredicto opuesto. Ver `docs/DECISIONS.md` §D-219 para el análisis completo.
- El bloque EDA heredado (Secciones 2-3) se verificó programáticamente contra el notebook
  original del Miniproyecto 1 con `scripts/verificar_eda_mp1.py`: 39 de 42 celdas idénticas
  byte a byte; las 3 restantes difieren solo en la codificación del guión largo, sin cambios
  de código ni de cifras. Ver `docs/DECISIONS.md` §D-217.
