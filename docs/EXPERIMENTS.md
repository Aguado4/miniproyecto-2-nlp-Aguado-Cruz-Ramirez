# EXPERIMENTS - Bitácora de resultados

> **Regla:** aquí solo se escriben números que efectivamente se ejecutaron.
> Nada de estimaciones ni de valores esperados. Si un experimento no se corrió,
> la fila se queda vacía.

> **Estado:** sin ejecutar. Las tablas están con su estructura definida y sus celdas
> pendientes; se llenan a medida que avanzan las fases del `PLAN.md`.

Entorno de la corrida de referencia:

| Campo | Valor |
|---|---|
| Fecha | — |
| Plataforma | — |
| GPU | — |
| Python / torch / transformers | — |
| Semilla | 42 |
| Tamaño de submuestra | 40,000 (Split A: 32,000 / 4,000 / 4,000) |
| Tokenizador | BPE (ByteLevel), vocabulario — |
| `MAX_LEN` (tokens BPE, P95) | — |
| Tasa de truncamiento | — |
| Tasa de `[UNK]` | — |

---

## 1. Baselines - Split A, test

Deben reproducir los del Miniproyecto 1. Si no lo hacen, el split no es el mismo y la
sección 5 de este documento queda invalidada.

| Baseline | Accuracy | macro-F1 | MAE | QWK |
|---|---:|---:|---:|---:|
| Clase mayoritaria (siempre 5★) | | | | |
| Azar estratificado | | | | |

Valores de referencia del Miniproyecto 1: accuracy 0.6565, macro-F1 0.1585, MAE 0.549,
QWK 0.0000.

## 2. Modelos principales - Split A, test

| # | Modelo | macro-F1 | Accuracy | MAE | QWK | Params | Tiempo entren. (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| A | MLP simple | | | | | | |
| B | Transformer desde cero | | | | | | |

### Desglose de parámetros del Transformer

| Componente | Parámetros | % del total |
|---|---:|---:|
| Embeddings (token + posicional) | | |
| Bloques codificadores | | |
| Clasificador | | |

### F1 por clase

| Modelo | 1★ | 2★ | 3★ | 4★ | 5★ |
|---|---:|---:|---:|---:|---:|
| MLP simple | | | | | |
| Transformer desde cero | | | | | |

**Hipótesis H1** (`SPEC.md` §1.2): el Transformer superará al MLP principalmente en las
clases negativas, donde la negación es el rasgo dominante.
Resultado: —

## 3. Ablaciones estructurales (§8)

Todas con `CFG_ABL` y semilla fijada antes de cada corrida. **Comparan entre sí, no contra
la tabla de §2.**

| Variante | macro-F1 | Δ vs. completo | MAE | QWK | Params | Tiempo (s) |
|---|---:|---:|---:|---:|---:|---:|
| Modelo completo (referencia de esta tabla) | | — | | | | |
| Sin conexiones residuales (versión del guía) | | | | | | |
| `Flatten` en vez de pooling enmascarado | | | | | | |
| Pooling sin máscara | | | | | | |
| Sin *label smoothing* | | | | | | |
| **Sin señal posicional** | | | | | | |

**Hipótesis H2** (`SPEC.md` §1.2): sin señal posicional el Transformer debería caer hacia el
nivel del MLP. Si no cae, su ventaja no viene de usar el orden.
Resultado: —

## 4. Más allá del paper (§9)

Misma configuración reducida que §3.

| Variante | macro-F1 | Δ vs. base | MAE | QWK | Tiempo (s) |
|---|---:|---:|---:|---:|---:|
| Base (forma original, Post-LN, pooling promedio) | | — | | | |
| Pre-LN | | | | | |
| Token `[CLS]` | | | | | |
| *Label smoothing* 0.1 (aislado) | | | | | |
| RoPE (posición relativa) | | | | | |

RoPE es la variante que ataca H1 más directamente (posición relativa vs. absoluta).
Si no se implementó por presupuesto, declararlo aquí en vez de omitir la fila: —

## 5. Barrido de hiperparámetros (§10)

Barrido univariado desde la configuración base.

| Hiperparámetro | Valor | macro-F1 | Accuracy | Tiempo (s) |
|---|---|---:|---:|---:|
| Tasa de aprendizaje | 2e-5 *(la del guía)* | | | |
| | 1e-4 | | | |
| | 3e-4 *(base)* | | | |
| | 1e-3 | | | |
| Nº de bloques | 1 / 2 / 4 | | | |
| Nº de cabezas | 1 / 4 / 8 | | | |
| Dropout | 0.0 / 0.1 / 0.3 | | | |

**Verificación de `DECISIONS.md` §D-204.** ¿Se queda el macro-F1 con `lr = 2e-5` cerca de
0,16 (el del baseline mayoritario)? Y en tal caso, ¿qué accuracy habría reportado un
notebook que copiara ese valor y midiera solo accuracy? — 

## 6. Costo cuadrático de la atención (§11)

| `MAX_LEN` | Segundos/época | Memoria pico (MB) | macro-F1 | % truncado |
|---:|---:|---:|---:|---:|
| 64 | | | | |
| 128 | | | | |
| 256 | | | | |
| 512 | | | | |

P95 elegido en §4.2: — · ¿Se observa crecimiento supralineal del tiempo? — ·
¿Dónde satura el macro-F1? — · Costo estimado del `MAX_LEN = 2048` del guía: —

## 7. Tarea de control: `Type` (§12)

| Modelo | Tarea | macro-F1 | Accuracy |
|---|---|---:|---:|
| Transformer desde cero | Polaridad (5 clases, ordinal, desbalanceada) | | |
| Transformer desde cero | `Type` (3 clases, balanceada) | | |
| BiLSTM + spaCy *(Miniproyecto 1)* | `Type` | 0.9489 | 0.9510 |

La diferencia entre ambos macro-F1 del Transformer mide cuánta dificultad pertenece a la
tarea y no al modelo. Resultado: —

## 8. Comparación con el Miniproyecto 1 - Split A, test

Los tres primeros modelos provienen de `../miniproyecto 1/docs/EXPERIMENTS.md` §2, medidos
sobre esta misma submuestra y este mismo split. **La columna de entorno es obligatoria**
(ver `DECISIONS.md` §D-209).

| Modelo | Entrega | macro-F1 | Accuracy | MAE | QWK | Params | Tiempo (s) | Entorno |
|---|---|---:|---:|---:|---:|---:|---:|---|
| TF-IDF + LogReg | MP1 | 0.5235 | 0.6785 | 0.376 | 0.7261 | 60,000 (features) | 27.0 | Local RTX 3050 |
| LSTM desde cero | MP1 | 0.2951 | 0.5755 | 0.651 | 0.3552 | 3,972,741 | 50.1 | Local RTX 3050 |
| BiLSTM + spaCy (afinada) | MP1 | 0.4863 | 0.6358 | 0.431 | 0.6942 | 9,441,605 | 38.9 | Local RTX 3050 |
| BiLSTM + atención (Ext. C) | MP1 | 0.5274 | 0.6758 | 0.362 | 0.7546 | 9,441,862 | 88.4 | Local RTX 3050 |
| MLP simple | MP2 | | | | | | | |
| Transformer desde cero | MP2 | | | | | | | |

Calibración de hardware (TF-IDF reentrenado en este notebook, si se hizo): —

## 9. Interpretabilidad - pesos de atención (§14)

Contraste contra el léxico distintivo del EDA §3.5:

| Pregunta | Resultado |
|---|---|
| ¿Los tokens más atendidos coinciden con el léxico distintivo del EDA? (en el MP1, 8 de 20) | — |
| ¿El modelo liga los adverbios de negación con el término que modifican? | — |
| ¿Se especializan las cabezas en cosas distintas? | — |

## 10. Demo - pruebas de estrés dirigidas (§15)

| Par de prueba | Predicción A | Predicción B | ¿Cambia? |
|---|---|---|---|
| «la comida estuvo buena» / «la comida no estuvo buena» | | | |
| «el hotel bien, la comida pésima» / orden invertido | | | |
| Reseña larga con la queja al final / sin ella | | | |

Mismo ejercicio con el MLP, para contrastar: —

Evidencia anecdótica; complementa las métricas de §2, no las sustituye.

## 11. Análisis de errores (§16)

| Métrica | Valor |
|---|---|
| Reseñas donde fallan todos los modelos | — |
| % de errores a distancia ≥ 2 estrellas (Transformer) | — |
| % de errores a distancia ≥ 2 estrellas (MLP) | — |

Causas observadas:

| Causa | Ejemplo observado |
|---|---|
| | |

## 12. Veredicto sobre las hipótesis

| Hipótesis | Evidencia | Veredicto |
|---|---|---|
| **H1** — el Transformer gana sobre todo en clases negativas, por la negación | §2 (F1 por clase), §9, §10 | — |
| **H2** — sin señal posicional cae hacia el nivel del MLP | §3 (última fila) | — |

## 13. Notas de la corrida

—
