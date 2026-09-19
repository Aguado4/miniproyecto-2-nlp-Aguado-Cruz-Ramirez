# PLAN - Ejecución por fases

> Estado global: **Fases 0-7 completadas.** El notebook tiene 124 celdas, ejecutadas de
> punta a punta sin errores, con resultados reales en `docs/EXPERIMENTS.md`. Durante esta
> sesión se añadieron celdas de código y lectura para completar §10-§17, y se eliminaron 4
> comentarios de andamiaje huérfanos en §5-§9 (`DECISIONS.md` §D-220).
>
> Marca cada tarea al completarla. No avances de fase sin cerrar la anterior:
> el orden está pensado para que cada fase produzca los insumos de la siguiente.

Leyenda: `[ ]` pendiente · `[~]` en curso · `[x]` hecho

---

## Fase 0 - Scaffolding y especificación

- [x] Revisar `consigna.txt` y `rubrica.txt`, extraer restricciones duras
- [x] Revisar `Sesion2/1-transformers-from-scratch.ipynb`, identificar qué heredar y qué corregir
- [x] Confirmar con el profesor la reutilización del corpus y la condición del EDA completo
- [x] `CLAUDE.md`, `SPEC.md`, `DATASET.md`, `DECISIONS.md`, `PLAN.md`, `EXPERIMENTS.md`
- [x] `README.md`, `requirements.txt`, `.gitignore`
- [x] Estructura de carpetas
- [x] Copiar el bloque EDA sin modificar + verificar programáticamente que es idéntico
      (`scripts/verificar_eda_mp1.py`, añadido en esta sesión — ver `DECISIONS.md` §D-217)
- [x] Esqueleto de las secciones 4–17 con su narrativa
- [x] `git init`, primer commit, repositorio remoto
      (`miniproyecto-2-nlp-Aguado-Cruz-Ramirez`, privado)

---

## Fase 1 - Entorno, corpus y EDA (SPEC §0–3)

- [x] Ejecutar las secciones 1–3 completas y comprobar que corren sin errores
- [x] Verificar que las cifras coinciden con `DATASET.md` §3 (coinciden exactamente)
- [x] Comprobar que `CFG` imprime la configuración activa y que `CFG_ABL` está definida
- [x] Revisar que las dos celdas puente digan la verdad sobre lo que viene y lo que vino
      (se corrigió una afirmación desactualizada sobre `MAX_LEN`/BPE en la celda puente
      posterior a §3.8, y se añadió la nota sobre `scripts/verificar_eda_mp1.py`)

**Criterio de salida:** cumplido. El bloque heredado es 39/42 celdas idéntico byte a byte a
MP1 (3 diferencias cosméticas de codificación, documentadas en `DECISIONS.md` §D-217).

---

## Fase 2 - Preprocesamiento y protocolo (SPEC §4)

- [x] 4.1 Tokenizador y vocabulario heredados de MP1 (D-214), con tasa de `[UNK]` e ida y vuelta
- [x] 4.2 Submuestra 40k idéntica al Miniproyecto 1, con verificación impresa
- [x] 4.3 Split A y vocabulario definitivo (30,000 tokens, solo train)
- [x] 4.4 `Dataset` y `DataLoader` (`input_ids`, `attention_mask`, `y`)
- [x] 4.5 Función `evaluar(...)` única + baselines

**Criterio de salida (bloqueante): cumplido.** Los baselines reproducen los del
Miniproyecto 1 exactamente (accuracy 0.6565, macro-F1 0.1585).

---

## Fase 3 - Los dos modelos (SPEC §5–6)

- [x] Modelo A - MLP simple (embeddings promediados, D-215)
- [x] 6.1 Token + positional embeddings sinusoidales (escalado √d_model + visualización)
- [x] 6.2 Multi-head self-attention propia, con máscara y `return_attention`
- [x] 6.3 Bloque codificador con residuales (parámetros `residual` y `norma`)
- [x] 6.4 Clasificador con pooling enmascarado (parámetro `pooling`)
- [x] 6.5 Bucle compartido: warmup, label smoothing, early stopping por macro-F1 de validación
- [x] Modelo B entrenado, cronometrado y evaluado (macro-F1 0.4160)

**Criterio de salida: cumplido.** Verificado: no aparece `nn.MultiheadAttention` ni
`nn.TransformerEncoderLayer` en ninguna celda.

---

## Fase 4 - Comparación principal (SPEC §7)

- [x] Tabla MLP vs. Transformer con las siete columnas
- [x] F1 por clase y Δ por clase (contraste de H1: mejora relativa mayor en 1★/2★)
- [x] Matrices de confusión lado a lado
- [x] Ejemplos con negación que el MLP falla y el Transformer acierta (362 casos)

**Criterio de salida: cumplido.** La atención compensa su costo frente al MLP (gana en las
seis métricas), pero no frente al resto de modelos del curso — eso se responde en la Fase 6.

---

## Fase 5 - Estudios sobre la arquitectura (SPEC §8–§11)

**§8 · Ablaciones estructurales**
- [x] Sin conexiones residuales
- [x] `Flatten` vs. pooling enmascarado
- [x] Pooling sin máscara
- [x] Sin label smoothing
- [x] **Sin señal posicional** (control de H2) — resultado: **H2 no se sostiene** en esta
      corrida (ver `DECISIONS.md` §D-219 sobre la discrepancia con una corrida anterior)

**§9 · Más allá del paper**
- [x] Pre-LN
- [x] Token `[CLS]` — única variante que mejora claramente sobre la forma original
- [x] Label smoothing aislado (efecto en MAE y QWK, leído junto a §8)
- [x] BPE vs. palabras (D-216, reemplaza a RoPE por presupuesto)

**§10 · Barrido de hiperparámetros**
- [x] Tasa de aprendizaje (incluyendo el `2e-5` del guía — confirma D-204, más severo de lo
      anticipado: el accuracy también colapsa, no solo el macro-F1)
- [x] Número de bloques · número de cabezas · dropout

**§11 · Costo cuadrático**
- [x] `MAX_LEN` ∈ {64, 128, 256, 512}: tiempo/época, memoria pico, macro-F1
- [x] Gráfica de doble eje con el P95 y el 2.048 del guía marcados
- [x] Micro-benchmark aislando solo la capa de atención (añadido en esta sesión; el
      borrador original no lo tenía) + extrapolación a `MAX_LEN=2048` (~26 min/época)

**Criterio de salida: cumplido.** Se declaró explícitamente que estas filas comparan entre
sí y no contra la tabla de §7. Toda variante que no mejoró se reportó igual, con su
explicación (incluyendo el resultado inesperado de H2).

---

## Fase 6 - Control, comparación entre entregas y demo (SPEC §12–§15)

- [x] §12 Tarea de control `Type`, comparada contra polaridad (+0.5040) y contra el 0.9489
      del MP1 (diferencia de solo 0.029, mucho menor que en polaridad)
- [x] §13 Tabla global con los modelos de las dos entregas y su entorno de medición
- [x] §13 Gráfica costo vs. beneficio
- [x] §13 Reentrenado TF-IDF aquí para calibrar el hardware (añadido en esta sesión; el
      borrador original no lo tenía) — confirma datos/split coherentes, pero aclara que el
      tiempo de TF-IDF (CPU) no calibra el tiempo de los modelos de PyTorch (GPU)
- [x] §14 Mapa de calor de atención, texto coloreado, comparación entre cabezas
- [x] §14 Contraste agregado contra el léxico distintivo del EDA §3.5 (cuantificación sobre
      2,000 reseñas añadida en esta sesión; el borrador original solo tenía un ejemplo)
- [x] §15 `predecir_resena(...)` + los tres pares de prueba con texto fijo
- [ ] §15 Widget de `ipywidgets` como extra opcional — no implementado, declarado como tal
      en vez de omitido en silencio

**Criterio de salida: cumplido.** La demo corre sin intervención manual (los pares son texto
fijo).

---

## Fase 7 - Errores, cierre y entrega (SPEC §16–§17)

- [x] §16 Errores por distancia en estrellas (Transformer y MLP; el borrador original solo
      tenía el Transformer)
- [x] §16 Casos donde fallan todos los modelos, agrupados por causa, con tabla de causas
      contables (truncamiento, longitud, mezcla de polaridades, error grave — añadida en
      esta sesión; el borrador original solo narraba ejemplos sin contar)
- [x] §17 Conclusiones: veredicto sobre H1 (parcial) y H2 (no se sostiene), qué pieza
      resultó imprescindible (pooling enmascarado), qué dijo la tarea de control,
      limitaciones y trabajo futuro
- [x] «Restart & Run All» limpio, cronometrado (local, GPU RTX 3050, sin errores, ~61 min)
- [x] Repasar el checklist de `SPEC.md` §5 punto por punto
- [x] Repasar `rubrica.txt` contra la tabla de cobertura de `SPEC.md` §6
- [x] Re-verificar que el bloque EDA sigue idéntico al del Miniproyecto 1
- [x] `EXPERIMENTS.md` con los números de la corrida final
- [x] `README.md` actualizado con los resultados principales
- [x] Quitar `TODO`, celdas vacías, código muerto y comentarios de andamiaje
- [x] Notebook con salidas guardadas, commit — **push hecho a la rama
      `mp2-secciones-10-en-adelante`, a la espera de que se confirme el merge a `main`**

---

## Presupuesto de tiempo (GPU T4 estimado vs. real en RTX 3050 Laptop)

| Fase | Estimado | Real (esta corrida) |
|---|---|---|
| Carga + EDA (heredado) | 3–4 min | ~1 min (corpus ya en caché de HuggingFace tras la primera descarga) |
| Tokenizador + preprocesamiento | 2–3 min | < 1 min |
| Modelo A (MLP) | 1–2 min | ~0.4 min |
| Modelo B (Transformer, 2 bloques) | 5–7 min | ~3.3 min |
| §8 Ablaciones (6 corridas reducidas) | 6–8 min | ~10.2 min |
| §9 Variantes posteriores (4 corridas, incl. BPE) | 5–7 min | ~5.7 min |
| §10 Barrido (13 corridas cortas) | 6–9 min | ~19.7 min |
| §11 Costo cuadrático (4 corridas + micro-benchmark) | 4–6 min | ~3.9 min |
| §12 Tarea de control | 2 min | ~1.4 min |
| §13 Calibración TF-IDF | — (no estaba presupuestado) | ~0.3 min |
| §14–§16 Interpretabilidad, demo y errores | 3–4 min | ~1 min |
| **Total medido** | **~38–52 min** | **~61 min** |

**El total real superó la estimación** — principalmente por §10 (13 corridas en vez de las
~9 previstas, al desglosar cada hiperparámetro en filas separadas) y por el tiempo de
arranque de cada corrida de entrenamiento (carga de `DataLoader`, warmup). Se documenta en
`docs/EXPERIMENTS.md` §13 y en la Sección 17 del notebook como limitación honesta en vez de
ajustar el número a la baja.

**Nunca se recortó:** el EDA, la ablación sin posicional (control de H2), la fila de
`lr = 2e-5` en §10, ni el análisis de errores — tal como este documento lo pedía.
