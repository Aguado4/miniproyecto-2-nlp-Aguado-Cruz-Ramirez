# PLAN - Ejecución por fases

> Estado global: **Fase 0 completada** (scaffolding y especificación). El notebook tiene sus
> 112 celdas estructuradas: las secciones 2–3 heredadas intactas del Miniproyecto 1, y las
> secciones 4–17 con su narrativa escrita y sus celdas de código por implementar.
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
- [x] Esqueleto de las secciones 4–17 con su narrativa
- [ ] `git init`, primer commit, repositorio remoto
      (`miniproyecto-2-nlp-Aguado-Cruz-Ramirez`, privado)

---

## Fase 1 - Entorno, corpus y EDA (SPEC §0–3)

El EDA viene heredado, pero hay que ejecutarlo, no darlo por bueno.

- [ ] Ejecutar las secciones 1–3 completas y comprobar que corren sin errores
- [ ] Verificar que las cifras coinciden con `DATASET.md` §3 (si no coinciden, el dataset
      cambió: actualizar `DATASET.md` antes de seguir)
- [ ] Comprobar que `CFG` imprime la configuración activa y que `CFG_ABL` está definida
- [ ] Revisar que las dos celdas puente digan la verdad sobre lo que viene y lo que vino

**Criterio de salida:** las secciones 1–3 corren limpias y el bloque heredado sigue siendo
idéntico al del Miniproyecto 1 (volver a correr la verificación si se tocó algo).

---

## Fase 2 - Preprocesamiento y protocolo (SPEC §4)

La fase que hace posible todo lo demás: si el split no reproduce el del Miniproyecto 1, la
Sección 13 no se puede escribir.

- [ ] 4.1 Tokenizador BPE entrenado **solo sobre train**, con tasa de `[UNK]` e ida y vuelta
- [ ] 4.2 `MAX_LEN` = P95 sobre tokens BPE + tasa de truncamiento
- [ ] 4.3 Submuestra 40k y Split A idénticos al Miniproyecto 1, con verificación impresa
- [ ] 4.4 `Dataset` y `DataLoader` (`input_ids`, `attention_mask`, `y`)
- [ ] 4.5 Función `evaluar(...)` única + baselines

**Criterio de salida (bloqueante):** los baselines reproducen los del Miniproyecto 1
(accuracy 0.6565, macro-F1 0.1585). Si no, parar y corregir el split antes de modelar.

---

## Fase 3 - Los dos modelos (SPEC §5–6)

Cada uno se cierra con sus métricas en `EXPERIMENTS.md` antes de pasar al siguiente.

- [ ] Modelo A - MLP simple
- [ ] 6.1 Token + positional embeddings sinusoidales (escalado √d_model + visualización)
- [ ] 6.2 Multi-head self-attention propia, con máscara y `return_attention`
- [ ] 6.3 Bloque codificador con residuales (parámetros `residual` y `norma`)
- [ ] 6.4 Clasificador con pooling enmascarado (parámetro `pooling`)
- [ ] 6.5 Bucle compartido: warmup, label smoothing, early stopping por macro-F1 de validación
- [ ] Modelo B entrenado, cronometrado y evaluado

**Criterio de salida:** dos filas en la tabla, mismo test, mismas métricas, mismo bucle.
Verificar que no aparece `nn.MultiheadAttention` ni `nn.TransformerEncoderLayer` en ninguna
celda.

**Riesgo principal: que el Transformer no converja.** Síntoma: macro-F1 estancado cerca de
0,16 (el del baseline mayoritario). Causas probables, en orden: tasa de aprendizaje demasiado
baja (`DECISIONS.md` §D-204), falta de warmup, ausencia de escalado √d_model, o máscara mal
aplicada. **No** concluir «los transformers no sirven para esta tarea» sin descartar las
cuatro — y la §10 existe precisamente para dejar esa comprobación por escrito.

---

## Fase 4 - Comparación principal (SPEC §7)

- [ ] Tabla MLP vs. Transformer con las siete columnas
- [ ] F1 por clase y Δ por clase (contraste de H1)
- [ ] Matrices de confusión lado a lado
- [ ] Ejemplos con negación que el MLP falla y el Transformer acierta

**Criterio de salida:** está escrita la respuesta a «¿compensa la atención su costo?», con
números y no con expectativas.

---

## Fase 5 - Estudios sobre la arquitectura (SPEC §8–§11)

El grueso del aporte propio, y la fase con mayor riesgo de presupuesto. Todas las corridas
usan `CFG_ABL` y **fijan la semilla antes de empezar**.

**§8 · Ablaciones estructurales**
- [ ] Sin conexiones residuales
- [ ] `Flatten` vs. pooling enmascarado
- [ ] Pooling sin máscara
- [ ] Sin label smoothing
- [ ] **Sin señal posicional** (control de H2) — *no recortable*

**§9 · Más allá del paper**
- [ ] Pre-LN
- [ ] Token `[CLS]`
- [ ] Label smoothing aislado (efecto en MAE y QWK)
- [ ] RoPE — *la más ambiciosa; si no entra, declararla como trabajo pendiente*

**§10 · Barrido de hiperparámetros**
- [ ] Tasa de aprendizaje (incluyendo el `2e-5` del guía) — *no recortable*
- [ ] Número de bloques · número de cabezas · dropout

**§11 · Costo cuadrático**
- [ ] `MAX_LEN` ∈ {64, 128, 256, 512}: tiempo/época, memoria pico, macro-F1
- [ ] Gráfica de doble eje con el P95 y el 2.048 del guía marcados

**Criterio de salida:** se declara explícitamente que estas filas comparan entre sí y no
contra la tabla de §7. Toda variante que no mejore se reporta igual, con su explicación.

---

## Fase 6 - Control, comparación entre entregas y demo (SPEC §12–§15)

- [ ] §12 Tarea de control `Type`, comparada contra polaridad y contra el 0,949 del MP1
- [ ] §13 Tabla global con los modelos de las dos entregas y su entorno de medición
- [ ] §13 Gráfica costo vs. beneficio
- [ ] §13 (Recomendado) Reentrenar TF-IDF aquí para calibrar los tiempos
- [ ] §14 Mapa de calor de atención, texto coloreado, comparación entre cabezas
- [ ] §14 Contraste agregado contra el léxico distintivo del EDA §3.5
- [ ] §15 `predecir_resena(...)` + los tres pares de prueba con texto fijo
- [ ] §15 Widget de `ipywidgets` como extra opcional

**Criterio de salida:** la demo corre sin intervención manual (los pares son texto fijo).

---

## Fase 7 - Errores, cierre y entrega (SPEC §16–§17)

- [ ] §16 Errores por distancia en estrellas
- [ ] §16 Casos donde fallan todos los modelos, agrupados por causa
- [ ] §17 Conclusiones: veredicto sobre H1 y H2, qué pieza resultó imprescindible,
      qué dijo la tarea de control, limitaciones y trabajo futuro
- [ ] «Restart & Run All» limpio, cronometrado
- [ ] Repasar el checklist de `SPEC.md` §5 punto por punto
- [ ] Repasar `rubrica.txt` contra la tabla de cobertura de `SPEC.md` §6
- [ ] Re-verificar que el bloque EDA sigue idéntico al del Miniproyecto 1
- [ ] `EXPERIMENTS.md` con los números de la corrida final
- [ ] `README.md` actualizado con los resultados principales
- [ ] Quitar `TODO`, celdas vacías, código muerto y comentarios de andamiaje
- [ ] Notebook con salidas guardadas, commit y push

---

## Presupuesto de tiempo (GPU T4)

| Fase | Ejecución estimada |
|---|---|
| Carga + EDA (heredado) | 3–4 min |
| Tokenizador BPE + preprocesamiento | 2–3 min |
| Modelo A (MLP) | 1–2 min |
| Modelo B (Transformer, 2 bloques) | 5–7 min |
| §8 Ablaciones (5 corridas reducidas) | 6–8 min |
| §9 Variantes posteriores (4 corridas) | 5–7 min |
| §10 Barrido (≈9 corridas cortas) | 6–9 min |
| §11 Costo cuadrático (4 corridas) | 4–6 min |
| §12 Tarea de control | 2 min |
| §14–§16 Interpretabilidad, demo y errores | 3–4 min |
| **Total** | **~38–52 min** |

**El presupuesto está ajustado y probablemente se pase de los 30 min.** Es el precio de las
secciones que dan los puntos de Innovación. Orden de recorte, si hace falta:

1. Valores intermedios de §10 (dejar los extremos, que es donde está la señal).
2. `MAX_LEN` = 512 en §11 (es el más caro y la tendencia ya se ve con tres puntos).
3. RoPE en §9 — declararla como trabajo pendiente, no borrarla en silencio.
4. Épocas del modelo principal.

**Nunca recortar:** el EDA, la ablación sin posicional (control de H2), la fila de
`lr = 2e-5` en §10, ni el análisis de errores.

**La perilla que más pesa es `MAX_LEN`**, porque el costo de la atención es cuadrático en la
longitud. Antes de recortar épocas, comprobar que salió del P95 real y no de un número
heredado. La §11 mide exactamente eso: si el tiempo se dispara, ahí está la explicación.
