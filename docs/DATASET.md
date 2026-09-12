# DATASET - Ficha del corpus

> Todas las cifras de este documento fueron verificadas contra la API de estadísticas de
> HuggingFace el 2026-09-05, **antes** de escribir una sola línea del notebook del
> Miniproyecto 1. El EDA de este notebook debe reproducirlas; si no coinciden, o bien hay un
> bug en el EDA o bien el dataset cambió en el Hub, y hay que actualizar esta ficha antes de
> seguir.
>
> **Este corpus se reutiliza del Miniproyecto 1** con autorización explícita del profesor.
> Ver `DECISIONS.md` §D-201. La ficha es la misma; lo que cambia es el uso, en §5.

---

## 1. Identificación

| Campo | Valor |
|---|---|
| Identificador | `vg055/Rest-Mex2025` |
| Fuente | HuggingFace Hub |
| Licencia | CC-BY-4.0 |
| Idioma | Español (variedad mexicana, con anglicismos) |
| Dominio | Reseñas turísticas (estilo TripAdvisor) sobre destinos de México |
| Origen | Corpus del shared task **Rest-Mex 2025**, IberLEF |
| Formato | CSV, un único split `train` |
| Registros | **208,051** |

Carga:

```python
from datasets import load_dataset
ds = load_dataset("vg055/Rest-Mex2025", split="train")
df = ds.to_pandas()
```

## 2. Esquema

| Columna | Tipo | Descripción | Nulos |
|---|---|---|---|
| `Title` | texto | Título corto de la reseña | 2 |
| `Review` | texto | Cuerpo de la reseña | 0 |
| `Polarity` | float 1–5 | Estrellas asignadas por el autor | 0 |
| `Town` | categórica (40) | Pueblo o destino | 0 |
| `Region` | categórica (19) | Estado de México | 0 |
| `Type` | categórica (3) | `Hotel` / `Restaurant` / `Attractive` | 0 |

## 3. Distribuciones verificadas

### 3.1 `Polarity` - desbalance extremo

| Estrellas | n | % |
|---|---:|---:|
| 5 | 136,561 | 65.64% |
| 4 | 45,034 | 21.64% |
| 3 | 15,519 | 7.46% |
| 2 | 5,496 | 2.64% |
| 1 | 5,441 | 2.61% |

Media 4.45. **Baseline de clase mayoritaria: accuracy 0.6564, macro-F1 ≈ 0.158.**

Consecuencia directa: el accuracy no puede ser la métrica de progreso. Ver `SPEC.md` §3.2.

### 3.2 `Type` - razonablemente balanceada

| Tipo | n | % |
|---|---:|---:|
| Restaurant | 86,720 | 41.68% |
| Attractive | 69,921 | 33.61% |
| Hotel | 51,410 | 24.71% |

Este contraste con §3.1 es deliberado y es la razón por la que `Type` entra como tarea de
control en `SPEC.md` §13.

### 3.3 `Region` - sesgo geográfico severo

19 regiones. Las principales:

| Región | n | % |
|---|---:|---:|
| QuintanaRoo | 85,993 | 41.33% |
| Chiapas | 23,532 | 11.31% |
| Estado_de_Mexico | 19,439 | 9.34% |
| Yucatan | 13,678 | 6.57% |
| Jalisco | 11,168 | 5.37% |
| Baja_CaliforniaSur | 10,125 | 4.87% |
| Nayarit | 7,337 | 3.53% |
| Puebla | 6,832 | 3.28% |
| Queretaro | 4,879 | 2.34% |
| Michoacan | 4,454 | 2.14% |
| Guerrero | 4,201 | 2.02% |
| Morelos | 3,445 | 1.66% |

### 3.4 `Town` - 40 destinos, igual de concentrados

Tulum (45,345 · 21.8%), Isla_Mujeres (29,826 · 14.3%), San_Cristobal_de_las_Casas (13,060),
Valladolid (11,637), Bacalar (10,822), Palenque (9,512), Sayulita (7,337),
Valle_de_Bravo (5,959), Teotihuacan (5,810), Loreto (5,525), TodosSantos (4,600),
Patzcuaro (4,454).

Solo Tulum e Isla Mujeres son el **36%** del corpus.

### 3.5 Longitudes en caracteres

| Campo | mín | media | máx |
|---|---:|---:|---:|
| `Review` | 33 | 357.4 | 8,247 |
| `Title` | 1 | 24.9 | 166 |

Distribución de `Review` muy asimétrica: ~96% de las reseñas caben en 855 caracteres; la
cola llega a 8,247. El notebook debe calcular el **percentil 95 en tokens** y derivar de
ahí `MAX_LEN`, en lugar de fijar 256 o 512 por costumbre.

## 4. Sesgos y riesgos conocidos

1. **Sesgo de positividad.** Dos tercios de las reseñas son 5★. Es un sesgo real de las
   plataformas de reseñas (quien reserva ya está predispuesto, y hay incentivo a puntuar
   alto). No es un defecto a corregir, es una propiedad del dominio que el modelo enfrentará
   en producción; se documenta y se maneja con métricas y ponderación, no eliminándolo.
2. **Sesgo geográfico.** El Caribe mexicano domina. Un modelo entrenado con split aleatorio
   puede aprender atajos ligados al destino. Es exactamente lo que mide la Extensión B.
3. **Ruido de etiqueta.** La polaridad la pone el propio autor; hay reseñas con texto
   negativo y 5★ (y viceversa). Esto impone un techo al rendimiento alcanzable y debe
   aparecer en el análisis de errores, no atribuirse al modelo.
4. **Fronteras difusas entre clases contiguas.** La diferencia entre 4★ y 5★ es en buena
   medida idiosincrásica del autor. Es el argumento central a favor del tratamiento ordinal.
5. **Título y cuerpo son campos distintos.** El título suele ser más polar y más corto.
   El notebook debe decidir y justificar si los concatena o usa solo `Review`.

## 5. Uso en este proyecto

- **EDA:** corpus completo, 208,051 registros (207,869 tras eliminar duplicados exactos).
- **Modelado:** submuestra estratificada por `Polarity` de **40,000** registros, `SEED = 42`,
  preservando las proporciones de §3.1. **Idéntica a la del Miniproyecto 1**, que es lo que
  permite comparar los modelos de las dos entregas en una sola tabla. Ver `DECISIONS.md`
  §D-207.
- **Split A (aleatorio):** 80/10/10 estratificado por `Polarity`. Es el único split de esta
  entrega; todas las comparaciones, internas y entre proyectos, se hacen sobre él.
- **Split B (geográfico):** no se usa aquí. Su pregunta —generalización a destinos nunca
  vistos— se respondió en el Miniproyecto 1 y no es la pregunta de este trabajo.

### 5.1 Tokenización

A diferencia del Miniproyecto 1, que construía un vocabulario de 30,000 palabras completas
por frecuencia, aquí se entrena un tokenizador **ByteLevel BPE** sobre el split de
entrenamiento. El motivo sale de §3.7 del propio EDA: el 26 % de los tipos son nombres
propios o variantes ortográficas que un vocabulario por palabras colapsa en `[UNK]`.
Justificación completa en `DECISIONS.md` §D-203.

Consecuencia práctica: **`MAX_LEN` no se hereda**. El percentil 95 de 150 unidades del
Miniproyecto 1 está medido en palabras; BPE produce más unidades, así que se recalcula sobre
la tokenización efectiva (`DECISIONS.md` §D-206).

## 6. Cita

El corpus proviene del foro de evaluación IberLEF, tarea Rest-Mex 2025. Referenciar al
publicarlo conforme a CC-BY-4.0, indicando la fuente en HuggingFace y el shared task.
