
# LLM-Agents-TFM

Trabajo de Fin de Máster: **Agentes basados en LLM con arquitectura RAG** para generación y corrección de exámenes.  
El proyecto usa [Ollama](https://ollama.com/) junto con `llama-index`, `chromadb` y scripts en Python.

---

## 🚀 Requisitos

- **Python 3.10+**
- Instalar dependencias:

  ```bash
  pip install -r requirements.txt
  ```

(requisitos principales: `pandas`, `numpy`, `requests`, `llama-index`, `chromadb`, `rapidfuzz`, `openpyxl`)

* **Ollama instalado** y modelos descargados:

  ```bash
  ollama pull llama3
  ollama pull llama3:8b
  ollama pull mistral
  ollama pull deepseek-r1
  ollama pull nomic-embed-text
  ollama pull all-minilm
  ollama pull mxbai-embed-large
  ```

---

## 🐳 Ejecución con Docker

Incluye `docker-compose.yml` para levantar el entorno con soporte GPU:

```bash
docker compose up -d
```

Para pararlo:

```bash
docker compose down
```

👉 Si no tienes GPU, elimina la sección `deploy.resources.devices` del `docker-compose.yml`.

---

## 📂 Dataset: Simpsons Multilingual

Corpus de prueba en tres idiomas:

* `simpsons_data_es`: 13 episodios en **español**
* `simpsons_data_en`: 13 episodios en **inglés**
* `simpsons_data_ca`: 13 episodios en **catalán**

Formato de cada archivo:

* **ES**: Título, Temporada, Episodio, Sinopsis, Personajes destacados
* **EN**: Title, Season, Episode, Synopsis, Main characters
* **CA**: Títol, Temporada, Episodi, Sinopsi, Personatges destacats

**Sugerencias**:

* Indexar cada idioma en colecciones distintas de ChromaDB para comparar *recall/precision*.
* Usar archivos homogéneos (ej. `s01eXX.txt`) para análisis cruzado.

---

## ⚙️ Scripts principales

### 1. `simpsons_script.py`

Genera el dataset en la carpeta `simpsons_data` (solo ES, el resto fue traducido con ChatGPT). Los documentos quedan en `/data`.

```bash
python simpsons_script.py
```

---

### 2. `build_index.py`

Crea la base de datos e índice en **ChromaDB** a partir del dataset.
Es un archivo de prueba: en `benchmark_rag_tokenized.py` se reutiliza esta funcionalidad dentro del propio código.

```bash
python build_index.py --data_dir simpsons_data_es --collection simpsons_es
```

---

### 3. `benchmark_rag_tokenized.py`

Ejecuta benchmarks de RAG con recuperación + generación.
Mide latencias, tokens y métricas de groundedness.

Ejemplo de uso:

```bash
python benchmark_rag_tokenized.py \
  --data_dir simpsons_data_es \
  --gold gold_test_es.csv \
  --llm llama3 \
  --embed nomic-embed-text \
  --top_k 3 \
  --lang es \
  --output results_es.csv
```

**Parámetros destacados:**

* `--lang` → idioma (`es`, `en`, `ca`)
* `--llm` → modelo LLM en Ollama
* `--embed` → modelo de embeddings
* `--top_k` → nº de chunks recuperados
* `--use_chroma` → activa Chroma como vector store
* `--groundedness semantic|lexical` → cómo medir groundedness (*semantic* por defecto)

---

### 4. `aggregate_with_audit.py`

Unifica múltiples resultados (CSV/JSONL) en un único archivo y genera auditoría.
De esta manera se asegura que los archivos contienen la cantidad de filas correcta.

```bash
python aggregate_with_audit.py \
  --input-dir bench_out \
  --output bench_out/aggregate.csv \
  --audit bench_out/_audit.csv
```

**Opcional:**

* `--schema-json` o `--schema-csv` para fijar un esquema de columnas canónico.
* `--parquet` para exportar también en formato Parquet (no se ha llegado a utilizar).

---

### 5. `analyze_schema_bench.py`

Analiza el `aggregate.csv` y produce métricas textuales y cuantitativas.

```bash
python analyze_schema_bench.py \
  --in bench_out/aggregate.csv \
  --outdir reports
```

**Genera:**

* Métricas textuales: **EM, F1, ROUGE-L, Levenshtein**
* Latencias y eficiencia (`tokens_per_sec`)
* Correlaciones y leaderboard de mejores combinaciones (en este caso no se explotó mucho)
* Excel resumen (`reports/summary.xlsx`)
* Tablas por idioma, LLM, embedding, top\_k y preguntas

---

## 📊 Flujo de trabajo

1. Generar dataset → `simpsons_script.py`
2. Indexar documentos → `build_index.py`
3. Ejecutar benchmark → `benchmark_rag_tokenized.py`
4. Agregar resultados → `aggregate_with_audit.py`
5. Analizar métricas → `analyze_schema_bench.py`

---

## 📎 Notas

* Los resultados incluyen métricas de groundedness y soporte de recuperación (`groundedness_pred_0_1`, `retrieval_support_gold_0_1`).
* Para más fidelidad, se recomienda usar `--groundedness semantic` (requiere embeddings).
* `rapidfuzz` mejora el cálculo de similitud de Levenshtein.
