# -*- coding: utf-8 -*-
"""
Benchmark RAG (Ollama + LlamaIndex)
-----------------------------------
- Recuperación con LlamaIndex (top-k)
- Generación directa con Ollama (/api/generate) para capturar métricas:
  prompt_tokens (prompt_eval_count), completion_tokens (eval_count),
  total_tokens y tokens_per_sec (eval_count / eval_duration)
- Latencias separadas: retrieval / generación / total
- Gold test con LLM evaluador (parseo robusto del JSON)
- Heurísticas ligeras: factual_consistency_0_10, hallucination_rate_0_1
- Lectura robusta del CSV (UTF-8/UTF-8-BOM/CP1252 y delimitador ,/;)

Ejemplos:
  python benchmark_rag.py --data_dir simpsons_data --gold gold_examples_simpsons.csv ^
    --llm llama3 --embed nomic-embed-text --top_k 3 --lang es --output results.csv
"""

import os
import csv
import time
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# LlamaIndex
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Chroma (opcional)
try:
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# --- NUEVOS defaults (arriba, junto a imports) ---
DEFAULT_BASE_DIR = "./data"
DEFAULT_GOLD_PREFIX = "gold_test_"
DEFAULT_GOLD_EXT = ".csv"
DEFAULT_COLLECTION_PREFIX = "simpsons_data_"
AUTO = "AUTO"



# ---------------------------- Dataclasses -------------------------------------

@dataclass
class QuantMetrics:
    latency_total_ms: float
    latency_retrieval_ms: float
    latency_generation_ms: float
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    tokens_per_sec: Optional[float] = None


@dataclass
class QualMetrics:
    gold_score_0_10: Optional[float]
    gold_feedback: Optional[str]
    factual_consistency_0_10: Optional[float] = None
    answer_relevance_0_10: Optional[float] = None
    hallucination_rate_0_1: Optional[float] = None


# --------------------- Utilidades: Ollama ensure models -----------------------

def ensure_ollama_models(models: List[str]) -> None:
    """Verifica (y si hace falta descarga) modelos en Ollama."""
    import requests
    url = "http://localhost:11434/api/pull"
    headers = {"Content-Type": "application/json"}
    for name in models:
        if not name:
            continue
        try:
            payload = {"name": name}
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    if data.get("status") == "success":
                        break
        except Exception:
            # Si falla la comprobación, seguimos; el error real aparecerá al usar el modelo.
            pass


# ----------------------------- CSV robusto ------------------------------------

def read_gold_robust(path: str) -> List[Dict[str, str]]:
    """
    Lee el CSV del gold set detectando separador y tolerando UTF-8, UTF-8 BOM y CP1252.
    Requiere cabeceras: question,gold_answer[,language]
    """
    encodings_to_try = ["utf-8-sig", "utf-8", "cp1252"]
    last_err = None
    for enc in encodings_to_try:
        try:
            with open(path, "r", encoding=enc, newline="") as f:
                sample = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t"])
                except csv.Error:
                    dialect = csv.get_dialect("excel")
                reader = csv.DictReader(f, dialect=dialect)
                items = list(reader)
                headers = [h.strip().lower() for h in (reader.fieldnames or [])]
                required = {"question", "gold_answer"}
                if not required.issubset(set(headers)):
                    raise ValueError(
                        f"Cabeceras detectadas: {reader.fieldnames}. "
                        "Se requieren columnas: question,gold_answer[,language]"
                    )
                return items
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"No se pudo leer el CSV '{path}': {last_err}")


# ------------------------------ Core RAG --------------------------------------

def build_index(
    data_dir: str,
    embed_model_name: str,
    use_chroma: bool = False,
    chroma_path: str = "./chroma_db",
    collection_name: str = "rag_collection",
    llm_name: str = "llama3",
):
    """Construye (o conecta) el índice con embeddings de Ollama. Soporta Chroma."""
    Settings.llm = Ollama(model=llm_name)
    Settings.embed_model = OllamaEmbedding(model_name=embed_model_name)

    if use_chroma:
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb no está instalado. Ejecuta: pip install chromadb")
        os.makedirs(chroma_path, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=chroma_path)
        chroma_collection = chroma_client.get_or_create_collection(collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        if chroma_collection.count() == 0:
            docs = SimpleDirectoryReader(data_dir).load_data()
            index = VectorStoreIndex.from_documents(docs, vector_store=vector_store)
        else:
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    else:
        docs = SimpleDirectoryReader(data_dir).load_data()
        index = VectorStoreIndex.from_documents(docs)

    return index


# --------- Construcción de prompt y generación con tokens vía Ollama -----------

def _oneliner(s: str) -> str:
    import re
    return re.sub(r"\s+", " ", s or "").strip()

def _strip_reasoning(text: str) -> str:
    """Elimina bloques de razonamiento comunes (DeepSeek-R1 y similares)."""
    import re
    if not text:
        return text
    # 1) DeepSeek-R1: <think> ... </think>
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 2) Bloques markdown con etiquetas típicas
    text = re.sub(r"```(?:reasoning|thought|inner_monologue).*?```", "", text, flags=re.DOTALL | re.IGNORECASE)
    # 3) Líneas prefijadas
    text = re.sub(r"(?im)^\s*(razonamiento|reasoning)\s*:\s*.*$", "", text)
    return text.strip()

def build_prompt(query: str, nodes, language: str = "es") -> str:
    # ✅ Buenas prácticas: pedimos UNA sola frase para reducir ruido léxico.
    sys_instructions = {
        "es": ("Eres un asistente experto. Responde SÓLO con la información del CONTEXTO.\n"
               "Si no está en el contexto, indícalo explícitamente.\n"
               "Devuelve UNA sola frase (máx. 25 palabras), concisa y exacta.\n"
               "NO muestres razonamiento ni pasos intermedios.\n"),
        "en": ("You are an expert assistant. Answer ONLY using the CONTEXT.\n"
               "If it's not in the context, say it explicitly.\n"
               "Return ONE single sentence (max 25 words), concise and precise.\n"
               "Do NOT show chain-of-thought.\n"),
        "ca": ("Ets un assistent expert. Respon NOMÉS amb la informació del CONTEXT.\n"
               "Si no és al context, digues-ho explícitament.\n"
               "Retorna UNA sola frase (màx. 25 paraules), concisa i precisa.\n"
               "No mostris el raonament.\n"),
    }
    context_chunks = []
    for i, n in enumerate(nodes, start=1):
        src = n.metadata.get("file_name", f"chunk_{i}")
        context_chunks.append(f"[Fuente: {src}]\n{n.node.get_text()}")
    context_text = "\n\n".join(context_chunks)

    prompt = (
        f"{sys_instructions.get(language, sys_instructions['es'])}\n"
        f"--- CONTEXTO ---\n{context_text}\n"
        f"--- PREGUNTA ---\n{query}\n"
        f"--- RESPUESTA ---\n"
    )
    return prompt


def ollama_generate_with_metrics(model: str, prompt: str, timeout: int = 600):
    import requests
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    t0 = time.perf_counter()
    r = requests.post(url, json=payload, timeout=timeout)
    t1 = time.perf_counter()
    if not r.ok:
        raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
    
    data = r.json()
    text_raw = (data.get("response") or "").strip()
    # Limpiar razonamiento antes de devolver la respuesta (para CSV y Gold test)
    text = _oneliner(_strip_reasoning(text_raw))

    # Métricas proporcionadas por Ollama
    prompt_tokens = data.get("prompt_eval_count")
    completion_tokens = data.get("eval_count")
    total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    eval_duration_ns = data.get("eval_duration") or 0
    gen_secs = eval_duration_ns / 1e9 if eval_duration_ns else (t1 - t0)
    tokens_per_sec = (completion_tokens or 0) / gen_secs if gen_secs > 0 else None

    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_sec": tokens_per_sec,
        "eval_duration_ns": eval_duration_ns,
        "wall_clock_s": (t1 - t0),
    }


def time_retrieval_and_generation(index, query: str, top_k: int = 3, llm_name: str = "llama3", language: str = "es"):
    """Recupera con LlamaIndex, construye el prompt y genera con Ollama (tokens)."""
    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)

    t0 = time.perf_counter()
    # retrieval
    t_r0 = time.perf_counter()
    nodes = retriever.retrieve(query)
    t_r1 = time.perf_counter()

    # build prompt + generación con Ollama
    prompt = build_prompt(query, nodes, language=language)
    t_g0 = time.perf_counter()
    gen = ollama_generate_with_metrics(model=llm_name, prompt=prompt)
    t_g1 = time.perf_counter()
    t1 = time.perf_counter()

    q = QuantMetrics(
        latency_total_ms=(t1 - t0) * 1000.0,
        latency_retrieval_ms=(t_r1 - t_r0) * 1000.0,
        latency_generation_ms=(t_g1 - t_g0) * 1000.0,
        prompt_tokens=gen["prompt_tokens"],
        completion_tokens=gen["completion_tokens"],
        total_tokens=gen["total_tokens"],
        tokens_per_sec=gen["tokens_per_sec"],
    )

    class _SimpleResponse:
        def __init__(self, text): self.text = text
        def __str__(self): return self.text
        @property
        def response(self): return self.text

    return _SimpleResponse(gen["text"]), nodes, q

# --------------------------- Evaluación cualitativa ----------------------------

def grade_with_llm(
    grader_model: str,
    question: str,
    predicted_answer: str,
    gold_answer: str,
    language: str = "es",
) -> QualMetrics:
    """
    Evalúa la respuesta con un 'Gold test' (0-10 + feedback).
    - JSON en el prompt con llaves escapadas {{ }}.
    - Parseo robusto: intenta JSON directo, luego primer bloque {...}, luego número suelto.
    - Feedback y respuesta normalizados a una sola línea.
    """
    prompts = {
        "es": (
            "Eres un evaluador estricto. Compara la RESPUESTA con la RESPUESTA_CORRECTA.\n"
            "Devuelve una nota de 0 a 10 (decimales permitidos) y una breve justificación.\n"
            "Pregunta: {q}\nRespuesta del sistema: {a}\nRespuesta correcta: {g}\n"
            "Formato de salida JSON: {{\"score\": 0-10, \"feedback\": \"texto\"}}\n"
            "IMPORTANTE: Devuelve ÚNICAMENTE el JSON, sin texto adicional, sin razonamiento ni bloques de código."
        ),
        "en": (
            "You are a strict grader. Compare the ANSWER to the GOLD_ANSWER.\n"
            "Return a score from 0 to 10 (decimals allowed) and a brief justification.\n"
            "Question: {q}\nSystem answer: {a}\nGold answer: {g}\n"
            "Output JSON: {{\"score\": 0-10, \"feedback\": \"text\"}}\n"
            "IMPORTANT: Return JSON ONLY, no extra text, no reasoning, no code fences."
        ),
        "ca": (
            "Ets un avaluador estricte. Compara la RESPOSTA amb la RESPOSTA_CORRECTA.\n"
            "Retorna una nota de 0 a 10 (amb decimals) i una breu justificació.\n"
            "Pregunta: {q}\nResposta del sistema: {a}\nResposta correcta: {g}\n"
            "Format de sortida JSON: {{\"score\": 0-10, \"feedback\": \"text\"}}\n"
            "IMPORTANT: Retorna NOMÉS el JSON, sense text extra, sense raonament ni blocs de codi."
        ),
    }

    prompt = prompts.get(language, prompts["es"]).format(
        q=question, a=predicted_answer, g=gold_answer
    )

    import requests, re
    url = "http://localhost:11434/api/generate"
    payload = {"model": grader_model, "prompt": prompt, "stream": False}

    try:
        r = requests.post(url, json=payload, timeout=180)
        if not r.ok:
            return QualMetrics(None, f"Error grader: {r.status_code} {r.text}")

        raw = (r.json().get("response") or "").strip()
        # normalizar comillas “”
        raw = raw.replace("“", "\"").replace("”", "\"").replace("’", "'")

        # 1) JSON directo
        score, feedback = None, None
        try:
            parsed = json.loads(raw)
            score = float(parsed.get("score"))
            feedback = str(parsed.get("feedback"))
        except Exception:
            # 2) buscar el ÚLTIMO bloque {...} (a veces el modelo escribe varias cosas)
            blocks = re.findall(r"\{.*?\}", raw, flags=re.DOTALL)
            for m in reversed(blocks):
                try:
                    parsed = json.loads(m)
                    score = float(parsed.get("score"))
                    feedback = str(parsed.get("feedback"))
                    break
                except Exception:
                    continue
            # 3) fallback: número suelto + texto entero
            if score is None:
                mnum = re.search(r"(\d+(?:\.\d+)?)", raw)
                score = float(mnum.group(1)) if mnum else None
            if feedback is None:
                feedback = raw

        return QualMetrics(score, _oneliner(feedback))
    except Exception as e:
        return QualMetrics(None, f"Excepción grader: {e}")


def consistency_lexical(predicted_answer: str, retrieved_nodes: List, language: str = "es") -> Dict[str, Any]:
    """Heurística léxica (mejorada) para consistencia factual y alucinaciones."""
    import re
    import unicodedata
    from collections import Counter
    def norm(text: str) -> str:
        if not text: return ""
        t = unicodedata.normalize("NFKD", text)
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        t = re.sub(r"[^\w\s]", " ", t.lower(), flags=re.UNICODE)
        t = re.sub(r"\s+", " ", t).strip()
        return t
    stop_es = {"el","la","los","las","un","una","unos","unas","de","del","al","y","o","u","en","es","son","ser","se","que","por","con","para","como"}
    stop_en = {"the","a","an","of","and","or","to","in","is","are","be","that","for","with","as","on","by","at"}
    stop_ca = {"el","la","els","les","un","una","uns","unes","de","del","al","i","o","u","en","es","són","ser","que","per","amb","com"}
    stops = {"es": stop_es, "en": stop_en, "ca": stop_ca}.get(language, stop_es)
    def toks(s: str):
        return [w for w in s.split() if len(w) >= 3 and w not in stops]
    ctx_text = " ".join([(getattr(n, "node", None).get_text() if getattr(n, "node", None) else getattr(n, "text", "")) or "" for n in (retrieved_nodes or [])])
    ctx_toks = toks(norm(ctx_text))
    ans_toks = toks(norm(predicted_answer))
    if not ans_toks or not ctx_toks:
        return {"factual_consistency_0_10": 0.0, "hallucination_rate_0_1": 1.0}
    def bigrams(seq): 
        return list(zip(seq, seq[1:])) if len(seq) >= 2 else []
    ctx_set = set(ctx_toks)
    uni_hits = sum(1 for w in ans_toks if w in ctx_set)
    uni_cov = uni_hits / max(1, len(ans_toks))
    ans_bi = set(bigrams(ans_toks))
    ctx_bi = set(bigrams(ctx_toks))
    bi_cov = (len(ans_bi & ctx_bi) / max(1, len(ans_bi))) if ans_bi else 0.0
    support = min(1.0, max(0.0, 0.75 * uni_cov + 0.25 * bi_cov))
    halluc = round(1.0 - support, 3)
    factual = round(10.0 * support, 2)
    return {
        "factual_consistency_0_10": factual,
        "hallucination_rate_0_1": halluc,
        "_mode": "lexical",
    }

def _ollama_embed(texts, model="nomic-embed-text", timeout=120):
    """
    Robusto a distintas versiones de Ollama:
      - /api/embed        {"model":..., "input":[...]}      -> {"embeddings":[[...],[...]]}
      - /api/embeddings   {"model":...,"prompt":"..."}      -> {"embedding":[...]}
    Si recibe lista, hace loop single en la API antigua.
    """
    import requests
    def _single(txt: str):
        # 1) endpoint nuevo (/api/embed)
        try:
            r = requests.post(
                "http://localhost:11434/api/embed",
                json={"model": model, "input": [txt]},
                timeout=timeout,
            )
            if r.ok:
                js = r.json()
                if "embeddings" in js and isinstance(js["embeddings"], list):
                    vec = js["embeddings"][0]
                    if isinstance(vec, list):
                        return vec
        except Exception:
            pass
        # 2) endpoint antiguo (/api/embeddings) – single
        r = requests.post(
            "http://localhost:11434/api/embeddings",
            json={"model": model, "prompt": txt},
            timeout=timeout,
        )
        r.raise_for_status()
        js = r.json()
        if "embedding" in js:
            return js["embedding"]
        if "data" in js and js["data"] and "embedding" in js["data"][0]:
            return js["data"][0]["embedding"]
        raise RuntimeError("No se pudo parsear embedding de Ollama")

    if isinstance(texts, str):
        return [_single(texts)]
    return [_single(t) for t in texts]

def _cos(a, b):
    import math
    da = math.sqrt(sum(x*x for x in a)) or 1.0
    db = math.sqrt(sum(x*x for x in b)) or 1.0
    return sum(x*y for x,y in zip(a,b)) / (da*db)

def consistency_semantic(predicted_answer: str, retrieved_nodes: List, language: str = "es", embed_model: str = "nomic-embed-text") -> Dict[str, Any]:
    """Groundedness semántico: similitud coseno entre la respuesta y el mejor chunk."""
    ctxs = [(getattr(n, "node", None).get_text() if getattr(n, "node", None) else getattr(n, "text", "")) or "" for n in (retrieved_nodes or [])]
    if not predicted_answer or not ctxs:
        return {"factual_consistency_0_10": 0.0, "hallucination_rate_0_1": 1.0}
    try:
        vec_a = _ollama_embed(predicted_answer, model=embed_model)[0]
        vec_cs = _ollama_embed(ctxs, model=embed_model)
        sim = max(_cos(vec_a, v) for v in vec_cs)  # similitud con el mejor chunk
        support = max(0.0, min(1.0, (sim + 1.0) / 2.0))  # [-1,1] → [0,1]
        return {
            "factual_consistency_0_10": round(10.0 * support, 2),
            "hallucination_rate_0_1": round(1.0 - support, 3),
            "_mode": "semantic",
        }
    except Exception:
        # Fallback: marca explícitamente que caímos a léxico
        out = consistency_lexical(predicted_answer, retrieved_nodes, language=language)
        out["_mode"] = "lexical_fallback"
        return out

# ------------------------------ Runner ----------------------------------------

def run_benchmark(
    data_dir: str,
    gold_path: str,
    llm_name: str,
    embed_name: str,
    top_k: int,
    language: str,
    grader_model: Optional[str] = None,
    use_chroma: bool = False,
    chroma_path: str = "./chroma_db",
    collection_name: str = "rag_collection",
    output_csv: str = "results.csv",
    ensure_models_flag: bool = True,
    groundedness: str = "semantic",
) -> str:

    # Elegir modelo evaluador por defecto
    effective_grader = grader_model or "llama3:8b"

    if ensure_models_flag:
        ensure_ollama_models([llm_name, embed_name, effective_grader])

    index = build_index(
        data_dir=data_dir,
        embed_model_name=embed_name,
        use_chroma=use_chroma,
        chroma_path=chroma_path,
        collection_name=collection_name,
        llm_name=llm_name,
    )

    # Usar el grader efectivo calculado arriba
    grader_model = effective_grader

    rows: List[Dict[str, Any]] = []
    # selector de métrica de groundedness
    def _ground_fn(answer, nodes, lang):
        if groundedness == "semantic":
            return consistency_semantic(answer, nodes, language=lang, embed_model=embed_name)
        return consistency_lexical(answer, nodes, language=lang)

    gold_items = read_gold_robust(gold_path)

    for item in gold_items:
        q = item["question"]
        gold_ans = item["gold_answer"]
        lang = item.get("language", language) or language

        # 1) Retrieval + generación con tokens
        response, nodes, qmetrics = time_retrieval_and_generation(
            index, q, top_k=top_k, llm_name=llm_name, language=lang
        )
        answer = _oneliner(str(response))

        # 2) Gold grading
        qual = grade_with_llm(
            grader_model=grader_model,
            question=q,
            predicted_answer=answer,
            gold_answer=gold_ans,
            language=lang
        )

        # 3) Groundedness/heurísticas
        extra = _ground_fn(answer, nodes, lang)
        qual.factual_consistency_0_10 = extra["factual_consistency_0_10"]
        qual.hallucination_rate_0_1 = extra["hallucination_rate_0_1"]

        # 3b) Evaluación de retrieval: ¿el GOLD está soportado por el contexto?
        gold_support = _ground_fn(gold_ans, nodes, lang)
        groundedness_pred = round((qual.factual_consistency_0_10 or 0)/10.0, 3)
        retrieval_support_gold = round((gold_support["factual_consistency_0_10"] or 0)/10.0, 3)


        rows.append({
            "q_index10": item.get("q_index10"),
            "question": q,
            "gold_answer": gold_ans,
            "predicted_answer": answer,
            "language": lang,
            "llm": llm_name,
            "embedding": embed_name,
            "top_k": top_k,
            "latency_total_ms": round(qmetrics.latency_total_ms, 2),
            "latency_retrieval_ms": round(qmetrics.latency_retrieval_ms, 2),
            "latency_generation_ms": round(qmetrics.latency_generation_ms, 2),
            "prompt_tokens": qmetrics.prompt_tokens or 0,
            "completion_tokens": qmetrics.completion_tokens or 0,
            "total_tokens": qmetrics.total_tokens or 0,
            "tokens_per_sec": qmetrics.tokens_per_sec if qmetrics.tokens_per_sec is not None else 0,
            "gold_score_0_10": qual.gold_score_0_10,
            "gold_feedback": _oneliner(qual.gold_feedback),
            "factual_consistency_0_10": qual.factual_consistency_0_10,
            "hallucination_rate_0_1": qual.hallucination_rate_0_1,
            "groundedness_pred_0_1": groundedness_pred,
            "retrieval_support_gold_0_1": retrieval_support_gold,
            # modos usados realmente (útil para detectar fallbacks)
            "groundedness_mode_pred": extra.get("_mode", ""),
            "retrieval_support_mode_gold": gold_support.get("_mode", ""),
            # qué modo pediste al script
            "groundedness_requested": groundedness,
        })

    out_path = os.path.abspath(output_csv)
    if not rows:
        raise RuntimeError("No se han generado filas de resultados. ¿Está vacío el CSV de gold?")

    # utf-8-sig => Excel muestra bien acentos
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
            quoting=csv.QUOTE_MINIMAL
        )
        writer.writeheader()
        writer.writerows(rows)

    return out_path


# --- main(): sustituye el parser por este bloque extendido ---
def main():
    parser = argparse.ArgumentParser()
    # NUEVOS: configuración por idioma
    parser.add_argument("--lang", default="es")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR,
                        help="Carpeta base donde viven gold y colecciones por idioma")
    parser.add_argument("--gold_prefix", default=DEFAULT_GOLD_PREFIX)
    parser.add_argument("--gold_ext", default=DEFAULT_GOLD_EXT)
    parser.add_argument("--collection_prefix", default=DEFAULT_COLLECTION_PREFIX)

    # Permite AUTO o placeholders con {lang}
    parser.add_argument("--data_dir", default=AUTO,
                        help="Carpeta con documentos para RAG. "
                             "Usa AUTO para {base_dir}/{collection_prefix}{lang} o un path con {lang}.")
    parser.add_argument("--gold", default=AUTO,
                        help="CSV gold. Usa AUTO para {base_dir}/{gold_prefix}{lang}{gold_ext} o un path con {lang}.")
    parser.add_argument("--collection", default=AUTO,
                        help="Nombre de colección. Usa AUTO para {collection_prefix}{lang} o un valor con {lang}.")

    parser.add_argument("--llm", default="llama3")
    parser.add_argument("--embed", default="nomic-embed-text")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--grader", default=None,
                        help="Modelo de evaluación (si no, se usa 'llama3:8b' por defecto dentro de run_benchmark)")
    parser.add_argument("--use_chroma", action="store_true")
    parser.add_argument("--chroma_path", default="./chroma_db",
                        help="Ruta Chroma (admite {lang} si quieres particionar por idioma)")
    parser.add_argument("--output", default="results_{lang}.csv",
                        help="Permite {lang} en el nombre de salida")
    parser.add_argument("--no_ensure_models", action="store_true",
                        help="No comprobar/descargar modelos en Ollama")
    parser.add_argument("--groundedness", choices=["semantic","lexical"], default="semantic",
                        help="Cómo calcular groundedness/alucinación (semantic usa embeddings de Ollama).")    
    args = parser.parse_args()

    # --------- Resolución por idioma ----------
    lang = args.lang

    # data_dir
    if args.data_dir == AUTO:
        data_dir = os.path.join(args.base_dir, f"{args.collection_prefix}{lang}")
    else:
        data_dir = args.data_dir.format(lang=lang)

    # gold
    if args.gold == AUTO:
        gold_path = os.path.join(args.base_dir, f"{args.gold_prefix}{lang}{args.gold_ext}")
    else:
        gold_path = args.gold.format(lang=lang)

    # collection
    if args.collection == AUTO:
        collection_name = f"{args.collection_prefix}{lang}"
    else:
        collection_name = args.collection.format(lang=lang)

    # chroma_path y output aceptan {lang}
    chroma_path = (args.chroma_path or "./chroma_db").format(lang=lang)
    output_csv = (args.output or "results.csv").format(lang=lang)

    # --------- Ejecutar benchmark ----------
    try:
        out_csv = run_benchmark(
            data_dir=data_dir,
            gold_path=gold_path,
            llm_name=args.llm,
            embed_name=args.embed,
            top_k=args.top_k,
            language=lang,
            grader_model=args.grader,
            use_chroma=args.use_chroma,
            chroma_path=chroma_path,
            collection_name=collection_name,
            output_csv=output_csv,
            ensure_models_flag=not args.no_ensure_models,
            groundedness=args.groundedness,
        )
    except PermissionError:
        base, ext = os.path.splitext(output_csv)
        ts = time.strftime("%Y%m%d_%H%M%S")
        alt_output = f"{base}_{ts}{ext or '.csv'}"
        out_csv = run_benchmark(
            data_dir=data_dir,
            gold_path=gold_path,
            llm_name=args.llm,
            embed_name=args.embed,
            top_k=args.top_k,
            language=lang,
            grader_model=args.grader,
            use_chroma=args.use_chroma,
            chroma_path=chroma_path,
            collection_name=collection_name,
            output_csv=alt_output,
            ensure_models_flag=not args.no_ensure_models,
            groundedness=args.groundedness,
        )

    print(f"✅ Resultados guardados en: {out_csv}")



if __name__ == "__main__":
    main()
