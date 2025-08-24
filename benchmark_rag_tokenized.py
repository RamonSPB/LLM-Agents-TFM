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

def build_prompt(query: str, nodes, language: str = "es") -> str:
    sys_instructions = {
        "es": ("Eres un asistente experto. Responde SÓLO con la información del CONTEXTO.\n"
               "Si no está en el contexto, indícalo explícitamente.\n"
               "Sé conciso y exacto.\n"),
        "en": ("You are an expert assistant. Answer ONLY using the CONTEXT.\n"
               "If it's not in the context, say it explicitly.\n"
               "Be concise and precise.\n"),
        "ca": ("Ets un assistent expert. Respon NOMÉS amb la informació del CONTEXT.\n"
               "Si no és al context, digues-ho explícitament.\n"
               "Sigues concís i precís.\n"),
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
    text = _oneliner((data.get("response") or "").strip())

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
            "Formato de salida JSON: {{\"score\": 0-10, \"feedback\": \"texto\"}}"
        ),
        "en": (
            "You are a strict grader. Compare the ANSWER to the GOLD_ANSWER.\n"
            "Return a score from 0 to 10 (decimals allowed) and a brief justification.\n"
            "Question: {q}\nSystem answer: {a}\nGold answer: {g}\n"
            "Output JSON: {{\"score\": 0-10, \"feedback\": \"text\"}}"
        ),
        "ca": (
            "Ets un avaluador estricte. Compara la RESPOSTA amb la RESPOSTA_CORRECTA.\n"
            "Retorna una nota de 0 a 10 (amb decimals) i una breu justificació.\n"
            "Pregunta: {q}\nResposta del sistema: {a}\nResposta correcta: {g}\n"
            "Format de sortida JSON: {{\"score\": 0-10, \"feedback\": \"text\"}}"
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
            # 2) primer bloque {...} en el texto
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    score = float(parsed.get("score"))
                    feedback = str(parsed.get("feedback"))
                except Exception:
                    pass
            # 3) fallback: número suelto + texto entero
            if score is None:
                mnum = re.search(r"(\d+(?:\.\d+)?)", raw)
                score = float(mnum.group(1)) if mnum else None
            if feedback is None:
                feedback = raw

        return QualMetrics(score, _oneliner(feedback))
    except Exception as e:
        return QualMetrics(None, f"Excepción grader: {e}")


def simple_consistency_checks(predicted_answer: str, retrieved_nodes: List, language: str = "es") -> Dict[str, Any]:
    """Heurísticas ligeras para consistencia factual y alucinaciones."""
    import re
    context_text = " ".join([n.node.get_text() for n in retrieved_nodes]) if retrieved_nodes else ""
    resp_sents = [s.strip() for s in re.split(r"[\.!?]\s+", predicted_answer) if s.strip()]
    hits = 0
    for s in resp_sents:
        words = [w.lower() for w in re.findall(r"[\wÀ-ÿ]{5,}", s)]
        if not words:
            continue
        covered = sum(1 for w in words if w in context_text.lower())
        if covered >= max(1, len(words)//4):
            hits += 1
    halluc_rate = 1.0 - (hits / len(resp_sents)) if resp_sents else 1.0
    factual = max(0.0, 10.0 * (1.0 - halluc_rate))
    return {
        "factual_consistency_0_10": round(factual, 2),
        "hallucination_rate_0_1": round(halluc_rate, 3)
    }


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
) -> str:

    if ensure_models_flag:
        ensure_ollama_models([llm_name, embed_name, grader_model or llm_name])

    index = build_index(
        data_dir=data_dir,
        embed_model_name=embed_name,
        use_chroma=use_chroma,
        chroma_path=chroma_path,
        collection_name=collection_name,
        llm_name=llm_name,
    )

    if grader_model is None:
        grader_model = llm_name

    rows: List[Dict[str, Any]] = []
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

        # 3) Heurísticas extra
        extra = simple_consistency_checks(answer, nodes, language=lang)
        qual.factual_consistency_0_10 = extra["factual_consistency_0_10"]
        qual.hallucination_rate_0_1 = extra["hallucination_rate_0_1"]

        rows.append({
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Carpeta con documentos para RAG")
    parser.add_argument("--gold", required=True, help="CSV con columnas: question,gold_answer[,language]")
    parser.add_argument("--llm", default="llama3")
    parser.add_argument("--embed", default="nomic-embed-text")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--lang", default="es")
    parser.add_argument("--grader", default=None, help="Modelo de evaluación (por defecto igual a --llm)")
    parser.add_argument("--use_chroma", action="store_true")
    parser.add_argument("--chroma_path", default="./chroma_db")
    parser.add_argument("--collection", default="rag_collection")
    parser.add_argument("--output", default="results.csv")
    parser.add_argument("--no_ensure_models", action="store_true", help="No comprobar/descargar modelos en Ollama")
    args = parser.parse_args()

    # Si el CSV está abierto en Excel, puede dar PermissionError; añade sufijo de fecha si pasa
    try:
        out_csv = run_benchmark(
            data_dir=args.data_dir,
            gold_path=args.gold,
            llm_name=args.llm,
            embed_name=args.embed,
            top_k=args.top_k,
            language=args.lang,
            grader_model=args.grader,
            use_chroma=args.use_chroma,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            output_csv=args.output,
            ensure_models_flag=not args.no_ensure_models,
        )
    except PermissionError:
        base, ext = os.path.splitext(args.output)
        ts = time.strftime("%Y%m%d_%H%M%S")
        alt_output = f"{base}_{ts}{ext or '.csv'}"
        out_csv = run_benchmark(
            data_dir=args.data_dir,
            gold_path=args.gold,
            llm_name=args.llm,
            embed_name=args.embed,
            top_k=args.top_k,
            language=args.lang,
            grader_model=args.grader,
            use_chroma=args.use_chroma,
            chroma_path=args.chroma_path,
            collection_name=args.collection,
            output_csv=alt_output,
            ensure_models_flag=not args.no_ensure_models,
        )

    print(f"✅ Resultados guardados en: {out_csv}")


if __name__ == "__main__":
    main()
