# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
# (Actualizado) Añadidas métricas groundedness_pred_0_1 y retrieval_support_gold_0_1 y nuevo leaderboard sin q_index10.
# Analiza aggregate_all.csv con el esquema:
# db_lang,question,gold_answer,predicted_answer,language,llm,embedding,top_k,
# latency_total_ms,latency_retrieval_ms,latency_generation_ms,
# prompt_tokens,completion_tokens,total_tokens,tokens_per_sec,
# gold_score_0_10,gold_feedback,factual_consistency_0_10,hallucination_rate_0_1
#
# Genera métricas para respuestas TEXTUALES (no MCQ):
# - EM normalizado
# - F1 por tokens
# - ROUGE-L F1 (LCS)
# - Similitud de Levenshtein (rapidfuzz si está, si no fallback)
# - Accuracy "suave" a partir de umbrales (ROUGE-L>=0.7, LEV_SIM>=0.8)
#
# Produce tablas por idioma, LLM, embedding, top_k y combinaciones, con IC95 por bootstrap.
#
# Uso:
#   python analyze_schema_bench.py --in aggregate_all.csv --outdir reports

# Requisitos: pandas, numpy, (opcional) rapidfuzz
"""
import argparse
import re
import ast
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------- utilidades de texto/metricas -----------------

_ws_re = re.compile(r"\s+", flags=re.UNICODE)

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = s.replace("’","'").replace("´","'").replace("`","'")
    s = s.replace("“",'"').replace("”",'"')
    s = _ws_re.sub(" ", s)
    # conserva letras acentuadas, dígitos y guiones; quita el resto
    s = re.sub(r"[^\wáéíóúñüçàèìòùäëïöâêîôû-]+", " ", s, flags=re.UNICODE)
    return _ws_re.sub(" ", s).strip()

def tokens(s: str):
    return normalize_text(s).split()

def em(a: str, b: str) -> int:
    return int(normalize_text(a) == normalize_text(b))

from collections import Counter
def f1_tokens(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb: return 1.0
    if not ta or not tb:  return 0.0
    ca, cb = Counter(ta), Counter(tb)
    common = sum((ca & cb).values())
    if common == 0: return 0.0
    prec = common/len(ta); rec = common/len(tb)
    return 2*prec*rec/(prec+rec)

def lcs_len(a_tokens, b_tokens):
    n, m = len(a_tokens), len(b_tokens)
    if n == 0 or m == 0: return 0
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0
        for j in range(1, m+1):
            temp = dp[j]
            if a_tokens[i-1] == b_tokens[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[m]

def rouge_l_f1(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb: return 1.0
    if not ta or not tb:  return 0.0
    lcs = lcs_len(ta, tb)
    prec = lcs / len(ta) if len(ta) else 0.0
    rec  = lcs / len(tb) if len(tb) else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec) else 0.0

def lev_similarity(a: str, b: str) -> float:
    """Similitud de Levenshtein en rango [0,1]."""
    try:
        from rapidfuzz.distance import Levenshtein
        sim = Levenshtein.normalized_similarity(normalize_text(a), normalize_text(b))
        # Guard compatible: si alguna versión devuelve 0..100, normalizamos.
        if sim > 1.0:
            sim = sim / 100.0
        return float(sim)
    except Exception:
        # Fallback O(n*m) para cadenas razonables
        aa, bb = normalize_text(a), normalize_text(b)
        if not aa and not bb: return 1.0
        if not aa or not bb:  return 0.0
        n, m = len(aa), len(bb)
        if n*m > 10000:
            # fallback más barato: Jaccard sobre tokens
            A, B = set(aa.split()), set(bb.split())
            inter = len(A & B); uni = len(A | B) or 1
            return inter/uni
        dp = list(range(m+1))
        for i in range(1, n+1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, m+1):
                tmp = dp[j]
                cost = 0 if aa[i-1]==bb[j-1] else 1
                dp[j] = min(dp[j]+1, dp[j-1]+1, prev+cost)
                prev = tmp
        dist = dp[m]
        return 1.0 - dist / max(n, m)

def parse_refs(x):
    # Admite string; '|||'; lista como string; lista real
    if isinstance(x, list): return x
    if isinstance(x, str):
        xs = x.strip()
        if '|||' in xs:
            return [t.strip() for t in xs.split('|||') if t.strip()]
        if xs.startswith('[') and xs.endswith(']'):
            try:
                lst = ast.literal_eval(xs)
                return [str(t) for t in lst]
            except Exception:
                return [x]
        return [x]
    return ['']

def best_of_refs(pred: str, refs, scorer):
    return max(scorer(pred, r) for r in refs)

def bootstrap_ci_mean(arr, n_boot=2000, alpha=0.05, seed=42):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0: return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))

# ----------------- construcción de métricas -----------------

def build_metrics(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # numéricos seguros
    for c in ['top_k','prompt_tokens','completion_tokens','total_tokens','tokens_per_sec',
              'gold_score_0_10','factual_consistency_0_10','hallucination_rate_0_1',
              'groundedness_pred_0_1','retrieval_support_gold_0_1']:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors='coerce')

    for c in ['latency_total_ms','latency_retrieval_ms','latency_generation_ms']:
        if c in d.columns: d[c] = pd.to_numeric(d[c], errors='coerce')
    # latencias a segundos
    for c in ['latency_total_ms','latency_retrieval_ms','latency_generation_ms']:
        if c in d.columns:
            d[c.replace('_ms','_s')] = d[c] / 1000.0

    # longitudes
    d['q_len']    = d['question'].map(lambda s: len(tokens(s)) if isinstance(s,str) else np.nan)
    d['gold_len'] = d['gold_answer'].map(lambda s: len(tokens(s)) if isinstance(s,str) else np.nan)
    d['pred_len'] = d['predicted_answer'].map(lambda s: len(tokens(s)) if isinstance(s,str) else np.nan)

    # métricas textuales (mejor de múltiples refs si hay)
    gold_refs = d['gold_answer'].map(parse_refs)
    preds = d['predicted_answer'].fillna('')

    d['EM']       = [best_of_refs(p, r, em)            for p, r in zip(preds, gold_refs)]
    d['F1']       = [best_of_refs(p, r, f1_tokens)     for p, r in zip(preds, gold_refs)]
    d['ROUGE_L']  = [best_of_refs(p, r, rouge_l_f1)    for p, r in zip(preds, gold_refs)]
    d['LEV_SIM']  = [best_of_refs(p, r, lev_similarity)for p, r in zip(preds, gold_refs)]

    # accuracies suaves por umbral
    d['ACC_rouge70'] = (d['ROUGE_L'] >= 0.70).astype(int)
    d['ACC_lev80']   = (d['LEV_SIM'] >= 0.80).astype(int)

    # eficiencia si falta tokens_per_sec
    if ('tokens_per_sec' not in d.columns or d['tokens_per_sec'].isna().all()) and        ('completion_tokens' in d.columns and 'latency_generation_s' in d.columns):
        d['tokens_per_sec'] = d['completion_tokens'] / d['latency_generation_s']

    return d

# ----------------- agregación con IC -----------------

def agg_with_ci(d: pd.DataFrame, by):
    def _safe_mean(series):
        return pd.to_numeric(series, errors='coerce').mean()

    agg = d.groupby(by, dropna=False).agg(
        n=('question','count'),
        EM_mean=('EM','mean'),
        F1_mean=('F1','mean'),
        ROUGE_L_mean=('ROUGE_L','mean'),
        LEV_SIM_mean=('LEV_SIM','mean'),
        ACC_rouge70=('ACC_rouge70','mean'),
        ACC_lev80=('ACC_lev80','mean'),
        gold_score_mean=('gold_score_0_10', _safe_mean),
        fact_cons_mean=('factual_consistency_0_10', _safe_mean),
        halluc_rate_mean=('hallucination_rate_0_1', _safe_mean),
        groundedness_mean=('groundedness_pred_0_1', _safe_mean),
        retrieval_support_mean=('retrieval_support_gold_0_1', _safe_mean),
        lat_total_s_mean=('latency_total_s', _safe_mean),
        lat_gen_s_mean=('latency_generation_s', _safe_mean),
        lat_ret_s_mean=('latency_retrieval_s', _safe_mean),
        tok_total_mean=('total_tokens', _safe_mean),
        tok_prompt_mean=('prompt_tokens', _safe_mean),
        tok_comp_mean=('completion_tokens', _safe_mean),
        tps_mean=('tokens_per_sec', _safe_mean),
    ).reset_index()

    # IC95 para EM / F1 / ROUGE-L
    cis = d.groupby(by, dropna=False, group_keys=False).apply(
        lambda g: pd.Series({
            'EM_ci_lo': bootstrap_ci_mean(g['EM'])[0],
            'EM_ci_hi': bootstrap_ci_mean(g['EM'])[1],
            'F1_ci_lo': bootstrap_ci_mean(g['F1'])[0],
            'F1_ci_hi': bootstrap_ci_mean(g['F1'])[1],
            'ROUGE_L_ci_lo': bootstrap_ci_mean(g['ROUGE_L'])[0],
            'ROUGE_L_ci_hi': bootstrap_ci_mean(g['ROUGE_L'])[1],
        })
    ).reset_index(drop=True)
    agg = pd.concat([agg, cis], axis=1)
    return agg

# ----------------- main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='input_csv', default='aggregate_all.csv')
    ap.add_argument('--outdir', default='reports')
    ap.add_argument('--sep', default=',')
    ap.add_argument('--encoding', default='utf-8-sig')
    args = ap.parse_args()

    inp = Path(args.input_csv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(inp, sep=args.sep, encoding=args.encoding)
    d = build_metrics(df)
    
    # Helper para guardar CSVs opcionales
    def _save(name, table):
        table.to_csv(outdir/f"{name}.csv", index=False, encoding="utf-8-sig") if (table is not None and not table.empty) else None

    # === tablas clave (db_lang vs language + q_index10) ===
    have_db   = "db_lang"   in d.columns
    have_ans  = "language"  in d.columns
    have_llm  = "llm"       in d.columns
    have_emb  = "embedding" in d.columns
    have_k    = "top_k"     in d.columns
    have_qi   = "q_index10" in d.columns

    tables = {}

    # 1) Por idioma de la BD (embedding / corpus)
    if have_db:
        tables["by_db_lang"] = agg_with_ci(d, ["db_lang"])

    # 2) Por idioma de pregunta/respuesta
    if have_ans:
        tables["by_answer_lang"] = agg_with_ci(d, ["language"])

    # 3) Por índice cíclico de pregunta (1..10, cada bloque de 3 preguntas)
    if have_qi:
        tables["by_qindex10"] = agg_with_ci(d, ["q_index10"])

    # 4) Cruces con q_index10 (útiles para ver interacciones)
    if have_db and have_qi:
        tables["by_db_vs_qindex10"] = agg_with_ci(d, ["db_lang","q_index10"])
    if have_ans and have_qi:
        tables["by_anslang_vs_qindex10"] = agg_with_ci(d, ["language","q_index10"])

    # 5) Cruce BD vs respuesta (cross-lingual)
    if have_db and have_ans:
        tables["by_db_vs_answer_lang"] = agg_with_ci(d, ["db_lang","language"])

    # 6) Por LLM, embedding y top_k
    if have_llm:
        tables["by_llm"] = agg_with_ci(d, ["llm"])
    if have_emb:
        tables["by_embedding"] = agg_with_ci(d, ["embedding"])
    if have_k:
        tables["by_topk"] = agg_with_ci(d, ["top_k"])

    # 7) Combinación completa    
    combo_keys = [c for c in ["db_lang","language","llm","embedding","top_k","q_index10"] if c in d.columns]
    if combo_keys:
        tables["by_db_answer_llm_emb_k_qi"] = agg_with_ci(d, combo_keys)    

    # ===== NUEVO: Leaderboard sin q_index10 (language + llm + embedding), filtrando top_k=3 si existe =====
    d_top3 = d
    if "top_k" in d.columns:
        try:
            d_top3 = d[pd.to_numeric(d["top_k"], errors="coerce")==3].copy()
            if d_top3.empty:
                d_top3 = d.copy()
        except Exception:
            d_top3 = d.copy()
    needed = {"language","llm","embedding"}
    if needed.issubset(d_top3.columns):
        tbl_lang_llm_emb = agg_with_ci(d_top3, ["language","llm","embedding"])
        leaderboard = tbl_lang_llm_emb.sort_values(["ROUGE_L_mean","EM_mean"], ascending=False).head(15)
    else:
        tbl_lang_llm_emb = pd.DataFrame()
        leaderboard = pd.DataFrame()

    # Correlaciones
    corr = pd.DataFrame({
        "EM_vs_lat_total":        [d["EM"].corr(d.get("latency_total_s"))],
        "F1_vs_lat_total":        [d["F1"].corr(d.get("latency_total_s"))],
        "ROUGE_vs_lat_total":     [d["ROUGE_L"].corr(d.get("latency_total_s"))],
        "EM_vs_tokens":           [d["EM"].corr(d.get("total_tokens"))],
        "F1_vs_tokens":           [d["F1"].corr(d.get("total_tokens"))],
        "ROUGE_vs_tokens":        [d["ROUGE_L"].corr(d.get("total_tokens"))],
        # relaciones con las nuevas métricas (si existen):
        "ROUGE_vs_groundedness":    [d.get("ROUGE_L").corr(d.get("groundedness_pred_0_1")) if "groundedness_pred_0_1" in d.columns else np.nan],
        "ROUGE_vs_retrievalsup":  [d.get("ROUGE_L").corr(d.get("retrieval_support_gold_0_1")) if "retrieval_support_gold_0_1" in d.columns else np.nan],
        "Halluc_vs_fact":         [d.get("hallucination_rate_0_1").corr(d.get("factual_consistency_0_10"))],
    })


    # ===== NUEVAS TABLAS SOLICITADAS =====
    # (1) Mejor combinación LLM+Embedding por lengua (con top_k=3 si existe)
    if not tbl_lang_llm_emb.empty:
        best_per_lang = (
            tbl_lang_llm_emb
            .sort_values(["language","ROUGE_L_mean","EM_mean"], ascending=[True,False,False])
            .drop_duplicates(subset=["language"], keep="first")
            .reset_index(drop=True)
        )
        _save("by_language_best_llm_embedding_topk3", best_per_lang)

    # (2) Estadísticas por pregunta y (2b) por pregunta + lengua + embedding
    q_key = "question_id" if "question_id" in d.columns else ("q_index10" if "q_index10" in d.columns else None)
    if q_key is None:
        d["q_index10"] = (pd.Series(range(1, len(d)+1)) % 10).replace(0,10)
        q_key = "q_index10"
    by_question = agg_with_ci(d, [q_key])
    _save("by_question", by_question)
    if {"language","embedding"}.issubset(d.columns):
        by_question_lang_embedding = agg_with_ci(d, [q_key,"language","embedding"])
        _save("by_question_lang_embedding", by_question_lang_embedding)

    # (3) Estadísticas por LLM
    if "llm" in d.columns:
        by_llm = agg_with_ci(d, ["llm"])
        tables["by_llm"] = by_llm
        _save("by_llm", by_llm)

    # (4) Estadísticas por Embedding
    if "embedding" in d.columns:
        by_embedding = agg_with_ci(d, ["embedding"])
        tables["by_embedding"] = by_embedding
        _save("by_embedding", by_embedding)

    # ------ exportaciones ------
    def save_all(name, table):
        if table is None or table.empty: return
        table.to_csv(outdir/f"{name}.csv", index=False, encoding="utf-8-sig")
        try:
            with open(outdir/f"{name}.tex","w",encoding="utf-8") as f:
                f.write(table.to_latex(index=False, float_format="%.3f"))
        except Exception:
            pass

    for name, tbl in tables.items():
        save_all(name, tbl)

    save_all("leaderboard_top15", leaderboard)
    corr.to_csv(outdir/"correlations.csv", index=False, encoding="utf-8-sig")

    # Excel con todo
    with pd.ExcelWriter(outdir/"summary.xlsx") as xw:
        d.to_excel(xw, index=False, sheet_name="rows")
        for name, tbl in tables.items():
            if tbl is not None and not tbl.empty:
                sheet = name[:31]  # límite Excel
                tbl.to_excel(xw, index=False, sheet_name=sheet)
        # nuevas pestañas útiles
        if not tbl_lang_llm_emb.empty:
            tbl_lang_llm_emb.to_excel(xw, index=False, sheet_name="by_lang_llm_emb")
            # best por lengua
            try:
                best_per_lang.to_excel(xw, index=False, sheet_name="best_by_lang")
            except Exception:
                pass
        if not leaderboard.empty:
            leaderboard.to_excel(xw, index=False, sheet_name="leaderboard_no_qi")
        try:
            by_question.to_excel(xw, index=False, sheet_name="by_question")
        except Exception:
            pass
        try:
            by_question_lang_embedding.to_excel(xw, index=False, sheet_name="by_q_lang_emb")
        except Exception:
            pass
        try:
            tables.get("by_llm", pd.DataFrame()).to_excel(xw, index=False, sheet_name="by_llm")
            tables.get("by_embedding", pd.DataFrame()).to_excel(xw, index=False, sheet_name="by_embedding")
        except Exception:
            pass
        corr.to_excel(xw, index=False, sheet_name="correlations")

    # resumen ejecutivo en consola
    def fmt_pct(x):
        try: return f"{100*float(x):.1f}%"
        except: return "nan"

    print("\n=== RESUMEN ===")
    if have_db and "by_db_lang" in tables and not tables["by_db_lang"].empty:
        print("ROUGE-L por idioma de BD (db_lang):")
        for _, r in tables["by_db_lang"].sort_values("ROUGE_L_mean", ascending=False).iterrows():
            print(f"  {r['db_lang']}: ROUGE-L={r['ROUGE_L_mean']:.3f}  EM={fmt_pct(r['EM_mean'])}  (n={int(r['n'])})")

    if have_ans and "by_answer_lang" in tables and not tables["by_answer_lang"].empty:
        print("ROUGE-L por idioma de respuesta (language):")
        for _, r in tables["by_answer_lang"].sort_values("ROUGE_L_mean", ascending=False).iterrows():
            print(f"  {r['language']}: ROUGE-L={r['ROUGE_L_mean']:.3f}  EM={fmt_pct(r['EM_mean'])}  (n={int(r['n'])})")

    if have_qi and "by_qindex10" in tables and not tables["by_qindex10"].empty:
        print("ROUGE-L por índice de pregunta (q_index10):")
        for _, r in tables["by_qindex10"].sort_values("q_index10").iterrows():
            print(f"  idx {int(r['q_index10'])}: ROUGE-L={r['ROUGE_L_mean']:.3f}  EM={fmt_pct(r['EM_mean'])}  (n={int(r['n'])})")

    if not leaderboard.empty:
        best = leaderboard.iloc[0]
        tags = ", ".join([f"{c}={best[c]}" for c in ["language","llm","embedding"] if c in best.index])
        print(f"Mejor combinación: {tags}")
        print(f"  ROUGE-L={best['ROUGE_L_mean']:.3f}  EM={fmt_pct(best['EM_mean'])}  F1={best['F1_mean']:.3f}"
              f"  lat_total≈{best.get('lat_total_s_mean', np.nan):.2f}s")

    print(f"Salida en: {outdir.resolve()}")
    print("Tablas añadidas:")
    print(" - by_language_best_llm_embedding_topk3.csv")
    print(" - by_question.csv")
    print(" - by_question_lang_embedding.csv")
    print(" - by_llm.csv, by_embedding.csv, leaderboard_top15.csv (sin q_index10)")

if __name__ == '__main__':
    main()