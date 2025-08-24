# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 23:10:38 2025

@author: Admin
"""
# aggregate_results.py
import os, glob, pandas as pd, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Carpeta con CSVs individuales")
    ap.add_argument("--out", default="combined_summary.csv", help="CSV combinado")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.in_dir, "*.csv"))
    if not files:
        raise SystemExit(f"No se encontraron CSVs en {args.in_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(f, encoding="utf-8")
        df["__file"] = os.path.basename(f)
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)

    # Guardar combinado
    full.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"✅ Combinado guardado: {args.out} ({len(full)} filas)")

    # Agregados por (llm, embedding, language, top_k)
    metrics = [
        "latency_total_ms","latency_retrieval_ms","latency_generation_ms",
        "prompt_tokens","completion_tokens","total_tokens","tokens_per_sec",
        "gold_score_0_10","factual_consistency_0_10","hallucination_rate_0_1"
    ]
    group_cols = ["llm","embedding","language","top_k"]
    agg = full.groupby(group_cols, dropna=False)[metrics].mean().reset_index()
    agg_out = os.path.splitext(args.out)[0] + "_grouped.csv"
    agg.to_csv(agg_out, index=False, encoding="utf-8-sig")
    print(f"📊 Resumen por configuración: {agg_out}")

if __name__ == "__main__":
    main()
