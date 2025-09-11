
#!/usr/bin/env python3
"""
robust_aggregate_bench.py

Objetivo: Agregar resultados de benchmarks ubicados en una carpeta (por defecto `bench_out/`)
sin mezclar columnas, aun cuando:
- Hay CSV con diferentes delimitadores (coma, punto y coma, tab)
- Hay filas con comas dentro de texto
- Hay JSONL mezclado con CSV
- Hay encabezados con espacios/acentos/inconsistencias
- Hay columnas que aparecen en unos archivos y en otros no

Salida: CSV seguro para Excel/LibreOffice (todas las celdas entrecomilladas, UTF-8 con BOM)
"""

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


PREFERRED_ORDER = [
    # ajusta este orden a tu esquema estándar
    "run_id","dataset","split","question_id","question","options","answer",
    "gold","gold_explanation","prediction","prediction_explanation",
    "is_correct","score","latency_ms","duration_ms",
    "model","temperature","top_p","seed",
    "prompt_tokens","completion_tokens","total_tokens",
    "n_tokens_prompt","n_tokens_completion",
    "timestamp","source_file"
]

def norm_col(name: str) -> str:
    if name is None:
        return ""
    s = name.strip()
    # normaliza: minúsculas, sin espacios duplicados, reemplaza espacios/símbolos por _
    s = s.lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("º", "o").replace("ª", "a")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s

def sniff_delimiter(sample: bytes) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"), delimiters=[",",";","\t","|"])
        return dialect.delimiter
    except Exception:
        return ","  # fallback

def read_csv_any(path: Path) -> pd.DataFrame:
    # lee unas cuantas KB para detectar delimitador
    with path.open("rb") as f:
        sample = f.read(65536)
    sep = sniff_delimiter(sample)
    # usa engine=python para manejar mejor quotes extraños
    df = pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
        quotechar='"',
        quoting=csv.QUOTE_MINIMAL,
        escapechar="\\",
        keep_default_na=False
    )
    return df

def read_jsonl(path: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    rows.append({"value": obj})
            except Exception:
                # Línea corrupta -> saltar
                continue
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def coerce_str_frame(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = df[c].astype(str).fillna("")
    return df

def aggregate(input_dir: Path, output_csv: Path, output_parquet: Optional[Path]=None) -> None:
    files = []
    for ext in ("*.csv","*.tsv","*.jsonl","*.ndjson"):
        files.extend(input_dir.rglob(ext))
    if not files:
        print(f"[WARN] No se encontraron archivos en {input_dir}", file=sys.stderr)
        # crear salida vacía con BOM y encabezado mínimo
        pd.DataFrame().to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        return

    frames = []
    for p in sorted(files):
        try:
            if p.suffix.lower() in [".csv",".tsv"]:
                df = read_csv_any(p)
            elif p.suffix.lower() in [".jsonl",".ndjson"]:
                df = read_jsonl(p)
            else:
                continue
            if df is None or df.empty:
                continue

            # normaliza encabezados
            df.columns = [norm_col(c) for c in df.columns]

            # añade source_file para rastrear el origen
            df["source_file"] = str(p.relative_to(input_dir))

            frames.append(coerce_str_frame(df))
        except Exception as e:
            print(f"[WARN] No se pudo leer {p}: {e}", file=sys.stderr)
            continue

    if not frames:
        print(f"[WARN] No se pudieron agregar archivos de {input_dir}", file=sys.stderr)
        pd.DataFrame().to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        return

    # unión de columnas, luego reordenar
    agg = pd.concat(frames, axis=0, ignore_index=True, sort=True)

    # orden preferente + el resto alfabético
    cols_present = list(agg.columns)
    ordered = [c for c in PREFERRED_ORDER if c in cols_present]
    rest = sorted([c for c in cols_present if c not in ordered])
    agg = agg[ordered + rest]

    # escritura segura para Excel: UTF-8 con BOM + todas las celdas entrecomilladas
    agg.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

    if output_parquet:
        try:
            agg.to_parquet(output_parquet, index=False)
        except Exception as e:
            print(f"[WARN] No se pudo escribir parquet: {e}", file=sys.stderr)

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Agrega salidas de benchmark sin mezclar columnas.")
    ap.add_argument("--input-dir", "-i", default="bench_out", type=str, help="Carpeta con los archivos a agregar.")
    ap.add_argument("--output", "-o", default="aggregate.csv", type=str, help="Ruta del CSV de salida.")
    ap.add_argument("--parquet", "-p", default=None, type=str, help="Ruta opcional para exportar Parquet.")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output).expanduser().resolve()
    output_parquet = Path(args.parquet).expanduser().resolve() if args.parquet else None

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_parquet:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)

    aggregate(input_dir, output_csv, output_parquet)
    print(f"[OK] CSV escrito en {output_csv}")
    if output_parquet:
        print(f"[OK] Parquet escrito en {output_parquet}")

if __name__ == "__main__":
    main()
