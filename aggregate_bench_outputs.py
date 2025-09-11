# -*- coding: utf-8 -*-
"""
Unifica los CSV generados en bench_out_{lang}/* y añade la primera columna 'db_lang'.

Uso:
  python aggregate_bench_outputs.py \
      --roots bench_out_ca bench_out_en bench_out_es \
      --pattern "*.csv" \
      --out aggregate_all.csv
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

def infer_lang_from_path(path: Path) -> str:
    """
    Busca en la ruta un directorio que empiece por 'bench_out_' y devuelve el sufijo como idioma.
    Ej: bench_out_es -> 'es'
    """
    for part in path.parts:
        if part.startswith("bench_out_"):
            return part.split("bench_out_")[-1]
    # fallback: último sufijo tras '_'
    name = path.name
    if "_" in name:
        return name.split("_")[-1]
    return "unknown"

def collect_csvs(roots, pattern):
    files = []
    for root in roots:
        root_p = Path(root)
        if not root_p.exists():
            print(f"[AVISO] No existe la carpeta: {root_p}", file=sys.stderr)
            continue
        files.extend(root_p.rglob(pattern))
    return sorted(files)

def read_csv_safely(fp: Path):
    # intenta con separador por defecto y codificaciones comunes
    encodings = ["utf-8", "utf-8-sig", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(fp, encoding=enc)
            return df
        except Exception as e:
            last_err = e
    raise last_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Carpetas raíz (p.ej. bench_out_ca bench_out_en bench_out_es)")
    ap.add_argument("--pattern", default="*.csv",
                    help="Patrón de búsqueda de ficheros (por defecto: *.csv)")
    ap.add_argument("--out", default="aggregate_all.csv",
                    help="Nombre del CSV de salida")
    ap.add_argument("--keep_source_path", action="store_true",
                    help="Añade columna 'source_path' con la ruta del CSV original")
    args = ap.parse_args()

    files = collect_csvs(args.roots, args.pattern)
    if not files:
        print("[ERROR] No se encontraron CSV con el patrón indicado.", file=sys.stderr)
        sys.exit(2)

    all_rows = []
    for fp in files:
        try:
            df = read_csv_safely(fp)
        except Exception as e:
            print(f"[AVISO] No se pudo leer {fp}: {e}", file=sys.stderr)
            continue

        lang = infer_lang_from_path(fp)
        # Inserta 'db_lang' como PRIMERA columna
        df.insert(0, "db_lang", lang)

        if args.keep_source_path:
            df["source_path"] = str(fp)

        all_rows.append(df)

    if not all_rows:
        print("[ERROR] No se pudo leer ningún CSV.", file=sys.stderr)
        sys.exit(3)

    # Alinea columnas (puede haber CSV con columnas distintas)
    # Esto hace outer concat; columnas ausentes quedarán como NaN.
    merged = pd.concat(all_rows, ignore_index=True, sort=False)

    # Opcional: ordena por db_lang primero
    if "db_lang" in merged.columns:
        merged = merged.sort_values(by=["db_lang"], kind="stable").reset_index(drop=True)

    # Guarda salida
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] Generado: {out_path} ({len(merged)} filas, {len(merged.columns)} columnas)")

if __name__ == "__main__":
    main()
