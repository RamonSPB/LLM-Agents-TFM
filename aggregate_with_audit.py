
#!/usr/bin/env python3
"""
aggregate_with_audit.py

Mejoras clave vs robust_aggregate_bench.py:
1) Soporta un ESQUEMA CANÓNICO opcional (lista de columnas) vía --schema-json o --schema-csv.
   - Si se provee, todas las tablas se reindexan a ese orden y se rellenan faltantes con "".
   - Si no se provee, usa columnas de la PRIMERA tabla leída como canónico y añade extras al final.

2) Emite un informe de auditoría CSV (--audit) con:
   - archivo, n_rows, n_bad_rows_parse, cols_encontradas, cols_faltantes, ejemplo_bad_lines
   - Útil para detectar líneas mal delimitadas/entrecomilladas o filas truncadas.

3) Valida longitudes de fila vs encabezado a bajo nivel con csv.reader para detectar
   parseos incorrectos por comillas/escapes.
"""
import argparse
import csv
import io
import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd

def norm_col(name: str) -> str:
    if name is None:
        return ""
    s = name.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.replace("º", "o").replace("ª", "a")
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s

def sniff_delimiter(sample: bytes) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample.decode("utf-8", errors="ignore"), delimiters=[",",";","\t","|"])
        return dialect.delimiter
    except Exception:
        return ","

def read_csv_any(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        sample = f.read(65536)
    sep = sniff_delimiter(sample)
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
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
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
                # skip corrupt line
                continue
    return pd.DataFrame(rows)

def coerce_str_frame(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        df[c] = df[c].astype(str)
    return df

def low_level_audit_csv(path: Path) -> Tuple[int,int,list,str]:
    """
    Recorre el archivo con csv.reader para contar filas "mal parseadas":
    - Usa delimitador detectado y quotechar="\""
    - Si el número de campos por fila != len(header), marca como "bad"
    Retorna: (n_rows, n_bad, ejemplos_bad_lines (máx 3), delimiter)
    """
    with path.open("rb") as fb:
        sample = fb.read(65536)
    delimiter = sniff_delimiter(sample)

    examples = []
    n_rows = 0
    n_bad = 0
    header_len = None
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar='"', escapechar="\\")
            for row in reader:
                if header_len is None:
                    header_len = len(row)
                else:
                    n_rows += 1
                    if len(row) != header_len:
                        n_bad += 1
                        if len(examples) < 3:
                            examples.append(",".join(row)[:200])
    except Exception:
        # Si no se puede auditar (binario o lo que sea), lo dejamos en 0
        pass

    return n_rows, n_bad, examples, delimiter

def load_schema_from_args(schema_json: Optional[str], schema_csv: Optional[str]) -> Optional[List[str]]:
    if schema_json:
        p = Path(schema_json)
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [norm_col(x) for x in data]
    if schema_csv:
        p = Path(schema_csv)
        import pandas as pd
        df = pd.read_csv(p, nrows=0)
        return [norm_col(c) for c in df.columns.tolist()]
    return None

def aggregate(input_dir: Path, output_csv: Path, output_parquet: Optional[Path], audit_csv: Optional[Path],
              schema_json: Optional[str], schema_csv: Optional[str]) -> None:
    files = []
    for ext in ("*.csv","*.tsv","*.jsonl","*.ndjson"):
        files.extend(input_dir.rglob(ext))

    frames = []
    audit_rows = []

    canonical = load_schema_from_args(schema_json, schema_csv)

    first_cols = None

    for p in sorted(files):
        df = None
        try:
            if p.suffix.lower() in [".csv",".tsv"]:
                df = read_csv_any(p)
            elif p.suffix.lower() in [".jsonl",".ndjson"]:
                df = read_jsonl(p)
            else:
                continue
        except Exception as e:
            continue

        if df is None or df.empty:
            continue

        # normaliza encabezados
        df.columns = [norm_col(c) for c in df.columns]

        # auditoría de líneas "mal parseadas" (solo CSV/TSV)
        n_rows, n_bad, examples, delim = (0,0,[], "")
        if p.suffix.lower() in [".csv",".tsv"]:
            n_rows, n_bad, examples, delim = low_level_audit_csv(p)

        # inicializa canon si no se pasó y es el primer archivo
        if canonical is None and first_cols is None:
            first_cols = df.columns.tolist()

        # aplica schema canónico si existe
        if canonical:
            # añade columnas faltantes
            for c in canonical:
                if c not in df.columns:
                    df[c] = ""
            # preserva extras también (al final)
            extras = [c for c in df.columns if c not in canonical]
            df = df[canonical + extras]
        elif first_cols is not None:
            # usa primer encabezado como base
            extras = [c for c in df.columns if c not in first_cols]
            missing = [c for c in first_cols if c not in df.columns]
            for c in missing:
                df[c] = ""
            df = df[first_cols + extras]

        df["source_file"] = str(p.relative_to(input_dir))
        df = coerce_str_frame(df)
        frames.append(df)

        cols_found = set(df.columns.tolist())
        cols_expected = set(canonical or first_cols or df.columns.tolist())
        missing_cols = sorted([c for c in cols_expected if c not in cols_found])

        audit_rows.append({
            "file": str(p.relative_to(input_dir)),
            "delimiter": delim,
            "n_rows_data": n_rows,
            "n_bad_rows_parse": n_bad,
            "cols_expected_count": len(cols_expected),
            "cols_found_count": len(cols_found),
            "cols_missing": "|".join(missing_cols),
            "bad_examples": " || ".join(examples)
        })

    if not frames:
        # escribir salidas vacías
        pd.DataFrame().to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        if audit_csv:
            pd.DataFrame().to_csv(audit_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
        return

    agg = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # escritura segura
    agg.to_csv(output_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)
    if output_parquet:
        try:
            agg.to_parquet(output_parquet, index=False)
        except Exception:
            pass

    if audit_csv:
        pd.DataFrame(audit_rows).to_csv(audit_csv, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_ALL)

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir","-i", default="bench_out", type=str)
    ap.add_argument("--output","-o", default="bench_out/aggregate.csv", type=str)
    ap.add_argument("--parquet","-p", default=None, type=str)
    ap.add_argument("--audit", default="bench_out/_audit.csv", type=str, help="Ruta del informe de auditoría CSV (o vacío para desactivar)")
    ap.add_argument("--schema-json", default=None, type=str, help="Ruta a JSON con lista de columnas en orden canónico")
    ap.add_argument("--schema-csv", default=None, type=str, help="Ruta a CSV cuya CABECERA define el orden canónico")
    return ap.parse_args(argv)

def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_csv = Path(args.output).expanduser().resolve()
    output_parquet = Path(args.parquet).expanduser().resolve() if args.parquet else None
    audit_csv = Path(args.audit).expanduser().resolve() if args.audit else None

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_parquet:
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
    if audit_csv:
        audit_csv.parent.mkdir(parents=True, exist_ok=True)

    aggregate(input_dir, output_csv, output_parquet, audit_csv, args.schema_json, args.schema_csv)
    print(f"[OK] CSV -> {output_csv}")
    if output_parquet:
        print(f"[OK] Parquet -> {output_parquet}")
    if audit_csv:
        print(f"[OK] Auditoría -> {audit_csv}")

if __name__ == "__main__":
    main()
