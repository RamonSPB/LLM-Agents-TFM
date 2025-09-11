# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 17:38:43 2025

@author: Admin
"""
import pandas as pd
import os

# Ruta del archivo original
archivo = "gold_examples_simpsons.csv"

# Carpeta de salida
out_dir = "csv_por_idioma"
os.makedirs(out_dir, exist_ok=True)

# Leer CSV detectando encoding automáticamente
df = pd.read_csv(archivo, encoding="utf-8-sig")

# Verificar si existe la columna 'language'
if "language" not in df.columns:
    raise ValueError("El CSV no tiene una columna 'language' para separar por idioma.")

# Guardar un CSV por idioma
for lang, subset in df.groupby("language"):
    # Resetear índice para que quede consecutivo (1, 2, 3, ...)
    subset = subset.reset_index(drop=True)
    subset.insert(0, "q_index10", subset.index + 1)  # Nueva columna al inicio

    # Nombre de archivo
    nombre_salida = os.path.join(out_dir, f"gold_examples_simpsons_{lang}.csv")
    subset.to_csv(nombre_salida, index=False, encoding="utf-8-sig")

    print(f"✅ Guardado {nombre_salida} con {len(subset)} preguntas.")

print("🎉 Archivos separados por idioma listos.")
