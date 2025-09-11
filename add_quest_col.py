# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 16:16:33 2025

@author: Admin
"""
import pandas as pd

# Leer el CSV original
df = pd.read_csv("aggregate_all.csv", encoding="utf-8")

# Número total de preguntas únicas (1 a 10 en bucle)
num_preguntas = 10

# Generar índice de preguntas: cada 3 filas corresponde a la misma pregunta
df["q_index10"] = [(i // 3) % num_preguntas + 1 for i in range(len(df))]

# Guardar el nuevo CSV
output_file = "aggregate_all_with_questions.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"✅ Archivo guardado en: {output_file}")
