# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 21:03:57 2025

@author: Admin
"""
import requests
import json


def descargar_modelo(nombre_modelo):
    url = "http://localhost:11434/api/pull"
    headers = {"Content-Type": "application/json"}
    payload = {"name": nombre_modelo}

    print(f"\n⬇️ Verificando modelo '{nombre_modelo}'...")
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if status == "success":
                        print(f"✅ Modelo '{nombre_modelo}' ya está disponible.")
                        break
                    else:
                        print(f"📦 {status}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al descargar el modelo: {e}")


# Descargar modelo antes de comenzar el bucle de preguntas
descargar_modelo("llama3")

def descargar_embedding(nombre_embedding):
    url = "http://localhost:11434/api/pull"
    headers = {"Content-Type": "application/json"}
    payload = {"name": nombre_embedding}

    print(f"\n⬇️ Verificando modelo de embeddings '{nombre_embedding}'...")
    try:
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=600) as r:
            for line in r.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if status == "success":
                        print(f"✅ Embedding '{nombre_embedding}' disponible.")
                        break
                    else:
                        print(f"📦 {status}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al descargar embedding: {e}")

descargar_embedding("nomic-embed-text")


def preguntar_a_ollama(pregunta):
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "llama3",
        "messages": [
            {"role": "user", "content": pregunta}
        ]
    }

    respuesta = ""
    try:
        print("🔍 Enviando POST a Ollama...")
        with requests.post(url, headers=headers, json=payload, stream=True, timeout=30) as r:
            print("📡 Respuesta recibida. Procesando...")
            for line in r.iter_lines():
                if line:                    
                    data = json.loads(line)
                    mensaje = data.get("message", {}).get("content", "")
                    respuesta += mensaje
        return respuesta

    except requests.exceptions.RequestException as e:
        return f"❌ Error de conexión con Ollama: {e}"

# Programa principal
if __name__ == "__main__":
    while True:
        pregunta = input("\n🧠 Pregunta a Ollama (o escribe 'salir'): ")
        if pregunta.lower() in ["salir", "exit", "quit"]:
            print("👋 Hasta luego.")
            break

        respuesta = preguntar_a_ollama(pregunta)
        print("\n💬 Respuesta:")
        print(respuesta)

