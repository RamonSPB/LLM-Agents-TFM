# -*- coding: utf-8 -*-
import requests
import json

def descargar_modelo(nombre_modelo):
    url = "http://localhost:11434/api/pull"
    headers = {"Content-Type": "application/json"}
    payload = {"name": nombre_modelo}

    print(f"\n⬇️ Verificando modelo '{nombre_modelo}'...")
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        if r.ok:
            # La respuesta tiene varias líneas JSON, tomamos la última válida
            lineas = r.text.strip().split("\n")
            ultimo = json.loads(lineas[-1])
            status = ultimo.get("status", "")
            if status == "success":
                print(f"✅ Modelo '{nombre_modelo}' ya está disponible.")
            else:
                print(f"📦 Estado final: {status}")
        else:
            print(f"❌ Error al descargar el modelo: {r.status_code} {r.text}")
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
        r = requests.post(url, headers=headers, json=payload, timeout=600)
        if r.ok:
            lineas = r.text.strip().split("\n")
            ultimo = json.loads(lineas[-1])
            status = ultimo.get("status", "")
            if status == "success":
                print(f"✅ Embedding '{nombre_embedding}' disponible.")
            else:
                print(f"📦 Estado final: {status}")
        else:
            print(f"❌ Error al descargar embedding: {r.status_code} {r.text}")
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

    try:
        print("🔍 Enviando POST a Ollama...")
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.ok:
            respuesta = ""
            lineas = r.text.strip().split("\n")
            for linea in lineas:
                try:
                    data = json.loads(linea)
                    # Capturar solo los fragmentos de respuesta
                    if "message" in data and "content" in data["message"]:
                        respuesta += data["message"]["content"]
                except json.JSONDecodeError:
                    pass

            return respuesta.strip() if respuesta else "⚠️ No se recibió contenido del modelo."
        else:
            return f"❌ Error {r.status_code}: {r.text}"
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
