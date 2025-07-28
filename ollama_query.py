import requests

def consultar_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False  # <-- esto fuerza a que devuelva la respuesta completa en vez de en trozos
    }

    response = requests.post(url, json=data)
    if response.ok:
        content = response.json()
        print("\n🧠 Respuesta:")
        print(content["response"])
    else:
        print("❌ Error:", response.status_code, response.text)

# Prueba
consultar_ollama("¿Qué es la inteligencia artificial?")