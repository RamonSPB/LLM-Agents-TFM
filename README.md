# LLM-Agents-TFM
LLM project to generate agents able to do exams and correct them

# 🧠 Proyecto LLM con Ollama y Python 3.8

Este proyecto utiliza modelos de lenguaje (LLM) a través de [Ollama](https://ollama.com/) en combinación con scripts de Python para consultas, pruebas RAG y generación de embeddings, usando `llama-index` y `requests`.

Está diseñado para ser ejecutado de forma sencilla mediante Docker Compose, incluyendo soporte para GPU si está disponible.

# Levantar Docker-Compose

El archivo docker-compose.yml puede levantarse lanzando el siguiente comando desde bash en la ruta del proyecto:

docker compose up -d

Dicho .yml aplica la aceleración por GPU mediante CUDA. En caso de no poder utilizarla se requiere eliminar la siguiente sección del archivo:

deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]

Para apagar el servicio se haría mediante el siguiente comando:

docker compose down

# build_index.py y simpsons_script.py

El archivo simpsons_script.py sirve para generar un dataset al mismo nivel que el proyecto en una carpeta llamada simpsons_data.

El script de build_index.py toma la ruta generada del dataset anterior y genera la base de datos e indexación en ChromaDB.

El efecto de ambos scripts es persistente, se guarda memoria en el docker-compose.yml para tal fin, así la aplicación solo ha de realizarse una vez en el contenedor.
