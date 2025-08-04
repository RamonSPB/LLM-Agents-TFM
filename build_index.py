# -*- coding: utf-8 -*-
"""
Script para crear un índice vectorial persistente de Los Simpson usando ChromaDB y Ollama Embeddings.
Se ejecuta una sola vez para construir la base de datos vectorial.
"""

import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

# 📂 Ruta de datos y base de datos
DATA_PATH = "simpsons_data"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "simpsons_episodes"

def build_index():
    print("📥 Cargando documentos...")
    documents = SimpleDirectoryReader(DATA_PATH).load_data()
    print(f"✅ {len(documents)} documentos cargados.")

    print("🔗 Conectando a ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("🧠 Creando embeddings e índice vectorial...")
    embed_model = OllamaEmbedding(model_name="nomic-embed-text")

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model
    )

    print("💾 Índice creado y almacenado en ChromaDB correctamente.")

if __name__ == "__main__":
    build_index()