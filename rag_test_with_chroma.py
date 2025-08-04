import os
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "simpsons_episodes")

# Conexión a Chroma persistente
chroma_client = chromadb.PersistentClient(path=DB_PATH)
chroma_collection = chroma_client.get_or_create_collection(COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

embed_model = OllamaEmbedding(model_name="nomic-embed-text")
llm = Ollama(model="llama3")

# Cargar índice directamente desde Chroma (sin load_index_from_storage)
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize",
    system_prompt="Eres un experto en Los Simpson. Usa SOLO la información recuperada.",
    llm=llm
)

pregunta = "¿En qué episodio Bart corta la cabeza de una estatua?"
respuesta = query_engine.query(pregunta)

print("🧠 Pregunta:", pregunta)
print("\n📚 Respuesta:\n", respuesta.response)

print("\n🔎 Fuentes utilizadas:")
for i, node in enumerate(respuesta.source_nodes):
    texto = node.node.get_text().strip()[:300]
    archivo = node.metadata.get("file_name", "desconocido")
    print(f"\n📄 Fragmento #{i+1} (de {archivo}):")
    print(texto, "...")
