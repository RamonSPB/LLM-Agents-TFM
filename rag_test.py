from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configurar LLM y embeddings globalmente usando Settings
Settings.llm = Ollama(model="llama3")
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Cargar documentos desde carpeta
documents = SimpleDirectoryReader("simpsons_data").load_data()

# Crear índice directamente (sin ServiceContext)
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="tree_summarize",  # o "refine" si quieres mayor control
    system_prompt="Eres un experto en Los Simpson. Usa el contexto completo para dar una respuesta bien explicada."
)

pregunta = "¿En que episodio Bart corta la cabeza de una estatua?"
respuesta = query_engine.query(pregunta)

print("🧠 Pregunta:", pregunta)
print("\n📚 Respuesta:\n", respuesta.response)

print("\n🔎 Fuentes utilizadas:")
for i, node in enumerate(respuesta.source_nodes):
    texto = node.node.get_text().strip()[:300]
    archivo = node.metadata.get("file_name", "desconocido")
    print(f"\n📄 Fragmento #{i+1} (de {archivo}):")
    print(texto, "...")


