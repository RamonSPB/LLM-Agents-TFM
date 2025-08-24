@echo off
echo 🔧 Creando entorno virtual llamado "mi_entorno_rag"...
python -m venv mi_entorno_rag

echo ✅ Entorno creado.
echo 🔄 Activando entorno...
call mi_entorno_rag\Scripts\activate

echo 📦 Instalando dependencias necesarias...
pip install --upgrade pip

pip install ^
    requests ^
    pandas ^
    llama-index ^
    llama-index-llms-ollama ^
    llama-index-embeddings-ollama ^
    PyMuPDF ^
    python-docx

echo 🟢 Entorno y dependencias instaladas correctamente.
pause
