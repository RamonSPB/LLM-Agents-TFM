param(
  # Carpeta base donde viven gold y colecciones por idioma
  [string]$BaseDir = "./data",

  # Prefijos/convención de nombres por idioma (sin tocar Python)
  [string]$GoldPrefix = "gold_examples_simpsons_",
  [string]$GoldExt    = ".csv",
  [string]$CollectionPrefix = "simpsons_data_",

  # Modelos a evaluar
  [string[]]$Embeds = @("nomic-embed-text","all-minilm","mxbai-embed-large"),
  [string[]]$LLMs   = @("llama3","mistral","deepseek-r1"),

  # Idiomas a iterar
  [string[]]$Langs  = @("es","ca","en"),

  [int]$TopK = 3,

  # Chroma (opcional)
  [switch]$UseChroma = $false,
  [string]$ChromaPath = "./chroma_db",
  # Base para el nombre de colección; luego añadimos _{embedSanitizado}_{lang}
  [string]$BaseCollection = "bench",

  # Salida
  [string]$OutDir = "./bench_out",

  # Si se indica, borra la colección Chroma antes de indexar (por embedding+idioma)
  [switch]$Reindex = $false
)

# Ejecutar desde la carpeta del script
Set-Location -Path $PSScriptRoot

# Python del venv (PS 5.1 compatible)
$py = "python"
$venv1 = Join-Path $PSScriptRoot "mi_entorno_rag\Scripts\python.exe"
$venv2 = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venv1) { $py = $venv1 } elseif (Test-Path $venv2) { $py = $venv2 }

# Salida
if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir | Out-Null }

# Pull automático (opcional)
function Ensure-OllamaModel {
  param([string]$Name)
  try { ollama pull $Name | Out-Null } catch {
    try {
      $uri = "http://localhost:11434/api/pull"
      $body = @{ name = $Name } | ConvertTo-Json
      Invoke-WebRequest -Method POST -Uri $uri -Body $body -ContentType "application/json" -TimeoutSec 600 -UseBasicParsing | Out-Null
    } catch {}
  }
}

# Borrar colección por embedding+idioma (vía API Chroma)
function Remove-ChromaCollection {
  param([string]$Path,[string]$Collection)
  $code = @"
import chromadb
client = chromadb.PersistentClient(path=r'$Path')
try:
    client.delete_collection('$Collection')
    print('deleted:$Collection')
except Exception as e:
    print('skip:', e)
"@
  & $py -c $code | Out-Null
}

# Asegura primero embeddings y luego llms
foreach ($emb in ($Embeds | Select-Object -Unique)) { Write-Host "⬇️  $emb"; Ensure-OllamaModel $emb }
foreach ($llm in ($LLMs   | Select-Object -Unique)) { Write-Host "⬇️  $llm"; Ensure-OllamaModel $llm }

# ======= ORDEN ÓPTIMO: EMBEDDING -> MODELO -> LENGUAJE =======
foreach ($embed in $Embeds) {

  # Evita mezclar dimensiones: colección separada por embedding y por idioma
  $safeEmbed = ($embed -replace '[^A-Za-z0-9]+','_')

  foreach ($llm in $LLMs) {
    foreach ($lang in $Langs) {

      # Rutas por idioma (sin tocar Python):
      $DataDir = Join-Path $BaseDir ("{0}{1}" -f $CollectionPrefix, $lang)   # ./data/simpsons_data_es
      $Gold    = Join-Path $BaseDir ("{0}{1}{2}" -f $GoldPrefix, $lang, $GoldExt)  # ./data/gold_test_es.csv

      # Nombre de colección por embedding+idioma (si usas Chroma)
      $collection = "{0}_{1}_{2}" -f $BaseCollection, $safeEmbed, $lang

      # (Re)indexación: borra la colección antes si lo pides
      if ($UseChroma -and $Reindex) {
        Write-Host "🧹 Reindex: borrando colección '$collection' en '$ChromaPath'..." -ForegroundColor Yellow
        Remove-ChromaCollection -Path $ChromaPath -Collection $collection
      }

      # Archivo de salida (por combinación)
      $outfile = Join-Path $OutDir ("results_{0}_{1}_{2}_k{3}.csv" -f $llm, $safeEmbed, $lang, $TopK)

      # Argumentos para el script Python (tal cual está ahora)
      $argsList = @(
        "benchmark_rag_tokenized.py",
        "--data_dir", $DataDir,
        "--gold", $Gold,
        "--llm", $llm,
        "--embed", $embed,
        "--top_k", $TopK,
        "--lang", $lang,
        "--output", $outfile
      )

      if ($UseChroma) {
        $argsList += @("--use_chroma", "--chroma_path", $ChromaPath, "--collection", $collection)
      }

      Write-Host "`n🚀 Ejecutando: $py $($argsList -join ' ')" -ForegroundColor Cyan
      & $py $argsList
      if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Falló EMBED=$embed LLM=$llm LANG=$lang" -ForegroundColor Red
      } else {
        Write-Host "✅ OK → $outfile" -ForegroundColor Green
      }
    }
  }
}

Write-Host "`n🎉 Lote finalizado. CSVs en: $OutDir" -ForegroundColor Green
