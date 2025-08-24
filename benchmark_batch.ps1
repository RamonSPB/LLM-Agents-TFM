param(
  [string]$DataDir = "simpsons_data",
  [string]$Gold = "gold_examples_simpsons.csv",

  # Pásalos como arrays en la llamada:
  [string[]]$Embeds = @("nomic-embed-text","all-minilm"),
  [string[]]$LLMs   = @("llama3","mistral"),
  [string[]]$Langs  = @("es","ca","en"),

  [int]$TopK = 3,

  [switch]$UseChroma = $false,
  [string]$ChromaPath = "./chroma_db",
  [string]$BaseCollection = "bench",

  [string]$OutDir = "./bench_out",

  # NUEVO: si se indica, borra la colección de Chroma por embedding antes de indexar
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

# Borrar colección por embedding (vía API de chroma, no borrar carpetas a mano)
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

  # Colección por embedding (evita mezclar dimensiones: p.ej. 768 vs 384)
  $safeEmbed = ($embed -replace '[^A-Za-z0-9]+','_')
  $collection = "${BaseCollection}_${safeEmbed}"

  # NUEVO: reindex → borrar colección del embedding antes de usarla
  if ($UseChroma -and $Reindex) {
    Write-Host "🧹 Reindex: borrando colección '$collection' en '$ChromaPath'..." -ForegroundColor Yellow
    Remove-ChromaCollection -Path $ChromaPath -Collection $collection
  }

  foreach ($llm in $LLMs) {
    foreach ($lang in $Langs) {

      $outfile = Join-Path $OutDir ("results_{0}_{1}_{2}_k{3}.csv" -f $llm, $safeEmbed, $lang, $TopK)

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
