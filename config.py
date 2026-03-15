"""
SIAA — Configuración Central
Todos los parámetros ajustables en un solo lugar.
Importar con: from config import CHUNK_SIZE, MODEL, ...
"""
import os

# ── Versión ──────────────────────────────────────────────────────
VERSION = "2.1.28"

# ── LLM / Ollama ─────────────────────────────────────────────────
OLLAMA_URL             = "http://localhost:11434"
MODEL                  = "qwen2.5:3b"
MAX_OLLAMA_SIMULTANEOS = 2
TIMEOUT_CONEXION       = 8
TIMEOUT_RESPUESTA      = 180
TIMEOUT_HEALTH         = 5

# ── Servidor Flask/Waitress ───────────────────────────────────────
HILOS_SERVIDOR  = 8
SERVIDOR_IP     = os.environ.get("SIAA_SERVER_IP", "")
SERVIDOR_PUERTO = os.environ.get("SIAA_SERVER_PORT", "5000")

# ── Rutas ─────────────────────────────────────────────────────────
CARPETA_FUENTES = "/opt/siaa/fuentes"
LOG_ARCHIVO     = "/opt/siaa/logs/calidad.jsonl"
LOG_MAX_LINEAS  = 5000

# ── RAG / Chunking ────────────────────────────────────────────────
CHUNK_SIZE          = 800
CHUNK_OVERLAP       = 300
MAX_CHUNKS_CONTEXTO = 3
MAX_DOCS_CONTEXTO   = 2

# ── TF-IDF / Keywords ────────────────────────────────────────────
TOP_KEYWORDS_POR_DOC = 20
MIN_FREQ_KEYWORD     = 1
MIN_LEN_KEYWORD      = 3

# ── Caché LRU ────────────────────────────────────────────────────
CACHE_MAX_ENTRADAS = 200
CACHE_TTL_SEGUNDOS = 3600
CACHE_SOLO_DOC     = True
