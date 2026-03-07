"""
╔══════════════════════════════════════════════════════════════════╗
║         SIAA — Proxy Inteligente  v2.1.6                        ║
║         Seccional Bucaramanga — Rama Judicial                    ║
╠══════════════════════════════════════════════════════════════════╣
║  CORRECCIONES v2.1.6 (sobre v2.1.5)                              ║
║                                                                   ║
║  [FIX-1] Códigos alfanuméricos en índice de densidad            ║
║      tokenizar() ya no filtra tokens con dígitos si tienen       ║
║      letras mezcladas (psaa16, pcsja19, art5, ley270...).        ║
║      Solo filtra secuencias puramente numéricas cortas (<4 dig). ║
║      Impacto: "psaa16" y "10476" ahora tienen densidad en        ║
║      acuerdo_no._psaa16-10476.md → enrutador lo sube al top.     ║
║                                                                   ║
║  [FIX-2] Bonus de artículo aumentado y con patrón exacto        ║
║      Antes: +3 por cualquier "artículo N"                        ║
║      Ahora: +10 si el párrafo contiene el artículo con grado    ║
║             (artículo 5°, art. 5°, artículo 5o)                  ║
║             +5  si contiene "artículo N" sin grado               ║
║                                                                   ║
║  [FIX-3] Chunking con solapamiento (sliding window)             ║
║      Reemplaza el split por \n+ con chunks de tamaño fijo        ║
║      (CHUNK_SIZE chars) y solapamiento (CHUNK_OVERLAP chars).    ║
║      Evita que pasos de procedimientos queden partidos.          ║
║      Los chunks preservan el contexto del encabezado anterior.   ║
║                                                                   ║
║  [FIX-4] Números largos (>=4 dígitos) en índice                 ║
║      "10476" ahora entra al índice de densidad como token        ║
║      independiente porque tiene 5 dígitos.                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, re, json, math, threading, time, requests
from collections import defaultdict, Counter
from flask import Flask, request, Response, stream_with_context, jsonify
from flask_cors import CORS

# ================================================================
#  CACHÉ LRU — Problema D
#
#  Almacena respuestas completas de preguntas documentales frecuentes.
#  Para saludos y conversación general NO se cachea (son únicos cada vez).
#
#  DISEÑO:
#    - Clave: hash SHA256 de la pregunta normalizada (lowercase, sin puntuación)
#    - Valor: texto completo de la respuesta + cita fuente
#    - TTL: 3600 segundos (1 hora) — después de ese tiempo se considera stale
#    - Máximo: 200 entradas (LRU desaloja la menos usada recientemente)
#    - Thread-safe: protegido con Lock
#
#  IMPACTO ESPERADO:
#    - Hit de caché: respuesta en ~5ms vs 44s sin caché (8800x más rápido)
#    - Los 26 despachos hacen preguntas similares → hit rate estimado 30-40%
#    - Sin caché: cada consulta gasta RAM y CPU del modelo
#    - Con caché: 2da consulta idéntica cuesta 0 recursos de IA
# ================================================================

import hashlib
from collections import OrderedDict

CACHE_MAX_ENTRADAS = 200    # Máximo de respuestas almacenadas
CACHE_TTL_SEGUNDOS = 3600   # 1 hora — tiempo de vida por entrada
CACHE_SOLO_DOC     = True   # Solo cachear consultas documentales (no saludos)

# Estructura de cada entrada del caché:
# { "respuesta": str, "cita": str, "ts": float, "hits": int }
_cache_respuestas = OrderedDict()
_cache_lock       = threading.Lock()
_cache_hits       = 0
_cache_misses     = 0


def _clave_cache(texto: str) -> str:
    """
    Genera clave de caché normalizada — insensible a tildes, puntuación y mayúsculas.
    "¿Cuándo debo reportar?" == "cuando debo reportar" == "CUANDO DEBO REPORTAR"
    """
    import unicodedata
    t = texto.lower()
    t = re.sub(r'[^\w\s]', '', t)
    # Eliminar tildes: "cuándo" → "cuando", "información" → "informacion"
    t = ''.join(c for c in unicodedata.normalize('NFD', t)
                if unicodedata.category(c) != 'Mn')
    t = re.sub(r'\s+', ' ', t).strip()
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def cache_get(pregunta: str) -> dict | None:
    """
    Busca la pregunta en el caché.

    Returns:
        dict con {respuesta, cita} si hay hit válido, None si miss o expirado.
    """
    global _cache_hits, _cache_misses
    clave = _clave_cache(pregunta)

    with _cache_lock:
        if clave not in _cache_respuestas:
            _cache_misses += 1
            return None

        entrada = _cache_respuestas[clave]

        # Verificar TTL
        if time.time() - entrada["ts"] > CACHE_TTL_SEGUNDOS:
            del _cache_respuestas[clave]
            _cache_misses += 1
            return None

        # HIT — mover al final (LRU: más recientemente usado)
        _cache_respuestas.move_to_end(clave)
        entrada["hits"] += 1
        _cache_hits += 1
        return {"respuesta": entrada["respuesta"], "cita": entrada["cita"]}


def cache_set(pregunta: str, respuesta: str, cita: str):
    """
    Guarda una respuesta en el caché.
    Si el caché está lleno, desaloja la entrada menos usada (la del frente).
    """
    if not respuesta.strip():
        return  # No cachear respuestas vacías

    # No cachear respuestas de "no encontré" — son negativas y pueden cambiar
    if "no encontré esa información" in respuesta.lower():
        return

    clave = _clave_cache(pregunta)
    with _cache_lock:
        # Si ya existe, actualizar
        if clave in _cache_respuestas:
            _cache_respuestas.move_to_end(clave)
            _cache_respuestas[clave].update({"respuesta": respuesta, "cita": cita, "ts": time.time()})
            return

        # Si está lleno, desalojar el más antiguo (frente del OrderedDict)
        while len(_cache_respuestas) >= CACHE_MAX_ENTRADAS:
            _cache_respuestas.popitem(last=False)

        _cache_respuestas[clave] = {
            "respuesta": respuesta,
            "cita":      cita,
            "ts":        time.time(),
            "hits":      0,
        }


def cache_stats() -> dict:
    """Estadísticas del caché para el endpoint /siaa/status."""
    with _cache_lock:
        total   = _cache_hits + _cache_misses
        hit_rate = round(_cache_hits / total * 100, 1) if total > 0 else 0
        return {
            "entradas":  len(_cache_respuestas),
            "max":       CACHE_MAX_ENTRADAS,
            "hits":      _cache_hits,
            "misses":    _cache_misses,
            "hit_rate":  f"{hit_rate}%",
            "ttl_seg":   CACHE_TTL_SEGUNDOS,
        }


app = Flask(__name__)
CORS(app)

# ================================================================
#  CONFIGURACIÓN CENTRAL
# ================================================================

OLLAMA_URL             = "http://localhost:11434"
MODEL                  = "qwen2.5:3b"
VERSION                = "2.1.19"

# ================================================================
#  SISTEMA DE MONITOREO DE CALIDAD — Problema F
#
#  Registra cada consulta en un archivo JSONL (una línea JSON por consulta).
#  JSONL permite análisis con cualquier herramienta: grep, jq, Python, Excel.
#
#  Cada registro contiene:
#    - timestamp, tipo (CONV/DOC/CACHE_HIT), pregunta
#    - documentos usados, chars de contexto
#    - primeros 300 chars de la respuesta
#    - tiempo de respuesta en segundos
#    - indicador de posible alucinación (respuesta "No encontré" con docs)
#
#  El administrador puede revisar el log con:
#    curl http://localhost:5000/siaa/log             ← últimas 50 entradas
#    curl http://localhost:5000/siaa/log?n=100       ← últimas 100
#    curl http://localhost:5000/siaa/log?tipo=ERROR  ← solo errores
# ================================================================

LOG_ARCHIVO    = "/opt/siaa/logs/calidad.jsonl"   # Una línea JSON por consulta
LOG_MAX_LINEAS = 5000   # Rotar al llegar a 5000 entradas (~2MB)
_log_lock      = threading.Lock()


def _asegurar_carpeta_log():
    """Crea la carpeta de logs si no existe."""
    os.makedirs(os.path.dirname(LOG_ARCHIVO), exist_ok=True)


def registrar_consulta(
    tipo: str,          # "CONV", "DOC", "CACHE_HIT", "ERROR"
    pregunta: str,
    respuesta: str,
    docs: list,
    ctx_chars: int,
    tiempo_seg: float,
    cache_hit: bool = False,
):
    """
    Escribe una línea JSONL en el archivo de log de calidad.

    Detecta automáticamente posibles problemas:
      - POSIBLE_ALUCINACION: el modelo respondió "No encontré" pero SÍ había
        documentos relevantes (el extractor encontró contexto pero el modelo
        lo ignoró o el contexto era incorrecto).
      - SIN_CONTEXTO: pregunta documental sin documentos encontrados.
    """
    try:
        _asegurar_carpeta_log()

        # Detección automática de problemas
        no_encontro  = "no encontré esa información" in respuesta.lower()
        habia_docs   = len(docs) > 0 and ctx_chars > 100

        if no_encontro and habia_docs:
            alerta = "POSIBLE_ALUCINACION"   # Tenía docs pero dijo que no encontró
        elif no_encontro and not habia_docs:
            alerta = "SIN_CONTEXTO"           # No había docs — correcto decir "no encontré"
        elif tipo == "ERROR":
            alerta = "ERROR"
        else:
            alerta = "OK"

        entrada = {
            "ts":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tipo":      "CACHE_HIT" if cache_hit else tipo,
            "alerta":    alerta,
            "pregunta":  pregunta[:200],
            "respuesta": respuesta[:300],
            "docs":      docs,
            "ctx_chars": ctx_chars,
            "tiempo_s":  round(tiempo_seg, 2),
        }

        with _log_lock:
            # Rotar si excede el máximo
            try:
                with open(LOG_ARCHIVO, "r", encoding="utf-8") as f:
                    lineas = f.readlines()
                if len(lineas) >= LOG_MAX_LINEAS:
                    # Conservar las últimas 4000 líneas
                    with open(LOG_ARCHIVO, "w", encoding="utf-8") as f:
                        f.writelines(lineas[-4000:])
            except FileNotFoundError:
                pass  # Primera escritura

            with open(LOG_ARCHIVO, "a", encoding="utf-8") as f:
                f.write(json.dumps(entrada, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[LOG] Error escribiendo log: {e}", flush=True)


MAX_OLLAMA_SIMULTANEOS = 2
HILOS_SERVIDOR         = 16
TIMEOUT_CONEXION       = 8
TIMEOUT_RESPUESTA      = 180
TIMEOUT_HEALTH         = 5
CARPETA_FUENTES        = "/opt/siaa/fuentes"
MAX_DOCS_CONTEXTO      = 2    # [v2.1.8] 2×3×800=4800c≈1200tok — cabe en num_ctx=2048
TOP_KEYWORDS_POR_DOC   = 20
MIN_FREQ_KEYWORD       = 2
MIN_LEN_KEYWORD        = 3

# ── Parámetros de chunking ───────────────────────────────────────
# [FIX-3] Tamaño de chunk y solapamiento
# CHUNK_SIZE:    máximo de chars por chunk
# CHUNK_OVERLAP: chars compartidos entre chunks consecutivos
#                evita que un artículo quede partido entre dos chunks
# MAX_CHUNKS_CONTEXTO: cuántos chunks enviar al modelo por documento
CHUNK_SIZE             = 800
CHUNK_OVERLAP = 300   # [v2.1.15]
MAX_CHUNKS_CONTEXTO    = 3   # [v2.1.8] 3 × 800 = 2400 chars máx por doc

# ================================================================
#  PATRONES DE DOCUMENTOS ESPECÍFICOS
# ================================================================

PATRON_DOC_ESPECIFICO = re.compile(
    r'\b(psaa|pcsja|acuerdo|circular|resolución|resolucion|decreto)\s*'
    r'[\w\-\.]+',
    re.IGNORECASE
)


def detectar_doc_especifico(pregunta: str) -> bool:
    return bool(PATRON_DOC_ESPECIFICO.search(pregunta))


# ================================================================
#  SYSTEM PROMPTS
# ================================================================

SYSTEM_CONVERSACIONAL = """Eres SIAA (Sistema Inteligente de Apoyo Administrativo), el asistente oficial de la Seccional Bucaramanga de la Rama Judicial de Colombia.

SIAA significa exactamente: "Sistema Inteligente de Apoyo Administrativo". No significa nada más.

Responde con cordialidad en español formal.
Para saludos y preguntas generales sobre ti mismo, responde directamente.
Recuerda que puedes ayudar con consultas sobre procesos judiciales, administrativos y normativos."""

SYSTEM_DOCUMENTAL = """Eres SIAA, asistente judicial de la Seccional Bucaramanga.
Responde SOLO con información de los bloques [DOC:...]. No inventes nada externo.
Si el contexto tiene fechas, artículos, roles o sanciones, cítalos literalmente.
Si un rol preguntado no está en la lista del contexto, di cuáles roles sí existen.
Si la información no está en el contexto, responde: "No encontré esa información en los documentos disponibles."
Español formal. Máximo 4 líneas."""

PATRONES_CONVERSACION = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "buen día",
    "buenas", "hey", "saludos", "qué tal", "como estas", "cómo estás",
    "adiós", "adios", "hasta luego", "chao", "nos vemos",
    "gracias", "muchas gracias", "mil gracias", "muy amable",
    "de nada", "con gusto", "a la orden",
    "quién eres", "quien eres", "qué es siaa", "que es siaa",
    "qué significa siaa", "que significa siaa",
    "para qué sirves", "para que sirves",
    "ok", "bien", "entendido", "de acuerdo", "claro", "perfecto", "listo",
]


def es_conversacion_general(texto: str) -> bool:
    t = texto.lower().strip()
    if len(t) < 15:
        return True
    return any(p in t for p in PATRONES_CONVERSACION)


# ================================================================
#  MONITOR OLLAMA + WARM-UP
# ================================================================

ollama_estado = {
    "disponible":   False,
    "ultimo_check": 0,
    "fallos":       0,
    "warmup_done":  None
}
ollama_lock = threading.Lock()


def verificar_ollama() -> bool:
    try:
        r  = requests.get(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT_HEALTH)
        ok = (r.status_code == 200)
    except Exception:
        ok = False

    with ollama_lock:
        ollama_estado["disponible"]   = ok
        ollama_estado["ultimo_check"] = time.time()
        ollama_estado["fallos"]       = 0 if ok else ollama_estado["fallos"] + 1
        warmup_pendiente = ok and ollama_estado["warmup_done"] is None

    if warmup_pendiente:
        try:
            print(f"  [Ollama] Precargando {MODEL} en RAM...", flush=True)
            requests.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model":    MODEL,
                    "messages": [{"role": "user", "content": "ok"}],
                    "stream":   False,
                    "options":  {"num_predict": 1, "num_ctx": 64}
                },
                timeout=(10, 35)
            )
            with ollama_lock:
                ollama_estado["warmup_done"] = True
            print(f"  [Ollama] {MODEL} listo en RAM ✓", flush=True)
        except Exception as e:
            print(f"  [Ollama] Warm-up falló: {e}", flush=True)
            with ollama_lock:
                ollama_estado["warmup_done"] = False

    return ok


def _monitor_loop():
    while True:
        verificar_ollama()
        time.sleep(15)


threading.Thread(target=_monitor_loop, daemon=True).start()

ollama_semaforo  = threading.Semaphore(MAX_OLLAMA_SIMULTANEOS)
usuarios_activos = 0
total_atendidos  = 0
contadores_lock  = threading.Lock()


def inc_activos():
    global usuarios_activos, total_atendidos
    with contadores_lock:
        usuarios_activos += 1
        total_atendidos  += 1


def dec_activos():
    global usuarios_activos
    with contadores_lock:
        usuarios_activos = max(0, usuarios_activos - 1)


# ================================================================
#  [FIX-1 + FIX-4] TOKENIZADOR MEJORADO
#
#  REGLAS:
#    - Tokens alfanuméricos con letras: SIEMPRE incluir (psaa16, art5)
#    - Tokens solo dígitos con 4+ cifras: incluir (10476, 2016)
#    - Tokens solo dígitos con <4 cifras: descartar (1, 22, 999)
#    - Stopwords: descartar
#
#  Ejemplos:
#    "psaa16"  → INCLUIR  (tiene letras)
#    "10476"   → INCLUIR  (5 dígitos)
#    "2016"    → INCLUIR  (4 dígitos — año relevante)
#    "art5"    → INCLUIR  (tiene letras)
#    "42"      → descartar (solo 2 dígitos)
# ================================================================

STOPWORDS_ES = {
    "para", "como", "este", "esta", "esto", "estos", "estas", "pero",
    "más", "también", "cuando", "donde", "porque", "aunque", "sino",
    "desde", "hasta", "entre", "sobre", "bajo", "ante", "tras",
    "dentro", "fuera", "hacia", "según", "durante", "mediante",
    "cada", "todo", "toda", "todos", "todas", "otro", "otra", "otros",
    "dicho", "dicha", "mismo", "misma", "algún", "alguna", "ningún",
    "debe", "puede", "tiene", "hace", "será", "sido", "estar", "tener",
    "hacer", "poder", "deber", "haber", "caso", "forma", "parte",
    "tipo", "modo", "manera", "través", "respecto", "relación",
}


def tokenizar(texto: str) -> list:
    """
    Tokenizador alfanumérico con reglas de filtrado inteligentes.

    [FIX-1] Incluye tokens con letras+dígitos (psaa16, pcsja19, art5).
    [FIX-4] Incluye números puros de 4+ dígitos (10476, 2016, 1096).
    """
    # Capturar tokens alfanuméricos (letras, dígitos, tildes)
    tokens_raw = re.findall(r'\b[a-záéíóúüñ0-9]{3,}\b', texto.lower())

    resultado = []
    for p in tokens_raw:
        if p in STOPWORDS_ES:
            continue
        es_solo_digitos = p.isdigit()
        if es_solo_digitos:
            # Solo incluir números con 4+ dígitos (años, códigos largos)
            if len(p) >= 4:
                resultado.append(p)
            # Descartar números cortos (1, 22, 999)
        else:
            # Token con letras (con o sin dígitos): incluir siempre
            resultado.append(p)

    return resultado


def calcular_tfidf_coleccion(documentos: dict) -> dict:
    if not documentos:
        return {}
    tokens_por_doc = {n: tokenizar(d["contenido"]) for n, d in documentos.items()}
    N  = len(documentos)
    df = defaultdict(int)
    for tokens in tokens_por_doc.values():
        for t in set(tokens):
            df[t] += 1
    keywords_por_doc = {}
    for nombre, tokens in tokens_por_doc.items():
        if not tokens:
            keywords_por_doc[nombre] = []
            continue
        conteo       = Counter(tokens)
        total_tokens = len(tokens)
        scores       = {}
        for termino, freq in conteo.items():
            if freq < MIN_FREQ_KEYWORD or len(termino) < MIN_LEN_KEYWORD:
                continue
            tf  = freq / total_tokens
            idf = math.log((N + 1) / (df[termino] + 1)) + 1
            scores[termino] = tf * idf
        top = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        keywords_por_doc[nombre] = top[:TOP_KEYWORDS_POR_DOC]
    return keywords_por_doc


# ================================================================
#  [FIX-3] CHUNKING CON SOLAPAMIENTO
#
#  Divide el documento en chunks de CHUNK_SIZE chars con
#  CHUNK_OVERLAP chars de solapamiento entre chunks consecutivos.
#
#  Ventajas sobre split por \n+:
#    - Un artículo nunca queda partido entre dos chunks
#    - Los pasos de procedimientos siempre están juntos
#    - El contexto de encabezados se preserva en el overlap
#
#  Cada chunk recuerda el último encabezado visto antes de su inicio
#  para que la cita de sección sea correcta.
# ================================================================

def _ultimo_encabezado(texto: str) -> str:
    """Encuentra el último encabezado Markdown en un texto."""
    encabezados = re.findall(r'^#{1,3}\s+(.+)$', texto, re.MULTILINE)
    if encabezados:
        return re.sub(r'[*_`]', '', encabezados[-1]).strip().upper()
    return "INICIO"


def chunking_con_solapamiento(contenido: str) -> list:
    """
    Divide el contenido en chunks de tamaño fijo con solapamiento.

    Returns:
        Lista de dicts: {"texto": str, "seccion": str, "indice": int}
    """
    chunks  = []
    inicio  = 0
    total   = len(contenido)
    idx     = 0

    while inicio < total:
        fin  = min(inicio + CHUNK_SIZE, total)

        # Extender hasta el próximo salto de línea para no cortar palabras
        if fin < total:
            salto = contenido.find('\n', fin)
            if salto != -1 and salto - fin < 100:
                fin = salto

        texto_chunk = contenido[inicio:fin]

        # Determinar la sección activa usando el texto previo al chunk
        contexto_previo = contenido[max(0, inicio - 500):inicio]
        seccion = _ultimo_encabezado(contexto_previo + texto_chunk)

        chunks.append({
            "texto":   texto_chunk,
            "seccion": seccion,
            "indice":  idx,
        })

        idx    += 1
        inicio += CHUNK_SIZE - CHUNK_OVERLAP   # Avanzar con solapamiento

    return chunks


# ================================================================
#  GESTOR DE COLECCIONES
# ================================================================

colecciones          = {}
colecciones_lock     = threading.Lock()
documentos_cargados  = {}
documentos_lock      = threading.Lock()
indice_densidad      = {}
indice_densidad_lock = threading.Lock()
# Nuevo: índice de chunks pre-calculados por documento
chunks_por_doc       = {}
chunks_lock          = threading.Lock()


def _tokens_nombre_archivo(nombre_clave: str) -> set:
    sin_ext = os.path.splitext(nombre_clave)[0]
    partes  = re.split(r'[_\s\-\.]+', sin_ext.lower())
    # [FIX-1] incluir partes que tengan dígitos (psaa16, 10476)
    return {p for p in partes if len(p) >= 3}


def cargar_documentos():
    global documentos_cargados, colecciones, indice_densidad, chunks_por_doc

    if not os.path.exists(CARPETA_FUENTES):
        os.makedirs(CARPETA_FUENTES, exist_ok=True)
        return

    nuevas_colecciones = {}
    todos_los_docs     = {}
    nuevos_chunks      = {}

    subcarpetas = [("general", CARPETA_FUENTES)]
    try:
        for entrada in os.scandir(CARPETA_FUENTES):
            if entrada.is_dir():
                subcarpetas.append((entrada.name.lower(), entrada.path))
    except Exception as e:
        print(f"  [Docs] Error escaneando: {e}")

    for nombre_col, ruta_col in subcarpetas:
        try:
            archivos = [
                os.path.join(ruta_col, f)
                for f in os.listdir(ruta_col)
                if os.path.isfile(os.path.join(ruta_col, f))
                and f.lower().endswith(('.md', '.txt'))
            ]
        except Exception as e:
            print(f"  [Docs] Error listando {ruta_col}: {e}")
            continue

        if not archivos:
            continue

        docs_col = {}
        for ruta in archivos:
            try:
                nombre_original = os.path.basename(ruta)
                nombre_clave    = nombre_original.lower()

                with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
                    contenido = f.read()

                palabras     = set(tokenizar(contenido))
                tokens       = tokenizar(contenido)
                token_count  = Counter(tokens)
                total_tokens = len(tokens)

                # [FIX-3] Pre-calcular chunks con solapamiento
                chunks = chunking_con_solapamiento(contenido)
                nuevos_chunks[nombre_clave] = chunks

                doc_entry = {
                    "ruta":            ruta,
                    "nombre_original": nombre_original,
                    "contenido":       contenido,
                    "palabras":        palabras,
                    "tamano":          len(contenido),
                    "coleccion":       nombre_col,
                    "token_count":     token_count,
                    "total_tokens":    total_tokens,
                    "tokens_nombre":   _tokens_nombre_archivo(nombre_clave),
                    "num_chunks":      len(chunks),
                }
                docs_col[nombre_clave]       = doc_entry
                todos_los_docs[nombre_clave] = doc_entry
                print(
                    f"  [Doc] [{nombre_col}] {nombre_original} "
                    f"({len(contenido):,} chars, {len(chunks)} chunks)"
                )

            except Exception as e:
                print(f"  [Doc] Error leyendo {ruta}: {e}")

        if not docs_col:
            continue

        print(f"  [KW]  TF-IDF '{nombre_col}' ({len(docs_col)} docs)...")
        keywords_col = calcular_tfidf_coleccion(docs_col)

        for nd, kws in list(keywords_col.items())[:2]:
            print(f"  [KW]  {nd}: {kws[:6]}")

        nuevas_colecciones[nombre_col] = {
            "docs":     docs_col,
            "keywords": keywords_col,
        }

    # Índice de densidad con tokens alfanuméricos
    nuevo_indice = defaultdict(list)
    for nombre_doc, doc in todos_los_docs.items():
        total = doc["total_tokens"]
        if total == 0:
            continue
        for termino, freq in doc["token_count"].items():
            if len(termino) >= MIN_LEN_KEYWORD:
                nuevo_indice[termino].append((freq / total, nombre_doc))

    for t in nuevo_indice:
        nuevo_indice[t].sort(reverse=True)

    with colecciones_lock:
        colecciones = nuevas_colecciones
    with documentos_lock:
        documentos_cargados = todos_los_docs
    with indice_densidad_lock:
        indice_densidad = dict(nuevo_indice)
    with chunks_lock:
        chunks_por_doc = nuevos_chunks

    print(f"  [Doc] Total: {len(todos_los_docs)} docs en {len(nuevas_colecciones)} colecciones ✓")
    print(f"  [IDX] Índice densidad: {len(nuevo_indice):,} términos ✓")
    print(f"  [CHK] Chunks pre-calculados: {sum(len(v) for v in nuevos_chunks.values())} total ✓")


# ================================================================
#  ENRUTADOR MULTI-NIVEL
# ================================================================

def detectar_documentos(pregunta: str, max_docs: int = MAX_DOCS_CONTEXTO) -> list:
    p = pregunta.lower()

    # [FIX-1] palabras alfanuméricas incluyendo números largos
    palabras_pregunta  = set(tokenizar(p))
    palabras_3plus     = set(re.findall(r'\b[a-záéíóúüñ0-9]{3,}\b', p))
    palabras_filtradas = [w for w in palabras_pregunta if w not in STOPWORDS_ES]

    with colecciones_lock:
        snap_cols = dict(colecciones)
    with documentos_lock:
        snap_docs = dict(documentos_cargados)
    with indice_densidad_lock:
        snap_idx = dict(indice_densidad)

    N = len(snap_docs) or 1

    # [v2.1.14] Keywords manuales fijas — complementan las auto-generadas por TF-IDF
    # Útil para términos específicos que no alcanzan top-TF-IDF por baja frecuencia
    KEYWORDS_MANUALES = {
        "acuerdo_pcsja19-11207..md": [
            "capacitacion", "capacita", "capacitar", "quien capacita",
            "cendoj", "udae", "unidad de desarrollo", "analisis estadistico",
            "presentacion de informes", "primer informe",
        ],
        # psaa16: fuente principal de reglas del SIERJU
        # [v2.1.15] Agregar sanciones, roles y responsabilidades
        "acuerdo_no._psaa16-10476.md": [
            # Periodicidad y plazos
            "sierju", "periodicidad", "reportar", "quinto dia habil",
            "formulario", "recoleccion", "discrepancia",
            # Roles (artículo 7)
            "roles", "rol", "super administrador", "administrador nacional",
            "administrador seccional", "funcionario", "juez", "magistrado",
            "quien puede", "puede cargar", "asistente",
            # Responsabilidad (artículo 5)
            "responsable", "quien carga", "quien diligencia",
            "cargar informacion", "diligenciar formulario",
            # Sanciones (artículos 19 y 20)
            "sancion", "sanción", "incumplimiento", "no reporto", "qué pasa",
            "que pasa", "consecuencia", "disciplinario", "disciplinaria",
            "no reportar", "a tiempo", "fuera de tiempo", "castigo",
        ],
        # acuerdo_no__psaa16-10476.md es el mismo con guiones
        "acuerdo_no__psaa16-10476.md": [
            "sierju", "periodicidad", "reportar", "quinto dia habil",
            "formulario", "recoleccion", "discrepancia",
            "roles", "rol", "funcionario", "juez", "magistrado",
            "responsable", "quien carga", "cargar informacion",
            "sancion", "sanción", "incumplimiento", "no reporto", "qué pasa",
            "que pasa", "consecuencia", "disciplinario", "no reportar",
        ],
    }

    # Nivel 1: TF-IDF (auto-generado) + keywords manuales
    scores_tfidf = defaultdict(float)
    for col in snap_cols.values():
        for nombre_doc, keywords in col.get("keywords", {}).items():
            if nombre_doc not in snap_docs:
                continue
            for kw in keywords:
                if kw in p:
                    scores_tfidf[nombre_doc] += 1.0

    # Bonus por keywords manuales (peso x2 — son más específicas)
    for nombre_doc, kws_manuales in KEYWORDS_MANUALES.items():
        if nombre_doc in snap_docs:
            for kw in kws_manuales:
                if kw in p:
                    scores_tfidf[nombre_doc] += 2.0

    # Nivel 2: Densidad
    scores_densidad = defaultdict(float)
    for termino in palabras_filtradas:
        if termino not in snap_idx:
            continue
        df_t      = len(snap_idx[termino])
        idf_aprox = math.log((N + 1) / (df_t + 1)) + 1
        for densidad, nombre_doc in snap_idx[termino][:5]:
            scores_densidad[nombre_doc] += densidad * idf_aprox

    # Nivel 3: Nombre de archivo
    scores_nombre = defaultdict(float)
    for nombre_doc, doc in snap_docs.items():
        tokens_nombre = doc.get("tokens_nombre", set())
        coincidencias = tokens_nombre & palabras_3plus
        if coincidencias:
            scores_nombre[nombre_doc] = len(coincidencias) / (len(tokens_nombre) or 1)

    scores_combinados = defaultdict(float)
    for doc, s in scores_tfidf.items():
        scores_combinados[doc] += s * 2.0
    for doc, s in scores_densidad.items():
        scores_combinados[doc] += s * 1.0
    for doc, s in scores_nombre.items():
        scores_combinados[doc] += s * 1.5

    if scores_combinados:
        ordenados  = sorted(scores_combinados.keys(),
                            key=lambda d: scores_combinados[d], reverse=True)
        resultado  = ordenados[:max_docs]
        log_scores = [(d, round(scores_combinados[d], 4)) for d in resultado]
        print(f"  [ENRUTADOR] max={max_docs} {log_scores}", flush=True)
        return resultado

    # Fallback vocabulario
    scored = []
    for nombre, doc in snap_docs.items():
        c = len(palabras_pregunta & doc["palabras"])
        if c >= 2:
            scored.append((c, nombre))
    scored.sort(reverse=True)
    return [n for _, n in scored[:max_docs]]


# ================================================================
#  [FIX-2 + FIX-3] EXTRACTOR v4 — chunks con solapamiento
#                                  + bonus artículo aumentado
#
#  El extractor ahora trabaja sobre chunks pre-calculados
#  en lugar de párrafos separados por \n+.
#  Cada chunk tiene CHUNK_OVERLAP chars de contexto anterior,
#  garantizando que listas numeradas y artículos largos
#  nunca queden partidos.
# ================================================================

TERMINOS_PRIORITARIOS_BASE = {
    "procedimiento", "pasos", "proceso", "tramite", "trámite",
    "requisito", "requisitos", "ingreso", "registro", "módulo",
    "modulo", "sistema", "diligencia", "diligenciar", "registrar",
    "sierju", "siglo", "módulos", "formulario", "artículo", "articulo",
    "funcionarios", "responsables", "acuerdo", "resolución",
    # [v2.1.10] Términos temporales y de plazo — críticos para preguntas "cuándo"
    "plazo", "fecha", "periodicidad", "hábil", "habiles", "trimestre",
    "período", "periodo", "días", "quinto", "corte", "vencimiento",
    "reportar", "reporte", "publicación",
}

# [FIX-2] Patrones de artículo con símbolo de grado — máxima prioridad
PATRON_ARTICULO_GRADO = re.compile(
    r'art[íi]culo\s+\d+[\s°ºo]|art\.\s*\d+[\s°ºo]',
    re.IGNORECASE
)
PATRON_ARTICULO_SIMPLE = re.compile(
    r'art[íi]culo\s+\d+|art\.\s*\d+',
    re.IGNORECASE
)


def obtener_terminos_prioritarios(nombre_doc: str) -> set:
    terminos = set(TERMINOS_PRIORITARIOS_BASE)
    with colecciones_lock:
        snap = dict(colecciones)
    for col in snap.values():
        if nombre_doc in col.get("keywords", {}):
            terminos.update(col["keywords"][nombre_doc][:10])
            break
    return terminos


def puntuar_chunk(chunk: dict, palabras: set, pregunta_norm: str,
                  terminos_prio: set,
                  idf_local: dict = None) -> float:
    """
    Puntúa un chunk según su relevancia para la pregunta.

    Sistema de puntuación v2.1.7:
      +base*idf_local por cada palabra — términos raros en el doc pesan MÁS
      +15 si la pregunta completa aparece en el chunk
      +10 si el chunk contiene artículo con grado (art. 5°)
      +5  si el chunk contiene artículo simple (artículo 5)
      +4  si el chunk contiene lista numerada (procedimientos)

    idf_local: dict {palabra: idf} donde idf = log(total_chunks/(chunks_con_termino+1))
    Un término que aparece en 1 de 38 chunks → idf=log(38/2)=2.9
    Un término que aparece en todos 38    → idf=log(38/39)≈0
    """
    texto  = chunk["texto"].lower()
    puntos = 0.0

    for w in palabras:
        count = texto.count(w)   # [v2.1.10] TF: frecuencia real en el chunk
        if count > 0:
            tf   = 1.0 + math.log(count)   # log-normalizado: 1→1.0, 2→1.69, 5→2.61
            base = 3.0 if w in terminos_prio else 1.0
            if idf_local and w in idf_local:
                base *= idf_local[w]
            puntos += tf * base   # TF-IDF completo

    # Pregunta completa — máxima señal posible
    if pregunta_norm in texto:
        puntos += 15.0

    # Bonus artículo con grado
    if PATRON_ARTICULO_GRADO.search(chunk["texto"]):
        puntos += 10.0
    elif PATRON_ARTICULO_SIMPLE.search(chunk["texto"]):
        puntos += 5.0

    # Bonus lista numerada
    if re.search(r'^\s*\d+[\.\)]\s+\S', chunk["texto"], re.MULTILINE):
        puntos += 4.0

    # [v2.1.16] BONO DE PROXIMIDAD — estrategia "Francotirador"
    # Desliza ventanas de 150 chars sobre el chunk y mide DENSIDAD de co-ocurrencia.
    # Densidad = keywords_en_ventana / total_keywords_buscadas
    # Un chunk con "incumplimiento disciplinario sanción reportar" en 120 chars
    # obtiene densidad≈1.0 y bono máximo (+20), mientras que el mismo chunk
    # con esas palabras dispersas en 800 chars obtiene densidad≈0.25 y bono mínimo.
    # A diferencia del conteo simple, la densidad NO se deja engañar por repeticiones.
    if len(palabras) >= 2:
        VENTANA   = 150   # chars ≈ 1-2 frases cortas
        PASO      = 50    # solapamiento del 67%
        n_pal     = len(palabras)
        max_densidad = 0.0
        for i in range(0, max(1, len(texto) - VENTANA), PASO):
            v = texto[i:i+VENTANA]
            matches = sum(1 for w in palabras if w in v)
            if matches >= 2:
                d = matches / n_pal
                if d > max_densidad:
                    max_densidad = d
        # Escala de bonos por densidad
        if   max_densidad >= 0.90: puntos += 20.0  # ≥90% keywords juntas → respuesta exacta
        elif max_densidad >= 0.70: puntos += 12.0  # ≥70% → muy probable
        elif max_densidad >= 0.50: puntos +=  6.0  # ≥50% → probable
        elif max_densidad >= 0.30: puntos +=  2.0  # ≥30% → señal débil

    return puntos


# [v2.1.17] Detector de preguntas de LISTADO
# Las preguntas de listado requieren ver TODOS los ítems → nunca Francotirador
# Patrones: "cuáles son", "qué secciones", "enumera", "lista", "cuántas", "nombra"
PATRONES_LISTADO = [
    "cuáles son", "cuales son", "qué secciones", "que secciones",
    "qué partes", "que partes", "qué tipos", "que tipos",
    "cuántas secciones", "cuantas secciones", "cuántos tipos", "cuantos tipos",
    "enumera", "enumere", "lista los", "lista las",
    "nombra", "menciona", "describe las", "describe los",
    "qué incluye", "que incluye", "qué contiene", "que contiene",
    "qué campos", "que campos", "qué formularios", "que formularios",
    "cuáles son los deberes", "cuales son los deberes",
    "cuáles son los roles", "cuales son los roles",
    "cuáles son las", "cuales son las",
    "cuáles son los", "cuales son los",
]

def es_pregunta_listado(pregunta: str) -> bool:
    """
    Detecta si la pregunta pide una LISTA de ítems.
    
    ¿Por qué importa?
    Las preguntas de listado tienen una propiedad única: la respuesta completa
    está DISTRIBUIDA en múltiples chunks (cada sección/rol/artículo en su chunk).
    El modo Francotirador solo envía 1 chunk → la respuesta queda truncada.
    
    Solución: para listados, forzar mínimo 2 chunks (modo Binóculo como mínimo).
    """
    p = pregunta.lower()
    return any(pat in p for pat in PATRONES_LISTADO)


def extraer_fragmento(nombre_doc: str, pregunta: str) -> str:
    with documentos_lock:
        doc = documentos_cargados.get(nombre_doc)
    if not doc:
        return ""
    with chunks_lock:
        chunks = chunks_por_doc.get(nombre_doc, [])

    if not chunks:
        return ""

    # Preparar tokens de la pregunta con el tokenizador mejorado
    palabras      = set(tokenizar(pregunta.lower()))
    pregunta_norm = pregunta.lower().replace('?', '').replace('¿', '').strip()
    terminos_prio = obtener_terminos_prioritarios(nombre_doc)

    # [FIX v2.1.11] Query expansion — agregar keywords temáticas según el tipo de pregunta
    # El usuario dice "cuándo" pero el documento dice "periodicidad", "plazo", "hábil"
    p_lower = pregunta.lower()
    if any(w in p_lower for w in ["cuándo", "cuando", "plazo", "fecha", "término", "termino"]):
        palabras.update(["periodicidad", "plazo", "hábil", "habiles", "trimestre",
                         "vencimiento", "quinto", "corte", "periodo", "período"])
    if any(w in p_lower for w in ["fecha", "año", "mes", "expidió", "expedido",
                                   "suscrito", "dado en", "firmado"]):
        palabras.update(["febrero", "enero", "marzo", "abril", "mayo", "junio",
                         "julio", "agosto", "septiembre", "octubre", "noviembre",
                         "diciembre", "2019", "2016", "2017", "2018", "2020",
                         "2021", "2022", "2023", "2024", "2025",
                         "dado", "bogotá", "presidenta", "firmado"])
    if any(w in p_lower for w in ["quién", "quien", "responsable", "cargo", "encargado"]):
        palabras.update(["responsable", "funcionario", "competencia", "administrador",
                         "seccional", "sala"])
    # [v2.1.15] Preguntas de responsabilidad de carga de datos → artículo 5 y 7
    if any(w in p_lower for w in ["cargar", "carga", "diligenciar", "responsable",
                                   "quien carga", "quién carga", "quien diligencia"]):
        palabras.update(["funcionario", "magistrado", "juez", "diligenciar",
                         "reportar", "formulario", "rol", "roles", "despacho",
                         "artículo", "corresponde"])
    # [v2.1.15] Preguntas sobre permisos/roles ("puede un asistente") → artículo 7
    if any(w in p_lower for w in ["puede", "asistente", "auxiliar", "empleado",
                                   "secretario", "quien puede", "quién puede",
                                   "está permitido", "autorizado"]):
        palabras.update(["rol", "roles", "funcionario", "juez", "magistrado",
                         "administrador", "seccional", "nacional",
                         "súper", "super", "autorizado", "cargo"])
    # [v2.1.15] Preguntas de capacitación → artículo 2 pcsja19
    if any(w in p_lower for w in ["capacita", "capacitacion", "capacitación", "entrena",
                                   "instruye", "quien capacita", "quién capacita"]):
        palabras.update(["capacitación", "capacita", "cendoj", "udae",
                         "unidad", "desarrollo", "estadístico", "cronograma",
                         "informes", "presentacion"])
    if any(w in p_lower for w in ["cómo", "como", "pasos", "procedimiento", "diligenciar"]):
        palabras.update(["procedimiento", "pasos", "diligenciar", "registrar",
                         "formulario", "módulo"])
    if any(w in p_lower for w in ["discrepancia", "diferencia", "error", "inconsistencia"]):
        palabras.update(["discrepancia", "inconsistencia", "corrección", "unidad",
                         "verificación", "hábiles"])
    # [v2.1.15] Preguntas de consecuencias/sanciones → artículos 19 y 20 psaa16
    if any(w in p_lower for w in ["qué pasa", "que pasa", "consecuencia", "sanción",
                                   "sancion", "incumplimiento", "no reporto", "no reportar",
                                   "tarde", "vencido", "vence", "multa", "castigo",
                                   "a tiempo", "fuera de tiempo", "disciplinario"]):
        palabras.update(["sanción", "sancion", "disciplinaria", "incumplimiento",
                         "juramento", "veraz", "oportunamente", "acarreará",
                         "código", "disciplinario", "sanciones", "artículo"])

    # [FIX-1 v2.1.7] IDF local por chunk — términos raros pesan más
    # [FIX-2 v2.1.11] Palabras ultra-comunes (>80% chunks) excluidas del score
    total_chunks = max(len(chunks), 1)
    idf_local = {}
    for w in palabras:
        chunks_con_w = sum(1 for c in chunks if w in c["texto"].lower())
        if chunks_con_w > 0:
            idf_val = math.log((total_chunks + 1) / (chunks_con_w + 1)) + 1.0
            # Excluir palabras ultra-comunes: aparecen en >85% de chunks → ruido puro
            if chunks_con_w / total_chunks <= 0.85:
                idf_local[w] = idf_val

    # Puntuar todos los chunks con IDF local
    scored = []
    for chunk in chunks:
        pts = puntuar_chunk(chunk, palabras, pregunta_norm, terminos_prio, idf_local)
        if pts > 0:
            scored.append((pts, chunk["indice"], chunk))

    scored.sort(reverse=True)

    # [v2.1.16] PODA DINÁMICA — "Francotirador vs Escopeta"
    # [v2.1.17] EXCEPCIÓN: preguntas de LISTADO nunca usan Francotirador.
    # Razón: en listados la respuesta está DISTRIBUIDA en múltiples chunks.
    # Enviar solo 1 chunk garantiza una respuesta incompleta (p.ej. "cuáles son
    # las secciones" devuelve solo las 2 primeras de 7 porque el resto está
    # en el chunk 2, que Francotirador descarta).
    es_listado = es_pregunta_listado(pregunta)
    
    if len(scored) >= 2:
        s1, s2 = scored[0][0], scored[1][0]
        ratio  = s1 / max(s2, 0.01)
        if es_listado:
            # Listado: mínimo 2 chunks siempre para capturar todos los ítems
            chunks_a_usar = MAX_CHUNKS_CONTEXTO if ratio < 1.8 else 2
            modo = "LISTADO-BINÓCULO" if chunks_a_usar == 2 else "LISTADO-ESCOPETA"
        elif ratio >= 3.0:
            chunks_a_usar = 1   # Certeza alta  → 1 chunk  ≈200 tok
            modo = "FRANCOTIRADOR"
        elif ratio >= 1.8:
            chunks_a_usar = 2   # Certeza media → 2 chunks ≈400 tok
            modo = "BINÓCULO"
        else:
            chunks_a_usar = MAX_CHUNKS_CONTEXTO   # Ambiguo → 3 chunks
            modo = "ESCOPETA"
        print(f"  [PODA] {nombre_doc[:25]} ratio={ratio:.2f} → {chunks_a_usar}chunk [{modo}]", flush=True)
    else:
        chunks_a_usar = min(len(scored), MAX_CHUNKS_CONTEXTO)

    # Tomar los mejores chunks (sin repetir índices adyacentes ya incluidos)
    seleccionados   = []
    indices_usados  = set()
    chars_acum      = 0

    for pts, idx, chunk in scored[:chunks_a_usar * 2]:
        if idx in indices_usados:
            continue
        texto = chunk["texto"]
        if chars_acum + len(texto) > CHUNK_SIZE * chunks_a_usar:
            break

        # Metadato de sección
        meta = f"[SEC: {chunk['seccion'][:60]} | CHUNK: {idx}]"
        seleccionados.append(meta + "\n" + texto)
        indices_usados.add(idx)
        chars_acum += len(meta) + len(texto)

        if len(seleccionados) >= chunks_a_usar:
            break

    # Fallback: si no se encontró nada, usar los primeros chunks
    if not seleccionados:
        print(f"  [EXTRACTOR] Fallback primeros chunks para {nombre_doc}", flush=True)
        for chunk in chunks[:2]:
            meta = f"[SEC: {chunk['seccion'][:60]} | CHUNK: {chunk['indice']}]"
            seleccionados.append(meta + "\n" + chunk["texto"])

    nombre_original = doc.get("nombre_original", nombre_doc)
    nombre_display  = os.path.splitext(nombre_original)[0].upper()
    coleccion       = doc.get("coleccion", "")
    etiqueta        = (f"[DOC: {nombre_display}]"
                       if not coleccion or coleccion == "general"
                       else f"[DOC: {nombre_display} | {coleccion.upper()}]")

    separador = "\n" + "═" * 60 + "\n"
    return etiqueta + "\n" + "\n\n".join(seleccionados) + separador


# ================================================================
#  CLIENTE OLLAMA
# ================================================================

def llamar_ollama(mensajes: list, num_predict: int = 150) -> list:
    adquirido = ollama_semaforo.acquire(timeout=30)
    if not adquirido:
        return ["COLA_LLENA"]

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model":    MODEL,
                "messages": mensajes,
                "stream":   True,
                "think":    False,
                "options":  {
                    "temperature":    0.0,    # [v2.1.9] 0=determinístico, sin generación desde memoria
                    "num_predict":    num_predict,         # [v2.1.17] dinámico: 150 normal, 300 listados
                    "num_ctx":        2048,   # [v2.1.8] alineado: 2×3×800≈1200tok + overhead=1780tok < 2048
                    "num_thread":     6,      # [v2.1.8] Ryzen 5 2600: 6 núcleos físicos (SMT causa thrashing)
                    "num_batch":      512,    # [v2.1.8] batch más grande → menos ciclos de prefill → TTFT menor
                    "repeat_penalty": 1.1,
                    "stop": [
                        "\n\n\n",
                        "Espero que",
                        "Es importante destacar",
                        "Cabe destacar que",
                        "En conclusión,",
                        "Sin embargo,",
                        "Por otro lado,",
                        "Cabe mencionar",
                    ]
                }
            },
            stream=True,
            timeout=(TIMEOUT_CONEXION, TIMEOUT_RESPUESTA)
        )

        if resp.status_code != 200:
            return [f"ERROR_HTTP_{resp.status_code}"]

        chunks = []
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj         = json.loads(line.decode("utf-8"))
                content_tok = obj.get("message", {}).get("content", "")
                done        = obj.get("done", False)
                if content_tok:
                    safe = json.dumps(content_tok)[1:-1]
                    chunks.append(
                        f'data: {{"choices":[{{"delta":{{"content":"{safe}"}}}}]}}'
                    )
                if done:
                    chunks.append("data: [DONE]")
            except Exception as e:
                print(f"[PARSER] {e} | {str(line)[:80]}", flush=True)

        return chunks

    except requests.exceptions.ConnectTimeout:
        with ollama_lock:
            ollama_estado["disponible"] = False
        return ["TIMEOUT_CONEXION"]
    except requests.exceptions.ReadTimeout:
        return ["TIMEOUT_RESPUESTA"]
    except requests.exceptions.ConnectionError:
        with ollama_lock:
            ollama_estado["disponible"] = False
        return ["OLLAMA_CAIDO"]
    except Exception as e:
        return [f"ERROR:{e}"]
    finally:
        ollama_semaforo.release()


# ================================================================
#  ENDPOINT PRINCIPAL — /siaa/chat
# ================================================================

@app.route("/siaa/chat", methods=["POST"])
def chat():
    inc_activos()
    try:
        data     = request.get_json()
        messages = data.get("messages", []) if data else []

        if not messages:
            dec_activos()
            return Response("Sin mensajes.", status=400)

        with ollama_lock:
            disponible = ollama_estado["disponible"]
        if not disponible:
            disponible = verificar_ollama()
        if not disponible:
            dec_activos()
            return Response(
                'data: {"choices":[{"delta":{"content":"⚠ Servidor IA no disponible."}}]}\ndata: [DONE]\n',
                content_type="text/event-stream"
            )

        ultima_pregunta = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )

        es_conv         = es_conversacion_general(ultima_pregunta)
        contexto        = ""
        docs_relevantes = []

        # ── CACHÉ: consultar antes de procesar ────────────────────────────────
        # Solo para preguntas documentales — las conversacionales son únicas
        if not es_conv and CACHE_SOLO_DOC:
            hit = cache_get(ultima_pregunta)
            if hit:
                print(
                    f"[CACHÉ HIT] pregunta={ultima_pregunta[:50]!r} "
                    f"stats={cache_stats()}",
                    flush=True
                )
                registrar_consulta(
                    tipo="DOC", pregunta=ultima_pregunta,
                    respuesta=hit["respuesta"], docs=[],
                    ctx_chars=0, tiempo_seg=0.0, cache_hit=True
                )
                dec_activos()
                respuesta_cached = hit["respuesta"]
                cita_cached      = hit["cita"]

                def _stream_cached():
                    # Enviar respuesta token a token simulando streaming
                    # (el HTML espera SSE, no podemos enviar todo de golpe)
                    chunk_size = 40  # chars por "token" simulado
                    for i in range(0, len(respuesta_cached), chunk_size):
                        trozo = respuesta_cached[i:i+chunk_size]
                        safe  = json.dumps(trozo)[1:-1]
                        yield f'data: {{"choices":[{{"delta":{{"content":"{safe}"}}}}]}}\n\n'
                    if cita_cached:
                        safe_cita = json.dumps(cita_cached)[1:-1]
                        yield f'data: {{"choices":[{{"delta":{{"content":"{safe_cita}"}}}}]}}\n\n'
                    yield "data: [DONE]\n\n"

                return Response(
                    stream_with_context(_stream_cached()),
                    content_type="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache",
                             "X-Cache": "HIT"}
                )

        if not es_conv:
            es_doc_esp = detectar_doc_especifico(ultima_pregunta)
            max_docs   = 1 if es_doc_esp else MAX_DOCS_CONTEXTO

            docs_relevantes = detectar_documentos(ultima_pregunta, max_docs=max_docs)
            for nombre_doc in docs_relevantes:
                contexto += extraer_fragmento(nombre_doc, ultima_pregunta)

        # [v2.1.7] Cita bibliográfica  [v2.1.18] + link nueva pestaña (A) + artículo exacto (D)
        cita_fuente = ""
        if not es_conv and docs_relevantes:
            partes_cita = []
            links_ver   = []
            host_ip     = request.host.split(":")[0]   # IP del servidor
            with documentos_lock:
                for nd in docs_relevantes:
                    doc_info = documentos_cargados.get(nd, {})
                    nombre_d = os.path.splitext(
                        doc_info.get("nombre_original", nd)
                    )[0].upper()
                    colec = doc_info.get("coleccion", "")
                    etiq  = nombre_d if not colec or colec == "general" \
                            else f"{nombre_d} ({colec.upper()})"
                    partes_cita.append(etiq)
                    # Link A: abre el doc completo en nueva pestaña
                    url = f"http://{host_ip}:5000/siaa/ver/{nd}"
                    links_ver.append(f"[📖 Ver {nombre_d}]({url})")

            cita_fuente = "\n\n📄 **Fuente:** " + " · ".join(partes_cita)

            # Links de acceso directo al documento
            if links_ver:
                cita_fuente += "\n\n" + "  ".join(links_ver)

            # Sugerencia D: para preguntas de listado indicar dónde consultar más
            if es_pregunta_listado(ultima_pregunta):
                nombres_docs = " · ".join(
                    os.path.splitext(documentos_cargados.get(nd, {})
                    .get("nombre_original", nd))[0].upper()
                    for nd in docs_relevantes
                    if nd in documentos_cargados
                )
                cita_fuente += (
                    "\n\n💡 *Para la información completa consulte el documento "
                    f"fuente haciendo clic en **Ver {nombres_docs}** arriba.*"
                )

        print(
            f"[CHAT] tipo={'CONV' if es_conv else 'DOC'} "
            f"especifico={detectar_doc_especifico(ultima_pregunta) if not es_conv else False} "
            f"pregunta={ultima_pregunta[:50]!r} "
            f"docs={docs_relevantes} ctx={len(contexto)}chars",
            flush=True
        )

        system_prompt = SYSTEM_CONVERSACIONAL if es_conv else SYSTEM_DOCUMENTAL
        ollama_msgs   = [{"role": "system", "content": system_prompt}]

        if not es_conv:
            if contexto.strip():
                ollama_msgs.append({
                    "role":    "system",
                    "content": f"CONTEXTO DOCUMENTAL:\n{contexto}"
                })
            else:
                ollama_msgs.append({
                    "role":    "system",
                    "content": (
                        "No se encontraron documentos. Responde EXACTAMENTE:\n"
                        "'No encontré esa información en los documentos disponibles.\n"
                        "📄 Fuente: Sin documentos encontrados'"
                    )
                })

        for msg in messages[-6:]:
            rol = msg.get("role", "")
            if rol == "user":
                ollama_msgs.append({"role": "user", "content": msg["content"]})
            elif rol == "assistant":
                ollama_msgs.append(msg)

        # [v2.1.17] num_predict dinámico — listados necesitan más tokens
        # Lista de 7 secciones con nombres largos ≈ 200 tokens.
        # Respuesta puntual (fecha, artículo) ≈ 60-80 tokens.
        _es_listado = not es_conv and es_pregunta_listado(ultima_pregunta)
        _num_predict = 300 if _es_listado else 150
        if _es_listado:
            print(f"  [LISTADO] num_predict=300 para: {ultima_pregunta[:50]!r}", flush=True)

        t_inicio = time.time()   # [v2.1.13] Para medir tiempo de respuesta

        def generate():
            try:
                result = llamar_ollama(ollama_msgs, num_predict=_num_predict)
                respuesta_completa = ""   # [v2.1.12] Acumular para caché
                if not result:
                    yield 'data: {"choices":[{"delta":{"content":"Sin respuesta."}}]}\n\n'
                    return

                ERRORES = {
                    "COLA_LLENA":        "⏳ Sistema ocupado. Intente en 30 segundos.",
                    "TIMEOUT_CONEXION":  "⚠ IA no responde. Intente de nuevo.",
                    "TIMEOUT_RESPUESTA": "⏱ Consulta tomó demasiado tiempo.",
                    "OLLAMA_CAIDO":      "⚠ Servidor IA reiniciándose. Espere 1 minuto.",
                }

                if result[0] in ERRORES:
                    yield f'data: {{"choices":[{{"delta":{{"content":"{ERRORES[result[0]]}"}}}}]}}\n\n'
                    return
                if result[0].startswith("ERROR"):
                    print(f"[CHAT] Error Ollama: {result[0]}", flush=True)
                    yield 'data: {"choices":[{"delta":{"content":"⚠ Error interno."}}]}\n\n'
                    return

                for chunk in result:
                    if chunk == "data: [DONE]":
                        # Guardar en caché antes del [DONE]
                        if not es_conv and respuesta_completa.strip():
                            cache_set(ultima_pregunta, respuesta_completa, cita_fuente)
                        # [v2.1.13] Registrar en log de calidad
                        registrar_consulta(
                            tipo="CONV" if es_conv else "DOC",
                            pregunta=ultima_pregunta,
                            respuesta=respuesta_completa,
                            docs=docs_relevantes,
                            ctx_chars=len(contexto),
                            tiempo_seg=time.time() - t_inicio,
                        )
                        # [FIX-2 v2.1.7] Inyectar cita antes del [DONE]
                        if cita_fuente:
                            cita_escaped = json.dumps(cita_fuente)[1:-1]
                            yield f'data: {{"choices":[{{"delta":{{"content":"{cita_escaped}"}}}}]}}\n\n'
                    else:
                        try:
                            raw = chunk.replace("data: ", "", 1)
                            tok = json.loads(raw)
                            tok_c = tok.get("choices",[{}])[0].get("delta",{}).get("content","")
                            if tok_c:
                                respuesta_completa += tok_c
                        except Exception:
                            pass
                    yield chunk + "\n\n"
            except Exception as ex_gen:
                # Registrar error en log de calidad
                registrar_consulta(
                    tipo="ERROR", pregunta=ultima_pregunta,
                    respuesta=f"ERROR: {ex_gen}", docs=docs_relevantes,
                    ctx_chars=len(contexto), tiempo_seg=time.time()-t_inicio
                )
                raise
            finally:
                dec_activos()

        return Response(
            stream_with_context(generate()),
            content_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"}
        )

    except Exception as e:
        dec_activos()
        print(f"[CHAT] Excepción: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


# ================================================================
#  ENDPOINTS ADMINISTRATIVOS
# ================================================================

@app.route("/siaa/ver/<path:nombre_doc>", methods=["GET"])
def ver_documento(nombre_doc):
    """
    Sirve el documento .md como página HTML completa en nueva pestaña.
    ?sec=ARTÍCULO 8  → resalta y hace scroll a esa sección.
    ?q=<pregunta>    → resalta todas las ocurrencias de keywords.
    """
    nombre_doc = nombre_doc.lower().strip()
    with documentos_lock:
        doc = documentos_cargados.get(nombre_doc)
        if not doc:
            for k, v in documentos_cargados.items():
                if nombre_doc.replace(" ", "_") in k or nombre_doc in k:
                    doc, nombre_doc = v, k
                    break

    if not doc:
        return f"<h2 style='font-family:sans-serif;color:#b91c1c'>Documento '{nombre_doc}' no encontrado.</h2>", 404

    seccion  = request.args.get("sec", "").strip()
    contenido = doc["contenido"]
    nombre_display = os.path.splitext(doc.get("nombre_original", nombre_doc))[0].upper()
    coleccion      = doc.get("coleccion", "").upper()

    # Escapar y resaltar sección objetivo
    import html as _h, re as _re
    lineas_out = []
    primer_match = True
    for linea in contenido.splitlines():
        esc = _h.escape(linea)
        # Negrita **texto** e _itálica_
        esc = _re.sub(r'\*\*(.*?)\*\*', r'<strong></strong>', esc)
        esc = _re.sub(r'_(.*?)_', r'<em></em>', esc)
        # Encabezados
        if linea.startswith("#### "): esc = f"<h4>{_h.escape(linea[5:])}</h4>"
        elif linea.startswith("### "): esc = f"<h3>{_h.escape(linea[4:])}</h3>"
        elif linea.startswith("## "):  esc = f"<h2>{_h.escape(linea[3:])}</h2>"
        elif linea.startswith("# "):   esc = f"<h1>{_h.escape(linea[2:])}</h1>"
        elif linea.strip() == "":      esc = ""
        else:                          esc = f"<p>{esc}</p>"
        # Resaltar sección objetivo
        if seccion and seccion.lower() in linea.lower() and primer_match:
            esc = f'<div id="objetivo" class="resaltado">{esc}</div>'
            primer_match = False
        lineas_out.append(esc)

    cuerpo = "\n".join(lineas_out)
    scroll_js = '<script>document.getElementById("objetivo")?.scrollIntoView({behavior:"smooth",block:"center"});</script>' if seccion else ""

    html_pagina = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>{nombre_display}</title>
<style>
  :root{{--blue:#1a2a6c;--gold:#c5a059}}
  *{{box-sizing:border-box}}
  body{{font-family:'Segoe UI',sans-serif;margin:0;padding:0;color:#1e293b;line-height:1.75}}
  header{{background:var(--blue);color:white;padding:14px 32px;
          border-bottom:4px solid var(--gold);display:flex;
          align-items:center;justify-content:space-between;position:sticky;top:0;z-index:10}}
  header h1{{margin:0;font-size:.95rem;letter-spacing:1.5px}}
  .badge{{background:var(--gold);color:var(--blue);padding:2px 10px;
          border-radius:4px;font-size:.72rem;font-weight:700;margin-left:10px}}
  .contenido{{max-width:860px;margin:0 auto;padding:28px 32px 80px}}
  h1,h2,h3,h4{{color:var(--blue);margin-top:1.8em}}
  h2{{border-left:4px solid var(--gold);padding-left:12px}}
  strong{{color:#0f172a}}
  p{{margin:.4em 0}}
  .resaltado{{background:#fef9c3;border-left:5px solid var(--gold);
             padding:8px 14px;border-radius:4px;margin:6px 0;
             animation:blink 1s ease .3s 3}}
  @keyframes blink{{0%,100%{{background:#fef9c3}}50%{{background:#fde047}}}}
  .btn-imprimir{{position:fixed;bottom:24px;right:24px;
                background:var(--blue);color:white;border:none;
                padding:10px 22px;border-radius:8px;cursor:pointer;
                font-weight:700;font-size:.82rem;box-shadow:0 4px 14px rgba(0,0,0,.25)}}
  .btn-imprimir:hover{{background:#243580}}
  @media print{{header,.btn-imprimir{{display:none}}.contenido{{padding:0}}}}
</style>
</head>
<body>
<header>
  <h1>📄 {nombre_display} <span class="badge">{coleccion}</span></h1>
  <span style="color:var(--gold);font-size:.78rem;font-weight:700">SIAA · Seccional Bucaramanga</span>
</header>
<div class="contenido">{cuerpo}</div>
<button class="btn-imprimir" onclick="window.print()">🖨 Imprimir</button>
{scroll_js}
</body>
</html>"""

    return Response(html_pagina, content_type="text/html; charset=utf-8")


@app.route("/siaa/status", methods=["GET"])
def status():
    with ollama_lock:
        ol_ok  = ollama_estado["disponible"]
        fallos = ollama_estado["fallos"]
        warmup = ollama_estado["warmup_done"]
    with colecciones_lock:
        info_cols = {
            n: {"docs": list(c["docs"].keys()), "total": len(c["docs"])}
            for n, c in colecciones.items()
        }
    with documentos_lock:
        total_docs = len(documentos_cargados)
    with indice_densidad_lock:
        total_terminos = len(indice_densidad)
    with chunks_lock:
        total_chunks = sum(len(v) for v in chunks_por_doc.values())

    return jsonify({
        "version":           VERSION,
        "estado":            "ok" if ol_ok else "error",
        "cache":             cache_stats(),
        "ollama":            ol_ok,
        "ollama_fallos":     fallos,
        "modelo":            MODEL,
        "warmup_completado": warmup,
        "usuarios_activos":  usuarios_activos,
        "total_atendidos":   total_atendidos,
        "total_documentos":  total_docs,
        "total_chunks":      total_chunks,
        "indice_terminos":   total_terminos,
        "chunk_size":        CHUNK_SIZE,
        "chunk_overlap":     CHUNK_OVERLAP,
        "colecciones":       info_cols,
    })


@app.route("/siaa/keywords/<nombre_doc>", methods=["GET"])
def ver_keywords(nombre_doc: str):
    nombre_doc = nombre_doc.lower().replace("+", " ")
    with colecciones_lock:
        for col in colecciones.values():
            if nombre_doc in col.get("keywords", {}):
                return jsonify({
                    "documento": nombre_doc,
                    "keywords":  col["keywords"][nombre_doc],
                })
    return jsonify({"error": f"'{nombre_doc}' no encontrado"}), 404


@app.route("/siaa/densidad/<termino>", methods=["GET"])
def ver_densidad(termino: str):
    termino = termino.lower()
    with indice_densidad_lock:
        entradas = indice_densidad.get(termino, [])
    if not entradas:
        return jsonify({"error": f"'{termino}' no en índice"}), 404
    return jsonify({
        "termino":    termino,
        "top_docs":   [{"doc": d, "densidad": round(den, 6)} for den, d in entradas[:10]],
        "total_docs": len(entradas)
    })


@app.route("/siaa/enrutar", methods=["GET"])
def diagnostico_enrutador():
    pregunta   = request.args.get("q", "")
    if not pregunta:
        return jsonify({"error": "Parámetro 'q' requerido"}), 400
    es_doc_esp = detectar_doc_especifico(pregunta)
    max_docs   = 1 if es_doc_esp else MAX_DOCS_CONTEXTO
    docs       = detectar_documentos(pregunta, max_docs=max_docs)
    with documentos_lock:
        resultado = [
            {
                "doc":       d,
                "tamano":    documentos_cargados[d]["tamano"] if d in documentos_cargados else 0,
                "coleccion": documentos_cargados[d].get("coleccion", "") if d in documentos_cargados else "",
                "chunks":    documentos_cargados[d].get("num_chunks", 0) if d in documentos_cargados else 0,
            }
            for d in docs
        ]
    return jsonify({
        "pregunta":         pregunta,
        "doc_especifico":   es_doc_esp,
        "max_docs_usados":  max_docs,
        "docs_encontrados": resultado,
    })


@app.route("/siaa/fragmento", methods=["GET"])
def ver_fragmento():
    """
    Diagnóstico: muestra exactamente el fragmento que llega al modelo.
    curl "http://IP:5000/siaa/fragmento?doc=acuerdo_no._psaa16-10476.md&q=funcionarios+responsables"
    """
    nombre_doc = request.args.get("doc", "").lower()
    pregunta   = request.args.get("q", "")
    if not nombre_doc or not pregunta:
        return jsonify({"error": "Parámetros 'doc' y 'q' requeridos"}), 400

    fragmento = extraer_fragmento(nombre_doc, pregunta)
    return jsonify({
        "documento": nombre_doc,
        "pregunta":  pregunta,
        "fragmento": fragmento,
        "chars":     len(fragmento),
    })


@app.route("/siaa/cache", methods=["GET", "DELETE"])
def cache_endpoint():
    """
    GET  /siaa/cache  → estadísticas del caché
    DELETE /siaa/cache → vaciar el caché completamente

    Útil cuando se actualizan documentos y las respuestas cacheadas
    pueden estar desactualizadas.
    """
    if request.method == "DELETE":
        with _cache_lock:
            _cache_respuestas.clear()
        return jsonify({"vaciado": True, "mensaje": "Caché limpiado correctamente"})
    return jsonify(cache_stats())


@app.route("/siaa/log", methods=["GET"])
def ver_log():
    """
    Muestra las últimas N entradas del log de calidad.

    Parámetros URL:
      ?n=50          → últimas N consultas (máx 500, defecto 50)
      ?tipo=ERROR    → filtrar por tipo: OK, ERROR, POSIBLE_ALUCINACION, etc.
      ?alerta=OK     → filtrar por alerta
      ?formato=txt   → salida en texto plano (más fácil de leer en terminal)

    Ejemplo: curl http://localhost:5000/siaa/log?n=20&alerta=POSIBLE_ALUCINACION
    """
    try:
        n       = min(int(request.args.get("n", 50)), 500)
        filtro_tipo   = request.args.get("tipo", "").upper()
        filtro_alerta = request.args.get("alerta", "").upper()
        fmt     = request.args.get("formato", "json")

        _asegurar_carpeta_log()

        try:
            with open(LOG_ARCHIVO, "r", encoding="utf-8") as f:
                lineas = f.readlines()
        except FileNotFoundError:
            return jsonify({"error": "Sin consultas registradas aún", "entradas": []})

        # Parsear y filtrar
        entradas = []
        for linea in reversed(lineas):
            linea = linea.strip()
            if not linea:
                continue
            try:
                e = json.loads(linea)
                if filtro_tipo and e.get("tipo","") != filtro_tipo:
                    continue
                if filtro_alerta and e.get("alerta","") != filtro_alerta:
                    continue
                entradas.append(e)
                if len(entradas) >= n:
                    break
            except Exception:
                continue

        # Calcular resumen
        todas_lineas = [json.loads(l) for l in lineas if l.strip()]
        total   = len(todas_lineas)
        errores = sum(1 for e in todas_lineas if e.get("alerta") == "ERROR")
        alucs   = sum(1 for e in todas_lineas if e.get("alerta") == "POSIBLE_ALUCINACION")
        hits    = sum(1 for e in todas_lineas if e.get("tipo") == "CACHE_HIT")
        t_prom  = round(
            sum(e.get("tiempo_s", 0) for e in todas_lineas if e.get("tiempo_s", 0) > 0) /
            max(sum(1 for e in todas_lineas if e.get("tiempo_s", 0) > 0), 1), 1
        )

        if fmt == "txt":
            lineas_txt = [
                f"=== Log SIAA — últimas {len(entradas)} de {total} consultas ===",
                f"Errores: {errores} | Posibles alucinaciones: {alucs} | Cache hits: {hits} | T.prom: {t_prom}s",
                ""
            ]
            for e in entradas:
                alerta_str = f" ⚠ [{e['alerta']}]" if e["alerta"] != "OK" else ""
                lineas_txt.append(
                    f"[{e['ts']}]{alerta_str} {e['tipo']} {e['tiempo_s']}s\n"
                    f"  P: {e['pregunta'][:80]}\n"
                    f"  R: {e['respuesta'][:100]}\n"
                    f"  Docs: {e['docs']}"
                )
                lineas_txt.append("")
            return Response("\n".join(lineas_txt), content_type="text/plain; charset=utf-8")

        return jsonify({
            "resumen": {
                "total_consultas":        total,
                "errores":                errores,
                "posibles_alucinaciones": alucs,
                "cache_hits":             hits,
                "tiempo_promedio_s":      t_prom,
            },
            "entradas": entradas,
            "mostrando": len(entradas),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/siaa/recargar", methods=["GET"])
def recargar():
    cargar_documentos()
    with _cache_lock:
        _cache_respuestas.clear()  # Docs cambiaron → caché puede ser obsoleto
    with colecciones_lock:
        info = {n: len(c["docs"]) for n, c in colecciones.items()}
    with chunks_lock:
        total_chunks = sum(len(v) for v in chunks_por_doc.values())
    return jsonify({
        "recargado":    True,
        "colecciones":  info,
        "total_docs":   sum(info.values()),
        "total_chunks": total_chunks,
    })


# ================================================================
#  ARRANQUE
# ================================================================

if __name__ == "__main__":
    print("=" * 62)
    print(f"  SIAA — Proxy Inteligente v{VERSION}")
    print(f"  Sistema Inteligente de Apoyo Administrativo")
    print(f"  Seccional Bucaramanga — Rama Judicial")
    print("=" * 62)
    print(f"  Modelo:         {MODEL}")
    print(f"  Fuentes:        {CARPETA_FUENTES}")
    print(f"  Chunk size:     {CHUNK_SIZE} chars")
    print(f"  Chunk overlap:  {CHUNK_OVERLAP} chars")
    print(f"  Max chunks/doc: {MAX_CHUNKS_CONTEXTO}")
    print(f"  Tokenizador:    alfanumérico + números ≥4 dígitos")
    print(f"  Artículo bonus: +10 con grado°, +5 sin grado")
    print(f"  Doc específico: max_docs=1 para PSAA/PCSJA/acuerdo")
    print("=" * 62)

    cargar_documentos()
    verificar_ollama()

    with ollama_lock:
        ok = "OK ✓" if ollama_estado["disponible"] else "NO DISPONIBLE ✗"
    print(f"  Ollama: {ok}")
    print("=" * 62)

    try:
        from waitress import serve
        print("  Servidor: Waitress — 0.0.0.0:5000")
        serve(app, host="0.0.0.0", port=5000,
              threads=HILOS_SERVIDOR, channel_timeout=200)
    except ImportError:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
