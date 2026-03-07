"""
╔══════════════════════════════════════════════════════════════════╗
║         SIAA — Proxy Inteligente  v2.1.8                       ║
║         Seccional Bucaramanga — Rama Judicial                    ║
╠══════════════════════════════════════════════════════════════════╣
║  CORRECCIONES v2.1.8(sobre v2.1.7)                              ║
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

app = Flask(__name__)
CORS(app)

# ================================================================
#  CONFIGURACIÓN CENTRAL
# ================================================================

OLLAMA_URL             = "http://localhost:11434"
MODEL                  = "qwen2.5:3b"
VERSION                = "2.1.8"
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
CHUNK_OVERLAP          = 200
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

SYSTEM_DOCUMENTAL = """Eres SIAA, asistente de la Seccional Bucaramanga, Rama Judicial.
USA SOLO el contexto [DOC:...]. PROHIBIDO conocimiento propio.
Si no está en el contexto: "No encontré esa información en los documentos disponibles."
Español formal. Máximo 5 líneas. Cita artículos y pasos del contexto literalmente.
No mezcles bloques ═══."""

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

    # Nivel 1: TF-IDF
    scores_tfidf = defaultdict(float)
    for col in snap_cols.values():
        for nombre_doc, keywords in col.get("keywords", {}).items():
            if nombre_doc not in snap_docs:
                continue
            for kw in keywords:
                if kw in p:
                    scores_tfidf[nombre_doc] += 1.0

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
        if w in texto:
            base = 3.0 if w in terminos_prio else 1.0
            # Multiplicar por IDF local si está disponible
            if idf_local and w in idf_local:
                base *= idf_local[w]
            puntos += base

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

    return puntos


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

    # [FIX-1 v2.1.7] IDF local por chunk — términos raros pesan más
    # "discrepancia" en 1/38 chunks  → idf≈3.97 (peso ALTO)
    # "judicial"     en 35/38 chunks → idf≈1.08 (peso BAJO)
    total_chunks = max(len(chunks), 1)
    idf_local = {}
    for w in palabras:
        chunks_con_w = sum(1 for c in chunks if w in c["texto"].lower())
        if chunks_con_w > 0:
            idf_local[w] = math.log((total_chunks + 1) / (chunks_con_w + 1)) + 1.0

    # Puntuar todos los chunks con IDF local
    scored = []
    for chunk in chunks:
        pts = puntuar_chunk(chunk, palabras, pregunta_norm, terminos_prio, idf_local)
        if pts > 0:
            scored.append((pts, chunk["indice"], chunk))

    scored.sort(reverse=True)

    # Tomar los mejores chunks (sin repetir índices adyacentes ya incluidos)
    seleccionados   = []
    indices_usados  = set()
    chars_acum      = 0

    for pts, idx, chunk in scored[:MAX_CHUNKS_CONTEXTO * 2]:
        if idx in indices_usados:
            continue
        texto = chunk["texto"]
        if chars_acum + len(texto) > CHUNK_SIZE * MAX_CHUNKS_CONTEXTO:
            break

        # Metadato de sección
        meta = f"[SEC: {chunk['seccion'][:60]} | CHUNK: {idx}]"
        seleccionados.append(meta + "\n" + texto)
        indices_usados.add(idx)
        chars_acum += len(meta) + len(texto)

        if len(seleccionados) >= MAX_CHUNKS_CONTEXTO:
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

def llamar_ollama(mensajes: list) -> list:
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
                    "temperature":    0.05,
                    "num_predict":    200,
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

        if not es_conv:
            es_doc_esp = detectar_doc_especifico(ultima_pregunta)
            max_docs   = 1 if es_doc_esp else MAX_DOCS_CONTEXTO

            docs_relevantes = detectar_documentos(ultima_pregunta, max_docs=max_docs)
            for nombre_doc in docs_relevantes:
                contexto += extraer_fragmento(nombre_doc, ultima_pregunta)

        # [FIX-2 v2.1.7] Construir cita bibliográfica desde el proxy
        # No depende del modelo — siempre aparece, siempre es correcta
        cita_fuente = ""
        if not es_conv and docs_relevantes:
            partes_cita = []
            with documentos_lock:
                for nd in docs_relevantes:
                    doc_info = documentos_cargados.get(nd, {})
                    nombre_d = os.path.splitext(
                        doc_info.get("nombre_original", nd)
                    )[0].upper()
                    colec    = doc_info.get("coleccion", "")
                    etiq     = f"{nombre_d}" if not colec or colec == "general"                                else f"{nombre_d} ({colec.upper()})"
                    partes_cita.append(etiq)
            cita_fuente = "\n\n📄 **Fuente:** " + " · ".join(partes_cita)

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

        def generate():
            try:
                result = llamar_ollama(ollama_msgs)
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
                    if chunk == "data: [DONE]" and cita_fuente:
                        # [FIX-2 v2.1.7] Inyectar cita antes del [DONE]
                        cita_escaped = json.dumps(cita_fuente)[1:-1]
                        yield f'data: {{"choices":[{{"delta":{{"content":"{cita_escaped}"}}}}]}}\n\n'
                    yield chunk + "\n\n"
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


@app.route("/siaa/recargar", methods=["GET"])
def recargar():
    cargar_documentos()
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
