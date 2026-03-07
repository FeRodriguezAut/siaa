"""
╔══════════════════════════════════════════════════════════════════╗
║         SIAA — Proxy Inteligente  v2.1.2                         ║
║         Seccional Bucaramanga — Rama Judicial                    ║
╠══════════════════════════════════════════════════════════════════╣
║  CAMBIOS v2.1.2 vs v2.1.1                                        ║
║                                                                  ║
║  [FIX-1] Carga de archivos con mayúsculas y espacios             ║
║      'CADENA CONTROL PROCESO MODULOSv3.md' ahora carga           ║
║      correctamente. Linux es case-sensitive en rutas.            ║
║                                                                  ║
║  [FIX-2] Tokenizador case-insensitive completo                   ║
║      Documentos con "SIERJU" en mayúsculas ahora se indexan      ║
║      como "sierju". El índice de densidad los encuentra.         ║
║                                                                  ║
║  [FIX-3] Enrutador para preguntas SIERJU sin juzgado             ║
║      Si la pregunta menciona SIERJU sin especificar juzgado,     ║
║      devuelve los 3 docs con mayor densidad de "sierju"          ║
║      para dar contexto amplio al modelo.                         ║
║                                                                  ║
║  [FIX-4] Pregunta compuesta: SIERJU + tipo de juzgado            ║
║      "sierju en juzgado civil municipal" → prioriza              ║
║      juzgado_civil_municipal.md sobre los demás.                 ║
║                                                                  ║
║  [FUENTES] Citación obligatoria en cada respuesta (v2.1.1)       ║
║  [ENRUTADOR] TF-IDF + densidad + vocabulario (v2.1.1)            ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os, re, glob, json, math, threading, time, requests
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
VERSION                = "2.1.2"
MAX_OLLAMA_SIMULTANEOS = 2
HILOS_SERVIDOR         = 16
TIMEOUT_CONEXION       = 8
TIMEOUT_RESPUESTA      = 180
TIMEOUT_HEALTH         = 5
CARPETA_FUENTES        = "/opt/siaa/fuentes"
MAX_CHARS_FRAGMENTO    = 2500
MAX_DOCS_CONTEXTO      = 3
TOP_KEYWORDS_POR_DOC   = 20
MIN_FREQ_KEYWORD       = 2
MIN_LEN_KEYWORD        = 4

# ================================================================
#  SYSTEM PROMPTS DUALES — con citación de fuentes obligatoria
# ================================================================

SYSTEM_CONVERSACIONAL = """Eres SIAA (Sistema Inteligente de Apoyo Administrativo), el asistente de la Seccional Bucaramanga de la Rama Judicial de Colombia.

Tu misión es optimizar y agilizar los procesos administrativos de la Seccional.
Responde con cordialidad y naturalidad en español formal.
Para saludos, despedidas o preguntas generales sobre ti mismo, responde directamente.
Recuerda amablemente que puedes ayudar con consultas sobre procesos judiciales, administrativos y normativos."""

SYSTEM_DOCUMENTAL = """Eres SIAA (Sistema Inteligente de Apoyo Administrativo), asistente especializado de la Seccional Bucaramanga de la Rama Judicial de Colombia.

REGLAS ABSOLUTAS — SIN EXCEPCIÓN:

1. Responde ÚNICAMENTE con información del contexto [DOC:...] proporcionado.
2. Si el contexto no contiene la respuesta, responde EXACTAMENTE:
   "No encontré esa información en los documentos disponibles.
   📄 Fuente: Sin documentos encontrados"
3. Responde en español formal y preciso.
4. Máximo 8 líneas de respuesta — conciso pero completo.
5. PROHIBIDO inventar, suponer o completar con conocimiento propio.
6. Si hay pasos o procedimientos, lístalos en orden numerado.
7. Si la información viene de múltiples documentos, indica cuál corresponde a cada parte.

REGLA DE CITACIÓN — ABSOLUTAMENTE OBLIGATORIA:
La ÚLTIMA línea de CADA respuesta DEBE ser exactamente:
📄 Fuente: [nombre del documento tal como aparece en el contexto entre corchetes]

Ejemplos correctos:
  📄 Fuente: JUZGADO CIVIL MUNICIPAL
  📄 Fuentes: JUZGADO CIVIL MUNICIPAL, CADENA CONTROL PROCESO MODULOSV3

NUNCA termines una respuesta sin la línea 📄 Fuente:"""


# ── Patrones de conversación ─────────────────────────────────────
PATRONES_CONVERSACION = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "buen día",
    "buenas", "hey", "saludos", "qué tal", "como estas", "cómo estás",
    "como está", "cómo está usted", "cómo le va",
    "adiós", "adios", "hasta luego", "chao", "chau", "nos vemos",
    "hasta pronto", "hasta mañana",
    "gracias", "muchas gracias", "mil gracias", "muy amable",
    "de nada", "con gusto", "a la orden",
    "quién eres", "quien eres", "qué eres", "que eres", "qué es siaa",
    "que es siaa", "qué hace siaa", "que hace siaa", "para qué sirves",
    "para que sirves", "qué puedes", "que puedes", "cómo me ayudas",
    "ok", "bien", "entendido", "de acuerdo", "claro", "perfecto", "listo",
]


def es_conversacion_general(texto: str) -> bool:
    t = texto.lower().strip()
    if len(t) < 15:
        return True
    return any(p in t for p in PATRONES_CONVERSACION)


# ================================================================
#  MONITOR DE OLLAMA + WARM-UP
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
            print(f"  [Ollama] Warm-up falló (no crítico): {e}", flush=True)
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
#  [FIX-1] CARGA DE DOCUMENTOS — soporte para nombres con
#  mayúsculas, espacios y caracteres especiales en Linux.
#
#  El problema: 'CADENA CONTROL PROCESO MODULOSv3.md' tiene
#  mayúsculas y espacios. glob lo encuentra, pero la clave
#  interna se normaliza a minúsculas para búsquedas consistentes.
#
#  NORMALIZACIÓN:
#    nombre_original: 'CADENA CONTROL PROCESO MODULOSv3.md'
#    nombre_clave:    'cadena control proceso modulosv3.md'
#    nombre_display:  'CADENA CONTROL PROCESO MODULOSV3'
# ================================================================

# ── TF-IDF — stopwords español ───────────────────────────────────
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
    "primera", "segundo", "tercero", "siguiente", "anterior",
    "general", "especial", "nacional", "local", "dicho",
}


def tokenizar(texto: str) -> list:
    """
    [FIX-2] Tokeniza con lowercase completo.
    'SIERJU' en el doc → 'sierju' en el índice.
    Permite buscar 'sierju' en documentos que usan mayúsculas.
    """
    # lower() antes de findall: captura SIERJU, Sierju, sierju igual
    palabras = re.findall(r'\b[a-záéíóúüñ]{4,}\b', texto.lower())
    return [p for p in palabras if p not in STOPWORDS_ES]


def calcular_tfidf_coleccion(documentos: dict) -> dict:
    if not documentos:
        return {}

    tokens_por_doc = {n: tokenizar(d["contenido"]) for n, d in documentos.items()}
    N = len(documentos)

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


# ── Estado global ────────────────────────────────────────────────
colecciones          = {}
colecciones_lock     = threading.Lock()
documentos_cargados  = {}
documentos_lock      = threading.Lock()
indice_densidad      = {}
indice_densidad_lock = threading.Lock()


def cargar_documentos():
    """
    Carga todos los documentos con soporte completo para:
    - Nombres con mayúsculas: 'CADENA CONTROL...' → clave lowercase
    - Nombres con espacios: manejados por glob normalmente
    - Subcarpetas: cada subcarpeta = colección independiente
    """
    global documentos_cargados, colecciones, indice_densidad

    if not os.path.exists(CARPETA_FUENTES):
        os.makedirs(CARPETA_FUENTES, exist_ok=True)
        print(f"  [Docs] Carpeta creada: {CARPETA_FUENTES}")
        return

    nuevas_colecciones = {}
    todos_los_docs     = {}

    # Descubrir colecciones
    subcarpetas = [("general", CARPETA_FUENTES)]
    try:
        for entrada in os.scandir(CARPETA_FUENTES):
            if entrada.is_dir():
                subcarpetas.append((entrada.name.lower(), entrada.path))
    except Exception as e:
        print(f"  [Docs] Error escaneando carpetas: {e}")

    for nombre_col, ruta_col in subcarpetas:
        # [FIX-1] glob con ** no es necesario aquí; usamos os.listdir
        # para capturar archivos con cualquier capitalización
        try:
            if nombre_col == "general":
                # Solo archivos directamente en CARPETA_FUENTES
                archivos = [
                    os.path.join(ruta_col, f)
                    for f in os.listdir(ruta_col)
                    if os.path.isfile(os.path.join(ruta_col, f))
                    and f.lower().endswith(('.md', '.txt'))
                ]
            else:
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
                # [FIX-1] nombre_original conserva la capitalización real
                nombre_original = os.path.basename(ruta)
                # nombre_clave: siempre minúsculas para búsquedas consistentes
                nombre_clave    = nombre_original.lower()

                with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
                    contenido = f.read()

                # [FIX-2] palabras en lowercase para búsqueda case-insensitive
                palabras     = set(re.findall(r'\b[a-záéíóúüñ]{4,}\b', contenido.lower()))
                tokens       = tokenizar(contenido)
                token_count  = Counter(tokens)
                total_tokens = len(tokens)

                doc_entry = {
                    "ruta":             ruta,
                    "nombre_original":  nombre_original,  # Para display
                    "contenido":        contenido,
                    "palabras":         palabras,
                    "tamano":           len(contenido),
                    "coleccion":        nombre_col,
                    "token_count":      token_count,
                    "total_tokens":     total_tokens,
                }

                # Usar nombre_clave como índice (lowercase siempre)
                docs_col[nombre_clave]       = doc_entry
                todos_los_docs[nombre_clave] = doc_entry
                print(f"  [Doc] [{nombre_col}] {nombre_original} ({len(contenido):,} chars)")

            except Exception as e:
                print(f"  [Doc] Error leyendo {ruta}: {e}")

        if not docs_col:
            continue

        print(f"  [KW]  Calculando TF-IDF para colección '{nombre_col}' ({len(docs_col)} docs)...")
        keywords_col = calcular_tfidf_coleccion(docs_col)

        # Log muestra de keywords (primeros 3 docs)
        for nd, kws in list(keywords_col.items())[:3]:
            print(f"  [KW]  {nd}: {kws[:6]}")

        nuevas_colecciones[nombre_col] = {
            "docs":     docs_col,
            "keywords": keywords_col,
        }

    # Construir índice de densidad
    nuevo_indice = defaultdict(list)
    for nombre_doc, doc in todos_los_docs.items():
        total = doc["total_tokens"]
        if total == 0:
            continue
        for termino, freq in doc["token_count"].items():
            if len(termino) >= MIN_LEN_KEYWORD:
                densidad = freq / total
                nuevo_indice[termino].append((densidad, nombre_doc))

    for termino in nuevo_indice:
        nuevo_indice[termino].sort(reverse=True)

    with colecciones_lock:
        colecciones = nuevas_colecciones
    with documentos_lock:
        documentos_cargados = todos_los_docs
    with indice_densidad_lock:
        indice_densidad = dict(nuevo_indice)

    total_docs = len(todos_los_docs)
    print(f"  [Doc] Total: {total_docs} docs en {len(nuevas_colecciones)} colecciones ✓")
    print(f"  [IDX] Índice densidad: {len(nuevo_indice):,} términos ✓")


# ================================================================
#  ENRUTADOR MULTI-NIVEL
#
#  Nivel 1 — TF-IDF: términos exclusivos de un doc
#  Nivel 2 — Densidad: términos transversales (SIERJU, proceso...)
#  Nivel 3 — Vocabulario: fallback general
# ================================================================

def detectar_documentos(pregunta: str) -> list:
    p = pregunta.lower()
    palabras_pregunta  = set(re.findall(r'\b[a-záéíóúüñ]{4,}\b', p))
    palabras_filtradas = [w for w in palabras_pregunta if w not in STOPWORDS_ES]

    with colecciones_lock:
        snap_cols = dict(colecciones)
    with documentos_lock:
        snap_docs = dict(documentos_cargados)
    with indice_densidad_lock:
        snap_idx = dict(indice_densidad)

    N = len(snap_docs) or 1

    # ── Nivel 1: TF-IDF ──────────────────────────────────────────
    scores_tfidf = defaultdict(float)
    for col in snap_cols.values():
        for nombre_doc, keywords in col.get("keywords", {}).items():
            if nombre_doc not in snap_docs:
                continue
            for kw in keywords:
                if kw in p:
                    scores_tfidf[nombre_doc] += 1.0

    # ── Nivel 2: Densidad ─────────────────────────────────────────
    scores_densidad = defaultdict(float)
    for termino in palabras_filtradas:
        if termino not in snap_idx:
            continue
        df_termino = len(snap_idx[termino])
        idf_aprox  = math.log((N + 1) / (df_termino + 1)) + 1
        for densidad, nombre_doc in snap_idx[termino][:5]:
            scores_densidad[nombre_doc] += densidad * idf_aprox

    # ── Combinar Nivel 1 + 2 ─────────────────────────────────────
    scores_combinados = defaultdict(float)
    for doc, s in scores_tfidf.items():
        scores_combinados[doc] += s * 2.0
    for doc, s in scores_densidad.items():
        scores_combinados[doc] += s

    if scores_combinados:
        ordenados = sorted(
            scores_combinados.keys(),
            key=lambda d: scores_combinados[d],
            reverse=True
        )
        resultado = ordenados[:MAX_DOCS_CONTEXTO]
        log_scores = [(d, round(scores_combinados[d], 4)) for d in resultado]
        print(f"  [ENRUTADOR] {log_scores}", flush=True)
        return resultado

    # ── Nivel 3: Vocabulario ──────────────────────────────────────
    scored = []
    for nombre, doc in snap_docs.items():
        coincidencias = len(palabras_pregunta & doc["palabras"])
        if coincidencias >= 2:
            scored.append((coincidencias, nombre))
    scored.sort(reverse=True)
    return [n for _, n in scored[:MAX_DOCS_CONTEXTO]]


# ================================================================
#  EXTRACTOR DE FRAGMENTOS v2
# ================================================================

TERMINOS_PRIORITARIOS_BASE = {
    "procedimiento", "pasos", "proceso", "tramite", "trámite",
    "requisito", "requisitos", "ingreso", "registro", "módulo",
    "modulo", "sistema", "diligencia", "diligenciar", "registrar",
    "sierju", "siglo", "módulos",
}


def obtener_terminos_prioritarios(nombre_doc: str) -> set:
    terminos = set(TERMINOS_PRIORITARIOS_BASE)
    with colecciones_lock:
        snap = dict(colecciones)
    for col in snap.values():
        if nombre_doc in col.get("keywords", {}):
            terminos.update(col["keywords"][nombre_doc][:10])
            break
    return terminos


def extraer_fragmento(nombre_doc: str, pregunta: str) -> str:
    """
    Extrae los fragmentos más relevantes del documento.
    Usa el nombre_clave (lowercase) para buscar en el índice.
    Usa nombre_original para la etiqueta [DOC: ...] que verá el modelo.
    """
    with documentos_lock:
        doc = documentos_cargados.get(nombre_doc)
    if not doc:
        return ""

    palabras      = set(re.findall(r'\b[a-záéíóúüñ]{4,}\b', pregunta.lower()))
    pregunta_norm = pregunta.lower().replace('?', '').replace('¿', '').strip()
    parrafos      = re.split(r'\n{1,}', doc["contenido"])
    terminos_prio = obtener_terminos_prioritarios(nombre_doc)

    scored = []
    for idx, parrafo in enumerate(parrafos):
        if len(parrafo.strip()) < 10:
            continue
        # [FIX-2] comparar en lowercase para capturar SIERJU y sierju igual
        p_lower = parrafo.lower()
        puntos  = 0

        for w in palabras:
            if w in p_lower:
                puntos += 2 if w in terminos_prio else 1

        if parrafo.strip().startswith('#') and puntos > 0:
            puntos += 6
        if pregunta_norm in p_lower:
            puntos += 15
        if puntos > 0 and re.search(r'^\s*\d+[\.\)]\s+\S', parrafo, re.MULTILINE):
            puntos += 4

        if puntos > 0:
            scored.append((puntos, idx, parrafo))

    scored.sort(reverse=True)

    fragmento      = []
    chars_usados   = 0
    indices_usados = set()

    for _, idx, parrafo in scored[:5]:
        if idx > 0 and (idx - 1) not in indices_usados:
            prev = parrafos[idx - 1]
            if prev.strip() and chars_usados + len(prev) <= MAX_CHARS_FRAGMENTO:
                fragmento.append(prev)
                indices_usados.add(idx - 1)
                chars_usados += len(prev)

        if idx not in indices_usados and chars_usados + len(parrafo) <= MAX_CHARS_FRAGMENTO:
            fragmento.append(parrafo)
            indices_usados.add(idx)
            chars_usados += len(parrafo)

        for offset in [1, 2]:
            idx_sig = idx + offset
            if (idx_sig < len(parrafos)
                    and idx_sig not in indices_usados
                    and chars_usados + len(parrafos[idx_sig]) <= MAX_CHARS_FRAGMENTO):
                fragmento.append(parrafos[idx_sig])
                indices_usados.add(idx_sig)
                chars_usados += len(parrafos[idx_sig])

        if chars_usados >= MAX_CHARS_FRAGMENTO * 0.85:
            break

    if not fragmento:
        fragmento = [doc["contenido"][:MAX_CHARS_FRAGMENTO]]

    # Usar nombre_original para la etiqueta (lo que verá el modelo y citará)
    nombre_original = doc.get("nombre_original", nombre_doc)
    nombre_display  = os.path.splitext(nombre_original)[0].upper()
    coleccion       = doc.get("coleccion", "")
    etiqueta        = f"[DOC: {nombre_display}]" if not coleccion or coleccion == "general" \
                      else f"[DOC: {nombre_display} | {coleccion.upper()}]"

    return etiqueta + "\n" + "\n".join(fragmento) + "\n\n"


# ================================================================
#  CLIENTE OLLAMA
# ================================================================

def llamar_ollama(mensajes: list, thinking: bool = False) -> list:
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
                "think":    thinking,
                "options":  {
                    "temperature":    0.1,
                    "num_predict":    400,    # Aumentado para incluir línea 📄 Fuente
                    "num_ctx":        1536,
                    "num_thread":     10,
                    "num_batch":      256,
                    "repeat_penalty": 1.1,
                    "stop": [
                        "\n\n\n",
                        "Espero que",
                        "Es importante destacar",
                        "Cabe destacar que",
                        "En conclusión,",
                        "En resumen,",
                        "Por otro lado,",
                        "Cabe mencionar",
                        "Finalmente,",
                    ]
                }
            },
            stream=True,
            timeout=(TIMEOUT_CONEXION, TIMEOUT_RESPUESTA)
        )

        if resp.status_code != 200:
            return [f"ERROR_HTTP_{resp.status_code}"]

        chunks        = []
        hubo_thinking = False

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                obj          = json.loads(line.decode("utf-8"))
                thinking_tok = obj.get("thinking", "")
                content_tok  = obj.get("message", {}).get("content", "")
                done         = obj.get("done", False)

                if thinking_tok:
                    hubo_thinking = True
                    safe = json.dumps(thinking_tok)[1:-1]
                    chunks.append(f'data: {{"choices":[{{"delta":{{"thinking":"{safe}"}}}}]}}')

                if content_tok:
                    safe = json.dumps(content_tok)[1:-1]
                    chunks.append(f'data: {{"choices":[{{"delta":{{"content":"{safe}"}}}}]}}')

                if done:
                    if hubo_thinking:
                        chunks.append('data: {"choices":[{"delta":{"thinking_end":true}}]}')
                    chunks.append("data: [DONE]")

            except Exception as e:
                print(f"[PARSER] Error: {e} | {str(line)[:80]}", flush=True)

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
            docs_relevantes = detectar_documentos(ultima_pregunta)
            for nombre_doc in docs_relevantes:
                contexto += extraer_fragmento(nombre_doc, ultima_pregunta)

        print(
            f"[CHAT] tipo={'CONV' if es_conv else 'DOC'} "
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
                        "No se encontraron documentos específicos para esta consulta. "
                        "Debes responder EXACTAMENTE: "
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

        usar_thinking = False

        def generate():
            try:
                chunks = llamar_ollama(ollama_msgs, thinking=usar_thinking)

                if not chunks:
                    yield 'data: {"choices":[{"delta":{"content":"Sin respuesta del modelo."}}]}\n\n'
                    return

                ERRORES = {
                    "COLA_LLENA":        "⏳ Sistema ocupado. Intente en 30 segundos.",
                    "TIMEOUT_CONEXION":  "⚠ IA no responde. Intente de nuevo.",
                    "TIMEOUT_RESPUESTA": "⏱ Consulta tomó demasiado tiempo. Intente de nuevo.",
                    "OLLAMA_CAIDO":      "⚠ Servidor IA reiniciándose. Espere 1 minuto.",
                }

                if chunks[0] in ERRORES:
                    yield f'data: {{"choices":[{{"delta":{{"content":"{ERRORES[chunks[0]]}"}}}}]}}\n\n'
                    return

                if chunks[0].startswith("ERROR"):
                    print(f"[CHAT] Error Ollama: {chunks[0]}", flush=True)
                    yield 'data: {"choices":[{"delta":{"content":"⚠ Error interno. Contacte al administrador."}}]}\n\n'
                    return

                for chunk in chunks:
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
        "indice_terminos":   total_terminos,
        "colecciones":       info_cols,
    })


@app.route("/siaa/keywords/<nombre_doc>", methods=["GET"])
def ver_keywords(nombre_doc: str):
    """Ver keywords TF-IDF de un documento. Ejemplo:
    curl http://IP:5000/siaa/keywords/juzgado_civil_municipal.md
    """
    nombre_doc = nombre_doc.lower().replace("+", " ")
    with colecciones_lock:
        for col in colecciones.values():
            if nombre_doc in col.get("keywords", {}):
                return jsonify({
                    "documento": nombre_doc,
                    "keywords":  col["keywords"][nombre_doc],
                    "total":     len(col["keywords"][nombre_doc])
                })
    return jsonify({"error": f"'{nombre_doc}' no encontrado"}), 404


@app.route("/siaa/densidad/<termino>", methods=["GET"])
def ver_densidad(termino: str):
    """
    Diagnóstico: qué documentos mencionan más un término.
    Imprescindible para depurar términos transversales como 'sierju'.
    Ejemplo: curl http://IP:5000/siaa/densidad/sierju
    """
    termino = termino.lower()
    with indice_densidad_lock:
        entradas = indice_densidad.get(termino, [])
    if not entradas:
        return jsonify({"error": f"Término '{termino}' no en índice"}), 404
    return jsonify({
        "termino":    termino,
        "top_docs":   [{"doc": d, "densidad": round(den, 6)} for den, d in entradas[:10]],
        "total_docs": len(entradas)
    })


@app.route("/siaa/recargar", methods=["GET"])
def recargar():
    """Recarga docs y regenera keywords + índice de densidad.
    Usar cuando se agreguen o modifiquen archivos .md.
    """
    cargar_documentos()
    with colecciones_lock:
        info = {n: len(c["docs"]) for n, c in colecciones.items()}
    return jsonify({
        "recargado":  True,
        "colecciones": info,
        "total_docs":  sum(info.values())
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
    print(f"  Modelo:       {MODEL}")
    print(f"  Fuentes:      {CARPETA_FUENTES}")
    print(f"  Contexto/doc: {MAX_CHARS_FRAGMENTO} chars")
    print(f"  Hilos CPU:    10 / 12  (Ryzen PRO 2600)")
    print(f"  Keywords:     TF-IDF automático + índice densidad")
    print(f"  Fuentes resp: Obligatorias (📄 Fuente: ...)")
    print("=" * 62)

    cargar_documentos()
    verificar_ollama()

    with ollama_lock:
        estado_str = "OK ✓" if ollama_estado["disponible"] else "NO DISPONIBLE ✗"
    print(f"  Ollama: {estado_str}")
    print("=" * 62)

    try:
        from waitress import serve
        print("  Servidor: Waitress — 0.0.0.0:5000")
        serve(app, host="0.0.0.0", port=5000,
              threads=HILOS_SERVIDOR, channel_timeout=200)
    except ImportError:
        print("  Usando Flask desarrollo...")
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)