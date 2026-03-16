"""
SIAA — Extractor de fragmentos relevantes (RAG Layer 2).

Recibe un documento y la pregunta, devuelve los chunks más relevantes.

Sistema de puntuación por chunk:
  +TF×IDF  por cada keyword encontrada (términos raros pesan más)
  +15      si la pregunta completa aparece en el chunk
  +10      si el chunk contiene artículo con grado (art. 5°)
  +5       si contiene artículo simple (artículo 5)
  +4       si contiene lista numerada
  +20/12/6 bono de proximidad (densidad de keywords en ventana 150 chars)
  +bonus   por frase exacta de 2-4 palabras

Estrategia de poda dinámica (Francotirador vs Escopeta):
  ratio ≥ 3.0 → 1 chunk  (certeza alta)
  ratio ≥ 1.8 → 2 chunks (certeza media)
  ratio < 1.8 → 3 chunks (respuesta distribuida)
  listado     → mínimo 2 chunks siempre
"""
import math
import re
from config import CHUNK_SIZE, MAX_CHUNKS_CONTEXTO
from rag.tokenizer import tokenizar, normalizar
from rag.document_store import store

# ── Patrones de artículo ─────────────────────────────────────────
_PAT_ART_GRADO  = re.compile(r'art[íi]culo\s+\d+[\s°ºo]|art\.\s*\d+[\s°ºo]', re.IGNORECASE)
_PAT_ART_SIMPLE = re.compile(r'art[íi]culo\s+\d+|art\.\s*\d+', re.IGNORECASE)

# ── Términos prioritarios base ────────────────────────────────────
_TERMINOS_PRIO = {
    "procedimiento", "pasos", "proceso", "tramite", "trámite",
    "requisito", "ingreso", "registro", "módulo", "modulo",
    "sistema", "diligencia", "diligenciar", "registrar",
    "sierju", "formulario", "artículo", "articulo",
    "funcionarios", "responsables", "acuerdo", "resolución",
    "plazo", "fecha", "periodicidad", "hábil", "habiles",
    "trimestre", "período", "dias", "quinto", "corte",
    "vencimiento", "reportar", "reporte",
}

# ── Patrones de pregunta de listado ──────────────────────────────
_PATRONES_LISTADO = [
    "cuáles son", "cuales son", "qué secciones", "que secciones",
    "qué partes", "que partes", "qué tipos", "que tipos",
    "enumera", "enumere", "nombra", "menciona",
    "qué incluye", "que incluye", "qué contiene", "que contiene",
    "qué campos", "que campos", "cuáles son los", "cuales son los",
]


def _es_listado(pregunta: str) -> bool:
    p = pregunta.lower()
    return any(pat in p for pat in _PATRONES_LISTADO)


def _expandir_query(palabras: set, pregunta: str) -> set:
    """
    Agrega términos semánticamente relacionados según el tipo de pregunta.
    El usuario dice 'cuándo' pero el doc dice 'periodicidad' y 'plazo'.
    """
    p = pregunta.lower()
    expandidas = set(palabras)

    if any(w in p for w in ["cuándo", "cuando", "plazo", "fecha", "término"]):
        expandidas.update(["periodicidad", "plazo", "habil", "habiles",
                           "trimestre", "vencimiento", "quinto", "corte"])

    if any(w in p for w in ["quién", "quien", "responsable", "cargo"]):
        expandidas.update(["responsable", "funcionario", "competencia",
                           "administrador", "seccional"])

    if any(w in p for w in ["cómo", "como", "pasos", "procedimiento"]):
        expandidas.update(["procedimiento", "pasos", "diligenciar",
                           "registrar", "formulario", "modulo"])

    if any(w in p for w in ["qué es", "que es", "para qué", "para que",
                             "explica", "define", "describir"]):
        expandidas.update(["sistema", "herramienta", "estadistico",
                           "informacion", "instrumento", "judicial",
                           "recoleccion", "registro", "datos"])

    if any(w in p for w in ["sanción", "sancion", "incumplimiento",
                             "consecuencia", "que pasa", "disciplinario"]):
        expandidas.update(["sancion", "disciplinaria", "incumplimiento",
                           "acarreara", "codigo", "sanciones"])

    if any(w in p for w in ["cargar", "diligenciar", "quien carga",
                             "responsable", "puede"]):
        expandidas.update(["funcionario", "magistrado", "juez",
                           "rol", "roles", "despacho", "corresponde"])

    if any(w in p for w in ["campo", "sección", "seccion", "apartado",
                             "columna", "que informacion", "que dato"]):
        expandidas.update(["campo", "apartado", "columna", "formulario",
                           "dato", "clase", "proceso", "inventario",
                           "ingreso", "egreso", "efectivo"])

    return expandidas


def _puntuar(chunk: dict, palabras: set, pregunta_norm: str,
             terminos_prio: set, idf_local: dict) -> float:
    texto  = normalizar(chunk["texto"].lower())
    puntos = 0.0

    for w in palabras:
        count = texto.count(w)
        if count > 0:
            tf   = 1.0 + math.log(count)
            base = 3.0 if w in terminos_prio else 1.0
            if w in idf_local:
                base *= idf_local[w]
            puntos += tf * base

    if pregunta_norm in texto:
        puntos += 15.0
    if _PAT_ART_GRADO.search(chunk["texto"]):
        puntos += 10.0
    elif _PAT_ART_SIMPLE.search(chunk["texto"]):
        puntos += 5.0
    if re.search(r'^\s*\d+[\.\)]\s+\S', chunk["texto"], re.MULTILINE):
        puntos += 4.0

    # Bono de proximidad — densidad de keywords en ventana de 150 chars
    if len(palabras) >= 2:
        n_pal = len(palabras)
        max_d = 0.0
        for i in range(0, max(1, len(texto) - 150), 50):
            v = texto[i:i+150]
            m = sum(1 for w in palabras if w in v)
            if m >= 2:
                d = m / n_pal
                if d > max_d:
                    max_d = d
        if   max_d >= 0.90: puntos += 20.0
        elif max_d >= 0.70: puntos += 12.0
        elif max_d >= 0.50: puntos +=  6.0
        elif max_d >= 0.30: puntos +=  2.0

    return puntos


def extraer(nombre_doc: str, pregunta: str) -> str:
    """
    Extrae el fragmento más relevante del documento para la pregunta.

    Returns:
        Texto formateado con metadatos de sección, listo para el LLM.
        Ejemplo:
          [DOC: JUZGADO_CIVIL_MUNICIPAL | SIERJU]
          [SEC: INVENTARIO FINAL | CHUNK: 42]
          El inventario final corresponde a...
    """
    doc    = store.get_doc(nombre_doc)
    chunks = store.get_chunks(nombre_doc)
    if not doc or not chunks:
        return ""

    # Preparar tokens con query expansion
    palabras      = set(tokenizar(pregunta.lower()))
    pregunta_norm = normalizar(pregunta.lower().replace('?','').replace('¿','').strip())
    palabras      = _expandir_query(palabras, pregunta)

    # IDF local — términos raros en este doc específico pesan más
    total_chunks = max(len(chunks), 1)
    idf_local: dict = {}
    for w in palabras:
        chunks_con_w = sum(1 for c in chunks if w in c["texto"].lower())
        if 0 < chunks_con_w <= total_chunks * 0.85:
            idf_local[w] = math.log((total_chunks + 1) / (chunks_con_w + 1)) + 1.0

    # Términos prioritarios del doc
    terminos_prio = set(_TERMINOS_PRIO)
    cols = store.get_colecciones()
    for col in cols.values():
        if nombre_doc in col.get("keywords", {}):
            terminos_prio.update(col["keywords"][nombre_doc][:10])
            break

    # Bonus por frases exactas de 2-4 palabras
    frases: list = []
    ptokens = pregunta_norm.lower().split()
    for n in range(4, 1, -1):
        for i in range(len(ptokens) - n + 1):
            frase = " ".join(ptokens[i:i+n])
            if len(frase) > 6:
                frases.append(frase)

    # Puntuar todos los chunks
    scored: list = []
    for chunk in chunks:
        pts = _puntuar(chunk, palabras, pregunta_norm, terminos_prio, idf_local)
        for frase in frases:
            if frase in chunk["texto"].lower():
                pts += len(frase.split()) * 12.0
                break
        if pts > 0:
            scored.append((pts, chunk["indice"], chunk))
    scored.sort(reverse=True)

    # Position bias para preguntas de definición (inicio del doc)
    es_def = any(w in pregunta.lower() for w in [
        "qué es", "que es", "para qué", "para que", "explica", "define"
    ])
    if es_def and scored:
        scored = [(pts + max(0.0, (5 - idx) * 3.0), idx, c)
                  for pts, idx, c in scored]
        scored.sort(reverse=True)

    # Poda dinámica: Francotirador / Binóculo / Escopeta
    es_listado = _es_listado(pregunta)
    max_chunks = MAX_CHUNKS_CONTEXTO
    if len(chunks) > 80:
        max_chunks = 4

    if len(scored) >= 2:
        s1, s2 = scored[0][0], scored[1][0]
        ratio  = s1 / max(s2, 0.01)
        if es_listado:
            chunks_usar = max_chunks if ratio < 1.8 else 2
            modo = "LISTADO"
        elif ratio >= 3.0:
            chunks_usar = 1
            modo = "FRANCOTIRADOR"
        elif ratio >= 1.8:
            chunks_usar = 2
            modo = "BINOCULO"
        else:
            chunks_usar = max_chunks
            modo = "ESCOPETA"
        print(f"  [Extractor] {nombre_doc[:25]} ratio={ratio:.2f} [{modo}]", flush=True)
    else:
        chunks_usar = min(len(scored), MAX_CHUNKS_CONTEXTO)

    # Seleccionar los mejores chunks sin duplicados
    seleccionados: list = []
    usados:        set  = set()
    chars_acum         = 0

    for pts, idx, chunk in scored[:chunks_usar * 2]:
        if idx in usados:
            continue
        if chars_acum + len(chunk["texto"]) > CHUNK_SIZE * chunks_usar:
            break
        meta = f"[SEC: {chunk['seccion'][:60]} | CHUNK: {idx}]"
        seleccionados.append(meta + "\n" + chunk["texto"])
        usados.add(idx)
        chars_acum += len(meta) + len(chunk["texto"])
        if len(seleccionados) >= chunks_usar:
            break

    # Fallback: si no encontró nada, usar los primeros chunks
    if not seleccionados:
        for chunk in chunks[:2]:
            meta = f"[SEC: {chunk['seccion'][:60]} | CHUNK: {chunk['indice']}]"
            seleccionados.append(meta + "\n" + chunk["texto"])

    nombre_display = doc.get("nombre_original", nombre_doc).replace(".md","").upper()
    coleccion      = doc.get("coleccion", "")
    etiqueta = (f"[DOC: {nombre_display}]" if not coleccion or coleccion == "general"
                else f"[DOC: {nombre_display} | {coleccion.upper()}]")

    separador = "\n" + "═" * 60 + "\n"
    return etiqueta + "\n" + "\n\n".join(seleccionados) + separador
