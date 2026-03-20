"""
SIAA — Enrutador multi-nivel de documentos (RAG Layer 1).

Decide qué documentos consultar antes de extraer fragmentos.
Sin el enrutador, habría que buscar en los 63 documentos — lento.

Niveles de scoring (en orden de prioridad):
  1. TF-IDF automático + keywords manuales  (peso ×2.0)
  2. Densidad del término en el documento   (peso ×1.0)
  3. Coincidencia con nombre de archivo     (peso ×1.5)

Score final = TF-IDF×2 + densidad×1 + nombre×1.5
Devuelve los top MAX_DOCS_CONTEXTO documentos.
"""
import math
import re
from collections import defaultdict
from config import MAX_DOCS_CONTEXTO
from rag.tokenizer import tokenizar, normalizar, STOPWORDS_ES
from rag.document_store import store

# ── Keywords manuales ────────────────────────────────────────────
# Complementan el TF-IDF automático para términos específicos del dominio
# que no alcanzan el top-20 TF-IDF por baja frecuencia absoluta.
# Peso doble (×2) porque son más específicas que las automáticas.
KEYWORDS_MANUALES: dict[str, list[str]] = {
    "faq_acuerdo_pcsja19-11207.md": [
        "capacitacion", "capacita", "capacitar", "quien capacita",
        "cendoj", "udae", "unidad de desarrollo", "analisis estadistico",
        "presentacion de informes", "primer informe",
    ],
    "faq_acuerdo_psaa16-10476.md": [
        "que es sierju", "que es el sierju", "para que sirve", "objeto",
        "sistema de informacion", "recoleccion de informacion", "estadistica",
        "de que trata", "proposito", "descripcion", "finalidad",
        "sierju", "periodicidad", "reportar", "quinto dia habil",
        "formulario", "recoleccion", "discrepancia",
        "roles", "rol", "super administrador", "administrador nacional",
        "administrador seccional", "funcionario", "juez", "magistrado",
        "responsable", "quien carga", "quien diligencia",
        "sancion", "sanción", "incumplimiento", "no reporto",
        "que pasa", "consecuencia", "disciplinario", "no reportar",
    ],
}

# ── Patrón para detectar documento específico ───────────────────
_PATRON_DOC_ESP = re.compile(
    r'\b(psaa|pcsja|acuerdo|circular|resolución|resolucion|decreto)\s*[\w\-\.]+',
    re.IGNORECASE
)


def es_doc_especifico(pregunta: str) -> bool:
    """True si la pregunta menciona un documento concreto por nombre."""
    return bool(_PATRON_DOC_ESP.search(pregunta))


def detectar(pregunta: str, max_docs: int = MAX_DOCS_CONTEXTO) -> list[str]:
    """
    Encuentra los documentos más relevantes para la pregunta.

    Returns:
        Lista de nombres de archivo ordenados por relevancia.
        Ejemplo: ['juzgado_civil_municipal.md', 'faq_acuerdo_psaa16-10476.md']
    """
    p = normalizar(pregunta.lower())

    palabras_pregunta  = set(tokenizar(p))
    palabras_3plus     = set(re.findall(r'\b[a-záéíóúüñ0-9]{3,}\b', p))
    palabras_filtradas = [w for w in palabras_pregunta if w not in STOPWORDS_ES]

    snap_cols = store.get_colecciones()
    snap_docs = store.get_todos_docs()
    snap_dens = store.get_densidad()
    N = len(snap_docs) or 1

    # ── Nivel 1: TF-IDF automático ───────────────────────────────
    scores_tfidf: dict = defaultdict(float)
    for col in snap_cols.values():
        for nombre_doc, keywords in col.get("keywords", {}).items():
            if nombre_doc not in snap_docs:
                continue
            for kw in keywords:
                if kw in p:
                    scores_tfidf[nombre_doc] += 1.0

    # Bonus keywords manuales (peso ×2)
    for nombre_doc, kws in KEYWORDS_MANUALES.items():
        if nombre_doc in snap_docs:
            for kw in kws:
                if kw in p:
                    scores_tfidf[nombre_doc] += 2.0

    # ── Nivel 2: Densidad ────────────────────────────────────────
    scores_dens: dict = defaultdict(float)
    for termino in palabras_filtradas:
        if termino not in snap_dens:
            continue
        df_t      = len(snap_dens[termino])
        idf_aprox = math.log((N + 1) / (df_t + 1)) + 1
        for densidad, nombre_doc in snap_dens[termino][:5]:
            scores_dens[nombre_doc] += densidad * idf_aprox

    # ── Nivel 3: Nombre de archivo ───────────────────────────────
    scores_nombre: dict = defaultdict(float)
    for nombre_doc, doc in snap_docs.items():
        tokens_nombre = doc.get("tokens_nombre", set())
        coincidencias = tokens_nombre & palabras_3plus
        if coincidencias:
            scores_nombre[nombre_doc] = len(coincidencias) / (len(tokens_nombre) or 1)

    # ── Score combinado ──────────────────────────────────────────
    scores: dict = defaultdict(float)
    for doc, s in scores_tfidf.items():
        scores[doc] += s * 2.0
    for doc, s in scores_dens.items():
        scores[doc] += s * 1.0
    for doc, s in scores_nombre.items():
        scores[doc] += s * 1.5

    if scores:
        ordenados = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        resultado = ordenados[:max_docs]
        print(f"  [Router] {[(d, round(scores[d],2)) for d in resultado]}", flush=True)
        return resultado

    # Fallback: coincidencia de vocabulario básica
    scored = []
    for nombre, doc in snap_docs.items():
        c = len(palabras_pregunta & doc["palabras"])
        if c >= 2:
            scored.append((c, nombre))
    scored.sort(reverse=True)
    return [n for _, n in scored[:max_docs]]

# Actualización: agregar identificadores directos de documentos
KEYWORDS_MANUALES["faq_acuerdo_psaa16-10476.md"].extend([
    "psaa16", "psaa", "10476", "psaa16-10476",
])

# ── Keywords manuales RRHH ───────────────────────────────────────
# Agregadas porque el router no detectaba estas preguntas frecuentes
# del banco de preguntas de los 26 despachos.
KEYWORDS_MANUALES["faq_recursos_humanos.md"] = [
    # Certificaciones
    "certificacion ingresos", "certificado ingresos", "certificacion laboral",
    "certificado laboral", "certificacion salarial", "constancia laboral",
    "certificar ingresos", "carta laboral",
    # Vacaciones
    "solicitar vacaciones", "como pedir vacaciones", "dias de vacaciones",
    "disfrute vacaciones", "vacaciones judiciales", "periodo vacacional",
    # Licencias
    "licencia maternidad", "licencia paternidad", "licencia no remunerada",
    "licencia remunerada", "como solicitar licencia", "tipos de licencia",
    # Permisos
    "permiso remunerado", "permiso sindical", "como solicitar permiso",
    "dias de permiso",
    # Nomina y salario
    "cuando se paga", "fecha pago nomina", "pago salario",
    "como se liquida", "liquidacion cesantias", "cesantias",
    # Prima
    "prima servicios", "prima productividad", "prima navidad",
    "prima vacaciones", "como se calcula prima", "calculo prima",
    # Posesion y vinculacion
    "posesionarse cargo", "documentos posesion", "vinculacion judicial",
    "requisitos posesion", "como posesionarse",
    # Evaluacion
    "evaluacion desempeno", "calificacion servicios", "como evaluan",
    "proceso evaluacion",
    # Traslados
    "solicitar traslado", "traslado judicial", "como pedir traslado",
    # Bienestar
    "bienestar judicial", "programas bienestar", "actividades bienestar",
    "beneficios servidor judicial",
    # Seguridad social
    "eps judicial", "pension judicial", "seguridad social rama",
    "afiliacion eps", "caja compensacion",
    # Escalafon
    "escalafon judicial", "carrera judicial", "ascenso judicial",
    "categoria judicial",
]

# ── Keywords para preguntas sobre el sistema SIAA ───────────────
KEYWORDS_MANUALES["sierju_faq_general.md"] = [
    # Contenido real del doc — preguntas FAQ SIERJU generales
    "udae", "unidad de desarrollo", "analisis estadistico",
    "que es la udae", "para que sirve la udae",
    "sancion", "sanciones", "incumplimiento", "no reportar",
    "que pasa si no reporto", "consecuencias", "disciplinario",
    "administrador seccional", "que hace el administrador",
    "deberes administrador", "rol administrador",
    "salas administrativas", "responsabilidades salas",
    "deberes funcionario", "deberes del funcionario",
    "cuando reportar", "plazo reporte", "cinco dias habiles",
    "periodicidad", "frecuencia reporte",
    "problema tecnico sierju", "falla tecnica",
    "quien capacita", "capacitacion sierju",
    "informacion publica", "es publica",
    "corregir error", "como corrijo", "error sierju",
    "roles sierju", "super administrador",
]

# ── Keywords prestaciones sociales → cap3 ───────────────────────
KEYWORDS_MANUALES["cap3_salarios_prestaciones.md"] = [
    "cesantias", "cesantías", "cuando se pagan cesantias",
    "liquidacion cesantias", "intereses cesantias",
    "prima servicios calculo", "como se calcula prima",
    "calculo liquidacion", "liquidacion definitiva",
    "auxilio cesantias", "fondo cesantias",
    "cuando pagan cesantias", "fecha pago cesantias",
]
