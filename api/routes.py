"""
SIAA — Rutas Flask v3.0.2
Streaming real + sources event + limite de contexto para docs grandes.
"""
import json, os, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, Response, request, stream_with_context, jsonify
from flask_cors import CORS

from llm.classifier import es_conversacion
from llm.client     import (chat_stream, verificar, disponible,
                             SYSTEM_CONVERSACIONAL, SYSTEM_DOCUMENTAL)
from rag.document_store import store
from rag.router         import detectar
from rag.extractor      import extraer

VERSION      = "3.0.2"
LOG_ARCHIVO  = "/opt/siaa/logs/calidad.jsonl"
MAX_CONTEXTO = 3000   # chars maximos al modelo — qwen2.5:3b se pierde con mas

PATRONES_LISTADO = [
    "cuales son", "que secciones", "que campos", "que incluye",
    "enumera", "enumere", "nombra", "menciona", "que documentos",
]

app = Flask(__name__)
CORS(app)


def _nombre_display(nd: str) -> str:
    return os.path.splitext(nd)[0].replace('_',' ').replace('-',' ').title()


def _coleccion_doc(nd: str) -> str:
    doc = store.get_doc(nd)
    return doc.get("coleccion","general") if doc else "general"


def _sse_token(tok: str) -> str:
    return f"data: {json.dumps({'choices':[{'delta':{'content':tok},'finish_reason':None}]})}\n\n"


def _sse_sources(nombres: list) -> str:
    fuentes = [
        {
            "n":         i+1,
            "doc":       nd,
            "nombre":    _nombre_display(nd),
            "coleccion": _coleccion_doc(nd),
            "url":       f"/siaa/ver/{nd}",
        }
        for i, nd in enumerate(nombres)
    ]
    return f"data: {json.dumps({'sources': fuentes})}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _registrar(pregunta, respuesta, docs, tiempo_s, tipo="DOCUMENTAL"):
    entrada = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tipo": tipo, "pregunta": pregunta[:200],
        "respuesta": respuesta[:300], "docs": docs,
        "tiempo_s": round(tiempo_s, 2), "alerta": "OK",
    }
    try:
        os.makedirs(os.path.dirname(LOG_ARCHIVO), exist_ok=True)
        with open(LOG_ARCHIVO, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _construir_contexto(docs_enc: list, pregunta: str) -> tuple[str, list]:
    """
    Construye el contexto RAG respetando MAX_CONTEXTO chars.
    Si el primer documento ya supera el limite, solo lo usa a el.
    """
    partes, nombres = [], []
    chars_total = 0
    for nd in docs_enc:
        frag = extraer(nd, pregunta)
        if not frag:
            continue
        if chars_total + len(frag) > MAX_CONTEXTO and partes:
            # Ya tenemos contexto suficiente, no agregar mas
            print(f"  [Routes] Contexto limitado a {chars_total} chars — omitiendo {nd}", flush=True)
            break
        partes.append(frag)
        nombres.append(nd)
        chars_total += len(frag)
    return "\n\n".join(partes), nombres


@app.route("/siaa/chat", methods=["POST"])
def chat():
    datos = request.get_json(silent=True) or {}

    if "messages" in datos:
        msgs      = datos.get("messages", [])
        historial = [m for m in msgs[:-1] if m.get("role") in ("user","assistant")]
        ultima    = next((m for m in reversed(msgs) if m.get("role") == "user"), None)
        pregunta  = (ultima or {}).get("content","").strip()
    else:
        pregunta  = (datos.get("message") or "").strip()
        historial = datos.get("historial", [])

    if not pregunta:
        return jsonify({"error": "No se encontro pregunta"}), 400

    if not disponible():
        verificar()
    t0 = time.time()

    # ── Conversacional ───────────────────────────────────────────
    # ── Respuestas hardcodeadas para preguntas sobre el sistema ──
    pregunta_lower = pregunta.lower().strip()
    _IDENTIDAD = {
        "que es siaa", "qué es siaa", "que eres", "quién eres", "quien eres",
        "para que sirves", "para qué sirves", "que significa siaa",
        "qué significa siaa", "como te llamas", "cómo te llamas",
        "que haces", "qué haces", "presentate", "preséntate",
    }
    if any(p in pregunta_lower for p in _IDENTIDAD):
        respuesta_id = (
            "Soy **SIAA** — Sistema Inteligente de Apoyo Administrativo "
            "de la Seccional Bucaramanga, Rama Judicial de Colombia.\n\n"
            "Estoy aquí para ayudar a los 26 despachos judiciales con:\n"
            "- **SIERJU**: diligenciamiento, plazos, roles y normativa\n"
            "- **Recursos Humanos**: nómina, vacaciones, licencias, prestaciones\n"
            "- **Normativa**: Acuerdos PSAA y PCSJA vigentes\n\n"
            "Funciono completamente en modo local — sin internet, "
            "con documentos institucionales de la Seccional."
        )
        def _gen_id():
            yield f"data: {json.dumps({'no_docs': True})}\n\n"
            chunk = 50
            for i in range(0, len(respuesta_id), chunk):
                yield _sse_token(respuesta_id[i:i+chunk])
            yield _sse_done()
            _registrar(pregunta, respuesta_id, [], time.time()-t0, "IDENTIDAD")
        return Response(stream_with_context(_gen_id()),
                        content_type="text/event-stream")

    if es_conversacion(pregunta):
        mensajes = historial + [{"role": "user", "content": pregunta}]
        def _gen_conv():
            # Señal al frontend para limpiar panel de documentos
            yield f"data: {json.dumps({'no_docs': True})}\n\n"
            buf = []
            for tok in chat_stream(SYSTEM_CONVERSACIONAL, mensajes):
                buf.append(tok)
                yield _sse_token(tok)
            yield _sse_done()
            _registrar(pregunta, "".join(buf), [], time.time()-t0, "CONVERSACIONAL")
        return Response(stream_with_context(_gen_conv()),
                        content_type="text/event-stream")

    # ── Documental ───────────────────────────────────────────────
    docs_enc            = detectar(pregunta)
    contexto, nombres   = _construir_contexto(docs_enc, pregunta)
    contexto_chars      = len(contexto)

    print(f"  [Routes] Contexto: {contexto_chars} chars, docs: {nombres}", flush=True)

    prompt_u = f"{contexto}\n\nPregunta: {pregunta}" if contexto else pregunta
    mensajes = historial[-4:] + [{"role": "user", "content": prompt_u}]
    es_lista = any(p in pregunta.lower() for p in PATRONES_LISTADO)

    def _gen_doc():
        buf = []
        for tok in chat_stream(SYSTEM_DOCUMENTAL, mensajes,
                               contexto_chars=contexto_chars,
                               es_listado=es_lista):
            buf.append(tok)
            yield _sse_token(tok)
        if nombres:
            yield _sse_sources(nombres)
        yield _sse_done()
        _registrar(pregunta, "".join(buf), nombres, time.time()-t0)

    return Response(stream_with_context(_gen_doc()),
                    content_type="text/event-stream")


@app.route("/siaa/estado", methods=["GET"])
def estado():
    from collections import Counter
    docs = store.get_todos_docs()
    cols = dict(Counter(v.get("coleccion","general") for v in docs.values()))
    return jsonify({
        "version": VERSION, "ollama_disponible": disponible(),
        "documentos_cargados": len(docs), "colecciones": cols,
    })


@app.route("/siaa/status", methods=["GET"])
def status():
    from collections import Counter
    docs = store.get_todos_docs()
    cols = dict(Counter(v.get("coleccion","general") for v in docs.values()))
    return jsonify({
        "status":"ok","estado":"ok","version":VERSION,
        "ollama_disponible": disponible(),
        "warmup_completado": disponible(),
        "documentos_cargados": len(docs),
        "colecciones": cols,
        "usuarios_activos": 0,
        "cache": {"hit_rate":0,"entradas":0},
    })


@app.route("/siaa/enrutar", methods=["GET"])
def diag_enrutar():
    q = request.args.get("q","").strip()
    if not q:
        return jsonify({"error": "Parametro q requerido"}), 400
    docs = detectar(q)
    return jsonify({
        "pregunta": q, "docs": docs,
        "docs_encontrados": [{"doc":d,"coleccion":_coleccion_doc(d)} for d in docs],
    })


@app.route("/siaa/fragmento", methods=["GET"])
def ver_fragmento():
    doc = request.args.get("doc","")
    q   = request.args.get("q","")
    if not doc or not q:
        return jsonify({"error":"Parametros doc y q requeridos"}), 400
    frag = extraer(doc, q)
    return jsonify({"doc":doc,"pregunta":q,"fragmento":frag,"chars":len(frag)})


@app.route("/siaa/recargar", methods=["GET"])
def recargar():
    store.cargar()
    return jsonify({"recargado":True,"documentos":store.total_docs()})


# ── Endpoint ver documento (compatibilidad con frontend) ─────────
@app.route("/siaa/ver/<path:nombre_doc>", methods=["GET"])
def ver_documento(nombre_doc):
    """Sirve el documento .md como HTML para el panel lateral."""
    import html as _html
    nombre_doc = nombre_doc.lower().strip()
    doc = store.get_doc(nombre_doc)
    if not doc:
        # Buscar por coincidencia parcial
        todos = store.get_todos_docs()
        for k, v in todos.items():
            if nombre_doc.replace(" ","_") in k or nombre_doc in k:
                doc = v
                nombre_doc = k
                break
    if not doc:
        return f"<h2 style='font-family:sans-serif;color:#b91c1c;padding:2rem'>Documento '{nombre_doc}' no encontrado.</h2>", 404

    contenido   = doc.get("contenido", "")
    nombre_disp = os.path.splitext(doc.get("nombre_original", nombre_doc))[0].upper()
    coleccion   = doc.get("coleccion", "").upper()

    # Convertir markdown básico a HTML
    lineas_html = []
    for linea in contenido.splitlines():
        esc = _html.escape(linea)
        if esc.startswith("# "):
            esc = f"<h1>{esc[2:]}</h1>"
        elif esc.startswith("## "):
            esc = f"<h2>{esc[3:]}</h2>"
        elif esc.startswith("### "):
            esc = f"<h3>{esc[4:]}</h3>"
        elif esc.startswith("**") and esc.endswith("**"):
            esc = f"<strong>{esc[2:-2]}</strong>"
        else:
            import re as _re
            esc = _re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', esc)
        lineas_html.append(f"<p>{esc}</p>" if esc and not esc.startswith("<h") else esc)

    cuerpo = "\n".join(lineas_html)

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>{nombre_disp}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; padding: 1.5rem 2rem;
         color: #1e293b; line-height: 1.7; max-width: 860px; margin: 0 auto; }}
  h1 {{ font-size: 1.2rem; color: #1a3a8f; border-bottom: 2px solid #1a3a8f;
       padding-bottom: .5rem; margin-top: 1.5rem; }}
  h2 {{ font-size: 1rem; color: #1a3a8f; margin-top: 1.2rem; }}
  h3 {{ font-size: .9rem; color: #475569; }}
  strong {{ color: #1a3a8f; }}
  p {{ margin: .4rem 0; font-size: .88rem; }}
  .header {{ background: #1a3a8f; color: white; padding: .6rem 1rem;
             margin: -1.5rem -2rem 1.5rem; font-size: .8rem; }}
  .header span {{ opacity: .7; margin-left: .5rem; }}
</style>
</head>
<body>
<div class="header">{nombre_disp}<span>{coleccion}</span></div>
{cuerpo}
</body>
</html>""", 200, {"Content-Type": "text/html; charset=utf-8"}


@app.route("/siaa/rating", methods=["POST"])
def rating():
    """
    Recibe la calificación del usuario sobre una respuesta.
    Body: {"pregunta": str, "respuesta": str, "rating": 1|-1, "comentario": str}
    Guarda en calidad.jsonl con tipo=RATING para análisis posterior.
    """
    datos = request.get_json(silent=True) or {}
    entrada = {
        "ts":         time.strftime("%Y-%m-%dT%H:%M:%S"),
        "tipo":       "RATING",
        "pregunta":   datos.get("pregunta","")[:200],
        "respuesta":  datos.get("respuesta","")[:300],
        "rating":     datos.get("rating", 0),
        "comentario": datos.get("comentario","")[:200],
        "docs":       datos.get("docs", []),
        "alerta":     "NEGATIVO" if datos.get("rating",-1) < 0 else "OK",
    }
    try:
        os.makedirs(os.path.dirname(LOG_ARCHIVO), exist_ok=True)
        with open(LOG_ARCHIVO, "a", encoding="utf-8") as f:
            f.write(json.dumps(entrada, ensure_ascii=False) + "\n")
        return jsonify({"guardado": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
