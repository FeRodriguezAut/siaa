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
    if es_conversacion(pregunta):
        mensajes = historial + [{"role": "user", "content": pregunta}]
        def _gen_conv():
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
