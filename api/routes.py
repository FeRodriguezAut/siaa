"""
SIAA — Rutas Flask (Fase 4)
Orquestador: une rag/, llm/ y log en el pipeline completo.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, Response, request, stream_with_context, jsonify
from flask_cors import CORS

from llm.classifier import es_conversacion
from llm.client     import (chat_stream, verificar, disponible,
                             SYSTEM_CONVERSACIONAL, SYSTEM_DOCUMENTAL)
from rag.document_store import store
from rag.router         import detectar        # ← nombre real de la Fase 2
from rag.extractor      import extraer         # ← nombre real de la Fase 2

VERSION     = "3.0.0"
LOG_ARCHIVO = "/opt/siaa/logs/calidad.jsonl"

PATRONES_LISTADO = [
    "cuáles son", "cuales son", "qué secciones", "que secciones",
    "qué campos", "que campos", "qué incluye", "que incluye",
    "enumera", "enumere", "nombra", "menciona",
]

app = Flask(__name__)
CORS(app)


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


@app.route("/siaa/chat", methods=["POST"])
def chat():
    datos     = request.get_json(silent=True) or {}
    pregunta  = (datos.get("message") or datos.get("pregunta") or "").strip()
    historial = datos.get("historial", [])
    if not pregunta:
        return jsonify({"error": "Campo 'message' requerido"}), 400
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
                yield f"data: {json.dumps({'token': tok})}\n\n"
            yield "data: [DONE]\n\n"
            _registrar(pregunta, "".join(buf), [], time.time()-t0, "CONVERSACIONAL")
        return Response(stream_with_context(_gen_conv()), content_type="text/event-stream")

    # ── Documental (RAG) ─────────────────────────────────────────
    docs_enc = detectar(pregunta)          # ← antes era detectar_documentos
    partes   = []
    nombres  = []
    for nd in docs_enc:
        frag = extraer(nd, pregunta)       # ← antes era extraer_fragmento
        if frag:
            partes.append(frag)
            nombres.append(nd)

    contexto       = "\n\n".join(partes)
    contexto_chars = len(contexto)
    prompt_u = f"{contexto}\n\nPregunta: {pregunta}" if contexto else pregunta
    mensajes = historial + [{"role": "user", "content": prompt_u}]
    es_lista = any(p in pregunta.lower() for p in PATRONES_LISTADO)

    def _gen_doc():
        buf = []
        for tok in chat_stream(SYSTEM_DOCUMENTAL, mensajes,
                               contexto_chars=contexto_chars, es_listado=es_lista):
            buf.append(tok)
            yield f"data: {json.dumps({'token': tok})}\n\n"
        yield "data: [DONE]\n\n"
        _registrar(pregunta, "".join(buf), nombres, time.time()-t0)

    return Response(stream_with_context(_gen_doc()), content_type="text/event-stream")


@app.route("/siaa/estado", methods=["GET"])
def estado():
    from collections import Counter
    docs = store.get_todos_docs()
    cols = dict(Counter(v.get("coleccion","general") for v in docs.values()))
    return jsonify({
        "version": VERSION, "ollama_disponible": disponible(),
        "documentos_cargados": len(docs), "colecciones": cols,
    })


@app.route("/siaa/enrutar", methods=["GET"])
def diag_enrutar():
    q = request.args.get("q","").strip()
    if not q:
        return jsonify({"error": "Parámetro 'q' requerido"}), 400
    return jsonify({"pregunta": q, "docs": detectar(q)})


@app.route("/siaa/fragmento", methods=["GET"])
def ver_fragmento():
    doc = request.args.get("doc","")
    q   = request.args.get("q","")
    if not doc or not q:
        return jsonify({"error": "Parámetros 'doc' y 'q' requeridos"}), 400
    frag = extraer(doc, q)
    return jsonify({"doc": doc, "pregunta": q, "fragmento": frag, "chars": len(frag)})


@app.route("/siaa/recargar", methods=["GET"])
def recargar():
    store.cargar()
    return jsonify({"recargado": True, "documentos": store.total_docs()})
