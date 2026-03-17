"""
SIAA v3.0.0 — Punto de entrada arquitectura modular.
Corre en puerto 5001 (paralelo al monolito en 5000).
"""
import sys, os
sys.path.insert(0, "/opt/siaa")

from api.routes         import app
from llm.client         import verificar
from rag.document_store import store

if __name__ == "__main__":
    print("=" * 55)
    print("  SIAA v3.0.0 — Arquitectura Modular")
    print("  Puerto 5001 (monolito sigue en 5000)")
    print("=" * 55)
    store.cargar()
    verificar()
    print(f"  Documentos: {store.total_docs()}")
    print("  Servidor:   Waitress 0.0.0.0:5001")
    print("=" * 55)
    from waitress import serve
    serve(app, host="0.0.0.0", port=5001, threads=16, channel_timeout=200)
