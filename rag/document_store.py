"""
SIAA — Almacén de documentos en memoria (DocumentStore).

Carga los 65 documentos .md de /opt/siaa/fuentes/ en RAM al arrancar.
Sin lectura de disco en cada consulta — los HDD no son el cuello de botella.

Colecciones:
  fuentes/           → general (FAQs)
  fuentes/normativa/ → acuerdos PSAA, PCSJA
  fuentes/sierju/    → manuales por tipo de juzgado
  fuentes/rrhh/      → cartilla laboral, calendario nómina

Índices precalculados en RAM:
  _docs       → contenido completo + metadatos por documento
  _chunks     → chunks pre-divididos por documento
  _colecciones→ keywords TF-IDF por colección
  _densidad   → índice invertido término→(densidad, doc)

Singleton: usar `store` (instancia global al final del archivo).
Thread-safe: 4 locks independientes por índice.
"""
import math
import os
import re
import threading
from collections import Counter, defaultdict
from config import (
    CARPETA_FUENTES, TOP_KEYWORDS_POR_DOC,
    MIN_FREQ_KEYWORD, MIN_LEN_KEYWORD
)
from rag.tokenizer import tokenizar
from rag.chunker import chunkear


class DocumentStore:
    """
    Almacén central de documentos institucionales.

    Uso:
        from rag.document_store import store
        doc = store.get_doc("acuerdo_psaa16.md")
        chunks = store.get_chunks("acuerdo_psaa16.md")
    """

    def __init__(self):
        # Estado interno — nunca acceder directamente desde fuera
        self._docs:        dict = {}
        self._chunks:      dict = {}
        self._colecciones: dict = {}
        self._densidad:    dict = {}

        # Un lock por índice — evita bloqueos cruzados
        self._lock_docs  = threading.Lock()
        self._lock_chunks = threading.Lock()
        self._lock_cols  = threading.Lock()
        self._lock_dens  = threading.Lock()

    # ── Accesores thread-safe ────────────────────────────────────

    def get_doc(self, nombre: str) -> dict | None:
        with self._lock_docs:
            return self._docs.get(nombre)

    def get_chunks(self, nombre: str) -> list:
        with self._lock_chunks:
            return self._chunks.get(nombre, [])

    def get_colecciones(self) -> dict:
        with self._lock_cols:
            return dict(self._colecciones)

    def get_todos_docs(self) -> dict:
        with self._lock_docs:
            return dict(self._docs)

    def get_densidad(self) -> dict:
        with self._lock_dens:
            return dict(self._densidad)

    def total_docs(self) -> int:
        with self._lock_docs:
            return len(self._docs)

    def total_chunks(self) -> int:
        with self._lock_chunks:
            return sum(len(v) for v in self._chunks.values())

    def total_terminos(self) -> int:
        with self._lock_dens:
            return len(self._densidad)

    # ── Carga ────────────────────────────────────────────────────

    def cargar(self) -> None:
        """
        Carga todos los documentos .md en RAM y precalcula índices.
        Llamar al arrancar y tras /siaa/recargar.
        """
        nuevos_docs:  dict = {}
        nuevos_chunks: dict = {}
        nuevas_cols:  dict = {}

        # Descubrir subcarpetas de colecciones
        subcarpetas = [("general", CARPETA_FUENTES)]
        try:
            for entrada in os.scandir(CARPETA_FUENTES):
                if entrada.is_dir():
                    subcarpetas.append((entrada.name.lower(), entrada.path))
        except Exception as e:
            print(f"  [Store] Error escaneando fuentes: {e}")
            return

        for nombre_col, ruta_col in subcarpetas:
            try:
                archivos = [
                    os.path.join(ruta_col, f)
                    for f in os.listdir(ruta_col)
                    if os.path.isfile(os.path.join(ruta_col, f))
                    and f.lower().endswith(('.md', '.txt'))
                ]
            except Exception as e:
                print(f"  [Store] Error listando {ruta_col}: {e}")
                continue

            if not archivos:
                continue

            docs_col: dict = {}
            for ruta in archivos:
                try:
                    nombre_clave = os.path.basename(ruta).lower()
                    with open(ruta, "r", encoding="utf-8", errors="ignore") as f:
                        contenido = f.read()

                    tokens      = tokenizar(contenido)
                    token_count = Counter(tokens)
                    chunks      = chunkear(contenido)
                    nuevos_chunks[nombre_clave] = chunks

                    doc_entry = {
                        "ruta":            ruta,
                        "nombre_original": os.path.basename(ruta),
                        "contenido":       contenido,
                        "palabras":        set(tokens),
                        "tamano":          len(contenido),
                        "coleccion":       nombre_col,
                        "token_count":     token_count,
                        "total_tokens":    len(tokens),
                        "tokens_nombre":   self._tokens_nombre(nombre_clave),
                        "num_chunks":      len(chunks),
                    }
                    docs_col[nombre_clave]        = doc_entry
                    nuevos_docs[nombre_clave]     = doc_entry
                    print(f"  [Store] [{nombre_col}] {os.path.basename(ruta)} "
                          f"({len(contenido):,} chars, {len(chunks)} chunks)")

                except Exception as e:
                    print(f"  [Store] Error leyendo {ruta}: {e}")

            if not docs_col:
                continue

            keywords_col = self._calcular_tfidf(docs_col)
            nuevas_cols[nombre_col] = {
                "docs":     docs_col,
                "keywords": keywords_col,
            }

        # Índice de densidad invertido: término → [(densidad, nombre_doc)]
        nuevo_dens: dict = defaultdict(list)
        for nombre_doc, doc in nuevos_docs.items():
            total = doc["total_tokens"]
            if total == 0:
                continue
            for termino, freq in doc["token_count"].items():
                if len(termino) >= MIN_LEN_KEYWORD:
                    nuevo_dens[termino].append((freq / total, nombre_doc))
        for t in nuevo_dens:
            nuevo_dens[t].sort(reverse=True)

        # Actualizar estado con locks
        with self._lock_docs:
            self._docs = nuevos_docs
        with self._lock_chunks:
            self._chunks = nuevos_chunks
        with self._lock_cols:
            self._colecciones = nuevas_cols
        with self._lock_dens:
            self._densidad = dict(nuevo_dens)

        print(f"  [Store] Total: {len(nuevos_docs)} docs en "
              f"{len(nuevas_cols)} colecciones")
        print(f"  [Store] Chunks: {sum(len(v) for v in nuevos_chunks.values())}")
        print(f"  [Store] Términos en índice: {len(nuevo_dens):,}")

    # ── Utilidades internas ──────────────────────────────────────

    @staticmethod
    def _tokens_nombre(nombre_clave: str) -> set:
        sin_ext = os.path.splitext(nombre_clave)[0]
        partes  = re.split(r'[_\s\-\.]+', sin_ext.lower())
        return {p for p in partes if len(p) >= 3}

    @staticmethod
    def _calcular_tfidf(documentos: dict) -> dict:
        if not documentos:
            return {}
        tokens_por_doc = {n: tokenizar(d["contenido"]) for n, d in documentos.items()}
        N  = len(documentos)
        df: dict = defaultdict(int)
        for tokens in tokens_por_doc.values():
            for t in set(tokens):
                df[t] += 1
        keywords_por_doc: dict = {}
        for nombre, tokens in tokens_por_doc.items():
            if not tokens:
                keywords_por_doc[nombre] = []
                continue
            conteo       = Counter(tokens)
            total_tokens = len(tokens)
            scores: dict = {}
            for termino, freq in conteo.items():
                freq_min = MIN_FREQ_KEYWORD if len(termino) <= 4 else 1
                if freq < freq_min or len(termino) < MIN_LEN_KEYWORD:
                    continue
                tf  = freq / total_tokens
                idf = math.log((N + 1) / (df[termino] + 1)) + 1
                scores[termino] = tf * idf
            top = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
            keywords_por_doc[nombre] = top[:TOP_KEYWORDS_POR_DOC]
        return keywords_por_doc


# ── Singleton global ─────────────────────────────────────────────
# Todos los módulos importan este objeto, no la clase.
# from rag.document_store import store
store = DocumentStore()
