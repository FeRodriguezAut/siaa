"""
SIAA — Caché LRU de respuestas documentales.

Propósito: evitar que el LLM reprocese preguntas idénticas.
Los 26 despachos hacen preguntas muy similares → hit rate 30-40%.

Impacto medido:
  Sin caché : ~15-20s por consulta RAG nueva
  Con caché : ~5ms (3000x más rápido en hit)

No tiene dependencias externas — solo stdlib Python.
Thread-safe: todos los accesos protegidos con Lock.
"""
import hashlib
import threading
import time
import re
import unicodedata
from collections import OrderedDict
from config import CACHE_MAX_ENTRADAS, CACHE_TTL_SEGUNDOS, CACHE_SOLO_DOC


# ── Estado interno ────────────────────────────────────────────────
# OrderedDict mantiene el orden de inserción — base del algoritmo LRU.
# move_to_end() mueve una entrada al frente (más reciente).
# popitem(last=False) elimina del fondo (menos reciente).
_cache: dict = OrderedDict()
_lock         = threading.Lock()
_hits         = 0
_misses       = 0


def _clave(texto: str) -> str:
    """
    Genera clave de caché normalizada.
    Insensible a tildes, puntuación y mayúsculas.

    "¿Cuándo debo reportar?" == "cuando debo reportar"
    """
    t = texto.lower()
    t = re.sub(r'[^\w\s]', '', t)
    t = ''.join(
        c for c in unicodedata.normalize('NFD', t)
        if unicodedata.category(c) != 'Mn'
    )
    t = re.sub(r'\s+', ' ', t).strip()
    return hashlib.sha256(t.encode()).hexdigest()[:16]


def get(pregunta: str) -> dict | None:
    """
    Busca la pregunta en el caché.

    Returns:
        dict con {respuesta, cita} si hay hit válido.
        None si miss o entrada expirada (TTL).
    """
    global _hits, _misses
    clave = _clave(pregunta)

    with _lock:
        if clave not in _cache:
            _misses += 1
            return None

        entrada = _cache[clave]

        # TTL: descartar si lleva más de 1 hora sin actualizarse
        if time.time() - entrada["ts"] > CACHE_TTL_SEGUNDOS:
            del _cache[clave]
            _misses += 1
            return None

        # HIT — mover al frente (más reciente = no será desalojado pronto)
        _cache.move_to_end(clave)
        entrada["hits"] += 1
        _hits += 1
        return {"respuesta": entrada["respuesta"], "cita": entrada["cita"]}


def set(pregunta: str, respuesta: str, cita: str) -> None:
    """
    Guarda una respuesta en el caché.
    Si el caché está lleno, desaloja la entrada más fría (fondo del OrderedDict).
    No guarda respuestas vacías ni respuestas negativas ("No encontré...").
    """
    if not respuesta.strip():
        return
    if "no encontré esa información" in respuesta.lower():
        return

    clave = _clave(pregunta)

    with _lock:
        if clave in _cache:
            _cache.move_to_end(clave)
            _cache[clave].update({"respuesta": respuesta, "cita": cita, "ts": time.time()})
            return

        # Desalojar entradas frías si está lleno
        while len(_cache) >= CACHE_MAX_ENTRADAS:
            _cache.popitem(last=False)

        _cache[clave] = {
            "respuesta": respuesta,
            "cita":      cita,
            "ts":        time.time(),
            "hits":      0,
        }


def stats() -> dict:
    """Estadísticas para el endpoint /siaa/status."""
    with _lock:
        total    = _hits + _misses
        hit_rate = round(_hits / total * 100, 1) if total > 0 else 0
        return {
            "entradas":  len(_cache),
            "max":       CACHE_MAX_ENTRADAS,
            "hits":      _hits,
            "misses":    _misses,
            "hit_rate":  f"{hit_rate}%",
            "ttl_seg":   CACHE_TTL_SEGUNDOS,
        }


def clear() -> None:
    """Vaciar el caché completamente (útil tras recargar documentos)."""
    global _hits, _misses
    with _lock:
        _cache.clear()
