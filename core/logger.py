"""
SIAA — Log de calidad JSONL.

Registra cada consulta con detección automática de problemas:
  POSIBLE_ALUCINACION : respondió "No encontré" pero había documentos
  SIN_CONTEXTO        : respondió "No encontré" y era correcto
  OK                  : respuesta normal
  ERROR               : excepción en el sistema

Una línea JSON por consulta → analizable con grep, jq, Python, Excel.
Thread-safe. Rota automáticamente al llegar a LOG_MAX_LINEAS.
"""
import json
import os
import threading
import time
from config import LOG_ARCHIVO, LOG_MAX_LINEAS

_lock = threading.Lock()


def _asegurar_carpeta() -> None:
    os.makedirs(os.path.dirname(LOG_ARCHIVO), exist_ok=True)


def registrar(
    tipo: str,
    pregunta: str,
    respuesta: str,
    docs: list,
    ctx_chars: int,
    tiempo_seg: float,
    cache_hit: bool = False,
) -> None:
    """
    Escribe una línea JSONL en el log de calidad.

    Args:
        tipo       : "CONV", "DOC" o "ERROR"
        pregunta   : texto de la pregunta del usuario
        respuesta  : respuesta generada (primeros 300 chars)
        docs       : lista de nombres de documentos usados
        ctx_chars  : chars de contexto enviados al LLM
        tiempo_seg : segundos desde que llegó la pregunta
        cache_hit  : True si la respuesta vino del caché
    """
    try:
        _asegurar_carpeta()

        no_encontro = "no encontré esa información" in respuesta.lower()
        habia_docs  = len(docs) > 0 and ctx_chars > 100

        if no_encontro and habia_docs:
            alerta = "POSIBLE_ALUCINACION"
        elif no_encontro and not habia_docs:
            alerta = "SIN_CONTEXTO"
        elif tipo == "ERROR":
            alerta = "ERROR"
        else:
            alerta = "OK"

        entrada = {
            "ts":        time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tipo":      "CACHE_HIT" if cache_hit else tipo,
            "alerta":    alerta,
            "pregunta":  pregunta[:200],
            "respuesta": respuesta[:300],
            "docs":      docs,
            "ctx_chars": ctx_chars,
            "tiempo_s":  round(tiempo_seg, 2),
        }

        with _lock:
            # Rotar si supera el máximo
            try:
                with open(LOG_ARCHIVO, "r", encoding="utf-8") as f:
                    lineas = f.readlines()
                if len(lineas) >= LOG_MAX_LINEAS:
                    with open(LOG_ARCHIVO, "w", encoding="utf-8") as f:
                        f.writelines(lineas[-4000:])
            except FileNotFoundError:
                pass

            with open(LOG_ARCHIVO, "a", encoding="utf-8") as f:
                f.write(json.dumps(entrada, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"[LOG] Error escribiendo log: {e}", flush=True)


def leer(n: int = 50, filtro_tipo: str = "", filtro_alerta: str = "") -> dict:
    """
    Lee las últimas N entradas del log con filtros opcionales.
    Usado por el endpoint /siaa/log.
    """
    try:
        _asegurar_carpeta()
        with open(LOG_ARCHIVO, "r", encoding="utf-8") as f:
            lineas = f.readlines()
    except FileNotFoundError:
        return {"entradas": [], "resumen": {}}

    entradas = []
    for linea in reversed(lineas):
        linea = linea.strip()
        if not linea:
            continue
        try:
            e = json.loads(linea)
            if filtro_tipo   and e.get("tipo", "")   != filtro_tipo:
                continue
            if filtro_alerta and e.get("alerta", "") != filtro_alerta:
                continue
            entradas.append(e)
            if len(entradas) >= n:
                break
        except Exception:
            continue

    todas = [json.loads(l) for l in lineas if l.strip()]
    total  = len(todas)
    return {
        "resumen": {
            "total_consultas":        total,
            "errores":                sum(1 for e in todas if e.get("alerta") == "ERROR"),
            "posibles_alucinaciones": sum(1 for e in todas if e.get("alerta") == "POSIBLE_ALUCINACION"),
            "cache_hits":             sum(1 for e in todas if e.get("tipo") == "CACHE_HIT"),
            "tiempo_promedio_s":      round(
                sum(e.get("tiempo_s", 0) for e in todas if e.get("tiempo_s", 0) > 0) /
                max(sum(1 for e in todas if e.get("tiempo_s", 0) > 0), 1), 1
            ),
        },
        "entradas": entradas,
    }
