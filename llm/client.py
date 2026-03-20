"""
SIAA — Cliente LLM para Ollama v3.0.1
Optimizado para qwen2.5:3b con contextos documentales.
"""
import json, requests, threading, time

OLLAMA_URL        = "http://localhost:11434"
MODEL             = "qwen2.5:3b"
TIMEOUT_CONEXION  = 8
TIMEOUT_RESPUESTA = 120
TIMEOUT_HEALTH    = 5

_semaforo = threading.Semaphore(2)
_lock     = threading.Lock()
_activos  = 0          # ← agregar esta línea
_estado   = {"disponible": False, "ultimo_check": 0.0, "warmup_hecho": False}

STOP_SEQUENCES = [
    "\n\n\n", "Espero que", "Es importante destacar",
    "Cabe destacar que", "En conclusion,", "Por otro lado,",
    "Cabe mencionar", "I hope", "The document", "Please note"
]

SYSTEM_CONVERSACIONAL = """Eres SIAA — Sistema Inteligente de Apoyo Administrativo de la Seccional Bucaramanga, Rama Judicial de Colombia.

SOBRE TI:
- Tu nombre completo es: Sistema Inteligente de Apoyo Administrativo
- Fuiste creado para ayudar a los 26 despachos judiciales de Bucaramanga
- Respondes preguntas sobre SIERJU, normativa judicial, recursos humanos y procesos administrativos
- Funciones completamente en modo local, sin internet, con documentos institucionales

COMPORTAMIENTO:
- Responde en español formal y cordial
- Para saludos y preguntas sobre ti mismo, responde directamente y con brevedad
- Si preguntan que significa SIAA: Sistema Inteligente de Apoyo Administrativo"""

SYSTEM_DOCUMENTAL = """Eres SIAA, asistente judicial de la Seccional Bucaramanga. Respondes SIEMPRE en español.

INSTRUCCION: Usa SOLO la informacion de los bloques [DOC:...] para responder.

REGLAS ESTRICTAS:
- SIEMPRE responde en español formal institucional
- Cita articulos, fechas, plazos y valores exactos del documento
- Si el documento no tiene la informacion exacta, di lo que SI tiene relacionado
- Maximo 8 lineas de respuesta
- NUNCA respondas en ingles
- NUNCA inventes informacion"""


def _num_ctx_dinamico(contexto_chars: int) -> int:
    tokens_est = contexto_chars / 4
    if tokens_est < 300:
        return 1024
    elif tokens_est < 700:
        return 2048
    return 3072


def _opciones(contexto_chars: int, es_listado: bool = False) -> dict:
    return {
        "temperature":    0.0,
        "num_predict":    350 if es_listado else 200,
        "num_ctx":        _num_ctx_dinamico(contexto_chars),
        "num_thread":     6,
        "num_batch":      512,
        "repeat_penalty": 1.1,
        "stop":           STOP_SEQUENCES,
    }


def disponible() -> bool:
    with _lock:
        return _estado["disponible"]


def verificar(forzar: bool = False) -> bool:
    with _lock:
        ultimo = _estado["ultimo_check"]
    if not forzar and (time.time() - ultimo) < 15:
        return disponible()
    try:
        r  = requests.get(f"{OLLAMA_URL}/api/tags", timeout=TIMEOUT_HEALTH)
        ok = (r.status_code == 200)
    except Exception:
        ok = False
    with _lock:
        _estado["disponible"]   = ok
        _estado["ultimo_check"] = time.time()
    if ok:
        with _lock:
            ya_hecho = _estado["warmup_hecho"]
        if not ya_hecho:
            _warm_up()
    return ok


def _warm_up():
    try:
        requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "ok"}],
            "stream": False,
            "options": {"num_predict": 1, "num_ctx": 64},
        }, timeout=60)
        with _lock:
            _estado["warmup_hecho"] = True
        print("[LLM] Warm-up completado — modelo en RAM", flush=True)
    except Exception as e:
        print(f"[LLM] Warm-up fallo: {e}", flush=True)


def chat_stream(system: str, mensajes: list, contexto_chars: int = 0,
                es_listado: bool = False):
    payload = {
        "model":    MODEL,
        "system":   system,
        "messages": mensajes,
        "stream":   True,
        "options":  _opciones(contexto_chars, es_listado),
    }
    global _activos
    with _semaforo:
        with _lock:
            _activos += 1
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload, stream=True,
                timeout=(TIMEOUT_CONEXION, TIMEOUT_RESPUESTA),
            )
            for linea in resp.iter_lines():
                if not linea:
                    continue
                try:
                    datos = json.loads(linea)
                    token = datos.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if datos.get("done"):
                        break
                except Exception:
                    continue
        except requests.exceptions.Timeout:
            yield "\n\n[SIAA: tiempo de respuesta excedido — intente de nuevo]"
        except Exception as e:
            yield f"\n\n[SIAA: error de conexion — {e}]"
        finally:
            with _lock:
                _activos -= 1


def chat_completo(system: str, mensajes: list,
                  contexto_chars: int = 0, es_listado: bool = False) -> str:
    return "".join(chat_stream(system, mensajes, contexto_chars, es_listado))

def get_activos() -> int:
    """Devuelve el número de slots LLM actualmente ocupados."""
    with _lock:
        return _activos
