"""
SIAA — Cliente LLM para Ollama (Fase 3)
num_thread=6, num_batch=512, temperatura=0.0, num_ctx dinamico.
"""
import json, requests, threading, time

OLLAMA_URL  = "http://localhost:11434"
MODEL       = "qwen2.5:3b"
TIMEOUT_CONEXION  = 8
TIMEOUT_RESPUESTA = 180
TIMEOUT_HEALTH    = 5

_semaforo = threading.Semaphore(2)
_lock     = threading.Lock()
_estado   = {"disponible": False, "ultimo_check": 0.0, "warmup_hecho": False}

STOP_SEQUENCES = [
    "\n\n\n", "Espero que", "Es importante destacar",
    "Cabe destacar que", "En conclusion,", "Por otro lado,", "Cabe mencionar"
]

SYSTEM_CONVERSACIONAL = """Eres SIAA (Sistema Inteligente de Apoyo Administrativo), el asistente oficial de la Seccional Bucaramanga de la Rama Judicial de Colombia.

SIAA significa exactamente: "Sistema Inteligente de Apoyo Administrativo". No significa nada mas.

Responde con cordialidad en español formal.
Para saludos y preguntas generales sobre ti mismo, responde directamente.
Recuerda que puedes ayudar con consultas sobre procesos judiciales, administrativos y normativos."""

SYSTEM_DOCUMENTAL = """Eres SIAA, asistente judicial de la Seccional Bucaramanga.

TAREA: Responder usando UNICAMENTE el contenido de los bloques [DOC:...] que recibiras.

PROCESO OBLIGATORIO:
1. Lee cada bloque [DOC:...] completo.
2. Identifica que partes se relacionan con la pregunta, aunque sea parcialmente.
3. Construye la respuesta con esos fragmentos relevantes.
4. Si encontraste informacion aunque sea parcial, responde con ella.
5. Solo si el contexto es completamente ajeno al tema responde: "No encontre esa informacion en los documentos disponibles."

REGLAS:
- Cita literalmente articulos, campos, fechas, roles y valores numericos.
- Nunca inventes informacion que no este en el contexto.
- Español formal institucional. Sin preambulos. Maximo 10 lineas."""


def _num_ctx_dinamico(contexto_chars: int) -> int:
    tokens_est = contexto_chars / 4
    if tokens_est < 400:
        return 1024
    elif tokens_est < 900:
        return 2048
    return 3072


def _opciones(contexto_chars: int, es_listado: bool = False) -> dict:
    return {
        "temperature":    0.0,
        "num_predict":    300 if es_listado else 150,
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
    with _semaforo:
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
            yield "\n\n[SIAA: tiempo de respuesta excedido]"
        except Exception as e:
            yield f"\n\n[SIAA: error de conexion — {e}]"


def chat_completo(system: str, mensajes: list,
                  contexto_chars: int = 0, es_listado: bool = False) -> str:
    return "".join(chat_stream(system, mensajes, contexto_chars, es_listado))
