"""
SIAA — Clarificador de preguntas ambiguas.

Detecta preguntas documentales que necesitan más contexto antes
de buscar en documentos. Responde en <1ms sin llamar al LLM.

Casos gestionados:
  "juzgado civil"    → ¿Municipal, Circuito, Ejecución...?
  "juzgado penal"    → ¿Municipal, Circuito, Especializado...?
  "laboral"          → ¿Circuito, Pequeñas Causas, Tribunal?
  "promiscuo"        → ¿Municipal o Circuito?
  "administrativo"   → ¿Juzgado o Tribunal?

No tiene dependencias externas — función pura, sin estado.
"""

# Cada entrada define:
#   condicion  : lambda que recibe el texto en minúsculas → True si es ambigua
#   opciones   : lista de tipos específicos para mostrar al usuario
#   pregunta   : texto de clarificación a mostrar
_CLARIFICACIONES = [
    {
        "condicion": lambda t: "civil" in t and not any(x in t for x in [
            "municipal", "circuito", "familia", "tierras",
            "ejecucion", "ejecución", "pequeñas", "pequenas",
        ]),
        "opciones": [
            "Juzgado Civil Municipal",
            "Juzgado Civil del Circuito",
            "Juzgado Civil del Circuito Especializado",
            "Juzgado Civil de Ejecución de Sentencias",
            "Juzgado de Familia",
        ],
        "pregunta": "¿A qué tipo de Juzgado Civil se refiere su consulta?",
    },
    {
        "condicion": lambda t: "penal" in t and not any(x in t for x in [
            "municipal", "circuito", "adolescente", "adolescentes",
            "especializado", "ejecucion", "ejecución",
            "conocimiento", "garantias", "garantías",
        ]),
        "opciones": [
            "Juzgado Penal Municipal",
            "Juzgado Penal del Circuito",
            "Juzgado Penal del Circuito Especializado",
            "Juzgado Penal de Adolescentes",
        ],
        "pregunta": "¿A qué tipo de Juzgado Penal se refiere su consulta?",
    },
    {
        "condicion": lambda t: "laboral" in t and not any(x in t for x in [
            "pequeñas", "pequenas", "causas", "sala", "tribunal",
        ]),
        "opciones": [
            "Juzgado Laboral del Circuito",
            "Juzgado de Pequeñas Causas Laborales",
            "Sala Laboral del Tribunal",
        ],
        "pregunta": "¿A qué instancia laboral se refiere su consulta?",
    },
    {
        "condicion": lambda t: "promiscuo" in t and not any(x in t for x in [
            "municipal", "circuito",
        ]),
        "opciones": [
            "Juzgado Promiscuo Municipal",
            "Juzgado Promiscuo del Circuito",
        ],
        "pregunta": "¿Se refiere al Juzgado Promiscuo Municipal o del Circuito?",
    },
    {
        "condicion": lambda t: (
            "administrativo" in t
            and not any(x in t for x in ["tribunal", "juzgado", "sala"])
            and "sierju" not in t
            and "acuerdo" not in t
        ),
        "opciones": [
            "Juzgado Administrativo",
            "Tribunal Administrativo",
        ],
        "pregunta": "¿Se refiere al Juzgado Administrativo o al Tribunal Administrativo?",
    },
]


def detectar(pregunta: str) -> dict | None:
    """
    Detecta si la pregunta es ambigua y necesita clarificación.

    Returns:
        dict con {pregunta_clarificacion, opciones} si es ambigua.
        None si la pregunta es suficientemente específica.
    """
    t = pregunta.lower()
    for cfg in _CLARIFICACIONES:
        if cfg["condicion"](t):
            return {
                "pregunta_clarificacion": cfg["pregunta"],
                "opciones":               cfg["opciones"],
            }
    return None
