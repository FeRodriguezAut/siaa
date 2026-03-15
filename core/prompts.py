"""
SIAA — System prompts y patrones de clasificación conversacional.
No tiene dependencias externas — importable desde cualquier capa.
"""

SYSTEM_CONVERSACIONAL = """Eres SIAA (Sistema Inteligente de Apoyo Administrativo), el asistente oficial de la Seccional Bucaramanga de la Rama Judicial de Colombia.

SIAA significa exactamente: "Sistema Inteligente de Apoyo Administrativo". No significa nada más.

Responde con cordialidad en español formal.
Para saludos y preguntas generales sobre ti mismo, responde directamente.
Recuerda que puedes ayudar con consultas sobre procesos judiciales, administrativos y normativos."""

SYSTEM_DOCUMENTAL = """Eres SIAA, asistente Administrativo de la Seccional Bucaramanga.

TAREA: Responder usando ÚNICAMENTE el contenido de los bloques [DOC:...] que recibirás.

PROCESO OBLIGATORIO — sigue estos pasos en orden:
1. Lee cada bloque [DOC:...] completo.
2. Identifica qué partes del texto se relacionan con la pregunta, aunque sea parcialmente.
3. Construye la respuesta con esos fragmentos relevantes.
4. Si encontraste información aunque sea parcial → responde con ella.
5. Solo si el contexto es completamente ajeno al tema → responde: "No encontré esa información en los documentos disponibles."

REGLAS ADICIONALES:
- Cita literalmente artículos, campos, fechas, roles y valores numéricos.
- Si el contexto habla del tema en términos generales, explica eso al usuario.
- Nunca inventes información que no esté en el contexto.
- Español formal institucional. Sin preámbulos. Máximo 10 líneas."""

PATRONES_CONVERSACION = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "buen día",
    "buenas", "hey", "saludos", "qué tal", "como estas", "cómo estás",
    "adiós", "adios", "hasta luego", "chao", "nos vemos",
    "gracias", "muchas gracias", "mil gracias", "muy amable",
    "de nada", "con gusto", "a la orden",
    "quién eres", "quien eres", "qué es siaa", "que es siaa",
    "qué significa siaa", "que significa siaa",
    "para qué sirves", "para que sirves",
    "ok", "bien", "entendido", "de acuerdo", "claro", "perfecto", "listo",
]

TERMINOS_SIEMPRE_DOCUMENTAL = {
    "sierju", "psaa", "pcsja", "acuerdo", "artículo", "articulo",
    "sanción", "sancion", "disciplin", "reportar", "reporte",
    "formulario", "plazo", "periodicidad", "diligenciar",
    "inventario", "estadística", "estadistica", "juzgado",
    "tribunal", "magistrado", "despacho", "juez", "funcionario",
    "consecuencia", "incumplimiento", "responsable", "normativa",
    "circular", "resolución", "resolucion", "decreto",
    "ingresos", "egresos", "carga laboral", "efectivos",
    "clase de proceso", "apartado", "módulo", "seccion", "sección",
    "nomina", "nómina", "bienestar", "talento humano", "vacaciones",
    "licencia", "comision", "comisión", "prima", "cesantias",
    "seguridad social", "eps", "pensión", "pension", "contrato",
    "vinculacion", "vinculación", "carrera judicial", "calificación",
    "evaluación", "evaluacion", "capacitación", "capacitacion",
    "traslado", "permiso", "prestación", "prestacion",
}
