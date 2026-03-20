"""
SIAA — Clasificador de intención (conversacional vs documental).
"""

PATRONES_CONVERSACION = [
    "hola", "buenos días", "buenas tardes", "buenas noches", "buen día",
    "buenas", "hey", "saludos", "qué tal", "como estas", "cómo estás",
    "adiós", "adios", "hasta luego", "chao", "nos vemos",
    "gracias", "muchas gracias", "mil gracias", "muy amable",
    "de nada", "con gusto", "a la orden",
    "quién eres", "quien eres", "qué es siaa", "que es siaa",
    "qué significa siaa", "que significa siaa",
    "para qué sirves", "para que sirves",
    "qué es siaa", "que es siaa", "qué significa siaa", "que significa siaa",
    "para qué sirve siaa", "para que sirve siaa", "quién eres siaa",
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
    "clase de proceso", "apartado", "módulo", "modulo",
    "seccion", "sección",
    "nomina", "nómina", "salario", "prima", "cesantias", "cesantías",
    "vacaciones", "licencia", "incapacidad", "comision", "comisión",
    "prestaciones", "seguridad social", "pension", "pensión",
    "remuneracion", "remuneración", "servidor judicial", "carrera judicial",
    "permiso remunerado", "retiro servicio", "situacion administrativa",
    "cartilla laboral", "calendario nomina", "fecha pago", "novedades",
    "prenomina", "prenómina", "deajc26", "recursos humanos", "talento humano",
    "eps", "contrato", "vinculacion", "vinculación", "calificación",
    "evaluación", "evaluacion", "capacitación", "capacitacion",
    "traslado", "prestación", "prestacion",
}


def es_conversacion(texto: str) -> bool:
    t = texto.lower().strip()
    if any(term in t for term in TERMINOS_SIEMPRE_DOCUMENTAL):
        return False
    if len(t) < 8:
        return True
    return any(p in t for p in PATRONES_CONVERSACION)
