"""
SIAA — Chunker con ventana deslizante y solapamiento.

Divide documentos en chunks de tamaño fijo con solapamiento
para garantizar que artículos y procedimientos nunca queden
partidos entre dos chunks consecutivos.

Por qué ventana deslizante vs split por párrafos:
  - Un artículo largo puede tener 1200 chars (supera CHUNK_SIZE)
  - Split por párrafos lo parte → la segunda mitad pierde contexto
  - Con solapamiento: el chunk siguiente empieza 300 chars antes
    del fin del anterior → la idea completa siempre está en algún chunk

Cada chunk recuerda su sección (último encabezado Markdown visto)
para que la cita de fuente sea precisa.

Funciones puras — sin estado, sin dependencias externas.
"""
import re
from config import CHUNK_SIZE, CHUNK_OVERLAP


def ultimo_encabezado(texto: str) -> str:
    """
    Encuentra el último encabezado Markdown en un bloque de texto.
    Usado para etiquetar cada chunk con su sección de origen.

    '## Artículo 5. Responsabilidad' → 'ARTÍCULO 5. RESPONSABILIDAD'
    Si no hay encabezado → 'INICIO'
    """
    encabezados = re.findall(r'^#{1,3}\s+(.+)$', texto, re.MULTILINE)
    if encabezados:
        return re.sub(r'[*_`]', '', encabezados[-1]).strip().upper()
    return "INICIO"


def chunkear(contenido: str) -> list[dict]:
    """
    Divide el contenido en chunks con solapamiento.

    Returns:
        Lista de dicts:
          {
            "texto":   str,  — contenido del chunk
            "seccion": str,  — último encabezado antes del chunk
            "indice":  int   — posición (0, 1, 2, ...)
          }

    Ejemplo con CHUNK_SIZE=800, CHUNK_OVERLAP=300:
      Chunk 0: chars   0 → 800
      Chunk 1: chars 500 → 1300  (avanza 500 = 800-300)
      Chunk 2: chars 1000 → 1800
    """
    chunks  = []
    inicio  = 0
    total   = len(contenido)
    idx     = 0

    while inicio < total:
        fin = min(inicio + CHUNK_SIZE, total)

        # Extender hasta el próximo salto de línea para no cortar palabras
        if fin < total:
            salto = contenido.find('\n', fin)
            if salto != -1 and salto - fin < 100:
                fin = salto

        texto_chunk = contenido[inicio:fin]

        # La sección activa es el último encabezado ANTES de este chunk
        contexto_previo = contenido[max(0, inicio - 500):inicio]
        seccion = ultimo_encabezado(contexto_previo + texto_chunk)

        chunks.append({
            "texto":   texto_chunk,
            "seccion": seccion,
            "indice":  idx,
        })

        idx    += 1
        inicio += CHUNK_SIZE - CHUNK_OVERLAP  # avanzar con solapamiento

    return chunks
