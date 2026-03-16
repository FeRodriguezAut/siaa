"""
SIAA — Tokenizador alfanumérico con normalización de tildes.

Reglas específicas para el dominio judicial colombiano:
  - Incluye tokens con letras+dígitos: psaa16, pcsja19, art5
  - Incluye números de 4+ dígitos: 10476, 2016 (códigos y años)
  - Descarta números cortos: 1, 22, 999 (sin significado semántico)
  - Descarta stopwords en español
  - Normaliza tildes: información → informacion (búsqueda insensible)

Funciones puras — sin estado, sin dependencias externas.
"""
import re
import unicodedata
from config import MIN_LEN_KEYWORD

STOPWORDS_ES = {
    "para", "como", "este", "esta", "esto", "estos", "estas", "pero",
    "más", "también", "cuando", "donde", "porque", "aunque", "sino",
    "desde", "hasta", "entre", "sobre", "bajo", "ante", "tras",
    "dentro", "fuera", "hacia", "según", "durante", "mediante",
    "cada", "todo", "toda", "todos", "todas", "otro", "otra", "otros",
    "dicho", "dicha", "mismo", "misma", "algún", "alguna", "ningún",
    "debe", "puede", "tiene", "hace", "será", "sido", "estar", "tener",
    "hacer", "poder", "deber", "haber", "caso", "forma", "parte",
    "tipo", "modo", "manera", "través", "respecto", "relación",
}


def normalizar(texto: str) -> str:
    """
    Elimina tildes y diacríticos para búsqueda insensible.

    "información" → "informacion"
    "quinto día hábil" → "quinto dia habil"
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', texto)
        if unicodedata.category(c) != 'Mn'
    )


def tokenizar(texto: str) -> list[str]:
    """
    Convierte texto en lista de tokens relevantes para búsqueda.

    Incluye:
      - Tokens alfanuméricos con letras (psaa16, art5, pcsja19)
      - Números puros de 4+ dígitos (10476, 2016)
    Excluye:
      - Stopwords en español
      - Números cortos (< 4 dígitos)
      - Tokens de menos de MIN_LEN_KEYWORD caracteres
    """
    tokens_raw = re.findall(r'\b[a-záéíóúüñ0-9]{3,}\b', texto.lower())
    resultado = []
    for token in tokens_raw:
        if token in STOPWORDS_ES:
            continue
        if token.isdigit():
            if len(token) >= 4:
                resultado.append(token)
        else:
            resultado.append(token)
    return resultado
