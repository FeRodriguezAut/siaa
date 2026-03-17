# SIAA — Sistema Inteligente de Apoyo Administrativo

> Proyecto Piloto Asistente institucional de IA para la 

[![Estado][piloto - beta testing](https://img.shields.io/badge/piloto-beta_testing-green)](https://github.com/FeRodriguezAut/siaa)
[![Versión](https://img.shields.io/badge/versión-2.1.27-blue)](https://github.com/FeRodriguezAut/siaa)
[![Modelo](https://img.shields.io/badge/modelo-qwen2.5%3A3b-orange)](https://ollama.com/library/qwen2.5)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/licencia-uso%20institucional-lightgrey)](LICENSE)

---

## ¿Qué es SIAA?

SIAA es un chatbot RAG (Retrieval-Augmented Generation) completamente local que responde preguntas sobre documentos institucionales . Funciona sin internet, sin GPU y está optimizado para hardware de oficina estándar.

- **100% local:** Sin APIs externas. Todos los datos permanecen en el servidor.
- **RAG inteligente:** Recupera fragmentos precisos de 59 documentos `.md` usando TF-IDF multinivel.
- **Caché LRU:** Respuestas instantáneas (~5ms) para preguntas frecuentes de los 26 despachos.
- **Streaming SSE:** Respuestas token a token en tiempo real.
- **Diagnóstico completo:** Endpoints para inspeccionar enrutador, fragmentos, keywords y logs.

---

## Inicio Rápido

```bash
# 1. Clonar el repositorio
git clone https://github.com/FeRodriguezAut/siaa.git /opt/siaa
cd /opt/siaa

# 2. Instalar dependencias
pip install flask flask-cors waitress requests python-docx pandas openpyxl \
            pymupdf4llm pdf2image pytesseract --break-system-packages

# 3. Instalar Ollama y el modelo
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen2.5:3b

# 4. Crear estructura de directorios
sudo mkdir -p /opt/siaa/fuentes/{normativa,sierju} /opt/siaa/logs

# 5. Agregar documentos .md a /opt/siaa/fuentes/

# 6. Iniciar el servidor
python3 siaa_proxy.py

# 7. Verificar
curl http://localhost:5000/siaa/estado
```

---

## Arquitectura

```
Usuario (navegador)
      ↓
   Nginx :80
      ↓
SIAA Proxy Flask+Waitress :5000
  ├── Enrutador TF-IDF (59 docs en RAM, 3 colecciones)
  ├── Extractor de chunks (ventana deslizante, solapamiento)
  ├── Caché LRU (200 entradas, TTL 1h, ~5ms hit)
  └── Streaming SSE → Ollama :11434
              ↓
        qwen2.5:3b (CPU, AVX2, ~1.8GB RAM)
```

---

## Hardware de Producción

| Componente | Detalle |
|-----------|---------|
| Equipo | HP EliteDesk 705 G4 |
| CPU | AMD Ryzen 5 PRO 2600 (6 núcleos, AVX2) |
| RAM | 64 GB DDR4 |
| GPU | ❌ RX 550 (sin soporte ROCm — no utilizada) |
| OS | Fedora 43 |
| Latencia promedio | ~15s/consulta (sin caché) |

---

## Estructura del Proyecto

```
/opt/siaa/
├── siaa_proxy.py        # Servidor principal (Flask + RAG) — v2.1.27
├── convertidor.py       # Word/Excel → Markdown + SQLite
├── convertidor_pdf.py   # PDF → Markdown (pymupdf4llm + OCR Tesseract)
├── index.html           # Frontend web
├── fuentes/             # Documentos .md indexados (no en repo)
│   ├── *.md             # Colección general
│   ├── normativa/       # Acuerdos PSAA, PCSJA
│   └── sierju/          # Manuales por tipo de juzgado
├── instructivos/        # Archivos Word/Excel originales (no en repo)
├── pdfs_origen/         # PDFs para convertir (no en repo)
└── logs/
    └── calidad.jsonl    # Log de calidad JSONL (no en repo)
```

---

## Documentación

La documentación completa está en la carpeta `/docs/` (formato Mintlify):

- [Introducción y arquitectura](docs/introduccion/bienvenida.mdx)
- [Guía de instalación paso a paso](docs/instalacion/paso-a-paso.mdx)
- [Recuperación inteligente de documentos](docs/caracteristicas/recuperacion-inteligente.mdx)
- [Referencia de la API](docs/referencia-api/chat.mdx)
- [Solución de problemas](docs/mantenimiento/solucionar-problemas.mdx)

Documentación online: [ferodriguezaut-siaa.mintlify.app](https://ferodriguezaut-siaa.mintlify.app)

---

## API

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/siaa/chat` | POST | Consulta al asistente (streaming SSE) |
| `/siaa/estado` | GET | Estado del sistema y métricas |
| `/siaa/recargar` | GET | Recargar índice de documentos |
| `/siaa/cache` | GET/DELETE | Estadísticas / limpiar caché |
| `/siaa/log` | GET | Log de calidad (JSONL) |
| `/siaa/enrutar?q=` | GET | Diagnóstico del enrutador |
| `/siaa/fragmento?doc=&q=` | GET | Fragmento que llega al modelo |
| `/siaa/keywords/{doc}` | GET | Keywords TF-IDF de un documento |
| `/siaa/densidad/{term}` | GET | Densidad de un término en el índice |
| `/siaa/debug_tokens?q=` | GET | Diagnóstico de tokenización |

---

## Licencia

Uso institucional Piloto interno — .
