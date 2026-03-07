"""
╔══════════════════════════════════════════════════════════════╗
║       SIAA — Convertidor PDF → Markdown  v2.0               ║
║       Seccional Bucaramanga — Rama Judicial                  ║
╠══════════════════════════════════════════════════════════════╣
║  Modo 1: pymupdf4llm  → PDFs con texto nativo (rápido)      ║
║  Modo 2: OCR Tesseract → PDFs escaneados (fallback auto)     ║
║  Umbral: si pymupdf extrae < MIN_CHARS → activa OCR          ║
╚══════════════════════════════════════════════════════════════╝

INSTALACIÓN (ya hecha en el servidor):
  pip install pymupdf4llm pdf2image pytesseract --break-system-packages
  sudo dnf install tesseract tesseract-langpack-spa poppler-utils -y

USO:
  python3 convertidor_pdf_v2.py                  <- convierte todos
  python3 convertidor_pdf_v2.py --forzar-ocr     <- OCR en todos
  python3 convertidor_pdf_v2.py --reconvertir    <- solo MDs vacíos
"""

import os, re, sys, datetime
from pathlib import Path

try:
    import pymupdf4llm
    PYMUPDF_OK = True
except ImportError:
    PYMUPDF_OK = False
    print("  ⚠ pymupdf4llm no disponible")

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_OK = True
except ImportError:
    OCR_OK = False
    print("  ⚠ pytesseract/pdf2image no disponible")

# ── Rutas ─────────────────────────────────────────────────────────────────────
if sys.platform == "win32":
    CARPETA_ENTRADA = r"C:\SIAA\pdfs_origen"
    CARPETA_SALIDA  = r"C:\SIAA\Documentos_MD"
else:
    CARPETA_ENTRADA = "/opt/siaa/pdfs_origen"
    CARPETA_SALIDA  = "/opt/siaa/fuentes/normativa"

MIN_CHARS = 200   # Menos de esto → PDF escaneado → OCR
OCR_DPI   = 300
OCR_LANG  = "spa"


def sanitizar_nombre(nombre):
    nombre = nombre.lower().replace(" ", "_")
    nombre = re.sub(r'[^\w\-.]', '_', nombre)
    return re.sub(r'_+', '_', nombre)


def limpiar_ocr(texto):
    lineas = []
    for linea in texto.split('\n'):
        linea = linea.strip()
        if len(re.findall(r'[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]', linea)) < 3:
            continue
        lineas.append(re.sub(r' {3,}', '  ', linea))
    resultado = '\n'.join(lineas)
    return re.sub(r'\n{4,}', '\n\n\n', resultado)


def convertir_con_pymupdf(ruta_pdf):
    if not PYMUPDF_OK:
        return "", "sin_pymupdf"
    try:
        texto = pymupdf4llm.to_markdown(ruta_pdf)
        texto = re.sub(r'<!--.*?-->', '', texto, flags=re.DOTALL).strip()
        return texto, "pymupdf"
    except Exception as e:
        return "", f"pymupdf_error:{e}"


def convertir_con_ocr(ruta_pdf):
    if not OCR_OK:
        return "", "sin_ocr"
    try:
        print(f"    📷 Convirtiendo a imágenes (DPI={OCR_DPI})...")
        paginas = convert_from_path(ruta_pdf, dpi=OCR_DPI)
        print(f"    📄 {len(paginas)} página(s)")
        partes = []
        for i, pagina in enumerate(paginas, 1):
            print(f"    🔍 OCR página {i}/{len(paginas)}...", end="\r")
            texto_pag = pytesseract.image_to_string(pagina, lang=OCR_LANG)
            texto_pag = limpiar_ocr(texto_pag)
            if texto_pag.strip():
                partes.append(f"\n\n<!-- Página {i} -->\n\n{texto_pag}")
        print()
        return "\n".join(partes).strip(), "ocr_tesseract"
    except Exception as e:
        return "", f"ocr_error:{e}"


def convertir_un_pdf(ruta_pdf, forzar_ocr=False):
    nombre_pdf  = Path(ruta_pdf).name
    nombre_md   = sanitizar_nombre(Path(ruta_pdf).stem) + ".md"
    ruta_salida = os.path.join(CARPETA_SALIDA, nombre_md)

    texto, metodo = "", "ninguno"

    if not forzar_ocr:
        texto, metodo = convertir_con_pymupdf(ruta_pdf)

    if len(texto) < MIN_CHARS:
        if metodo == "pymupdf":
            print(f"    ⚠ pymupdf extrajo {len(texto)} chars → OCR...")
        texto, metodo = convertir_con_ocr(ruta_pdf)

    fecha = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    metodo_str = "pymupdf4llm" if metodo == "pymupdf" else "OCR Tesseract"

    if texto.strip():
        encabezado = f"<!-- Origen: {nombre_pdf} | Método: {metodo_str} | Convertido: {fecha} -->\n\n"
        md_final = encabezado + texto
        exito, icono = True, "✅"
    else:
        md_final = (
            f"<!-- Origen: {nombre_pdf} | ERROR: Sin texto extraíble | {fecha} -->\n\n"
            f"**AVISO:** No fue posible extraer texto de este documento.\n"
        )
        exito, icono = False, "❌"

    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    with open(ruta_salida, "w", encoding="utf-8") as f:
        f.write(md_final)

    print(f"    {icono} {nombre_md} → {len(texto):,} chars [{metodo_str}]")
    return {"nombre_md": nombre_md, "metodo": metodo, "chars": len(texto), "exito": exito}


def convertir_todos(forzar_ocr=False, solo_vacios=False):
    if not os.path.exists(CARPETA_ENTRADA):
        print(f"❌ No existe: {CARPETA_ENTRADA}")
        return

    archivos = sorted(f for f in os.listdir(CARPETA_ENTRADA) if f.lower().endswith('.pdf'))
    if not archivos:
        print(f"⚠ Sin PDFs en: {CARPETA_ENTRADA}")
        return

    print(f"\n{'='*55}")
    print(f"  SIAA Convertidor PDF v2.0")
    print(f"  PDFs: {len(archivos)} | Modo: {'OCR forzado' if forzar_ocr else 'Auto'}")
    print(f"  Salida: {CARPETA_SALIDA}")
    print(f"{'='*55}\n")

    ok, ocr_count, errores = 0, 0, 0

    for archivo in archivos:
        ruta_pdf  = os.path.join(CARPETA_ENTRADA, archivo)
        nombre_md = sanitizar_nombre(Path(archivo).stem) + ".md"
        ruta_md   = os.path.join(CARPETA_SALIDA, nombre_md)

        if solo_vacios and os.path.exists(ruta_md):
            if os.path.getsize(ruta_md) > MIN_CHARS + 100:
                print(f"  ⏭ {archivo} — ya convertido")
                ok += 1
                continue

        print(f"  📂 {archivo}")
        res = convertir_un_pdf(ruta_pdf, forzar_ocr=forzar_ocr)
        if res["exito"]:
            if "ocr" in res["metodo"]: ocr_count += 1
            else: ok += 1
        else:
            errores += 1

    print(f"\n{'='*55}")
    print(f"  ✅ pymupdf: {ok}  |  🔍 OCR: {ocr_count}  |  ❌ Error: {errores}")
    print(f"  Recarga: curl http://localhost:5000/siaa/recargar")
    print(f"{'='*55}")


if __name__ == "__main__":
    forzar_ocr  = "--forzar-ocr"  in sys.argv
    solo_vacios = "--reconvertir" in sys.argv
    convertir_todos(forzar_ocr=forzar_ocr, solo_vacios=solo_vacios)
