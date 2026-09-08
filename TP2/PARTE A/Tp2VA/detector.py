"""
Detector y clasificador de figuras geométricas
Universidad Austral - Visión Artificial - Proyecto 1

Estructura esperada:
  refs/circulo.png     <- imagen de referencia (fondo blanco, figura negra)
  refs/triangulo.png
  refs/cuadrado.png
  test.png             <- imagen a analizar (si no existe, usa webcam)

Controles (trackbars):
  Umbral      -> threshold blanco/negro
  Morfologia  -> tamaño del kernel morfológico para eliminar ruido
  Dist x100   -> distancia máxima de matchShapes para reconocer una figura
  Area minima -> descarta contornos más pequeños que este valor (px²)

Teclas: q / ESC para salir
"""

import cv2
import numpy as np
import os
import sys

REFS_DIR = "refs"
TEST_IMG = "test.png"

COLORES = {
    0: (50,  200,  50),
    1: (255, 150,  30),
    2: (30,  130, 255),
    3: (200,  30, 200),
}
COLOR_DESCONOCIDO = (0, 0, 220)


# ---------------------------------------------------------------------------
# Carga de referencias
# ---------------------------------------------------------------------------

def extraer_contorno_ref(path: str):
    """Carga una imagen de referencia y devuelve su contorno principal."""
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold fijo: figura negra (0) sobre fondo blanco (255)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnts:
        # Intentar al revés: figura blanca sobre fondo negro
        _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def cargar_referencias(refs_dir: str) -> dict:
    refs = {}
    if not os.path.isdir(refs_dir):
        return refs
    for fname in sorted(os.listdir(refs_dir)):
        if os.path.splitext(fname)[1].lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        nombre   = os.path.splitext(fname)[0]
        contorno = extraer_contorno_ref(os.path.join(refs_dir, fname))
        if contorno is not None:
            refs[nombre] = contorno
            print(f"  [OK] '{nombre}'  ({len(contorno)} puntos)")
        else:
            print(f"  [WARN] sin contorno en {fname}")
    return refs


# ---------------------------------------------------------------------------
# Referencias sintéticas de respaldo
# ---------------------------------------------------------------------------

def _triangulo_eq(cx, cy, r):
    pts = []
    for i in range(3):
        a = np.radians(90 + 120 * i)
        pts.append([int(cx + r * np.cos(a)), int(cy - r * np.sin(a))])
    return np.array(pts, np.int32)


def generar_referencias_sinteticas() -> dict:
    tam  = 300
    refs = {}

    img = np.zeros((tam, tam), np.uint8)
    cv2.circle(img, (tam // 2, tam // 2), tam // 2 - 20, 255, -1)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    refs["circulo"] = max(cnts, key=cv2.contourArea)

    img = np.zeros((tam, tam), np.uint8)
    cv2.fillPoly(img, [_triangulo_eq(tam // 2, tam // 2, tam // 2 - 20)], 255)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    refs["triangulo"] = max(cnts, key=cv2.contourArea)

    img = np.zeros((tam, tam), np.uint8)
    cv2.rectangle(img, (20, 20), (tam - 20, tam - 20), 255, -1)
    cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    refs["cuadrado"] = max(cnts, key=cv2.contourArea)

    return refs


# ---------------------------------------------------------------------------
# Procesamiento de un frame
# ---------------------------------------------------------------------------

def procesar_frame(frame, referencias: dict,
                   umbral: int, tam_morfo: int,
                   dist_max: float, area_min: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Threshold
    _, binaria = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY_INV)

    # 2. Morfología para eliminar ruido
    if tam_morfo > 0:
        k      = 2 * tam_morfo + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
        binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN,  kernel)

    # 3. Contornos
    cnts, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    salida  = frame.copy()
    nombres = list(referencias.keys())

    for cnt in cnts:
        # 4. Filtrar por área mínima
        if cv2.contourArea(cnt) < area_min:
            continue

        # 5. matchShapes contra cada referencia
        mejor_nombre    = None
        mejor_distancia = float("inf")
        for nombre, ref_cnt in referencias.items():
            dist = cv2.matchShapes(cnt, ref_cnt, cv2.CONTOURS_MATCH_I2, 0)
            if dist < mejor_distancia:
                mejor_distancia = dist
                mejor_nombre    = nombre

        # 6. Clasificar
        x, y, w, h = cv2.boundingRect(cnt)
        if mejor_distancia <= dist_max:
            idx      = nombres.index(mejor_nombre) % len(COLORES)
            color    = COLORES[idx]
            etiqueta = f"{mejor_nombre}  d={mejor_distancia:.3f}"
        else:
            color    = COLOR_DESCONOCIDO
            etiqueta = f"desconocido  d={mejor_distancia:.3f}"

        # 7. Anotar
        cv2.rectangle(salida, (x, y), (x + w, y + h), color, 2)
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(salida, (x, y - th - 8), (x + tw + 4, y), color, -1)
        cv2.putText(salida, etiqueta, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    return salida, binaria


# ---------------------------------------------------------------------------
# Panel lateral de referencias
# ---------------------------------------------------------------------------

def construir_panel_refs(referencias: dict, alto: int) -> np.ndarray:
    pw    = 170
    panel = np.zeros((alto, pw, 3), np.uint8)
    if not referencias:
        return panel

    nombres   = list(referencias.keys())
    h_item    = max(10, alto // len(referencias))

    cv2.putText(panel, "REFERENCIAS", (8, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    for i, (nombre, cnt) in enumerate(referencias.items()):
        y0     = i * h_item + 20
        color  = COLORES[i % len(COLORES)]
        canvas = np.zeros((h_item - 30, pw - 20, 3), np.uint8)

        xc, yc, wc, hc = cv2.boundingRect(cnt)
        sc = min((canvas.shape[1] - 10) / max(wc, 1),
                 (canvas.shape[0] - 10) / max(hc, 1))
        nc = cnt.astype(np.float32).copy()
        nc[:, :, 0] = (nc[:, :, 0] - xc) * sc + 5
        nc[:, :, 1] = (nc[:, :, 1] - yc) * sc + 5
        cv2.drawContours(canvas, [nc.astype(np.int32)], -1, color, 2)
        cv2.putText(canvas, nombre, (3, canvas.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        h_c = canvas.shape[0]
        if y0 + h_c <= alto:
            panel[y0:y0 + h_c, 10:10 + canvas.shape[1]] = canvas

    return panel


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 50)
    print(" Detector de Figuras Geometricas")
    print("=" * 50)

    print(f"\nCargando referencias desde '{REFS_DIR}/'...")
    referencias = cargar_referencias(REFS_DIR)

    if not referencias:
        print("No se encontraron referencias. Usando sintéticas.")
        referencias = generar_referencias_sinteticas()
        print(f"Figuras disponibles: {list(referencias.keys())}")
    else:
        print(f"Figuras cargadas: {list(referencias.keys())}")

    usar_webcam = not os.path.isfile(TEST_IMG)
    if usar_webcam:
        print("\nNo se encontró 'test.png' → usando webcam.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: no se pudo abrir la webcam.")
            sys.exit(1)
    else:
        print(f"\nImagen de prueba: {TEST_IMG}")

    WIN_CFG = "Configuracion"
    WIN_OUT = "Deteccion de Figuras Geometricas"
    WIN_BIN = "Imagen Binaria"

    cv2.namedWindow(WIN_CFG, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_CFG, 420, 140)
    cv2.createTrackbar("Umbral",      WIN_CFG, 127,  255,   lambda v: None)
    cv2.createTrackbar("Morfologia",  WIN_CFG,   2,   20,   lambda v: None)
    cv2.createTrackbar("Dist x100",   WIN_CFG,  15,  200,   lambda v: None)
    cv2.createTrackbar("Area minima", WIN_CFG, 500, 10000,  lambda v: None)

    cv2.namedWindow(WIN_OUT, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WIN_BIN, cv2.WINDOW_NORMAL)

    print("\nPresiona 'q' o ESC para salir.\n")

    while True:
        umbral   = cv2.getTrackbarPos("Umbral",      WIN_CFG)
        morfo    = cv2.getTrackbarPos("Morfologia",  WIN_CFG)
        dist_max = cv2.getTrackbarPos("Dist x100",   WIN_CFG) / 100.0
        area_min = max(1, cv2.getTrackbarPos("Area minima", WIN_CFG))

        if usar_webcam:
            ret, frame = cap.read()
            if not ret:
                print("ERROR leyendo webcam.")
                break
        else:
            frame = cv2.imread(TEST_IMG)
            if frame is None:
                print(f"ERROR: no se pudo leer {TEST_IMG}")
                break

        salida, binaria = procesar_frame(
            frame, referencias, umbral, morfo, dist_max, area_min
        )

        panel          = construir_panel_refs(referencias, salida.shape[0])
        salida_c_panel = np.hstack([salida, panel])

        cv2.imshow(WIN_OUT, salida_c_panel)
        cv2.imshow(WIN_BIN, binaria)

        espera = 1 if usar_webcam else 30
        if cv2.waitKey(espera) & 0xFF in (27, ord("q")):
            break

    if usar_webcam:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()