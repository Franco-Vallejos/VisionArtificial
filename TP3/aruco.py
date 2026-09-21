#!/usr/bin/env python3
"""
TP: Localización Homográfica — Visión Artificial
Prof. Alejandro Silvestri — UNLaM

USO:
    python aruco.py                  # solo localización
    python aruco.py kuri.WEBP        # localización + overlay AR de la imagen

Controles:
    R     → Registrar el plano métrico (hacelo con el marcador visible)
    Q/ESC → Salir
"""

import sys, os
import numpy as np
import cv2

# ──────────────────────────────────────────────────────────────────────────────
#  PARÁMETROS
# ──────────────────────────────────────────────────────────────────────────────

MARKER_SIZE_MM = 100                     # lado físico real del marcador (mm)
ARUCO_DICT     = cv2.aruco.DICT_4X4_50   # diccionario ArUco a usar
W2D_SIZE       = 650                     # tamaño de la ventana W2D (px, cuadrada)
PX_PER_MM      = 2.0                     # escala de la vista cenital

# ──────────────────────────────────────────────────────────────────────────────
#  CARGAR IMAGEN DE OVERLAY (argumento opcional)
# ──────────────────────────────────────────────────────────────────────────────

overlay_img = None

if len(sys.argv) > 1:
    nombre = sys.argv[1]
    # Buscar en la carpeta del script Y en el directorio desde donde se corre
    rutas = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), nombre),
        os.path.abspath(nombre),
    ]
    for ruta in rutas:
        if os.path.exists(ruta):
            data = np.frombuffer(open(ruta, "rb").read(), dtype=np.uint8)
            overlay_img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # soporta WEBP/PNG/JPG
            if overlay_img is not None:
                print(f"[OK] Overlay: {ruta}")
                break
    if overlay_img is None:
        print(f"[AVISO] No se pudo cargar '{nombre}'. Corriendo sin overlay.")

# ──────────────────────────────────────────────────────────────────────────────
#  DETECTOR ArUco — parámetros relajados para detectar en pantallas/tablets
# ──────────────────────────────────────────────────────────────────────────────

_dict   = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
_params = cv2.aruco.DetectorParameters()

# Ventanas de umbral más amplias → detecta marcadores en pantallas con brillo uniforme
_params.adaptiveThreshWinSizeMin      = 3
_params.adaptiveThreshWinSizeMax      = 53
_params.adaptiveThreshWinSizeStep     = 4
_params.adaptiveThreshConstant        = 7

# Permite marcadores pequeños o muy grandes en el encuadre
_params.minMarkerPerimeterRate        = 0.01
_params.maxMarkerPerimeterRate        = 4.0

# Más tolerancia a bits sucios o reflejos
_params.errorCorrectionRate           = 0.6
_params.polygonalApproxAccuracyRate   = 0.08

# Refinamiento subpíxel → esquinas más precisas para mejor homografía
_params.cornerRefinementMethod        = cv2.aruco.CORNER_REFINE_SUBPIX

_detector = cv2.aruco.ArucoDetector(_dict, _params)

# CLAHE: mejora el contraste localmente → el detector ve mejor los bordes del marcador
# aunque haya zonas muy brillantes (pantalla) o muy oscuras en la misma imagen
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def detectar(frame: np.ndarray):
    """
    Preprocesa el frame y detecta marcadores ArUco.

    Estrategia: intentar con CLAHE (mejor para pantallas);
    si no encuentra, intentar con simple escala de grises (mejor para papel impreso).

    Retorna: (corners, ids) — corners es lista de arrays (1,4,2); ids array (N,1)
    """
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # escala de grises
    gray_clahe = _clahe.apply(gray)                       # contraste adaptativo

    # Intento 1: con CLAHE
    corners, ids, _ = _detector.detectMarkers(gray_clahe)
    if ids is not None and len(ids) > 0:
        return corners, ids

    # Intento 2: sin CLAHE (marcador en papel impreso con buena iluminación)
    corners, ids, _ = _detector.detectMarkers(gray)
    return corners, ids


# ──────────────────────────────────────────────────────────────────────────────
#  ESTADO GLOBAL
# ──────────────────────────────────────────────────────────────────────────────

H_img2world = None   # homografía imagen → coordenadas en mm
H_img2viz   = None   # homografía imagen → píxeles W2D
bg_image    = None   # vista cenital capturada al registrar (no viva)
registered  = False  # si el plano fue registrado


# ──────────────────────────────────────────────────────────────────────────────
#  REGISTRO DEL PLANO MÉTRICO (tecla R)
# ──────────────────────────────────────────────────────────────────────────────

def registrar(frame: np.ndarray, corners_marcador: np.ndarray):
    """
    Calcula las dos homografías y captura la imagen de fondo cenital.
    Se ejecuta una sola vez (o cada vez que el usuario presiona R).

    corners_marcador: shape (1,4,2) — esquinas del marcador en la imagen de cámara.
    Orden ArUco: top-left, top-right, bot-right, bot-left (y↓ en imagen).
    """
    global H_img2world, H_img2viz, bg_image, registered

    half    = MARKER_SIZE_MM / 2.0         # mitad del lado del marcador en mm
    cx = cy = W2D_SIZE / 2.0              # centro de la imagen W2D en px
    half_px = half * PX_PER_MM            # mitad del lado en px de la vista W2D

    # Esquinas del marcador en la imagen de cámara
    img_pts = corners_marcador[0].astype(np.float32)   # shape (4,2)

    # ── Homografía 1: imagen → mundo (mm) ────────────────────────────────────
    # Sistema de referencia: origen en el CENTRO del marcador, x→derecha, y→ARRIBA.
    # top-left  imagen → x negativo, y positivo en mundo
    # top-right imagen → x positivo, y positivo en mundo
    # bot-right imagen → x positivo, y negativo en mundo
    # bot-left  imagen → x negativo, y negativo en mundo
    world_pts = np.float32([
        [-half,  half],   # top-left
        [ half,  half],   # top-right
        [ half, -half],   # bot-right
        [-half, -half],   # bot-left
    ])
    H_img2world, _ = cv2.findHomography(img_pts, world_pts)

    # ── Homografía 2: imagen → visualización W2D (px) ────────────────────────
    # El marcador queda centrado en la imagen W2D.
    # En la imagen W2D el eje y apunta hacia ABAJO (convención imagen).
    viz_pts = np.float32([
        [cx - half_px, cy - half_px],   # top-left
        [cx + half_px, cy - half_px],   # top-right
        [cx + half_px, cy + half_px],   # bot-right
        [cx - half_px, cy + half_px],   # bot-left
    ])
    H_img2viz, _ = cv2.findHomography(img_pts, viz_pts)

    # ── Vista cenital de fondo (SOLO aquí, no en el bucle) ───────────────────
    # warpPerspective aplica H_img2viz para rectificar la perspectiva:
    # cada píxel (u,v) de la imagen W2D se obtiene aplicando H⁻¹ sobre (u,v) en cámara.
    bg_image   = cv2.warpPerspective(frame, H_img2viz, (W2D_SIZE, W2D_SIZE))
    registered = True


# ──────────────────────────────────────────────────────────────────────────────
#  LOCALIZACIÓN (bucle continuo)
# ──────────────────────────────────────────────────────────────────────────────

def localizar(corners_marcador: np.ndarray):
    """
    Proyecta las esquinas del marcador al sistema de referencia del mundo (mm)
    y a la vista W2D (px).

    Retorna:
        centro_mm  : (x,y) en mm en el sistema de referencia del mundo
        angulo_deg : orientación en grados desde +x, positiva en sentido CCW
        viz_corners: esquinas proyectadas en la vista W2D (int px)
        centro_viz : centro del marcador en la vista W2D (int px)
    """
    pts = corners_marcador.reshape(-1, 1, 2).astype(np.float32)   # (4,1,2) para perspectiveTransform

    # Proyectar al mundo → coordenadas en mm
    world_pts  = cv2.perspectiveTransform(pts, H_img2world).reshape(4, 2)
    centro_mm  = world_pts.mean(axis=0)                           # centro = promedio de esquinas

    # Orientación: vector de top-left a top-right = dirección +x del marcador en mundo
    dx         = world_pts[1] - world_pts[0]
    angulo_deg = np.degrees(np.arctan2(dx[1], dx[0]))            # ángulo CCW desde +x (mundo y↑)

    # Proyectar a la vista W2D → píxeles
    viz_corners = cv2.perspectiveTransform(pts, H_img2viz).reshape(4, 2).astype(np.int32)
    centro_viz  = viz_corners.mean(axis=0).astype(np.int32)

    return centro_mm, angulo_deg, viz_corners, centro_viz


# ──────────────────────────────────────────────────────────────────────────────
#  DIBUJAR VENTANA W2D
# ──────────────────────────────────────────────────────────────────────────────

def dibujar_w2d(centro_mm, angulo_deg, viz_corners, centro_viz) -> np.ndarray:
    """
    Construye el frame de la ventana W2D con:
      - Fondo cenital (capturado al registrar)
      - Ejes canónicos centrados: x→derecha (rojo), y→arriba (verde)
      - Contorno cuadrado del marcador
      - Flecha de orientación (dirección +x del marcador)
      - Etiquetas: coordenadas en mm y ángulo
    """
    canvas = bg_image.copy()            # partir del fondo registrado, nunca modificarlo
    ox = oy = W2D_SIZE // 2             # origen de los ejes en el centro de la imagen

    # Ejes canónicos del sistema de referencia
    cv2.arrowedLine(canvas, (ox, oy), (ox+70, oy),   (0, 0, 220),  2, cv2.LINE_AA, 0, 0.2)  # x → rojo
    cv2.arrowedLine(canvas, (ox, oy), (ox, oy-70),   (0, 200, 0),  2, cv2.LINE_AA, 0, 0.2)  # y ↑ verde
    cv2.putText(canvas, "x", (ox+76, oy+6),   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 220),  1, cv2.LINE_AA)
    cv2.putText(canvas, "y", (ox-16, oy-74),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0),  1, cv2.LINE_AA)
    cv2.circle(canvas, (ox, oy), 4, (255, 255, 255), -1)   # punto en el origen

    # Contorno cuadrado del marcador
    cv2.polylines(canvas, [viz_corners], True, (255, 180, 0), 2, cv2.LINE_AA)

    # Flecha de orientación: desde el centro en la dirección +x del marcador
    # En la imagen W2D: x→derecha (+col), y→arriba (−fila)
    rad = np.radians(angulo_deg)
    tip = (int(centro_viz[0] + 60 * np.cos(rad)),
           int(centro_viz[1] - 60 * np.sin(rad)))    # -sin porque y imagen va hacia abajo
    cv2.arrowedLine(canvas, tuple(centro_viz), tip, (0, 255, 255), 3, cv2.LINE_AA, 0, 0.25)
    cv2.circle(canvas, tuple(centro_viz), 5, (0, 255, 255), -1)

    # Etiquetas
    lx, ly = int(centro_viz[0]) + 12, int(centro_viz[1]) - 12
    cv2.putText(canvas, f"({centro_mm[0]:+.1f}, {centro_mm[1]:+.1f}) mm",
                (lx, ly),    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{angulo_deg:.1f} deg",
                (lx, ly+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1, cv2.LINE_AA)

    return canvas


# ──────────────────────────────────────────────────────────────────────────────
#  OVERLAY AR: superponer imagen sobre el marcador en el frame de cámara
# ──────────────────────────────────────────────────────────────────────────────

def aplicar_overlay(frame: np.ndarray, corners_marcador: np.ndarray) -> np.ndarray:
    """
    Proyecta overlay_img sobre el área del marcador usando homografía.
      1. Calcula H: esquinas del overlay → esquinas del marcador en cámara
      2. warpPerspective: deforma el overlay para que encaje sobre el marcador
      3. Máscara binaria del área del marcador
      4. Composición: frame[máscara] = overlay deformado
    """
    h, w       = frame.shape[:2]
    ov_h, ov_w = overlay_img.shape[:2]

    src_pts = np.float32([[0,0],[ov_w,0],[ov_w,ov_h],[0,ov_h]])  # esquinas del overlay
    dst_pts = corners_marcador[0].astype(np.float32)              # esquinas del marcador en cámara

    H, _    = cv2.findHomography(src_pts, dst_pts)                # overlay → marcador
    warped  = cv2.warpPerspective(overlay_img, H, (w, h))         # deformar overlay

    mask    = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [dst_pts.astype(np.int32)], 255)           # área del marcador = blanco
    mask3   = cv2.merge([mask, mask, mask])                       # expandir a 3 canales

    return np.where(mask3 == 255, warped, frame).astype(np.uint8) # componer


# ──────────────────────────────────────────────────────────────────────────────
#  BUCLE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)           # abrir cámara por defecto
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        sys.exit(1)

    print("=" * 40)
    print("  R     → Registrar plano")
    print("  Q/ESC → Salir")
    print("=" * 40)

    # Frame inicial de W2D antes de registrar
    w2d_frame = np.zeros((W2D_SIZE, W2D_SIZE, 3), dtype=np.uint8)
    cv2.putText(w2d_frame, "Presione R para registrar el plano",
                (20, W2D_SIZE // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 150, 150), 1, cv2.LINE_AA)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Fallo al leer la cámara.")
            break

        # ── Detección ─────────────────────────────────────────────────────────
        corners, ids = detectar(frame)                             # detectar marcadores ArUco
        n = len(ids) if ids is not None else 0                     # cantidad detectada

        # ── Teclado ───────────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):                                  # Q o ESC → salir
            break
        if key == ord('r'):                                        # R → registrar
            if n > 0:
                registrar(frame, corners[0])                       # usar el primer marcador
                print(f"[OK] Plano registrado — marcador id={ids[0][0]}")
            else:
                print("[AVISO] No hay marcador visible. No se puede registrar.")

        # ── Ventana CAM ───────────────────────────────────────────────────────
        cam_frame = frame.copy()

        if n > 0:
            # Overlay AR si se pasó imagen como argumento
            if overlay_img is not None:
                cam_frame = aplicar_overlay(cam_frame, corners[0])

            # Contorno verde y etiqueta de CADA marcador detectado
            for corn, mid in zip(corners, ids):
                pts  = corn[0].astype(np.int32)                   # esquinas del marcador
                cv2.polylines(cam_frame, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
                cx_m = int(pts[:, 0].mean())
                cy_m = int(pts[:, 1].mean())
                cv2.putText(cam_frame, f"id={mid[0]}", (cx_m - 24, cy_m - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        # Estado + contador de marcadores
        if registered:
            cv2.putText(cam_frame, "REGISTRADO | R = re-registrar", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(cam_frame, "Presione R para registrar el plano", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2, cv2.LINE_AA)

        cv2.putText(cam_frame, f"Marcadores detectados: {n}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)

        # ── Ventana W2D ───────────────────────────────────────────────────────
        # Solo se actualiza si hay homografías registradas Y hay marcador visible
        if registered and n > 0:
            c_mm, ang, viz_c, c_viz = localizar(corners[0])
            w2d_frame = dibujar_w2d(c_mm, ang, viz_c, c_viz)
        elif registered:
            # Registrado pero marcador fuera del campo de visión → mostrar fondo fijo
            w2d_frame = bg_image.copy()
            cv2.putText(w2d_frame, "Marcador no detectado", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2, cv2.LINE_AA)

        cv2.imshow("Cam", cam_frame)    # ventana 1: cámara anotada
        cv2.imshow("W2D", w2d_frame)    # ventana 2: vista cenital del mundo 2D

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()