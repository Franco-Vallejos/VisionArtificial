"""Detector didáctico de objetos reales por contorno.

Usa la webcam y las imágenes de ``refs/`` como referencias.  No requiere
modelo entrenado: reconoce únicamente las clases cuya silueta se parezca a
una de las referencias (teléfono, mate y mouse en esta primera versión).

Controles:
  Umbral binario  : separa objeto y fondo en la webcam.
  Morfologia      : tamaño del elemento estructural (0 la desactiva).
  Area min x100   : descarta regiones pequeñas antes de clasificar.
  Distancia x100  : tolerancia máxima de ``matchShapes``.
  Umbral refs     : vuelve a extraer las siluetas de las fotos de referencia.

Teclas: q o ESC para salir.
"""

from pathlib import Path
import sys

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
REFS_DIR = BASE_DIR / "refs"

COLORES = [
    (50, 200, 50),
    (255, 150, 30),
    (30, 130, 255),
    (200, 30, 200),
]
COLOR_DESCONOCIDO = (0, 0, 220)


def extraer_contorno_referencia(ruta: Path, umbral: int):
    # Obtiene contorno exterior y proporcion del mayor agujero interno.
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        return None

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Fondo negro y objeto claro: el objeto debe quedar blanco.
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
    contornos, jerarquia = cv2.findContours(
        binaria, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contornos or jerarquia is None:
        return None

    externos = [i for i, h in enumerate(jerarquia[0]) if h[3] == -1]
    indice = max(externos, key=lambda i: cv2.contourArea(contornos[i]))
    area = max(cv2.contourArea(contornos[indice]), 1.0)
    hijos = [i for i, h in enumerate(jerarquia[0]) if h[3] == indice]
    proporcion_agujero = max(
        (cv2.contourArea(contornos[i]) for i in hijos), default=0.0
    ) / area
    return contornos[indice], proporcion_agujero


def cargar_referencias(umbral: int) -> dict:
    referencias = {}
    extensiones = {".png", ".jpg", ".jpeg", ".bmp"}
    for ruta in sorted(REFS_DIR.iterdir() if REFS_DIR.exists() else []):
        if ruta.suffix.lower() not in extensiones:
            continue
        datos = extraer_contorno_referencia(ruta, umbral)
        if datos is not None:
            contorno, proporcion_agujero = datos
            referencias[ruta.stem] = {
                "contorno": contorno,
                "proporcion_agujero": proporcion_agujero,
            }
    return referencias


def procesar_frame(frame, referencias, umbral, morfologia, area_min, dist_max):
    # Ejecuta las etapas y usa el agujero central como señal extra del CD.
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Fondo negro y objetos claros: las regiones de interes quedan blancas.
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)

    morfologica = binaria.copy()
    if morfologia > 0:
        lado = 2 * morfologia + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
        morfologica = cv2.morphologyEx(morfologica, cv2.MORPH_CLOSE, kernel)
        morfologica = cv2.morphologyEx(morfologica, cv2.MORPH_OPEN, kernel)

    contornos, jerarquia = cv2.findContours(
        morfologica, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if jerarquia is None:
        contornos, jerarquia = [], np.empty((1, 0, 4), dtype=np.int32)
    externos = [
        i for i, h in enumerate(jerarquia[0])
        if h[3] == -1 and cv2.contourArea(contornos[i]) >= area_min
    ]

    mascara_segmentada = np.zeros_like(morfologica)
    vista_contornos = cv2.cvtColor(morfologica, cv2.COLOR_GRAY2BGR)
    for indice in externos:
        cv2.drawContours(mascara_segmentada, contornos, indice, 255, cv2.FILLED)
        cv2.drawContours(vista_contornos, contornos, indice, (0, 255, 255), 2)
        hijos = [i for i, h in enumerate(jerarquia[0]) if h[3] == indice]
        for hijo in hijos:
            cv2.drawContours(mascara_segmentada, contornos, hijo, 0, cv2.FILLED)
            cv2.drawContours(vista_contornos, contornos, hijo, (255, 0, 255), 2)

    resultado = frame.copy()
    nombres = list(referencias)
    for indice in externos:
        contorno = contornos[indice]
        area = max(cv2.contourArea(contorno), 1.0)
        hijos = [i for i, h in enumerate(jerarquia[0]) if h[3] == indice]
        proporcion_agujero = max(
            (cv2.contourArea(contornos[i]) for i in hijos), default=0.0
        ) / area

        nombre, distancia = None, float("inf")
        for candidato, datos in referencias.items():
            # El CD debe tener un agujero central de proporcion comparable.
            diferencia_agujero = abs(
                proporcion_agujero - datos["proporcion_agujero"]
            )
            if candidato.lower() == "cd" and diferencia_agujero > 0.015:
                continue
            valor = cv2.matchShapes(
                contorno, datos["contorno"], cv2.CONTOURS_MATCH_I2, 0
            )
            if valor < distancia:
                nombre, distancia = candidato, valor

        x, y, ancho, alto = cv2.boundingRect(contorno)
        if nombre is not None and distancia <= dist_max:
            color = COLORES[nombres.index(nombre) % len(COLORES)]
            detalle = f"  h={proporcion_agujero:.2f}" if nombre.lower() == "cd" else ""
            etiqueta = f"{nombre}{detalle}  d={distancia:.3f}"
        else:
            color = COLOR_DESCONOCIDO
            etiqueta = f"desconocido  h={proporcion_agujero:.2f}"

        cv2.rectangle(resultado, (x, y), (x + ancho, y + alto), color, 2)
        for hijo in hijos:
            cv2.drawContours(resultado, contornos, hijo, (255, 0, 255), 2)
        cv2.putText(
            resultado, etiqueta, (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA,
        )

    return {
        "Original": frame,
        "1. Monocromatica": gris,
        "2. Binaria": binaria,
        "3. Morfologia": morfologica,
        "4. Segmentacion": mascara_segmentada,
        "5. Contornos": vista_contornos,
        "6. Clasificacion": resultado,
    }


def panel(titulo: str, imagen: np.ndarray, ancho=360, alto=250):
    """Prepara una imagen con título y tamaño uniforme para el tablero."""
    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

    factor = min(ancho / imagen.shape[1], (alto - 30) / imagen.shape[0])
    nuevo = cv2.resize(imagen, None, fx=factor, fy=factor)
    lienzo = np.full((alto, ancho, 3), 238, dtype=np.uint8)
    y = 30 + (alto - 30 - nuevo.shape[0]) // 2
    x = (ancho - nuevo.shape[1]) // 2
    lienzo[y:y + nuevo.shape[0], x:x + nuevo.shape[1]] = nuevo
    cv2.putText(lienzo, titulo, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.58, (25, 25, 25), 2, cv2.LINE_AA)
    return lienzo


def crear_tablero(etapas: dict[str, np.ndarray]):
    vistas = [panel(titulo, imagen) for titulo, imagen in etapas.items()]
    # Original + seis etapas: completa la ultima celda para formar una grilla.
    if len(vistas) % 2:
        vistas.append(np.full((250, 360, 3), 238, dtype=np.uint8))
    filas = [np.hstack(vistas[i:i + 2]) for i in range(0, len(vistas), 2)]
    return np.vstack(filas)


def main():
    ventana_config = "Controles"
    ventana_proceso = "Proceso de reconocimiento"

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: no se pudo abrir la webcam.")
        sys.exit(1)

    cv2.namedWindow(ventana_config, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana_config, 470, 220)
    cv2.createTrackbar("Umbral binario", ventana_config, 127, 255, lambda _: None)
    cv2.createTrackbar("Morfologia", ventana_config, 2, 20, lambda _: None)
    cv2.createTrackbar("Area min x100", ventana_config, 5, 100, lambda _: None)
    cv2.createTrackbar("Distancia x100", ventana_config, 15, 200, lambda _: None)
    cv2.createTrackbar("Umbral refs", ventana_config, 127, 255, lambda _: None)

    cv2.namedWindow(ventana_proceso, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana_proceso, 900, 900)

    umbral_refs_anterior = None
    referencias = {}
    print("Presiona q o ESC para salir.")

    while True:
        umbral = cv2.getTrackbarPos("Umbral binario", ventana_config)
        morfologia = cv2.getTrackbarPos("Morfologia", ventana_config)
        area_min = max(1, cv2.getTrackbarPos("Area min x100", ventana_config) * 100)
        dist_max = cv2.getTrackbarPos("Distancia x100", ventana_config) / 100.0
        umbral_refs = cv2.getTrackbarPos("Umbral refs", ventana_config)

        # Recalcular sólo cuando cambia el control que afecta las referencias.
        if umbral_refs != umbral_refs_anterior:
            referencias = cargar_referencias(umbral_refs)
            umbral_refs_anterior = umbral_refs
            print(f"Referencias activas: {list(referencias)}")

        ok, frame = cap.read()
        if not ok:
            print("ERROR: no se pudo leer un cuadro de la webcam.")
            break

        etapas = procesar_frame(
            frame, referencias, umbral, morfologia, area_min, dist_max
        )
        cv2.imshow(ventana_proceso, crear_tablero(etapas))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
