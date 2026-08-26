"""Detecta objetos en las imagenes de la carpeta 'imagen'."""

from pathlib import Path
import urllib.request

import cv2
import mediapipe as mp


CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_IMAGENES = CARPETA_PROYECTO / "Imagenes"
CARPETA_RESULTADOS = CARPETA_PROYECTO / "Resultados"
RUTA_MODELO = CARPETA_PROYECTO / "modelo" / "efficientdet_lite0.tflite"
THRESHOLD = 0.1
MAX_RESULT = 15

URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/object_detector/"
    "efficientdet_lite0/float32/1/efficientdet_lite0.tflite"
)


def descargar_modelo():
    if not RUTA_MODELO.exists():
        RUTA_MODELO.parent.mkdir(exist_ok=True)
        print("Descargando modelo...")
        urllib.request.urlretrieve(URL_MODELO, RUTA_MODELO)


def crear_detector():
    opciones = mp.tasks.vision.ObjectDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(RUTA_MODELO)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        score_threshold=THRESHOLD,
        max_results=MAX_RESULT,
    )
    return mp.tasks.vision.ObjectDetector.create_from_options(opciones)


def detectar_objetos(detector, imagen):
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    imagen_mediapipe = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=imagen_rgb,
    )

    resultado = detector.detect(imagen_mediapipe)
    imagen_resultado = imagen.copy()

    for deteccion in resultado.detections:
        categoria = deteccion.categories[0]
        nombre = categoria.category_name
        confianza = categoria.score
        caja = deteccion.bounding_box

        inicio = (caja.origin_x, caja.origin_y)
        fin = (
            caja.origin_x + caja.width,
            caja.origin_y + caja.height,
        )

        cv2.rectangle(
            imagen_resultado,  # Imagen donde se dibuja
            inicio,             # Esquina superior izquierda
            fin,                # Esquina inferior derecha
            (40, 200, 40),      # Color
            2                   # Grosor de la línea
        )

        texto = f"{nombre}: {confianza:.0%}"
        texto_y = max(25, caja.origin_y - 10)
        cv2.putText(
            imagen_resultado,
            texto,
            (caja.origin_x, texto_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4, # Tamaño de la fuente
            (40, 200, 40), # Color
            2, # Grosor de la fuente
        )

    return imagen_resultado


def main():
    CARPETA_IMAGENES.mkdir(exist_ok=True)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)

    imagenes = [
        ruta
        for ruta in CARPETA_IMAGENES.iterdir()
        if ruta.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not imagenes:
        print("No hay imagenes en la carpeta:", CARPETA_IMAGENES)
        return

    descargar_modelo()

    detector = crear_detector()

    for ruta_imagen in imagenes:
        imagen = cv2.imread(str(ruta_imagen))
        imagen_resultado = detectar_objetos(detector, imagen)
        ruta_resultado = CARPETA_RESULTADOS / f"{ruta_imagen.stem}_resultado.jpg"
        cv2.imwrite(str(ruta_resultado), imagen_resultado)


if __name__ == "__main__":
    main()
