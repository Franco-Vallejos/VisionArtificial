from pathlib import Path

import cv2
from ultralytics import YOLO


RAIZ = Path(__file__).resolve().parent
ARCHIVO_MODELO = RAIZ / "resultados" / "figuras_yolov8n" / "weights" / "best.pt"
CAMARA_WEB_INDEX = 0
CONFIANZA = 0.40
TAMANIO_IMAGEN = 320
VENTANA = "Deteccion YOLO"
TECLA_ESCAPE = 27
TECLA_SALIR = ord("q")


def main():
    """Carga los pesos entrenados y ejecuta inferencia YOLO sobre la camara en tiempo real."""
    if not ARCHIVO_MODELO.exists():
        raise SystemExit(f"No existe el modelo entrenado: {ARCHIVO_MODELO}")

    modelo = YOLO(str(ARCHIVO_MODELO))
    camara = cv2.VideoCapture(CAMARA_WEB_INDEX)
    if not camara.isOpened():
        raise SystemExit(f"No se pudo abrir la camara con indice {CAMARA_WEB_INDEX}")

    try:
        while True:
            cuadro_disponible, frame = camara.read()
            if not cuadro_disponible:
                raise SystemExit("La camara dejo de entregar imagenes")

            resultado = modelo.predict(source=frame, conf=CONFIANZA, imgsz=TAMANIO_IMAGEN, verbose=False)[0]
            cv2.imshow(VENTANA, resultado.plot())
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (TECLA_ESCAPE, TECLA_SALIR):
                break
    finally:
        camara.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
