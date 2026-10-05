from pathlib import Path

import cv2
from ultralytics import YOLO


RAIZ = Path(__file__).resolve().parent
ARCHIVO_MODELO = RAIZ / "resultados" / "figuras_yolo26n" / "weights" / "best.pt"
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
    resultados = modelo.predict(source=CAMARA_WEB_INDEX, conf=CONFIANZA, imgsz=TAMANIO_IMAGEN, stream=True, verbose=False)

    for resultado in resultados:
        frame_anotado = resultado.plot()
        cv2.imshow(VENTANA, frame_anotado)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (TECLA_ESCAPE, TECLA_SALIR):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
