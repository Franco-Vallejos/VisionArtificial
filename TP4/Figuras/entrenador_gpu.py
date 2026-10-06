from pathlib import Path

import torch
from ultralytics import YOLO


RAIZ = Path(__file__).resolve().parent
ARCHIVO_DATASET = RAIZ / "data.yaml"
MODELO_BASE = "yolov8n.pt"
DISPOSITIVO_GPU = 0
EPOCAS = 50
TAMANIO_IMAGEN = 320
TAMANIO_LOTE = 8
TRABAJADORES = 4
CARPETA_RESULTADOS = RAIZ / "resultados"
NOMBRE_ENTRENAMIENTO = "figuras_yolov8n"
ROTACION_GRADOS = 45.0
TRASLACION = 0.15
ESCALA = 0.40
ESPEJO_HORIZONTAL = 0.50
VARIACION_TONO = 0.05
VARIACION_SATURACION = 0.30
VARIACION_BRILLO = 0.25
MOSAICO = 1.0


def main():
    """Entrena el detector YOLO utilizando una GPU NVIDIA disponible mediante CUDA."""
    if not ARCHIVO_DATASET.exists():
        raise SystemExit(f"No existe la configuracion del dataset: {ARCHIVO_DATASET}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no esta disponible. Verificar la GPU NVIDIA y la instalacion de PyTorch con CUDA.")

    print(f"GPU: {torch.cuda.get_device_name(DISPOSITIVO_GPU)}")
    modelo = YOLO(MODELO_BASE)
    modelo.train(data=str(ARCHIVO_DATASET), device=DISPOSITIVO_GPU, epochs=EPOCAS, imgsz=TAMANIO_IMAGEN, batch=TAMANIO_LOTE, workers=TRABAJADORES, degrees=ROTACION_GRADOS, translate=TRASLACION, scale=ESCALA, fliplr=ESPEJO_HORIZONTAL, hsv_h=VARIACION_TONO, hsv_s=VARIACION_SATURACION, hsv_v=VARIACION_BRILLO, mosaic=MOSAICO, project=str(CARPETA_RESULTADOS), name=NOMBRE_ENTRENAMIENTO, exist_ok=True, plots=True)
    mejor_modelo = CARPETA_RESULTADOS / NOMBRE_ENTRENAMIENTO / "weights" / "best.pt"
    print(f"Entrenamiento terminado. Mejor modelo: {mejor_modelo}")


if __name__ == "__main__":
    main()
