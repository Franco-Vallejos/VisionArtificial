from pathlib import Path

from ultralytics import YOLO


RAIZ = Path(__file__).resolve().parent
ARCHIVO_DATASET = RAIZ / "dataset.yaml"
MODELO_BASE = "yolo26n.pt"
EPOCAS = 20
TAMANIO_IMAGEN = 320
TAMANIO_LOTE = 32
TRABAJADORES = 0
SEMILLA = 42
CARPETA_RESULTADOS = RAIZ / "resultados"
NOMBRE_ENTRENAMIENTO = "figuras_yolo26n"
ROTACION_GRADOS = 25.0
TRASLACION = 0.15
ESCALA = 0.40
ESPEJO_HORIZONTAL = 0.50
VARIACION_TONO = 0.05
VARIACION_SATURACION = 0.30
VARIACION_BRILLO = 0.25
MOSAICO = 1.0


def main():
    """Entrena por transferencia un detector YOLO nano con el dataset de figuras."""
    if not (RAIZ / "dataset_yolo").exists():
        raise SystemExit(f"No existe el dataset: {RAIZ / 'dataset_yolo'}")

    modelo = YOLO(MODELO_BASE)
    modelo.train(data=str(ARCHIVO_DATASET), epochs=EPOCAS, imgsz=TAMANIO_IMAGEN, batch=TAMANIO_LOTE, workers=TRABAJADORES, seed=SEMILLA, deterministic=True, degrees=ROTACION_GRADOS, translate=TRASLACION, scale=ESCALA, fliplr=ESPEJO_HORIZONTAL, hsv_h=VARIACION_TONO, hsv_s=VARIACION_SATURACION, hsv_v=VARIACION_BRILLO, mosaic=MOSAICO, project=str(CARPETA_RESULTADOS), name=NOMBRE_ENTRENAMIENTO, exist_ok=True, plots=True)
    mejor_modelo = CARPETA_RESULTADOS / NOMBRE_ENTRENAMIENTO / "weights" / "best.pt"
    print(f"Entrenamiento terminado. Mejor modelo: {mejor_modelo}")


if __name__ == "__main__":
    main()
