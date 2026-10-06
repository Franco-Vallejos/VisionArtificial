from pathlib import Path

from ultralytics import YOLO


RAIZ = Path(__file__).resolve().parent
ARCHIVO_DATASET = RAIZ / "data.yaml"
ARCHIVO_MODELO = RAIZ / "resultados" / "figuras_yolov8n" / "weights" / "best.pt"
CARPETA_RESULTADOS = RAIZ / "resultados"
NOMBRE_EVALUACION_METRICAS = "evaluacion_metricas"
NOMBRE_EVALUACION_OPERATIVA = "evaluacion_operativa"
TAMANIO_IMAGEN = 320
TAMANIO_LOTE = 32
CONFIANZA_OPERATIVA = 0.55


def main():
    """Evalua el mejor modelo sobre la particion test que no participo del entrenamiento."""
    if not ARCHIVO_DATASET.exists():
        raise SystemExit(f"No existe la configuracion del dataset: {ARCHIVO_DATASET}")
    if not ARCHIVO_MODELO.exists():
        raise SystemExit(f"No existe el modelo entrenado: {ARCHIVO_MODELO}")

    modelo = YOLO(str(ARCHIVO_MODELO))
    metricas = modelo.val(data=str(ARCHIVO_DATASET), split="test", imgsz=TAMANIO_IMAGEN, batch=TAMANIO_LOTE, project=str(CARPETA_RESULTADOS), name=NOMBRE_EVALUACION_METRICAS, exist_ok=True, plots=True)
    operativa = modelo.val(data=str(ARCHIVO_DATASET), split="test", imgsz=TAMANIO_IMAGEN, batch=TAMANIO_LOTE, conf=CONFIANZA_OPERATIVA, project=str(CARPETA_RESULTADOS), name=NOMBRE_EVALUACION_OPERATIVA, exist_ok=True, plots=True)
    print(f"mAP50: {metricas.box.map50:.4f}")
    print(f"mAP50-95: {metricas.box.map:.4f}")
    print(f"Resultados de metricas: {metricas.save_dir}")
    print(f"Resultados operativos: {operativa.save_dir}")


if __name__ == "__main__":
    main()
