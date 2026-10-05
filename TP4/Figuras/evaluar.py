from pathlib import Path

from ultralytics import YOLO


RAIZ = Path(__file__).resolve().parent
ARCHIVO_DATASET = RAIZ / "dataset.yaml"
ARCHIVO_MODELO = RAIZ / "resultados" / "figuras_yolo26n" / "weights" / "best.pt"
CARPETA_RESULTADOS = RAIZ / "resultados"
NOMBRE_EVALUACION = "evaluacion_test"
TAMANIO_IMAGEN = 320
TAMANIO_LOTE = 32


def main():
    """Evalua el mejor modelo sobre la particion test que no participo del entrenamiento."""
    if not ARCHIVO_MODELO.exists():
        raise SystemExit(f"No existe el modelo entrenado: {ARCHIVO_MODELO}")

    modelo = YOLO(str(ARCHIVO_MODELO))
    metricas = modelo.val(data=str(ARCHIVO_DATASET), split="test", imgsz=TAMANIO_IMAGEN, batch=TAMANIO_LOTE, project=str(CARPETA_RESULTADOS), name=NOMBRE_EVALUACION, plots=True)
    print(f"mAP50: {metricas.box.map50:.4f}")
    print(f"mAP50-95: {metricas.box.map:.4f}")
    print(f"Resultados: {metricas.save_dir}")


if __name__ == "__main__":
    main()
