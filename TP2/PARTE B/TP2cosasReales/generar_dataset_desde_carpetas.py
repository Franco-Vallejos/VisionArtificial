"""Genera el dataset Hu a partir de carpetas de imágenes etiquetadas."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import cv2
import numpy as np

from config import BASE_DIR, DATASET_FILE, ETIQUETAS, cargar_parametros
from vision import COLUMNAS_DATASET, invariantes_hu, procesar_imagen


EXTENSIONES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convierte imágenes separadas por categoría en un dataset Hu."
    )
    parser.add_argument("--ejemplares", type=Path, default=BASE_DIR / "ejemplares")
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument(
        "--agregar", action="store_true",
        help="Agrega filas al CSV existente. Sin esta opción, el CSV se reconstruye.",
    )
    parser.add_argument(
        "--mostrar", action="store_true",
        help="Muestra el contorno elegido para revisar cada ejemplar.",
    )
    return parser.parse_args()


def leer_imagen(path: Path) -> np.ndarray | None:
    """Lee rutas con caracteres especiales, incluso en Windows."""
    try:
        datos = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def imagenes_de(carpeta: Path) -> list[Path]:
    if not carpeta.is_dir():
        return []
    return sorted(
        path for path in carpeta.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONES
    )


def extraer_fila(path: Path, etiqueta: int, parametros: dict[str, int]):
    imagen = leer_imagen(path)
    if imagen is None:
        return None, None, "no se pudo leer"

    proceso = procesar_imagen(
        imagen,
        parametros["umbral"],
        bool(parametros["invertir"]),
        parametros["morfologia"],
        max(1, parametros["area_min_x100"] * 100),
        parametros["area_max_pct"],
    )
    indices = proceso["indices_validos"]
    if not indices:
        return None, imagen, "no se encontró un contorno válido"

    contorno = proceso["contornos"][indices[0]]
    hu = invariantes_hu(contorno)
    if not np.isfinite(hu).all():
        return None, imagen, "los invariantes Hu no son finitos"

    vista = imagen.copy()
    cv2.drawContours(vista, [contorno], -1, (0, 255, 0), 3)
    x, y, ancho, alto = cv2.boundingRect(contorno)
    cv2.rectangle(vista, (x, y), (x + ancho, y + alto), (0, 255, 0), 2)
    cv2.putText(
        vista, f"{ETIQUETAS[etiqueta]} - {path.name}", (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA,
    )
    fila = [f"{float(valor):.17e}" for valor in hu] + [etiqueta]
    return fila, vista, None


def main() -> None:
    args = parsear_argumentos()
    parametros = cargar_parametros()
    filas: list[list[str | int]] = []
    cantidades: Counter[int] = Counter()
    rechazadas: list[tuple[Path, str]] = []

    print(f"Carpeta de ejemplares: {args.ejemplares.resolve()}")
    print(
        "Parámetros: "
        f"umbral={parametros['umbral']}, invertir={parametros['invertir']}, "
        f"morfología={parametros['morfologia']}, "
        f"área mínima={parametros['area_min_x100'] * 100}, "
        f"área máxima={parametros['area_max_pct']}%"
    )

    for etiqueta, nombre in ETIQUETAS.items():
        carpeta = args.ejemplares / nombre
        archivos = imagenes_de(carpeta)
        print(f"{nombre}: {len(archivos)} archivo(s)")
        for path in archivos:
            fila, vista, error = extraer_fila(path, etiqueta, parametros)
            if error:
                rechazadas.append((path, error))
                print(f"  RECHAZADA {path.name}: {error}")
                continue

            if args.mostrar and vista is not None:
                cv2.imshow("Revisión de ejemplares", vista)
                tecla = cv2.waitKey(0) & 0xFF
                if tecla in (27, ord("q")):
                    cv2.destroyAllWindows()
                    print("Proceso cancelado; no se modificó el dataset.")
                    return
                if tecla in (ord("r"), ord("n")):
                    rechazadas.append((path, "rechazada manualmente"))
                    print(f"  RECHAZADA {path.name}: revisión manual")
                    continue

            filas.append(fila)
            cantidades[etiqueta] += 1

    cv2.destroyAllWindows()
    if not filas:
        print("ERROR: no se encontró ningún ejemplar utilizable.")
        sys.exit(1)

    args.dataset.parent.mkdir(parents=True, exist_ok=True)
    modo = "a" if args.agregar else "w"
    escribir_cabecera = (
        not args.agregar
        or not args.dataset.is_file()
        or args.dataset.stat().st_size == 0
    )
    with args.dataset.open(modo, newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        if escribir_cabecera:
            escritor.writerow(COLUMNAS_DATASET)
        escritor.writerows(filas)

    accion = "actualizado" if args.agregar else "reconstruido"
    print(f"Dataset {accion}: {args.dataset.resolve()}")
    for etiqueta, nombre in ETIQUETAS.items():
        print(f"  {nombre}: {cantidades[etiqueta]} muestra(s) aceptada(s)")
    print(f"  Rechazadas: {len(rechazadas)}")


if __name__ == "__main__":
    main()
