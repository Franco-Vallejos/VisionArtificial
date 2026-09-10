"""Genera dataset.csv a partir de las imagenes guardadas en ejemplares/."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


# Rutas y categorias
CARPETA_PROYECTO = Path(__file__).resolve().parent
CARPETA_EJEMPLARES = CARPETA_PROYECTO / "ejemplares"
ARCHIVO_DATASET = CARPETA_PROYECTO / "dataset.csv"
CATEGORIAS = {
    0: "Circulos",
    1: "Cuadrados",
    2: "Rectangulos",
    3: "Triangulos",
}
EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Procesamiento de imagen
TAMANIO_KERNEL = (3, 3)
ITERACIONES_MORFOLOGIA = 1
VALOR_BINARIO_MAXIMO = 255
UMBRAL_INICIAL_OTSU = 0
AREA_MINIMA_RELATIVA = 0.005
AREA_MAXIMA_RELATIVA = 0.95
MARGEN_BORDE = 2

# Columnas del CSV
PRIMER_INDICE_HU = 1
CANTIDAD_INVARIANTES_HU = 7
COLUMNAS_HU = [
    f"hu_{indice}"
    for indice in range(
        PRIMER_INDICE_HU,
        PRIMER_INDICE_HU + CANTIDAD_INVARIANTES_HU,
    )
]
COLUMNA_ETIQUETA = "etiqueta"
COLUMNAS_DATASET = COLUMNAS_HU + [COLUMNA_ETIQUETA]


def cargar_imagen(ruta: Path) -> np.ndarray | None:
    """Carga rutas con espacios o caracteres acentuados en Windows."""
    datos = np.fromfile(ruta, dtype=np.uint8)
    return cv2.imdecode(datos, cv2.IMREAD_COLOR)


def toca_borde(contorno: np.ndarray, ancho: int, alto: int) -> bool:
    x, y, w, h = cv2.boundingRect(contorno)
    return (
        x <= MARGEN_BORDE
        or y <= MARGEN_BORDE
        or x + w >= ancho - MARGEN_BORDE
        or y + h >= alto - MARGEN_BORDE
    )


def detectar_contornos(imagen: np.ndarray) -> list[np.ndarray]:
    """Segmenta objetos claros u oscuros y devuelve contornos validos."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, TAMANIO_KERNEL)
    alto, ancho = gris.shape
    area_imagen = alto * ancho
    area_minima = area_imagen * AREA_MINIMA_RELATIVA
    area_maxima = area_imagen * AREA_MAXIMA_RELATIVA
    validos: list[np.ndarray] = []

    for tipo in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, binaria = cv2.threshold(
            gris,
            UMBRAL_INICIAL_OTSU,
            VALOR_BINARIO_MAXIMO,
            tipo | cv2.THRESH_OTSU,
        )
        binaria = cv2.morphologyEx(
            binaria,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=ITERACIONES_MORFOLOGIA,
        )
        contornos, _ = cv2.findContours(
            binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if (
                area_minima <= area <= area_maxima
                and not toca_borde(contorno, ancho, alto)
            ):
                validos.append(contorno)

    return validos


def invariantes_hu(contorno: np.ndarray) -> np.ndarray:
    """Calcula los siete invariantes de Hu de un contorno."""
    momentos = cv2.moments(contorno)
    return cv2.HuMoments(momentos).flatten().astype(np.float64)


def imagenes_de(carpeta: Path) -> list[Path]:
    return sorted(
        ruta
        for ruta in carpeta.iterdir()
        if ruta.is_file() and ruta.suffix.lower() in EXTENSIONES_IMAGEN
    )


def main() -> None:
    filas: list[list[float | int]] = []
    rechazadas = 0

    for etiqueta, nombre in CATEGORIAS.items():
        carpeta_categoria = CARPETA_EJEMPLARES / nombre
        if not carpeta_categoria.is_dir():
            print(f"AVISO: no existe {carpeta_categoria}")
            continue

        aceptadas_categoria = 0
        for ruta_imagen in imagenes_de(carpeta_categoria):
            imagen = cargar_imagen(ruta_imagen)
            if imagen is None:
                print(f"RECHAZADA: no se pudo abrir {ruta_imagen.name}")
                rechazadas += 1
                continue

            contornos = detectar_contornos(imagen)
            if not contornos:
                print(f"RECHAZADA: no se encontro una forma en {ruta_imagen.name}")
                rechazadas += 1
                continue

            contorno_principal = max(contornos, key=cv2.contourArea)
            descriptor = invariantes_hu(contorno_principal)
            filas.append([float(valor) for valor in descriptor] + [etiqueta])
            aceptadas_categoria += 1

        print(f"{nombre}: {aceptadas_categoria} muestras aceptadas")

    if not filas:
        raise SystemExit("ERROR: no se pudo generar ninguna muestra.")

    with ARCHIVO_DATASET.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(COLUMNAS_DATASET)
        escritor.writerows(filas)

    print(f"Dataset guardado: {ARCHIVO_DATASET}")
    print(f"Muestras totales: {len(filas)}")
    print(f"Imagenes rechazadas: {rechazadas}")


if __name__ == "__main__":
    main()
