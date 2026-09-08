"""Aplicacion 1: genera el dataset de invariantes de Hu con la webcam.

Mostra un solo objeto por vez, selecciona su etiqueta y presiona ESPACIO.
La aplicacion guarda el contorno valido de mayor area en datos/dataset_hu.csv.
"""

from __future__ import annotations

from collections import Counter
import csv
import sys

import cv2

from config import DATASET_FILE, ETIQUETAS, Controles
from vision import COLUMNAS_DATASET, crear_tablero, invariantes_hu, procesar_imagen


VENTANA_CONTROLES = "Controles - generador"
VENTANA_PROCESO = "Generador de descriptores"


def cargar_cantidades() -> Counter[int]:
    cantidades: Counter[int] = Counter()
    if not DATASET_FILE.is_file():
        return cantidades
    with DATASET_FILE.open("r", newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            try:
                cantidades[int(fila["etiqueta"])] += 1
            except (KeyError, TypeError, ValueError):
                continue
    return cantidades


def guardar_muestra(hu, etiqueta: int) -> None:
    DATASET_FILE.parent.mkdir(exist_ok=True)
    nuevo = not DATASET_FILE.is_file() or DATASET_FILE.stat().st_size == 0
    with DATASET_FILE.open("a", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        if nuevo:
            escritor.writerow(COLUMNAS_DATASET)
        escritor.writerow([f"{float(valor):.17e}" for valor in hu] + [etiqueta])


def main() -> None:
    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("ERROR: no se pudo abrir la webcam.")
        sys.exit(1)

    cv2.namedWindow(VENTANA_CONTROLES, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_CONTROLES, 520, 260)
    controles = Controles(
        VENTANA_CONTROLES,
        ["umbral", "invertir", "morfologia", "area_min_x100", "area_max_pct"],
    )
    cv2.createTrackbar(
        "Etiqueta", VENTANA_CONTROLES, 1, max(ETIQUETAS), lambda _valor: None
    )

    cv2.namedWindow(VENTANA_PROCESO, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_PROCESO, 1000, 900)
    cantidades = cargar_cantidades()

    print("Categorias:")
    for numero, nombre in ETIQUETAS.items():
        print(f"  {numero}: {nombre}")
    print("ESPACIO guarda una muestra. q o ESC finaliza.")

    while True:
        ok, frame = captura.read()
        if not ok:
            print("ERROR: no se pudo leer un cuadro de la webcam.")
            break

        valores = controles.leer()
        controles.guardar_si_cambiaron(valores)
        etiqueta = max(1, cv2.getTrackbarPos("Etiqueta", VENTANA_CONTROLES))
        if etiqueta not in ETIQUETAS:
            etiqueta = min(ETIQUETAS)
            cv2.setTrackbarPos("Etiqueta", VENTANA_CONTROLES, etiqueta)

        proceso = procesar_imagen(
            frame,
            valores["umbral"],
            bool(valores["invertir"]),
            valores["morfologia"],
            max(1, valores["area_min_x100"] * 100),
            valores["area_max_pct"],
        )

        seleccion = frame.copy()
        indices = proceso["indices_validos"]
        contorno_elegido = None
        for posicion, indice in enumerate(indices):
            contorno = proceso["contornos"][indice]
            color = (0, 255, 0) if posicion == 0 else (0, 255, 255)
            grosor = 3 if posicion == 0 else 1
            cv2.drawContours(seleccion, [contorno], -1, color, grosor)
            if posicion == 0:
                contorno_elegido = contorno
                x, y, ancho, alto = cv2.boundingRect(contorno)
                cv2.rectangle(
                    seleccion, (x, y), (x + ancho, y + alto), color, 2
                )

        nombre = ETIQUETAS[etiqueta]
        resumen = "  ".join(
            f"{ETIQUETAS[numero]}={cantidades[numero]}" for numero in ETIQUETAS
        )
        cv2.putText(
            seleccion,
            f"Etiqueta actual: {etiqueta} - {nombre}",
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            seleccion,
            resumen,
            (15, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        etapas = {
            "0. Original": frame,
            "1. Monocromatica": proceso["gris"],
            "2. Binaria": proceso["binaria"],
            "3. Morfologia": proceso["morfologica"],
            "4. Contornos validos": proceso["vista_contornos"],
            "5. Contorno a guardar": seleccion,
        }
        cv2.imshow(VENTANA_PROCESO, crear_tablero(etapas))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (27, ord("q")):
            break
        if tecla == ord(" "):
            if contorno_elegido is None:
                print("No se guardo: no hay un contorno valido.")
                continue
            hu = invariantes_hu(contorno_elegido)
            guardar_muestra(hu, etiqueta)
            cantidades[etiqueta] += 1
            valores_texto = ", ".join(f"{valor:.8e}" for valor in hu)
            print(f"{nombre}: [{valores_texto}]")
            print(f"Muestras acumuladas: {dict(cantidades)}")

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
