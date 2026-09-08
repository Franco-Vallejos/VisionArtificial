"""Aplicacion 3: clasifica en tiempo real con el modelo entrenado."""

from __future__ import annotations

import sys

import cv2
from joblib import load
import numpy as np

from config import (
    COLORES,
    COLOR_DESCONOCIDO,
    MODEL_FILE,
    Controles,
)
from vision import COLUMNAS_HU, crear_tablero, invariantes_hu, procesar_imagen


VENTANA_CONTROLES = "Controles - clasificador"
VENTANA_PROCESO = "Clasificacion con Machine Learning"


def cargar_modelo():
    if not MODEL_FILE.is_file():
        raise ValueError(
            f"No existe {MODEL_FILE}. Primero ejecuta entrenar_modelo.py."
        )
    artefacto = load(MODEL_FILE)
    if not isinstance(artefacto, dict) or "modelo" not in artefacto:
        raise ValueError("El archivo no contiene un modelo valido de la Parte B.")
    if artefacto.get("columnas") != COLUMNAS_HU:
        raise ValueError("El modelo no usa los siete invariantes de Hu esperados.")
    etiquetas = {int(k): v for k, v in artefacto.get("etiquetas", {}).items()}
    return artefacto["modelo"], etiquetas


def predecir(modelo, hu: np.ndarray) -> tuple[int, float]:
    muestra = hu.reshape(1, -1)
    etiqueta = int(modelo.predict(muestra)[0])
    confianza = 1.0
    if hasattr(modelo, "predict_proba"):
        confianza = float(np.max(modelo.predict_proba(muestra)[0]))
    return etiqueta, confianza


def clasificar_frame(frame, modelo, etiquetas, valores):
    """Procesa y clasifica un cuadro; se comparte entre webcam y video."""
    proceso = procesar_imagen(
        frame,
        valores["umbral"],
        bool(valores["invertir"]),
        valores["morfologia"],
        max(1, valores["area_min_x100"] * 100),
        valores["area_max_pct"],
    )

    anotada = frame.copy()
    confianza_minima = valores["confianza_x100"] / 100.0
    for indice in proceso["indices_validos"]:
        contorno = proceso["contornos"][indice]
        hu = invariantes_hu(contorno)
        etiqueta, confianza = predecir(modelo, hu)
        reconocido = etiqueta in etiquetas and confianza >= confianza_minima
        if reconocido:
            nombre = etiquetas[etiqueta]
            color = COLORES.get(etiqueta, (0, 200, 0))
            texto = f"{nombre}  p={confianza:.2f}"
        else:
            color = COLOR_DESCONOCIDO
            texto = f"desconocido  p={confianza:.2f}"

        x, y, ancho, alto = cv2.boundingRect(contorno)
        cv2.rectangle(anotada, (x, y), (x + ancho, y + alto), color, 2)
        cv2.drawContours(anotada, [contorno], -1, color, 2)
        cv2.putText(
            anotada,
            texto,
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    return {
        "0. Original": frame,
        "1. Monocromatica": proceso["gris"],
        "2. Binaria": proceso["binaria"],
        "3. Morfologia": proceso["morfologica"],
        "4. Contornos validos": proceso["vista_contornos"],
        "5. Clasificacion ML": anotada,
    }


def main() -> None:
    try:
        modelo, etiquetas = cargar_modelo()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    captura = cv2.VideoCapture(0)
    if not captura.isOpened():
        print("ERROR: no se pudo abrir la webcam.")
        sys.exit(1)

    cv2.namedWindow(VENTANA_CONTROLES, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_CONTROLES, 520, 300)
    controles = Controles(
        VENTANA_CONTROLES,
        [
            "umbral",
            "invertir",
            "morfologia",
            "area_min_x100",
            "area_max_pct",
            "confianza_x100",
        ],
    )
    cv2.namedWindow(VENTANA_PROCESO, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_PROCESO, 1000, 900)
    print(f"Modelo cargado: {MODEL_FILE}")
    print("q o ESC finaliza.")

    while True:
        ok, frame = captura.read()
        if not ok:
            print("ERROR: no se pudo leer un cuadro de la webcam.")
            break

        valores = controles.leer()
        controles.guardar_si_cambiaron(valores)
        etapas = clasificar_frame(frame, modelo, etiquetas, valores)
        cv2.imshow(VENTANA_PROCESO, crear_tablero(etapas))

        tecla = cv2.waitKey(1) & 0xFF
        if tecla in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
