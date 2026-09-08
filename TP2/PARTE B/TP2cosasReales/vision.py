"""Pipeline de segmentacion, contornos y descriptores compartido."""

from __future__ import annotations

import cv2
import numpy as np


COLUMNAS_HU = [f"hu_{indice}" for indice in range(1, 8)]
COLUMNAS_DATASET = COLUMNAS_HU + ["etiqueta"]


def invariantes_hu(contorno: np.ndarray) -> np.ndarray:
    """Calcula los siete invariantes de Hu de un contorno."""
    momentos = cv2.moments(contorno)
    return cv2.HuMoments(momentos).flatten().astype(np.float64)


def procesar_imagen(
    frame: np.ndarray,
    umbral: int,
    invertir: bool,
    morfologia: int,
    area_min: float,
    area_max_pct: float,
    margen_borde: int = 1,
) -> dict[str, object]:
    """Segmenta y devuelve todos los contornos externos considerados validos."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY
    _, binaria = cv2.threshold(gris, umbral, 255, tipo)

    morfologica = binaria.copy()
    if morfologia > 0:
        lado = 2 * morfologia + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
        morfologica = cv2.morphologyEx(
            morfologica, cv2.MORPH_CLOSE, kernel
        )
        morfologica = cv2.morphologyEx(
            morfologica, cv2.MORPH_OPEN, kernel
        )

    contornos, jerarquia = cv2.findContours(
        morfologica, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    if jerarquia is None:
        jerarquia = np.empty((1, 0, 4), dtype=np.int32)

    alto, ancho = morfologica.shape
    area_max = alto * ancho * max(0.0, min(area_max_pct, 100.0)) / 100.0
    indices_validos: list[int] = []
    descartados: list[int] = []

    for indice, datos in enumerate(jerarquia[0]):
        if datos[3] != -1:
            continue
        contorno = contornos[indice]
        area = cv2.contourArea(contorno)
        x, y, w, h = cv2.boundingRect(contorno)
        toca_borde = (
            x <= margen_borde
            or y <= margen_borde
            or x + w >= ancho - margen_borde
            or y + h >= alto - margen_borde
        )
        if area_min <= area <= area_max and not toca_borde:
            indices_validos.append(indice)
        else:
            descartados.append(indice)

    indices_validos.sort(
        key=lambda indice: cv2.contourArea(contornos[indice]), reverse=True
    )

    mascara = np.zeros_like(morfologica)
    vista_contornos = cv2.cvtColor(morfologica, cv2.COLOR_GRAY2BGR)
    for indice in descartados:
        cv2.drawContours(vista_contornos, contornos, indice, (80, 80, 80), 1)
    for indice in indices_validos:
        cv2.drawContours(vista_contornos, contornos, indice, (0, 255, 255), 2)
        cv2.drawContours(mascara, contornos, indice, 255, cv2.FILLED)
        for hijo, datos in enumerate(jerarquia[0]):
            if datos[3] == indice:
                cv2.drawContours(mascara, contornos, hijo, 0, cv2.FILLED)

    return {
        "gris": gris,
        "binaria": binaria,
        "morfologica": morfologica,
        "mascara": mascara,
        "vista_contornos": vista_contornos,
        "contornos": contornos,
        "jerarquia": jerarquia,
        "indices_validos": indices_validos,
        "descartados": descartados,
    }


def panel(titulo: str, imagen: np.ndarray, ancho=400, alto=285) -> np.ndarray:
    """Escala una imagen sin deformarla y agrega un titulo."""
    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    factor = min(ancho / imagen.shape[1], (alto - 34) / imagen.shape[0])
    nuevo_ancho = max(1, round(imagen.shape[1] * factor))
    nuevo_alto = max(1, round(imagen.shape[0] * factor))
    reducida = cv2.resize(
        imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA
    )
    lienzo = np.full((alto, ancho, 3), 238, dtype=np.uint8)
    x = (ancho - nuevo_ancho) // 2
    y = 34 + (alto - 34 - nuevo_alto) // 2
    lienzo[y:y + nuevo_alto, x:x + nuevo_ancho] = reducida
    cv2.putText(
        lienzo, titulo, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (25, 25, 25), 2, cv2.LINE_AA,
    )
    return lienzo


def crear_tablero(etapas: dict[str, np.ndarray]) -> np.ndarray:
    """Organiza las etapas en dos columnas."""
    vistas = [panel(titulo, imagen) for titulo, imagen in etapas.items()]
    if len(vistas) % 2:
        vistas.append(np.full((285, 400, 3), 238, dtype=np.uint8))
    filas = [
        np.hstack(vistas[indice:indice + 2])
        for indice in range(0, len(vistas), 2)
    ]
    return np.vstack(filas)
