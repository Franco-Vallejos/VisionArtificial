"""Clasifica un video de ``video/`` en loop como si fuera la webcam.

Uso:
    python clasificador_videos.py nombre_del_video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

from clasificador import cargar_modelo, clasificar_frame
from config import BASE_DIR, MODEL_FILE, Controles
from vision import crear_tablero


VIDEO_DIR = BASE_DIR / "video"
EXTENSIONES_VIDEO = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
VENTANA_CONTROLES = "Controles - clasificador de video"
VENTANA_PROCESO = "Clasificacion ML - video"


def parsear_argumentos() -> Path:
    parser = argparse.ArgumentParser(
        description="Clasifica en loop un video ubicado dentro de video/."
    )
    parser.add_argument("video", help="nombre del archivo ubicado dentro de video/")
    argumentos = parser.parse_args()

    nombre = Path(argumentos.video)
    if nombre.name != argumentos.video or nombre.suffix.lower() not in EXTENSIONES_VIDEO:
        extensiones = ", ".join(sorted(EXTENSIONES_VIDEO))
        parser.error(
            f"Indica solo el nombre del archivo dentro de video/ ({extensiones})."
        )

    ruta = VIDEO_DIR / nombre.name
    if not ruta.is_file():
        parser.error(f"No existe el video: {ruta}")
    return ruta


def abrir_video(ruta: Path) -> cv2.VideoCapture:
    captura = cv2.VideoCapture(str(ruta))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta}")
    return captura


def main() -> None:
    try:
        ruta_video = parsear_argumentos()
        captura = abrir_video(ruta_video)
        modelo, etiquetas = cargar_modelo()
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}")
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

    fps = captura.get(cv2.CAP_PROP_FPS)
    espera_ms = max(1, round(1000 / fps)) if fps > 0 else 33
    pausado = False

    print(f"Modelo cargado: {MODEL_FILE}")
    print(f"Video en loop: {ruta_video.name}")
    print("ESPACIO pausa/reanuda. q o ESC finaliza.")

    ultimo_tablero = None
    while True:
        if not pausado:
            ok, frame = captura.read()
            if not ok:
                captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = captura.read()
                if not ok:
                    print("ERROR: no se pudo reiniciar el video.")
                    break

            valores = controles.leer()
            controles.guardar_si_cambiaron(valores)
            etapas = clasificar_frame(frame, modelo, etiquetas, valores)
            ultimo_tablero = crear_tablero(etapas)

        if ultimo_tablero is not None:
            cv2.imshow(VENTANA_PROCESO, ultimo_tablero)

        tecla = cv2.waitKey(espera_ms if not pausado else 30) & 0xFF
        if tecla in (27, ord("q")):
            break
        if tecla == ord(" "):
            pausado = not pausado

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
