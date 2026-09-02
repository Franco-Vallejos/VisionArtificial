"""Ejecuta el detector existente usando un MP4 en loop como si fuera webcam.

Uso:
    python detector_video.py nombre_del_video.mp4
"""

import sys

import cv2

from config import Controles, abrir_video, parsear_argumentos
from detector_objetos_reales import cargar_referencias, crear_tablero, procesar_frame


def main():
    try:
        ruta_video = parsear_argumentos()
        captura = abrir_video(ruta_video)
    except RuntimeError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    ventana_config = "Controles"
    ventana_proceso = "Proceso de reconocimiento - video"
    cv2.namedWindow(ventana_config, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana_config, 470, 220)
    controles = Controles(ventana_config)

    cv2.namedWindow(ventana_proceso, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ventana_proceso, 900, 900)

    fps = captura.get(cv2.CAP_PROP_FPS)
    espera_ms = max(1, round(1000 / fps)) if fps > 0 else 33
    umbral_refs_anterior = None
    referencias = {}

    print(f"Video en loop: {ruta_video.name}")
    print("Presiona q o ESC para salir.")
    while True:
        ok, frame = captura.read()
        if not ok:
            # Fin del MP4: vuelve al primer cuadro y continúa como una webcam.
            captura.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = captura.read()
            if not ok:
                print("ERROR: no se pudo reiniciar el video.")
                break

        valores = controles.leer()
        if valores["umbral_refs"] != umbral_refs_anterior:
            referencias = cargar_referencias(valores["umbral_refs"])
            umbral_refs_anterior = valores["umbral_refs"]
            print(f"Referencias activas: {list(referencias)}")

        etapas = procesar_frame(
            frame,
            referencias,
            valores["umbral"],
            valores["morfologia"],
            valores["area_min_x100"] * 100,
            valores["distancia_x100"] / 100.0,
        )
        cv2.imshow(ventana_proceso, crear_tablero(etapas))

        tecla = cv2.waitKey(espera_ms) & 0xFF
        if tecla in (27, ord("q")):
            break

    captura.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
