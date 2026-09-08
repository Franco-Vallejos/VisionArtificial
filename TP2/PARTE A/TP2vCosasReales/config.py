"""Argumentos, video y persistencia de controles del detector por video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "video"
CFG_DIR = BASE_DIR / ".cfg"
CFG_FILE = CFG_DIR / "parametros.json"

LIMITES = {
    "umbral": 255,
    "morfologia": 20,
    "area_min_x100": 100,
    "distancia_x100": 200,
    "umbral_refs": 255,
}
ETIQUETAS = {
    "umbral": "Umbral binario",
    "morfologia": "Morfologia",
    "area_min_x100": "Area min x100",
    "distancia_x100": "Distancia x100",
    "umbral_refs": "Umbral refs",
}
PREDETERMINADOS = {
    "umbral": 127,
    "morfologia": 2,
    "area_min_x100": 5,
    "distancia_x100": 15,
    "umbral_refs": 127,
}


def parsear_argumentos() -> Path:
    """Recibe sólo el nombre de un MP4 ubicado dentro de ``video/``."""
    parser = argparse.ArgumentParser(
        description="Usa un MP4 en loop como entrada del detector."
    )
    parser.add_argument("video", help="nombre del archivo .mp4 dentro de video/")
    argumentos = parser.parse_args()

    nombre = Path(argumentos.video)
    if nombre.name != argumentos.video or nombre.suffix.lower() != ".mp4":
        parser.error("Indicá sólo el nombre de un archivo .mp4 dentro de video/")

    ruta = VIDEO_DIR / nombre.name
    if not ruta.is_file():
        parser.error(f"No existe el video: {ruta}")
    return ruta


def abrir_video(ruta: Path) -> cv2.VideoCapture:
    """Abre el MP4 indicado y falla con un mensaje claro si no es legible."""
    captura = cv2.VideoCapture(str(ruta))
    if not captura.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta}")
    return captura


def cargar_parametros() -> dict[str, int]:
    """Carga valores válidos; crea la configuración inicial si aún no existe."""
    parametros = PREDETERMINADOS.copy()
    if CFG_FILE.is_file():
        try:
            datos = json.loads(CFG_FILE.read_text(encoding="utf-8"))
            for nombre, limite in LIMITES.items():
                if isinstance(datos.get(nombre), int):
                    parametros[nombre] = max(0, min(datos[nombre], limite))
        except (OSError, json.JSONDecodeError):
            print("Advertencia: no se pudo leer la configuracion; se usan valores iniciales.")
    guardar_parametros(parametros)
    return parametros


def guardar_parametros(parametros: dict[str, int]) -> None:
    """Guarda los controles en .cfg/parametros.json."""
    CFG_DIR.mkdir(exist_ok=True)
    datos = {nombre: int(parametros[nombre]) for nombre in LIMITES}
    CFG_FILE.write_text(json.dumps(datos, indent=2) + "\n", encoding="utf-8")


class Controles:
    """Barras de OpenCV que persisten cada ajuste inmediatamente."""

    def __init__(self, ventana: str):
        self.ventana = ventana
        self.parametros = cargar_parametros()
        self._listo = False
        for nombre, limite in LIMITES.items():
            cv2.createTrackbar(
                ETIQUETAS[nombre], ventana, self.parametros[nombre], limite,
                self._al_cambiar,
            )
        self._listo = True

    def _al_cambiar(self, _valor: int) -> None:
        # OpenCV invoca este callback cada vez que el usuario mueve una barra.
        if not self._listo:
            return
        self.parametros = self.leer()
        guardar_parametros(self.parametros)

    def leer(self) -> dict[str, int]:
        return {
            nombre: cv2.getTrackbarPos(ETIQUETAS[nombre], self.ventana)
            for nombre in LIMITES
        }
