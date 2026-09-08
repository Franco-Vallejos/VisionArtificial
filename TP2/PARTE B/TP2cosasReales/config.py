"""Configuracion compartida por las tres aplicaciones de la Parte B."""

from __future__ import annotations

import json
from pathlib import Path

import cv2


BASE_DIR = Path(__file__).resolve().parent
DATOS_DIR = BASE_DIR / "datos"
MODELOS_DIR = BASE_DIR / "modelos"
CFG_DIR = BASE_DIR / ".cfg"

DATASET_FILE = DATOS_DIR / "dataset_hu.csv"
MODEL_FILE = MODELOS_DIR / "clasificador_formas.joblib"
TREE_FILE = MODELOS_DIR / "arbol_decision.txt"
METRICS_FILE = MODELOS_DIR / "metricas.json"
CFG_FILE = CFG_DIR / "parametros.json"

# El numero es la etiqueta que se guarda en el dataset y predice el modelo.
ETIQUETAS = {
    1: "CD",
    2: "Flor",
    3: "Huevo",
    4: "Taza",
}

COLORES = {
    1: (50, 200, 50),
    2: (255, 150, 30),
    3: (30, 180, 255),
    4: (200, 30, 200),
}
COLOR_DESCONOCIDO = (0, 0, 220)

LIMITES = {
    "umbral": 255,
    "invertir": 1,
    "morfologia": 20,
    "area_min_x100": 100,
    "area_max_pct": 100,
    "confianza_x100": 100,
}

NOMBRES_CONTROLES = {
    "umbral": "Umbral binario",
    "invertir": "Invertir blanco/negro",
    "morfologia": "Morfologia",
    "area_min_x100": "Area min x100",
    "area_max_pct": "Area max %",
    "confianza_x100": "Confianza min %",
}

PREDETERMINADOS = {
    "umbral": 127,
    "invertir": 0,
    "morfologia": 2,
    "area_min_x100": 5,
    "area_max_pct": 80,
    "confianza_x100": 0,
}


def cargar_parametros() -> dict[str, int]:
    """Carga controles persistidos y completa valores ausentes."""
    parametros = PREDETERMINADOS.copy()
    if CFG_FILE.is_file():
        try:
            datos = json.loads(CFG_FILE.read_text(encoding="utf-8"))
            for nombre, limite in LIMITES.items():
                valor = datos.get(nombre)
                if isinstance(valor, int):
                    parametros[nombre] = max(0, min(valor, limite))
        except (OSError, json.JSONDecodeError):
            print("Advertencia: no se pudo leer la configuracion guardada.")
    return parametros


def guardar_parametros(parametros: dict[str, int]) -> None:
    """Guarda todos los controles para compartirlos entre aplicaciones."""
    CFG_DIR.mkdir(exist_ok=True)
    datos = {
        nombre: int(parametros.get(nombre, PREDETERMINADOS[nombre]))
        for nombre in LIMITES
    }
    CFG_FILE.write_text(
        json.dumps(datos, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class Controles:
    """Barras de OpenCV que comparten y persisten sus valores."""

    def __init__(self, ventana: str, nombres: list[str]):
        self.ventana = ventana
        self.nombres = nombres
        self.parametros = cargar_parametros()
        for nombre in nombres:
            cv2.createTrackbar(
                NOMBRES_CONTROLES[nombre],
                ventana,
                self.parametros[nombre],
                LIMITES[nombre],
                lambda _valor: None,
            )
        self._ultimos = self.leer()

    def leer(self) -> dict[str, int]:
        valores = self.parametros.copy()
        for nombre in self.nombres:
            valores[nombre] = cv2.getTrackbarPos(
                NOMBRES_CONTROLES[nombre], self.ventana
            )
        return valores

    def guardar_si_cambiaron(self, valores: dict[str, int]) -> None:
        actuales = {nombre: valores[nombre] for nombre in self.nombres}
        if actuales != self._ultimos:
            guardar_parametros(valores)
            self.parametros = valores.copy()
            self._ultimos = actuales
