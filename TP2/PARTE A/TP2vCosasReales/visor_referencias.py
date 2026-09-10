"""Visualiza paso a paso el procesamiento de las imagenes de referencia.

Uso:
    python visor_referencias.py

Controles:
    Selector visual elige una sola imagen de ``refs/`` mediante miniaturas.
    Umbral refs     ajusta la binarizacion.
    Morfologia      ajusta el radio del elemento estructural; 0 la desactiva.
    Area min x100   descarta contornos externos pequenos.

Teclas:
    n, d o flecha derecha   referencia siguiente
    p, a o flecha izquierda referencia anterior
    q o ESC                 salir

El detector principal actualmente carga las referencias sin morfologia. Por eso
este visor comienza con Morfologia=0. El control permite estudiar si un cierre y
una apertura mejorarian una referencia antes de decidir si conviene incorporar
esas operaciones al detector.
"""

from __future__ import annotations

import base64
from pathlib import Path
import sys
import tkinter as tk

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
REFS_DIR = BASE_DIR / "refs"
EXTENSIONES = {".png", ".jpg", ".jpeg", ".bmp"}

VENTANA_CONTROLES = "Controles de referencias"
VENTANA_PROCESO = "Proceso de las referencias"
VENTANA_SELECTOR = "Seleccion de referencia"

ANCHO_PANEL = 400
ALTO_PANEL = 285
GROSOR_CONTORNOS_EXTERNOS = 10
GROSOR_CONTORNO_ELEGIDO = 12
GROSOR_CONTORNOS_INTERNOS = 10


def listar_referencias() -> list[Path]:
    """Devuelve las imagenes de referencia en un orden estable."""
    if not REFS_DIR.is_dir():
        return []
    return [
        ruta
        for ruta in sorted(REFS_DIR.iterdir())
        if ruta.suffix.lower() in EXTENSIONES
    ]


def procesar_referencia(
    imagen: np.ndarray,
    umbral: int,
    morfologia: int,
    area_min: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Genera todas las etapas y reproduce la eleccion del contorno principal."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY_INV)

    cierre = binaria.copy()
    apertura = binaria.copy()
    if morfologia > 0:
        lado = 2 * morfologia + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lado, lado))
        cierre = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
        apertura = cv2.morphologyEx(cierre, cv2.MORPH_OPEN, kernel)

    contornos, jerarquia = cv2.findContours(
        apertura, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    vista_contornos = cv2.cvtColor(apertura, cv2.COLOR_GRAY2BGR)
    vista_elegido = imagen.copy()
    mascara_elegida = np.zeros_like(apertura)

    externos: list[int] = []
    if jerarquia is not None:
        externos = [
            indice
            for indice, datos in enumerate(jerarquia[0])
            if datos[3] == -1 and cv2.contourArea(contornos[indice]) >= area_min
        ]

    for indice in externos:
        cv2.drawContours(
            vista_contornos,
            contornos,
            indice,
            (0, 255, 255),
            GROSOR_CONTORNOS_EXTERNOS,
        )

    indice_elegido = None
    proporcion_agujero = 0.0
    area_elegida = 0.0
    toca_borde = False

    if externos:
        indice_elegido = max(
            externos, key=lambda indice: cv2.contourArea(contornos[indice])
        )
        contorno = contornos[indice_elegido]
        area_elegida = max(cv2.contourArea(contorno), 1.0)
        x, y, ancho, alto = cv2.boundingRect(contorno)
        toca_borde = (
            x <= 0
            or y <= 0
            or x + ancho >= imagen.shape[1]
            or y + alto >= imagen.shape[0]
        )

        hijos = [
            indice
            for indice, datos in enumerate(jerarquia[0])
            if datos[3] == indice_elegido
        ]
        area_agujero = max(
            (cv2.contourArea(contornos[indice]) for indice in hijos),
            default=0.0,
        )
        proporcion_agujero = area_agujero / area_elegida

        cv2.drawContours(
            vista_elegido,
            contornos,
            indice_elegido,
            (0, 255, 0),
            GROSOR_CONTORNO_ELEGIDO,
        )
        cv2.rectangle(
            vista_elegido, (x, y), (x + ancho, y + alto), (255, 180, 0), 2
        )
        cv2.drawContours(
            mascara_elegida, contornos, indice_elegido, 255, cv2.FILLED
        )
        for hijo in hijos:
            cv2.drawContours(
                vista_elegido,
                contornos,
                hijo,
                (255, 0, 255),
                GROSOR_CONTORNOS_INTERNOS,
            )
            cv2.drawContours(
                mascara_elegida, contornos, hijo, 0, cv2.FILLED
            )

    etapas = {
        "0. Original": imagen,
        "1. Monocromatica": gris,
        f"2. Threshold ({umbral})": binaria,
        "3. Cierre morfologico": cierre,
        "4. Apertura morfologica": apertura,
        "5. Contornos externos": vista_contornos,
        "6. Contorno seleccionado": vista_elegido,
        "7. Mascara seleccionada": mascara_elegida,
    }
    diagnostico = {
        "cantidad_contornos": len(contornos),
        "cantidad_externos": len(externos),
        "indice_elegido": indice_elegido,
        "area_elegida": area_elegida,
        "proporcion_agujero": proporcion_agujero,
        "toca_borde": toca_borde,
    }
    return etapas, diagnostico


def preparar_panel(titulo: str, imagen: np.ndarray) -> np.ndarray:
    """Escala una etapa sin deformarla y agrega su titulo."""
    if imagen.ndim == 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)

    alto_disponible = ALTO_PANEL - 34
    factor = min(
        ANCHO_PANEL / imagen.shape[1],
        alto_disponible / imagen.shape[0],
    )
    nuevo_ancho = max(1, round(imagen.shape[1] * factor))
    nuevo_alto = max(1, round(imagen.shape[0] * factor))
    escalada = cv2.resize(
        imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA
    )

    panel = np.full((ALTO_PANEL, ANCHO_PANEL, 3), 238, dtype=np.uint8)
    x = (ANCHO_PANEL - nuevo_ancho) // 2
    y = 34 + (alto_disponible - nuevo_alto) // 2
    panel[y:y + nuevo_alto, x:x + nuevo_ancho] = escalada
    cv2.putText(
        panel,
        titulo,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    return panel


def crear_tablero(
    nombre: str,
    etapas: dict[str, np.ndarray],
    diagnostico: dict[str, object],
) -> np.ndarray:
    """Organiza ocho etapas y agrega una franja con el diagnostico."""
    paneles = [
        preparar_panel(titulo, imagen) for titulo, imagen in etapas.items()
    ]
    filas = [
        np.hstack(paneles[indice:indice + 2])
        for indice in range(0, len(paneles), 2)
    ]
    tablero = np.vstack(filas)

    color_estado = (
        (0, 0, 210) if diagnostico["toca_borde"] else (20, 130, 20)
    )
    borde = "SI" if diagnostico["toca_borde"] else "no"
    texto = (
        f"{nombre} | contornos={diagnostico['cantidad_contornos']} | "
        f"externos validos={diagnostico['cantidad_externos']} | "
        f"area elegida={diagnostico['area_elegida']:.0f} | "
        f"agujero/area={diagnostico['proporcion_agujero']:.4f} | "
        f"toca borde={borde}"
    )

    franja = np.full((46, tablero.shape[1], 3), 245, dtype=np.uint8)
    cv2.putText(
        franja,
        texto,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color_estado,
        2,
        cv2.LINE_AA,
    )
    return np.vstack([franja, tablero])


def sin_accion(_valor: int) -> None:
    """Callback requerido por las barras de OpenCV."""


def crear_miniatura(ruta: Path, ancho=180, alto=135) -> tk.PhotoImage:
    """Crea una miniatura compatible con Tk sin depender de Pillow."""
    imagen = cv2.imread(str(ruta))
    if imagen is None:
        lienzo = np.full((alto, ancho, 3), 220, dtype=np.uint8)
        cv2.putText(
            lienzo, "No disponible", (12, alto // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 180), 1, cv2.LINE_AA,
        )
    else:
        factor = min(ancho / imagen.shape[1], alto / imagen.shape[0])
        nuevo_ancho = max(1, round(imagen.shape[1] * factor))
        nuevo_alto = max(1, round(imagen.shape[0] * factor))
        reducida = cv2.resize(
            imagen, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_AREA
        )
        lienzo = np.full((alto, ancho, 3), 235, dtype=np.uint8)
        x = (ancho - nuevo_ancho) // 2
        y = (alto - nuevo_alto) // 2
        lienzo[y:y + nuevo_alto, x:x + nuevo_ancho] = reducida

    correcto, png = cv2.imencode(".png", lienzo)
    if not correcto:
        raise RuntimeError(f"No se pudo generar la miniatura de {ruta}")
    datos = base64.b64encode(png.tobytes()).decode("ascii")
    return tk.PhotoImage(data=datos)


def crear_selector_referencias(
    referencias: list[Path],
) -> tuple[tk.Tk, tk.IntVar, dict[str, bool], list[tk.PhotoImage]]:
    """Muestra todas las referencias y permite seleccionar exactamente una."""
    raiz = tk.Tk()
    raiz.title(VENTANA_SELECTOR)
    raiz.resizable(False, False)

    columnas = min(4, len(referencias))
    tk.Label(
        raiz,
        text="Elegir una imagen de referencia",
        font=("Segoe UI", 13, "bold"),
        padx=12,
        pady=10,
    ).grid(row=0, column=0, columnspan=columnas, sticky="w")

    indice_seleccionado = tk.IntVar(value=0)
    miniaturas: list[tk.PhotoImage] = []
    for indice, ruta in enumerate(referencias):
        miniatura = crear_miniatura(ruta)
        miniaturas.append(miniatura)
        tk.Radiobutton(
            raiz,
            text=ruta.stem,
            image=miniatura,
            compound="top",
            variable=indice_seleccionado,
            value=indice,
            indicatoron=True,
            font=("Segoe UI", 10, "bold"),
            padx=8,
            pady=6,
            relief=tk.GROOVE,
            borderwidth=1,
        ).grid(
            row=1 + indice // 4,
            column=indice % 4,
            padx=6,
            pady=6,
            sticky="nsew",
        )

    ultima_fila = 2 + (len(referencias) - 1) // 4
    tk.Label(
        raiz,
        text="Solo una referencia puede quedar seleccionada.",
        font=("Segoe UI", 9),
        padx=12,
        pady=8,
    ).grid(row=ultima_fila, column=0, columnspan=columnas, sticky="w")

    estado = {"cerrado": False}

    def cerrar() -> None:
        estado["cerrado"] = True
        raiz.destroy()

    def avanzar(_evento=None) -> None:
        indice_seleccionado.set(
            (indice_seleccionado.get() + 1) % len(referencias)
        )

    def retroceder(_evento=None) -> None:
        indice_seleccionado.set(
            (indice_seleccionado.get() - 1) % len(referencias)
        )

    raiz.protocol("WM_DELETE_WINDOW", cerrar)
    raiz.bind("<KeyPress-n>", avanzar)
    raiz.bind("<KeyPress-d>", avanzar)
    raiz.bind("<Right>", avanzar)
    raiz.bind("<KeyPress-p>", retroceder)
    raiz.bind("<KeyPress-a>", retroceder)
    raiz.bind("<Left>", retroceder)
    raiz.bind("<KeyPress-q>", lambda _evento: cerrar())
    raiz.bind("<Escape>", lambda _evento: cerrar())
    raiz.update_idletasks()
    return raiz, indice_seleccionado, estado, miniaturas


def main() -> None:
    referencias = listar_referencias()
    if not referencias:
        print(f"ERROR: no hay imagenes de referencia en {REFS_DIR}")
        sys.exit(1)

    raiz, indice_seleccionado, estado, miniaturas = crear_selector_referencias(
        referencias
    )

    cv2.namedWindow(VENTANA_CONTROLES, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_CONTROLES, 520, 210)
    cv2.createTrackbar("Umbral refs", VENTANA_CONTROLES, 127, 255, sin_accion)
    cv2.createTrackbar("Morfologia", VENTANA_CONTROLES, 0, 20, sin_accion)
    cv2.createTrackbar("Area min x100", VENTANA_CONTROLES, 1, 100, sin_accion)

    cv2.namedWindow(VENTANA_PROCESO, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(VENTANA_PROCESO, 1000, 900)

    print("Referencias:")
    for indice, ruta in enumerate(referencias):
        print(f"  {indice}: {ruta.name}")
    print("Selecciona una miniatura; tambien puedes usar n/p o las flechas.")

    while not estado["cerrado"]:
        try:
            raiz.update_idletasks()
            raiz.update()
        except tk.TclError:
            break

        indice = indice_seleccionado.get()
        umbral = cv2.getTrackbarPos("Umbral refs", VENTANA_CONTROLES)
        morfologia = cv2.getTrackbarPos("Morfologia", VENTANA_CONTROLES)
        area_min = max(
            1, cv2.getTrackbarPos("Area min x100", VENTANA_CONTROLES) * 100
        )

        ruta = referencias[indice]
        imagen = cv2.imread(str(ruta))
        if imagen is None:
            print(f"ERROR: no se pudo leer {ruta}")
            break

        etapas, diagnostico = procesar_referencia(
            imagen, umbral, morfologia, area_min
        )
        tablero = crear_tablero(ruta.name, etapas, diagnostico)
        cv2.imshow(VENTANA_PROCESO, tablero)

        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (27, ord("q")):
            break
        if tecla in (ord("n"), ord("d"), 83):
            indice_seleccionado.set((indice + 1) % len(referencias))
        elif tecla in (ord("p"), ord("a"), 81):
            indice_seleccionado.set((indice - 1) % len(referencias))

    if not estado["cerrado"]:
        raiz.destroy()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
