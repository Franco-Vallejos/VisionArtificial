"""Clasifica formas en tiempo real usando la camara y modelo.joblib."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from joblib import load


# Rutas y modelo
CARPETA_PROYECTO = Path(__file__).resolve().parent
ARCHIVO_MODELO = CARPETA_PROYECTO / "modelo.joblib"
PRIMER_INDICE_HU = 1
CANTIDAD_INVARIANTES_HU = 7
COLUMNAS_HU = [
    f"hu_{indice}"
    for indice in range(
        PRIMER_INDICE_HU,
        PRIMER_INDICE_HU + CANTIDAD_INVARIANTES_HU,
    )
]

# Camara e interfaz
INDICE_CAMARA = 0
VENTANA_FILTROS = "Filtros aplicados"
VENTANA_CONTROLES = "Panel de controles"
NOMBRE_CONTROL_UMBRAL = "Umbral (0=Otsu)"
NOMBRE_CONTROL_MORFOLOGIA = "Morfologia"
TECLA_ESCAPE = 27
TECLA_SALIR = ord("q")
COLOR_RECONOCIDO = (0, 255, 0)
COLOR_DESCONOCIDO = (0, 0, 255)
GROSOR_CONTORNO = 2
GROSOR_TEXTO = 2
ESCALA_TEXTO = 0.65
DESPLAZAMIENTO_TEXTO = 8
ALTURA_MINIMA_TEXTO = 24
FUENTE_TEXTO = cv2.FONT_HERSHEY_SIMPLEX
TODOS_LOS_CONTORNOS = -1
ESPERA_TECLA_MILISEGUNDOS = 1
MASCARA_CODIGO_TECLA = 0xFF
DIVISOR_GRILLA = 2
ANCHO_VENTANA_CONTROLES = 430
ALTO_CONTENIDO_CONTROLES = 1
ALTO_BANDA_TITULO_FILTRO = 28
POSICION_TITULO_FILTRO = (8, 20)
ESCALA_TITULO_FILTRO = 0.55
GROSOR_TITULO_FILTRO = 2
COLOR_BANDA_FILTRO = (20, 20, 20)
COLOR_TITULO_FILTRO = (255, 255, 255)

# Procesamiento de imagen: debe coincidir con generar_dataset.py
ITERACIONES_MORFOLOGIA = 1
VALOR_BINARIO_MAXIMO = 255
AREA_MINIMA_RELATIVA = 0.005
AREA_MAXIMA_RELATIVA = 0.95
MARGEN_BORDE = 2
EPSILON_HU = 1e-30
UMBRAL_AUTOMATICO = 0
UMBRAL_INICIAL = UMBRAL_AUTOMATICO
UMBRAL_MAXIMO = 255
MORFOLOGIA_INICIAL = 1
MORFOLOGIA_MAXIMA = 10
FACTOR_DIAMETRO_KERNEL = 2
AJUSTE_KERNEL_IMPAR = 1


def toca_borde(contorno: np.ndarray, ancho: int, alto: int) -> bool:
    x, y, w, h = cv2.boundingRect(contorno)
    return (
        x <= MARGEN_BORDE
        or y <= MARGEN_BORDE
        or x + w >= ancho - MARGEN_BORDE
        or y + h >= alto - MARGEN_BORDE
    )


def detectar_contornos(
    imagen: np.ndarray,
    umbral: int,
    morfologia: int,
) -> tuple[list[np.ndarray], dict[str, np.ndarray]]:
    """Detecta contornos y conserva copias intermedias solo para mostrarlas."""
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    alto, ancho = gris.shape
    area_imagen = alto * ancho
    area_minima = area_imagen * AREA_MINIMA_RELATIVA
    area_maxima = area_imagen * AREA_MAXIMA_RELATIVA
    validos: list[np.ndarray] = []

    # SOLO VISUALIZACION: este diccionario no participa en la deteccion ni
    # se entrega al modelo. Guarda imagenes para ilustrar las fases.
    etapas: dict[str, np.ndarray] = {
        "Imagen original": imagen.copy(),
        "Escala de grises": gris,
    }

    for tipo in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        tipo_umbral = tipo
        if umbral == UMBRAL_AUTOMATICO:
            tipo_umbral |= cv2.THRESH_OTSU
        _, binaria = cv2.threshold(
            gris,
            umbral,
            VALOR_BINARIO_MAXIMO,
            tipo_umbral,
        )

        # SOLO VISUALIZACION: la copia permite mostrar el umbral antes de
        # aplicar la morfologia. Los contornos se calculan con `binaria`.
        umbralizada = binaria.copy()
        if morfologia > 0:
            lado_kernel = (
                FACTOR_DIAMETRO_KERNEL * morfologia + AJUSTE_KERNEL_IMPAR
            )
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (lado_kernel, lado_kernel),
            )
            binaria = cv2.morphologyEx(
                binaria,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=ITERACIONES_MORFOLOGIA,
            )

        # SOLO VISUALIZACION
        if tipo == cv2.THRESH_BINARY_INV:
            etapas["Umbral invertido"] = umbralizada
            etapas["Morfologia"] = binaria
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

    return validos, etapas


def preparar_etapa(
    imagen: np.ndarray,
    titulo: str,
    ancho: int,
    alto: int,
) -> np.ndarray:
    """SOLO VISUALIZACION: prepara una fase para la ventana de filtros."""
    if imagen.ndim == 2:
        imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        imagen_color = imagen.copy()
    vista = cv2.resize(imagen_color, (ancho, alto), interpolation=cv2.INTER_AREA)
    cv2.rectangle(
        vista,
        (0, 0),
        (ancho, ALTO_BANDA_TITULO_FILTRO),
        COLOR_BANDA_FILTRO,
        cv2.FILLED,
    )
    cv2.putText(
        vista,
        titulo,
        POSICION_TITULO_FILTRO,
        FUENTE_TEXTO,
        ESCALA_TITULO_FILTRO,
        COLOR_TITULO_FILTRO,
        GROSOR_TITULO_FILTRO,
        cv2.LINE_AA,
    )
    return vista


def construir_vista_filtros(
    etapas: dict[str, np.ndarray],
    ancho: int,
    alto: int,
) -> np.ndarray:
    """SOLO VISUALIZACION: arma la cuadricula; no modifica la prediccion."""
    ancho_celda = ancho // DIVISOR_GRILLA
    alto_celda = alto // DIVISOR_GRILLA
    vistas = [
        preparar_etapa(imagen, titulo, ancho_celda, alto_celda)
        for titulo, imagen in etapas.items()
    ]
    fila_superior = np.hstack(vistas[:DIVISOR_GRILLA])
    fila_inferior = np.hstack(vistas[DIVISOR_GRILLA:])
    return np.vstack((fila_superior, fila_inferior))


def invariantes_hu(contorno: np.ndarray) -> np.ndarray:
    """Calcula los siete invariantes de Hu de un contorno."""
    momentos = cv2.moments(contorno)
    return cv2.HuMoments(momentos).flatten().astype(np.float64)

def cargar_modelo():
    if not ARCHIVO_MODELO.is_file():
        raise ValueError("No existe modelo.joblib. Ejecuta primero entrenar_modelo.py.")
    artefacto = load(ARCHIVO_MODELO)
    if artefacto.get("columnas_hu") != COLUMNAS_HU:
        raise ValueError("El modelo no utiliza los siete invariantes de Hu esperados.")
    return artefacto["modelo"], artefacto["categorias"]


def main():
    try:
        modelo, categorias = cargar_modelo()
    except (OSError, KeyError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error

    camara = cv2.VideoCapture(INDICE_CAMARA)
    if not camara.isOpened():
        raise SystemExit("ERROR: no se pudo abrir la camara.")

    cv2.namedWindow(VENTANA_FILTROS, cv2.WINDOW_NORMAL)
    cv2.namedWindow(VENTANA_CONTROLES, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        VENTANA_CONTROLES,
        ANCHO_VENTANA_CONTROLES,
        ALTO_CONTENIDO_CONTROLES,
    )
    cv2.createTrackbar(
        NOMBRE_CONTROL_UMBRAL,
        VENTANA_CONTROLES,
        UMBRAL_INICIAL,
        UMBRAL_MAXIMO,
        lambda _valor: None,
    )

    # SOLO VISUALIZACION: OpenCV necesita una imagen minima para mantener
    # visible la ventana que contiene los trackbars.
    contenido_controles = np.zeros(
        (
            ALTO_CONTENIDO_CONTROLES,
            ANCHO_VENTANA_CONTROLES,
            3,
        ),
        dtype=np.uint8,
    )
    cv2.imshow(VENTANA_CONTROLES, contenido_controles)
    cv2.createTrackbar(
        NOMBRE_CONTROL_MORFOLOGIA,
        VENTANA_CONTROLES,
        MORFOLOGIA_INICIAL,
        MORFOLOGIA_MAXIMA,
        lambda _valor: None,
    )
    print("Clasificador iniciado. Presiona q o ESC para salir.")
    while True:
        lectura_correcta, frame = camara.read()
        if not lectura_correcta:
            print("ERROR: no se pudo leer la camara.")
            break

        umbral = cv2.getTrackbarPos(NOMBRE_CONTROL_UMBRAL, VENTANA_CONTROLES)
        morfologia = cv2.getTrackbarPos(
            NOMBRE_CONTROL_MORFOLOGIA, VENTANA_CONTROLES
        )
        contornos, etapas = detectar_contornos(frame, umbral, morfologia)

        if contornos:
            descriptores = np.asarray(
                [invariantes_hu(contorno) for contorno in contornos],
                dtype=np.float64,
            )
            etiquetas_modelo = modelo.predict(descriptores)
            for contorno, etiqueta_modelo in zip(contornos, etiquetas_modelo):
                etiqueta = int(etiqueta_modelo)
                reconocido = etiqueta in categorias
                nombre = categorias[etiqueta] if reconocido else "Desconocido"
                color = COLOR_RECONOCIDO if reconocido else COLOR_DESCONOCIDO

                # SOLO VISUALIZACION: dibujar el contorno y escribir la etiqueta
                # ocurre despues de predecir; no cambia los Hu ni la categoria.
                x, y, _, _ = cv2.boundingRect(contorno)
                cv2.drawContours(
                    frame,
                    [contorno],
                    TODOS_LOS_CONTORNOS,
                    color,
                    GROSOR_CONTORNO,
                )
                cv2.putText(
                    frame,
                    nombre,
                    (x, max(ALTURA_MINIMA_TEXTO, y - DESPLAZAMIENTO_TEXTO)),
                    FUENTE_TEXTO,
                    ESCALA_TEXTO,
                    color,
                    GROSOR_TEXTO,
                    cv2.LINE_AA,
                )

        # SOLO VISUALIZACION: estas operaciones construyen y muestran las
        # ventanas. El procedimiento termina en `modelo.predict` de arriba.
        vista_filtros = construir_vista_filtros(
            etapas,
            frame.shape[1],
            frame.shape[0],
        )
        cv2.imshow(VENTANA_FILTROS, vista_filtros)
        tecla = cv2.waitKey(ESPERA_TECLA_MILISEGUNDOS) & MASCARA_CODIGO_TECLA
        if tecla in (TECLA_ESCAPE, TECLA_SALIR):
            break

    camara.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
