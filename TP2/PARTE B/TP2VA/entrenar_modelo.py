"""Entrena un arbol de decision usando dataset.csv."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Rutas y categorias
CARPETA_PROYECTO = Path(__file__).resolve().parent
ARCHIVO_DATASET = CARPETA_PROYECTO / "dataset.csv"
ARCHIVO_MODELO = CARPETA_PROYECTO / "modelo.joblib"
CATEGORIAS = {
    0: "Circulos",
    1: "Cuadrados",
    2: "Rectangulos",
    3: "Triangulos",
}

# Dataset y entrenamiento
PRIMER_INDICE_HU = 1
CANTIDAD_INVARIANTES_HU = 7
NUMERO_PRIMERA_FILA_DATOS = 2
COLUMNAS_HU = [
    f"hu_{indice}"
    for indice in range(
        PRIMER_INDICE_HU,
        PRIMER_INDICE_HU + CANTIDAD_INVARIANTES_HU,
    )
]
COLUMNA_ETIQUETA = "etiqueta"
PROPORCION_PRUEBA = 0.10
SEMILLA_ALEATORIA = 42
PROFUNDIDAD_MAXIMA = None
MINIMO_CATEGORIAS = 2
MINIMO_MUESTRAS_POR_CATEGORIA = 2


def cargar_dataset() -> tuple[np.ndarray, np.ndarray]:
    if not ARCHIVO_DATASET.is_file():
        raise ValueError("No existe dataset.csv. Ejecuta primero generar_dataset.py.")

    muestras: list[list[float]] = []
    etiquetas: list[int] = []
    with ARCHIVO_DATASET.open("r", newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        columnas_requeridas = set(COLUMNAS_HU + [COLUMNA_ETIQUETA])
        if lector.fieldnames is None or not columnas_requeridas.issubset(lector.fieldnames):
            raise ValueError("El CSV no contiene siete Hu y una etiqueta.")

        for numero_fila, fila in enumerate(
            lector, start=NUMERO_PRIMERA_FILA_DATOS
        ):
            try:
                muestra = [float(fila[columna]) for columna in COLUMNAS_HU]
                etiqueta = int(fila[COLUMNA_ETIQUETA])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Fila {numero_fila} invalida: {error}") from error
            if etiqueta not in CATEGORIAS:
                raise ValueError(f"Fila {numero_fila}: etiqueta {etiqueta} desconocida.")
            if not np.isfinite(muestra).all():
                raise ValueError(f"Fila {numero_fila}: contiene valores no finitos.")
            muestras.append(muestra)
            etiquetas.append(etiqueta)

    return (
        np.asarray(muestras, dtype=np.float64),
        np.asarray(etiquetas, dtype=np.int32),
    )


def validar_dataset(etiquetas: np.ndarray):
    cantidades = Counter(int(etiqueta) for etiqueta in etiquetas)
    if len(cantidades) < MINIMO_CATEGORIAS:
        raise ValueError("Se necesitan muestras de al menos dos categorias.")
    insuficientes = [
        CATEGORIAS[etiqueta]
        for etiqueta, cantidad in cantidades.items()
        if cantidad < MINIMO_MUESTRAS_POR_CATEGORIA
    ]
    if insuficientes:
        raise ValueError(
            "Se necesitan al menos dos muestras de: " + ", ".join(insuficientes)
        )


def main():
    try:
        muestras, etiquetas = cargar_dataset()
        validar_dataset(etiquetas)
    except ValueError as error:
        raise SystemExit(f"ERROR: {error}") from error

    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(
        muestras,
        etiquetas,
        test_size=PROPORCION_PRUEBA,
        random_state=SEMILLA_ALEATORIA,
        stratify=etiquetas,
    )

    modelo = DecisionTreeClassifier(
        max_depth=PROFUNDIDAD_MAXIMA,
        random_state=SEMILLA_ALEATORIA,
    )
    modelo.fit(x_entrenamiento, y_entrenamiento)
    predicciones = modelo.predict(x_prueba)

    artefacto = {
        "modelo": modelo,
        "categorias": CATEGORIAS,
        "columnas_hu": COLUMNAS_HU,
    }
    dump(artefacto, ARCHIVO_MODELO)

    precision = accuracy_score(y_prueba, predicciones)
    clases_presentes = sorted(set(int(valor) for valor in etiquetas))
    matriz = confusion_matrix(y_prueba, predicciones, labels=clases_presentes)
    print(f"Muestras de entrenamiento: {len(y_entrenamiento)}")
    print(f"Muestras de prueba: {len(y_prueba)}")
    print(f"Precision: {precision:.2%}")
    print("Matriz de confusion:")
    print(matriz)
    print(f"Modelo guardado: {ARCHIVO_MODELO}")


if __name__ == "__main__":
    main()
