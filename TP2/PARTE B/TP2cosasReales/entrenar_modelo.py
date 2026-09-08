"""Aplicacion 2: entrena y guarda un arbol de decision."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import sys

from joblib import dump
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from config import DATASET_FILE, ETIQUETAS, METRICS_FILE, MODEL_FILE, TREE_FILE
from vision import COLUMNAS_DATASET, COLUMNAS_HU


def parsear_argumentos() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena el clasificador del TP2.")
    parser.add_argument("--dataset", type=Path, default=DATASET_FILE)
    parser.add_argument("--modelo", type=Path, default=MODEL_FILE)
    parser.add_argument("--arbol", type=Path, default=TREE_FILE)
    parser.add_argument("--metricas", type=Path, default=METRICS_FILE)
    parser.add_argument("--test-size", type=float, default=0.10)
    parser.add_argument("--max-depth", type=int, default=None)
    return parser.parse_args()


def cargar_dataset(ruta: Path) -> tuple[np.ndarray, np.ndarray]:
    if not ruta.is_file():
        raise ValueError(
            f"No existe {ruta}. Primero ejecuta generar_dataset.py."
        )

    muestras: list[list[float]] = []
    etiquetas: list[int] = []
    with ruta.open("r", newline="", encoding="utf-8") as archivo:
        lector = csv.DictReader(archivo)
        if lector.fieldnames is None or not set(COLUMNAS_DATASET).issubset(lector.fieldnames):
            raise ValueError(
                "El CSV debe contener hu_1..hu_7 y la columna etiqueta."
            )
        for numero_fila, fila in enumerate(lector, start=2):
            try:
                muestra = [float(fila[columna]) for columna in COLUMNAS_HU]
                etiqueta = int(fila["etiqueta"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Fila {numero_fila} invalida: {error}") from error
            if not np.isfinite(muestra).all():
                raise ValueError(f"Fila {numero_fila} contiene valores no finitos.")
            if etiqueta not in ETIQUETAS:
                raise ValueError(f"Fila {numero_fila}: etiqueta desconocida {etiqueta}.")
            muestras.append(muestra)
            etiquetas.append(etiqueta)

    if not muestras:
        raise ValueError("El dataset esta vacio.")
    return np.asarray(muestras, dtype=np.float64), np.asarray(etiquetas, dtype=np.int32)


def dividir_dataset(X, y, test_size: float):
    clases = sorted(set(int(valor) for valor in y))
    cantidades = Counter(int(valor) for valor in y)
    if len(clases) < 2:
        raise ValueError("Se necesitan al menos dos categorias para clasificar.")
    if min(cantidades.values()) < 2:
        raise ValueError("Cada categoria necesita al menos dos muestras.")
    if not 0 < test_size < 1:
        raise ValueError("--test-size debe estar entre 0 y 1.")

    cantidad_test = max(len(clases), math.ceil(len(y) * test_size))
    if cantidad_test > len(y) - len(clases):
        raise ValueError(
            "No hay suficientes muestras para dejar al menos una de cada clase "
            "en entrenamiento y prueba. Agrega mas muestras."
        )
    return train_test_split(
        X,
        y,
        test_size=cantidad_test,
        random_state=42,
        stratify=y,
    )


def omitir_clases_incompletas(X, y):
    """Omite temporalmente clases con menos de dos muestras."""
    cantidades = Counter(int(valor) for valor in y)
    incompletas = sorted(
        etiqueta for etiqueta, cantidad in cantidades.items() if cantidad < 2
    )
    if not incompletas:
        return X, y

    nombres = ", ".join(
        f"{ETIQUETAS[etiqueta]} ({cantidades[etiqueta]} muestra)"
        for etiqueta in incompletas
    )
    print(f"ADVERTENCIA: se omiten categorias incompletas: {nombres}.")
    mascara = np.asarray(
        [int(etiqueta) not in incompletas for etiqueta in y], dtype=bool
    )
    return X[mascara], y[mascara]


def main() -> None:
    args = parsear_argumentos()
    try:
        X, y = cargar_dataset(args.dataset)
        X, y = omitir_clases_incompletas(X, y)
        if len(set(int(valor) for valor in y)) < 3:
            print(
                "ADVERTENCIA: entrenamiento provisorio con menos de tres "
                "categorias; la entrega final requiere al menos tres."
            )
        X_train, X_test, y_train, y_test = dividir_dataset(X, y, args.test_size)
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    clasificador = tree.DecisionTreeClassifier(
        random_state=42, max_depth=args.max_depth
    )
    clasificador.fit(X_train, y_train)
    predicciones = clasificador.predict(X_test)
    precision = float(accuracy_score(y_test, predicciones))
    matriz = confusion_matrix(y_test, predicciones, labels=sorted(ETIQUETAS))

    artefacto = {
        "modelo": clasificador,
        "etiquetas": ETIQUETAS,
        "columnas": COLUMNAS_HU,
        "creado": datetime.now().isoformat(timespec="seconds"),
    }
    args.modelo.parent.mkdir(exist_ok=True)
    dump(artefacto, args.modelo)

    args.arbol.parent.mkdir(exist_ok=True)
    args.arbol.write_text(
        tree.export_text(
            clasificador,
            feature_names=COLUMNAS_HU,
            class_names=[ETIQUETAS[numero] for numero in clasificador.classes_],
            show_weights=True,
        ),
        encoding="utf-8",
    )

    metricas = {
        "muestras_totales": int(len(y)),
        "muestras_entrenamiento": int(len(y_train)),
        "muestras_prueba": int(len(y_test)),
        "muestras_por_clase": {
            ETIQUETAS[numero]: int((y == numero).sum()) for numero in sorted(ETIQUETAS)
        },
        "precision_prueba": precision,
        "clases_matriz": [ETIQUETAS[numero] for numero in sorted(ETIQUETAS)],
        "matriz_confusion": matriz.tolist(),
    }
    args.metricas.parent.mkdir(exist_ok=True)
    args.metricas.write_text(
        json.dumps(metricas, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Dataset: {args.dataset}")
    print(f"Entrenamiento: {len(y_train)} muestras")
    print(f"Prueba: {len(y_test)} muestras")
    print(f"Precision de prueba: {precision:.2%}")
    print("Matriz de confusion:")
    print(matriz)
    print(f"Modelo guardado: {args.modelo}")
    print(f"Arbol exportado: {args.arbol}")
    print(f"Metricas guardadas: {args.metricas}")


if __name__ == "__main__":
    main()
