import json                           # para leer los datos del generador

from joblib import dump               # para guardar el modelo en disco
import matplotlib                    # libreria de graficos
import numpy as np                    # operaciones matematicas con arrays
from sklearn import tree              # arbol de decision de scikit-learn


# Archivos y claves
ARCHIVO_DATOS = "datos.json"
ARCHIVO_MODELO = "modelo.joblib"
ARCHIVO_ARBOL = "arbol_decision.png"
CLAVE_MUESTRAS = "X"
CLAVE_ETIQUETAS = "Y"
CODIFICACION_ARCHIVO = "utf-8"

# Etiquetas del modelo
NOMBRES = {
    1: "Circulos",
    2: "Cuadrados",
    3: "Rectangulos",
    4: "Triangulos",
}

# Transformacion y entrenamiento
EPSILON_LOGARITMO = 1e-10
PROFUNDIDAD_MAXIMA_ARBOL = 10
FACTOR_PORCENTAJE = 100.0

# Grafico del arbol
BACKEND_MATPLOTLIB = "Agg"
TAMANIO_FIGURA = (20, 10)
CANTIDAD_MOMENTOS_HU = 7
TAMANIO_FUENTE_ARBOL = 7
RESOLUCION_ARBOL_DPI = 150
PRIMER_INDICE_HU = 0


matplotlib.use(BACKEND_MATPLOTLIB)
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Leer datos generados por generador.py
# -----------------------------------------------------------------------------
with open(ARCHIVO_DATOS, encoding=CODIFICACION_ARCHIVO) as archivo_entrada:
    datos = json.load(archivo_entrada)

X = datos[CLAVE_MUESTRAS]
Y = datos[CLAVE_ETIQUETAS]

print(f"Muestras cargadas: {len(X)}")
print(f"Clases presentes: {set(Y)}")

# -----------------------------------------------------------------------------
# Transformacion logaritmica
# -----------------------------------------------------------------------------
# Los Momentos de Hu tienen una escala numerica muy amplia. El logaritmo
# facilita que el arbol compare sus valores conservando el signo.
X_np = np.array(X)
X_log = -np.sign(X_np) * np.log10(np.abs(X_np) + EPSILON_LOGARITMO)

# -----------------------------------------------------------------------------
# Entrenar arbol de decision
# -----------------------------------------------------------------------------
clasificador = tree.DecisionTreeClassifier(max_depth=PROFUNDIDAD_MAXIMA_ARBOL)
clasificador.fit(X_log, Y)

accuracy = clasificador.score(X_log, Y) * FACTOR_PORCENTAJE
print(f"Accuracy en training: {accuracy:.1f}%")
print(f"Clases entrenadas: {[NOMBRES[c] for c in clasificador.classes_]}")

# -----------------------------------------------------------------------------
# Guardar imagen del arbol de decision
# -----------------------------------------------------------------------------
plt.figure(figsize=TAMANIO_FIGURA)
nombres_hu = [f"Hu{indice}" for indice in range(PRIMER_INDICE_HU, CANTIDAD_MOMENTOS_HU)]
nombres_clases = [NOMBRES[c] for c in clasificador.classes_]
tree.plot_tree(clasificador, feature_names=nombres_hu, class_names=nombres_clases, filled=True, fontsize=TAMANIO_FUENTE_ARBOL)
plt.tight_layout()
plt.savefig(ARCHIVO_ARBOL, dpi=RESOLUCION_ARBOL_DPI)
plt.close()
print(f"Arbol guardado: {ARCHIVO_ARBOL}")

# -----------------------------------------------------------------------------
# Guardar modelo para usar en clasificador.py
# -----------------------------------------------------------------------------
dump(clasificador, ARCHIVO_MODELO)
print(f"Modelo guardado: {ARCHIVO_MODELO}")
