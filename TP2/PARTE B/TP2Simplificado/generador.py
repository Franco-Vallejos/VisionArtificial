import json                  # para guardar X e Y en un archivo
import os                    # para recorrer carpetas del dataset

import cv2                   # OpenCV para leer imagenes y extraer contornos


# Rutas y nombres
CARPETA_DATASET = r".\dataset"
ARCHIVO_DATOS = "datos.json"
CLAVE_MUESTRAS = "X"
CLAVE_ETIQUETAS = "Y"
EXTENSIONES_IMAGEN = (".jpg", ".png")
CODIFICACION_ARCHIVO = "utf-8"

# Etiqueta numerica - nombre de carpeta
ETIQUETAS = {
    1: "Circulos",
    2: "Cuadrados",
    3: "Rectangulos",
    4: "Triangulos",
}

# Procesamiento de imagen
TAMANIO_FILTRO_GAUSSIANO = (5, 5)
SIGMA_FILTRO_GAUSSIANO = 0
UMBRAL_INICIAL_OTSU = 0
VALOR_BINARIO_MAXIMO = 255
TAMANIO_KERNEL_MORFOLOGICO = (5, 5)
AREA_MINIMA_CONTORNO = 500


X = []   # vectores con 7 Momentos de Hu, uno por imagen
Y = []   # etiquetas numericas correspondientes

for num_etiqueta, nombre_clase in ETIQUETAS.items():
    carpeta = os.path.join(CARPETA_DATASET, nombre_clase)

    if not os.path.exists(carpeta):
        print(f"No existe: {carpeta}")
        continue

    archivos = [nombre for nombre in os.listdir(carpeta) if nombre.lower().endswith(EXTENSIONES_IMAGEN)]

    if not archivos:
        print(f"AVISO: carpeta vacia: {carpeta}")
        continue

    for archivo in archivos:
        ruta_imagen = os.path.join(carpeta, archivo)
        img = cv2.imread(ruta_imagen)
        if img is None:
            continue

        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gris = cv2.GaussianBlur(gris, TAMANIO_FILTRO_GAUSSIANO, SIGMA_FILTRO_GAUSSIANO)
        _, bw = cv2.threshold(gris, UMBRAL_INICIAL_OTSU, VALOR_BINARIO_MAXIMO, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, TAMANIO_KERNEL_MORFOLOGICO)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

        contornos, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos:
            continue

        contorno = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(contorno) < AREA_MINIMA_CONTORNO:
            continue

        momentos = cv2.moments(contorno)
        hu = cv2.HuMoments(momentos).flatten()

        X.append(hu.tolist())
        Y.append(num_etiqueta)
        print(f"  {nombre_clase} | {archivo}")

with open(ARCHIVO_DATOS, "w", encoding=CODIFICACION_ARCHIVO) as archivo_salida:
    json.dump({CLAVE_MUESTRAS: X, CLAVE_ETIQUETAS: Y}, archivo_salida)

print(f"\nTotal muestras: {len(X)}")
print(f"Datos guardados en {ARCHIVO_DATOS}")
