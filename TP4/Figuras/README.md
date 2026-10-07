# TP4 - Deteccion de figuras con YOLOv8

Este proyecto entrena un detector Ultralytics YOLOv8 para reconocer figuras geometricas dibujadas a mano. Las clases actuales son `Circle`, `Rectangle`, `Square` y `Triangle`.

## Dependencias

```powershell
python -m pip install ultralytics opencv-python
```

Para usar `entrenador_gpu.py` tambien se necesita una instalacion de PyTorch compatible con CUDA y una GPU NVIDIA. El script comprueba la disponibilidad de CUDA antes de entrenar.

## Estructura del dataset

El dataset exportado desde Roboflow ya esta preparado para YOLO y se incluye en el repositorio:

```text
Figuras/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
  dataset_original/
```

- `train`: 271 imagenes para ajustar los pesos.
- `valid`: 32 imagenes para medir el modelo durante el entrenamiento.
- `test`: 33 imagenes reservadas para la evaluacion final.
- `dataset_original`: conserva las fotografias originales para auditoria y trazabilidad.
- `auditoria_dataset`: contiene copias de cuadrados y rectangulos ambiguos para su revision.
- `data.yaml`: define las rutas y las cuatro clases de la version 5 del dataset.

Cada archivo de `labels` tiene el mismo nombre base que su imagen. El primer valor de cada anotacion es el identificador de clase: `0` para `Circle`, `1` para `Rectangle`, `2` para `Square` y `3` para `Triangle`. Cada fila contiene una caja YOLO con centro `x`, centro `y`, ancho y alto normalizados entre 0 y 1. Los poligonos exportados por Roboflow fueron convertidos a sus cajas envolventes para mantener todo el dataset en un unico formato de deteccion.

Los archivos `README.dataset.txt` y `README.roboflow.txt` conservan los metadatos de la exportacion de Roboflow.

## 1. Entrenamiento

### Con GPU NVIDIA

```powershell
python entrenador_gpu.py
```

`entrenador_gpu.py` utiliza `yolov8n.pt`, exige CUDA y aplica aumentacion mediante rotacion, traslacion, escala, espejo, variaciones de color y mosaico.

### Con CPU

```powershell
python entrenar.py
```

La version por CPU usa el mismo modelo, dataset, aumentaciones y nombre de salida. Es considerablemente mas lenta. Se debe ejecutar uno de los dos entrenadores, no ambos para una misma corrida.

Los entrenadores guardan el mejor modelo en:

```text
resultados/figuras_yolov8n/weights/best.pt
```

Con `plots=True`, Ultralytics tambien genera `results.png`, curvas de precision, recall y F1, matrices de confusion, muestras del dataset y otros resultados del entrenamiento.

## 2. Evaluacion

Despues de entrenar:

```powershell
python evaluar.py
```

El script evalua `best.pt` sobre `test` dos veces y mantiene los entregables separados:

```text
resultados/evaluacion_metricas/
resultados/evaluacion_operativa/
```

- `evaluacion_metricas` usa el umbral predeterminado de Ultralytics para calcular correctamente mAP y las curvas completas.
- `evaluacion_operativa` usa `CONFIANZA_OPERATIVA = 0.55` para producir una matriz de confusion representativa del uso real.

El valor de mAP informado por consola corresponde a `evaluacion_metricas`.

## 3. Deteccion con camara

Despues de entrenar:

```powershell
python detectar.py
```

`detectar.py` abre la camara indicada por `CAMARA_WEB_INDEX`, procesa cada cuadro en tiempo real y dibuja las cajas, clases y confianzas. Se cierra con `Q` o `Esc`.

La deteccion tambien necesita que exista:

```text
resultados/figuras_yolov8n/weights/best.pt
```

## Archivos generados

Las carpetas `resultados/`, los pesos `*.pt`, los ZIP originales y los archivos `*.cache` estan ignorados por Git. En una copia nueva del repositorio primero se debe entrenar; Ultralytics descargara `yolov8n.pt` si no esta disponible localmente.

## Referencias

- [Entrenamiento con Ultralytics YOLO](https://docs.ultralytics.com/modes/train)
- [Formato de datasets de deteccion](https://docs.ultralytics.com/datasets/detect)
- [Evaluacion con Ultralytics YOLO](https://docs.ultralytics.com/modes/val)
- [Inferencia con Ultralytics YOLO](https://docs.ultralytics.com/modes/predict)
