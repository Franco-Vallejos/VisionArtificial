# TP4 - Deteccion de figuras con YOLO

Este proyecto utiliza Ultralytics YOLO para detectar figuras geometricas. A diferencia del TP2, durante la deteccion no se segmenta por color ni se calculan Momentos de Hu: YOLO recibe directamente cada cuadro de la camara y devuelve la clase, la confianza y la caja de cada figura.

## Dependencias

Instalar una sola vez:

```powershell
python -m pip install ultralytics opencv-python
```

## Dataset preparado

El proyecto ya contiene el dataset convertido al formato requerido por YOLO Detect en la carpeta `dataset_yolo`. No es necesario ejecutar un script de preparacion.

El conjunto tiene 90.000 imagenes de 200 x 200 pixeles. Cada imagen contiene una figura geometrica sobre un fondo uniforme y posee un archivo de texto con su caja y su clase.

El dataset original es [2D geometric shapes dataset](https://data.mendeley.com/datasets/wzr2yv7r53/1), publicado por EL KORCHI Anas con DOI `10.17632/wzr2yv7r53.1` y licencia CC BY 4.0.

### Clases

Los identificadores utilizados por YOLO son:

| ID | Clase |
|---:|---|
| 0 | Circle |
| 1 | Square |
| 2 | Triangle |
| 3 | Pentagon |
| 4 | Hexagon |
| 5 | Heptagon |
| 6 | Octagon |
| 7 | Nonagon |
| 8 | Star |

La clase `Rectangle` no forma parte de este dataset.

### Estructura

```text
dataset_yolo/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
  revision/
    train/
    val/
    test/
```

El contenido de cada carpeta es:

- `images/train`: 81.000 imagenes usadas para ajustar los pesos del modelo.
- `images/val`: 4.500 imagenes usadas por YOLO para controlar el aprendizaje durante el entrenamiento.
- `images/test`: 4.500 imagenes reservadas para la evaluacion final.
- `labels`: contiene una etiqueta con el mismo nombre base que cada imagen.
- `revision`: contiene muestras con las cajas dibujadas para comprobar visualmente las anotaciones.

Cada particion conserva las nueve clases. El reparto original es aproximadamente 90 % entrenamiento, 5 % validacion y 5 % prueba.

### Formato de las etiquetas

Por cada imagen existe un archivo `.txt` del mismo nombre. Por ejemplo:

```text
images/train/Circle_abc123.png
labels/train/Circle_abc123.txt
```

Cada fila de la etiqueta representa un objeto:

```text
clase centro_x centro_y ancho alto
```

Un ejemplo podria ser:

```text
0 0.500000 0.500000 0.420000 0.420000
```

Esto significa:

- `0`: la figura pertenece a la clase `Circle`.
- `centro_x` y `centro_y`: posicion del centro de la caja.
- `ancho` y `alto`: dimensiones de la caja.

Las cuatro coordenadas estan normalizadas entre 0 y 1. No representan pixeles, sino proporciones del ancho y alto de la imagen.

### Archivo `dataset.yaml`

El archivo `dataset.yaml` le indica a Ultralytics donde estan las particiones y que nombre corresponde a cada ID:

```yaml
path: "C:/Users/fnval/Desktop/git/vision artificial/TP4/Figuras/dataset_yolo"
train: images/train
val: images/val
test: images/test
```

Tambien contiene la lista completa de las nueve clases. Los scripts pasan este archivo a YOLO para que pueda relacionar imagenes, etiquetas y nombres. Si el proyecto se mueve a otra ubicacion, se debe actualizar el valor de `path`.

## Entrenar

Ejecutar:

```powershell
python entrenar.py
```

`entrenar.py` carga el modelo preentrenado `yolo26n.pt` y realiza transferencia de aprendizaje con el dataset de figuras.

Las constantes principales son:

- `EPOCAS = 20`: cantidad maxima de recorridos completos sobre `train`.
- `TAMANIO_IMAGEN = 320`: resolucion usada por YOLO.
- `TAMANIO_LOTE = 32`: cantidad de imagenes procesadas antes de actualizar los pesos.
- `SEMILLA = 42`: permite repetir el experimento con la misma configuracion aleatoria.

Durante entrenamiento se aplican rotacion, traslacion, escala, espejo, cambios de color y mosaico. Estas variantes se generan solamente sobre `train`; `val` y `test` permanecen sin aumentacion para que la medicion sea independiente.

Ultralytics guarda curvas, metricas, matriz de confusion y pesos en:

```text
resultados/figuras_yolo26n/
```

Los dos pesos principales son:

- `weights/last.pt`: estado de la ultima epoca.
- `weights/best.pt`: estado que obtuvo el mejor resultado de validacion.

Los scripts posteriores utilizan `best.pt`.

### Entrenar con GPU NVIDIA

Para forzar el entrenamiento mediante CUDA se incluye un entrenador separado:

```powershell
python entrenador_gpu.py
```

Sus diferencias principales son:

- `DISPOSITIVO_GPU = 0`: selecciona la primera GPU NVIDIA.
- `TAMANIO_LOTE = -1`: Ultralytics calcula automaticamente un lote acorde con la memoria disponible.
- `TRABAJADORES = 4`: prepara varias imagenes en paralelo para alimentar la GPU.
- `amp=True`: utiliza precision mixta para reducir memoria y acelerar el entrenamiento.

Antes de comenzar, el script comprueba `torch.cuda.is_available()` y muestra el nombre de la GPU seleccionada. Si CUDA no esta disponible, finaliza sin iniciar un entrenamiento accidentalmente en CPU.

Los resultados se guardan en la misma ubicacion esperada por `evaluar.py` y `detectar.py`:

```text
resultados/figuras_yolo26n/
```

Se debe ejecutar `entrenar.py` o `entrenador_gpu.py`, no ambos para una misma corrida.

## Validar y evaluar

Despues de entrenar, ejecutar:

```powershell
python evaluar.py
```

`evaluar.py` carga `best.pt` y evalua solamente la particion `test`, que no se utilizo para ajustar los pesos. Al finalizar muestra:

- `mAP50`: precision media usando una coincidencia de caja IoU 0,50.
- `mAP50-95`: promedio mas exigente calculado entre distintos valores de IoU.

Tambien guarda graficos de evaluacion en:

```text
resultados/evaluacion_test/
```

Para el informe deben observarse ademas precision, recall, perdidas y matriz de confusion, identificando que figuras se confunden con mayor frecuencia.

## Detectar con la camara

Configurar al comienzo de `detectar.py`:

```python
CAMARA_WEB_INDEX = 0
CONFIANZA = 0.40
```

- `CAMARA_WEB_INDEX`: indice de la camara; normalmente la webcam principal es `0`.
- `CONFIANZA`: probabilidad minima requerida para mostrar una deteccion.

Ejecutar:

```powershell
python detectar.py
```

El script carga:

```text
resultados/figuras_yolo26n/weights/best.pt
```

Luego procesa continuamente la camara, dibuja las cajas, clases y confianzas, y muestra el resultado en pantalla. La ventana se cierra con `Q` o `Esc`.

## Limitacion principal

El dataset es grande y balanceado, pero sus imagenes son sinteticas y tienen fondos uniformes. Las metricas sobre `test` pueden ser muy altas sin garantizar el mismo rendimiento frente a figuras impresas, iluminacion real, perspectiva, sombras o fondos complejos. La prueba con la camara permite observar esa diferencia.

## Fuentes tecnicas

- [Entrenamiento con Ultralytics YOLO](https://docs.ultralytics.com/modes/train)
- [Formato de datasets de deteccion](https://docs.ultralytics.com/datasets/detect)
- [Inferencia con Ultralytics YOLO](https://docs.ultralytics.com/modes/predict)
