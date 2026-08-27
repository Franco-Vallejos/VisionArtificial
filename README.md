Claro. Te lo dejo listo para copiar directamente a tu `README.md`, manteniendo el contenido y ordenándolo con títulos, tablas y formato Markdown:

# Visión Artificial

**Docentes:**

* Alesio Esteban Sinopoli
* Jorgen Alejandro Silvestri

**Alumno:** Marcos León Rodríguez y Franco Vallejos
**Cuatrimestre:** 2C – 2026

---

# Clase 1 – 20/08/2026

## Tareas Principales en Visión Artificial

### Detección (Detection)

Tarea de identificar la presencia de un objeto específico dentro de una imagen y determinar su ubicación aproximada mediante una **caja delimitadora (bounding box)**.

**Ejemplo:**
Encuadrar cada vehículo en una autopista con un rectángulo verde y devolver sus coordenadas.

---

### Reconocimiento / Clasificación (Recognition / Classification)

Asignar una **categoría o etiqueta conceptual** a una imagen completa o a una región detectada previamente.

**Ejemplo:**

* Determinar si la cara detectada en la cámara corresponde al `"Usuario A"`.
* Determinar si un vehículo es un `"Auto"`, `"Camión"` o `"Moto"`.

---

### Localización (Localization)

Determinar la **posición precisa `(x, y)`** o las coordenadas espaciales dentro de la imagen de un elemento de interés.

**Ejemplo:**
Encontrar la posición exacta de los tornillos en una placa de circuito impreso para verificar su presencia.

---

### Segmentación (Segmentation)

Clasificar la imagen **a nivel de píxel**, delimitando el contorno o la silueta exacta del objeto.

A diferencia de la detección, no se limita a un simple rectángulo.

**Ejemplo:**
Pintar de un color específico únicamente la silueta de una célula en la imagen de un microscopio para separar su área del fondo.

---

### Puntos Clave / Malla (Landmarks / Keypoints)

Identificación de **puntos geométricos de interés específicos** en una estructura flexible o compleja para medir posturas o movimientos.

**Ejemplo:**

* Mapear los **33 puntos de las articulaciones del cuerpo** mediante Pose.
* Detectar los puntos de la boca y los ojos mediante Face Mesh.

---

# Conceptos de Desarrollo y Metodología

## Inspección Automática

Uso de **visión artificial en entornos industriales** para realizar controles de calidad mediante pruebas de ausencia/presencia o medición en líneas de producción.

**Ejemplo:**
Un sistema que toma una foto a cada producto en la cinta de montaje y valida si falta un cable o si está mal colocado.

---

## Debugging Visual

Proceso de diagnóstico del programa donde se generan **imágenes intermedias** —en escala de grises, binarias o anotadas— para verificar manualmente qué está "viendo" el algoritmo en cada etapa.

**Ejemplo:**
Mostrar 4 ventanas simultáneas:

1. Webcam original.
2. Imagen con bordes detectados mediante **Canny**.
3. Contornos encontrados.
4. Objetos marcados.

---

# ¿Qué es Computer Vision / Visión Artificial?

Es el campo de la informática que busca que una computadora pueda **obtener información a partir de imágenes o videos y tomar decisiones sobre ellos**.

### Ejemplos

La Visión Artificial permite:

* Detectar una cara.
* Detectar una mano.
* Reconocer objetos.
* Contar personas.
* Analizar movimientos.
* Identificar características dentro de una imagen.

### Idea general

```text
Imagen / Video
      ↓
Procesamiento
      ↓
Extracción de información
      ↓
Interpretación
      ↓
Decisión / Acción
```

---

# ¿Qué es OpenCV?

**OpenCV (Open Source Computer Vision Library)** es una biblioteca de programación para **visión artificial y procesamiento de imágenes y video**.

Proporciona herramientas para trabajar con:

* Imágenes.
* Videos.
* Cámaras.
* Transformaciones de imágenes.
* Detección de bordes.
* Dibujo y anotaciones.
* Procesamiento de imágenes.

## Principales métodos de OpenCV

| Método                    | ¿Qué hace?                                                                |
| ------------------------- | ------------------------------------------------------------------------- |
| `cv2.VideoCapture()`      | Abre una webcam o carga un archivo de video.                              |
| `cv2.cvtColor()`          | Cambia el espacio de color de una imagen, por ejemplo a escala de grises. |
| `cv2.rectangle()`         | Dibuja rectángulos sobre una imagen.                                      |
| `cv2.imshow()`            | Muestra una imagen o frame en una ventana.                                |
| `cv2.waitKey()`           | Espera una tecla y permite pausar la ejecución.                           |
| `cv2.imread()`            | Carga una imagen desde un archivo.                                        |
| `cv2.destroyAllWindows()` | Cierra todas las ventanas abiertas por OpenCV.                            |
| `cv2.Canny()`             | Detecta los bordes dentro de una imagen.                                  |

---

# ¿Qué es MediaPipe?

**MediaPipe** es un framework/biblioteca desarrollado por **Google** que proporciona soluciones ya preparadas para determinadas tareas de **percepción y visión artificial**.

En lugar de tener que desarrollar desde cero un algoritmo para determinadas tareas, MediaPipe proporciona modelos y pipelines especializados.

## Módulos / Soluciones de MediaPipe

| Módulo / Solución             | ¿Qué hace?                                                                                         |
| ----------------------------- | -------------------------------------------------------------------------------------------------- |
| **Hand Detection / Tracking** | Detecta y realiza el seguimiento en tiempo real de las manos.                                      |
| **Face Detection**            | Identifica la presencia y ubicación de rostros en una imagen o video.                              |
| **Face Landmarks**            | Localiza puntos clave en el rostro mediante una malla facial para mapear expresiones y rasgos.     |
| **Pose Estimation**           | Detecta la postura corporal y el esqueleto de una persona mediante sus principales articulaciones. |
| **Object Detection**          | Identifica y ubica múltiples objetos de diferentes categorías.                                     |
| **Gesture Recognition**       | Reconoce gestos específicos realizados con las manos o el cuerpo.                                  |

---

# Diferencia entre OpenCV y MediaPipe

La diferencia fundamental puede resumirse de esta manera:

| OpenCV                                                                                    | MediaPipe                                                               |
| ----------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Proporciona herramientas generales para trabajar con imágenes y video.                    | Proporciona modelos y pipelines especializados.                         |
| Permite procesar y modificar imágenes.                                                    | Permite interpretar determinados elementos dentro de imágenes o videos. |
| Incluye funciones como captura de video, conversión de colores, detección de bordes, etc. | Incluye soluciones como detección de manos, rostros, pose y gestos.     |
| Es más general.                                                                           | Es más específico para determinadas tareas de percepción.               |

### Ejemplo

**OpenCV** puede encargarse de:

```text
Webcam
   ↓
Capturar frame
   ↓
Procesar imagen
   ↓
Mostrar resultado
```

Mientras que **MediaPipe** puede encargarse de interpretar el contenido:

```text
Imagen / Video
      ↓
MediaPipe
      ↓
Detectar mano
      ↓
Obtener puntos de referencia
```

Por ejemplo, MediaPipe puede detectar una mano y devolver **puntos de referencia (landmarks)** correspondientes a diferentes partes de la mano.
