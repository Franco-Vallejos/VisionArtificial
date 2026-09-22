# Guía paso a paso de `aruco_tp3_clase.py`

Esta guía explica el programa desde cero, suponiendo que quien la lee no tiene conocimientos previos de visión artificial. El objetivo es entender qué problema resuelve, qué significa cada transformación y cómo se relaciona la implementación con los conceptos desarrollados en las clases del 10 y 17 de septiembre.

Archivos relacionados:

- Código: [`aruco_tp3_clase.py`](./aruco_tp3_clase.py)
- Marcador ArUco: [`aruco_4x4_50_id0_100mm.png`](./aruco_4x4_50_id0_100mm.png)
- PDF para imprimir: [`aruco_4x4_50_id0_100mm_A4.pdf`](../output/pdf/aruco_4x4_50_id0_100mm_A4.pdf)
- Resumen del 10/09: [`Resumen_clase_20260910.md`](../videos_clases/clase_20260910/Resumen_clase_20260910.md)
- Resumen del 17/09: [`Resumen_clase_20260917.md`](../videos_clases/clase_20260917/Resumen_clase_20260917.md)

## 1. ¿Qué hace el programa?

La cámara observa una mesa o superficie plana desde una posición inclinada. En esa superficie se coloca un marcador ArUco cuadrado de `100 mm` de lado.

El programa permite:

1. Detectar el marcador en la imagen de la cámara.
2. Usarlo para registrar un sistema de coordenadas real medido en milímetros.
3. Crear una vista cenital, como si la mesa se observara perfectamente desde arriba.
4. Mover el marcador y conocer su posición y orientación dentro del plano registrado.
5. Dibujar la misma grilla métrica en la vista cenital y en perspectiva.
6. Seleccionar un punto de la vista cenital y medir su distancia al marcador.
7. Reproyectar ese punto y esa medición sobre la imagen en tiempo real.

La idea central es convertir coordenadas de la cámara, medidas en **píxeles**, a coordenadas del mundo real, medidas en **milímetros**.

## 2. Conceptos mínimos de visión artificial

### 2.1 Una imagen es una matriz

Una imagen digital puede pensarse como una grilla de píxeles. Cada píxel se identifica con dos coordenadas:

- `u`: columna, que aumenta hacia la derecha;
- `v`: fila, que aumenta hacia abajo.

El punto `(0, 0)` está en la esquina superior izquierda de la imagen.

Estas coordenadas solamente indican una ubicación dentro de la imagen. Por sí solas no dicen cuántos centímetros hay entre dos objetos.

### 2.2 Coordenadas físicas

El programa también define un sistema de coordenadas sobre la mesa:

- `x` aumenta hacia la derecha;
- `y` aumenta hacia arriba;
- el origen `(0, 0)` queda en el centro del ArUco al registrar;
- las unidades son milímetros.

El eje `y` físico y el eje vertical de la imagen apuntan en sentidos contrarios. Por eso aparecen restas e inversiones de signo al pasar de un sistema al otro.

### 2.3 ¿Qué es un ArUco?

Un ArUco es un marcador fiduciario: un patrón cuadrado diseñado para que un programa pueda detectar con precisión:

- su identificador;
- sus cuatro esquinas;
- su centro;
- su orientación.

El programa usa el diccionario `DICT_4X4_50` y el marcador de ID `0`. El diccionario del código debe coincidir con el usado para generar el marcador.

No hay un *dataset*, entrenamiento ni un clasificador aprendido. La detección del ArUco y el resto del proceso pertenecen a la visión geométrica clásica.

### 2.4 ¿Por qué necesitamos conocer el tamaño del marcador?

Una fotografía no contiene una escala física absoluta. Un cuadrado visto en una imagen podría medir `10 cm`, `1 m` o `10 m`.

Como sabemos que el ArUco mide `100 mm`, cada lado real va desde `-50 mm` hasta `+50 mm` respecto de su centro. Esa medida conocida permite establecer la escala entre píxeles y milímetros.

### 2.5 ¿Qué es una homografía?

Una homografía es una matriz `3 x 3` que relaciona dos representaciones del mismo plano. En este proyecto relaciona la mesa vista en perspectiva con:

- un plano físico medido en milímetros;
- una imagen cenital medida en píxeles.

De forma conceptual, un punto se transforma así:

```text
[a]       [u]
[b] = H · [v]
[w]       [1]

x = a / w
y = b / w
```

`H` es la homografía, `(u, v)` es el punto original y `(x, y)` es el punto transformado. La división por `w` se llama normalización o deshomogeneización. `cv2.perspectiveTransform` realiza internamente estas operaciones.

Una homografía tiene ocho grados de libertad efectivos y necesita como mínimo cuatro correspondencias. Las cuatro esquinas del ArUco proporcionan exactamente esos cuatro pares de puntos.

## 3. Los tres sistemas de coordenadas del programa

| Sistema | Unidad | Origen | Uso |
|---|---|---|---|
| Imagen de cámara | píxeles | esquina superior izquierda | Detectar el ArUco y mostrar `Cam` |
| Mundo físico | milímetros | centro del ArUco al registrar | Medir posición y distancia reales |
| Vista `W2D` | píxeles | centro de la ventana cenital | Dibujar una representación fácil de interpretar |

Es importante no mezclar estos sistemas. Un valor de `50` puede significar `50 píxeles` o `50 mm` según el sistema en el que se encuentre.

## 4. Las dos homografías principales

El programa calcula dos homografías distintas durante el registro.

### 4.1 `h_image_to_world`

Transforma:

```text
píxeles de la cámara -> milímetros del mundo
```

Se usa para localizar el marcador y calcular distancias reales.

### 4.2 `h_image_to_view`

Transforma:

```text
píxeles de la cámara -> píxeles de W2D
```

Se usa para crear la vista cenital y colocar allí las anotaciones.

Aunque las dos parten de la misma cámara, sus destinos tienen unidades distintas. Por eso no deben intercambiarse.

### 4.3 `h_world_to_image`

Es la inversa de `h_image_to_world`:

```text
milímetros del mundo -> píxeles de la cámara
```

Se usa para proyectar la grilla métrica y el punto seleccionado sobre la imagen en perspectiva.

## 5. Flujo general

```text
Cámara
  |
  v
Cuadro BGR -> imagen gris -> detección ArUco -> ID y cuatro esquinas
                                      |
                         se presiona R una vez
                                      |
                                      v
                   registro de origen, escala y ejes
                                      |
                   +------------------+------------------+
                   |                                     |
                   v                                     v
        H cámara -> mundo                     H cámara -> W2D
             píxeles -> mm                       píxeles -> píxeles
                   |                                     |
                   v                                     v
     localización y mediciones                 fondo cenital estático

En cada cuadro posterior:
esquinas actuales -> transformar puntos -> centro, ángulo y dibujos

Clic en W2D:
píxel W2D -> punto en mm -> distancia -> reproyección sobre Cam
```

## 6. Constantes y configuración

Al comienzo del archivo se definen las decisiones generales del programa:

```python
CAMERA_INDEX = 0
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 100.0
WORLD_WIDTH_MM = 400.0
WORLD_HEIGHT_MM = 300.0
PIXELS_PER_MM = 1.0
GRID_STEP_MM = 50.0
```

Su significado es:

- `CAMERA_INDEX`: cámara que abre OpenCV; normalmente `0` es la cámara principal.
- `MARKER_SIZE_MM`: tamaño real de cada lado del marcador.
- `WORLD_WIDTH_MM` y `WORLD_HEIGHT_MM`: región física representada por `W2D`.
- `PIXELS_PER_MM`: escala gráfica de `W2D`; con valor `1`, un milímetro se representa con un píxel.
- `GRID_STEP_MM`: separación de `50 mm` entre líneas de la grilla.

Con estos valores, `W2D` mide `400 x 300 píxeles` y representa una región física de `400 x 300 mm`:

- `x` va aproximadamente de `-200` a `+200 mm`;
- `y` va aproximadamente de `-150` a `+150 mm`.

## 7. Explicación de cada función

### 7.1 `detectar(frame)`

Su tarea es encontrar los ArUco presentes en un cuadro.

1. Recibe una imagen BGR de la cámara.
2. La convierte a escala de grises con `cv2.cvtColor`.
3. Llama a `detector.detectMarkers(gray)`.
4. Devuelve las esquinas y los identificadores detectados.

La escala de grises alcanza porque el patrón ArUco codifica información mediante regiones claras y oscuras; el color no aporta información necesaria.

La forma típica de una detección es:

```text
corners: (cantidad_de_marcadores, 1, 4, 2)
                                  |  |
                                  |  +-- columnas x, y
                                  +----- cuatro esquinas
```

### 7.2 `elegir_marcador(corners, ids, marker_id=None)`

Esta función decide qué marcador usar.

- Si no hay detecciones, devuelve `None`.
- Antes del registro, elige el primer marcador visible.
- Después del registro, busca siempre el mismo ID.

Mantener el ID evita que el sistema cambie accidentalmente de referencia si aparecen varios ArUco en la imagen.

### 7.3 `registrar(frame, marker_corners)`

Esta es la función central del registro. Sólo se ejecuta cuando se presiona `R` y existe un marcador visible.

#### Paso 1: definir las esquinas detectadas

`image_points_matrix` contiene las cuatro esquinas del marcador en píxeles de cámara. Cada fila es un punto `[x, y]` y se conserva el orden:

1. superior izquierda;
2. superior derecha;
3. inferior derecha;
4. inferior izquierda.

El orden es crítico: cada fila de origen debe corresponder con la misma esquina en el destino.

#### Paso 2: definir el marcador en milímetros

Para un lado de `100 mm`, la mitad es `50 mm`. La matriz física es:

```text
[-50, +50]  superior izquierda
[+50, +50]  superior derecha
[+50, -50]  inferior derecha
[-50, -50]  inferior izquierda
```

Así, el centro queda en `(0, 0)`, `x` apunta a la derecha e `y` hacia arriba.

#### Paso 3: definir el marcador dentro de W2D

`view_points_matrix` describe el mismo cuadrado, pero en píxeles de la vista cenital. El marcador registrado se coloca en el centro de la ventana.

Con una vista de `400 x 300` y escala `1 píxel/mm`, su centro es `(200, 150)` y sus esquinas quedan a `50 píxeles` de ese centro.

#### Paso 4: calcular las homografías

`cv2.getPerspectiveTransform` recibe exactamente cuatro puntos de origen y cuatro de destino:

```python
h_image_to_world = cv2.getPerspectiveTransform(image_points_matrix, world_points_matrix)
h_image_to_view = cv2.getPerspectiveTransform(image_points_matrix, view_points_matrix)
```

Se utiliza `getPerspectiveTransform` porque el problema tiene cuatro correspondencias conocidas. `findHomography` sería más apropiado si hubiera muchos puntos, ruido o valores atípicos.

#### Paso 5: calcular la transformación inversa

```python
h_world_to_image = np.linalg.inv(h_image_to_world)
```

Esta matriz permite tomar un punto expresado en milímetros y encontrar dónde debe dibujarse en la imagen original.

#### Paso 6: crear el fondo cenital

```python
background = cv2.warpPerspective(frame, h_image_to_view, tamaño_w2d)
```

`warpPerspective` transforma todos los píxeles del cuadro. Se ejecuta una sola vez durante el registro porque es más costoso que transformar unos pocos puntos.

El resultado queda guardado como fondo estático. Mover el ArUco no vuelve a generar ese fondo.

### 7.4 `localizar(marker_corners, h_image_to_world, h_image_to_view)`

Esta función calcula la pose actual del marcador.

1. Ordena sus cuatro esquinas actuales como una matriz `4 x 2`.
2. Las adapta al formato `N x 1 x 2` requerido por OpenCV.
3. Transforma las esquinas al mundo físico usando `h_image_to_world`.
4. Transforma las mismas esquinas a `W2D` usando `h_image_to_view`.
5. Promedia las cuatro esquinas para obtener el centro.
6. Usa el lado superior para calcular la orientación.

El vector de orientación se obtiene restando:

```text
dirección = esquina_superior_derecha - esquina_superior_izquierda
```

Luego se calcula:

```python
angle_deg = degrees(atan2(direccion_y, direccion_x))
```

`atan2` conserva el cuadrante y permite distinguir giros positivos y negativos.

### 7.5 `dibujar_grilla_cam(frame, h_world_to_image)`

La grilla se define primero en el sistema físico, donde es sencillo pedir líneas cada `50 mm`.

Para cada línea:

1. Se crean sus dos extremos en milímetros.
2. Se adaptan al formato requerido por OpenCV.
3. Se transforman mediante `h_world_to_image`.
4. Se dibuja el segmento entre los dos píxeles resultantes.

Por eso la grilla aparece deformada en `Cam`: no está mal dibujada, sino proyectada con la misma perspectiva que la mesa.

La grilla queda fija respecto del plano registrado. El marcador puede moverse sobre ella.

### 7.6 `dibujar_grilla_w2d(frame)`

En la vista cenital no hace falta una homografía para dibujar la grilla, porque ya se trabaja en un sistema frontal regular.

La conversión utiliza:

```text
columna = centro_x + x_mm · píxeles_por_mm
fila    = centro_y - y_mm · píxeles_por_mm
```

La resta en `fila` aparece porque `y` físico crece hacia arriba, mientras que las filas de una imagen crecen hacia abajo.

### 7.7 `manejar_click(event, x, y, state)`

OpenCV llama a esta función cuando ocurre un evento de mouse en `W2D`. Si se presiona el botón izquierdo, se guarda el punto `[x, y]` seleccionado.

Ese punto está inicialmente expresado en píxeles de la vista cenital.

### 7.8 `convertir_w2d_a_mm(point)`

Convierte el clic de píxeles W2D a milímetros del mundo:

```text
x_mm = (x_pixel - centro_x) / píxeles_por_mm
y_mm = (centro_y - y_pixel) / píxeles_por_mm
```

Ejemplo con la configuración actual:

- centro de W2D: `(200, 150)`;
- clic: `(250, 100)`;
- resultado: `(50 mm, 50 mm)`.

### 7.9 `dibujar_medicion_cam(...)`

Esta función muestra en `Cam` la medición iniciada desde `W2D`.

1. Convierte el clic a milímetros.
2. Lo coloca en una matriz con el formato requerido por `perspectiveTransform`.
3. Aplica `h_world_to_image` para obtener el píxel equivalente en la cámara.
4. Calcula el centro actual del ArUco en la imagen.
5. Calcula la distancia real entre el punto y el centro del marcador.
6. Dibuja el punto, la línea y el valor en milímetros.

La distancia euclídea es:

```text
distancia = sqrt((x_punto - x_marcador)² + (y_punto - y_marcador)²)
```

El punto seleccionado representa una ubicación fija del plano. Si el ArUco se mueve, el punto permanece fijo, la línea cambia y la distancia se actualiza.

### 7.10 `dibujar_w2d(...)`

Esta función compone la ventana cenital:

1. Copia el fondo estático obtenido al registrar.
2. Dibuja la grilla.
3. Dibuja los ejes `x` e `y`.
4. Dibuja el contorno actual del ArUco.
5. Dibuja una flecha con su orientación.
6. Escribe su posición en milímetros y su ángulo.
7. Si existe un clic, dibuja el punto, la línea y la distancia.

Al calcular la punta de la flecha se resta el seno en la coordenada vertical porque el eje `y` del mundo y las filas de la imagen tienen sentidos opuestos.

### 7.11 `main()`

`main` coordina todo el programa.

#### Inicio

1. Abre la cámara.
2. Inicializa las homografías como `None` porque todavía no hay registro.
3. Crea las ventanas `Cam` y `W2D`.
4. Asocia el mouse de `W2D` con `manejar_click`.

#### Bucle en tiempo real

Para cada cuadro:

1. Lee la cámara.
2. Detecta todos los ArUco.
3. Si ya existe un registro, dibuja la grilla en perspectiva.
4. Dibuja los contornos e identificadores detectados.
5. Busca el mismo ID utilizado durante el registro.
6. Si lo encuentra, actualiza posición, orientación y ambas ventanas.
7. Si existe un punto seleccionado, actualiza también la medición.
8. Lee el teclado.

#### Teclas

- `R`: registra o vuelve a registrar el plano.
- `Q`: termina el programa.
- `Esc`: termina el programa.

Al volver a registrar se borran el punto y la medición anteriores porque el sistema de referencia físico acaba de cambiar.

## 8. Registro frente a localización

Esta diferencia es fundamental.

### Registro

Ocurre una vez al presionar `R`:

- fija el origen;
- fija los ejes;
- fija la escala;
- calcula las homografías;
- captura el fondo cenital.

### Localización

Ocurre en todos los cuadros posteriores:

- detecta el marcador actual;
- reutiliza las homografías;
- transforma sólo sus puntos;
- calcula centro y ángulo;
- actualiza los dibujos.

No se recalcula una homografía cada vez que el marcador se mueve. La homografía describe la relación entre la cámara y el plano, no la posición particular del marcador.

## 9. Relación directa con las clases

| Concepto desarrollado | Implementación en el código |
|---|---|
| Pasar de píxeles a unidades físicas | `h_image_to_world` convierte las esquinas detectadas a milímetros |
| La escala no surge de una fotografía | `MARKER_SIZE_MM = 100.0` aporta una medida real conocida |
| Una homografía necesita cuatro correspondencias | Se usan las cuatro esquinas del ArUco en el mismo orden |
| Usar `getPerspectiveTransform` en el caso mínimo | `registrar` calcula las dos homografías con cuatro puntos |
| Diferenciar homografía métrica y visual | `h_image_to_world` entrega mm y `h_image_to_view` entrega píxeles W2D |
| Diferenciar `warpPerspective` de `perspectiveTransform` | El primero crea el fondo una vez; el segundo transforma puntos en tiempo real |
| Cámara fija y registro discreto | Las matrices permanecen constantes hasta volver a presionar `R` |
| Grilla pensada en el plano frontal | Sus extremos se definen en mm y se llevan a `Cam` con la inversa |
| Marcador fiduciario para origen, orientación y escala | El ArUco registrado define `(0, 0)`, los ejes y los `100 mm` |
| Clic sobre un punto y distancia física | El clic se convierte a mm y se mide respecto del centro del ArUco |

La clase del 10/09 introduce la diferencia entre píxeles y unidades métricas, las homografías, la cámara fija y la conveniencia de transformar puntos en lugar de deformar la imagen completa en cada cuadro.

La clase del 17/09 completa el procedimiento: cuatro correspondencias, `getPerspectiveTransform`, dos homografías con destinos diferentes, uso de la inversa para la grilla y el ArUco de tamaño conocido como referencia métrica.

La selección de un punto y su distancia es una aplicación directa del ejemplo de localizar un punto de una mesa de billar: primero se obtiene una coordenada en la imagen y luego se expresa dentro del plano físico.

## 10. Cómo probarlo paso a paso

### Paso 1: imprimir el marcador

Imprimir el PDF en tamaño real, sin opciones como “rellenar página” o “ajustar al área imprimible”. Después de imprimir, medir el cuadrado para comprobar que su lado físico sea de `100 mm`.

Si el tamaño impreso no es correcto, las distancias calculadas tampoco serán correctas.

### Paso 2: preparar la escena

1. Usar una mesa o superficie plana.
2. Fijar la cámara para que no se mueva.
3. Inclinarla lo suficiente para observar perspectiva, pero no tanto como para impedir la detección.
4. Mantener todo el marcador dentro de la imagen.
5. Evitar sombras fuertes, reflejos y desenfoque.

### Paso 3: ejecutar

Desde una terminal:

```powershell
cd "C:\Users\fnval\Desktop\git\vision artificial\TP3"
python aruco_tp3_clase.py
```

Si faltan dependencias, el proyecto necesita una versión de OpenCV con el módulo ArUco y NumPy. Habitualmente se instalan con:

```powershell
python -m pip install opencv-contrib-python numpy
```

### Paso 4: registrar

1. Colocar el ArUco donde se desea establecer el origen.
2. Verificar que `Cam` dibuje su contorno e ID.
3. Presionar `R`.
4. Confirmar en la terminal el mensaje de registro.
5. Observar el fondo cenital, los ejes y la grilla.

### Paso 5: mover el marcador

Mover el ArUco sobre la misma superficie sin mover la cámara. En `W2D` deberían cambiar:

- sus coordenadas `(x, y)` en milímetros;
- su contorno;
- la flecha de orientación;
- su ángulo.

La grilla y el fondo deben permanecer fijos porque representan el plano registrado.

### Paso 6: medir un punto

1. Hacer clic en cualquier punto de `W2D`.
2. Observar el punto y la línea magenta en `W2D`.
3. Ver el mismo punto reproyectado sobre `Cam`.
4. Mover el marcador y comprobar que la distancia cambia.

### Paso 7: volver a registrar

Si se mueve la cámara, cambia el zoom o se desea otro origen, colocar nuevamente el ArUco y presionar `R`. La selección anterior se borra porque pertenecía al sistema de referencia previo.

## 11. Qué debería observarse

### Antes de registrar

- `Cam` muestra la cámara y las detecciones ArUco.
- `W2D` solicita presionar `R`.
- Todavía no hay grilla proyectada ni coordenadas métricas válidas.

### Después de registrar

- aparece la grilla sobre `Cam`;
- aparece el fondo cenital estático;
- el origen coincide con la posición inicial del centro del ArUco;
- la posición inicial es aproximadamente `(0, 0) mm`;
- la orientación inicial es aproximadamente `0°`.

Puede haber pequeñas variaciones por ruido de detección, perspectiva, impresión o calidad de la cámara.

## 12. Errores comunes

### “No se pudo abrir la cámara”

Otra aplicación puede estar usando la cámara o el índice puede ser incorrecto. Probar `CAMERA_INDEX = 1` si existe más de una cámara.

### No detecta el ArUco

Comprobar:

- que el marcador pertenezca a `DICT_4X4_50`;
- que el borde negro esté completo;
- que no haya reflejos ni desenfoque;
- que el marcador no se vea demasiado pequeño;
- que la perspectiva no sea extremadamente rasante.

### Las distancias son incorrectas

Las causas más frecuentes son:

- el marcador no fue impreso a `100 mm`;
- la impresión aplicó escalado automático;
- la cámara se movió después del registro;
- cambió el zoom o el enfoque modificó demasiado el encuadre;
- el punto u objeto no pertenece al mismo plano de la mesa.

### La grilla ya no coincide con la mesa

La homografía representa una relación geométrica fija. Si se mueve la cámara o el plano, hay que presionar `R` nuevamente.

### El marcador desaparece temporalmente

Si el ID registrado no se detecta, el programa conserva la última vista `W2D` válida. La pose vuelve a actualizarse cuando el mismo marcador reaparece.

## 13. Limitaciones del enfoque

- La homografía es válida para un plano; un objeto elevado sobre la mesa no se proyecta correctamente.
- La cámara, el plano y el zoom deben permanecer fijos después del registro.
- No se realiza una calibración de lente ni una corrección explícita de distorsión.
- La precisión depende de la impresión, resolución, enfoque, iluminación y ángulo de cámara.
- El programa sigue el ID registrado, no cualquier objeto arbitrario.
- La vista cenital no crea información nueva: las regiones nunca observadas por la cámara no pueden reconstruirse.

## 14. Resumen para explicar el TP oralmente

Una explicación breve y completa podría ser:

> La cámara detecta las cuatro esquinas de un ArUco de tamaño conocido. Al presionar R, esas esquinas en píxeles se relacionan con cuatro posiciones conocidas en milímetros y con cuatro píxeles de una vista cenital. Con esas correspondencias se calculan dos homografías: una para medir en el mundo real y otra para visualizar. Mientras la cámara permanezca fija, las matrices se reutilizan en cada cuadro para transformar las esquinas actuales, calcular el centro y la orientación del marcador y actualizar las ventanas. La homografía inversa permite llevar la grilla y un punto físico de regreso a la imagen de cámara.

Las cinco ideas que no deberían confundirse son:

1. La cámara entrega píxeles, no milímetros.
2. El tamaño conocido del ArUco proporciona la escala.
3. El registro calcula las homografías; la localización solamente las reutiliza.
4. `warpPerspective` transforma una imagen y `perspectiveTransform` transforma puntos.
5. Si cambia la cámara o el plano, hay que registrar nuevamente.
