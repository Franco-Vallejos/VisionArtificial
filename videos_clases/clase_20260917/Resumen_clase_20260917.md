# Resumen de clase de Visión Artificial - 17/09/2026

Fuentes utilizadas: [transcripción](./transcripcion/Clase_20260917_transcripcion.md), [diapositivas recortadas](./ppt/Clase_20260917_ppt_recortadas.pdf) y la consigna [TP Localización homográfica](C:/Users/fnval/Downloads/TP%20Localizaci%C3%B3n%20homogr%C3%A1fica.pdf). No se utilizó el video para redactar el contenido.

## Idea central de la clase

La primera parte de la clase completa la teoría y el procedimiento práctico para calcular homografías a partir de correspondencias. El objetivo aplicado consiste en transformar posiciones medidas en píxeles a coordenadas físicas de un plano y, de forma independiente, producir una vista cenital para visualización.

El docente desarrolla casi todo el flujo del TP de localización homográfica: cámara fija, registro del plano mediante un marcador fiduciario, cálculo de dos homografías, transformación de puntos y actualización de la posición del marcador en unidades métricas.

La segunda parte introduce modelos de cámara, campo visual, sensores, lentes, artefactos ópticos, cámara estenopeica, cámaras de gran angular y proyecciones para imágenes omnidireccionales. Estos temas amplían la teoría de visión 3D. La conexión inmediata con el TP es que cualquier cambio físico o de zoom en la cámara invalida la homografía registrada.

## 1. Estimadores y correspondencias

Una transformación relaciona puntos correspondientes entre dos sistemas:

`p' = T p`

Cada correspondencia 2D aporta dos ecuaciones, una para cada coordenada. La cantidad mínima de correspondencias depende de los grados de libertad de la transformación:

| Transformación | Parámetros | Correspondencias mínimas |
|---|---:|---:|
| Rotación 2D | 1 | 1 |
| Traslación | 2 | 1 |
| Euclidiana o rototraslación | 3 | 2 |
| Similitud | 4 | 2 |
| Afín | 6 | 3 |
| Homografía | 8 | 4 |

Una matriz homogénea `3 x 3` contiene nueve valores, pero una homografía tiene ocho grados de libertad efectivos porque todas las matrices proporcionales representan la misma transformación.

Con el número mínimo de correspondencias se obtiene una solución directa. Con mediciones reales suelen utilizarse más puntos, estimadores y métodos robustos para reducir el efecto del ruido y los valores atípicos. En el TP se utiliza el caso mínimo controlado de cuatro correspondencias.

## 2. Vista frontal, vista cenital y plano físico

Una vista frontal observa un plano de manera perpendicular. Cuando ese plano es el suelo o una mesa y se lo observa desde arriba, se habla de vista cenital.

En una aplicación real la cámara puede estar inclinada. Incluso una instalación casi cenital tiene pequeñas diferencias a nivel de píxel, por lo que se registra el plano y se corrige mediante una homografía.

### Relación con el TP - 00:16:06 a 00:18:42

El docente define el TP3 como un problema de localización 2D sobre un plano. Para que el efecto sea visible, recomienda montar la cámara con una perspectiva marcada, aunque sin llegar a un ángulo que impida detectar el marcador.

El ejemplo de la mesa de billar resume el objetivo: se obtiene una coordenada `(x, y)` en píxeles y se debe devolver la posición equivalente en metros, centímetros o milímetros dentro del sistema de referencia de la mesa.

## 3. Registro del plano

El registro establece la relación entre la imagen de la cámara y un sistema de coordenadas conocido del mundo real.

Durante el registro se definen:

- el origen del sistema físico;
- la orientación de los ejes;
- la escala métrica;
- las correspondencias entre puntos de la imagen y puntos del plano;
- las homografías que se utilizarán después.

### Relación con el TP - 00:20:06 a 00:21:37

El caso de uso del TP supone una cámara fija y una escena fija. El origen y los ejes se establecen durante el registro. A partir de ese momento, una homografía convierte directamente coordenadas de la imagen en píxeles a coordenadas físicas, expresadas en milímetros.

La homografía permanece válida mientras no cambie la relación geométrica entre la cámara y el plano. Si se mueve la cámara, cambia su orientación o se modifica el zoom, hay que registrar nuevamente.

## 4. Las dos homografías del TP

La clase insiste en que existen dos transformaciones distintas:

### Homografía de cámara a mundo físico

- Entrada: coordenadas en píxeles de la imagen de cámara.
- Salida: coordenadas métricas del plano, por ejemplo milímetros.
- Uso: obtener la posición real del marcador.
- Se aplica a puntos, no a una imagen completa.

### Homografía de cámara a visualización cenital

- Entrada: coordenadas en píxeles de la imagen de cámara.
- Salida: coordenadas en píxeles de la imagen cenital.
- Uso: generar el fondo cenital y transportar anotaciones.
- Se puede aplicar a la imagen completa mediante `warpPerspective`.

Las dos matrices pueden ser proporcionales o estar relacionadas por una transformación de escala y traslación, pero sus sistemas destino no son iguales. Una produce milímetros y la otra produce píxeles.

### Relación con el TP - 00:21:14 a 00:30:43

El docente advierte que no debe localizarse el objeto ejecutando un *warp* completo, detectando nuevamente sobre la imagen deformada y convirtiendo después esa posición. Ese procedimiento agrega costo y puede perder información por interpolación.

El flujo recomendado es:

1. Detectar el marcador en la imagen original de la cámara.
2. Transformar sus puntos directamente a coordenadas métricas.
3. Utilizar el *warp* solamente para la visualización cenital solicitada por el TP.

La consigna confirma esta arquitectura: el fondo cenital se genera durante el registro y no se recalcula en cada cuadro.

## 5. Grilla y homografía inversa

Para dibujar una grilla en perspectiva conviene definirla primero en el sistema frontal, donde las coordenadas son sencillas. Después se aplica la homografía inversa para transportar sus extremos a la imagen de cámara.

Por ejemplo, una grilla métrica puede definirse cada 20 cm en el plano físico. Cada segmento se representa mediante dos extremos. Todos los puntos se transforman juntos y luego se dibujan las líneas correspondientes sobre la vista en perspectiva.

### Relación con el TP - 00:30:33 a 00:36:14

La grilla forma parte de la visualización pedida. El concepto importante es pensar la geometría en el espacio donde resulta natural y después transportarla:

- para dibujar sobre la vista cenital, se trabaja en píxeles de la visualización;
- para una grilla con medidas reales, se trabaja en el plano físico;
- para dibujar cualquiera de ellas sobre la cámara, se usa la transformación inversa correspondiente.

## 6. Obtención de la homografía

Una homografía necesita cuatro correspondencias. Cada punto de origen debe relacionarse con el punto correcto del destino y ambos arreglos deben conservar el mismo orden.

OpenCV ofrece dos funciones que la clase diferencia:

- `cv2.getPerspectiveTransform`: solución directa para exactamente cuatro correspondencias.
- `cv2.findHomography`: método más general, útil con muchos puntos, ruido y estimadores robustos.

### Relación con el TP - 00:41:24 a 00:45:07

El docente recomienda explícitamente `getPerspectiveTransform` para este TP porque se trabaja con cuatro correspondencias conocidas. No hace falta utilizar `findHomography`.

El orden de los argumentos determina el sentido de la transformación. Si se intercambian origen y destino se obtiene la homografía inversa. Conviene nombrar las matrices indicando los sistemas que relacionan para evitar invertirlas por error.

Después de obtener la homografía visual, `warpPerspective` genera la vista cenital con el tamaño elegido por el programa.

## 7. Tamaño y relación de aspecto de la vista cenital

El tamaño de la imagen cenital no surge automáticamente de la cámara. Es una decisión de diseño. El programa debe definir el ancho y el alto de la visualización y preservar la relación de aspecto real del plano para evitar deformaciones.

La imagen rectificada tampoco crea información nueva. Las regiones que la cámara no observó quedan vacías y los objetos que no pertenecen al plano registrado pueden deformarse.

## 8. Transformación de puntos con OpenCV

`cv2.perspectiveTransform` permite transformar un conjunto de puntos utilizando una homografía. Internamente realiza:

1. Expansión a coordenadas homogéneas.
2. Multiplicación por la matriz `H`.
3. Normalización o deshomogeneización.
4. Retorno a puntos 2D.

La función resulta adecuada para transformar a la vez los cuatro vértices del marcador, los extremos de la grilla u otros puntos de anotación.

## 9. Marcadores fiduciarios: QR y ArUco

La clase utiliza QR para explicar la detección y el orden de las esquinas. Un detector devuelve cuatro vértices en un orden consistente, lo que permite construir las correspondencias con un cuadrado de tamaño conocido.

Si el marcador mide `L`, un sistema físico centrado puede asignar a sus esquinas combinaciones de `-L/2` y `L/2`. Para un marcador de 100 mm, los valores son `-50` y `50` mm.

El docente señala que ArUco resulta más apropiado para localización porque codifica un identificador y está diseñado como marcador fiduciario. La consigna formal pide específicamente un marcador ArUco, por lo que debe priorizarse ArUco aunque la explicación también muestre `QRCodeDetector`.

### Relación con el TP - 00:54:19 a 01:00:25

El tamaño real conocido del marcador aporta la escala que la imagen no contiene. Sus cuatro esquinas detectadas se relacionan con las cuatro coordenadas físicas conocidas y permiten calcular la homografía métrica.

## 10. Flujo completo descrito por el docente

Entre `01:06:03` y `01:10:12`, el docente resume el trabajo:

1. Imprimir o mostrar un marcador fiduciario.
2. Colocar una cámara fija observando una mesa o un plano en perspectiva.
3. Detectar el marcador.
4. Pulsar una tecla para registrar el plano.
5. Usar la posición del marcador durante el registro como origen del sistema físico.
6. Usar el tamaño conocido para establecer la escala.
7. Calcular las homografías una única vez.
8. Mover el marcador sin volver a registrar.
9. Transformar sus puntos detectados con la homografía guardada.
10. Mostrar en pantalla sus coordenadas métricas y la visualización cenital.

El docente advierte que una perspectiva demasiado rasante puede impedir la detección. Recomienda observar el plano desde arriba con una inclinación moderada. La cámara debe estar apoyada o fijada; no debe sostenerse con la mano.

Un marcador mostrado en un celular también puede funcionar, pero el brillo de la pantalla puede dificultar la detección. Imprimirlo suele producir un resultado más estable.

## 11. Modelos de cámara

La segunda mitad de la clase introduce conceptos generales que continuarán en clases posteriores.

### Campo visual

El campo visual indica cuánto de la escena entra en la imagen. Depende de la lente y del tamaño del sensor. Una lente con mayor campo visual permite observar más entorno, pero distribuye la resolución entre una región más amplia.

### Sensor y lente

El sensor convierte la luz en valores digitales. Su tamaño físico, resolución y capacidad de transmisión limitan la imagen obtenida. La lente forma la imagen sobre el sensor y debe quedar correctamente alineada con él.

### Artefactos no deseados

La clase menciona desenfoque, reflejos, destellos, movimiento borroso y limitaciones del zoom. Estos efectos pueden reducir la precisión de los algoritmos de visión.

### Modelo estenopeico

El modelo *pinhole* o estenopeico representa una cámara ideal mediante un centro de proyección y un plano de imagen. Sirve como base matemática para relacionar puntos 3D con coordenadas 2D.

### Cámaras de gran angular

Las lentes *fisheye* y omnidireccionales cubren campos visuales amplios, pero requieren modelos de proyección y distorsión diferentes del modelo convencional. La clase muestra rectificación, colineación y proyección equirectangular como formas de representar esas imágenes.

### Relación con el TP - 01:25:48 a 01:26:13

El zoom forma parte de la configuración geométrica de la cámara. Si se modifica después del registro, la homografía deja de ser válida y debe calcularse nuevamente. Para el TP, el zoom debe ajustarse primero y quedar fijo.

---

# TRABAJO PRÁCTICO - PUNTOS IMPORTANTES

## Objetivo confirmado

Desarrollar un sistema de localización 2D en tiempo real. Una cámara fija observa un plano en perspectiva y el programa determina la posición y orientación de un marcador ArUco en un sistema de referencia métrico registrado.

## Menciones y decisiones del TP en orden cronológico

| Momento | Explicación | Consecuencia práctica |
|---|---|---|
| 00:16:06 | El TP3 trata sobre localización 2D y utiliza una vista cenital como concepto intermedio. | El problema ocurre sobre un plano. |
| 00:17:50 | Se pide una perspectiva visible para apreciar la corrección homográfica. | Montar la cámara inclinada, pero no de forma rasante. |
| 00:18:20 | Un punto detectado en píxeles debe expresarse en unidades de la mesa. | La salida importante son coordenadas métricas. |
| 00:20:06 | Se define el caso cámara fija, escena fija y registro previo. | Separar registro de localización. |
| 00:21:14 | Se presentan dos homografías distintas. | Una matriz convierte a milímetros y otra a píxeles cenitales. |
| 00:22:53 | El *warp* se incluye por visualización y aprendizaje. | No usar la imagen deformada como fuente principal para volver a detectar. |
| 00:24:47 | La cámara entrega perspectiva y el destino es un plano frontal. | Las cuatro correspondencias relacionan esos dos sistemas. |
| 00:27:42 | Se insiste en que las dos homografías tienen escalas diferentes. | No asumir que un píxel equivale a un milímetro. |
| 00:30:33 | Se explica cómo transportar una grilla mediante la homografía inversa. | Definir la grilla en el plano cenital o métrico y proyectarla. |
| 00:41:24 | Se construyen cuatro correspondencias para calcular `H`. | Mantener el mismo orden de vértices en origen y destino. |
| 00:44:17 | Se recomienda `getPerspectiveTransform`. | Usar la solución directa de cuatro puntos, no `findHomography`. |
| 00:51:16 | Las coordenadas físicas forman parte de la configuración del trabajo. | Origen, ejes, escala y dimensiones deben estar definidos. |
| 00:54:19 | Se introduce el marcador fiduciario. | Sus esquinas proporcionan las correspondencias. |
| 00:58:18 | El tamaño conocido del marcador proporciona la escala. | Para 100 mm, las esquinas centradas utilizan `±50 mm`. |
| 01:06:03 | El docente describe el flujo completo de registro y movimiento posterior. | Calcular `H` una vez y reutilizarla mientras la cámara siga fija. |
| 01:08:32 | Se mencionan la grilla y la vista cenital como elementos adicionales. | Integrarlas en `W2D` sin recalcular el fondo por cuadro. |
| 01:25:48 | Cambiar el zoom invalida la homografía. | Bloquear la configuración de la cámara después del registro. |

## Requisitos formales de la consigna

### Ventana `Cam`

- Mostrar el *feed* de la cámara.
- Dibujar el contorno de los ArUco detectados.
- Mostrar la etiqueta de cada marcador.

### Ventana `W2D`

- Utilizar como fondo una vista cenital estática capturada durante el registro.
- Dibujar el contorno cuadrado del marcador.
- Dibujar una flecha que represente posición y orientación.
- Mostrar coordenadas en milímetros y ángulo.
- Mostrar ejes canónicos centrados en la imagen, con `x` hacia la derecha e `y` hacia arriba.

### Registro mediante la tecla `r`

- No registrar si no se detecta ningún marcador.
- Si hay varios, elegir uno con un criterio consistente.
- Utilizar el marcador para definir el sistema de referencia métrico.
- Calcular la homografía imagen a milímetros.
- Calcular la homografía imagen a visualización.
- Generar una sola vez el fondo cenital mediante `warpPerspective`.

### Localización en tiempo real

- Detectar el marcador en cada cuadro.
- Reutilizar las homografías calculadas durante el registro.
- Transformar puntos mediante `perspectiveTransform`.
- Actualizar ambas ventanas.
- No actualizar la pose de `W2D` si no existen homografías o no hay marcador detectado.

## Aclaraciones importantes para la implementación

- La explicación usa QR como ejemplo, pero la consigna pide ArUco.
- `getPerspectiveTransform` calcula la homografía a partir de cuatro correspondencias.
- `warpPerspective` transforma una imagen completa.
- `perspectiveTransform` transforma puntos y resulta apropiada para el bucle en tiempo real.
- La homografía métrica y la homografía visual no son intercambiables.
- El marcador debe tener un tamaño real conocido.
- La cámara, el zoom y el plano deben permanecer fijos después del registro.
- Mover el marcador no requiere recalcular la homografía.
- Si se altera la instalación, se debe repetir el registro.

## Lista de control para el TP

- [ ] Cámara fija con perspectiva visible y marcador detectable.
- [ ] ArUco impreso o mostrado con tamaño físico conocido.
- [ ] Detección consistente de identificador y cuatro vértices.
- [ ] Registro activado con `r` solamente si existe un marcador.
- [ ] Origen, ejes y escala definidos durante el registro.
- [ ] Cuatro correspondencias en el mismo orden.
- [ ] Dos homografías calculadas con `getPerspectiveTransform`.
- [ ] Fondo cenital generado una sola vez.
- [ ] Puntos transformados en tiempo real sin *warp* por cuadro.
- [ ] `Cam` con contornos y etiquetas.
- [ ] `W2D` con fondo, ejes, cuadrado, flecha, coordenadas y ángulo.
- [ ] Manejo correcto de estados sin registro y sin detección.
- [ ] Nuevo registro si cambia cámara, plano o zoom.
- [ ] El grupo puede explicar el sentido de cada homografía y cada sistema de coordenadas.

## Próximo trabajo anticipado

Cerca de `01:12:01`, el docente anticipa que la clase siguiente volverá al problema de detección y clasificación del TP2, pero utilizando *Deep Learning*. Es otro trabajo y no debe confundirse con el TP actual de localización homográfica.

## Conclusión

La clase convierte la teoría de homografías en una implementación concreta. El sistema registra un plano mediante cuatro puntos conocidos, conserva dos transformaciones distintas y localiza el marcador transformando sus coordenadas directamente. El *warp* cenital ayuda a visualizar el resultado, mientras que la salida realmente útil para robótica es la pose expresada en el sistema métrico del mundo.
