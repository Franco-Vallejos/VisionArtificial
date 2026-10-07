# Resumen de clase de Visión Artificial - 01/10/2026

Fuentes utilizadas: [transcripción](./transcripcion/Clase_20261001_transcripcion.md) y [diapositivas recortadas](./ppt/Clase_20261001_ppt_recortadas.pdf). No se utilizó el video para redactar el contenido.

> La transcripción fue generada automáticamente. En este resumen se normalizaron nombres técnicos que Whisper transcribió fonéticamente, como OpenCV, ArUco, ChArUco, PnP, `solvePnP`, IPPE, Rodrigues y coeficientes de distorsión.

## Idea central de la clase

La clase introduce los conocimientos necesarios para el TP5: calibrar una cámara y obtener la pose 3D de un marcador a partir de correspondencias entre puntos conocidos del objeto y sus posiciones en la imagen.

El recorrido comienza con la distorsión de lente y la calibración. Después amplía el panorama con cámaras industriales, cámaras espectrales, cámaras térmicas, sensores de profundidad y distintos problemas geométricos. La última parte desarrolla el problema PnP y lo conecta directamente con la localización tridimensional de un marcador ArUco o QR.

El concepto que une toda la clase es que una imagen entrega coordenadas 2D, pero para recuperar posición, orientación y escala 3D hace falta incorporar información conocida: la calibración de la cámara, la geometría real del objeto y las correspondencias entre sus puntos 3D y los píxeles detectados.

## 1. Distorsión de lente

Una cámara ideal representa rectas del mundo como rectas en la imagen. La distorsión introduce un apartamiento de esa colineación. Puede resultar poco visible en una fotografía común, pero afecta las mediciones geométricas y la localización precisa.

La clase distingue dos componentes:

- **Distorsión radial:** produce deformaciones simétricas respecto del centro, como barril o cojín. OpenCV suele representarla mediante coeficientes `k1`, `k2`, `k3` y, en modelos ampliados, otros términos.
- **Distorsión tangencial:** aparece por desalineaciones entre lente y sensor. Se representa principalmente mediante `p1` y `p2`.

OpenCV organiza habitualmente los coeficientes como `k1, k2, p1, p2, k3`. El orden importa porque sus valores son posicionales.

### Relación con trabajos anteriores y el TP5 - 00:02:25 a 00:03:42

Para reconocimiento o aprendizaje profundo, una distorsión pequeña puede no ser determinante. Para mediciones métricas sí importa. El docente conecta este punto con la localización del TP3 y aclara que la corrección se vuelve imprescindible para pasar a localización 3D en el TP5.

## 2. Antidistorsión

La antidistorsión aplica una transformación geométrica que compensa el modelo de la lente. OpenCV ofrece funciones para corregir una imagen completa y otras para corregir solamente coordenadas.

La clase remarca una decisión de implementación:

- generar una imagen completa sin distorsión sirve para visualización;
- para cálculos geométricos conviene trabajar sobre la imagen original y corregir los puntos detectados;
- transformar toda la imagen agrega costo y puede perder información por interpolación.

Esta recomendación repite el criterio usado con homografías: cuando el algoritmo necesita puntos, se transforman los puntos; el *warp* completo se reserva para mostrar el resultado.

## 3. Qué significa calibrar una cámara

Calibrar no significa ajustar físicamente la cámara. El procedimiento mide su comportamiento a partir de imágenes y obtiene dos conjuntos de datos:

1. **Matriz de cámara o parámetros intrínsecos.** Incluye distancia focal expresada en píxeles y punto principal.
2. **Coeficientes de distorsión.** Describen la distorsión radial y tangencial de esa cámara y esa lente.

Estos valores son particulares de cada dispositivo. Incluso dos cámaras del mismo modelo pueden producir calibraciones distintas debido a tolerancias de fabricación y montaje.

### Relación con el TP5 - 00:17:45 a 00:19:00

El docente dice explícitamente que PnP necesita la matriz intrínseca y los coeficientes de distorsión. Por lo tanto, la calibración no es teoría adicional: es un requisito previo del TP5. Hay que calibrar la misma cámara con la que se obtendrán las imágenes del sistema final.

## 4. Patrones de calibración

Los patrones más habituales son el tablero de ajedrez y el patrón de círculos. Están diseñados para que OpenCV detecte puntos con un orden conocido.

En el tablero se cuentan las esquinas internas, no los cuadrados. Conviene elegir una cantidad par por una cantidad impar, por ejemplo `6 x 7`, para evitar que el patrón resulte idéntico al rotarlo 180 grados.

El patrón debe permanecer perfectamente plano. Una hoja curvada introduce una deformación que el algoritmo podría atribuir erróneamente a la lente. Se puede pegar la impresión sobre una base rígida o mostrarla en una pantalla plana.

Si se usa un celular o monitor hay que controlar:

- que la cámara observe el patrón completo;
- que el brillo no haga crecer las zonas blancas sobre las negras;
- que no aparezcan franjas o interferencias por la captura de la pantalla;
- que el dispositivo no tenga una superficie curva.

## 5. Capturas necesarias para calibrar

El proceso combina muchas observaciones del patrón en posiciones y orientaciones diferentes. Cada detección aporta coordenadas conocidas que el algoritmo utiliza para minimizar un error global.

La clase menciona dos referencias:

- un procedimiento tradicional puede utilizar 50, 60 o 100 capturas;
- un método que selecciona posiciones informativas puede lograr una calibración óptima con unas 19 tomas.

El número por sí solo no garantiza calidad. Las imágenes deben cubrir:

- el centro y los bordes del encuadre;
- inclinaciones en ambos ejes;
- distintas posiciones del patrón;
- el patrón completo y bien detectado.

Tomar muchas fotografías casi iguales agrega poca información. La diversidad geométrica es más importante que repetir la misma vista.

## 6. Resultado y persistencia de la calibración

El procedimiento devuelve la matriz de cámara y los coeficientes de distorsión. Esos valores se calculan una vez para una configuración de cámara y luego se guardan para utilizarlos en el procesamiento.

La calibración deja de ser válida si cambia la geometría interna de captura, por ejemplo por una modificación relevante del zoom, del foco o de la resolución usada por el modelo de cámara.

## 7. Cámaras y arquitecturas de visión artificial

La clase amplía el panorama de hardware:

- cámaras industriales preparadas para montaje fijo y conexión estable;
- cámaras integradas que ejecutan procesamiento en el propio dispositivo;
- equipos de visión con computadora dedicada;
- sistemas centralizados donde varias cámaras envían imágenes a un servidor;
- procesamiento en la nube para casos que toleran latencia y transferencia de datos.

Las cámaras industriales pueden ofrecer ópticas intercambiables, control del foco, alta tasa de cuadros, sincronización y conexiones robustas. La arquitectura se elige según tiempo real, ancho de banda, costo y condiciones de instalación.

## 8. Cámaras espectrales, térmicas y sintéticas

Una cámara espectral registra bandas diferentes de la luz visible. Las cámaras multiespectrales combinan varias bandas para construir información que una cámara RGB no observa.

Una cámara térmica mide radiación infrarroja asociada con la temperatura. No entrega temperatura de manera perfecta en cualquier condición: emisividad, reflejos y distancia pueden alterar la medición.

La clase denomina imágenes sintéticas a modalidades que no surgen de proyectar radiación visible sobre un sensor convencional, por ejemplo ecografía y tomografía.

## 9. Cámaras de profundidad

Una cámara RGB-D entrega color y profundidad. El mapa de profundidad indica la distancia estimada para cada píxel, aunque su resolución y precisión pueden diferir de la imagen RGB.

La profundidad puede obtenerse mediante:

- paralaje entre cámaras;
- luz infrarroja estructurada;
- proyección de un patrón conocido y análisis de su deformación;
- otras combinaciones de sensores.

La clase aclara que estos equipos miden por triangulación y que la precisión disminuye al aumentar la distancia. La potencia y el alcance del proyector también limitan el rango útil.

## 10. Problemas geométricos

Se presentan problemas más amplios que la actividad inmediata:

- **Structure from Motion:** reconstrucción 3D a partir de imágenes obtenidas desde distintos puntos de vista.
- **Reconstrucción 3D:** construcción de una nube de puntos o superficie.
- **Reconocimiento de lugares:** identificación y geolocalización visual de un sitio.
- **Realidad aumentada:** alineación de objetos virtuales con la escena real.
- **Odometría visual:** estimación del movimiento de la cámara a partir del video.
- **Odometría visual-inercial:** combinación de cámara y sensores inerciales.
- **SLAM visual:** localización simultánea y construcción de un mapa.

El docente aclara que este bloque abarca más contenido que el TP5. Sirve para ubicar PnP y la pose 3D dentro de un conjunto mayor de problemas geométricos.

## 11. Pose 3D y problema PnP

La pose de un cuerpo rígido combina:

- una posición 3D, representada por una traslación;
- una orientación 3D, representada mediante una rotación.

Puede expresarse con una matriz homogénea `4 x 4`, o con un vector de traslación y un vector de rotación en notación de Rodrigues.

PnP significa *Perspective-n-Point*. El problema recibe:

- puntos 3D de un objeto conocido;
- las coordenadas 2D donde aparecen esos puntos en la imagen;
- la matriz intrínseca de la cámara;
- los coeficientes de distorsión.

El resultado es la pose del objeto respecto de la cámara, normalmente mediante `rvec` y `tvec`.

## 12. Correspondencias 3D-2D

Cada punto del modelo debe corresponder al mismo punto detectado en la imagen. El orden de ambos arreglos tiene que coincidir.

Tres correspondencias constituyen el mínimo matemático, pero pueden producir hasta cuatro soluciones. Con cuatro puntos se elimina normalmente la ambigüedad y se obtiene una solución única para el caso de interés.

Un marcador cuadrado resulta conveniente porque:

- tiene cuatro vértices fáciles de definir;
- sus esquinas pueden detectarse automáticamente;
- el detector las devuelve ordenadas;
- todas pertenecen al mismo plano;
- se conoce su tamaño físico.

## 13. Escala y unidades

La imagen no contiene escala métrica por sí sola. La escala se introduce en las coordenadas 3D del modelo.

Si el marcador mide 100 mm y los puntos del objeto se expresan en milímetros, `tvec` estará expresado en milímetros. Si se utilizan centímetros, la salida quedará en centímetros.

Para un marcador cuadrado de lado `L`, centrado en el origen y ubicado en el plano `Z = 0`, sus vértices se construyen con combinaciones de `-L/2` y `L/2`. El algoritmo especializado exige además un orden específico de esos puntos.

## 14. `solvePnP` y elección del algoritmo

OpenCV ofrece varias variantes de PnP. El algoritmo genérico funciona en muchos casos, pero no aprovecha toda la geometría conocida.

Para un ArUco o QR cuadrado y plano, el docente recomienda:

`cv2.SOLVEPNP_IPPE_SQUARE`

IPPE Square está especializado en cuatro puntos coplanares que forman un cuadrado. Por esa razón resulta más eficiente y preciso que una solución general para el caso del TP5.

La llamada a `solvePnP` recibe puntos 3D, puntos 2D, matriz de cámara, coeficientes de distorsión y la bandera del algoritmo. Devuelve un estado de éxito, `rvec` y `tvec`.

## 15. Pipeline de ejecución

El flujo explicado en clase es:

1. Calibrar la cámara y guardar matriz intrínseca y distorsión.
2. Definir una vez las coordenadas 3D del marcador.
3. Leer cada cuadro de la cámara.
4. Detectar las cuatro esquinas 2D del ArUco o QR.
5. Mantener el mismo orden entre puntos 3D y 2D.
6. Ejecutar `solvePnP` con `SOLVEPNP_IPPE_SQUARE`.
7. Obtener rotación y traslación.
8. Mostrar o utilizar la pose calculada.

Una vez obtenida la pose se pueden proyectar puntos 3D sobre la imagen. La clase muestra este recurso para dibujar ejes o un cubo virtual sobre el marcador. Esa anotación permite verificar visualmente que la pose acompaña correctamente la perspectiva.

## 16. Otros marcadores y detectores

La misma idea puede aplicarse a:

- QR;
- ArUco;
- ChArUco;
- patrones de calibración;
- balizas diseñadas a medida;
- objetos con puntos característicos conocidos;
- marcadores virtuales basados en textura.

Cuando se usan muchos puntos, un algoritmo robusto puede descartar correspondencias incorrectas. Sin embargo, esa extensión se presenta como contenido posterior y no como centro del TP5.

---

# TRABAJO PRÁCTICO 5 - PUNTOS IMPORTANTES

## Objetivo confirmado

Obtener en tiempo real la pose 3D de un marcador observado por una cámara. El docente indica que el marcador será ArUco; permite QR como alternativa, pero recomienda ArUco.

La salida central debe incluir la posición y la orientación del marcador respecto de la cámara. El método explicado utiliza calibración de cámara, correspondencias entre las cuatro esquinas del marcador y `solvePnP`.

## Menciones del TP en orden cronológico

| Momento | Explicación del docente | Consecuencia práctica |
|---|---|---|
| 00:02:25 | La distorsión afecta las mediciones métricas y la localización precisa. | La pose 3D debe calcularse usando la calibración de la cámara. |
| 00:03:00 | La calibración y antidistorsión son necesarias para el siguiente nivel de localización, que corresponde al TP5. | No se puede tratar PnP como un cálculo independiente de la cámara. |
| 00:17:45 | La calibración produce matriz intrínseca y coeficientes de distorsión. | Ambos datos deben guardarse y entregarse a `solvePnP`. |
| 00:18:40 | PnP pide explícitamente los parámetros de cámara y distorsión. | Calibrar la misma cámara usada durante la ejecución. |
| 01:04:48 | Los problemas geométricos introducen el TP5, aunque el bloque completo es más amplio que el trabajo. | No todo SfM, SLAM u odometría forma parte de la implementación pedida. |
| 01:15:06 | La presentación de casos de uso empieza a explicar cómo será el TP5. | El núcleo práctico es PnP aplicado a un marcador detectable. |
| 01:19:28 | La práctica se concentrará en marcadores fiduciarios. | No hace falta construir un detector arbitrario de objetos. |
| 01:23:13 | En cada cuadro se detectan puntos 2D que corresponden al modelo 3D conocido. | Mantener correctamente las correspondencias y su orden. |
| 01:25:19 | PnP también necesita la información de calibración. | La matriz y la distorsión forman parte de los datos mínimos. |
| 01:26:44 | Tres correspondencias son el mínimo, pero presentan varias soluciones; normalmente se usan cuatro. | Utilizar las cuatro esquinas del marcador. |
| 01:28:26 | La salida contiene traslación y rotación en Rodrigues. | Procesar o mostrar `tvec` y `rvec`. |
| 01:30:34 | El pipeline comienza con calibración, modelo 3D y detección por cuadro. | Separar datos de configuración del bucle de cámara. |
| 01:37:43 | El marcador del TP es un cuadrado plano de cuatro puntos. | Elegir una variante PnP especializada. |
| 01:38:03 | IPPE Square aprovecha específicamente la geometría cuadrada y coplanar. | Usar `cv2.SOLVEPNP_IPPE_SQUARE`. |
| 01:39:10 | Puntos 3D y 2D deben aparecer en el mismo orden. | Una permutación incorrecta produce una pose equivocada. |
| 01:39:31 | `solvePnP` recibe matriz de cámara, distorsión y bandera del algoritmo. | Pasar explícitamente todos esos argumentos. |
| 01:41:35 | El tamaño real del marcador define la escala. | Medir el lado y expresar `objectPoints` en la unidad deseada. |
| 01:45:26 | Se confirma que el TP5 pide la pose 3D de un ArUco; QR es aceptable, pero ArUco es preferible. | Priorizar ArUco para la solución final. |
| 01:46:22 | El origen del modelo cuadrado se coloca en el centro y `Z = 0`. | Construir vértices mediante `±L/2` con el orden requerido por IPPE Square. |

## Arquitectura mínima sugerida

### Etapa de calibración

1. Mostrar o imprimir un patrón plano.
2. Capturar vistas variadas que cubran el encuadre y distintas inclinaciones.
3. Detectar todas las esquinas internas.
4. Ejecutar la calibración de OpenCV.
5. Guardar la matriz de cámara y los coeficientes de distorsión.

### Etapa de pose en tiempo real

1. Cargar la calibración guardada.
2. Medir el lado real del ArUco.
3. Definir sus cuatro `objectPoints` centrados en el origen y con `Z = 0`.
4. Abrir la cámara y detectar el marcador en cada cuadro.
5. Obtener las cuatro esquinas ordenadas.
6. Ejecutar `solvePnP(..., flags=cv2.SOLVEPNP_IPPE_SQUARE)`.
7. Comprobar el valor booleano devuelto por la función.
8. Mostrar posición y orientación.
9. Proyectar ejes o un cubo para validar visualmente la pose, si se incluye visualización 3D.

## Errores que hay que evitar

- Calibrar una cámara y ejecutar el TP con otra.
- Curvar el patrón de calibración.
- Capturar siempre el patrón en la misma zona o con el mismo ángulo.
- Contar cuadrados en lugar de esquinas internas.
- Usar un patrón simétrico que pueda confundirse al rotarlo 180 grados.
- Cambiar resolución o zoom después de calibrar sin revisar la validez de los parámetros.
- Alterar el orden entre `objectPoints` e `imagePoints`.
- Omitir la medida física del marcador.
- Usar el algoritmo PnP genérico cuando se conoce que el objeto es un cuadrado plano.
- Aplicar un *warp* completo en cada cuadro si solo se necesitan puntos corregidos.

## Lo que se mostró como contexto y no debe confundirse con el TP

- Structure from Motion, SLAM, odometría visual y reconocimiento de lugares son problemas relacionados, pero más amplios.
- Las cámaras térmicas, espectrales y de profundidad amplían el panorama de sensores; no se presentan como hardware obligatorio.
- La pose sin marcador y los métodos robustos con muchos puntos quedan como extensiones posteriores.
- El llavero quirúrgico, la realidad aumentada y las balizas son ejemplos de aplicación.
- Dibujar un modelo 3D complejo no constituye el centro del TP; la prioridad es recuperar correctamente la pose del marcador.

## Checklist del TP5

- [ ] La cámara fue calibrada con la configuración que se usará en ejecución.
- [ ] Se guardaron matriz intrínseca y coeficientes de distorsión.
- [ ] El patrón permaneció plano y apareció completo en las capturas válidas.
- [ ] Las capturas cubren zonas y ángulos variados.
- [ ] Se eligió preferentemente un marcador ArUco.
- [ ] Se midió el lado real del marcador.
- [ ] Los cuatro puntos 3D están centrados, con `Z = 0` y en el orden requerido.
- [ ] Las cuatro esquinas 2D se corresponden con los puntos 3D en el mismo orden.
- [ ] Se usa `cv2.SOLVEPNP_IPPE_SQUARE`.
- [ ] Se verifica que `solvePnP` haya encontrado una solución.
- [ ] Se muestran o utilizan `rvec` y `tvec`.
- [ ] Las unidades de `tvec` coinciden con las usadas para definir el marcador.
- [ ] La pose se actualiza cuadro por cuadro.
- [ ] La visualización permite comprobar que la orientación y posición son coherentes.

## Síntesis final

El TP5 transforma la localización plana del trabajo anterior en una localización tridimensional completa. La calibración describe cómo forma imágenes la cámara; el marcador aporta geometría y escala conocidas; la detección genera los puntos 2D; y PnP recupera rotación y traslación. El éxito depende principalmente de una calibración consistente, correspondencias ordenadas y el algoritmo adecuado para un marcador cuadrado.
