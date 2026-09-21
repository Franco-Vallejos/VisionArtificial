# Resumen de clase de Vision Artificial - 10/09/2026

Fuentes utilizadas: [transcripcion](./transcripcion/Clase_20260910_transcripcion.md), [diapositivas recortadas](./ppt/Clase_20260910_ppt_recortadas.pdf) y la consigna [TP Localizacion homografica](C:/Users/fnval/Downloads/TP%20Localizaci%C3%B3n%20homogr%C3%A1fica.pdf). No se utilizo el video.

## Idea central de la clase

La clase introduce la vision geometrica necesaria para relacionar una imagen en perspectiva con un sistema de coordenadas del mundo real. El recorrido teorico va desde transformaciones lineales, afines y coordenadas homogeneas hasta homografias, modelo de camara y pose.

La aplicacion que conecta toda la clase es la siguiente: una camara fija observa un plano inclinado respecto de la imagen; el sistema detecta un objeto o marcador en pixeles y, mediante una homografia, calcula su posicion en unidades metricas dentro de ese plano. Esta es tambien la base del trabajo practico de localizacion homografica.

## 1. Motivacion: vision geometrica y localizacion

La clase comienza con ejemplos de asistencia a la conduccion y navegacion de robots. El docente remarca que detectar un objeto en una imagen solo proporciona coordenadas en pixeles. Para tomar decisiones en el mundo real se necesita conocer distancias y posiciones en metros, centimetros o milimetros.

Los marcadores fiduciarios ArUco resultan utiles porque:

- estan disenados para ser detectados con facilidad;
- cada marcador tiene un identificador;
- sus vertices y orientacion pueden localizarse en la imagen;
- su tamano real conocido aporta la escala que una fotografia por si sola no contiene.

En el ejemplo de una pista observada por una camara inclinada, las posiciones conocidas de marcadores o vertices permiten registrar el plano. A partir de ese registro se transforma la posicion en pixeles del robot a coordenadas metricas de la pista.

### Conexion con el TP - 00:09:25 a 00:10:05

Esta es la primera mencion explicita del trabajo practico. El docente anticipa que el problema sera parecido al del robot, pero simplificado: en lugar de controlar un robot se movera manualmente un marcador sobre un plano. El objetivo sera convertir su posicion detectada en la imagen a una posicion del mundo real.

## 2. Transformaciones lineales, afines y euclidianas

Una transformacion lineal tiene la forma `y = A x`. Puede rotar, escalar o deformar, pero mantiene el origen fijo.

Una transformacion afin agrega una traslacion:

`y = A x + b`

La rototraslacion es un caso particular importante: combina una rotacion con una traslacion. Cuando no hay escala ni deformacion, conserva las distancias relativas entre los puntos y se considera una transformacion euclidiana o isometria.

Las coordenadas homogeneas permiten representar la traslacion dentro de una unica multiplicacion matricial. Un punto 2D `(x, y)` se escribe como `(x, y, 1)` y una transformacion afin pasa a expresarse mediante una matriz `3 x 3`.

Conceptos importantes:

- una matriz afin 2D tiene seis parametros efectivos;
- la ultima fila normalizada es `(0, 0, 1)`;
- las transformaciones se pueden componer multiplicando matrices;
- el orden de las transformaciones importa;
- la transformacion inversa permite volver del sistema destino al sistema de origen.

## 3. Coordenadas homogeneas y normalizacion

En coordenadas homogeneas, multiples vectores proporcionales representan el mismo punto. Por ejemplo, `(x, y, 1)` y `(2x, 2y, 2)` describen la misma posicion proyectiva.

Luego de aplicar una homografia, el resultado generalmente no queda normalizado. Si se obtiene `(u, v, w)`, hay que deshomogeneizar:

- `x = u / w`
- `y = v / w`

Esta division final es indispensable cuando se implementa la operacion manualmente. Algunas funciones de OpenCV, como `perspectiveTransform`, realizan internamente la expansion homogenea, la multiplicacion y la normalizacion.

## 4. Homografia y transformacion de perspectiva

Una homografia es una matriz `3 x 3` que relaciona dos representaciones proyectivas de un mismo plano. Una transformacion afin es un caso particular de homografia; los elementos adicionales de la ultima fila permiten modelar el efecto de perspectiva.

La homografia puede utilizarse de dos maneras:

1. Transformar una imagen completa mediante un *warp* y generar una vista frontal o cenital.
2. Transformar solamente puntos, por ejemplo el centro o los vertices de un marcador, desde coordenadas de imagen hacia otro sistema de coordenadas.

La transformacion no recupera informacion que la camara nunca observo. Las zonas negras de una vista rectificada representan regiones para las que no existen pixeles de origen. Ademas, la homografia es valida para el plano registrado: los objetos que sobresalen de ese plano pueden quedar deformados.

### Conexion con el TP - 00:36:42 a 00:37:14

El docente relaciona la vista cenital con el trabajo de ArUco. Aclara que generar una imagen cenital es util para visualizar y depurar, pero el resultado importante para la aplicacion es obtener coordenadas metricas del marcador dentro del sistema de referencia del mundo.

### Ejemplo de la mesa de billar - 00:38:45 y 01:35:01

La mesa observada en perspectiva se transforma en una vista cenital. El mismo concepto permite tomar una bola detectada en pixeles y calcular donde se encuentra sobre la mesa en centimetros. El ejemplo puede resolverse con un clic del usuario o con una deteccion automatica previa.

Este ejemplo anticipa directamente el TP: el marcador ArUco reemplaza a la bola y evita tener que desarrollar un detector de objetos adicional.

## 5. Modelo de camara, sistemas de coordenadas y pose

La clase distingue varios sistemas de referencia:

- imagen o pixeles;
- camara;
- mundo;
- sistema local asociado a un objeto o cuerpo rigido.

La matriz de camara relaciona puntos 3D con su proyeccion en el plano de imagen. Sus parametros intrinsecos describen propiedades internas de la camara, mientras que los parametros extrinsecos describen su pose respecto del mundo.

La pose combina posicion y orientacion. En 3D tiene seis grados de libertad: tres de traslacion y tres de rotacion. Puede representarse mediante una matriz homogenea `4 x 4` que contiene una submatriz de rotacion, un vector de traslacion y la fila `(0, 0, 0, 1)`.

La clase tambien introduce distintas formas de expresar rotaciones, entre ellas angulos de Euler y el vector de Rodrigues. Estos conceptos son mas generales que el TP inmediato, que trabaja con pose 2D sobre un plano.

## 6. Escala y sistema de referencia metrico

Una imagen aislada no permite determinar la escala absoluta: visualmente no se puede saber si una escena es grande o una maqueta. Para obtener milimetros o centimetros hace falta incorporar una medida conocida.

En el TP, esa informacion proviene del tamano real del marcador ArUco. Al registrar el marcador se fija simultaneamente:

- el origen del sistema de referencia;
- la orientacion de los ejes;
- la escala entre pixeles y unidades metricas.

Mientras la camara y el plano permanezcan fijos, la homografia calculada durante el registro no cambia aunque luego se mueva el marcador.

## 7. Eficiencia: transformar imagenes o transformar puntos

`warpPerspective` transforma todos los pixeles y puede ser costoso, especialmente en equipos limitados. Para localizar un marcador no es necesario rectificar cada cuadro completo. Es mas eficiente detectar el marcador en la imagen original y aplicar la homografia solo a los puntos relevantes.

El docente menciona `perspectiveTransform` para transformar muchas coordenadas juntas. La funcion recibe los puntos y la homografia, trabaja internamente en coordenadas homogeneas y devuelve nuevamente coordenadas 2D.

### Conexion con el TP - 01:42:14 a 01:43:47

Esta explicacion define la arquitectura correcta:

- el *warp* de la imagen se usa para producir una referencia visual;
- la localizacion en tiempo real transforma puntos, no la imagen completa;
- no se debe ejecutar `warpPerspective` innecesariamente en cada cuadro.

La consigna formal confirma esta decision: la vista cenital de fondo se captura durante el registro y queda estatica.

---

# TRABAJO PRACTICO - LOCALIZACION HOMOGRAFICA

## Objetivo

Desarrollar una demostracion de localizacion 2D en tiempo real. Una camara fija observa en perspectiva un plano por el que se mueve un marcador ArUco. El sistema debe mostrar la posicion y orientacion 2D del marcador dentro de un sistema de referencia metrico registrado.

El ejemplo propuesto es un escritorio y un marcador ArUco de `100 mm` de lado, aunque tambien puede utilizarse una pantalla que muestre un marcador movil.

## Menciones del TP en orden cronologico

| Momento | Que explica el docente | Consecuencia para la implementacion |
|---|---|---|
| 00:09:25 - 00:10:05 | Anticipa un trabajo parecido a la localizacion del robot, reemplazando el robot por un marcador movido con la mano. | La entrada es una camara fija y el objeto movil es un ArUco. |
| 00:36:42 - 00:37:14 | Relaciona el TP con la generacion de una vista cenital, pero aclara que interesan especialmente las coordenadas metricas. | La vista cenital es una ayuda visual; la salida esencial es la pose en unidades reales. |
| 01:35:01 - 01:36:09 | Usa la mesa de billar para explicar el paso de un punto detectado en pixeles a centimetros del plano. | El centro o los vertices detectados deben transformarse mediante la homografia metrica. |
| 01:42:14 - 01:43:47 | Diferencia el *warp* de imagen de la transformacion de puntos y menciona `perspectiveTransform`. | En el bucle se transforman coordenadas; no hace falta rectificar cada cuadro. |
| 01:44:20 - 01:45:02 | Explica que, con camara fija, `H` se calcula una vez y permanece constante. | La homografia se obtiene durante una operacion discreta de registro. |
| 01:45:19 - 01:46:08 | Describe una camara fija, un escritorio y un ArUco impreso que se mueve con el dedo; deben verse sus coordenadas metricas. | Es el planteo concreto de la consigna. |
| 01:46:08 - 01:47:18 | Remarca que el TP esta organizado paso a paso y que el objetivo es comprender el procedimiento, no solo obtener un programa resuelto. | Hay que poder explicar registro, homografia, escala, transformacion y localizacion. |
| 01:47:18 - 01:48:46 | Vuelve a definir el objetivo como convertir pixeles a coordenadas metricas de un sistema arbitrario sobre el plano. | Deben definirse origen, ejes y unidades del mundo 2D. |
| 01:49:43 - 01:50:33 | El marcador fiduciario define origen, orientacion y escala; al pulsar una tecla se calcula la homografia y luego el marcador puede moverse. | Coincide con el registro mediante la tecla `r` pedido por la consigna. |
| 01:50:48 - 01:51:22 | Relaciona el problema con el pipeline de contornos del TP2. | La deteccion previa produce puntos; desde ahi comienza la etapa geometrica. Con ArUco no hace falta clasificar formas manualmente. |
| 01:53:19 - 01:54:05 | Aun una camara casi cenital requiere registro; en el TP se pedira una perspectiva marcada para mostrar el efecto. | Conviene montar la camara claramente inclinada, fija y con todo el plano visible. |
| 01:54:15 - 01:55:07 | Formula el ejemplo generico: hacer clic en una bola y devolver su posicion en centimetros. | Es la misma operacion matematica que localizar el marcador. |
| 01:55:07 - 01:56:02 | Distingue dos homografias: una genera la vista cenital en pixeles y otra entrega coordenadas en centimetros. | No reutilizar ciegamente una unica `H`; los sistemas destino y sus escalas son diferentes. |

### Nota sobre el numero del TP

El docente lo llama mayormente **TP3**. Cerca de `01:48:46` aparece una correccion verbal confusa entre TP3 y TP4, pero enseguida vuelve a identificar como TP3 el trabajo inmediato. La consigna entregada no muestra numero y lo titula **TP: Localizacion homografica**. Para evitar ambiguedad, este resumen usa ese titulo.

## Requisitos formales confirmados por la consigna

### Ventana `Cam`

Debe mostrar el *feed* de la camara y anotar:

- contorno de los marcadores ArUco detectados;
- etiqueta de cada marcador.

### Ventana `W2D`

Debe mostrar una representacion cenital del mundo 2D con:

- una flecha que indique posicion y orientacion del ArUco;
- el contorno cuadrado del marcador;
- coordenadas en milimetros;
- angulo de orientacion;
- ejes canonicos centrados en la imagen: `x` hacia la derecha e `y` hacia arriba;
- una imagen cenital estatica de fondo, capturada durante el registro.

### Operacion de registro

Se ejecuta al pulsar `r` y debe:

1. Comprobar que haya al menos un ArUco detectado.
2. Elegir uno si hay varios; la consigna permite cualquier criterio.
3. Usarlo para definir el sistema de referencia metrico del mundo 2D.
4. Calcular la homografia de imagen a coordenadas en milimetros.
5. Calcular otra homografia de imagen a coordenadas de la visualizacion cenital.
6. Generar y guardar la vista cenital estatica mediante `warpPerspective`.

Si no se detecta ningun marcador, el registro no debe realizarse.

### Operacion de localizacion

Se ejecuta continuamente y debe:

1. Detectar los ArUco del cuadro actual.
2. Dibujar contorno e identificador en `Cam`.
3. Transformar los puntos relevantes con las homografias registradas.
4. Calcular posicion y orientacion 2D.
5. Actualizar la flecha, el cuadrado, las coordenadas y el angulo en `W2D`.

`W2D` no puede actualizar la pose si todavia no existen homografias o si en el cuadro actual no se detecta ningun marcador.

## Distinciones que hay que poder explicar en la defensa

- **Pixeles frente a milimetros:** son sistemas distintos y necesitan una transformacion con escala conocida.
- **Registro frente a localizacion:** el registro calcula referencias y homografias una vez; la localizacion usa esos resultados continuamente.
- **Homografia metrica frente a homografia visual:** tienen destinos y escalas diferentes.
- **`warpPerspective` frente a `perspectiveTransform`:** la primera transforma una imagen; la segunda transforma puntos.
- **Vista cenital frente a informacion nueva:** rectificar no permite ver zonas ocultas ni reconstruir datos inexistentes.
- **Plano frente a objetos fuera del plano:** la homografia modela correctamente el plano registrado; otros objetos pueden deformarse.
- **Camara fija:** si la camara se mueve despues del registro, las homografias dejan de representar la escena y hay que registrar nuevamente.
- **Tamano conocido del ArUco:** proporciona la escala metrica que la imagen no contiene por si sola.

## Lista de control del TP

- [ ] La camara permanece fija y observa el plano con perspectiva visible.
- [ ] Se detectan identificador, cuatro vertices y orientacion del ArUco.
- [ ] La tecla `r` registra solamente cuando existe un marcador detectado.
- [ ] El marcador de registro define origen, ejes y escala metrica.
- [ ] Se calculan dos homografias independientes.
- [ ] El fondo cenital se genera una sola vez durante el registro.
- [ ] El bucle en tiempo real transforma puntos y no ejecuta un *warp* completo por cuadro.
- [ ] `Cam` muestra contornos y etiquetas.
- [ ] `W2D` muestra fondo estatico, ejes, cuadrado, flecha, coordenadas en mm y angulo.
- [ ] `W2D` maneja correctamente los estados sin registro o sin marcador visible.
- [ ] El grupo puede explicar por que la homografia es valida para un plano y por que depende de una camara fija.

## Conclusion

El TP convierte los conceptos geometricos de la clase en un flujo concreto: detectar un marcador en pixeles, registrar un plano metrico, calcular dos homografias y localizar en tiempo real la pose 2D del marcador. La vista cenital permite ver y depurar el resultado, pero la informacion central para una aplicacion robotica son las coordenadas y la orientacion expresadas en el sistema de referencia del mundo.
