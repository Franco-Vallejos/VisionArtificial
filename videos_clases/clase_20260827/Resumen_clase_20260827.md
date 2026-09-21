# Resumen de clase de Visión Artificial — 27/08/2026

Fuentes utilizadas: [transcripción](./Clase_20260827_transcripcion.md) y [diapositivas recortadas](./Clase_20260827_ppt_recortadas.pdf). No se utilizó el video.

## Idea central de la clase

La clase presenta el **TP2: detección y clasificación de formas** y desarrolla la primera parte de su pipeline. El recorrido comienza con una imagen de cámara, continúa con segmentación binaria y extracción de contornos, y termina con un descriptor de siete valores basado en los invariantes de Hu. La construcción del dataset y el entrenamiento del clasificador quedan para la clase siguiente.

## 1. Detección, clasificación y segmentación

La clase diferencia tres tareas de visión artificial:

- **Detección:** localiza un objeto en la imagen mediante coordenadas, un punto, un rectángulo u otra región.
- **Clasificación:** asigna el objeto a una categoría perteneciente a una lista finita y predefinida.
- **Segmentación:** clasifica cada píxel. Permite localizar el objeto con más precisión, aunque suele requerir más procesamiento que una detección.

El TP2 combina estas ideas: segmenta las figuras, obtiene sus contornos, extrae un descriptor y finalmente clasifica cada forma.

## 2. Pipeline heurístico

El docente presenta el trabajo como un pipeline heurístico, es decir, una secuencia de operaciones elegidas y programadas de forma explícita. Esto contrasta con un modelo entrenable, que ajusta su comportamiento a partir de datos.

Pipeline general introducido en la clase:

1. Capturar la imagen.
2. Convertirla a escala de grises.
3. Generar una imagen binaria mediante un umbral.
4. Extraer los contornos.
5. Seleccionar los contornos relevantes.
6. Calcular momentos e invariantes de Hu.
7. Usar el descriptor resultante para clasificar la forma. El clasificador se desarrolla en la clase siguiente.
8. Anotar en la imagen el contorno y la categoría reconocida.

Aunque el ejercicio utiliza algoritmos básicos, la estructura del pipeline es general. Cada etapa puede reemplazarse más adelante por un método más potente sin cambiar la lógica global del sistema.

## 3. Imágenes binarias y máscaras

Una imagen binaria contiene dos estados, normalmente representados como negro y blanco. En las operaciones posteriores, el objeto de interés debe quedar en blanco y el fondo en negro.

Una máscara binaria selecciona qué zona de una imagen se conserva o procesa. Las regiones blancas representan la parte activa de la máscara y las negras la parte descartada.

En el escenario previsto para el TP2, las figuras se dibujan con marcador negro sobre una hoja blanca. Por eso será necesario usar una binarización invertida para obtener las figuras blancas sobre fondo negro.

## 4. Umbralización

La función de umbral o `threshold` compara cada píxel de una imagen monocromática con un valor. Según el resultado, asigna negro o blanco.

Puntos importantes:

- Primero se convierte la imagen a escala de grises. Aplicar directamente un umbral a los tres canales de una imagen a color genera combinaciones de colores que no sirven para este TP.
- El valor del umbral determina qué intensidades pertenecen al fondo y cuáles al objeto.
- El docente recomienda colocar una barra deslizante o *trackbar* para ajustar el umbral manualmente y observar el resultado en tiempo real.
- Conviene mostrar las imágenes intermedias del pipeline: original, escala de grises y binaria. Esto facilita la calibración y el diagnóstico.

### Histogramas y umbral automático

El histograma cuenta cuántos píxeles existen para cada intensidad. En una imagen bimodal, un umbral razonable se ubica en el valle entre los dos picos.

Se mencionan los métodos automáticos **Otsu** y **Triangle**. Funcionan bien cuando la distribución de intensidades cumple sus supuestos, pero pueden fallar con histogramas ambiguos o con más de dos modos. El docente prefiere el ajuste manual para este ejercicio porque permite comprender cuándo la segmentación funciona.

### Umbral adaptativo

El umbral adaptativo calcula un valor local a partir de los píxeles vecinos. Puede servir cuando la iluminación cambia dentro de la imagen, pero requiere más procesamiento. Se presentan variantes basadas en promedio local y en ponderación gaussiana.

Para el TP2 se pide trabajar en un **ambiente controlado**, por lo que no debería ser necesario utilizar umbral adaptativo.

## 5. Ambiente controlado

Un ambiente controlado mantiene estables la iluminación, el fondo, la posición general de la cámara y las condiciones de captura. El objetivo es producir imágenes fáciles de segmentar de manera consistente.

Para el TP2 esto implica:

- Usar una hoja blanca y un marcador negro.
- Ajustar la iluminación para evitar sombras intensas, reflejos y cambios de brillo.
- Si la escena queda oscura, agregar o reposicionar una fuente de luz.
- Mantener condiciones similares entre las muestras usadas para preparar el sistema y las empleadas en la demostración.

Controlar la captura forma parte de la solución de ingeniería. Reduce la necesidad de aplicar algoritmos más complejos y menos previsibles.

## 6. Blobs y componentes conectados

Un *blob* es una mancha formada por píxeles conectados. Representa un primer nivel de interpretación: el sistema deja de tratar todos los píxeles como elementos aislados y reconoce conjuntos separados.

La conectividad puede definirse de dos maneras:

- **4 conexiones:** cada píxel se conecta con sus vecinos laterales.
- **8 conexiones:** también se consideran conectados los vecinos diagonales.

Los algoritmos de componentes conectados pueden asignar una etiqueta distinta a cada blob. La clase presenta el concepto, pero para el TP se utilizarán contornos porque ofrecen una representación más compacta y práctica.

## 7. Contornos

Un contorno es una secuencia de coordenadas que describe el límite de una región. Contiene información equivalente a la mancha para muchas operaciones, pero requiere procesar menos datos que todos sus píxeles internos.

La función `findContours` recibe una imagen binaria y devuelve un conjunto de contornos. Cada contorno es, a su vez, un conjunto de puntos. La función puede devolver además información jerárquica para indicar qué contornos contienen huecos u otros contornos.

Aspectos importantes:

- Los contornos se extraen de una imagen binaria bien segmentada.
- Puede solicitarse una aproximación que reemplace largas secuencias de puntos por segmentos y reduzca el uso de memoria.
- La jerarquía permite distinguir bordes exteriores, huecos y regiones anidadas.
- Para este TP, la jerarquía completa no parece necesaria, pero se debe elegir correctamente el modo de recuperación o filtrar después los contornos.

### Cuidado con los contornos externos

Si se solicita únicamente el contorno más externo, el algoritmo podría devolver el borde de la hoja en lugar de las figuras dibujadas dentro de ella. El grupo debe decidir qué modo de recuperación usar y cómo descartar la hoja, el fondo u otros contornos irrelevantes.

### Medidas y anotaciones posibles

A partir de un contorno se pueden obtener:

- Área y perímetro.
- Rectángulo delimitador horizontal o rotado.
- Envolvente convexa mediante `convexHull`.
- Concavidades o defectos de convexidad.
- Distancia entre un punto y el contorno con `pointPolygonTest`.
- Representación visual mediante `drawContours`.

Estas herramientas sirven para localizar, medir, filtrar y anotar objetos. No todas son obligatorias para el TP2.

## 8. Descriptores de forma

Un descriptor resume una entidad mediante un vector numérico. Para reconocer formas, el descriptor debe cambiar principalmente cuando cambia la forma y mantenerse estable ante transformaciones que no deberían alterar su categoría.

Para el TP se buscan tres invariancias:

- **Traslación:** la posición de la figura no debe modificar el descriptor.
- **Escala:** una versión más grande o más pequeña debe conservar valores comparables.
- **Rotación:** girar la figura no debe cambiar su categoría.

El tamaño, el perímetro o el color pueden servir en otros problemas, pero no cumplen necesariamente todas estas invariancias.

## 9. Momentos e invariantes de Hu

Los momentos condensan propiedades geométricas de una región:

- El momento espacial de orden cero representa el área.
- Los momentos de primer orden permiten calcular el centroide.
- Los momentos centrales toman el centroide como referencia y aportan invariancia frente a la traslación.
- Los momentos centrales normalizados incorporan invariancia frente a la escala.
- Los **siete invariantes de Hu** combinan momentos normalizados para obtener un descriptor que también resulta estable frente a la rotación.

En OpenCV, `moments` calcula los momentos de un contorno y `HuMoments` genera los siete valores finales. Calcularlos a partir del contorno resulta más eficiente que hacerlo sobre todos los píxeles de la imagen binaria.

Los siete valores deben interpretarse como un vector completo, no como mediciones independientes fáciles de leer. Su función es alimentar un clasificador que aprenda a separar las categorías en un espacio de siete dimensiones.

La reflexión tiene un comportamiento especial: seis invariantes se mantienen y uno cambia de signo. Este detalle no es central para el objetivo del TP2.

## 10. Operaciones morfológicas

La clase cierra con una introducción a operaciones morfológicas sobre imágenes binarias:

- **Dilatación:** expande las regiones blancas.
- **Erosión:** reduce las regiones blancas.
- **Apertura:** erosión seguida de dilatación. Elimina puntos blancos pequeños.
- **Cierre:** dilatación seguida de erosión. Rellena pequeños huecos negros.

El efecto depende del elemento estructural utilizado. Estas operaciones pueden limpiar ruido, separar objetos unidos o cerrar discontinuidades.

El docente aclara que probablemente no sean necesarias en el TP2 porque se trabajará con imágenes generadas por los propios estudiantes y en condiciones controladas. Deben usarse solo si solucionan un problema concreto observado en la imagen binaria.

---

# TRABAJO PRÁCTICO 2 — PUNTOS IMPORTANTES

## Objetivo general

Crear un sistema capaz de observar formas dibujadas, localizar sus contornos, obtener un descriptor mediante los siete invariantes de Hu y, después del entrenamiento explicado en la clase siguiente, indicar a qué categoría pertenece cada forma.

## Alcance de esta clase

El docente indica que con los contenidos del 27/08 se puede realizar aproximadamente el **80 % inicial del TP2**. La última parte, vinculada con el dataset y el clasificador entrenable, se completa en la clase siguiente.

## Pipeline que se puede implementar con esta clase

1. Capturar la escena con la cámara.
2. Convertir la imagen a escala de grises.
3. Aplicar `threshold` invertido para obtener figuras blancas sobre fondo negro.
4. Ajustar el umbral manualmente con una barra deslizante.
5. Mostrar las imágenes intermedias para comprobar cada etapa.
6. Extraer todos los contornos.
7. Filtrar el borde de la hoja, el fondo y cualquier contorno no deseado.
8. Calcular los momentos del contorno.
9. Obtener los siete invariantes de Hu.
10. Imprimir o almacenar el vector junto con la forma observada para preparar el futuro dataset.
11. Dibujar el contorno y, cuando se incorpore el clasificador, escribir la categoría reconocida.

## Condiciones de las muestras

- Las formas deben dibujarse a mano sobre una hoja blanca con marcador negro.
- Conviene que los dibujos no sean geométricamente perfectos. El sistema debe reconocer variaciones reales de una misma clase.
- Deben probarse cambios de posición, tamaño y rotación para comprobar el comportamiento de los invariantes.
- Las categorías las define el grupo. En la explicación se muestran cuadrados, triángulos, estrellas y otras formas, y se menciona un ejemplo de cinco categorías. Conviene confirmar en el enunciado escrito si cinco es una cantidad obligatoria.

## Recomendaciones técnicas

- Mantener el objeto de interés en blanco y el fondo en negro en la imagen binaria.
- No aplicar `threshold` directamente sobre la imagen a color.
- Priorizar un ambiente controlado y un umbral manual antes de sumar métodos automáticos o adaptativos.
- Verificar visualmente cada etapa del pipeline en ventanas separadas.
- No confiar ciegamente en `RETR_EXTERNAL`: puede recuperar solo el borde de la hoja.
- Filtrar los contornos por criterios coherentes, como área o ubicación, si aparecen regiones que no corresponden a las figuras.
- Calcular momentos e invariantes sobre cada contorno individual.
- Conservar siempre el mismo orden para los siete invariantes de Hu.
- Incorporar morfología solamente si existe ruido o un defecto de segmentación concreto.

## Qué queda para la clase siguiente

- Construir un dataset con muchos descriptores y sus categorías correctas.
- Entrenar un clasificador con esos datos.
- Guardar el modelo entrenado.
- Integrar el modelo al programa final para convertir cada descriptor en una categoría.

## Entrega y defensa

Al comienzo se explica la modalidad utilizada para los trabajos prácticos:

- La entrega se realiza compartiendo pantalla durante la clase.
- El grupo debe demostrar el funcionamiento del programa.
- Debe explicar las decisiones tomadas y responder las preguntas de los docentes.
- Esta clase no establece con claridad una fecha específica de entrega para el TP2. La fecha se comunica en la clase siguiente.

## Lista de control para avanzar

- [ ] La cámara produce una imagen estable y bien iluminada.
- [ ] Se visualizan la imagen original, la versión monocromática y la binaria.
- [ ] Una barra deslizante permite ajustar el umbral.
- [ ] Las figuras quedan blancas sobre fondo negro.
- [ ] El programa obtiene contornos separados para las figuras.
- [ ] El borde de la hoja y otros contornos irrelevantes se descartan.
- [ ] Se calculan los momentos de cada contorno.
- [ ] Se obtienen los siete invariantes de Hu en un orden constante.
- [ ] Se prueban traslaciones, rotaciones y cambios de escala.
- [ ] Se guardan o imprimen los descriptores para comenzar a construir el dataset.
- [ ] El grupo puede explicar la diferencia entre segmentación, detección y clasificación.

## Relación con la clase del 03/09

Esta clase construye la parte visual y el descriptor del TP2. La clase siguiente toma esos siete valores, explica cómo formar el dataset, entrenar un árbol de decisión y completar el clasificador final.
