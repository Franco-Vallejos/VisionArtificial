# Resumen de clase de Visión Artificial — 03/09/2026

Fuentes utilizadas: [transcripción](./Clase_20260903_transcripcion.md) y [diapositivas recortadas](./Clase_20260903_ppt_recortadas.pdf). No se utilizó el video.

## Idea central de la clase

La clase introduce los fundamentos de *Machine Learning* necesarios para completar el TP2. El foco principal está en construir un clasificador de formas a partir de los siete invariantes de Hu obtenidos de contornos. Luego comienza un segundo bloque sobre transformaciones geométricas de imágenes, interpolación y coordenadas homogéneas, que continuará en la clase siguiente.

## 1. Machine Learning y sus problemas principales

*Machine Learning* es el conjunto de técnicas que ajustan un modelo a partir de ejemplos o datos. El modelo no se programa caso por caso: durante el entrenamiento ajusta parámetros internos para aprender una regla general.

La clase distingue tres problemas:

- **Clasificación:** produce una categoría discreta. Es el problema que se aplica directamente en el TP2.
- **Regresión:** produce un número real. Se presenta como base conceptual y como antecedente para redes neuronales, pero no es lo que se implementará en el TP2.
- **Clustering:** agrupa muestras similares sin etiquetas previas. Es aprendizaje no supervisado y se presenta para mostrar que Machine Learning abarca más que clasificación y regresión.

También se ubica a *Deep Learning* como una especialización dentro de Machine Learning. Más adelante la materia retomará la clasificación con redes profundas y con imágenes como entrada.

## 2. Conceptos básicos

### Muestra, descriptor y espacio de descripción

Una muestra es el conjunto de características que describe un caso. En el contexto del TP2, cada muestra es un vector formado por los **siete invariantes de Hu**, siempre en el mismo orden. Cada invariante representa una dimensión del espacio de descripción.

### Etiquetas y categorías

En clasificación, la salida es una categoría. Las categorías se codifican mediante números enteros y se relacionan con nombres mediante un diccionario de etiquetas. La distancia numérica entre dos etiquetas no tiene significado: que una clase sea 1 y otra 2 no implica que sean parecidas.

### Entrenamiento e inferencia

- **Entrenamiento:** el modelo recibe muestras junto con la respuesta correcta y ajusta sus parámetros.
- **Inferencia o predicción:** el modelo ya entrenado recibe un descriptor nuevo y devuelve la categoría que considera más probable.
- El archivo con los parámetros aprendidos pertenece al modelo y a la configuración usados durante el entrenamiento. Si cambia la cantidad o el significado de las características, se debe entrenar nuevamente.

## 3. Dataset y evaluación

Un dataset es una colección de muestras. Para el TP2, cada fila debe contener:

1. Los siete invariantes de Hu de una forma.
2. Una octava columna con el entero que identifica la categoría correcta.

El docente usa como ejemplo un conjunto de formas como cuadrados, triángulos y estrellas. También menciona, a modo ilustrativo, cinco categorías y unas cincuenta imágenes. La transcripción no permite confirmar que esas cantidades sean requisitos rígidos, por lo que conviene verificar la consigna escrita del TP2.

Como regla general se propone separar aproximadamente:

- **90 % para entrenamiento.**
- **10 % para prueba.**

La prueba debe usar muestras que no hayan participado del entrenamiento. Evaluar con las mismas muestras puede ocultar el sobreajuste.

### Calidad de los datos

La selección de características y la construcción del dataset constituyen la parte más valiosa del proceso. El modelo solo ve números, por lo que depende de que las características elegidas sean discriminantes y de que los ejemplos representen la variedad real del problema.

Un dataset con muchas copias casi iguales enseña muy poco. Debe incluir casos variados y extremos. Los errores de etiquetado generan *outliers* y pueden forzar al modelo a aprender relaciones incorrectas. Algunos algoritmos son más robustos que otros frente a esos errores.

### Sobreajuste y generalización

El *overfitting* aparece cuando el modelo memoriza el conjunto de entrenamiento en lugar de aprender una regla general. Puede rendir muy bien con datos conocidos y fallar con muestras nuevas.

Las medidas explicadas para reducirlo incluyen aumentar la cantidad y diversidad de datos, reservar datos independientes para evaluar y, en modelos que lo admiten, reducir la tasa de aprendizaje. Cada recorrido completo del dataset durante el entrenamiento se denomina **época**.

## 4. Modelos explicados

### Regresión y perceptrones multicapa

La regresión aproxima una función cuyo resultado es un número real. Se usa para introducir el ajuste insuficiente, el buen ajuste y el sobreajuste. También se presenta el perceptrón: combina entradas mediante pesos, suma los resultados y aplica una función. Una red multicapa conecta muchos perceptrones y aprende ajustando sus pesos.

### Clasificación y árbol de decisión binario

Un clasificador aprende fronteras que separan categorías en el espacio de características. Para el TP2 se recomienda un **árbol de decisión binario** porque entrena rápido, es sencillo de utilizar y muestra con claridad el mecanismo de clasificación.

En cada nodo, el árbol selecciona una característica y la compara con un umbral. Según el resultado, sigue una rama hasta llegar a una categoría. La característica y el umbral de cada nodo surgen del entrenamiento.

### Clustering

El clustering busca grupos de descriptores similares sin conocer previamente sus identidades. Un ejemplo sería agrupar apariciones de un mismo rostro en grabaciones de seguridad aun cuando no se conozca el nombre de la persona.

## 5. Uso responsable de Machine Learning

El modelo puede reproducir los sesgos presentes en el dataset. La clase analiza ejemplos de clasificación de imágenes y de decisiones crediticias para mostrar que un sistema puede aprender discriminaciones históricas sin que nadie programe explícitamente esa regla.

La conclusión práctica es que definir las muestras, las etiquetas y la variedad del dataset transfiere conocimiento humano al modelo, pero también puede transferir errores y prejuicios. La calidad del resultado depende directamente de esas decisiones.

## 6. Segundo bloque: transformaciones geométricas

Una transformación geométrica o *warp* cambia la ubicación espacial asociada a los píxeles. Para generar la imagen destino se recorre cada coordenada de salida y se busca de qué coordenada de la imagen original debe obtenerse el color. Este mapeo inverso evita huecos y funciona aunque las imágenes de entrada y salida tengan tamaños distintos.

Casos presentados:

- **Resize:** cambio de tamaño mediante factores de escala.
- **Remap:** permite definir una deformación arbitraria mediante un mapa que indica, para cada píxel de salida, la coordenada de origen.
- **Corrección de distorsión de lente:** transforma imágenes de cámaras gran angulares en vistas más convencionales.
- **Homografía:** permite obtener vistas cenitales o de “vuelo de pájaro”, útiles en robótica, vehículos autónomos y análisis de estacionamientos.

Cuando la coordenada calculada no coincide con una posición entera de la grilla, se interpola el color:

- **Vecino más cercano:** toma un solo píxel. Es el método más rápido, pero produce bordes pixelados.
- **Bilineal:** combina cuatro píxeles cercanos. Produce transiciones más suaves con bajo costo computacional.
- **Bicúbica:** utiliza dieciséis píxeles. Suaviza más, pero cuesta más y puede borrar contrastes útiles para visión artificial.

El criterio recomendado es elegir el método más simple que satisfaga la aplicación. En visión artificial suelen priorizarse rendimiento y conservación de información útil para detectar bordes.

## 7. Introducción a coordenadas homogéneas

La clase cierra diferenciando transformaciones lineales y afines:

- Una transformación lineal tiene la forma `y = Ax` y lleva el origen al origen.
- Una transformación afín tiene la forma `y = Ax + b` y solo es lineal cuando `b = 0`.

Las coordenadas homogéneas permitirán expresar transformaciones afines como operaciones matriciales lineales. Este tema continuará en la clase siguiente y se aplicará a visión artificial y robótica.

---

# TRABAJO PRÁCTICO 2 — PUNTOS IMPORTANTES

## Objetivo

Construir un sistema que detecte contornos, calcule sus siete invariantes de Hu y clasifique cada forma en una categoría elegida por el grupo.

## Flujo de trabajo pedido

El docente describe **tres programas o etapas**:

1. **Generador del dataset**
   - Detectar y mostrar las formas o contornos.
   - Calcular los siete invariantes de Hu de cada contorno.
   - Imprimirlos o guardarlos para construir una tabla.
   - Agregar manualmente la etiqueta correcta de cada muestra. Esta es la parte supervisada.

2. **Entrenador**
   - Leer el dataset desde una hoja de cálculo, CSV u otro formato elegido.
   - Cargar las muestras y etiquetas en memoria.
   - Entrenar el clasificador.
   - Guardar en un archivo los parámetros o el modelo entrenado.

3. **Clasificador final**
   - Reutilizar la detección de contornos y el cálculo de invariantes de Hu.
   - Cargar el modelo ya entrenado.
   - Pasar los siete invariantes al clasificador.
   - Mostrar la categoría reconocida sobre el contorno, por ejemplo “cuadrado”.

## Decisiones y recomendaciones del docente

- Usar siempre los siete invariantes de Hu en el mismo orden.
- Elegir categorías claramente definidas y asignarles etiquetas enteras consistentes.
- Incluir ejemplos variados de cada forma, especialmente dibujos deformados o extremos: líneas no perfectamente rectas, concavidades, convexidades y otras variaciones reales.
- Evitar muchas muestras idénticas o casi idénticas.
- Revisar las etiquetas para no introducir errores en el dataset.
- Reservar muestras no vistas para comprobar que el sistema generaliza.
- Se recomienda el árbol de decisión binario por su simplicidad y rapidez, aunque el docente indica que puede utilizarse otro clasificador.
- Si se cambia el vector de características o la configuración del modelo, hay que volver a entrenar.

## Entrega y evaluación

- Al final de la clase se indica que deben presentar **las dos partes del TP2**.
- Fecha indicada: **jueves 10 de septiembre de 2026, durante la clase**.
- No se sube una entrega a una plataforma, según la explicación dada al cierre.
- El grupo debe **mostrar el funcionamiento, exponer y defender el trabajo**.
- La calificación se comunica en la clase y luego forma parte del promedio final de la materia.

> Nota: “tres programas” describe la arquitectura o el flujo técnico explicado por el docente. “Dos partes” parece referirse a la organización formal de la consigna del TP2. Para conocer exactamente qué integra cada parte, hay que contrastar este resumen con el enunciado escrito enlazado por la cátedra.

## Lista de control para la defensa

- [ ] El dataset tiene siete invariantes de Hu y una etiqueta por fila.
- [ ] Las etiquetas mantienen la misma correspondencia en todo el proyecto.
- [ ] Las muestras cubren variaciones representativas de cada forma.
- [ ] El entrenador lee los datos y guarda el modelo entrenado.
- [ ] El clasificador carga ese modelo sin reentrenar durante la ejecución normal.
- [ ] El sistema reconoce contornos nuevos y escribe la categoría correspondiente.
- [ ] Se probaron casos que no estaban en el conjunto de entrenamiento.
- [ ] El grupo puede explicar dataset, entrenamiento, inferencia, sobreajuste y elección del clasificador.

## Relación con los próximos trabajos

- El bloque de transformaciones geométricas anticipa el **TP3**.
- En el **TP4** se retomará la clasificación del TP2, pero con *Deep Learning* y entrenamiento directo a partir de imágenes, sin depender del vector manual de invariantes de Hu.
