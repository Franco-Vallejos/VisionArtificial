# Resumen de clase de Vision Artificial - 24/09/2026

Fuentes utilizadas: [transcripcion](./transcripcion/Clase_20260924_transcripcion.md) y [diapositivas recortadas](./ppt/Clase_20260924_ppt_recortadas.pdf). No se utilizo el video para redactar el contenido.

> La transcripcion fue generada automaticamente. En este resumen se normalizaron nombres tecnicos que Whisper transcribio de forma fonetica, como YOLO, Ultralytics, PyTorch, TensorFlow, Kaggle y segmentacion.

## Idea central de la clase

La clase funciona como introduccion y preparacion directa para el TP4. El objetivo es recorrer de punta a punta un proyecto de *deep learning* aplicado a vision artificial: elegir un dataset ya existente, adaptarlo al formato requerido, entrenar un modelo, interpretar los resultados y usar el archivo entrenado para realizar inferencia.

El docente recomienda Ultralytics y YOLO porque simplifican mucho el entrenamiento, aunque aclara que no son obligatorios. La experiencia buscada no consiste solamente en ejecutar una biblioteca, sino en comprender que datos recibe el modelo, que ocurre durante las epocas, como se evalua el aprendizaje, cuanto demora y que tan cerca queda el resultado de lo esperado.

La clase tambien repasa datasets, tipos de anotacion, aumentacion de datos, demostraciones en Google Colab y los principales frameworks de *deep learning*.

## 1. YOLO y su lugar en vision artificial

YOLO significa *You Only Look Once*. Nacio como una familia de detectores de objetos capaz de procesar una imagen en una sola pasada, lo que permitio detecciones mucho mas rapidas que las alternativas iniciales.

El docente explica que las versiones llamadas YOLO no forman necesariamente una unica linea mantenida por la misma organizacion. El proyecto fue cambiando de autores, implementaciones y frameworks. En la actualidad, Ultralytics consolida un ecosistema sencillo de instalar y usar, basado principalmente en PyTorch.

La recomendacion de Ultralytics responde a que ofrece una interfaz integrada para:

- descargar o cargar modelos;
- entrenar con un dataset propio;
- validar el resultado;
- generar metricas y graficos;
- realizar inferencia;
- resolver deteccion, segmentacion, estimacion de pose y otras tareas.

### Relacion con el TP - 00:04:46 a 00:05:49

Se puede elegir otro modelo o framework, pero el docente recomienda fuertemente Ultralytics porque reduce la friccion tecnica del entrenamiento. La contrapartida es que la facilidad de uso puede ocultar el proceso; por eso se pide interpretar lo que sucede y no limitarse a ejecutar comandos.

## 2. Estructura de un dataset para YOLO

Los datasets no tienen un formato universal. Cada herramienta puede exigir una estructura y un formato de anotacion diferentes. Ultralytics define una convencion estricta para automatizar el entrenamiento.

Para un dataset de deteccion, la estructura explicada incluye:

- una carpeta de imagenes;
- una carpeta de etiquetas o `labels`;
- subdivisiones equivalentes para entrenamiento, validacion y prueba;
- un archivo YAML que indica ubicaciones, divisiones y nombres de clases.

La division sugerida en clase es:

- 90 % para entrenamiento;
- 5 % para validacion;
- 5 % para prueba.

Cada imagen debe tener su archivo de etiqueta correspondiente. Para deteccion, cada renglon representa un objeto e incluye la categoria y cuatro valores asociados a su caja delimitadora. Las coordenadas se expresan normalizadas entre 0 y 1, no en pixeles.

La ventaja de normalizar coordenadas es que las etiquetas siguen siendo validas aunque se cambie la resolucion de las imagenes. Si estuvieran expresadas en pixeles, habria que recalcularlas cada vez que se redimensiona el dataset.

### Relacion con el TP - 00:17:14 a 00:24:14

El dataset elegido puede venir en JSON, otro esquema de carpetas o un formato de anotacion distinto. En ese caso hay que convertirlo al formato esperado por la herramienta. El docente menciona que esta conversion puede resolverse con un script y que es obligatorio organizar correctamente imagenes, etiquetas y particiones antes de entrenar.

## 3. Entrenamiento con Ultralytics y Google Colab

El flujo presentado es deliberadamente corto:

1. Instalar o importar `ultralytics`.
2. Crear una instancia de YOLO eligiendo una variante del modelo.
3. Indicar el archivo YAML del dataset.
4. Definir la cantidad de epocas.
5. Ejecutar el entrenamiento.
6. Conservar el modelo resultante y analizar sus resultados.

Una epoca equivale a recorrer una vez todo el conjunto de entrenamiento. Al comenzar una nueva epoca, el modelo vuelve a procesar el dataset y ajusta sus parametros un poco mas.

El numero de epocas no es fijo: depende del dataset, el modelo, el tiempo disponible y el comportamiento de las metricas. La diapositiva usa 100 como ejemplo, pero el docente indica que para el trabajo se podria comenzar con una cantidad menor, como 20, y observar el resultado.

Google Colab permite ejecutar este proceso en una computadora remota y evita depender por completo del hardware local. El entrenamiento puede tardar desde menos de una hora hasta varias horas, segun el dataset, el modelo y la cantidad de epocas.

## 4. Variantes y escala del modelo

Ultralytics ofrece variantes de distinto tamano, como `n` (*nano*), `s`, `m`, `l` y `x`. En general:

- un modelo mas pequeno entrena y ejecuta mas rapido, pero puede ser menos preciso;
- un modelo mas grande puede mejorar la precision, a costa de mas memoria y tiempo de procesamiento.

### Relacion con el TP - 00:31:40 a 00:32:53

Para el TP4 no se exige alcanzar una precision minima. El docente sugiere que probablemente alcance una variante pequena, como *nano*. Lo importante es atravesar el proceso, probar el modelo, reconocer que puede acertar y fallar, y poder explicar el comportamiento observado.

## 5. Evaluacion del entrenamiento

El entrenamiento genera graficos y metricas que permiten determinar si el modelo esta aprendiendo.

### Matriz de confusion

La matriz de confusion compara las clases reales con las clases predichas:

- la diagonal concentra los aciertos;
- los valores fuera de la diagonal muestran que clases se confunden entre si;
- una confusion reiterada puede indicar que faltan ejemplos de esas categorias o que las muestras no permiten diferenciarlas bien.

### Funciones de perdida

Las curvas de perdida deben disminuir. Su valor absoluto no siempre es interpretable de manera aislada; lo importante es observar la tendencia. Si la curva deja de bajar, el entrenamiento puede haberse estancado y agregar epocas podria no producir una mejora significativa.

### Precision y otras metricas normalizadas

Las metricas normalizadas suelen buscar valores cercanos a 1. Durante el entrenamiento, cada epoca agrega un punto al grafico y permite ver la evolucion del modelo.

### Relacion con el TP - 00:27:52 a 00:31:31

El docente lo declara parte del trabajo practico: cuando el entrenamiento produzca graficos, hay que explicar que significa cada grafico presentado. No alcanza con adjuntarlo. Debe interpretarse por que algunas curvas bajan, por que otras suben y que indica su evolucion sobre el aprendizaje.

## 6. Tareas de vision y anotaciones

La clase diferencia varios problemas:

- **Clasificacion:** asigna una categoria a una imagen completa.
- **Deteccion:** localiza objetos mediante cajas y asigna una clase a cada uno.
- **Segmentacion semantica:** asigna una categoria a cada pixel.
- **Estimacion de pose:** localiza puntos o articulaciones relevantes.
- **Prominencia o saliencia:** identifica regiones que atraen la atencion visual.

La deteccion es el punto fuerte historico de YOLO y suele ser suficiente para muchos casos de uso. La segmentacion produce una descripcion espacial mas detallada, pero normalmente requiere mas procesamiento.

Un dataset debe contener tanto las muestras como el resultado esperado. Por eso el tipo de anotacion depende de la tarea:

- una clase para clasificacion;
- cajas para deteccion;
- mascaras o poligonos para segmentacion;
- puntos para pose.

Algunos datasets son multiproposito y ofrecen varias anotaciones para la misma imagen.

## 7. Creacion, etiquetado y gestion de datasets

Crear un dataset propio exige capturar imagenes, definir categorias, anotar cada muestra, revisar errores y organizar particiones. En un proyecto real, esta puede ser la parte mas costosa y valiosa del desarrollo.

La clase muestra herramientas de anotacion y plataformas que permiten distribuir el trabajo entre varias personas. Tambien diferencia datasets publicos y privados:

- los publicos facilitan investigacion, comparacion y aprendizaje;
- los privados suelen representar problemas comerciales especificos y pueden constituir el activo principal de una solucion.

Para el objetivo academico del TP4 no se pide crear miles de anotaciones desde cero, porque repetir manualmente el etiquetado aporta poco aprendizaje una vez comprendido el procedimiento. Se prioriza usar un dataset existente y estudiar el ciclo de entrenamiento.

## 8. Aumentacion de datos

La aumentacion genera variantes de muestras existentes mediante transformaciones compatibles con el problema, por ejemplo:

- rotaciones pequenas;
- traslaciones;
- cambios de escala;
- reflexiones;
- transformaciones geometricas;
- cambios de iluminacion o color, cuando resulten validos.

La etiqueta debe transformarse junto con la imagen. En segmentacion, por ejemplo, la misma operacion geometrica debe aplicarse tanto a la imagen como a su mascara.

La aumentacion enriquece el entrenamiento y ayuda a que el modelo generalice, pero no reemplaza por completo la diversidad de datos reales. Los datasets sinteticos pueden complementar los reales, aunque tambien pueden introducir diferencias respecto del dominio donde se utilizara el modelo.

## 9. Datasets publicos mencionados

La clase presenta datasets ampliamente usados, entre ellos:

- MNIST y Fashion-MNIST para clasificacion;
- ImageNet para clasificacion a gran escala;
- COCO y Open Images para deteccion y otras tareas;
- Cityscapes para escenas urbanas;
- datasets medicos y biologicos;
- Oxford-IIIT Pet para segmentacion de mascotas.

Kaggle se recomienda como un sitio inicial para buscar. Hay que comprobar que el dataset sea realmente de imagenes y que incluya las anotaciones necesarias para la tarea seleccionada.

## 10. Demostraciones de entrenamiento

El docente muestra en Colab el entrenamiento de segmentadores semanticos. En una demostracion con mascotas, una U-Net mejora progresivamente durante 30 epocas. En otra, un dataset de microscopio utiliza aumentacion y se observa el progreso durante 15 epocas.

Estas secuencias muestran que:

- las primeras predicciones pueden no parecerse en nada al resultado esperado;
- el aprendizaje es gradual;
- mas epocas pueden mejorar el resultado, pero no garantizan perfeccion;
- conviene evaluar siempre la prediccion sobre ejemplos concretos, ademas de mirar las metricas.

Estas demostraciones ilustran el proceso. No obligan a implementar U-Net ni a reproducir esos dos datasets en el TP4.

## 11. Frameworks, backend e inferencia

Un framework de *deep learning* ofrece bibliotecas, modelos, entrenamiento, metricas y un backend que ejecuta los calculos. TensorFlow y PyTorch son los dos ecosistemas dominantes presentados en la clase.

El backend intenta aprovechar el hardware disponible:

- CPU y sus nucleos;
- GPU para calculo paralelo;
- TPU en infraestructura de Google;
- NPU para inferencia eficiente en dispositivos de borde.

Entrenar y hacer inferencia son etapas diferentes. El entrenamiento es mucho mas costoso y produce un archivo de pesos o modelo. La inferencia carga ese archivo y lo aplica a nuevas entradas. Un sistema desplegado puede usar un backend mas liviano si solo necesita ejecutar el modelo.

---

# TRABAJO PRACTICO 4 - PUNTOS IMPORTANTES

## Objetivo confirmado

Elegir un dataset existente de imagenes, entrenar un modelo de *deep learning*, evaluar su comportamiento y utilizar el archivo resultante para realizar inferencia sobre una nueva imagen. El foco esta en experimentar y comprender el ciclo completo, no en obtener una precision minima determinada.

## Menciones y requisitos en orden cronologico

| Momento | Indicacion del docente | Consecuencia practica |
|---|---|---|
| 00:01:09 | El TP4 consiste en entrenar un modelo de *deep learning* y permite elegir el dataset. | El tema y las categorias pueden seleccionarse segun el interes del grupo. |
| 00:03:26 | Para el trabajo se debe elegir un dataset existente, entrenar y observar como funciono. | No es necesario crear y etiquetar desde cero un dataset grande. |
| 00:04:46 | El modelo es libre, pero se recomienda fuertemente Ultralytics. | YOLO/Ultralytics es el camino sugerido, no una obligacion absoluta. |
| 00:17:03 | Ultralytics tambien permite segmentacion, pose y otras tareas. | Se puede elegir una funcion que se adapte al dataset. |
| 00:17:25 | El dataset debe respetar el formato esperado por la herramienta. | Hay que ordenar carpetas, particiones y anotaciones o convertirlas con un script. |
| 00:18:34 | Se sugiere dividir 90 % entrenamiento, 5 % validacion y 5 % prueba. | Mantener separados los conjuntos para no evaluar con las mismas muestras usadas al entrenar. |
| 00:26:45 | Las epocas indican cuantas veces se recorre el dataset. | Elegir una cantidad razonable y registrar cuanto tarda. |
| 00:30:06 | Explicar los graficos obtenidos es parte del TP. | Interpretar matriz de confusion, perdidas y metricas; no limitarse a pegarlas. |
| 00:32:12 | No se exige una precision minima para este proyecto. | Una variante pequena puede ser suficiente; deben analizarse aciertos y fallos. |
| 01:36:10 | Se reafirma que cada grupo puede buscar el dataset que quiera. | Kaggle y buscadores de datasets son puntos de partida. |
| 01:37:35 | El dataset debe contener imagenes y su resultado esperado. | Verificar que tenga etiquetas adecuadas, no solo archivos o mediciones tabulares. |
| 01:37:49 | Se recomienda apuntar principalmente a deteccion o segmentacion. | Son las dos opciones preferidas, aunque se admite otra tarea apropiada. |
| 01:38:19 | El entrenamiento produce un archivo que contiene el conocimiento aprendido. | Guardar e identificar el archivo del modelo entrenado. |
| 01:39:15 | Ese archivo debe cargarse y ejecutarse sobre una imagen. | Incluir una prueba de inferencia funcional. |
| 01:39:23 | La salida depende de la tarea: cajas y clases para deteccion; mascara para segmentacion. | Mostrar visualmente el resultado correspondiente al problema elegido. |
| 01:39:39 | El centro del TP es atravesar el proceso y analizar tiempo y calidad. | Documentar duracion, resultado final y cercania respecto de lo esperado. |

## Entregable funcional inferido de la explicacion

Aunque en esta clase no se muestra una consigna escrita completa, la explicacion oral deja como minimo el siguiente flujo demostrable:

1. Identificar el problema elegido y la tarea de vision.
2. Indicar de donde se obtuvo el dataset.
3. Verificar que contiene imagenes y anotaciones compatibles.
4. Dividir u organizar entrenamiento, validacion y prueba.
5. Convertir las etiquetas si no estan en el formato del framework.
6. Elegir el modelo y justificar brevemente su tamano o variante.
7. Entrenar durante una cantidad definida de epocas.
8. Guardar el archivo de pesos o modelo resultante.
9. Presentar e interpretar las metricas y curvas obtenidas.
10. Cargar el modelo entrenado en un proceso separado de inferencia.
11. Ejecutarlo sobre una imagen y visualizar el resultado.
12. Explicar aciertos, errores, tiempo de entrenamiento y posibles mejoras.

## Lo que no debe confundirse con un requisito

- Ultralytics esta fuertemente recomendado, pero el docente permite otras herramientas.
- Los ejemplos de mascotas y microscopio ilustran el aprendizaje; no son datasets obligatorios.
- U-Net aparece en una demostracion, pero no es el unico modelo aceptado.
- Las 100 epocas de la diapositiva son un ejemplo, no una cantidad obligatoria.
- No se exige construir un dataset nuevo ni alcanzar una precision minima especifica.
- La parte historica de YOLO y frameworks aporta contexto, pero no reemplaza la demostracion practica.

## Checklist breve para el TP4

- [ ] El dataset contiene imagenes y etiquetas adecuadas.
- [ ] La tarea elegida esta definida: preferentemente deteccion o segmentacion.
- [ ] Las carpetas y anotaciones respetan el formato del framework.
- [ ] Entrenamiento, validacion y prueba estan separados.
- [ ] El modelo y la cantidad de epocas estan documentados.
- [ ] Se conserva el archivo entrenado.
- [ ] Se interpretan las curvas, perdidas y metricas mostradas.
- [ ] El modelo se carga nuevamente para inferencia.
- [ ] Se prueba con una imagen y se visualiza la salida.
- [ ] Se informan tiempo de entrenamiento, aciertos, fallos y posibles mejoras.

## Sintesis final

El TP4 no busca solamente que una llamada a YOLO produzca una prediccion. Busca que el estudiante comprenda la cadena completa: datos anotados, formato, particiones, seleccion del modelo, entrenamiento por epocas, evaluacion, archivo aprendido e inferencia. La calidad final importa como objeto de analisis, pero la experiencia y la explicacion del proceso son el centro del trabajo.
