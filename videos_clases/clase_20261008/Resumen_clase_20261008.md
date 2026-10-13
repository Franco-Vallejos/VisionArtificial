# Resumen de clase de Visión Artificial - 08/10/2026

Fuentes utilizadas: [transcripción](./transcripcion/Clase_20261008_transcripcion.md) y [diapositivas recortadas](./ppt/Clase_20261008_ppt_recortadas.pdf). No se utilizó el video para redactar el contenido.

> La transcripción fue generada automáticamente. En este resumen se normalizaron términos que Whisper transcribió fonéticamente, como *keypoint*, *feature*, *embedding*, *matching*, DMatch, SIFT, SURF, ORB, BRIEF, FLANN, SuperPoint y SuperGlue.

## Idea central de la clase

La clase presenta dos familias de técnicas relacionadas. La primera parte desarrolla el pipeline completo de reconocimiento facial. La segunda explica cómo detectar puntos característicos, describirlos y encontrar correspondencias entre dos imágenes.

El docente aclara desde el comienzo que estos temas ya no pertenecen a los trabajos prácticos obligatorios. Pueden servir como base para el proyecto final. Reconocimiento facial aparece como una opción concreta, mientras que detección y matching de features permiten reconocimiento de lugares, seguimiento sin marcadores, homografías, reconstrucción 3D y realidad aumentada.

La conexión directa con trabajos anteriores ayuda a entender el contenido: el reconocimiento facial repite la lógica detección, descriptor y reconocimiento usada en el TP2; el matching extiende la detección de esquinas del TP3; y al final se relaciona esta tecnología con la pose y la realidad aumentada del TP5.

## 1. Pipeline de reconocimiento facial

El procedimiento mínimo tiene tres etapas:

1. Detectar una o más caras en la imagen.
2. Extraer un descriptor o embedding de cada cara.
3. Comparar o clasificar el embedding para determinar una identidad.

La versión más completa agrega dos pasos entre detección y descripción:

1. Detección de caras.
2. Detección de puntos faciales o *facial landmarks*.
3. Alineamiento de la cara.
4. Extracción del embedding.
5. Reconocimiento de la identidad.

El alineamiento mejora la consistencia porque los extractores suelen entrenarse con caras verticales y normalizadas. Si la cara aparece inclinada, los landmarks de los ojos permiten rotarla, trasladarla y escalarla antes de generar el embedding.

### Relación con el TP2 - 00:04:02 a 00:04:56

El docente compara el pipeline con el clasificador de formas:

- en el TP2 se detectaban contornos; aquí se detectan caras;
- en el TP2 se calculaban invariantes de Hu; aquí se extraen embeddings;
- en ambos casos puede entrenarse un clasificador usando los descriptores.

La comparación busca mostrar que cambia el detector y el descriptor, pero se conserva la arquitectura general del reconocimiento.

## 2. Detección de caras

La detección localiza la cara y devuelve una caja delimitadora. La clase repasa brevemente la evolución de los métodos:

- Viola-Jones y cascadas Haar como enfoque histórico;
- detectores basados en redes neuronales profundas;
- YuNet como detector moderno y liviano integrado en OpenCV.

Los métodos anteriores a *deep learning* todavía aparecen en tutoriales, pero el docente los considera obsoletos para una implementación nueva. Recomienda utilizar detectores DNN actuales.

## 3. Facial landmarks

Los landmarks son puntos anatómicos consistentes, por ejemplo ojos, nariz, boca y contorno facial. Sus usos incluyen:

- estimar la orientación de la cara;
- alinear el recorte;
- detectar expresiones;
- realizar anotaciones o efectos de realidad aumentada;
- normalizar la entrada del extractor de embeddings.

Algunos modelos devuelven pocos puntos principales y otros generan una malla facial densa. El número depende del modelo y del propósito de la aplicación.

## 4. Alineamiento de caras

El alineamiento usa los landmarks, especialmente los ojos, para llevar todas las caras a una configuración común. La transformación aplicada suele combinar rotación, traslación y escala.

El proceso busca que:

- la línea entre los ojos quede horizontal;
- los ojos ocupen posiciones predeterminadas;
- el recorte tenga un tamaño uniforme;
- el extractor reciba imágenes parecidas a las de su entrenamiento.

Esta normalización aumenta la calidad y estabilidad del embedding.

## 5. Embeddings faciales

Un embedding es un descriptor vectorial de una cara. FaceNet popularizó vectores de 128 dimensiones. Cada vector ocupa poco espacio y permite comparar identidades mediante una distancia.

El entrenamiento busca que:

- imágenes de una misma persona produzcan vectores cercanos;
- personas distintas queden separadas en el espacio de embeddings.

La clase menciona FaceNet como referencia histórica y ArcFace como enfoque posterior. OpenCV incorpora SFace como extractor de embeddings y YuNet como detector.

## 6. Clasificación, clustering y bases vectoriales

Hay varias estrategias para utilizar embeddings.

### Clasificador supervisado

Se reúnen varias imágenes de cada persona, se extraen sus embeddings y se entrena un clasificador. El problema es que agregar una identidad nueva puede obligar a entrenar nuevamente el modelo completo.

### Clustering

Un algoritmo no supervisado agrupa embeddings parecidos sin conocer nombres. Esto permite descubrir que la misma persona aparece varias veces aunque todavía no se sepa su identidad.

### Base de datos vectorial

Se almacena cada embedding junto con una identidad. Para reconocer una cara nueva se busca el vector más cercano. Este enfoque permite agregar personas sin volver a entrenar un clasificador.

Las bases vectoriales utilizan índices y algoritmos aproximados de vecinos cercanos para evitar comparar contra todos los registros. Esa aproximación mejora mucho la velocidad cuando existen miles de identidades.

## 7. Modelos actuales en OpenCV

La clase destaca dos componentes:

- **YuNet:** detector facial liviano.
- **SFace:** extractor de embeddings que también ofrece operaciones de alineamiento, extracción y comparación.

Esta integración refleja un pipeline ya consolidado. La API expone directamente las operaciones que antes debían implementarse por separado.

## 8. Detección y matching de features

La segunda parte introduce correspondencias visuales entre imágenes. Una *feature* es una característica localizada de la escena. Su posición se representa mediante un *keypoint* y su apariencia mediante un descriptor.

Las correspondencias permiten vincular puntos equivalentes aunque cambien:

- el punto de vista;
- la escala;
- la rotación;
- la iluminación;
- parte del contenido visible.

Entre las aplicaciones mencionadas aparecen detección de objetos, reconocimiento de lugares, composición de panoramas, odometría visual, SLAM y seguimiento sin marcadores.

## 9. Keypoints

Un buen punto característico debe volver a detectarse en distintas imágenes de la misma escena. Las esquinas suelen ser adecuadas porque tienen cambios de intensidad en más de una dirección.

La clase diferencia:

- **detector:** localiza los keypoints;
- **descriptor:** representa el entorno visual de cada keypoint;
- **matcher:** compara descriptores de distintas imágenes.

El objeto `cv2.KeyPoint` puede incluir posición, tamaño, orientación, respuesta del detector, octava de escala e identificador de clase.

## 10. Propiedades deseables

La característica principal es la repetibilidad: el mismo punto físico debe volver a detectarse bajo condiciones diferentes.

Según la aplicación, se busca invariancia o robustez frente a:

- cambio de escala;
- rotación;
- perspectiva;
- iluminación;
- ruido o desenfoque.

No todos los detectores resuelven todos los cambios. La elección depende del equilibrio entre velocidad, robustez y restricciones de licencia.

## 11. Detectores clásicos

La clase recorre varios algoritmos:

- **Harris:** detector de esquinas clásico, sensible a escala y rotación según su configuración.
- **Shi-Tomasi / `goodFeaturesToTrack`:** modificación de Harris con un criterio práctico para seleccionar buenas esquinas.
- **SIFT:** detecta en pirámides de escala, calcula orientación y produce descriptores robustos.
- **FAST:** muy rápido y repetible, aunque originalmente no aporta orientación ni escala.
- **AGAST:** variante adaptativa de FAST.
- **BRISK:** combina detección y descripción binaria.
- **AKAZE:** detector y descriptor robusto disponible en OpenCV.
- **SURF:** históricamente importante, pero con restricciones de patente o disponibilidad según la versión.

## 12. Detección mediante deep learning

SuperPoint utiliza una red neuronal para detectar puntos y extraer descriptores. Produce un mapa de probabilidad de keypoints y un descriptor denso que después se consulta en las posiciones seleccionadas.

Estos métodos buscan superar la robustez de los algoritmos clásicos, aunque no necesariamente su velocidad. La elección depende de si la aplicación prioriza tiempo real o tolerancia a cambios difíciles.

## 13. Descriptores

Un descriptor resume los píxeles de una región alrededor del keypoint en un vector de dimensión fija. Los descriptores de un mismo método pueden compararse entre sí mediante una distancia.

Hay dos grandes grupos:

### Descriptores de punto flotante

SIFT utiliza vectores de 128 valores. Ofrece buena robustez, pero su comparación requiere más memoria y operaciones aritméticas.

### Descriptores binarios

BRIEF, ORB, BRISK y otros representan comparaciones entre pares de píxeles mediante bits. Son compactos y permiten una comparación muy rápida con distancia de Hamming.

Los descriptores no deben mezclarse arbitrariamente. Un descriptor ORB se compara con otro ORB; no se compara directamente con SIFT porque su estructura y métrica son diferentes.

## 14. BRIEF y ORB

BRIEF construye un vector binario comparando intensidades de pares de píxeles dentro del parche. Resulta rápido y robusto a cambios globales de iluminación, pero su versión original no es invariante a escala ni rotación.

ORB combina:

- FAST para detectar puntos;
- una pirámide para manejar escala;
- orientación calculada sobre la vecindad;
- BRIEF rotado y optimizado como descriptor.

El resultado habitual es un descriptor de 256 bits, equivalente a 32 bytes. La clase lo presenta como una alternativa especialmente adecuada cuando importa la velocidad.

## 15. Distancias entre descriptores

La métrica depende del tipo de vector:

- **distancia euclidiana:** apropiada para descriptores de punto flotante como SIFT;
- **distancia de Hamming:** apropiada para descriptores binarios como ORB y BRIEF.

Hamming aplica una operación XOR y cuenta los bits diferentes. Para un descriptor de 256 bits, la distancia posible va de 0 a 256.

## 16. Matching de descriptores

El matcher recibe dos conjuntos de descriptores. OpenCV suele llamar `train` al conjunto de referencia y `query` al conjunto consultado.

El docente corrige explícitamente una posible confusión: el matching compara descriptores, no posiciones de keypoints. Las coordenadas se usan después para visualizar o verificar coherencia geométrica.

El resultado se representa mediante objetos `DMatch`, que almacenan:

- índice del descriptor de referencia;
- índice del descriptor de consulta;
- distancia calculada.

## 17. Fuerza bruta y FLANN

`BFMatcher` compara todos los descriptores contra todos. Si cada imagen tiene 1000 descriptores, el proceso necesita cerca de un millón de comparaciones.

FLANN utiliza estructuras de búsqueda aproximada para reducir el trabajo. Puede no garantizar el vecino matemáticamente más cercano en todos los casos, pero logra gran velocidad y suele resultar adecuado para tiempo real.

La clase recomienda FLANN cuando el matching debe ejecutarse en tiempo real. Para descriptores binarios hay que elegir el índice o algoritmo compatible con ese tipo de datos.

## 18. Filtrado de matches y outliers

Una distancia pequeña no garantiza por sí sola una correspondencia correcta. La clase presenta varios niveles de filtrado:

### Ratio test

Se comparan la mejor y la segunda mejor coincidencia. Si la primera es claramente menor, el match resulta distintivo. Si ambas distancias son parecidas, la correspondencia es ambigua y conviene descartarla.

### Coherencia geométrica

Después del matching se analizan las coordenadas de los keypoints. Una homografía, un estimador robusto como RANSAC o un filtro espacial permiten detectar correspondencias que no encajan con el movimiento común.

### GMS

Grid-based Motion Statistics analiza la coherencia espacial de los matches y elimina muchos outliers.

`cv2.drawMatches` permite unir las dos imágenes y dibujar las correspondencias. Es una herramienta útil para inspeccionar visualmente matches aceptados y descartados.

---

# RELACIÓN CON LOS TRABAJOS PRÁCTICOS Y EL PROYECTO FINAL

## Aclaración principal

El docente afirma al comienzo que los temas de esta clase **no corresponden a los trabajos prácticos obligatorios actuales**. Los presenta para ampliar las herramientas disponibles y para inspirar posibles proyectos finales.

Por lo tanto, reconocimiento facial, SuperPoint, SuperGlue, bases vectoriales y el pipeline completo de matching no deben interpretarse automáticamente como requisitos de entrega de un TP.

## Proyecto final - 00:00:22 a 00:02:08

Reconocimiento facial aparece como una de las opciones concretas para proyecto final. El docente comenta que suele haber grupos interesados y que mantiene actualizado el material por ese motivo.

Un proyecto de reconocimiento facial podría incluir:

1. Detección de caras.
2. Landmarks y alineamiento.
3. Extracción de embeddings.
4. Registro de identidades.
5. Reconocimiento mediante distancia o base vectorial.
6. Manejo de personas desconocidas.

## Conexión con el TP2 - 00:04:02 a 00:04:56 y 00:20:49 a 00:21:33

El pipeline de reconocimiento facial reutiliza la estructura conceptual del TP2:

| TP2 | Reconocimiento facial |
|---|---|
| Detectar contornos | Detectar caras |
| Calcular invariantes de Hu | Extraer embeddings |
| Entrenar un clasificador de formas | Clasificar identidades |

Esta relación es explicativa. No modifica la entrega ya realizada del TP2.

## Conexión con el TP3 - 01:24:27 a 01:25:27

En el TP3 se utilizaban las esquinas detectadas por un marcador ArUco. El matching de features permite localizar una imagen cualquiera sin depender de un marcador diseñado específicamente.

La idea abre la posibilidad de usar una fotografía, una portada o una textura plana como marcador virtual. Se detectan keypoints, se extraen descriptores, se buscan correspondencias y luego se verifica su coherencia geométrica.

Nuevamente, se presenta como extensión conceptual y no como corrección obligatoria del TP3.

## Conexión con el TP5 - 01:38:08 a 01:38:55

Al cierre, el docente relaciona el contenido con el TP5. La pose calculada en ese trabajo puede aprovecharse para:

- graficar ejes o representar visualmente la pose;
- controlar o seguir la orientación de un robot;
- colocar un objeto virtual sobre la imagen;
- construir una aplicación de realidad aumentada.

Aclara que el TP5 busca una implementación mínima para facilitar la entrega. Las aplicaciones 3D y de realidad aumentada muestran el potencial de la tecnología, pero no se presentan en esta clase como requisitos obligatorios adicionales.

## Menciones cronológicas

| Momento | Mención | Interpretación |
|---|---|---|
| 00:00:22 | Los temas ya no corresponden a TP obligatorios. | Usarlos como ampliación o inspiración para proyecto final. |
| 00:00:27 | Hay una lista de posibles proyectos finales. | Reconocimiento facial puede ser una opción de proyecto. |
| 00:01:45 | Reconocimiento facial fue un TP en años anteriores. | No implica que sea el TP actual. |
| 00:04:16 | Se compara el pipeline con el TP2. | Sirve para reutilizar el modelo mental detección, descriptor y clasificador. |
| 00:20:49 | Se explica cómo reconocer caras usando un clasificador como en el TP2. | Alternativa educativa, aunque las bases vectoriales permiten agregar identidades con más facilidad. |
| 00:41:46 | La detección de features se compara con la detección de objetos del TP2. | Relación conceptual, no requisito nuevo. |
| 01:24:27 | El matching se compara con las esquinas de ArUco del TP3. | Permite seguimiento sin marcadores predefinidos. |
| 01:38:18 | La pose y la realidad aumentada se vinculan con el TP5. | El TP5 se mantiene mínimo; los usos 3D son extensiones posibles. |

## Qué conviene rescatar para el TP5

Aunque la clase no agrega una nueva consigna obligatoria, estos conceptos pueden mejorar la demostración del TP5:

- usar la pose calculada para dibujar ejes o una figura 3D;
- comprobar visualmente que la anotación acompaña la orientación del marcador;
- entender que un marcador virtual podría reemplazar al ArUco en una versión avanzada;
- filtrar correspondencias erróneas antes de estimar una transformación;
- distinguir la similitud de descriptores de la consistencia geométrica.

## Qué no debe agregarse por obligación

- Un sistema de reconocimiento facial.
- Una base de datos vectorial.
- Entrenamiento de FaceNet, ArcFace o SFace.
- SuperPoint o SuperGlue.
- Seguimiento sin marcadores.
- FLANN, GMS o RANSAC si la solución del TP5 usa directamente un ArUco bien detectado.
- Una escena de realidad aumentada compleja.

## Checklist para un posible proyecto final de reconocimiento facial

- [ ] Detectar todas las caras de la imagen.
- [ ] Obtener landmarks suficientes para alinear cada rostro.
- [ ] Normalizar el tamaño y la orientación del recorte.
- [ ] Extraer embeddings con un modelo actual.
- [ ] Elegir clasificación, clustering o búsqueda vectorial según el caso.
- [ ] Definir una métrica y un umbral para aceptar una identidad.
- [ ] Manejar correctamente personas desconocidas.
- [ ] Evaluar cambios de iluminación, pose, escala y oclusiones.
- [ ] Considerar privacidad y autorización para almacenar datos biométricos.

## Checklist para un posible proyecto final de matching

- [ ] Elegir detector y descriptor compatibles.
- [ ] Usar la métrica correcta: euclidiana o Hamming.
- [ ] Comparar descriptores, no coordenadas de keypoints.
- [ ] Elegir fuerza bruta o FLANN según el costo permitido.
- [ ] Aplicar ratio test u otro filtro de ambigüedad.
- [ ] Verificar coherencia geométrica y descartar outliers.
- [ ] Visualizar los matches para diagnosticar errores.
- [ ] Medir velocidad y estabilidad ante rotación, escala y perspectiva.

## Síntesis final

La clase amplía las técnicas de reconocimiento y correspondencia visual, pero no agrega un nuevo TP obligatorio. Su valor principal está en mostrar cómo los mismos bloques conceptuales usados en trabajos anteriores evolucionan hacia sistemas más generales: caras representadas por embeddings, imágenes representadas por keypoints y descriptores, y correspondencias filtradas mediante criterios de distancia y geometría. Para el TP5, la conexión concreta es la visualización y explotación de la pose; para el proyecto final, reconocimiento facial o matching ofrecen líneas completas de desarrollo.
