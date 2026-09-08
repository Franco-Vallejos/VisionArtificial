#openCV sirve para abrir la cámara, capturar frames, convertir BGR a RGB, dibujar un recatangulo y mostrar vídeo
import cv2
#analiza la imagen, detecta la cara y devuelve la info
import mediapipe as mp

# NUEVO: herramientas para guardar y descargar el modelo de deteccion facial
from pathlib import Path
import urllib.request

# NUEVO: ubicacion y direccion del modelo requerido por MediaPipe Tasks
RUTA_MODELO = Path(__file__).resolve().parent / "blaze_face_short_range.tflite"
URL_MODELO = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)

# NUEVO: descarga el modelo solamente si todavia no existe
if not RUTA_MODELO.exists():
    print("Descargando modelo de deteccion facial...")
    urllib.request.urlretrieve(URL_MODELO, RUTA_MODELO)

# Abrimos la webcam
cap = cv2.VideoCapture(0)

# Inicializamos MediaPipe
FaceDetector = mp.tasks.vision.FaceDetector

# NUEVO: configuracion que utiliza la API moderna de MediaPipe Tasks
opciones = mp.tasks.vision.FaceDetectorOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(RUTA_MODELO)),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    min_detection_confidence=0.5,
)

# Creamos el detector
with FaceDetector.create_from_options(opciones) as face_detection:

    while True:

        # Capturamos un frame
        ret, frame = cap.read()

        if not ret:
            break

        # Convertimos BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # NUEVO: convertimos el frame RGB al formato de imagen de MediaPipe
        imagen_mediapipe = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # MediaPipe detecta las caras
        cara = face_detection.detect(imagen_mediapipe)

        # Dibujamos SOLO el rectángulo
        if cara.detections:

            for detection in cara.detections:

                bbox = detection.bounding_box

                x = bbox.origin_x
                y = bbox.origin_y
                ancho = bbox.width
                alto = bbox.height

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + ancho, y + alto),
                    (0, 255, 255),
                    3
                )

        # Mostramos la cámara
        cv2.imshow("MediaPipe - Deteccion de Rostro", frame)

        # ESC para salir
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
