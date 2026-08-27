#openCV sirve para abrir la cámara, capturar frames, convertir BGR a RGB, dibujar un recatangulo y mostrar vídeo
import cv2
#analiza la imagen, detecta la cara y devuelve la info
import mediapipe as mp

# Abrimos la webcam
cap = cv2.VideoCapture(1)

# Inicializamos MediaPipe
mp_face_detection = mp.solutions.face_detection

# Creamos el detector
with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.1) as face_detection:

    while True:

        # Capturamos un frame
        ret, frame = cap.read()

        if not ret: #valido nada x las dudas
            break

        # Convertimos BGR → RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cara = face_detection.process(rgb)

        # Dibujamos SOLO el rectángulo
        if cara.detections:

            for detection in cara.detections:

                bbox = detection.location_data.relative_bounding_box

                alto, ancho, _ = frame.shape

            #Convierto las coordenadas relativas a absolutas
                x = int(bbox.xmin * ancho)
                y = int(bbox.ymin * alto)
                ancho = int(bbox.width * ancho)
                alto = int(bbox.height * alto)

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
