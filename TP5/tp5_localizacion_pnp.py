# TP5 - Localización con PnP
# Base: poseARUCO.py del profesor (Laboratorio-de-Robotica/Marcadores-fiduciarios).
# Se conserva su código; los cambios están marcados con [TP5]:
#   1. cameraMatrix y distCoeffs salen de calibrate.py (antes eran aproximados).
#   2. Se usa el primer marcador detectado como sistema de referencia del mundo.
#   3. Se invierte Tcq para obtener la pose de la cámara en el mundo (Tqc).
#   4. Se envía Tqc a Foxglove para visualizarla en 3D en tiempo real.
# Foxglove: Open connection -> Foxglove WebSocket -> ws://localhost:8765,
#           panel 3D con Display frame = "mundo".

import cv2 as cv  # OpenCV (el profesor lo importa como cv)
import numpy as np  # Arreglos y álgebra matricial
import math  # [TP5] Seno y coseno para el cuaternión
import foxglove  # [TP5] SDK de Foxglove (servidor WebSocket)
from foxglove.schemas import FrameTransform, FrameTransforms, Quaternion, Vector3  # [TP5] Mensajes de transformación 3D

print('OpenCV version', cv.__version__)  # Muestra la versión (ArUco requiere 4.7+)
cam = cv.VideoCapture(0)  # Abre la cámara 0
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # Ancho 640 (misma resolución que en calibrate.py)
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # Alto 480

# [TP5] Pegar acá los valores que imprime calibrate.py al presionar 'c'.
# Valores obtenidos con calibrate.py (patrón 9x6 impreso, 25 capturas).
cameraMatrix = np.array(  # Matriz intrínseca K (fx, fy, cx, cy)
    [[543.74,   0.0,   320.65], # [TP5] fx, 0, cx  (calibrate.py, patrón 9x6 impreso, 25 capturas, 11/10)
    [  0.0,   543.82, 244.54],  # [TP5] 0, fy, cy
    [  0.0,     0.0,     1.0  ]], np.float32) # Última fila fija

distCoeffs = np.array([-0.08411395, 0.0761868, 0.0, 0.0, -0.0191586])  # [TP5] K1, K2, P1, P2, K3 (calibrate.py, 11/10)

squareLength = 10  # cm, lado del marcador (medilo con regla); fija la unidad del mundo
objectPoints = np.array(  # Esquinas del marcador en el mundo, centro en el origen, plano Z=0
               [[-squareLength / 2,  squareLength / 2, 0],  # Superior izquierda
                [ squareLength / 2,  squareLength / 2, 0],  # Superior derecha
                [ squareLength / 2, -squareLength / 2, 0],  # Inferior derecha
                [-squareLength / 2, -squareLength / 2, 0]])  # Inferior izquierda (orden horario, como detectMarkers)

det = cv.aruco.ArucoDetector()  # Detector ArUco con parámetros por defecto (igual que el profesor)

foxglove.start_server()  # [TP5] Levanta el servidor de Foxglove en ws://localhost:8765

print("ESC to quit.")  # Cómo salir
axis = np.float32([[0,0,0], [3,0,0], [0,3,0], [0,0,3]]).reshape(-1,3)  # Origen y extremos de los 3 ejes (3 cm)

while True:  # Bucle de localización en tiempo real
    _, im = cam.read()  # Lee un cuadro
    corners, ids, rejectedImgPoints = det.detectMarkers(im)  # Detecta marcadores: esquinas (N,4,2) e IDs
    im = cv.aruco.drawDetectedMarkers(im, corners)  # Dibuja el contorno de los marcadores detectados

    if len(corners) > 0:  # [TP5] Si hay al menos un marcador, el primero es el mundo
        imPoints = np.array(corners[0], np.float32).reshape((4, 2))  # Sus 4 vértices en píxeles

        ret, rvec, tvec = cv.solvePnP(objectPoints, imPoints, cameraMatrix, distCoeffs, flags=cv.SOLVEPNP_IPPE_SQUARE)  # PnP: pose del marcador respecto de la cámara

        if ret:  # Si PnP encontró solución
            imgpts, _ = cv.projectPoints(axis, rvec, tvec, cameraMatrix, distCoeffs)  # Proyecta los ejes 3D a la imagen

            imgpts = np.int32(imgpts).reshape(-1,2)  # Coordenadas enteras en píxeles
            origin = tuple(imgpts[0].ravel())  # Punto de origen de los ejes
            im = cv.line(im, origin, tuple(imgpts[1].ravel()), (255,0,0), 2)  # Eje X
            im = cv.line(im, origin, tuple(imgpts[2].ravel()), (0,255,0), 2)  # Eje Y
            im = cv.line(im, origin, tuple(imgpts[3].ravel()), (0,0,255), 2)  # Eje Z

            R, _ = cv.Rodrigues(rvec)  # Vector de rotación -> matriz 3x3
            Tcq = np.identity(4)  # Matriz homogénea 4x4: marcador (q) expresado en la cámara (c)
            Tcq[:3, :3] = R  # Bloque de rotación
            Tcq[:3, 3] = tvec.ravel()  # Columna de traslación

            Tqc = np.linalg.inv(Tcq)  # [TP5] Inversa: pose de la cámara expresada en el mundo (marcador)
            tqc = Tqc[:3, 3] / 100.0  # [TP5] Posición de la cámara, de cm a metros (unidad de Foxglove)
            distancia_cm = np.linalg.norm(tqc) * 100.0  # Distancia recta entre la cámara y el centro del marcador
            cv.putText(im, f"Distancia: {distancia_cm:.1f} cm", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)  # Muestra la distancia sobre el video
            rvecQc, _ = cv.Rodrigues(Tqc[:3, :3])  # [TP5] Rotación de la cámara como vector de Rodrigues
            ang = float(np.linalg.norm(rvecQc))  # [TP5] Ángulo de giro (norma del vector)
            if ang > 1e-9:  # [TP5] Si hay rotación
                eje = rvecQc.ravel() / ang  # [TP5] Eje de giro unitario
            else:  # [TP5] Sin rotación
                eje = np.array([1.0, 0.0, 0.0])  # [TP5] Eje arbitrario (el seno será 0)
            s = math.sin(ang / 2)  # [TP5] Seno del medio ángulo

            foxglove.log("/tf", FrameTransforms(transforms=[FrameTransform(  # [TP5] Envía la pose a Foxglove
                parent_frame_id="mundo",  # [TP5] Marco padre: el marcador (mundo)
                child_frame_id="camara",  # [TP5] Marco hijo: la cámara
                translation=Vector3(x=float(tqc[0]), y=float(tqc[1]), z=float(tqc[2])),  # [TP5] Posición en metros
                rotation=Quaternion(x=eje[0]*s, y=eje[1]*s, z=eje[2]*s, w=math.cos(ang / 2)),  # [TP5] Orientación como cuaternión
            )]))  # [TP5] Cierra el mensaje

    cv.imshow("webcam", im)  # Muestra la cámara anotada

    match cv.waitKey(30):  # Espera una tecla 30 ms
        case 27: # ESC
            break  # Sale del bucle
