# calibrate.py del profesor (AlejandroSilvestri/VisiAr/Calibration), SIN cambios de lógica.
# Única modificación, marcada con [TP5]: se fija la resolución en 640x480,
# las mismas dos líneas que usa poseARUCO.py, para que K sea válida en el TP5.
# Necesita en la misma carpeta la imagen "pattern_chessboard 6 x 9.png" del repo.

import numpy as np  # Arreglos numéricos
import cv2 as cv  # OpenCV (el profesor lo importa como cv)

print("""
      Usage:
      Space: take picture
      C: calibrate with taken pictures
      ESC: quit
      """)  # Muestra las teclas: espacio = capturar, c = calibrar, ESC = salir

ESC = chr(27)  # Carácter de la tecla ESC
cv.namedWindow("Detecciones", cv.WINDOW_NORMAL)  # Ventana que acumula las capturas tomadas
cv.namedWindow("Cam", cv.WINDOW_NORMAL)  # Ventana con la cámara en vivo
cv.namedWindow("Tablero", cv.WINDOW_NORMAL)  # Ventana que muestra el patrón a imprimir
defaultPrintOptions = np.get_printoptions()  # Guarda el formato de impresión de numpy para restaurarlo

chessBoard = (9,6)  # Esquinas internas del tablero (9 x 6)
gradualDarkness = 0.90  # Factor para oscurecer capturas viejas en "Detecciones"
cornersubpixTerminationCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Corte del refinamiento subpíxel
imgPoints = []  # Esquinas 2D detectadas, una lista por captura
objPoints = []  # Puntos 3D del tablero, una lista por captura

# chessboard 3D points
chessboardPointCloud3D = np.zeros((chessBoard[0]*chessBoard[1],3), np.float32)  # Una fila (X,Y,Z) por esquina
chessboardPointCloud3D[:,:2] = np.mgrid[0:chessBoard[0],0:chessBoard[1]].T.reshape(-1,2)  # Grilla plana Z=0 (unidad = 1 cuadro)

imChessboard = cv.imread("pattern_chessboard 6 x 9.png")  # Carga la imagen del patrón
cv.imshow("Tablero", imChessboard)  # La muestra (para imprimirla o mostrarla en pantalla)

cam = cv.VideoCapture(0)  # Abre la cámara 0
cam.set(cv.CAP_PROP_FRAME_WIDTH, 640)  # [TP5] Ancho 640, igual que poseARUCO.py
cam.set(cv.CAP_PROP_FRAME_HEIGHT, 480)  # [TP5] Alto 480, igual que poseARUCO.py
width = cam.get(cv.CAP_PROP_FRAME_WIDTH)  # Lee el ancho real de la cámara
height = cam.get(cv.CAP_PROP_FRAME_HEIGHT)  # Lee el alto real de la cámara
print("Cam resolution:", width, " x ", height)  # Muestra la resolución (debe decir 640 x 480)
newSize = (640, int(640 * height / width))  # Tamaño reducido para la vista previa
imBlack = np.zeros(newSize[::-1]+(3,), dtype=np.uint8)  # Imagen negra donde se acumulan las detecciones

while True:  # Bucle principal
    ret, im = cam.read()  # Lee un cuadro
    if ret:  # Si se leyó bien
        imLowRes = cv.resize(im, newSize)  # Versión reducida para detectar rápido
        imGrayLowRes = cv.cvtColor(imLowRes, cv.COLOR_BGR2GRAY)  # Pasa a gris
        ret, corners = cv.findChessboardCorners(imGrayLowRes, chessBoard, None)  # Busca el tablero en baja resolución
        if ret:  # Si lo encontró
            cv.drawChessboardCorners(imLowRes, chessBoard, corners, ret)  # Dibuja las esquinas en la vista previa

    cv.imshow('Cam', imLowRes)  # Muestra la vista previa

    key = cv.waitKey(33)  # Espera una tecla ~33 ms
    if key>=0:  # Si se presionó algo
        key = chr(key)  # Código -> carácter
        print(key)  # Muestra la tecla
        match key:  # Según la tecla
            case ' ':  # Espacio: capturar
                # Repite la detección en alta resolución y la registra
                imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)  # Cuadro completo en gris
                ret, precisionCorners = cv.findChessboardCorners(imGray, chessBoard, None)  # Detecta en resolución completa
                if ret:  # Si lo encontró
                    precisionCorners = cv.cornerSubPix(imGray, precisionCorners, (11,11), (-1,-1), cornersubpixTerminationCriteria)  # Refina a subpíxel
                    imgPoints.append(precisionCorners)  # Guarda los puntos 2D
                    objPoints.append(chessboardPointCloud3D)  # Guarda los puntos 3D correspondientes

                    # Anota en baja resolución
                    imBlack = cv.convertScaleAbs(imBlack, alpha=gradualDarkness, beta=0)  # Oscurece capturas anteriores
                    cv.drawChessboardCorners(imBlack, chessBoard, corners, ret)  # Dibuja la nueva captura
                    cv.imshow("Detecciones", imBlack)  # Muestra el acumulado

                    print(len(imgPoints), "pictures taken")  # Cuenta de capturas

            case 'c':  # c: calibrar
                # Calibra
                ret, K, distCoef, rvecs, tvecs = cv.calibrateCamera(objPoints, imgPoints, im.shape[:2][::-1], None, None, flags=cv.CALIB_ZERO_TANGENT_DIST)  # Calcula K y distorsión (tangencial fija en 0)

                # Muestra resultados
                print("Coeficientes de distorsión (K1, K2, P1, P2, K3):", distCoef)  # Imprime distorsión -> copiar a distCoeffs del TP5

                np.set_printoptions(precision=2, suppress=True)  # Formato corto para la matriz
                print("Matriz K", K)  # Imprime K -> copiar a cameraMatrix del TP5
                np.set_printoptions(**defaultPrintOptions)  # Restaura el formato

            case ESC:  # ESC: salir
                print("Terminando.")  # Mensaje de cierre
                break  # Sale del bucle
