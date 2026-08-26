import cv2

#nro de mi webCam, hasta ahora solo abro mi camarita

cap = cv2.VideoCapture(0)

#esto es pasa

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
while True:

    ret, frame = cap.read()

    #esto es para

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostro = faceCascade.detectMultiScale(gray, 1.1, 4)

    #dibujamos el rectangulo 

    for x, y, w, h in rostro:

        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 255), 5)

        #cv2.rectangle(te digo el frame, las dimensiones horizontale, las izq, el color, y el grosor de 5)

        roi = frame[y: y + h, x: x + w]

    cv2.imshow("Video", frame)

    k = cv2.waitKey(1)

    if k == 27: #esto sirve para tocar esc y salir

        break

cap.release()
cv2.destroyAllWindows()