import cv2                    # OpenCV para leer imagen y dibujar resultados
import numpy as np           # operaciones matematicas
from joblib import load      # para cargar el modelo entrenado


# Modelo y etiquetas
ARCHIVO_MODELO = "modelo.joblib"
NOMBRES = {
    1: "Circulos",
    2: "Cuadrados",
    3: "Rectangulos",
    4: "Triangulos",
}
COLORES = {
    1: (50, 200, 50),
    2: (200, 100, 50),
    3: (50, 50, 220),
    4: (50, 100, 255),
}
COLOR_DESCONOCIDO = (128, 128, 128)
EPSILON_LOGARITMO = 1e-10
INDICE_PRIMERA_PREDICCION = 0

# Webcam y ventanas
INDICE_CAMARA = 0
VENTANA_CLASIFICACION = "Clasificacion ML"
VENTANA_BINARIA = "Binaria"
ESPERA_TECLA_MILISEGUNDOS = 1
MASCARA_CODIGO_TECLA = 0xFF
TECLA_ESCAPE = 27
TECLA_SALIR = ord("q")

# Procesamiento de imagen
CANAL_SATURACION = 1
TAMANIO_FILTRO_GAUSSIANO = (5, 5)
SIGMA_FILTRO_GAUSSIANO = 0
UMBRAL_SATURACION = 30
VALOR_BINARIO_MAXIMO = 255
TAMANIO_KERNEL_MORFOLOGICO = (5, 5)
AREA_MINIMA_CONTORNO = 500

# Dibujo del resultado
TODOS_LOS_CONTORNOS = -1
GROSOR_CONTORNO = 2
FUENTE_TEXTO = cv2.FONT_HERSHEY_SIMPLEX
ESCALA_TEXTO = 0.65
GROSOR_TEXTO = 2
COLOR_TEXTO = (255, 255, 255)
DESPLAZAMIENTO_CAJA_SUPERIOR = 12
DESPLAZAMIENTO_CAJA_DERECHA = 8
DESPLAZAMIENTO_CAJA_INFERIOR = 2
DESPLAZAMIENTO_TEXTO_X = 4
DESPLAZAMIENTO_TEXTO_Y = 5
RELLENO_RECTANGULO = -1
PESO_CAPA = 0.55
PESO_RESULTADO = 0.45
GAMMA_MEZCLA = 0


clasificador = load(ARCHIVO_MODELO)
print("Modelo cargado.")


def predecir_contorno(contorno):
    """Calcula los 7 Momentos de Hu y predice la clase."""
    momentos = cv2.moments(contorno)
    hu = cv2.HuMoments(momentos).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + EPSILON_LOGARITMO)
    prediccion = clasificador.predict([hu_log])[INDICE_PRIMERA_PREDICCION]
    etiqueta = int(prediccion)
    nombre = NOMBRES.get(etiqueta, "Desconocido")
    color = COLORES.get(etiqueta, COLOR_DESCONOCIDO)
    return nombre, color


camara = cv2.VideoCapture(INDICE_CAMARA)
if not camara.isOpened():
    raise SystemExit("ERROR: no se pudo abrir la webcam")

print("Webcam iniciada. Presiona q o ESC para cerrar.")

while True:
    lectura_correcta, frame = camara.read()
    if not lectura_correcta:
        print("ERROR: no se pudo leer un frame de la webcam")
        break

    # Preprocesamiento usando el canal de saturacion HSV.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturacion = hsv[:, :, CANAL_SATURACION]
    saturacion = cv2.GaussianBlur(saturacion, TAMANIO_FILTRO_GAUSSIANO, SIGMA_FILTRO_GAUSSIANO)
    _, binaria = cv2.threshold(saturacion, UMBRAL_SATURACION, VALOR_BINARIO_MAXIMO, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, TAMANIO_KERNEL_MORFOLOGICO)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_CLOSE, kernel)
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultado = frame.copy()

    for contorno in contornos:
        if cv2.contourArea(contorno) < AREA_MINIMA_CONTORNO:
            continue

        nombre, color = predecir_contorno(contorno)

        cv2.drawContours(resultado, [contorno], TODOS_LOS_CONTORNOS, color, GROSOR_CONTORNO)
        x, y, _, _ = cv2.boundingRect(contorno)
        (ancho_texto, alto_texto), _ = cv2.getTextSize(nombre, FUENTE_TEXTO, ESCALA_TEXTO, GROSOR_TEXTO)
        capa = resultado.copy()
        inicio_caja = (x, y - alto_texto - DESPLAZAMIENTO_CAJA_SUPERIOR)
        fin_caja = (x + ancho_texto + DESPLAZAMIENTO_CAJA_DERECHA, y - DESPLAZAMIENTO_CAJA_INFERIOR)
        posicion_texto = (x + DESPLAZAMIENTO_TEXTO_X, y - DESPLAZAMIENTO_TEXTO_Y)
        cv2.rectangle(capa, inicio_caja, fin_caja, color, RELLENO_RECTANGULO)
        cv2.addWeighted(capa, PESO_CAPA, resultado, PESO_RESULTADO, GAMMA_MEZCLA, resultado)
        cv2.putText(resultado, nombre, posicion_texto, FUENTE_TEXTO, ESCALA_TEXTO, COLOR_TEXTO, GROSOR_TEXTO)

    cv2.imshow(VENTANA_CLASIFICACION, resultado)
    cv2.imshow(VENTANA_BINARIA, binaria)

    tecla = cv2.waitKey(ESPERA_TECLA_MILISEGUNDOS) & MASCARA_CODIGO_TECLA
    if tecla in (TECLA_ESCAPE, TECLA_SALIR):
        break

camara.release()
cv2.destroyAllWindows()
