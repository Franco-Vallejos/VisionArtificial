"""Demo por lote con OpenCV: procesa videos, dibuja detecciones y guarda cada resultado en MP4."""
import logging  # logging: mensajes del pipeline
import time  # time.perf_counter: reloj de alta precisión para medir la latencia
from collections import deque  # deque: lista de largo fijo para promediar los FPS de los últimos frames
from pathlib import Path  # Path: rutas portables

import cv2  # OpenCV: cámara/video, dibujo y ventana
import numpy as np  # numpy: percentiles de la latencia
from ultralytics import YOLO  # YOLO: carga el modelo entrenado y hace la inferencia

from configuracion import Config  # la configuración única del proyecto


def dibujar_caja(imagen: np.ndarray, caja: tuple[float, float, float, float], texto: str, color: tuple[int, ...]) -> None:
    """Entrada: imagen, caja (x1, y1, x2, y2) en píxeles, texto y color BGR. Salida: nada (dibuja sobre la imagen).

    La usan la demo, el EDA y los ejemplos de errores: así todo se ve con los mismos colores.
    """
    x1, y1, x2, y2 = (int(valor) for valor in caja)  # OpenCV necesita coordenadas enteras
    grosor = max(2, imagen.shape[1] // 400)  # línea más gruesa en imágenes más grandes
    cv2.rectangle(imagen, (x1, y1), (x2, y2), color, grosor)  # dibuja el rectángulo de la caja
    (ancho_texto, alto_texto), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)  # mide el texto
    y_texto = max(y1, alto_texto + 4)  # si la caja toca el borde de arriba, el texto baja para que se vea
    cv2.rectangle(imagen, (x1, y_texto - alto_texto - 4), (x1 + ancho_texto + 4, y_texto), color, -1)  # fondo del texto (-1 = relleno)
    cv2.putText(imagen, texto, (x1 + 2, y_texto - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)  # texto negro


def buscar_videos(carpeta: Path) -> list[Path]:
    """Entrada: carpeta de videos. Salida: archivos compatibles ordenados y sin resultados procesados anteriormente."""
    if not carpeta.is_dir():  # si la carpeta de entrada no existe...
        raise FileNotFoundError(f"No existe la carpeta de videos: {carpeta}")  # ...frena con una ruta clara
    extensiones = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}  # formatos de video que OpenCV suele poder abrir
    sufijos_procesados = ("_detectado", "_etiquetado")  # evita volver a procesar salidas viejas que quedaron en la entrada
    videos = sorted(  # orden estable para que la demo siempre recorra los archivos igual
        (ruta for ruta in carpeta.iterdir() if ruta.is_file() and ruta.suffix.lower() in extensiones and not ruta.stem.lower().endswith(sufijos_procesados)),  # filtra videos originales
        key=lambda ruta: ruta.name.lower(),  # ordena sin distinguir mayúsculas de minúsculas
    )
    if not videos:  # si no se encontró ninguna entrada válida...
        raise FileNotFoundError(f"No hay videos compatibles para procesar en: {carpeta}")  # ...evita una ejecución vacía
    logging.info("Videos encontrados para la demo: %d", len(videos))  # informa el tamaño del lote
    return videos  # devuelve las entradas listas para procesar


def construir_ruta_salida(video: Path, carpeta_salida: Path) -> Path:
    """Entrada: video original y carpeta de salida. Salida: ruta MP4 con sufijo _etiquetado."""
    return carpeta_salida / f"{video.stem}_etiquetado.mp4"  # conserva el nombre original y distingue claramente el resultado


def abrir_fuente_de_video(fuente: Path) -> cv2.VideoCapture:
    """Entrada: ruta de un video. Salida: captura de OpenCV abierta."""
    captura = cv2.VideoCapture(str(fuente))  # abre el archivo de video indicado
    if not captura.isOpened():  # si no se pudo abrir...
        raise RuntimeError(f"No se pudo abrir la fuente de video: {fuente}")  # ...se frena con un mensaje claro
    logging.info("Fuente de video abierta: %s", fuente)  # informa qué fuente se usa
    return captura  # devuelve la captura lista para leer frames


def crear_escritor_video(captura: cv2.VideoCapture, frame: np.ndarray, ruta: Path) -> cv2.VideoWriter:
    """Entrada: captura, primer frame y destino. Salida: escritor MP4 abierto con FPS y tamaño originales."""
    ruta.parent.mkdir(parents=True, exist_ok=True)  # crea la carpeta de salida si todavía no existe
    fps_origen = captura.get(cv2.CAP_PROP_FPS)  # obtiene los FPS declarados por el video o la cámara
    fps_salida = fps_origen if np.isfinite(fps_origen) and fps_origen > 1 else 30.0  # usa 30 FPS si la fuente no informa un valor válido
    alto, ancho = frame.shape[:2]  # conserva exactamente la resolución que entra al detector
    codec = cv2.VideoWriter_fourcc(*"mp4v")  # códec MP4 ampliamente soportado por OpenCV
    escritor = cv2.VideoWriter(str(ruta), codec, fps_salida, (ancho, alto))  # abre el archivo de salida
    if not escritor.isOpened():  # comprueba que OpenCV tenga disponible el códec solicitado
        raise RuntimeError(f"No se pudo crear el video de salida: {ruta}")  # evita ejecutar toda la demo sin guardar nada
    logging.info("Grabando demo en: %s | %.2f FPS | %dx%d", ruta, fps_salida, ancho, alto)  # informa el destino y formato
    return escritor  # devuelve el escritor listo para recibir frames BGR


def dibujar_detecciones(frame: np.ndarray, resultado, config: Config) -> None:
    """Entrada: frame, resultado de YOLO y Config. Salida: nada. Dibuja cada caja con el color de su clase."""
    cajas = resultado.boxes.xyxy.tolist()  # cajas en píxeles (x1, y1, x2, y2); .tolist() las trae de la GPU a la CPU
    clases = resultado.boxes.cls.tolist()  # id de clase de cada caja
    confianzas = resultado.boxes.conf.tolist()  # confianza de cada caja
    for caja, clase, confianza in zip(cajas, clases, confianzas):  # recorre las detecciones
        nombre = resultado.names[int(clase)]  # cada modelo lleva su propio catálogo de clases
        color = tuple(config.colores_bgr.get(nombre, [0, 180, 255]))  # color de respaldo para clases nuevas
        dibujar_caja(frame, caja, f"{nombre} {confianza:.2f}", color)  # dibuja la detección


def dibujar_rendimiento(frame: np.ndarray, fps: float, latencia_ms: float) -> None:
    """Entrada: frame, FPS y latencia. Salida: nada. Escribe el rendimiento arriba a la izquierda."""
    texto = f"FPS: {fps:.1f} | latencia: {latencia_ms:.1f} ms"  # texto a mostrar
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)  # borde negro (legible en cualquier fondo)
    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)  # texto blanco encima


def resumir_latencias(latencias_ms: list[float], duraciones_s: list[float]) -> dict[str, float]:
    """Entrada: latencias de procesamiento (ms) y duración total de cada frame (s). Salida: p50/p95/p99 y FPS promedio."""
    if not latencias_ms:  # si no se midió ningún frame (video muy corto o se cerró enseguida)...
        logging.warning("No hay frames medidos para resumir")  # ...avisa
        return {}  # ...y devuelve un resumen vacío
    resumen = {  # métricas de tiempo real
        "frames_medidos": len(latencias_ms),  # cantidad de frames que entraron en la estadística
        "latencia_p50_ms": float(np.percentile(latencias_ms, 50)),  # la mitad de los frames tarda menos que esto
        "latencia_p95_ms": float(np.percentile(latencias_ms, 95)),  # el 95% tarda menos que esto
        "latencia_p99_ms": float(np.percentile(latencias_ms, 99)),  # el 99% tarda menos: muestra los "tirones"
        "fps_promedio": len(duraciones_s) / sum(duraciones_s),  # frames por segundo de punta a punta (lectura + inferencia + pantalla)
    }
    for clave, valor in resumen.items():  # muestra cada métrica
        logging.info("  %-16s %.1f", clave, valor)  # una por línea en el log
    return resumen  # devuelve el resumen


def procesar_video(modelo: YOLO, video: Path, salida: Path, config: Config, mostrar_en_vivo: bool) -> tuple[dict[str, float], bool]:
    """Entrada: modelo, video, salida, Config y visualización. Salida: métricas y si el usuario pidió detener el lote."""
    captura = abrir_fuente_de_video(video)  # abre el video actual del lote
    escritor: cv2.VideoWriter | None = None  # se crea al recibir el primer frame, cuando ya conocemos su tamaño
    latencias_ms: list[float] = []  # latencia de procesamiento de cada frame (inferencia + dibujo)
    duraciones_s: list[float] = []  # duración total de cada frame (para los FPS)
    ultimas = deque(maxlen=30)  # duraciones de los últimos 30 frames (FPS suavizados en pantalla)
    numero_frame = 0  # contador de frames
    detener_lote = False  # se vuelve True si el usuario presiona q en la vista en vivo
    try:  # try/finally: la cámara y las ventanas se liberan aunque haya un error
        while True:  # un ciclo por frame
            inicio_frame = time.perf_counter()  # arranca el reloj del frame completo
            leido, frame = captura.read()  # lee el siguiente frame
            if not leido:  # se terminó el video o se cortó la cámara...
                break  # ...sale del ciclo
            inicio_proceso = time.perf_counter()  # arranca el reloj de procesamiento
            resultado = modelo.predict(  # inferencia en la GPU
                frame,  # el frame actual
                conf=config.umbral_confianza,  # descarta detecciones con poca confianza
                imgsz=config.imgsz,  # mismo tamaño que en el entrenamiento
                device=config.dispositivo,  # GPU configurada
                quantize=16 if config.media_precision else 32,  # FP16 o FP32 según la configuración
                verbose=False,  # sin imprimir una línea por frame
            )[0]  # predict devuelve una lista (una por imagen): se toma la única
            dibujar_detecciones(frame, resultado, config)  # dibuja las cajas (fuerza a esperar a la GPU)
            latencia_ms = (time.perf_counter() - inicio_proceso) * 1000  # latencia de procesamiento en milisegundos
            fps = len(ultimas) / sum(ultimas) if ultimas else 0.0  # FPS promedio de los últimos frames
            dibujar_rendimiento(frame, fps, latencia_ms)  # escribe FPS y latencia sobre el frame
            if escritor is None:  # el primer frame define resolución y FPS del archivo final
                escritor = crear_escritor_video(captura, frame, salida)  # abre una grabación independiente para el video actual
            escritor.write(frame)  # guarda el frame después de dibujar detecciones, FPS y latencia
            tecla = -1  # en modo sin ventana no se consulta el teclado ni se crea una interfaz gráfica
            if mostrar_en_vivo:  # solo abre la ventana en el modo interactivo predeterminado
                cv2.imshow(f"Detector de transito - {video.name} (q para salir)", frame)  # muestra el video actual con su nombre
                tecla = cv2.waitKey(1) & 0xFF  # espera una tecla y permite que OpenCV actualice la ventana
            duracion_s = time.perf_counter() - inicio_frame  # duración total del frame
            ultimas.append(duracion_s)  # la agrega a la ventana de FPS
            numero_frame += 1  # cuenta el frame
            if numero_frame > config.frames_calentamiento:  # los primeros frames son lentos (la GPU arranca): no cuentan
                latencias_ms.append(latencia_ms)  # guarda la latencia
                duraciones_s.append(duracion_s)  # guarda la duración
            if tecla == ord("q"):  # si se apretó q...
                detener_lote = True  # recuerda que también deben omitirse los videos siguientes
                break  # ...sale del ciclo
    finally:  # siempre, haya error o no:
        captura.release()  # libera la cámara o el archivo
        if escritor is not None:  # si llegó a crearse el archivo de salida...
            escritor.release()  # ...escribe el índice final y cierra correctamente el MP4
        if mostrar_en_vivo:  # si se creó una ventana...
            cv2.destroyAllWindows()  # ...la cierra antes de pasar al video siguiente
    if escritor is None:  # si el archivo no entregó ningún frame válido...
        raise RuntimeError(f"El video no contiene frames legibles: {video}")  # ...no informa falsamente que se generó una salida
    logging.info("Video procesado guardado en: %s", salida)  # confirma dónde quedó la grabación
    logging.info("Resumen de tiempo real (%s):", video.name)  # encabezado del resumen del archivo actual
    return resumir_latencias(latencias_ms, duraciones_s), detener_lote  # devuelve métricas y la orden de detener el lote


def detectar_en_tiempo_real(pesos: Path, config: Config, mostrar_en_vivo: bool = True) -> dict[str, dict[str, float]]:
    """Entrada: pesos, Config y visualización. Salida: métricas por video procesado."""
    modelo = YOLO(str(pesos))  # carga el modelo una sola vez para reutilizarlo en todo el lote
    videos = buscar_videos(config.carpeta_videos)  # descubre todos los videos originales de la carpeta configurada
    config.carpeta_videos_procesados.mkdir(parents=True, exist_ok=True)  # crea la carpeta separada de resultados
    resumenes: dict[str, dict[str, float]] = {}  # junta las métricas usando el nombre de cada video como clave
    for indice, video in enumerate(videos, start=1):  # procesa los archivos uno por uno en orden alfabético
        salida = construir_ruta_salida(video, config.carpeta_videos_procesados)  # genera un destino independiente para este video
        logging.info("Procesando video %d/%d: %s", indice, len(videos), video.name)  # muestra el progreso del lote
        resumen, detener_lote = procesar_video(modelo, video, salida, config, mostrar_en_vivo)  # ejecuta inferencia, overlay y grabación
        resumenes[video.name] = resumen  # conserva las métricas del archivo recién terminado
        if detener_lote:  # si el usuario presionó q durante la visualización...
            logging.info("Procesamiento por lote detenido por el usuario")  # informa por qué no se recorrieron los demás archivos
            break  # ...no abre el siguiente video
    logging.info("Videos procesados: %d | carpeta de salida: %s", len(resumenes), config.carpeta_videos_procesados)  # resume la corrida completa
    return resumenes  # devuelve todas las métricas para posibles usos posteriores
