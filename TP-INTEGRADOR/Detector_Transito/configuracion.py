"""Configuración y entorno: carga config.yaml en un objeto Config, fija semillas y verifica la GPU.

Es el único lugar donde se lee la configuración; después el objeto Config se pasa
explícitamente como parámetro a cada función del pipeline (cero variables globales).
"""
import logging  # logging: mensajes con hora y nivel (INFO, WARNING, ERROR) en vez de print sueltos
import random  # random: generador de números al azar de Python (se usa en el split)
from dataclasses import dataclass  # dataclass: arma una clase de datos sin escribir el __init__ a mano
from pathlib import Path  # Path: manejo de rutas portable entre Windows y Linux

import numpy as np  # numpy: cálculo numérico (también tiene su propio generador al azar)
import torch  # torch: PyTorch, el motor sobre el que corre YOLO (y quien maneja la GPU)
import yaml  # yaml: lee y escribe archivos .yaml (config.yaml, data.yaml)


@dataclass(frozen=True)  # frozen=True: nadie puede modificar la configuración por accidente durante la corrida
class Config:
    """Todos los parámetros del proyecto. Cada campo corresponde a una clave de config.yaml."""

    semilla: int  # semilla única para random, numpy, torch, el split y el entrenamiento
    carpeta_zips: Path  # carpeta con uno o más ZIPs exportados en formato YOLO
    zip_dataset_base: Path  # zip con el dataset público ya unificado (BDD100K + LISA + fotos propias)
    carpeta_dataset_base: Path  # carpeta donde se descomprime ese zip
    carpeta_datos_propios: Path  # carpeta con imágenes nuevas etiquetadas a mano (exportadas en formato YOLO)
    carpeta_dataset_final: Path  # carpeta donde se escribe el dataset final dividido en train/val/test
    carpeta_reportes: Path  # carpeta para el EDA (tablas, gráficos y ejemplos)
    carpeta_experimentos: Path  # carpeta donde cada entrenamiento guarda su versión (v1, v2, ...)
    clases: list[str]  # clases finales del detector, en orden (el índice es el id YOLO)
    mapeo_clases: dict[str, str]  # nombre de clase de origen -> nombre de clase final (lo que no está, se descarta)
    patrones_grupo: list[str]  # expresiones regulares que sacan el "grupo" (video/clip) del nombre del archivo
    proporciones_split: dict[str, float]  # proporción de imágenes para train, val y test
    sobremuestreo: dict[str, int]  # clase minoritaria -> cuántas veces aparece en train cada imagen que la contiene
    umbral_desbalance: float  # una clase con menos de esta fracción de la clase mayoritaria se marca como minoritaria
    ejemplos_eda: int  # cantidad de imágenes de ejemplo que se dibujan en el EDA
    modelo_base: str  # pesos preentrenados de partida (transfer learning)
    epochs: int  # cantidad máxima de pasadas completas por train
    batch: int  # imágenes por paso de entrenamiento
    cache: bool | str  # False, "ram" o "disk": cache de imágenes para acelerar las épocas
    batch_evaluacion: int  # lote acotado para validación/test; no usa el batch automático de entrenamiento
    imgsz: int  # lado (en píxeles) al que se redimensiona cada imagen
    optimizador: str  # algoritmo que actualiza los pesos (SGD respeta lr0)
    lr0: float  # learning rate inicial
    patience: int  # early stopping: épocas sin mejorar en val antes de cortar
    workers: int  # procesos que cargan imágenes en paralelo
    dispositivo: int | str  # 0 = primera GPU, 1 = segunda...; "cpu" = sin GPU (mucho más lento)
    aumentos: dict[str, float]  # data augmentation de Ultralytics (solo se aplica en train)
    umbral_confianza: float  # confianza mínima para aceptar una detección (demo y análisis de errores)
    umbral_iou: float  # superposición mínima para considerar que una predicción "encontró" un objeto real
    umbrales_a_probar: list[float]  # umbrales de confianza para la curva precision vs. recall
    ejemplos_errores: int  # cuántas imágenes con errores se guardan por tipo de error
    exportar_onnx: bool  # además de best.pt, exportar el modelo a ONNX
    carpeta_videos: Path  # carpeta de entrada con todos los videos que procesa la demo
    carpeta_videos_procesados: Path  # carpeta de salida para los MP4 con detecciones y etiquetas
    media_precision: bool  # FP16 en la GPU durante la demo (más rápido, misma calidad en la práctica)
    frames_calentamiento: int  # primeros frames que no cuentan en la latencia (la GPU recién arranca)
    colores_bgr: dict[str, list[int]]  # color de la caja de cada clase en formato BGR de OpenCV
    umbral_confianza_validacion: float = 0.20  # confianza mínima de la segunda evaluación en val


def configurar_logging() -> None:
    """Entrada: nada. Salida: nada. Deja configurado el formato de los mensajes de log para todo el pipeline."""
    logging.basicConfig(  # configura el logger raíz una sola vez
        level=logging.INFO,  # muestra INFO, WARNING y ERROR (oculta DEBUG)
        format="%(asctime)s | %(levelname)-7s | %(message)s",  # hora | nivel | mensaje
        datefmt="%H:%M:%S",  # la hora sin fecha, para que el log sea corto
    )


def cargar_configuracion(ruta: Path) -> Config:
    """Entrada: ruta a config.yaml. Salida: objeto Config. Es el primer paso del pipeline."""
    if not ruta.is_file():  # si el archivo de configuración no existe...
        raise FileNotFoundError(f"No existe el archivo de configuración: {ruta}")  # ...se corta con un mensaje claro
    datos = yaml.safe_load(ruta.read_text(encoding="utf-8"))  # lee el YAML y lo convierte en un diccionario
    for clave in datos:  # recorre todas las claves del YAML
        if clave.startswith(("carpeta_", "zip_")):  # las claves de rutas empiezan con "carpeta_" o "zip_"
            datos[clave] = (ruta.parent / datos[clave]).resolve()  # ruta relativa a config.yaml -> ruta absoluta
    config = Config(**datos)  # crea el Config (falla si falta o sobra alguna clave: así se detectan errores de tipeo)
    logging.info("Configuración cargada desde %s", ruta)  # deja constancia de qué archivo se usó
    return config  # devuelve la configuración para pasarla al resto del pipeline


def fijar_semillas(semilla: int) -> None:
    """Entrada: semilla. Salida: nada. Hace reproducible todo lo que usa azar (split, pesos iniciales, aumentos)."""
    random.seed(semilla)  # fija el azar de Python
    np.random.seed(semilla)  # fija el azar de numpy
    torch.manual_seed(semilla)  # fija el azar de PyTorch en CPU
    torch.cuda.manual_seed_all(semilla)  # fija el azar de PyTorch en todas las GPUs
    logging.info("Semilla fijada en %d", semilla)  # informa la semilla usada


def verificar_gpu(dispositivo: int | str) -> str:
    """Entrada: dispositivo de config.yaml. Salida: nombre de la GPU (o "cpu"). Frena si se pidió GPU y no hay."""
    if str(dispositivo).lower() == "cpu":  # si en config.yaml se eligió CPU a propósito...
        logging.warning("Se usa CPU (dispositivo: cpu en config.yaml): entrenar va a ser MUY lento")  # ...avisa
        return "cpu"  # ...y sigue sin verificar la GPU
    if not torch.cuda.is_available():  # True solo si hay GPU NVIDIA, driver y un PyTorch compilado con CUDA
        raise RuntimeError(  # se frena: entrenar en CPU tardaría días
            "No hay GPU con CUDA disponible. Instalá PyTorch con CUDA (ver README) y verificá el driver NVIDIA."
        )
    nombre = torch.cuda.get_device_name(dispositivo)  # nombre comercial de la GPU elegida
    memoria_gb = torch.cuda.get_device_properties(dispositivo).total_memory / 1024**3  # memoria de la GPU en GB
    logging.info("GPU detectada: %s (%.1f GB) | CUDA %s", nombre, memoria_gb, torch.version.cuda)  # muestra la GPU
    return nombre  # devuelve el nombre por si se quiere guardar en el experimento
