"""Fase de modelo: versionado de experimentos, entrenamiento, curvas, umbral (en val), evaluación (en test) y exportación."""
import csv  # csv: lee results.csv (métricas por época que escribe Ultralytics)
import gc  # gc: libera referencias cíclicas del entrenador antes de evaluar
import json  # json: guarda las métricas finales
import logging  # logging: mensajes del pipeline
import re  # re: reconoce las carpetas de versión (v1_..., v2_...)
import shutil  # shutil: copia archivos
from dataclasses import asdict  # asdict: convierte el Config en diccionario para guardarlo
from datetime import datetime  # datetime: fecha y hora en el nombre de cada experimento
from pathlib import Path  # Path: rutas portables

import cv2  # OpenCV: guarda las imágenes de ejemplos de errores
import matplotlib  # matplotlib: gráficos

matplotlib.use("Agg")  # dibuja a archivo, sin abrir ventanas
import matplotlib.pyplot as plt  # noqa: E402  interfaz de gráficos
import numpy as np  # noqa: E402  matrices y cálculos
import torch  # noqa: E402  libera la memoria CUDA que dejó el entrenamiento
import yaml  # noqa: E402  guarda la configuración usada
from ultralytics import YOLO  # noqa: E402  modelo YOLO de Ultralytics

from configuracion import Config  # noqa: E402  configuración única del proyecto
from datos import buscar_etiqueta, es_imagen, leer_imagen  # noqa: E402  utilidades de lectura de datos
from tiempo_real import dibujar_caja  # noqa: E402  mismo dibujo que la demo

# Cada "par" de la evaluación es una tupla: (clase_real, clase_predicha, confianza, caja_real, caja_predicha)
# clase_real = -1  -> falso positivo (se predijo algo donde no había nada)
# clase_predicha = -1 -> falso negativo (había un objeto y no se detectó)


def liberar_memoria_gpu() -> None:
    """Libera objetos sin referencias y la caché CUDA antes de cambiar de fase."""
    gc.collect()  # elimina ciclos de referencias del modelo, trainer y dataloaders
    if torch.cuda.is_available():  # solo existe caché CUDA cuando se usa una GPU NVIDIA
        torch.cuda.empty_cache()  # devuelve al asignador la VRAM reservada que ya no está en uso


def crear_carpeta_experimento(config: Config) -> Path:
    """Entrada: Config. Salida: carpeta nueva "vN_fecha_hora" para guardar todo lo de esta corrida."""
    config.carpeta_experimentos.mkdir(parents=True, exist_ok=True)  # crea la carpeta de experimentos si no existe
    versiones = [  # números de versión ya usados
        int(re.match(r"v(\d+)_", carpeta.name).group(1))  # saca el número de "v3_2026-..."
        for carpeta in config.carpeta_experimentos.iterdir()  # recorre lo que hay adentro
        if carpeta.is_dir() and re.match(r"v\d+_", carpeta.name)  # solo carpetas con nombre de versión
    ]
    carpeta = config.carpeta_experimentos / f"v{max(versiones, default=0) + 1}_{datetime.now():%Y-%m-%d_%H%M}"  # siguiente versión
    carpeta.mkdir()  # la crea
    logging.info("Experimento: %s", carpeta.name)  # informa la versión
    return carpeta  # devuelve la carpeta del experimento


def guardar_configuracion_del_experimento(config: Config, data_yaml: Path, carpeta: Path) -> None:
    """Entrada: Config, data.yaml y carpeta del experimento. Salida: nada. Guarda hiperparámetros y huella de los datos."""
    datos = {clave: str(valor) if isinstance(valor, Path) else valor for clave, valor in asdict(config).items()}  # Path -> texto
    (carpeta / "config_usada.yaml").write_text(yaml.safe_dump(datos, allow_unicode=True, sort_keys=False), encoding="utf-8")  # hiperparámetros
    shutil.copy2(data_yaml, carpeta / "data_usado.yaml")  # guarda el descriptor exacto del entrenamiento
    manifiesto = data_yaml.with_name("manifiesto.json")  # manifiesto del pipeline de preparación
    if not manifiesto.is_file():  # el dataset argentino conserva su procedencia en un índice
        manifiesto = data_yaml.with_name("indice.json")  # usa ese índice como registro del experimento
    if manifiesto.is_file():  # el manifiesto es metadato adicional al descriptor YOLO
        shutil.copy2(manifiesto, carpeta / "manifiesto_datos.json")  # guarda la procedencia de los datos


def entrenar_detector(data_yaml: Path, carpeta: Path, config: Config) -> Path:
    """Entrada: data.yaml, carpeta del experimento y Config. Salida: ruta de best.pt (mejor época según val).

    Transfer learning: parte de pesos preentrenados en COCO y los ajusta a nuestras clases.
    """
    modelo = YOLO(config.modelo_base)  # carga el modelo base preentrenado (lo descarga la primera vez)
    modelo.train(  # entrena; Ultralytics aplica los aumentos SOLO en train, nunca en val
        data=str(data_yaml),  # dataset final
        epochs=config.epochs,  # máximo de épocas
        batch=config.batch,  # imágenes por paso
        cache=config.cache,  # cachea en disco para no cargar todo el dataset en RAM
        imgsz=config.imgsz,  # tamaño de entrada
        optimizer=config.optimizador,  # SGD: con "auto" Ultralytics puede ignorar lr0
        lr0=config.lr0,  # learning rate inicial
        patience=config.patience,  # early stopping
        workers=config.workers,  # procesos de carga de imágenes
        device=config.dispositivo,  # GPU
        seed=config.semilla,  # misma semilla que el resto del pipeline
        deterministic=True,  # operaciones deterministas: misma corrida, mismo resultado (o casi, en GPU)
        project=str(carpeta),  # los resultados quedan dentro de la carpeta del experimento
        name="entrenamiento",  # subcarpeta del entrenamiento
        exist_ok=True,  # no crea "entrenamiento2" si ya existe
        plots=True,  # guarda gráficos de Ultralytics (results.png, ejemplos de batches)
        **config.aumentos,  # data augmentation (hsv_h=0 para no cambiar el color del semáforo)
    )
    pesos = Path(modelo.trainer.save_dir) / "weights" / "best.pt"  # Ultralytics guarda la mejor época como best.pt
    logging.info("Entrenamiento terminado. Mejores pesos: %s", pesos)  # informa dónde quedaron
    del modelo  # elimina el modelo de entrenamiento antes de cargar best.pt para evaluar
    liberar_memoria_gpu()  # evita conservar la VRAM ocupada por el trainer y sus lotes
    return pesos  # devuelve la ruta de los mejores pesos


def leer_resultados_por_epoca(carpeta_entrenamiento: Path) -> list[dict[str, float]]:
    """Entrada: carpeta del entrenamiento. Salida: una fila por época con pérdidas y métricas (de results.csv)."""
    with open(carpeta_entrenamiento / "results.csv", newline="", encoding="utf-8") as archivo:  # abre el CSV
        return [  # una fila por época
            {clave.strip(): float(valor) for clave, valor in fila.items() if valor and valor.strip()}  # quita espacios y pasa a número
            for fila in csv.DictReader(archivo)  # DictReader usa el encabezado como claves
        ]


def diagnosticar_ajuste(filas: list[dict[str, float]]) -> str:
    """Entrada: filas por época. Salida: diagnóstico en texto (overfitting, posible underfitting o ajuste razonable)."""
    val = [f["val/box_loss"] + f["val/cls_loss"] for f in filas]  # pérdida de validación por época
    train = [f["train/box_loss"] + f["train/cls_loss"] for f in filas]  # pérdida de entrenamiento por época
    mejor = int(np.argmin(val))  # época con menor pérdida en val
    if len(val) - 1 - mejor >= 5 and val[-1] > 1.05 * val[mejor] and train[-1] < train[mejor]:  # train baja y val sube
        diagnostico = f"Overfitting: val empeoró desde la época {mejor + 1} mientras train siguió bajando (se usa best.pt)"
    elif len(train) >= 6 and train[-1] < 0.95 * train[-6]:  # train todavía bajaba más de 5% en las últimas 5 épocas
        diagnostico = "Todavía aprendía al terminar: posible underfitting -> más épocas o un modelo más grande"
    else:  # ni una cosa ni la otra
        diagnostico = "Ajuste razonable: train y val se estabilizaron juntas"
    logging.info("Diagnóstico de ajuste: %s", diagnostico)  # lo muestra
    return diagnostico  # lo devuelve para guardarlo


def graficar_curvas_de_perdida(carpeta_entrenamiento: Path, carpeta: Path) -> Path:
    """Entrada: carpeta del entrenamiento y del experimento. Salida: ruta del gráfico de pérdidas train vs. val."""
    filas = leer_resultados_por_epoca(carpeta_entrenamiento)  # lee las métricas por época
    epocas = [f["epoch"] for f in filas]  # eje X
    perdidas = [p for p in ("box_loss", "cls_loss", "dfl_loss", "l1_loss") if f"train/{p}" in filas[0]]  # incluye L1 de YOLO26
    figura, ejes = plt.subplots(1, len(perdidas), figsize=(5 * len(perdidas), 4))  # un gráfico por pérdida
    for eje, perdida in zip(np.atleast_1d(ejes), perdidas):  # recorre cada pérdida
        eje.plot(epocas, [f[f"train/{perdida}"] for f in filas], color="#1464a0", label="Entrenamiento")  # curva de train
        eje.plot(epocas, [f[f"val/{perdida}"] for f in filas], color="#bd5200", linestyle="--", label="Validación")  # curva de val
        eje.set_title(perdida)  # nombre de la pérdida
        eje.set_xlabel("época")  # eje X
        eje.legend()  # leyenda
    figura.suptitle("Pérdidas: si val sube mientras train baja, hay overfitting")  # recordatorio para la defensa
    figura.tight_layout()  # ajusta márgenes
    ruta = carpeta / "curvas_de_perdida.png"  # archivo de salida
    figura.savefig(ruta, dpi=120)  # guarda el gráfico
    plt.close(figura)  # libera memoria
    (carpeta / "diagnostico_ajuste.txt").write_text(diagnosticar_ajuste(filas) + "\n", encoding="utf-8")  # guarda el diagnóstico
    return ruta  # devuelve la ruta del gráfico


def listar_imagenes(carpeta: Path) -> list[Path]:
    """Entrada: carpeta. Salida: imágenes que contiene, ordenadas."""
    return sorted(ruta for ruta in carpeta.iterdir() if es_imagen(ruta))  # solo archivos de imagen, en orden fijo


def leer_cajas_reales(etiqueta: Path | None) -> list[tuple[int, tuple[float, float, float, float]]]:
    """Entrada: .txt YOLO. Salida: lista de (clase, (x1, y1, x2, y2)) normalizados entre 0 y 1."""
    if etiqueta is None:  # imagen sin archivo de etiquetas...
        return []  # ...no tiene objetos
    cajas = []  # acá se juntan las cajas
    for linea in etiqueta.read_text(encoding="utf-8").splitlines():  # una caja por línea
        if linea.strip():  # ignora líneas vacías
            clase, xc, yc, w, h = (float(v) for v in linea.split())  # separa los valores
            cajas.append((int(clase), (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)))  # centro+tamaño -> esquinas
    return cajas  # devuelve las cajas reales


def calcular_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Entrada: dos cajas (x1, y1, x2, y2). Salida: IoU = área de intersección / área de unión (0 a 1)."""
    ancho = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))  # ancho de la intersección (0 si no se tocan)
    alto = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))  # alto de la intersección
    interseccion = ancho * alto  # área de la intersección
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - interseccion  # área de la unión
    return interseccion / union if union > 0 else 0.0  # IoU (protege contra división por cero)


def emparejar_predicciones(reales: list, predichas: list, umbral_iou: float) -> list[tuple]:
    """Entrada: cajas reales, predicciones (clase, confianza, caja) ordenadas por confianza y umbral IoU.

    Salida: lista de pares (clase_real, clase_predicha, confianza, caja_real, caja_predicha).
    Empareja sin mirar la clase: así se detecta cuando un rojo se predijo como verde.
    """
    libres = set(range(len(reales)))  # objetos reales todavía no encontrados
    pares = []  # resultado
    for clase_p, confianza, caja_p in predichas:  # de la predicción más confiable a la menos confiable
        candidatos = [(calcular_iou(reales[i][1], caja_p), i) for i in libres]  # IoU con cada objeto libre
        iou, indice = max(candidatos, default=(0.0, -1))  # el objeto que más se superpone
        if iou >= umbral_iou:  # si se superpone lo suficiente...
            libres.discard(indice)  # ...ese objeto queda encontrado
            pares.append((reales[indice][0], clase_p, confianza, reales[indice][1], caja_p))  # par real-predicho
        else:  # si no se superpone con ningún objeto...
            pares.append((-1, clase_p, confianza, None, caja_p))  # ...es un falso positivo
    pares += [(reales[i][0], -1, 0.0, reales[i][1], None) for i in sorted(libres)]  # objetos no encontrados: falsos negativos
    return pares  # devuelve los pares


def predecir_y_emparejar(modelo: YOLO, imagenes: list[Path], config: Config) -> list[dict]:
    """Entrada: modelo, imágenes y Config. Salida: por imagen, {"imagen": ruta, "pares": [...]}.

    Predice una sola vez con el umbral más bajo; después cualquier umbral más alto se simula filtrando.
    """
    resultados = []  # resultado por imagen
    umbral_minimo = min((config.umbral_confianza_validacion, *config.umbrales_a_probar))
    for inicio in range(0, len(imagenes), config.batch_evaluacion):  # divide la lista para no convertir todo val/test en un lote gigante
        lote = imagenes[inicio:inicio + config.batch_evaluacion]  # lote acotado por la configuración de evaluación
        predicciones = modelo.predict(  # predice únicamente este lote y libera sus tensores al avanzar
            source=[str(imagen) for imagen in lote], stream=True, batch=config.batch_evaluacion,  # nunca entrega toda la lista junta
            conf=umbral_minimo, imgsz=config.imgsz, device=config.dispositivo, verbose=False,
        )
        for resultado in predicciones:  # procesa cada imagen del lote antes de solicitar el siguiente
            imagen = Path(resultado.path)  # ruta de la imagen procesada
            predichas = sorted(  # predicciones ordenadas por confianza (de mayor a menor)
                zip(map(int, resultado.boxes.cls.tolist()), resultado.boxes.conf.tolist(), map(tuple, resultado.boxes.xyxyn.tolist())),  # (clase, confianza, caja)
                key=lambda p: p[1], reverse=True,  # ordena por la confianza
            )
            reales = leer_cajas_reales(buscar_etiqueta(imagen))  # cajas reales de esa imagen
            resultados.append({"imagen": imagen, "pares": emparejar_predicciones(reales, predichas, config.umbral_iou)})  # guarda
    return resultados  # devuelve todos los resultados


def aplicar_umbral(pares: list[tuple], umbral: float) -> list[tuple]:
    """Entrada: pares y umbral de confianza. Salida: pares como quedarían con ese umbral."""
    filtrados = []  # resultado
    for real, pred, confianza, caja_r, caja_p in pares:  # recorre cada par
        if pred != -1 and confianza < umbral:  # la predicción no supera el umbral: desaparece
            if real == -1:  # era un falso positivo...
                continue  # ...ya no existe
            pred, caja_p = -1, None  # había un objeto real: pasa a ser falso negativo
        filtrados.append((real, pred, confianza, caja_r, caja_p))  # guarda el par
    return filtrados  # devuelve los pares filtrados


def construir_matriz_confusion(resultados: list[dict], umbral: float, cantidad_clases: int) -> np.ndarray:
    """Entrada: resultados, umbral y cantidad de clases. Salida: matriz [real, predicha]; la última fila/columna es "fondo"."""
    fondo = cantidad_clases  # índice extra para "nada" (falsos positivos y negativos)
    matriz = np.zeros((cantidad_clases + 1, cantidad_clases + 1), dtype=int)  # matriz de ceros
    for resultado in resultados:  # cada imagen
        for real, pred, _, _, _ in aplicar_umbral(resultado["pares"], umbral):  # cada par con este umbral
            matriz[real if real != -1 else fondo, pred if pred != -1 else fondo] += 1  # suma en la celda [real, predicha]
    return matriz  # devuelve la matriz


def precision_y_recall_totales(matriz: np.ndarray) -> tuple[float, float]:
    """Entrada: matriz de confusión. Salida: (precision, recall) sumando todas las clases."""
    aciertos = np.trace(matriz[:-1, :-1])  # diagonal sin "fondo": clase correcta
    predichas = matriz[:, :-1].sum()  # todo lo que el modelo predijo
    reales = matriz[:-1, :].sum()  # todos los objetos reales
    return aciertos / max(predichas, 1), aciertos / max(reales, 1)  # precision = aciertos/predichas; recall = aciertos/reales


def metricas_por_clase_a_umbral(matriz: np.ndarray, clases: list[str]) -> dict:
    metricas = {}
    for indice, clase in enumerate(clases):
        tp = int(matriz[indice, indice])
        predichas = int(matriz[:, indice].sum())
        reales = int(matriz[indice, :].sum())
        metricas[clase] = {
            "soporte": reales,
            "tp": tp,
            "fp": predichas - tp,
            "fn": reales - tp,
            "precision": tp / predichas if predichas else None,
            "recall": tp / reales if reales else None,
        }
    return metricas


def analizar_umbral_confianza(modelo: YOLO, data_yaml: Path, carpeta: Path, config: Config) -> Path:
    """Entrada: modelo, data.yaml, carpeta del experimento y Config. Salida: gráfico precision vs. recall por umbral.

    Se hace en VAL (no en test): elegir el umbral mirando test sería "hacer trampa" con test.
    """
    resultados = predecir_y_emparejar(modelo, listar_imagenes(data_yaml.parent / "images" / "val"), config)  # predice en val
    filas, precisiones, recalls = ["umbral,precision,recall"], [], []  # tabla y series del gráfico
    for umbral in config.umbrales_a_probar:  # prueba cada umbral
        precision, recall = precision_y_recall_totales(construir_matriz_confusion(resultados, umbral, len(config.clases)))  # métricas
        precisiones.append(precision)  # guarda la precision
        recalls.append(recall)  # guarda el recall
        filas.append(f"{umbral},{precision:.4f},{recall:.4f}")  # fila de la tabla
        logging.info("  umbral %.2f -> precision %.3f | recall %.3f", umbral, precision, recall)  # la muestra
    (carpeta / "umbral_confianza_val.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")  # guarda la tabla
    figura, eje = plt.subplots(figsize=(7, 4))  # crea la figura
    eje.plot(config.umbrales_a_probar, precisiones, marker="o", label="precision")  # curva de precision
    eje.plot(config.umbrales_a_probar, recalls, marker="o", label="recall")  # curva de recall
    eje.axvline(config.umbral_confianza, color="gray", linestyle="--", label="umbral elegido")  # umbral de config.yaml
    eje.set_xlabel("umbral de confianza")  # eje X
    eje.set_title("Validación: subir el umbral sube la precision y baja el recall")  # título
    eje.legend()  # leyenda
    figura.tight_layout()  # ajusta márgenes
    ruta = carpeta / "umbral_confianza_val.png"  # archivo de salida
    figura.savefig(ruta, dpi=120)  # guarda el gráfico
    plt.close(figura)  # libera memoria
    return ruta  # devuelve la ruta


def graficar_curvas_validacion(metricas, carpeta: Path, config: Config, confianza_minima: float) -> None:
    """Guarda las curvas por clase con los mismos colores y estilos en ambas validaciones."""
    colores = (*matplotlib.colormaps["tab20"].colors, *matplotlib.colormaps["Dark2"].colors)  # colores categóricos
    estilos = ("-", "--", "-.", ":")  # distingue curvas incluso cuando se repite un color
    ejes = {"Confidence": "Confianza", "Precision": "Precisión", "Recall": "Recall", "F1": "F1"}  # rótulos
    nombres_archivos = ("BoxPR_curve", "BoxF1_curve", "BoxP_curve", "BoxR_curve")  # reemplaza los gráficos estándar
    for nombre, (x, series, eje_x, eje_y) in zip(nombres_archivos, metricas.curves_results, strict=True):  # cuatro curvas
        if not len(series):  # no hay curvas si el modelo no acertó ningún objeto
            continue  # conserva los reportes numéricos y la matriz de confusión
        figura, eje = plt.subplots(figsize=(11, 8), layout="constrained")  # espacio para la leyenda
        for clase, curva in zip(metricas.box.ap_class_index, series, strict=True):  # las filas siguen el orden de las clases evaluadas
            clase = int(clase)  # índice YOLO
            eje.plot(x, curva, color=colores[clase % len(colores)], linestyle=estilos[clase % len(estilos)],
                     label=config.clases[clase], linewidth=1.5)  # color y estilo estables por ID
        if eje_x == "Confidence":  # indica el punto operativo en las curvas de confianza
            umbral = config.umbral_confianza_validacion  # valor configurado para la segunda pasada
            eje.axvline(umbral, color="#222222", linestyle="--", label=f"Umbral {umbral:.2f}")  # marca el corte
        eje.set(xlabel=ejes[eje_x], ylabel=ejes[eje_y], xlim=(0, 1), ylim=(0, 1),
                title=f"Validación por clase · confianza mínima {confianza_minima:g}")  # identifica la pasada
        eje.grid(alpha=0.2)  # facilita leer las coordenadas
        eje.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, title="Clase")  # fuera del gráfico
        figura.savefig(carpeta / f"{nombre}.png", dpi=145)  # salida junto a las métricas
        plt.close(figura)  # libera la figura


def validar_detector(modelo: YOLO, data_yaml: Path, carpeta: Path, config: Config) -> dict:
    """Compara AP estándar y el punto operativo con confianza fija sobre el split val."""
    umbral = config.umbral_confianza_validacion  # umbral independiente de la demo
    if list(modelo.names.values()) != config.clases:  # las clases deben coincidir por ID
        raise ValueError("Las clases de los pesos no coinciden con las del dataset de validación")  # evita leyendas incorrectas
    if not 0.001 < umbral < 1:  # la segunda pasada debe tener un corte mayor a la estándar
        raise ValueError("umbral_confianza_validacion debe estar entre 0.001 y 1")  # evita una comparación inválida
    destino = carpeta / "validacion"  # reportes de val del experimento
    resumen = {"split": "val"}  # mantiene separados los resultados de test
    for nombre, confianza in (("estandar", 0.001), ("umbral_fijo", umbral)):  # evalúa los mismos pesos dos veces
        metricas = modelo.val(data=str(data_yaml), split="val", conf=confianza, imgsz=config.imgsz,
                               batch=config.batch_evaluacion, device=config.dispositivo, workers=config.workers,
                               project=str(destino), name=nombre, exist_ok=True, plots=True, verbose=False)  # evaluación oficial
        graficar_curvas_validacion(metricas, destino / nombre, config, confianza)  # curva por clase
        medicion = {"confianza_minima": confianza, "mAP50": float(metricas.box.map50),
                    "mAP50-95": float(metricas.box.map)}  # AP calculada con las detecciones disponibles
        if nombre == "umbral_fijo":  # umbral aplicado en la matriz oficial
            matriz = metricas.confusion_matrix.matrix  # orientación: predicha × real
            precision, recall = precision_y_recall_totales(matriz.T)  # agrega aciertos, predicciones y objetos reales
            medicion.update(precision_micro=float(precision), recall_micro=float(recall),
                            f1_micro=float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
                            iou_emparejamiento=0.45)  # métricas al corte fijo de confianza
        resumen[nombre] = medicion  # separa AP estándar y métricas del umbral
        liberar_memoria_gpu()  # libera temporales antes de la siguiente validación
    (destino / "metricas.json").write_text(json.dumps(resumen, indent=2) + "\n", encoding="utf-8")  # guarda la comparación
    return resumen  # devuelve las dos mediciones


def guardar_matriz_confusion(matriz: np.ndarray, config: Config, carpeta: Path) -> Path:
    """Entrada: matriz, Config y carpeta. Salida: ruta de la imagen de la matriz (también la guarda en CSV)."""
    etiquetas = [*config.clases, "fondo"]  # nombres de filas y columnas
    filas = ["real\\predicha," + ",".join(etiquetas)]  # encabezado del CSV
    filas += [f"{etiquetas[i]}," + ",".join(map(str, fila)) for i, fila in enumerate(matriz)]  # una fila por clase real
    (carpeta / "matriz_confusion.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")  # guarda el CSV
    figura, eje = plt.subplots(figsize=(8, 7))  # crea la figura
    eje.imshow(np.log1p(matriz), cmap="Blues")  # colores en escala log para que se vean también las celdas chicas
    for (i, j), valor in np.ndenumerate(matriz):  # escribe el número en cada celda
        eje.text(j, i, str(valor), ha="center", va="center", fontsize=8)  # número centrado
    eje.set_xticks(range(len(etiquetas)), etiquetas, rotation=45, ha="right")  # columnas: clase predicha
    eje.set_yticks(range(len(etiquetas)), etiquetas)  # filas: clase real
    eje.set_xlabel("predicha")  # título del eje X
    eje.set_ylabel("real")  # título del eje Y
    eje.set_title(f"Matriz de confusión en test (umbral {config.umbral_confianza})")  # título
    figura.tight_layout()  # ajusta márgenes
    ruta = carpeta / "matriz_confusion.png"  # archivo de salida
    figura.savefig(ruta, dpi=120)  # guarda la imagen
    plt.close(figura)  # libera memoria
    return ruta  # devuelve la ruta


def guardar_ejemplos_de_errores(resultados: list[dict], config: Config, carpeta: Path) -> dict[str, int]:
    """Entrada: resultados de test, Config y carpeta. Salida: cantidad de imágenes con cada tipo de error.

    Guarda imágenes con falsos positivos, falsos negativos y clases confundidas (ej.: rojo predicho como verde).
    Real = caja blanca "REAL"; predicción = caja del color de su clase "PRED".
    """
    conteo = {"falso_positivo": 0, "falso_negativo": 0, "clase_confundida": 0}  # imágenes con cada tipo de error
    for resultado in resultados:  # cada imagen de test
        pares = aplicar_umbral(resultado["pares"], config.umbral_confianza)  # pares con el umbral elegido
        tipos = {"falso_positivo" if r == -1 else "falso_negativo" if p == -1 else "clase_confundida" for r, p, *_ in pares if r != p}  # errores presentes
        for tipo in tipos:  # para cada tipo de error de la imagen...
            conteo[tipo] += 1  # ...lo cuenta
            if conteo[tipo] > config.ejemplos_errores:  # ya hay suficientes ejemplos de este tipo
                continue  # no guarda más
            imagen = leer_imagen(resultado["imagen"])  # lee la imagen
            alto, ancho = imagen.shape[:2]  # tamaño para pasar de coordenadas 0-1 a píxeles
            for real, pred, confianza, caja_r, caja_p in pares:  # dibuja todos los pares de la imagen
                if caja_r is not None:  # hay un objeto real...
                    dibujar_caja(imagen, (caja_r[0] * ancho, caja_r[1] * alto, caja_r[2] * ancho, caja_r[3] * alto), f"REAL {config.clases[real]}", (255, 255, 255))  # ...en blanco
                if caja_p is not None:  # hay una predicción...
                    nombre = config.clases[pred]  # nombre de la clase predicha
                    color = tuple(config.colores_bgr.get(nombre, [0, 180, 255]))  # color de respaldo para clases nuevas
                    dibujar_caja(imagen, (caja_p[0] * ancho, caja_p[1] * alto, caja_p[2] * ancho, caja_p[3] * alto), f"PRED {nombre} {confianza:.2f}", color)  # dibuja la predicción
            destino = carpeta / "ejemplos_errores" / tipo  # carpeta del tipo de error
            destino.mkdir(parents=True, exist_ok=True)  # la crea si no existe
            cv2_ok = cv2.imwrite(str(destino / f"{resultado['imagen'].stem}.jpg"), imagen)  # guarda la imagen
            if not cv2_ok:  # si OpenCV no pudo escribir...
                logging.warning("No se pudo guardar el ejemplo %s", resultado["imagen"].name)  # ...avisa
    logging.info("Imágenes de test con errores: %s", conteo)  # resumen
    return conteo  # devuelve los conteos


def evaluar_en_test(modelo: YOLO, data_yaml: Path, carpeta: Path, config: Config) -> dict:
    """Entrada: modelo, data.yaml, carpeta del experimento y Config. Salida: métricas finales.

    Es la ÚNICA vez que se usa test: mAP, precision/recall por clase, matriz de confusión y ejemplos de errores.
    """
    carpeta_eval = carpeta / "evaluacion_test"  # carpeta de la evaluación
    carpeta_eval.mkdir(exist_ok=True)  # la crea
    metricas = modelo.val(  # evaluación oficial de Ultralytics
        data=str(data_yaml), split="test", imgsz=config.imgsz, batch=config.batch_evaluacion, device=config.dispositivo,  # lote fijo seguro
        project=str(carpeta), name="evaluacion_test", exist_ok=True, plots=True, verbose=False,  # guarda sus gráficos acá
    )
    filas = ["clase,precision,recall,mAP50,mAP50-95"]  # encabezado de la tabla por clase
    por_clase = {}  # métricas por clase
    for indice, id_clase in enumerate(metricas.box.ap_class_index):  # solo las clases que aparecen en test
        precision, recall, map50, map50_95 = metricas.box.class_result(indice)  # métricas de esa clase
        nombre = config.clases[int(id_clase)]  # nombre de la clase
        por_clase[nombre] = {"precision": precision, "recall": recall, "mAP50": map50, "mAP50-95": map50_95}  # las guarda
        filas.append(f"{nombre},{precision:.4f},{recall:.4f},{map50:.4f},{map50_95:.4f}")  # fila de la tabla
        logging.info("  %-18s P=%.3f R=%.3f mAP50=%.3f mAP50-95=%.3f", nombre, precision, recall, map50, map50_95)  # log
    (carpeta_eval / "metricas_por_clase.csv").write_text("\n".join(filas) + "\n", encoding="utf-8")  # guarda la tabla
    resultados = predecir_y_emparejar(modelo, listar_imagenes(data_yaml.parent / "images" / "test"), config)  # pares en test
    umbral_operativo = config.umbral_confianza_validacion
    matriz_operativa = construir_matriz_confusion(resultados, umbral_operativo, len(config.clases))
    metricas_operativas = metricas_por_clase_a_umbral(matriz_operativa, config.clases)
    columnas = ("soporte", "tp", "fp", "fn", "precision", "recall")
    filas_operativas = [",".join(("clase", "confianza", *columnas))]
    for clase, valores in metricas_operativas.items():
        celdas = ["" if valores[columna] is None else str(valores[columna]) for columna in columnas]
        filas_operativas.append(",".join((clase, str(umbral_operativo), *celdas)))
    (carpeta_eval / "metricas_por_clase_umbral_operativo.csv").write_text(
        "\n".join(filas_operativas) + "\n", encoding="utf-8"
    )
    matriz = construir_matriz_confusion(resultados, config.umbral_confianza, len(config.clases))  # matriz con el umbral elegido
    guardar_matriz_confusion(matriz, config, carpeta_eval)  # la guarda (CSV + imagen)
    errores = guardar_ejemplos_de_errores(resultados, config, carpeta_eval)  # ejemplos visuales de errores
    resumen = {  # métricas finales del experimento
        "mAP50": float(metricas.box.map50),  # mAP con IoU 0.5
        "mAP50-95": float(metricas.box.map),  # mAP promedio entre IoU 0.5 y 0.95
        "precision_media": float(metricas.box.mp),  # precision promedio entre clases
        "recall_medio": float(metricas.box.mr),  # recall promedio entre clases
        "por_clase": por_clase,  # tabla por clase
        "umbral_operativo": {"confianza": umbral_operativo, "iou": config.umbral_iou, "por_clase": metricas_operativas},
        "imagenes_con_errores": errores,  # conteo de errores por tipo
    }
    if {"semaforo_rojo", "semaforo_verde"}.issubset(config.clases):  # solo aplica a clases que distinguen ambos estados
        rojo, verde = config.clases.index("semaforo_rojo"), config.clases.index("semaforo_verde")  # identifica ambos estados
        resumen["rojo_predicho_como_verde"] = int(matriz[rojo, verde])  # registra el error crítico
        resumen["verde_predicho_como_rojo"] = int(matriz[verde, rojo])  # registra el error inverso
        logging.info("Error más grave: rojo predicho como verde = %d | verde como rojo = %d", matriz[rojo, verde], matriz[verde, rojo])  # lo destaca
    (carpeta_eval / "metricas.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")  # guarda todo
    logging.info("Test: mAP50=%.3f | mAP50-95=%.3f", resumen["mAP50"], resumen["mAP50-95"])  # resumen en el log
    return resumen  # devuelve las métricas


def exportar_modelo(pesos: Path, carpeta: Path, config: Config) -> Path:
    """Entrada: best.pt, carpeta del experimento y Config. Salida: ruta del modelo final (best.pt; y best.onnx si se pide)."""
    destino = carpeta / "modelo"  # carpeta del modelo final de esta versión
    destino.mkdir(exist_ok=True)  # la crea
    final = destino / "best.pt"  # ruta final de los pesos
    shutil.copy2(pesos, final)  # copia best.pt (formato PyTorch, el que usa la demo)
    if config.exportar_onnx:  # si se pidió ONNX (formato estándar, sirve fuera de PyTorch)...
        ruta_onnx = YOLO(str(final)).export(format="onnx", imgsz=config.imgsz, simplify=False)  # exporta con ONNX sin dependencias de simplificación
        logging.info("Exportado a ONNX: %s", ruta_onnx)  # informa la ruta
    logging.info("Modelo final: %s", final)  # informa la ruta del modelo
    return final  # devuelve la ruta para la demo


def buscar_ultimo_modelo(config: Config) -> Path:
    """Entrada: Config. Salida: best.pt de la versión más nueva (para correr solo la demo)."""
    modelos = list(config.carpeta_experimentos.glob("v*_*/modelo/best.pt"))  # todos los modelos guardados
    if not modelos:  # si no hay ninguno...
        raise FileNotFoundError("No hay modelos entrenados: corré primero  python main.py --fase entrenar")  # ...indica qué hacer
    return max(modelos, key=lambda ruta: int(re.match(r"v(\d+)_", ruta.parent.parent.name).group(1)))  # el de versión más alta
