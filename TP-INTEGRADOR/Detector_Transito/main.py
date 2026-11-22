"""Punto de entrada: orquesta el pipeline completo en orden. No tiene lógica propia, solo llama a las fases.

Uso:
    python main.py                      # todo: datos -> entrenamiento -> evaluación -> exportación -> demo
    python main.py --fase datos         # solo prepara el dataset y el EDA
    python main.py --fase entrenar      # entrena, evalúa y exporta (usa el dataset ya preparado)
    python main.py --fase entrenar --data ../datos/dataset_unificado_argentina/data.yaml  # dataset argentino
    python main.py --fase validar --data ../datos/dataset_unificado_argentina/data.yaml --pesos ruta/best.pt
    python main.py --fase folds --data ../datos/dataset_unificado_argentina/data.yaml --folds 3
    python main.py --fase demo          # procesa todos los videos y los muestra uno por uno
    python main.py --fase demo --sin-ventana  # procesa todos los videos sin abrir la vista en vivo
"""
import argparse  # argparse: lee los argumentos de la línea de comandos
from dataclasses import replace  # ajusta las clases para un dataset ya preparado
from pathlib import Path  # Path: rutas portables

import yaml  # lee los nombres de clase del data.yaml
from ultralytics import YOLO  # YOLO: carga los pesos entrenados para evaluar

from configuracion import Config, cargar_configuracion, configurar_logging, fijar_semillas, verificar_gpu  # entorno
from datos import (  # fase de datos
    agrupar_imagenes_duplicadas,  # fusiona grupos con imágenes duplicadas
    analizar_dataset,  # EDA
    dividir_train_val_test_sin_fuga,  # split por grupo
    escribir_dataset_yolo,  # escribe el dataset final
    incorporar_imagenes_propias,  # suma fotos propias nuevas
    obtener_datasets_publicos,  # dataset público base
    sobremuestrear_clases_minoritarias,  # trata el desbalance (solo train)
    ubicar_dataset_final,  # encuentra un dataset ya preparado
    unificar_clases_y_formato_yolo,  # mismas clases para todas las fuentes
    verificar_sin_fuga,  # controla que no haya fuga de datos
)
from entrenamiento import (  # fase de modelo
    analizar_umbral_confianza,  # precision vs. recall en val
    buscar_ultimo_modelo,  # último modelo entrenado
    crear_carpeta_experimento,  # carpeta vN_fecha
    entrenar_detector,  # entrenamiento con transfer learning
    evaluar_en_test,  # evaluación final (única vez en test)
    exportar_modelo,  # best.pt (+ ONNX)
    graficar_curvas_de_perdida,  # curvas train vs. val
    guardar_configuracion_del_experimento,  # hiperparámetros y huella de datos
    liberar_memoria_gpu,  # limpia VRAM entre análisis, test y exportación
    validar_detector,  # validación estándar y con un umbral fijo
)
from tiempo_real import detectar_en_tiempo_real  # demo con OpenCV
from validacion_cruzada import crear_corrida_folds, entrenar_folds  # estratos y validación cruzada


def leer_argumentos() -> argparse.Namespace:
    """Entrada: nada (lee sys.argv). Salida: configuración, fase, pesos y modo de visualización."""
    parser = argparse.ArgumentParser(description="Detector de señales de tránsito y semáforos con YOLO")  # descripción del programa
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml", help="ruta a config.yaml")  # config
    parser.add_argument("--fase", choices=["todo", "datos", "entrenar", "validar", "folds", "demo"], default="todo", help="qué parte correr")  # fase
    parser.add_argument("--data", type=Path, help="data.yaml de un dataset ya preparado (entrenar, validar o preparar folds)")  # dataset externo
    parser.add_argument("--pesos", type=Path, default=None, help="pesos para validar o para la demo (por defecto, el último modelo)")  # pesos
    parser.add_argument("--sin-ventana", action="store_true", help="procesa la demo sin mostrar los videos en vivo")  # salida por lote sin interfaz gráfica
    parser.add_argument("--folds", type=int, help="cantidad de folds (3 por defecto con --fase folds)")  # cantidad de particiones
    parser.add_argument("--entrenar-folds", action="store_true", help="entrena y evalúa cada fold después de prepararlo")  # validación cruzada completa
    argumentos = parser.parse_args()  # lee los argumentos
    if argumentos.data is not None and argumentos.fase not in ("entrenar", "validar", "folds"):  # usa el descriptor indicado
        parser.error("--data requiere --fase entrenar, validar o folds")  # indica el uso correcto
    if argumentos.fase == "folds":  # los grupos se leen del índice que acompaña al dataset
        if not argumentos.data:  # requiere una versión concreta
            parser.error("--fase folds requiere --data con el data.yaml del dataset")  # informa el argumento que falta
        if argumentos.folds is not None and argumentos.folds < 2:  # un fold necesita al menos otro para entrenar
            parser.error("--folds debe ser al menos 2")  # igual que en el script independiente
    elif argumentos.folds is not None or argumentos.entrenar_folds:  # evita ignorar estos argumentos en otras fases
        parser.error("--folds y --entrenar-folds requieren --fase folds")  # informa cómo usarlos
    return argumentos  # devuelve los argumentos leídos


def cargar_clases_dataset(data_yaml: Path) -> list[str]:
    """Lee los nombres en el mismo orden que los IDs de las etiquetas YOLO."""
    datos = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))  # obtiene el descriptor del dataset
    nombres = datos["names"]  # Ultralytics admite una lista o un mapa por ID
    if isinstance(nombres, dict):  # normaliza las claves del YAML a enteros
        por_id = {int(indice): nombre for indice, nombre in nombres.items()}  # permite claves numéricas entre comillas
        if set(por_id) != set(range(len(nombres))):  # exige IDs consecutivos desde cero
            raise ValueError(f"IDs de clase inválidos en {data_yaml}")  # reporta el descriptor con el problema
        nombres = [por_id[indice] for indice in range(len(por_id))]  # alinea los nombres con los IDs
    if not isinstance(nombres, list) or not nombres or datos.get("nc", len(nombres)) != len(nombres):  # valida el catálogo
        raise ValueError(f"Nombres o cantidad de clases inválidos en {data_yaml}")  # evita evaluar con un catálogo incorrecto
    return nombres  # devuelve las clases ordenadas por ID


def preparar_datos(config: Config) -> Path:
    """Entrada: Config. Salida: data.yaml del dataset final. Fase 1: datos (el split va antes que cualquier análisis)."""
    carpetas_publicas = obtener_datasets_publicos(config)  # 1. todos los ZIPs públicos en formato YOLO
    carpeta_propia = incorporar_imagenes_propias(config)  # 2. fotos propias nuevas, si hay
    dataset = unificar_clases_y_formato_yolo(carpetas_publicas, carpeta_propia, config)  # 3. mismas clases, formato YOLO
    dataset, huellas = agrupar_imagenes_duplicadas(dataset)  # 4. duplicados -> mismo grupo
    splits = dividir_train_val_test_sin_fuga(dataset, config)  # 5. split por grupo (70/15/15)
    verificar_sin_fuga(splits, huellas)  # 6. control: nada compartido entre splits
    analizar_dataset(splits, config)  # 7. EDA sobre los splits reales
    splits = {**splits, "train": sobremuestrear_clases_minoritarias(splits["train"], config)}  # 8. desbalance: solo train
    return escribir_dataset_yolo(splits, config)  # 9. escribe el dataset final y su data.yaml


def entrenar_evaluar_y_exportar(data_yaml: Path, config: Config) -> Path:
    """Entrada: data.yaml y Config. Salida: ruta del modelo final. Fase 2: modelo (versionado en experimentos/vN)."""
    carpeta = crear_carpeta_experimento(config)  # 1. carpeta vN_fecha
    guardar_configuracion_del_experimento(config, data_yaml, carpeta)  # 2. guarda hiperparámetros y huella de datos
    pesos = entrenar_detector(data_yaml, carpeta, config)  # 3. entrena con early stopping
    graficar_curvas_de_perdida(pesos.parent.parent, carpeta)  # 4. curvas train vs. val y diagnóstico
    modelo = YOLO(str(pesos))  # carga la mejor época
    analizar_umbral_confianza(modelo, data_yaml, carpeta, config)  # 5. trade-off precision/recall en VAL
    liberar_memoria_gpu()  # libera temporales del análisis de val antes de evaluar test
    validar_detector(modelo, data_yaml, carpeta, config)  # val estándar y val con el umbral configurado
    evaluar_en_test(modelo, data_yaml, carpeta, config)  # 6. evaluación final, una sola vez en TEST
    liberar_memoria_gpu()  # libera temporales de test antes de exportar ONNX
    return exportar_modelo(pesos, carpeta, config)  # 7. best.pt (+ ONNX)


def main() -> None:
    """Entrada: argumentos de línea de comandos. Salida: nada. Corre las fases pedidas en orden."""
    argumentos = leer_argumentos()  # lee --config, --fase y --pesos
    configurar_logging()  # formato de los mensajes
    config = cargar_configuracion(argumentos.config)  # carga la configuración UNA sola vez
    if argumentos.fase == "folds":  # la preparación solo escribe listas y descriptores
        data_yaml = argumentos.data.resolve()  # ubica el descriptor y su indice.json
        if not data_yaml.is_file():  # evita resolver otra versión por accidente
            raise FileNotFoundError(f"No existe el descriptor del dataset: {data_yaml}")  # muestra la ruta pedida
        destino = config.carpeta_experimentos / f"validacion_cruzada_{data_yaml.parent.name}"  # separa corridas por versión
        descriptores, nombres = crear_corrida_folds(data_yaml.parent, destino, argumentos.folds or 3, config.semilla)  # prepara los folds
        if argumentos.entrenar_folds:  # el entrenamiento también valida y resume resultados por fold
            entrenar_folds(descriptores, argumentos.config.resolve(), config.semilla, nombres)  # reutiliza el entrenador
        return  # termina sin pasar por el entrenamiento general
    fijar_semillas(config.semilla)  # reproducibilidad
    verificar_gpu(config.dispositivo)  # GPU o CPU según config.yaml; si se pidió GPU y no hay, se frena acá
    if argumentos.fase in ("todo", "datos"):  # fase de datos
        data_yaml = preparar_datos(config)  # prepara el dataset
    elif argumentos.fase == "entrenar":  # solo entrenamiento...
        data_yaml = argumentos.data.resolve() if argumentos.data else ubicar_dataset_final(config)  # usa el dataset preparado
        if argumentos.data:  # usa las clases del dataset indicado en evaluación y exportación
            config = replace(config, clases=cargar_clases_dataset(data_yaml))  # ajusta el orden de clases a los IDs del dataset
    if argumentos.fase in ("todo", "entrenar"):  # fase de modelo
        pesos = entrenar_evaluar_y_exportar(data_yaml, config)  # entrena, evalúa y exporta
    elif argumentos.fase == "validar":  # vuelve a evaluar un checkpoint existente
        pesos = (argumentos.pesos or buscar_ultimo_modelo(config)).resolve()  # fija los pesos de la corrida
        carpeta = pesos.parent  # carpeta del checkpoint
        if carpeta.name == "modelo":  # pesos exportados por el proyecto
            carpeta = carpeta.parent  # coloca el informe en la carpeta del experimento
        elif carpeta.name == "weights" and carpeta.parent.name == "entrenamiento":  # pesos de la fase train
            carpeta = carpeta.parent.parent  # usa el mismo destino que con el modelo exportado
        data_yaml = argumentos.data or carpeta / "data_usado.yaml"  # descriptor explícito o guardado en el experimento
        config = replace(config, clases=cargar_clases_dataset(data_yaml))  # alinea las clases con el dataset
        csv_resultados = carpeta / "entrenamiento" / "results.csv"  # usa el historial existente si está disponible
        if csv_resultados.is_file():  # un checkpoint externo puede venir sin historial de entrenamiento
            graficar_curvas_de_perdida(csv_resultados.parent, carpeta)  # actualiza el gráfico train vs. val
        validar_detector(YOLO(str(pesos)), data_yaml.resolve(), carpeta, config)  # produce ambos informes en val
    elif argumentos.fase == "demo":  # solo demo...
        pesos = argumentos.pesos or buscar_ultimo_modelo(config)  # ...con los pesos indicados o el último modelo
    if argumentos.fase in ("todo", "demo"):  # fase de demo
        detectar_en_tiempo_real(pesos, config, mostrar_en_vivo=not argumentos.sin_ventana)  # procesa todos los videos, con ventana salvo que se desactive


if __name__ == "__main__":
    main()
