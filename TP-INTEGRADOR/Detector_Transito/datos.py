"""Fase de datos: obtener datasets, unificar clases, dividir sin fuga, EDA, sobremuestreo y escritura en formato YOLO."""
import hashlib  # hashlib: calcula la huella SHA-256 de la lista de archivos (versionado de datos)
import json  # json: guarda el manifiesto del dataset
import logging  # logging: mensajes del pipeline
import random  # random: mezcla de grupos con semilla fija
import re  # re: expresiones regulares para sacar el grupo (video/clip) del nombre del archivo
import shutil  # shutil: copiar archivos y borrar carpetas
import zipfile  # zipfile: descomprimir el dataset base
from dataclasses import dataclass, replace  # replace: copia una Muestra cambiando solo algunos campos
from pathlib import Path  # Path: rutas portables

import cv2  # OpenCV: leer imágenes, calcular huellas y dibujar ejemplos
import matplotlib  # matplotlib: gráficos del EDA

matplotlib.use("Agg")  # "Agg" dibuja a archivo sin abrir ventanas (funciona también sin pantalla)
import matplotlib.pyplot as plt  # noqa: E402  pyplot: la interfaz de gráficos de matplotlib
import numpy as np  # noqa: E402  numpy: lectura de bytes de imagen y cálculos
import yaml  # noqa: E402  yaml: leer y escribir data.yaml

from configuracion import Config  # noqa: E402  la configuración única del proyecto
from tiempo_real import dibujar_caja  # noqa: E402  misma función de dibujo que usa la demo (mismos colores)


@dataclass(frozen=True)  # frozen=True: una Muestra no se modifica, se crea una nueva con replace()
class Muestra:
    """Una imagen del dataset con sus etiquetas ya convertidas a las clases finales."""

    imagen: Path  # ruta a la imagen original
    nombre: str  # nombre único (sin extensión); su prefijo indica la fuente: bdd_, lisa_, propio_, nuevo_
    lineas: tuple[str, ...]  # líneas YOLO "clase x_centro y_centro ancho alto" con los ids finales
    grupo: str  # video o secuencia de origen: todas las imágenes de un grupo van al mismo split
    copia: int = 0  # 0 = original; 1, 2, ... = copia agregada por sobremuestreo (solo en train)


def es_imagen(ruta: Path) -> bool:
    """Entrada: ruta. Salida: True si es una imagen soportada. Filtro auxiliar al recorrer carpetas."""
    return ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}  # compara la extensión en minúsculas


def leer_imagen(ruta: Path, modo: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Entrada: ruta y modo de OpenCV. Salida: imagen. Lee también rutas con tildes en Windows."""
    imagen = cv2.imdecode(np.fromfile(str(ruta), dtype=np.uint8), modo)  # lee los bytes y los decodifica
    if imagen is None:  # imdecode devuelve None si el archivo está roto o no es una imagen
        raise ValueError(f"No se pudo leer la imagen: {ruta}")  # error claro con la ruta problemática
    return imagen  # devuelve la matriz de píxeles


def _extraer_zip_seguro(origen: Path, destino: Path) -> None:
    """Extrae un ZIP local sin permitir que una entrada escape de su carpeta destino."""
    destino.mkdir(parents=True, exist_ok=True)
    raiz = destino.resolve()
    with zipfile.ZipFile(origen) as archivo_zip:
        for entrada in archivo_zip.infolist():
            salida = (destino / entrada.filename).resolve()
            if salida != raiz and raiz not in salida.parents:
                raise ValueError(f"ZIP inseguro: {entrada.filename}")
        archivo_zip.extractall(destino)


def obtener_datasets_publicos(config: Config) -> list[Path]:
    """Extrae y devuelve todos los datasets YOLO encontrados en la carpeta de ZIPs."""
    zips = sorted(config.carpeta_zips.glob("*.zip")) if config.carpeta_zips.is_dir() else []
    if not zips and config.zip_dataset_base.is_file():
        zips = [config.zip_dataset_base]
    if not zips:
        raise FileNotFoundError(f"No se encontraron ZIPs YOLO en {config.carpeta_zips}")
    carpetas = []
    for zip_path in zips:
        destino = config.carpeta_dataset_base / zip_path.stem
        encontrados = list(destino.rglob("data.yaml")) if destino.is_dir() else []
        if not encontrados:
            logging.info("Descomprimiendo %s (puede tardar)...", zip_path.name)
            _extraer_zip_seguro(zip_path, destino)
            encontrados = list(destino.rglob("data.yaml"))
        if not encontrados:
            logging.warning("Se ignora %s: no contiene data.yaml", zip_path.name)
            continue
        carpetas.append(encontrados[0].parent)
        logging.info("Dataset público cargable: %s", encontrados[0].parent)
    if not carpetas:
        raise FileNotFoundError(f"Ningún ZIP de {config.carpeta_zips} contiene data.yaml")
    return carpetas


def incorporar_imagenes_propias(config: Config) -> Path | None:
    """Entrada: Config. Salida: carpeta con fotos propias nuevas etiquetadas, o None si no hay.

    Las fotos se etiquetan con Label Studio, CVAT o Roboflow y se exportan en formato YOLO
    (imágenes + .txt + classes.txt) dentro de config.carpeta_datos_propios.
    """
    carpeta = config.carpeta_datos_propios  # carpeta configurada para los datos propios
    if not carpeta.is_dir() or not any(es_imagen(r) for r in carpeta.rglob("*")):  # si no existe o no tiene imágenes...
        logging.warning("No hay imágenes propias nuevas en %s: se sigue solo con el dataset base", carpeta)  # ...avisa
        return None  # None = no hay datos propios para sumar
    logging.info("Datos propios encontrados en %s", carpeta)  # informa que se van a sumar
    return carpeta  # devuelve la carpeta para unificarla con el resto


def leer_nombres_de_clases(carpeta: Path) -> list[str]:
    """Entrada: carpeta de un dataset YOLO. Salida: nombres de sus clases en orden de id."""
    data_yaml = carpeta / "data.yaml"  # formato Ultralytics / Roboflow
    if data_yaml.is_file():  # si existe data.yaml...
        nombres = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]  # ...lee el campo "names"
        return [nombres[i] for i in sorted(nombres)] if isinstance(nombres, dict) else list(nombres)  # dict o lista
    for nombre_archivo in ("classes.txt", "obj.names"):  # classes.txt = Label Studio; obj.names = CVAT
        archivo = next(carpeta.rglob(nombre_archivo), None)  # busca el archivo en cualquier subcarpeta
        if archivo is not None:  # si lo encontró...
            return [linea.strip() for linea in archivo.read_text(encoding="utf-8").splitlines() if linea.strip()]  # una clase por línea
    raise FileNotFoundError(f"No hay data.yaml, classes.txt ni obj.names en {carpeta}")  # sin nombres no se puede mapear


def buscar_etiqueta(imagen: Path) -> Path | None:
    """Entrada: ruta de una imagen. Salida: su .txt de etiquetas, o None si no tiene."""
    candidatas = [imagen.with_suffix(".txt")]  # caso CVAT: el .txt está al lado de la imagen
    partes = list(imagen.parts)  # separa la ruta en carpetas
    if "images" in partes:  # caso Ultralytics / Label Studio: .../images/... -> .../labels/...
        indice = len(partes) - 1 - partes[::-1].index("images")  # posición de la última carpeta "images"
        partes[indice] = "labels"  # la reemplaza por "labels"
        candidatas.append(Path(*partes).with_suffix(".txt"))  # agrega esa ruta como candidata
    return next((c for c in candidatas if c.is_file()), None)  # devuelve la primera que exista


def remapear_lineas(lineas: list[str], nombres_origen: list[str], config: Config) -> tuple[str, ...]:
    """Entrada: líneas YOLO de origen y sus nombres de clase. Salida: líneas con los ids de las clases finales."""
    nuevas = []  # acá se juntan las líneas convertidas
    for linea in lineas:  # recorre cada objeto etiquetado
        valores = linea.split()  # separa "clase x y ancho alto"
        if len(valores) != 5:  # descarta líneas vacías o que no son cajas (por ejemplo polígonos)
            continue  # pasa a la siguiente línea
        nombre_final = config.mapeo_clases.get(nombres_origen[int(valores[0])])  # nombre de origen -> nombre final
        if nombre_final is None:  # la clase no está en el mapeo (por ejemplo "auto")...
            continue  # ...se descarta: el detector solo aprende semáforos y señales
        nuevas.append(" ".join([str(config.clases.index(nombre_final)), *valores[1:]]))  # mismo cuadro, id nuevo
    return tuple(nuevas)  # tupla: no se puede modificar después


def calcular_grupo(nombre: str, patrones: list[str]) -> str:
    """Entrada: nombre de archivo y patrones. Salida: grupo (video/clip). Base de la partición sin fuga."""
    for patron in patrones:  # prueba cada expresión regular de config.yaml
        coincidencia = re.match(patron, nombre)  # intenta reconocer el nombre
        if coincidencia:  # si coincide...
            return coincidencia.group(1)  # ...el grupo es lo capturado (ej.: "lisa_dayClip3")
    return nombre  # foto suelta: es su propio grupo


def cargar_muestras(carpeta: Path, prefijo: str, config: Config) -> list[Muestra]:
    """Entrada: carpeta YOLO, prefijo para los nombres y Config. Salida: lista de Muestras en clases finales."""
    nombres_origen = leer_nombres_de_clases(carpeta)  # clases tal como vienen en esa fuente
    if not set(nombres_origen).intersection(config.mapeo_clases):
        logging.warning("Se ignora %s: ninguna clase está mapeada en config.yaml (%s)", carpeta, nombres_origen)
        return []
    raiz = carpeta / "images" if (carpeta / "images").is_dir() else carpeta  # si hay images/, se busca solo ahí
    muestras, sin_etiqueta = [], 0  # lista de resultados y contador de imágenes sin .txt
    for imagen in sorted(r for r in raiz.rglob("*") if es_imagen(r)):  # recorre las imágenes en orden fijo
        etiqueta = buscar_etiqueta(imagen)  # busca su archivo de etiquetas
        if etiqueta is None:  # imagen sin etiquetar...
            sin_etiqueta += 1  # ...se cuenta...
            continue  # ...y se saltea (no se sabe qué tiene adentro)
        lineas = remapear_lineas(etiqueta.read_text(encoding="utf-8").splitlines(), nombres_origen, config)  # convierte clases
        nombre = prefijo + imagen.stem  # nombre único con el prefijo de la fuente
        muestras.append(Muestra(imagen, nombre, lineas, calcular_grupo(nombre, config.patrones_grupo)))  # guarda la muestra
    if sin_etiqueta:  # si hubo imágenes sin etiquetas...
        logging.warning("%d imágenes de %s no tienen etiquetas y se ignoran", sin_etiqueta, carpeta)  # ...avisa
    logging.info("%s: %d imágenes cargadas", carpeta.name, len(muestras))  # informa cuántas entraron
    return muestras  # devuelve las muestras de esta fuente


def unificar_clases_y_formato_yolo(carpetas_publicas: list[Path], carpeta_propia: Path | None, config: Config) -> list[Muestra]:
    """Entrada: carpetas del dataset base y de los datos propios. Salida: todas las Muestras con las mismas clases."""
    muestras = []
    for indice, carpeta_publica in enumerate(carpetas_publicas):
        muestras += cargar_muestras(carpeta_publica, f"fuente{indice}_", config)
    if carpeta_propia is not None:  # si hay fotos propias nuevas...
        muestras += cargar_muestras(carpeta_propia, "nuevo_", config)  # ...se suman con el prefijo "nuevo_"
    nombres = [m.nombre for m in muestras]  # lista de nombres para controlar repetidos
    if len(nombres) != len(set(nombres)):  # si dos imágenes tienen el mismo nombre se pisarían al escribir...
        raise ValueError("Hay imágenes con el mismo nombre: renombralas antes de seguir")  # ...se frena
    logging.info("Dataset unificado: %d imágenes | clases: %s", len(muestras), config.clases)  # resumen
    return muestras  # devuelve el dataset unificado (todavía sin dividir)


def calcular_huella(imagen: Path) -> int:
    """Entrada: ruta de una imagen. Salida: huella dHash de 64 bits. Dos imágenes casi iguales dan la misma huella."""
    gris = leer_imagen(imagen, cv2.IMREAD_REDUCED_GRAYSCALE_4)  # lee en gris y a 1/4 del tamaño (más rápido)
    chica = cv2.resize(gris, (9, 8), interpolation=cv2.INTER_AREA)  # la achica a 9x8 píxeles
    bits = (chica[:, 1:] > chica[:, :-1]).flatten()  # 64 bits: ¿cada píxel es más claro que el de su izquierda?
    return int("".join("1" if bit else "0" for bit in bits), 2)  # convierte los 64 bits en un número entero


def raiz_de_grupo(padres: dict[str, str], grupo: str) -> str:
    """Entrada: tabla de fusiones y un grupo. Salida: el grupo representante (union-find)."""
    while padres.get(grupo, grupo) != grupo:  # mientras el grupo haya sido fusionado con otro...
        grupo = padres[grupo]  # ...sube al grupo con el que se fusionó
    return grupo  # devuelve el representante final


def agrupar_imagenes_duplicadas(muestras: list[Muestra]) -> tuple[list[Muestra], dict[str, int]]:
    """Entrada: Muestras. Salida: Muestras con grupos fusionados y la huella de cada imagen.

    Si dos imágenes casi idénticas están en grupos distintos, se fusionan los grupos: así un
    duplicado nunca puede quedar uno en train y otro en test.
    """
    huellas = {}  # nombre de imagen -> huella
    for numero, muestra in enumerate(muestras, start=1):  # recorre las imágenes numerándolas
        huellas[muestra.nombre] = calcular_huella(muestra.imagen)  # calcula la huella
        if numero % 2000 == 0:  # cada 2000 imágenes...
            logging.info("Huellas calculadas: %d / %d", numero, len(muestras))  # ...muestra el avance
    padres: dict[str, str] = {}  # tabla de fusiones entre grupos
    primer_grupo: dict[int, str] = {}  # huella -> primer grupo en el que apareció
    fusiones = 0  # contador de fusiones hechas
    for muestra in muestras:  # recorre las imágenes
        huella = huellas[muestra.nombre]  # su huella
        if huella not in primer_grupo:  # primera vez que aparece esta huella...
            primer_grupo[huella] = muestra.grupo  # ...se anota su grupo
            continue  # y se sigue
        grupo_a = raiz_de_grupo(padres, primer_grupo[huella])  # representante del grupo anterior
        grupo_b = raiz_de_grupo(padres, muestra.grupo)  # representante del grupo actual
        if grupo_a != grupo_b:  # si son grupos distintos...
            padres[grupo_b] = grupo_a  # ...se fusionan
            fusiones += 1  # y se cuenta
    logging.info("Duplicados entre grupos distintos: %d fusiones de grupo", fusiones)  # informa cuántas hubo
    agrupadas = [replace(m, grupo=raiz_de_grupo(padres, m.grupo)) for m in muestras]  # aplica las fusiones
    return agrupadas, huellas  # devuelve las muestras y las huellas (para verificar después)


def dividir_train_val_test_sin_fuga(muestras: list[Muestra], config: Config) -> dict[str, list[Muestra]]:
    """Entrada: Muestras agrupadas y Config. Salida: {"train": [...], "val": [...], "test": [...]}.

    Partición por GRUPO: todas las imágenes de un mismo video caen en el mismo split.
    Se hace por fuente (bdd, lisa, propio, nuevo) para que cada split tenga de todas.
    """
    generador = random.Random(config.semilla)  # generador propio con semilla: el split siempre sale igual
    objetivos = config.proporciones_split  # por ejemplo {"train": 0.70, "val": 0.15, "test": 0.15}
    splits: dict[str, list[Muestra]] = {nombre: [] for nombre in objetivos}  # splits vacíos
    por_grupo: dict[str, list[Muestra]] = {}  # grupo -> sus imágenes
    for muestra in muestras:  # reparte cada imagen en su grupo
        por_grupo.setdefault(muestra.grupo, []).append(muestra)  # crea la lista del grupo si no existía
    grupos_por_fuente: dict[str, list[str]] = {}  # fuente -> sus grupos
    for grupo, imagenes in sorted(por_grupo.items()):  # recorre los grupos en orden fijo
        grupos_por_fuente.setdefault(imagenes[0].nombre.split("_")[0], []).append(grupo)  # fuente = prefijo del nombre
    for fuente, grupos in grupos_por_fuente.items():  # divide cada fuente por separado
        generador.shuffle(grupos)  # mezcla los grupos al azar (con semilla)
        grupos.sort(key=lambda g: len(por_grupo[g]), reverse=True)  # los más grandes primero (sort estable: mantiene el azar entre iguales)
        asignadas = {nombre: 0 for nombre in objetivos}  # imágenes de esta fuente ya puestas en cada split
        for grupo in grupos:  # reparte cada grupo
            total = sum(asignadas.values()) + len(por_grupo[grupo])  # total de la fuente si se agrega este grupo
            destino = max(objetivos, key=lambda s: objetivos[s] * total - asignadas[s])  # el split más lejos de su cupo
            splits[destino] += por_grupo[grupo]  # el grupo entero va a ese split
            asignadas[destino] += len(por_grupo[grupo])  # actualiza la cuenta
        logging.info("Split de %s: %s", fuente, asignadas)  # muestra cómo quedó esta fuente
    logging.info("Split total: %s", {nombre: len(lista) for nombre, lista in splits.items()})  # resumen general
    return splits  # devuelve los tres splits


def verificar_sin_fuga(splits: dict[str, list[Muestra]], huellas: dict[str, int]) -> None:
    """Entrada: splits y huellas. Salida: nada. Frena si un grupo o una imagen duplicada está en dos splits."""
    nombres = list(splits)  # ["train", "val", "test"]
    for i, split_a in enumerate(nombres):  # compara cada par de splits...
        for split_b in nombres[i + 1:]:  # ...sin repetir pares
            grupos = {m.grupo for m in splits[split_a]} & {m.grupo for m in splits[split_b]}  # grupos en ambos
            repetidas = {huellas[m.nombre] for m in splits[split_a]} & {huellas[m.nombre] for m in splits[split_b]}  # huellas en ambos
            if grupos or repetidas:  # si hay algo en común hay fuga de datos...
                raise RuntimeError(f"Fuga entre {split_a} y {split_b}: {len(grupos)} grupos, {len(repetidas)} duplicados")  # ...se frena
    logging.info("Verificado: ningún video ni imagen duplicada aparece en dos splits")  # todo en orden


def contar_instancias_por_clase(splits: dict[str, list[Muestra]], config: Config) -> dict[str, list[int]]:
    """Entrada: splits y Config. Salida: {split: [cantidad de cajas de cada clase]}."""
    conteos = {nombre: [0] * len(config.clases) for nombre in splits}  # un contador por clase y por split
    for nombre, muestras in splits.items():  # recorre cada split
        for muestra in muestras:  # cada imagen
            for linea in muestra.lineas:  # cada caja
                conteos[nombre][int(linea.split()[0])] += 1  # suma 1 a la clase de esa caja
    return conteos  # devuelve la tabla de conteos


def guardar_tabla_de_clases(conteos: dict[str, list[int]], config: Config, carpeta: Path) -> Path:
    """Entrada: conteos, Config y carpeta. Salida: ruta del CSV con las instancias por clase y por split."""
    ruta = carpeta / "instancias_por_clase.csv"  # archivo de salida
    filas = ["clase," + ",".join(conteos) + ",total,porcentaje"]  # encabezado: clase,train,val,test,total,porcentaje
    total_general = sum(sum(valores) for valores in conteos.values())  # cajas totales de todo el dataset
    for indice, clase in enumerate(config.clases):  # una fila por clase
        por_split = [conteos[nombre][indice] for nombre in conteos]  # cajas de esta clase en cada split
        porcentaje = 100 * sum(por_split) / max(total_general, 1)  # porcentaje sobre el total (max evita dividir por 0)
        filas.append(f"{clase}," + ",".join(map(str, por_split)) + f",{sum(por_split)},{porcentaje:.1f}")  # arma la fila
        logging.info("  %-18s %s  total=%d (%.1f%%)", clase, por_split, sum(por_split), porcentaje)  # la muestra en el log
    ruta.write_text("\n".join(filas) + "\n", encoding="utf-8")  # escribe el CSV
    return ruta  # devuelve la ruta del CSV


def graficar_instancias_por_clase(conteos: dict[str, list[int]], config: Config, carpeta: Path) -> Path:
    """Entrada: conteos, Config y carpeta. Salida: ruta del gráfico de barras (deja ver el desbalance)."""
    ruta = carpeta / "instancias_por_clase.png"  # archivo de salida
    posiciones = np.arange(len(config.clases))  # una posición en el eje X por clase
    ancho = 0.8 / len(conteos)  # ancho de cada barra para que entren las de los tres splits
    figura, eje = plt.subplots(figsize=(10, 5))  # crea la figura
    for numero, (nombre, valores) in enumerate(conteos.items()):  # una serie de barras por split
        eje.bar(posiciones + numero * ancho, valores, ancho, label=nombre)  # dibuja las barras de ese split
    eje.set_xticks(posiciones + ancho * (len(conteos) - 1) / 2)  # centra los nombres debajo de cada grupo de barras
    eje.set_xticklabels(config.clases, rotation=20)  # nombres de las clases
    eje.set_yscale("log")  # escala logarítmica: si no, las clases chicas casi no se ven
    eje.set_ylabel("cajas (escala log)")  # título del eje Y
    eje.set_title("Instancias por clase y por split")  # título del gráfico
    eje.legend()  # leyenda train / val / test
    figura.tight_layout()  # ajusta márgenes para que no se corten los textos
    figura.savefig(ruta, dpi=120)  # guarda el gráfico
    plt.close(figura)  # libera la memoria de la figura
    return ruta  # devuelve la ruta del gráfico


def graficar_tamanos_de_cajas(splits: dict[str, list[Muestra]], config: Config, carpeta: Path) -> Path:
    """Entrada: splits, Config y carpeta. Salida: ruta del histograma del tamaño de las cajas por clase."""
    ruta = carpeta / "tamanos_de_cajas.png"  # archivo de salida
    tamanos: list[list[float]] = [[] for _ in config.clases]  # una lista de tamaños por clase
    for muestras in splits.values():  # recorre todos los splits
        for muestra in muestras:  # cada imagen
            for linea in muestra.lineas:  # cada caja
                clase, _, _, ancho, alto = linea.split()  # separa los valores
                tamanos[int(clase)].append(100 * (float(ancho) * float(alto)) ** 0.5)  # lado equivalente en % de la imagen
    todas = [t for lista in tamanos for t in lista]  # todos los tamaños juntos
    chicas = 100 * sum(t < 2 for t in todas) / max(len(todas), 1)  # porcentaje de cajas de menos del 2% de la imagen
    logging.info("Cajas muy chicas (< 2%% del lado de la imagen): %.1f%%", chicas)  # dato clave: objetos lejanos
    figura, eje = plt.subplots(figsize=(10, 5))  # crea la figura
    eje.hist(tamanos, bins=np.logspace(-1, 2, 40), histtype="step", label=config.clases, linewidth=1.5)  # un histograma por clase
    eje.set_xscale("log")  # escala logarítmica: hay muchísimas cajas chicas
    eje.set_xlabel("tamaño de la caja (% del lado de la imagen, escala log)")  # título del eje X
    eje.set_ylabel("cantidad de cajas")  # título del eje Y
    eje.set_title("Distribución del tamaño de las cajas")  # título del gráfico
    eje.legend()  # leyenda con las clases
    figura.tight_layout()  # ajusta márgenes
    figura.savefig(ruta, dpi=120)  # guarda el gráfico
    plt.close(figura)  # libera la memoria
    return ruta  # devuelve la ruta del gráfico


def guardar_mosaico_de_ejemplos(muestras: list[Muestra], config: Config, carpeta: Path) -> Path:
    """Entrada: Muestras, Config y carpeta. Salida: ruta de un mosaico de ejemplos con las cajas dibujadas con OpenCV."""
    ruta = carpeta / "ejemplos_train.jpg"  # archivo de salida
    elegidas = random.Random(config.semilla).sample(muestras, min(config.ejemplos_eda, len(muestras)))  # ejemplos al azar
    lado, columnas = 400, 4  # cada ejemplo ocupa 400x400 píxeles, 4 por fila
    filas = (len(elegidas) + columnas - 1) // columnas  # cantidad de filas necesarias (división redondeando para arriba)
    mosaico = np.zeros((filas * lado, columnas * lado, 3), dtype=np.uint8)  # lienzo negro
    for numero, muestra in enumerate(elegidas):  # dibuja cada ejemplo
        imagen = leer_imagen(muestra.imagen)  # lee la imagen
        alto, ancho = imagen.shape[:2]  # tamaño en píxeles
        for linea in muestra.lineas:  # dibuja cada caja
            clase, xc, yc, w, h = linea.split()  # separa los valores normalizados
            x1, y1 = (float(xc) - float(w) / 2) * ancho, (float(yc) - float(h) / 2) * alto  # esquina superior izquierda en píxeles
            x2, y2 = x1 + float(w) * ancho, y1 + float(h) * alto  # esquina inferior derecha en píxeles
            nombre = config.clases[int(clase)]  # nombre de la clase
            dibujar_caja(imagen, (x1, y1, x2, y2), nombre, tuple(config.colores_bgr[nombre]))  # dibuja caja y texto
        escala = lado / max(alto, ancho)  # factor para que entre en 400x400 sin deformarse
        chica = cv2.resize(imagen, (int(ancho * escala), int(alto * escala)))  # achica la imagen
        fila, columna = divmod(numero, columnas)  # posición dentro del mosaico
        mosaico[fila * lado:fila * lado + chica.shape[0], columna * lado:columna * lado + chica.shape[1]] = chica  # la pega
    cv2.imwrite(str(ruta), mosaico)  # guarda el mosaico
    return ruta  # devuelve la ruta


def detectar_clases_minoritarias(conteos: dict[str, list[int]], config: Config) -> list[str]:
    """Entrada: conteos y Config. Salida: clases con muy pocas cajas en train (desbalance)."""
    train = conteos["train"]  # el desbalance que importa es el de train: de ahí aprende el modelo
    maximo = max(train)  # cajas de la clase más frecuente
    minoritarias = [c for c, n in zip(config.clases, train) if n < config.umbral_desbalance * maximo]  # clases muy por debajo
    for clase in minoritarias:  # para cada clase minoritaria...
        logging.warning("Clase minoritaria: %s (%d cajas vs. %d de la mayor)", clase, train[config.clases.index(clase)], maximo)  # ...avisa
    return minoritarias  # devuelve la lista


def analizar_dataset(splits: dict[str, list[Muestra]], config: Config) -> Path:
    """Entrada: splits y Config. Salida: carpeta del reporte EDA (tabla, gráficos, ejemplos y desbalance)."""
    carpeta = config.carpeta_reportes / "eda"  # carpeta del EDA
    carpeta.mkdir(parents=True, exist_ok=True)  # la crea si no existe
    logging.info("EDA - instancias por clase %s:", list(splits))  # encabezado de la tabla en el log
    conteos = contar_instancias_por_clase(splits, config)  # cuenta cajas por clase y split
    guardar_tabla_de_clases(conteos, config, carpeta)  # tabla CSV
    graficar_instancias_por_clase(conteos, config, carpeta)  # gráfico de barras
    graficar_tamanos_de_cajas(splits, config, carpeta)  # histograma de tamaños
    guardar_mosaico_de_ejemplos(splits["train"], config, carpeta)  # ejemplos dibujados con OpenCV
    minoritarias = detectar_clases_minoritarias(conteos, config)  # detecta el desbalance
    resumen = {"imagenes": {s: len(m) for s, m in splits.items()}, "cajas": conteos, "minoritarias": minoritarias}  # resumen
    (carpeta / "resumen_eda.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")  # lo guarda
    logging.info("Reporte EDA guardado en %s", carpeta)  # informa dónde quedó
    return carpeta  # devuelve la carpeta del reporte


def sobremuestrear_clases_minoritarias(train: list[Muestra], config: Config) -> list[Muestra]:
    """Entrada: Muestras de TRAIN y Config. Salida: train con copias extra de las imágenes con clases minoritarias.

    Solo se aplica a train: val y test tienen que reflejar la distribución real.
    """
    veces_por_id = {config.clases.index(c): veces for c, veces in config.sobremuestreo.items()}  # id de clase -> repeticiones
    resultado = list(train)  # arranca con todas las originales
    for muestra in train:  # revisa cada imagen
        ids = {int(linea.split()[0]) for linea in muestra.lineas}  # clases presentes en la imagen
        veces = max((veces_por_id[i] for i in ids if i in veces_por_id), default=1)  # repeticiones que le tocan
        resultado += [replace(muestra, copia=n) for n in range(1, veces)]  # agrega veces-1 copias
    logging.info("Sobremuestreo en train: %d -> %d imágenes", len(train), len(resultado))  # informa cuántas se sumaron
    return resultado  # devuelve el train aumentado


def escribir_dataset_yolo(splits: dict[str, list[Muestra]], config: Config) -> Path:
    """Entrada: splits y Config. Salida: ruta del data.yaml del dataset final, listo para Ultralytics.

    Escribe images/ y labels/ por split, el data.yaml y un manifiesto con la huella de los datos (versionado).
    """
    destino = config.carpeta_dataset_final  # carpeta del dataset final
    if destino.exists():  # si ya existe de una corrida anterior...
        if not (destino / "manifiesto.json").is_file():  # ...y no la generó este programa...
            raise RuntimeError(f"{destino} existe y no es un dataset generado por este programa: no se borra")  # ...no se toca
        shutil.rmtree(destino)  # se borra para regenerarla desde cero (es una carpeta generada)
    for nombre, muestras in splits.items():  # escribe cada split
        (destino / "images" / nombre).mkdir(parents=True)  # carpeta de imágenes del split
        (destino / "labels" / nombre).mkdir(parents=True)  # carpeta de etiquetas del split
        for muestra in muestras:  # cada imagen del split
            archivo = muestra.nombre if muestra.copia == 0 else f"{muestra.nombre}__copia{muestra.copia}"  # nombre único
            shutil.copy2(muestra.imagen, destino / "images" / nombre / f"{archivo}{muestra.imagen.suffix.lower()}")  # copia la imagen
            texto = "".join(linea + "\n" for linea in muestra.lineas)  # una caja por línea (vacío = imagen de fondo)
            (destino / "labels" / nombre / f"{archivo}.txt").write_text(texto, encoding="utf-8")  # escribe la etiqueta
        logging.info("Escrito %s: %d imágenes", nombre, len(muestras))  # avance por split
    data_yaml = destino / "data.yaml"  # archivo de configuración del dataset para Ultralytics
    contenido = {"path": str(destino), **{s: f"images/{s}" for s in splits}, "names": dict(enumerate(config.clases))}  # rutas y clases
    data_yaml.write_text(yaml.safe_dump(contenido, allow_unicode=True, sort_keys=False), encoding="utf-8")  # lo escribe
    manifiesto = {  # resumen para saber exactamente con qué datos se entrenó cada modelo
        s: {"imagenes": len(m), "huella_sha256": hashlib.sha256("\n".join(sorted(f"{x.nombre}#{x.copia}" for x in m)).encode()).hexdigest()}
        for s, m in splits.items()  # una entrada por split
    }
    (destino / "manifiesto.json").write_text(json.dumps(manifiesto, indent=2), encoding="utf-8")  # guarda el manifiesto
    logging.info("Dataset final listo: %s", data_yaml)  # informa dónde quedó
    return data_yaml  # devuelve la ruta para el entrenamiento


def ubicar_dataset_final(config: Config) -> Path:
    """Entrada: Config. Salida: data.yaml del dataset final ya generado (para correr solo la fase de entrenamiento)."""
    data_yaml = config.carpeta_dataset_final / "data.yaml"  # donde lo deja escribir_dataset_yolo
    if not data_yaml.is_file():  # si todavía no se generó...
        raise FileNotFoundError("No existe el dataset final: corré primero  python main.py --fase datos")  # ...indica qué hacer
    return data_yaml  # devuelve la ruta
