"""Prepara y, opcionalmente, entrena folds YOLO estratificados por clase y agrupados por video."""

import argparse  # argumentos de la línea de comandos
from collections import Counter, defaultdict  # conteos por clase y grupos de imágenes
from dataclasses import dataclass, replace  # datos de cada grupo y ajustes para el entrenamiento
import json  # índice del dataset, descriptores YOLO e informes
import logging  # progreso por fold
from pathlib import Path  # manejo de rutas
import random  # desempates reproducibles al repartir grupos
import re  # reconoce identificadores de clips BDD y fondos locales
from statistics import mean, stdev  # resumen de las métricas entre folds
from tempfile import mkdtemp  # crea una carpeta distinta para cada corrida


@dataclass
class Grupo:
    """Imágenes y anotaciones reales de un video, más los sintéticos que usan su fondo."""

    imagenes: list[Path]  # imágenes reales que se reparten juntas
    clases: Counter[int]  # cantidad de imágenes reales donde aparece cada clase
    sinteticas: list[Path]  # se usan solo si este grupo está en entrenamiento


def leer_argumentos() -> argparse.Namespace:
    carpeta = Path(__file__).resolve().parent  # permite ejecutar el script desde cualquier directorio
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=carpeta.parent / 'datos' / 'dataset_unificado_argentina')
    parser.add_argument('--config', type=Path, default=carpeta / 'config.yaml')
    parser.add_argument('--salida', type=Path, default=carpeta / 'experimentos' / 'validacion_cruzada')
    parser.add_argument('--folds', type=int, default=3, help='cantidad de folds (por defecto, 3)')
    parser.add_argument('--semilla', type=int, default=42, help='semilla del reparto y del entrenamiento')
    parser.add_argument('--entrenar', action='store_true', help='entrena y evalúa los folds después de prepararlos')
    argumentos = parser.parse_args()
    if argumentos.folds < 2:
        parser.error('--folds debe ser al menos 2')
    return argumentos


def origen(fila: dict) -> str:
    """Recupera el identificador de video del índice o del nombre del clip BDD."""
    if fila['tipo'] == 'sintetica':
        if fila.get('grupo_validacion'):
            return f"local/{fila['grupo_validacion']}"
        fondo = Path(fila['fondo_local_desenfocado']).stem
        coincidencia = re.fullmatch(r'(.+)_\d+ms', fondo)
        if coincidencia:
            return f'local/{coincidencia[1]}'  # mismo identificador que para su video de fondo
    elif fila.get('video'):
        return f"local/{fila.get('grupo_validacion', fila['video'])}"  # conecta videos con una escena compartida conocida
    elif fila['tipo'] == 'base_yolo_filtrada':
        clip = Path(fila['imagen_original']).stem
        if re.fullmatch(r'bdd_[0-9a-f]{8}-[0-9a-f]{8}', clip, flags=re.IGNORECASE):
            return f'bdd/{clip.lower()}'
    raise ValueError(f"No se pudo identificar el video de {fila['imagen']}")


def cargar_dataset(raiz: Path) -> tuple[dict[int, str], dict[str, Grupo], dict[str, Grupo]]:
    indice = json.loads((raiz / 'indice.json').read_text(encoding='utf-8-sig'))
    nombres = {int(id_clase): nombre for nombre, id_clase in indice['clases_yolo'].items()}
    desarrollo, test = {}, {}  # train/val entran al reparto y test se conserva fijo
    imagenes_vistas = set()
    for fila in indice['habilitadas']:
        clave = origen(fila)
        imagen, etiqueta = raiz / fila['imagen'], raiz / fila['etiqueta']
        if imagen in imagenes_vistas or not imagen.is_file() or not etiqueta.is_file():
            raise ValueError(f'Imagen repetida o falta el archivo de imagen o etiqueta: {imagen}')
        imagenes_vistas.add(imagen)
        split = fila['split']
        sintetica = fila['tipo'] == 'sintetica'
        if split not in {'train', 'val', 'test'} or (sintetica and split != 'train'):
            raise ValueError(f'Partición inesperada ({split}): {imagen}')
        grupos = test if split == 'test' else desarrollo
        grupo = grupos.setdefault(clave, Grupo([], Counter(), []))
        if sintetica:
            grupo.sinteticas.append(imagen)
            continue
        # Una imagen con varias cajas de una clase cuenta una vez para estratificar esa clase.
        presentes = {int(linea.split()[0]) for linea in etiqueta.read_text(encoding='utf-8-sig').splitlines() if linea.strip()}
        if presentes - nombres.keys():
            raise ValueError(f'IDs de clase desconocidos en {etiqueta}: {presentes - nombres.keys()}')
        grupo.imagenes.append(imagen)
        grupo.clases.update(presentes)
    compartidos = desarrollo.keys() & test.keys()
    if compartidos:
        raise ValueError(f'Test comparte videos con train o val: {sorted(compartidos)}')
    sin_original = [clave for clave, grupo in desarrollo.items() if not grupo.imagenes]
    if sin_original:
        raise ValueError(f'Fondos sintéticos sin video de origen en train o val: {sin_original}')
    if not test:
        raise ValueError('El índice no contiene un test fijo')
    return nombres, desarrollo, test


def repartir(grupos: dict[str, Grupo], cantidad: int, semilla: int) -> list[list[str]]:
    """Busca un reparto reproducible que equilibre tamaños y presencia de clases."""
    if len(grupos) < cantidad:
        raise ValueError(f'Hay {len(grupos)} grupos para {cantidad} folds')
    totales, soporte = Counter(), Counter()
    total_imagenes = sum(len(grupo.imagenes) for grupo in grupos.values())
    for grupo in grupos.values():
        totales.update(grupo.clases)
        soporte.update(grupo.clases.keys())
    azar = random.Random(semilla)
    mejor_puntaje, mejor_reparto = None, []
    for intento in range(24):
        claves = sorted(grupos)
        azar.shuffle(claves)  # desempata grupos con perfiles similares
        prioridad = {clave: sum((n / totales[c]) ** 2 for c, n in grupo.clases.items())
                     + (len(grupo.imagenes) / total_imagenes) ** 2 for clave, grupo in grupos.items()}
        claves.sort(key=lambda clave: prioridad[clave] * (1 + azar.random() * intento / 24), reverse=True)
        folds = [[] for _ in range(cantidad)]
        conteos = [Counter() for _ in folds]
        tamanos = [0 for _ in folds]
        for clave in claves:
            grupo = grupos[clave]
            fraccion = len(grupo.imagenes) / total_imagenes
            # Cambio de error cuadrático al sumar el grupo, más el incentivo de cubrir clases faltantes.
            puntajes = [
                (2 * tamanos[i] / total_imagenes + fraccion - 2 / cantidad) * fraccion
                + sum((2 * conteos[i][c] + n - 2 * totales[c] / cantidad) * n / totales[c] ** 2
                      - (2 / soporte[c] if not conteos[i][c] else 0) for c, n in grupo.clases.items())
                for i in range(cantidad)
            ]
            destino = min(range(cantidad), key=lambda i: (bool(folds[i]), puntajes[i], tamanos[i]))
            folds[destino].append(clave)
            conteos[destino].update(grupo.clases)
            tamanos[destino] += len(grupo.imagenes)
        faltantes = sum(min(soporte[c], cantidad) - sum(c in conteo for conteo in conteos) for c in totales)
        error = sum((n / total_imagenes - 1 / cantidad) ** 2 for n in tamanos)
        error += sum((conteo[c] / total - 1 / cantidad) ** 2
                     for c, total in totales.items() for conteo in conteos)
        puntaje = (faltantes, error)
        if mejor_puntaje is None or puntaje < mejor_puntaje:
            mejor_puntaje, mejor_reparto = puntaje, folds
    return mejor_reparto


def escribir_json(ruta: Path, datos: dict) -> None:
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def escribir_lista(ruta: Path, imagenes: list[Path]) -> None:
    ruta.write_text(''.join(f'{p.as_posix()}\n' for p in sorted(imagenes)), encoding='utf-8')


def resumen_particion(claves: list[str], grupos: dict[str, Grupo], nombres: dict[int, str], sinteticos: bool = False) -> dict:
    conteos = Counter()
    for clave in claves:
        conteos.update(grupos[clave].clases)
    return {
        'grupos': sorted(claves),
        'imagenes_reales': sum(len(grupos[clave].imagenes) for clave in claves),
        'imagenes_sinteticas': sum(len(grupos[clave].sinteticas) for clave in claves) if sinteticos else 0,
        'imagenes_reales_por_clase': {nombre: conteos[c] for c, nombre in nombres.items()},
    }


def preparar_folds(salida: Path, nombres: dict[int, str], desarrollo: dict[str, Grupo],
                   test: dict[str, Grupo], folds: list[list[str]], semilla: int) -> list[Path]:
    descriptores, resumen = [], []
    for numero, val in enumerate(folds, start=1):
        carpeta = salida / f'fold_{numero:02d}'
        carpeta.mkdir()
        train = sorted(desarrollo.keys() - set(val))
        train_txt, val_txt = carpeta / 'train.txt', carpeta / 'val.txt'
        escribir_lista(train_txt, [p for clave in train for p in desarrollo[clave].imagenes + desarrollo[clave].sinteticas])
        escribir_lista(val_txt, [p for clave in val for p in desarrollo[clave].imagenes])
        # JSON es válido como YAML y conserva las rutas absolutas, incluso con espacios en Windows.
        descriptor = carpeta / 'data.yaml'
        escribir_json(descriptor, {'path': carpeta.as_posix(), 'train': train_txt.as_posix(),
                                  'val': val_txt.as_posix(), 'names': nombres})
        descriptores.append(descriptor)
        resumen.append({'fold': numero, 'train': resumen_particion(train, desarrollo, nombres, sinteticos=True),
                        'val': resumen_particion(val, desarrollo, nombres)})
    soporte = Counter(c for grupo in desarrollo.values() for c in grupo.clases)
    escribir_json(salida / 'estratos.json', {
        'semilla': semilla, 'folds': resumen, 'test_reservado': resumen_particion(list(test), test, nombres),
        'grupos_reales_por_clase': {nombre: soporte[c] for c, nombre in nombres.items()},
    })
    return descriptores


def entrenar_folds(descriptores: list[Path], config_path: Path, semilla: int, nombres: dict[int, str]) -> None:
    from ultralytics import YOLO  # prepara los estratos sin importar PyTorch ni Ultralytics

    from configuracion import cargar_configuracion, fijar_semillas, verificar_gpu
    from entrenamiento import entrenar_detector, liberar_memoria_gpu

    config = cargar_configuracion(config_path)
    modelo_local = config_path.parent / config.modelo_base
    config = replace(config, semilla=semilla, modelo_base=str(modelo_local) if modelo_local.is_file() else config.modelo_base)
    verificar_gpu(config.dispositivo)
    fijar_semillas(semilla)
    resultados = []
    map_por_clase = defaultdict(list)
    for descriptor in descriptores:
        carpeta = descriptor.parent
        logging.info('Entrenando %s con %s', carpeta.name, config.modelo_base)
        pesos = entrenar_detector(descriptor, carpeta, config)  # inicializa el mismo modelo base en cada fold
        modelo = YOLO(str(pesos))
        metricas = modelo.val(data=str(descriptor), split='val', imgsz=config.imgsz,
                              batch=config.batch_evaluacion, device=config.dispositivo, workers=config.workers,
                              project=str(carpeta), name='evaluacion_val', exist_ok=True, plots=False)
        por_clase = {}
        for indice, clase in enumerate(metricas.box.ap_class_index):
            nombre = nombres[int(clase)]
            por_clase[nombre] = float(metricas.box.ap[indice])
            map_por_clase[nombre].append(por_clase[nombre])
        resultado = {'fold': carpeta.name, 'best_pt': pesos.as_posix(), 'mAP50': float(metricas.box.map50),
                     'mAP50-95': float(metricas.box.map), 'mAP50-95_por_clase': por_clase}
        escribir_json(carpeta / 'metricas.json', resultado)
        resultados.append(resultado)
        del modelo, metricas
        liberar_memoria_gpu()
    agregar = {'folds_medidos': len(resultados), 'folds': resultados}
    for metrica in ('mAP50', 'mAP50-95'):
        valores = [fold[metrica] for fold in resultados]
        agregar[metrica] = {'media': mean(valores), 'desvio_estandar': stdev(valores)}
    agregar['mAP50-95_por_clase'] = {
        nombre: {'media': mean(valores) if valores else None, 'folds_medidos': len(valores),
                 'desvio_estandar': stdev(valores) if len(valores) > 1 else None}
        for nombre in nombres.values() for valores in [map_por_clase[nombre]]
    }
    escribir_json(descriptores[0].parent.parent / 'metricas.json', agregar)


def crear_corrida_folds(dataset: Path, destino: Path, cantidad: int, semilla: int) -> tuple[list[Path], dict[int, str]]:
    """Prepara los descriptores YOLO y el informe de reparto de una corrida."""
    nombres, desarrollo, test = cargar_dataset(dataset.resolve())  # lee las clases, los grupos y el test fijo
    folds = repartir(desarrollo, cantidad, semilla)  # equilibra presencia de clases por video
    destino = destino.resolve()  # conserva rutas absolutas en los descriptores
    destino.mkdir(parents=True, exist_ok=True)  # crea la carpeta común de experimentos
    salida = Path(mkdtemp(prefix='cv_', dir=destino))  # conserva las corridas anteriores
    descriptores = preparar_folds(salida, nombres, desarrollo, test, folds, semilla)  # escribe las listas y el informe
    logging.info('Estratos preparados en %s', salida)
    return descriptores, nombres  # se reutilizan si se pidió entrenar


def main() -> None:
    argumentos = leer_argumentos()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    descriptores, nombres = crear_corrida_folds(argumentos.dataset, argumentos.salida, argumentos.folds, argumentos.semilla)
    if argumentos.entrenar:
        entrenar_folds(descriptores, argumentos.config.resolve(), argumentos.semilla, nombres)


if __name__ == '__main__':  # permite múltiples workers de Ultralytics en Windows
    main()
