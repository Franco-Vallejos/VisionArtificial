# Detector de tránsito con YOLO26s

El detector reconoce 23 clases de tránsito, incluidos vehículos, personas, señales y estados de semáforos.

| Modelo | Dataset | Entrenamiento | Validación | Test |
| --- | --- | ---: | ---: | ---: |
| v11 | `dataset_unificado_argentina_v11` | 7.727 | 1.656 | 1.656 |
| v12 | `dataset_argentina_v11_chequeadas` | 581 | 125 | 127 |

Los dos checkpoints y sus reportes de evaluación están en la [Release del proyecto](https://github.com/Franco-Vallejos/VisionArtificial/releases/tag/tp-integrador-modelos).

## Instalar

Desde `TP-INTEGRADOR/Detector_Transito`, activá un entorno virtual e instalá PyTorch para tu equipo y después las dependencias:

```powershell
python -m pip install -r requirements.txt
```

El archivo `config.yaml` usa la primera GPU CUDA. En un equipo sin CUDA, cambiá `dispositivo: 0` por `dispositivo: cpu`.

## Demo con v12

Descargá `detector-transito-v12-best.pt` de la Release y guardalo en `Detector_Transito/modelos/`. Colocá tus MP4 en `Detector_Transito/videos/` y corré:

```powershell
python main.py --fase demo --sin-ventana --pesos ".\modelos\detector-transito-v12-best.pt"
```

Los MP4 anotados quedan en `videos_procesados/`. Para usar v11, indicá el archivo `detector-transito-v11-best.pt`.

## Validación y entrenamiento

Para validar o entrenar necesitás un dataset YOLO propio con `data.yaml`, imágenes y etiquetas. Para reproducir las corridas de la tabla necesitás una copia local del dataset correspondiente. Los datasets se mantienen aparte de la publicación. Con tu copia de la v11 chequeada en `TP-INTEGRADOR/datos/`:

```powershell
python main.py --fase validar --data "..\datos\dataset_argentina_v11_chequeadas\data.yaml" --pesos ".\modelos\detector-transito-v12-best.pt"
python main.py --fase entrenar --data "..\datos\dataset_argentina_v11_chequeadas\data.yaml"
```

Para preparar tres folds estratificados por clase y agrupados por video:

```powershell
python main.py --fase folds --folds 3 --data "..\datos\dataset_argentina_v11_chequeadas\data.yaml"
```

Los reportes de v11 incluyen el historial y la validación guardada. Los de v12 también incluyen el test y las métricas por clase. Los pesos se entrenaron con [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/).
