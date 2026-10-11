# TP5 — Localización con PnP

Visión Artificial (UNLaM). Estima la **pose de la cámara** a partir de un marcador **ArUco** fijo usando **PnP** (`cv.solvePnP`) y la muestra en vivo en **Foxglove**. Si la cámara se mueve, su posición se actualiza en el visualizador.

> Explicación completa, conceptos, capturas y tutorial paso a paso: **`TP5_Localizacion_PnP_Manual.docx`**.

## Consigna

> En base a la detección de marcadores ARUCO, estimar la pose de la cámara utilizando PnP. Utilizando un visualizador (como foxglove) mostrar la localización de la cámara en el mundo. Si la cámara se mueve, la pose deberá actualizarse y moverse en el visualizador. Para poder realizar este trabajo deberán realizar el proceso de calibración de cámara.

## Archivos

| Archivo | Qué es |
|---|---|
| `calibrate.py` | Calibración de cámara. Es el `calibrate.py` del profesor ([VisiAr/Calibration](https://github.com/AlejandroSilvestri/VisiAr/tree/main/Calibration)); solo se agregó fijar la resolución en 640×480. |
| `pattern_chessboard 6 x 9.png` | Patrón de calibración del profesor (10×7 cuadros = 9×6 esquinas internas). **Tiene que estar en la misma carpeta, con este nombre exacto.** |
| `tp5_localizacion_pnp.py` | Localización. Basado en `poseARUCO.py` del profesor ([Marcadores-fiduciarios](https://github.com/Laboratorio-de-Robotica/Marcadores-fiduciarios)). Los cambios están marcados con `[TP5]`. |
| `TP5_Localizacion_PnP_Manual.docx` | Manual, tutorial y guía de estudio. |

## Requisitos

- **Python 3.10 o superior** (el código usa `match/case`).
- Librerías:
  ```
  python -m pip install opencv-python numpy foxglove-sdk
  ```
- Una webcam.
- **Patrón impreso:** `pattern_chessboard 6 x 9.png` en A4 horizontal, con margen blanco, pegado plano sobre cartón.
- **Marcador ArUco impreso:** diccionario **4x4** (DICT_4X4_50), por ejemplo desde chev.me/arucogen. Se usó uno de 10 cm.
- Cuenta gratuita en [app.foxglove.dev](https://app.foxglove.dev).

## Uso

Ejecutar **siempre desde una terminal parada en esta carpeta** (no con el botón ▶ de VS Code: busca el PNG en otra carpeta y falla).

### 1. Calibrar (una sola vez)

```
python calibrate.py
```

| Tecla | Acción |
|---|---|
| `espacio` | Captura (solo se guarda si el tablero está detectado: se ven líneas de colores y la terminal dice `N pictures taken`) |
| `c` | Calibra con las capturas tomadas e imprime K y la distorsión en la terminal |
| `ESC` | Sale |

Tomar **20–25 capturas variadas**: centro, esquinas y bordes de la imagen, cerca, lejos e inclinado. Después de `c`, **copiar los valores**: el programa no los guarda.

### 2. Cargar la calibración

En `tp5_localizacion_pnp.py`, reemplazar `cameraMatrix` y `distCoeffs` (líneas `[TP5]`) por los valores obtenidos, separados por comas. Ajustar `squareLength` al lado real del ArUco **en cm**.

Valores actuales (webcam usada en el TP, 640×480, 25 capturas):

```python
cameraMatrix = np.array(
    [[449.22, 0.0, 320.15],
     [0.0, 458.83, 311.39],
     [0.0, 0.0, 1.0]], np.float32)
distCoeffs = np.array([-0.08411395, 0.0761868, 0.0, 0.0, -0.0191586])
squareLength = 10  # cm
```

> Estos valores solo sirven para **esa** webcam a 640×480. Con otra cámara hay que recalibrar.

### 3. Localizar

Cerrar `calibrate.py` (la webcam la usa un solo programa a la vez) y ejecutar:

```
python tp5_localizacion_pnp.py
```

Apuntar la webcam al ArUco (fijo y plano). En la ventana `webcam` se ve el contorno verde y tres ejes. `ESC` para salir.

### 4. Ver en Foxglove

Con `tp5_localizacion_pnp.py` corriendo:

1. Entrar a **app.foxglove.dev** → **Open connection**.
2. Elegir **Foxglove WebSocket** (no "Remote Access", que es pago).
3. URL `ws://localhost:8765` → **Open**.
4. Apuntar la webcam al ArUco (si no lo detecta, no se envían datos).
5. En el panel 3D, **Frame → Display frame → `mundo`**.

Se ven dos sistemas de ejes: **`mundo`** (el ArUco, en el origen) y **`camara`** (la webcam). Al mover la cámara, `camara` se mueve.

## Cómo funciona

```
webcam → detectMarkers → solvePnP (+K, distorsión) → Tcq (marcador visto desde la cámara)
                                                        │ np.linalg.inv
                                                        ▼
                                    Tqc (cámara vista desde el marcador)
                                                        │ foxglove.log("/tf")
                                                        ▼
                                    Foxglove: frames "mundo" y "camara"
```

## Problemas frecuentes

| Síntoma | Solución |
|---|---|
| `can't open/read file` y error en `cv.imshow` | El PNG no está en la carpeta desde donde se ejecuta. Ejecutar desde esta carpeta. |
| `nimages > 0` al apretar `c` | No se guardó ninguna captura: capturar solo cuando se ven las líneas de colores. |
| El tablero no se detecta | Tiene que ser el patrón 10×7 cuadros (un tablero de ajedrez de juego, 8×8, no sirve). |
| La localización no abre la cámara | Cerrar `calibrate.py`. |
| Foxglove: Display frame vacío / "Waiting for next message" | Apuntar la webcam al ArUco. |
| Foxglove: advertencia "Age 497702 hr" | Los mensajes van sin timestamp. No afecta la pose; se puede ignorar. |
