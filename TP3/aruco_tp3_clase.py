#!/usr/bin/env python3
"""TP3 - Localizacion homografica minima con ArUco."""

import sys

import cv2
import numpy as np


CAMERA_INDEX = 0
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
MARKER_SIZE_MM = 100.0
WORLD_WIDTH_MM = 400.0
WORLD_HEIGHT_MM = 300.0
PIXELS_PER_MM = 2.0
GRID_STEP_MM = 50.0
W2D_WIDTH_PX = int(WORLD_WIDTH_MM * PIXELS_PER_MM)
W2D_HEIGHT_PX = int(WORLD_HEIGHT_MM * PIXELS_PER_MM)
WINDOW_CAM = "Cam"
WINDOW_W2D = "W2D"
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_MARKER = (0, 255, 0)
COLOR_GRID = (100, 100, 100)
COLOR_X = (0, 0, 255)
COLOR_Y = (0, 200, 0)
COLOR_POSE = (0, 255, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_MEASURE = (255, 0, 255)


# El diccionario debe coincidir con el utilizado para generar el marcador impreso.
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def detectar(frame):
    """Detecta marcadores ArUco en una version monocromatica del cuadro."""
    # El detector utiliza intensidad; el color no aporta informacion al patron binario.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def elegir_marcador(corners, ids, marker_id=None):
    """Selecciona el primer marcador visible o el marcador cuyo identificador ya fue registrado."""
    if ids is None:
        return None, None
    # OpenCV devuelve los identificadores como N x 1 y se aplastan para poder buscarlos por indice.
    flat_ids = ids.flatten()
    if marker_id is None:
        return corners[0], int(flat_ids[0])
    indices = np.where(flat_ids == marker_id)[0]
    if len(indices) == 0:
        return None, None
    index = int(indices[0])
    return corners[index], marker_id


def registrar(frame, marker_corners):
    """Define el sistema metrico desde el ArUco y calcula las homografias entre camara, mundo y vista cenital."""
    half = MARKER_SIZE_MM / 2.0
    center_x = W2D_WIDTH_PX / 2.0
    center_y = W2D_HEIGHT_PX / 2.0
    half_px = half * PIXELS_PER_MM
    # Cada fila es un punto [x, y] y el orden es: superior izquierdo, superior derecho, inferior derecho e inferior izquierdo.
    image_points_matrix = np.array([
        marker_corners[0][0],
        marker_corners[0][1],
        marker_corners[0][2],
        marker_corners[0][3]
    ], dtype=np.float32)
    # El centro del marcador es el origen del mundo, con x hacia la derecha e y hacia arriba.
    world_points_matrix = np.array([
        [-half, half],
        [half, half],
        [half, -half],
        [-half, -half]
    ], dtype=np.float32)
    # La vista W2D representa el mismo cuadrado fisico, centrado y expresado ahora en pixeles.
    view_points_matrix = np.array([
        [center_x - half_px, center_y - half_px],
        [center_x + half_px, center_y - half_px],
        [center_x + half_px, center_y + half_px],
        [center_x - half_px, center_y + half_px]
    ], dtype=np.float32)
    # Cuatro correspondencias coplanares permiten calcular la transformacion proyectiva entre ambos planos.
    h_image_to_world = cv2.getPerspectiveTransform(image_points_matrix, world_points_matrix)
    h_image_to_view = cv2.getPerspectiveTransform(image_points_matrix, view_points_matrix)
    # La inversa reproyecta coordenadas metricas del mundo sobre la imagen original de la camara.
    h_world_to_image = np.linalg.inv(h_image_to_world)
    # La imagen cenital queda congelada al registrar para usarla como referencia estatica.
    background = cv2.warpPerspective(frame, h_image_to_view, (W2D_WIDTH_PX, W2D_HEIGHT_PX))
    return h_image_to_world, h_image_to_view, h_world_to_image, background


def localizar(marker_corners, h_image_to_world, h_image_to_view):
    """Calcula la posicion y orientacion actuales del ArUco en el sistema metrico registrado."""
    image_points_matrix = marker_corners.reshape(4, 2).astype(np.float32)
    # perspectiveTransform requiere la forma N x 1 x 2 aunque cada punto conserve las columnas [x, y].
    image_points_cv_matrix = image_points_matrix.reshape(4, 1, 2)
    # Las mismas esquinas se transforman al plano metrico para medir y a W2D para dibujar.
    world_corners_cv_matrix = cv2.perspectiveTransform(image_points_cv_matrix, h_image_to_world)
    view_corners_cv_matrix = cv2.perspectiveTransform(image_points_cv_matrix, h_image_to_view)
    world_corners_matrix = world_corners_cv_matrix.reshape(4, 2)
    view_corners_matrix = view_corners_cv_matrix.reshape(4, 2)
    # El centroide fija la posicion y el lado superior del marcador determina su orientacion.
    center_mm = world_corners_matrix.mean(axis=0)
    center_view = view_corners_matrix.mean(axis=0)
    direction = world_corners_matrix[1] - world_corners_matrix[0]
    # atan2 conserva el cuadrante y entrega el giro con signo respecto del eje x del mundo.
    angle_deg = float(np.degrees(np.arctan2(direction[1], direction[0])))
    return center_mm, angle_deg, view_corners_matrix.astype(np.int32), center_view.astype(np.int32)


def dibujar_grilla_cam(frame, h_world_to_image):
    """Proyecta la grilla metrica del mundo sobre la perspectiva de la camara."""
    half_width = WORLD_WIDTH_MM / 2.0
    half_height = WORLD_HEIGHT_MM / 2.0
    # Cada linea se define por sus dos extremos en milimetros y luego se reproyecta sobre la camara.
    for x in np.arange(-half_width, half_width + GRID_STEP_MM, GRID_STEP_MM):
        world_line_matrix = np.array([
            [x, -half_height],
            [x, half_height]
        ], dtype=np.float32)
        world_line_cv_matrix = world_line_matrix.reshape(2, 1, 2)
        image_line_cv_matrix = cv2.perspectiveTransform(world_line_cv_matrix, h_world_to_image)
        image_line_matrix = image_line_cv_matrix.reshape(2, 2).astype(np.int32)
        cv2.line(frame, tuple(image_line_matrix[0]), tuple(image_line_matrix[1]), COLOR_GRID, 1, cv2.LINE_AA)
    for y in np.arange(-half_height, half_height + GRID_STEP_MM, GRID_STEP_MM):
        world_line_matrix = np.array([
            [-half_width, y],
            [half_width, y]
        ], dtype=np.float32)
        world_line_cv_matrix = world_line_matrix.reshape(2, 1, 2)
        image_line_cv_matrix = cv2.perspectiveTransform(world_line_cv_matrix, h_world_to_image)
        image_line_matrix = image_line_cv_matrix.reshape(2, 2).astype(np.int32)
        cv2.line(frame, tuple(image_line_matrix[0]), tuple(image_line_matrix[1]), COLOR_GRID, 1, cv2.LINE_AA)

def dibujar_grilla_w2d(frame):
    """Dibuja la grilla metrica regular sobre la vista cenital."""
    half_width = WORLD_WIDTH_MM / 2.0
    half_height = WORLD_HEIGHT_MM / 2.0
    # El origen W2D esta en el centro y el eje vertical de la imagen crece hacia abajo.
    for x in np.arange(-half_width, half_width + GRID_STEP_MM, GRID_STEP_MM):
        column = int(W2D_WIDTH_PX / 2.0 + x * PIXELS_PER_MM)
        cv2.line(frame, (column, 0), (column, W2D_HEIGHT_PX - 1), COLOR_GRID, 1, cv2.LINE_AA)
    for y in np.arange(-half_height, half_height + GRID_STEP_MM, GRID_STEP_MM):
        row = int(W2D_HEIGHT_PX / 2.0 - y * PIXELS_PER_MM)
        cv2.line(frame, (0, row), (W2D_WIDTH_PX - 1, row), COLOR_GRID, 1, cv2.LINE_AA)



def manejar_click(event, x, y, _flags, state):
    """Guarda el ultimo punto seleccionado con el boton izquierdo en la vista cenital."""
    if event == cv2.EVENT_LBUTTONDOWN:
        state["point"] = np.array([x, y], dtype=np.int32)


def convertir_w2d_a_mm(point):
    """Convierte un pixel de la vista cenital en coordenadas metricas respecto del origen."""
    # En x se resta el centro; en y se invierte la resta porque las filas crecen hacia abajo.
    x_mm = (point[0] - W2D_WIDTH_PX / 2.0) / PIXELS_PER_MM
    y_mm = (W2D_HEIGHT_PX / 2.0 - point[1]) / PIXELS_PER_MM
    return np.array([x_mm, y_mm], dtype=np.float32)


def dibujar_medicion_cam(frame, marker_corners, center_mm, clicked_view, h_world_to_image):
    """Reproyecta el punto elegido sobre la camara y dibuja su distancia al centro del ArUco."""
    clicked_mm = convertir_w2d_a_mm(clicked_view)
    # El punto recorre W2D -> mundo metrico -> perspectiva de la camara.
    clicked_world_matrix = np.array([
        [clicked_mm]
    ], dtype=np.float32)
    clicked_camera_matrix = cv2.perspectiveTransform(clicked_world_matrix, h_world_to_image)
    clicked_cam = clicked_camera_matrix.reshape(2).astype(np.int32)
    marker_image_matrix = marker_corners.reshape(4, 2)
    center_cam = marker_image_matrix.mean(axis=0).astype(np.int32)
    # La norma euclidea entrega la distancia real sobre el plano en milimetros.
    distance_mm = float(np.linalg.norm(clicked_mm - center_mm))
    cv2.line(frame, tuple(center_cam), tuple(clicked_cam), COLOR_MEASURE, 2, cv2.LINE_AA)
    cv2.circle(frame, tuple(clicked_cam), 5, COLOR_MEASURE, -1, cv2.LINE_AA)
    cv2.putText(frame, f"{distance_mm:.1f} mm", (int(clicked_cam[0]) + 10, max(20, int(clicked_cam[1]) - 10)), FONT, 0.55, COLOR_MEASURE, 2, cv2.LINE_AA)


def dibujar_w2d(background, center_mm, angle_deg, view_corners, center_view, clicked_view):
    """Compone la vista cenital con grilla, pose del ArUco y medicion seleccionada."""
    frame = background.copy()
    dibujar_grilla_w2d(frame)
    origin = (W2D_WIDTH_PX // 2, W2D_HEIGHT_PX // 2)
    cv2.arrowedLine(frame, origin, (origin[0] + 70, origin[1]), COLOR_X, 2, cv2.LINE_AA, 0, 0.2)
    cv2.arrowedLine(frame, origin, (origin[0], origin[1] - 70), COLOR_Y, 2, cv2.LINE_AA, 0, 0.2)
    cv2.putText(frame, "x", (origin[0] + 76, origin[1] + 6), FONT, 0.6, COLOR_X, 2, cv2.LINE_AA)
    cv2.putText(frame, "y", (origin[0] - 16, origin[1] - 74), FONT, 0.6, COLOR_Y, 2, cv2.LINE_AA)
    cv2.polylines(frame, [view_corners], True, COLOR_POSE, 2, cv2.LINE_AA)
    angle_rad = np.radians(angle_deg)
    # El seno se resta porque un y positivo del mundo apunta hacia arriba, contrario a las filas de la imagen.
    arrow_tip = (int(center_view[0] + 60 * np.cos(angle_rad)), int(center_view[1] - 60 * np.sin(angle_rad)))
    cv2.arrowedLine(frame, tuple(center_view), arrow_tip, COLOR_POSE, 3, cv2.LINE_AA, 0, 0.25)
    label_position = (int(center_view[0]) + 12, int(center_view[1]) - 12)
    cv2.putText(frame, f"({center_mm[0]:+.1f}, {center_mm[1]:+.1f}) mm", label_position, FONT, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{angle_deg:.1f} deg", (label_position[0], label_position[1] + 20), FONT, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
    if clicked_view is not None:
        clicked_mm = convertir_w2d_a_mm(clicked_view)
        distance_mm = float(np.linalg.norm(clicked_mm - center_mm))
        cv2.line(frame, tuple(center_view), tuple(clicked_view), COLOR_MEASURE, 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(clicked_view), 5, COLOR_MEASURE, -1, cv2.LINE_AA)
        cv2.putText(frame, f"{distance_mm:.1f} mm", (int(clicked_view[0]) + 10, max(20, int(clicked_view[1]) - 10)), FONT, 0.55, COLOR_MEASURE, 2, cv2.LINE_AA)
    return frame


def main():
    """Ejecuta la captura, el registro del plano y la actualizacion de ambas visualizaciones."""
    capture = cv2.VideoCapture(CAMERA_INDEX)
    if not capture.isOpened():
        print("ERROR: no se pudo abrir la camara.")
        sys.exit(1)

    # Las homografias permanecen fijas hasta que el usuario vuelve a registrar con R.
    h_image_to_world = None
    h_image_to_view = None
    h_world_to_image = None
    background = None
    registered_id = None
    click_state = {"point": None}
    w2d_frame = np.zeros((W2D_HEIGHT_PX, W2D_WIDTH_PX, 3), dtype=np.uint8)
    cv2.putText(w2d_frame, "Presione r para registrar", (20, 35), FONT, 0.7, COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.namedWindow(WINDOW_CAM)
    cv2.namedWindow(WINDOW_W2D)
    cv2.setMouseCallback(WINDOW_W2D, manejar_click, click_state)

    while True:
        ok, frame = capture.read()
        if not ok:
            print("ERROR: no se pudo leer la camara.")
            break

        # La deteccion se repite en cada cuadro, mientras que el sistema de referencia sigue siendo el registrado.
        corners, ids = detectar(frame)
        cam_frame = frame.copy()
        if h_world_to_image is not None:
            dibujar_grilla_cam(cam_frame, h_world_to_image)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(cam_frame, corners, ids)

        # Mantener el mismo identificador evita cambiar accidentalmente de referencia entre cuadros.
        current_corners, _ = elegir_marcador(corners, ids, registered_id)
        # La pose se actualiza solo cuando vuelve a verse el marcador registrado; de lo contrario se conserva la ultima W2D valida.
        if registered_id is not None and current_corners is not None:
            center_mm, angle_deg, view_corners, center_view = localizar(current_corners, h_image_to_world, h_image_to_view)
            w2d_frame = dibujar_w2d(background, center_mm, angle_deg, view_corners, center_view, click_state["point"])
            if click_state["point"] is not None:
                dibujar_medicion_cam(cam_frame, current_corners, center_mm, click_state["point"], h_world_to_image)

        cv2.imshow(WINDOW_CAM, cam_frame)
        cv2.imshow(WINDOW_W2D, w2d_frame)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            break
        if key in (ord("r"), ord("R")):
            marker_corners, marker_id = elegir_marcador(corners, ids)
            if marker_corners is None:
                print("No hay un marcador ArUco visible.")
            else:
                # El marcador visible pasa a definir el origen y la escala del plano registrado.
                h_image_to_world, h_image_to_view, h_world_to_image, background = registrar(frame, marker_corners)
                registered_id = marker_id
                click_state["point"] = None
                print(f"Plano registrado con marcador id={registered_id}.")

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
