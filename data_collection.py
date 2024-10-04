import cv2
import mediapipe as mp
import os

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inicializar la captura de video desde la webcam
cap = cv2.VideoCapture(0)

# Ruta base donde se guardarán las imágenes
BASE_DIR = "hand_gesture_data"

# Variable para guardar el número actual que se está capturando
current_number = input("Ingresa el número que deseas capturar (0-9): ")

# Crear la carpeta correspondiente si no existe
output_dir = os.path.join(BASE_DIR, current_number)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Contador de imágenes capturadas
img_counter = 0

# Capturar video en un bucle
while True:
    # Leer el cuadro actual de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Error al recibir cuadro de la cámara. Saliendo...")
        break

    # Convertir el cuadro a formato RGB (MediaPipe usa RGB en lugar de BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el cuadro con MediaPipe Hands
    result = hands.process(rgb_frame)

    # Si se detectan manos
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Dibujar las landmarks y las conexiones sobre el cuadro original
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

    # Mostrar el cuadro en una ventana
    cv2.imshow('Captura de Gestos de Manos', frame)

    # Guardar imagen cuando se presiona la tecla 's'
    key = cv2.waitKey(1)
    if key == ord('s'):
        img_name = os.path.join(output_dir, f"{current_number}_{img_counter}.png")
        cv2.imwrite(img_name, frame)
        print(f"Imagen {img_name} guardada!")
        img_counter += 1

    # Salir si se presiona la tecla 'q'
    if key == ord('q'):
        break

# Liberar los recursos de la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
