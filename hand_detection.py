import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Parámetros de confianza ajustables
mp_drawing = mp.solutions.drawing_utils

# Inicializar la captura de video desde la webcam
cap = cv2.VideoCapture(0)

# Comprobar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

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
    
    # Mostrar el cuadro con las detecciones
    cv2.imshow('Detección de Manos - MediaPipe', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar los recursos de la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
