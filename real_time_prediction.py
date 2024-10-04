import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Cargar el modelo entrenado
model = tf.keras.models.load_model('gesture_recognition_model.h5')

# Configuración de MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Definir el tamaño de la imagen que el modelo espera
IMG_SIZE = 64

# Iniciar captura de video
cap = cv2.VideoCapture(0)

while True:
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
            # Dibujar las landmarks sobre el cuadro
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Obtener el bounding box de la mano
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Recortar la región de la mano
            hand_region = frame[y_min:y_max, x_min:x_max]

            # Asegurarse de que la región de la mano tiene el tamaño adecuado
            if hand_region.shape[0] > 0 and hand_region.shape[1] > 0:
                # Redimensionar la imagen al tamaño esperado por el modelo
                hand_region_resized = cv2.resize(hand_region, (IMG_SIZE, IMG_SIZE))
                hand_region_gray = cv2.cvtColor(hand_region_resized, cv2.COLOR_BGR2GRAY)  # Convertir a escala de grises
                hand_region_normalized = hand_region_gray / 255.0  # Normalizar los valores de píxeles
                hand_region_reshaped = hand_region_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)  # Ajustar forma para el modelo

                # Hacer la predicción
                prediction = model.predict(hand_region_reshaped)
                predicted_number = np.argmax(prediction)

                # Mostrar el número predicho en la imagen
                cv2.putText(frame, f'Prediccion: {predicted_number}', (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el cuadro con las predicciones
    cv2.imshow('Predicción en tiempo real - Gestos de mano', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar los recursos de la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
