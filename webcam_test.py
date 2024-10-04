import cv2

# Inicializar la captura de video desde la webcam
cap = cv2.VideoCapture(0)  # El argumento '0' indica la cámara predeterminada

# Comprobar si la cámara se abrió correctamente
if not cap.isOpened():
    print("Error: No se puede abrir la cámara")
    exit()

# Capturar video en un bucle
while True:
    # Leer el cuadro actual de la webcam
    ret, frame = cap.read()

    # Si no se captura correctamente, salir del bucle
    if not ret:
        print("Error al recibir cuadro de la cámara. Saliendo...")
        break

    # Mostrar el cuadro en una ventana
    cv2.imshow('Webcam - Prueba', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) == ord('q'):
        break

# Liberar los recursos de la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
