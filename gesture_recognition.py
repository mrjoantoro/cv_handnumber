import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

#Paso 1: Preparación del dataset

# Configuraciones del dataset
IMG_SIZE = 64  # Tamaño al que redimensionaremos las imágenes
DATA_DIR = "hand_gesture_data"  # Directorio donde están las imágenes de gestos

# Función para cargar y preprocesar las imágenes
def load_data():
    images = []
    labels = []
    
    # Recorrer las carpetas de cada número (0-9)
    for label in range(10):
        path = os.path.join(DATA_DIR, str(label))
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Cargar imagen en escala de grises
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))  # Redimensionar la imagen
            images.append(img)
            labels.append(label)
    
    # Convertir a numpy arrays y normalizar
    images = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalización a valores entre 0 y 1
    labels = np.array(labels)
    
    return images, labels

# Cargar los datos
images, labels = load_data()

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convertir las etiquetas a formato one-hot (para clasificación multiclase)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

print(f"Dataset cargado: {len(X_train)} imágenes para entrenamiento, {len(X_test)} imágenes para prueba.")

#Paso 2: Construcción del modelo CNN

# Definir el modelo CNN
def create_model():
    model = models.Sequential()
    
    # Definir explícitamente la entrada usando layers.Input
    model.add(layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)))
    
    # Primera capa convolucional
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Segunda capa convolucional
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Tercera capa convolucional
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    # Aplanar y agregar capas densas
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))  # 10 clases (números 0-9)
    
    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Crear el modelo
model = create_model()
model.summary()  # Mostrar un resumen del modelo

#Paso 3: Entrenamiento del modelo

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Guardar el modelo entrenado
model.save('gesture_recognition_model.h5')

#Paso 4: Evaluación del modelo

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Precisión en el conjunto de prueba: {test_acc * 100:.2f}%")

#Paso 5: Visualización del rendimiento (opcional)

# Graficar precisión del entrenamiento y validación
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.xlabel('Epoch')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Graficar pérdida del entrenamiento y validación
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.xlabel('Epoch')
plt.ylabel('Pérdida')
plt.legend()
plt.show()