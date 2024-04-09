import tensorflow as tf
import cv2
import numpy as np

# Cargar el modelo desde el archivo .h5
modelo_cargado = tf.keras.models.load_model('C:\\Users\\user\\Desktop\\ProyectoFunda\\codigo\\modelo_perros_gatos.h5')
# Cargar y preprocesar la imagen de prueba
imagen_prueba = cv2.imread('C:\\Users\\user\\Desktop\\ProyectoFunda\\codigo\\Imagenes_prueba\\Bruce.jpg')
imagen_prueba = cv2.resize(imagen_prueba, (224, 224))
imagen_prueba = imagen_prueba / 255.0  # Normalizar la imagen
imagen_prueba = np.expand_dims(imagen_prueba, axis=0)  # Agregar dimensión de lote (batch dimension)

# Realizar la clasificación de la imagen
prediccion = modelo_cargado.predict(imagen_prueba)

# Obtener la etiqueta de la clase predicha
etiqueta_predicha = "Perro" if prediccion[0][0] > 0.5 else "Gato"

# Imprimir el resultado
print("La imagen es un:", etiqueta_predicha)
