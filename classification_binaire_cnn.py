import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# I. Chargement , prétraitement et exploration des données

# 1. Chargement des images depuis les dossiers avec augmentation de données

# Définition des chemins des datasets
train_dir = "/chest_xray_data/chest_xray/train/"
val_dir = "/chest_xray_data/chest_xray/val/"
test_dir = "/chest_xray_data/chest_xray/test/"


# Paramètres
IMAGE_HEIGHT= 224
IMAGE_WITHD= 224
BATCH_SIZE = 32
EPOCHS = 10

# Prétraitement des images
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(IMAGE_HEIGHT, IMAGE_WITHD),
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Modifier selon le nombre de classes
    subset='training'
)

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(IMAGE_HEIGHT, IMAGE_WITHD),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Définition du modèle CNN
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Utiliser softmax pour plusieurs classes
])

# Compilation du modèle
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',  # Modifier en categorical_crossentropy si plusieurs classes
    metrics=['accuracy']
)

# Entraînement du modèle
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# Affichage de l’évolution de la précision
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()

# Sauvegarde du modèle
model.save('cnn_model.h5')

