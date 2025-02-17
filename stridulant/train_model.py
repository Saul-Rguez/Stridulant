# -*- coding: utf-8 -*-
"""
Module for training a Convolutional Neural Network (CNN) to detect stridulations
in spectrogram images. The model is trained using a binary classification approach.

Functions:
    - train_model: Trains the CNN model using images from the specified directory.

Autor: Saul Rodriguez Martinez
Fecha de creación: 2025-02-15

"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_model(train_dir, target_size=(128, 128), batch_size=32, epochs=10, learning_rate=0.001):
    """
    Trains a Convolutional Neural Network (CNN) for binary classification of stridulations 
    in spectrogram images.

    Args:
        train_dir (str): Path to the directory containing the training images. 
        target_size (tuple): The target size to which each input image is resized (default is (128, 128)).
        batch_size (int): The number of images per batch used during training (default is 32).
        epochs (int): The number of epochs to train the model (default is 10).
        learning_rate (float): The learning rate for the optimizer (default is 0.001).

    Returns:
        model (tf.keras.Model): The trained CNN model.
        history (History): The training history object containing loss and accuracy values.
    """
    train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary',
        subset = 'training'
    )

    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size = target_size,
        batch_size = batch_size,
        class_mode = 'binary',
        subset = 'validation'
    )

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer = Adam(learning_rate=learning_rate),
                  loss = 'binary_crossentropy',
                  metrics = ['accuracy'])

    history = model.fit(
        train_generator,
        validation_data = val_generator,
        epochs = epochs
    )

    model.save('stridulation_detection_model.keras')

    return model, history
