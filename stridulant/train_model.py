# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 21:45:33 2025

@author: Saul
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def train_model(train_dir, target_size=(128, 128), batch_size=32, epochs=10, learning_rate=0.001):
    # Set up image data generators for data augmentation and normalization
    train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)  # 20% for validation

    # Load training data with validation split
    train_generator = train_datagen.flow_from_directory(
        train_dir,                                          # images folder
        target_size=target_size,                             # Resize to match input shape
        batch_size=batch_size,                              # Size of the training batch
        class_mode='binary',                                # 2 outputs, stridulation or not
        subset='training'                                   # For training
    )

    # Load validation data using validation split
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'                                  # For validation
    )

    # Define the CNN model
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
        Dense(1, activation='sigmoid')  # Binary classification
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs
    )

    # Save the model
    model.save('stridulation_detection_model.keras')

    return model, history