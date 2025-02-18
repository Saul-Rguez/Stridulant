# -*- coding: utf-8 -*-
"""
Module for training a Convolutional Neural Network (CNN) to detect stridulations
in spectrogram images. The model is trained using a binary classification approach.

Functions:
    - train_model: Trains the CNN model using images from the specified directory.

Author: Saul Rodriguez Martinez
Creation date: 2025-02-15

"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
from collections import Counter

def train_model(train_dir, target_size=(128, 128), batch_size=32, epochs=5, learning_rate=0.001, class_weights = None):
    """
    Trains a Convolutional Neural Network (CNN) for binary classification of stridulations 
    in spectrogram images.

    This function loads images from the specified directory, preprocesses them, and trains
    a CNN model to classify whether each image represents a stridulation or not. The model
    is saved as a .keras file, and the training history (including loss and accuracy) is
    stored in a CSV file.

    Args:
        train_dir (str): Path to the directory containing the training images. 
        target_size (tuple): The target size to which each input image is resized (default is (128, 128)).
        batch_size (int): The number of images per batch used during training (default is 32).
        epochs (int): The number of epochs to train the model (default is 5).
        learning_rate (float): The learning rate for the optimizer (default is 0.001).
        class_weights (dict, optional): A dictionary of class weights for handling class imbalance. 
                                         If None, the class weights are computed based on the class distribution.

    Returns:
        model (tf.keras.Model): The trained CNN model.
        history (History): The training history object containing loss and accuracy values.
    
    Example:
        model, history = train_model('path/to/training/images')

    """
    # Define image data generator with rescaling and validation split
    train_datagen = ImageDataGenerator(rescale=1.0/255, validation_split=0.2)

    # Load training images from the directory
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    # Load validation images from the directory
    val_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )
    
    # Calculate class weights if not provided
    if class_weights is None:
        class_counts = dict(Counter(train_generator.classes))  
        total_samples = sum(class_counts.values())
        class_weights = {cls: total_samples / count for cls, count in class_counts.items()}

    # Define CNN architecture
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

    # Compile the model with Adam optimizer and binary crossentropy loss
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model using the training and validation generators
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        class_weight=class_weights
    )

    # Save the trained model and training history
    model.save(os.path.join(train_dir, 'stridulation_detection_model.keras'))
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(os.path.join(train_dir, 'training_history.csv'), index=False)

    return model, history
