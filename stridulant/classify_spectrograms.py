# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 22:04:03 2025

@author: Saul
"""

import os
import shutil
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

def load_and_prepare_model(model_path: str):
    """
    Load the pre-trained model from the specified file path.
    
    Args:
    - model_path (str): Path to the pre-trained model file.

    Returns:
    - model (tf.keras.Model): The loaded TensorFlow model.
    """
    try:
        model = load_model(model_path)
        print(f"Model loaded successfully from: {model_path}")
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        raise

def preprocess_image(file_path: str, target_size=(128, 128)):
    """
    Preprocesses the image to match the model's expected input format.
    
    Args:
    - file_path (str): Path to the image file (spectrogram).
    - target_size (tuple): Dimensions to resize the image to.

    Returns:
    - img_array (np.ndarray): Preprocessed image ready for prediction.
    """
    try:
        # Load and resize the image
        img = image.load_img(file_path, target_size=target_size)
        img_array = image.img_to_array(img)  # Convert the image to a Numpy array
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image (assuming the model was trained with normalized images)
        return img_array
    except Exception as e:
        print(f"Error preprocessing the image {file_path}: {e}")
        raise

def classify_and_move_spectrograms(input_dir: str, model, output_dir_stridulations: str, output_dir_non_stridulations: str):
    """
    Classifies the spectrograms into two categories (stridulation and non-stridulation) 
    and moves the files to their respective folders.
    
    Args:
    - input_dir (str): Directory containing the spectrograms to classify.
    - model (tf.keras.Model): The loaded model for classification.
    - output_dir_stridulations (str): Directory where stridulation spectrograms will be moved.
    - output_dir_non_stridulations (str): Directory where non-stridulation spectrograms will be moved.
    """
    # Get all spectrogram files in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]  # Adjust as needed
    
    for file in files:
        file_path = os.path.join(input_dir, file)
        
        # Preprocess the image
        img_array = preprocess_image(file_path)
        
        # Make the prediction
        prediction = model.predict(img_array)  # The output of the model will be 0 or 1 (stridulation or non-stridulation)
        
        # Check the prediction and move the file
        if prediction >= 0.5:  # Classification threshold (adjustable)
            # If prediction is greater than 0.5, move to the stridulation folder
            shutil.move(file_path, os.path.join(output_dir_stridulations, file))
            print(f"File {file} classified as stridulation.")
        else:
            # If not, move to the non-stridulation folder
            shutil.move(file_path, os.path.join(output_dir_non_stridulations, file))
            print(f"File {file} classified as non-stridulation.")

def create_directories(output_dir_stridulations: str, output_dir_non_stridulations: str):
    """
    Creates the necessary folders to store the classified spectrograms if they don't exist.

    Args:
    - output_dir_stridulations (str): Directory for stridulation spectrograms.
    - output_dir_non_stridulations (str): Directory for non-stridulation spectrograms.
    """
    os.makedirs(output_dir_stridulations, exist_ok=True)
    os.makedirs(output_dir_non_stridulations, exist_ok=True)
    print("Output directories created if they didn't exist.")

def classify_spectrograms(input_dir: str, model_path: str, output_dir_stridulations: str, output_dir_non_stridulations: str):
    """
    Main function to load the model, create output directories, and classify the spectrograms.

    Args:
    - input_dir (str): Directory containing the spectrograms to classify.
    - model_path (str): Path to the pre-trained model.
    - output_dir_stridulations (str): Directory for stridulation spectrograms.
    - output_dir_non_stridulations (str): Directory for non-stridulation spectrograms.
    """
    # Load the pre-trained model
    model = load_and_prepare_model(model_path)
    
    # Create output directories
    create_directories(output_dir_stridulations, output_dir_non_stridulations)
    
    # Classify and move the spectrograms
    classify_and_move_spectrograms(input_dir, model, output_dir_stridulations, output_dir_non_stridulations)
