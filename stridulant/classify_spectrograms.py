# -*- coding: utf-8 -*-
"""
This script contains functions to load a pre-trained spectrogram classification model,
process spectrogram images, and classify them as stridulation or non-stridulation.
The main functions include:
- Load and prepare the trained model.
- Preprocess spectrogram images.
- Classify the images and move them to the appropriate folders.
- Create output directories if they do not exist.

Images are classified using a threshold of 0.5, where values greater than or equal to 0.5 are 
considered stridulations, and the rest are classified as non-stridulations.

Author: Saul Rodriguez Martinez
Creation date: 2025-02-15

"""

import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import shutil


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
        img = image.load_img(file_path, target_size=target_size)
        img_array = image.img_to_array(img)  
        img_array = np.expand_dims(img_array, axis=0) 
        img_array /= 255.0 
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

    files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for file in files:
        file_path = os.path.join(input_dir, file)
        

        img_array = preprocess_image(file_path)

        prediction = model.predict(img_array) 
        

        if prediction >= 0.5:
            shutil.move(file_path, os.path.join(output_dir_stridulations, file))
            print(f"File {file} classified as stridulation.")
        else:
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
    model = load_and_prepare_model(model_path)

    create_directories(output_dir_stridulations, output_dir_non_stridulations)

    classify_and_move_spectrograms(input_dir, model, output_dir_stridulations, output_dir_non_stridulations)
