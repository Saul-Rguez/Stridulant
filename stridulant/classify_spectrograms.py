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
    
    This function loads a TensorFlow model from the provided file path and prepares it for predictions.
    
    Args:
    - model_path (str): Path to the pre-trained model file.

    Returns:
    - model (tf.keras.Model): The loaded TensorFlow model.

    Raises:
    - Exception: If the model cannot be loaded from the given path.
    
    Example:
        model = load_and_prepare_model('path/to/model.keras')
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
    
    This function loads an image from the specified file path, resizes it to the target size, 
    normalizes its pixel values, and prepares it for classification by the model.

    Args:
    - file_path (str): Path to the image file (spectrogram).
    - target_size (tuple): Dimensions to resize the image to (default is (128, 128)).

    Returns:
    - img_array (np.ndarray): Preprocessed image ready for prediction.
    
    Example:
        img_array = preprocess_image('path/to/image.png')
    """
    try:
        img = image.load_img(file_path, target_size=target_size)
        img_array = image.img_to_array(img)  
        img_array = np.expand_dims(img_array, axis=0) 
        img_array /= 255.0  # Normalize image values to [0, 1]
        return img_array
    except Exception as e:
        print(f"Error preprocessing the image {file_path}: {e}")
        raise

def classify_and_move_spectrograms(input_dir: str, model, output_dir_pos: str, output_dir_neg: str):
    """
    Classifies the spectrograms into two categories (stridulation and non-stridulation) 
    and moves the files to their respective folders.
    
    This function processes each image in the input directory, classifies it using the model,
    and moves it to the appropriate directory (positive for stridulation, negative for non-stridulation).
    
    Args:
    - input_dir (str): Directory containing the spectrograms to classify.
    - model (tf.keras.Model): The loaded model for classification.
    - output_dir_pos (str): Directory where stridulation spectrograms will be moved.
    - output_dir_neg (str): Directory where non-stridulation spectrograms will be moved.

    Example:
        classify_and_move_spectrograms('path/to/input', model, 'path/to/positive', 'path/to/negative')
    """
    files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    for file in files:
        file_path = os.path.join(input_dir, file)
        
        # Preprocess the image for prediction
        img_array = preprocess_image(file_path)

        # Predict the class of the image (stridulation or non-stridulation)
        prediction = model.predict(img_array) 

        # Move the file to the corresponding directory based on prediction
        if prediction >= 0.5:
            shutil.move(file_path, os.path.join(output_dir_pos, file))
            print(f"File {file} classified as positive (stridulation).")
        else:
            shutil.move(file_path, os.path.join(output_dir_neg, file))
            print(f"File {file} classified as negative (non-stridulation).")

def classify_spectrograms(input_dir: str, model_path: str):
    """
    Main function to load the model, create output directories, and classify the spectrograms.

    This function orchestrates the classification process by first loading the pre-trained model,
    creating output directories for positive and negative classifications, and then classifying 
    and moving the spectrogram files into their corresponding folders.

    Args:
    - input_dir (str): Directory containing the spectrograms to classify.
    - model_path (str): Path to the pre-trained model.
    - output_dir_pos (str): Directory for stridulation spectrograms.
    - output_dir_neg (str): Directory for non-stridulation spectrograms.

    Example:
        classify_spectrograms('path/to/input', 'path/to/model.keras')
    """
    # Load the pre-trained model
    model = load_and_prepare_model(model_path)
    
    # Create output directories for positive and negative classes
    output_dir_pos = f"{input_dir}/positive"
    os.makedirs(output_dir_pos, exist_ok=True)
    output_dir_neg = f"{input_dir}/negative"
    os.makedirs(output_dir_neg, exist_ok=True)

    # Classify the spectrograms and move them to the appropriate directories
    classify_and_move_spectrograms(input_dir, model, output_dir_pos, output_dir_neg)
