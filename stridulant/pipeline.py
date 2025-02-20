# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 10:30:39 2025
Pipeline for the AI.
Author: Saul Rodriguez Martinez
Creation date: 2025-02-17

This script processes audio files, prepares data for training, performs data augmentation, 
and trains a machine learning model to classify stridulations and non-stridulations.
The steps are as follows:
1. Process audio files in a specified directory.
2. Annotate data based on annotation files and organize it into positives and negatives.
3. Perform data augmentation on positive samples to balance the dataset.
4. Train a model to classify spectrograms using the processed data.
5. Plot training loss and accuracy metrics.
6. Classify new spectrograms with the trained model.

"""

# Import necessary modules
import stridulant as st  # The custom package for processing and training
import os  # For file and directory operations
import shutil  # For copying files
import pandas as pd  # For handling CSV data
import matplotlib.pyplot as plt  # For plotting training results

# Set the path to the directory containing the audio files to be processed
path = 'F:/estridulaciones/anteater data/tagged_files'

# %%
# Process all audio files in the specified folder
# This step processes each audio file in the folder and generates necessary outputs.
for file in os.listdir(path):
    if file.endswith('.wav'):
        st.process_audio_file(os.path.join(path, file))

# %%
# Capture subdirectories created by the 'process_audio_file' function
# This is necessary to organize files into subfolders for further processing.
subdirectories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

# Now, we will divide the files into positives and negatives using the annotation files
# This block uses the 'stridulant' structure and naming conventions, so no need to modify it
# It also merges the data into 'Merged_audio' and 'Merged_spectrograms' for model training
for file in os.listdir(path):
    if file.endswith('.txt'):
        notes = os.path.join(path, file)
        for directory in subdirectories:
            if file[0:-4] in directory:
                snip = os.path.join(directory, "Audio_snippets")
                spec = os.path.join(directory, "Spectrograms")
                st.annotate_data(notes, snip, spec, snippet_duration=2.0, csv_delim='\t')


            # Define paths for merged audio and spectrograms
            merged_audio_pos_path = os.path.join(path, "Merged_audio/Merged_positives")
            merged_audio_neg_path = os.path.join(path, "Merged_audio/Merged_negatives")
            merged_spec_pos_path = os.path.join(path, "Merged_spectrograms/Merged_spectrogram_positives")
            merged_spec_neg_path = os.path.join(path, "Merged_spectrograms/Merged_spectrogram_negatives")
            
            #create the paths in case they didn't exist
            os.makedirs(merged_audio_pos_path, exist_ok=True)
            os.makedirs(merged_audio_neg_path, exist_ok=True)
            os.makedirs(merged_spec_pos_path, exist_ok=True)
            os.makedirs(merged_spec_neg_path, exist_ok=True)

            # Copy positive and negative audio files to the merged directories
            for file_name in os.listdir(os.path.join(snip, "positives")):
                source_path = os.path.join(snip, "positives", file_name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, merged_audio_pos_path)

            for file_name in os.listdir(os.path.join(snip, "negatives")):
                source_path = os.path.join(snip, "negatives", file_name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, merged_audio_neg_path)

            # Copy positive and negative spectrogram files to the merged directories
            for file_name in os.listdir(os.path.join(spec, "positives")):
                source_path = os.path.join(spec, "positives", file_name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, merged_spec_pos_path)

            for file_name in os.listdir(os.path.join(spec, "negatives")):
                source_path = os.path.join(spec, "negatives", file_name)
                if os.path.isfile(source_path):
                    shutil.copy2(source_path, merged_spec_neg_path)

# %%
# Handle data imbalance by augmenting positive samples
# This part creates variations of the stridulation samples and adds them to the positives folder
for file in os.listdir(merged_audio_pos_path):
    snippet = st.load_snippet(os.path.join(merged_audio_pos_path, file))
    st.create_variations(snippet, os.path.join(merged_audio_pos_path, 'synthetic'), num_variations=20)

# Create spectrograms for the augmented snippets and save them
for file in os.listdir(os.path.join(merged_audio_pos_path, 'synthetic')):
    snippet = st.load_snippet(os.path.join(merged_audio_pos_path, 'synthetic', file))
    spec = snippet.spectrogram('mel')
    spec.save(merged_spec_pos_path)

# %%
# Training the model
# The function 'train_model' is used to train a model on the merged spectrograms.
# The class weights are adjusted automatically based on the data imbalance.
# this adjustment is potentially too strict, at least in cases of extreme 
# I will have to keep testing and adjusting that, I recommend keeping the default
# auto-weight function, but if you see a lot of overfitting, you may try to define
# weights in this function. It is a kwarg that should look like this:
# {0:1,1:50}, meaning {class:weight, another_class:another_weight}. 
# the second class should be the positives and is the one you wat to give weight
# because it is a minority of the examples.
mod, hist = st.train_model('F:/estridulaciones/anteater data/tagged_files/Merged_spectrograms', epochs=10)

# %%
# Plot training history (loss and accuracy)
# This code loads the training history from the CSV file and plots the loss and accuracy curves.
hist = pd.read_csv(os.path.join(path, 'Merged_spectrograms/training_history.csv'))
plt.figure(figsize=(10, 6))

# Plot training and validation loss
plt.plot(hist['loss'], label='Training loss')
plt.plot(hist['val_loss'], label='Validation loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(hist['accuracy'], label='Training accuracy')
plt.plot(hist['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# %%
# Classify new spectrograms with the trained model
# After training, the model is used to classify new spectrograms in the 'test' directory.
spec_path = os.path.join(path, 'test')
mod = 'F:/estridulaciones/anteater data/tagged_files/Merged_spectrograms/stridulation_detection_model.keras'
st.classify_spectrograms(spec_path, mod)
