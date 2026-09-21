# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 11:31:54 2026
Pipeline for running feature-based scans on datasets, 
assuming a feature-based scan has already been created.
See the Python tutorials on how to create feature-based scans.
This pipeline handles interruptions of file processing by keeping a log. Upon
restart, it will read the log and skip already processed files.

These are the file collection steps:
    1. Check if a previous snippet log exists
    2. Collect previously processed files
    3. Collect all files to be processed
    3. Start a loop over all the files. Skip if has already been processed.
    
Inside the loop, each file will be:
    1. Prepped by normalizing, filtering, augmenting, etc
    2. Cut into snippets.
    
Each snippet will be:
    3. Checked for events using find_events.
    4. Checked for specific event using a feature-based scan.

Results will be entered in the log, which will be saved after the
completion of each file.

If CNN is to be implemented next, the spectrogram outputs of the feature-based scan may be
manually sorted into positives and negatives. A subset should be created to
use as for training data.

"""

import stridulant as st
import matplotlib.pyplot as plt
import pandas as pd
import os


# %%
# If applying a user-defined feature-based scan, make sure it is loaded here.
# For example:

def is_worm_rumble(
    snippet,
    min_event_duration=.3,       # mandatory aurgument needed for find_events()
    env_smooth=25,               # mandatory aurgument needed for find_events() and extract_features()
    threshold_percentile=50,     # mandatory aurgument needed for find_events()
    pulse_dist=1,                # mandatory aurgument needed for extract_features()
    # Feature arguments:
    min_pulse_density=160,
    min_pulse_count=100,
    max_tonal_variation=.3,
    min_spectral_rolloff=14000,
    min_dynamic_range=78
):

    # Detect events
    events = snippet.find_events(
        min_event_duration=min_event_duration,
        threshold_percentile=threshold_percentile,
        env_smooth=env_smooth
    )

    if not events:
        return False
    else:
        for event in events:
            features = snippet.extract_features(event, env_smooth=env_smooth, pulse_dist=pulse_dist)

            if (
                features["pulse_density"] > min_pulse_density
                and features["tonal_variation"] < max_tonal_variation
                and features["spectral_rolloff"] > min_spectral_rolloff
                and features["dynamic_range"] > min_dynamic_range
                and features["pulse_count"] > min_pulse_count
            ):
                return True

        return False


# %%
# --- Find or create log file ---

# Define where the results should be saved
output_folder = r"T:"

# Define which folder(s) to collect wav files from
folder_list = [r"T:\snippets"]

# Define the name of the file where the results will be logged.
log_name = "snippet_log"

# --- Find or create log file ---
# If a snippet log already exist, the names of the audio files that are already processed will be collected.

log_file = os.path.join(output_folder, f"{log_name}.csv")
processed_files = set()

if not os.path.exists(log_file):
    with open(log_file, "w") as f:
        f.write("Original_audio_path,Snippet_start_time, Snippet_end_time,Event,Result\n")
    print("New snippet log created.")

else:
    print(f"Existing log found: {log_file}. Checking which files have been processed already.")
    df = pd.read_csv(log_file)
    processed_files = set(df["Original_audio_path"].unique())

# --- Gather files ending with wav or flac ---

all_audio_files = []
for folder in folder_list:
    for root, dirs, files in os.walk(folder):
        all_audio_files.extend(
            os.path.join(root, f)
            for f in files
            if f.lower().endswith((".wav", ".flac"))
        )

# --- Sort files ---

all_audio_files.sort(key=str.lower, reverse=False)

# %%
# --- Find global max ---
# If audio should be normalized across files, find the global max across files.
#global_max = st.find_global_max(all_audio_files)

# %%
# --- Start loop --- #

# Open log file
log_f = open(log_file, "a")

print(f"Total files processed: {len(processed_files)}")
print(f"Total files remaining: {len(all_audio_files) - len(processed_files)}")

try:

    for file_path in all_audio_files:

        # Skip the file if has already been processed
        if file_path in processed_files:
            continue

        print(f"Processing: {file_path}")

        try:

            # --- Create names based on the naming convetion of the files --- #
            parts = os.path.normpath(file_path).split(os.sep)
            base_name = "_".join(parts[1:3])
            base_name = os.path.splitext(base_name)[0]

            # --- Load audio --- #
            audio, sr = st.load_audio(file_path) 
            
            print("Audio loaded")

            # --- Apply filters and normalization as needed --- #
            # audio = st.highpass_filter(audio, sr)
            # audio = st.lowpass_filter(audio, sr)
            # audio = st.normalize_audio(audio, global_max = global_max) # Global normalization
            # audio = st.normalize_audio(audio) # File-level normalization

            # --- Calculate the number of snippets to be made in the file. Skips if the file is too short. ---
            snippet_duration = 2  # How long the snippets should be
            total_duration = len(audio) / sr
            
            if total_duration < snippet_duration:
                print("Audio file too short (snippet duration does not fit in the length of the audio file).")
                log_f.write(f"{file_path},NA,NA,NA\n")
                continue
            
            num_snippets = int(total_duration // snippet_duration)  # How many times the snippet duration fits in the audio file length.
            print(f"Total snippets to be created: {num_snippets}")

            # Check for events in each snippet
            # If there are events, they will be scanned using the feature-based scan in the next step.

            # --- Loop over the file ---
            for i in range(0, num_snippets):

                start_time = i * snippet_duration
                end_time = start_time + snippet_duration

                # --- Create snippet name ---
                snippet_name = f"{base_name}_{start_time}_"

                # --- Create snippet and find events in snippet, normalize or add other functions if needed ---
                Snippet = st.create_snippet(
                    audio,
                    sr,
                    start_time,
                    snippet_duration
                )

                # --- Preparation on the snippet level ---
                # Snippet.normalize()
                
                # --- Find events ---
                events = Snippet.find_events(min_event_duration=.3, env_smooth=25, threshold_percentile=50)

                # --- Create binary event results: "Yes" if event(s) are found, "No" if snippet is empty ---
                if not events:
                    event_binary = "No"
                    result = "NA"

                    # Save to log
                    log_f.write(
                        f"{file_path},{start_time},{end_time},{event_binary},{result}\n"
                    )

                if events:
                    event_binary = "Yes"

                    # --- Run feature-based scan on snippet ---

                    # Make sure the feature-based scan you want to use is loaded. Here we use the is_worm_rumble defined above.
                    # bool makes sure the result will be returned as either True or False
                    
                    result = bool(is_worm_rumble(Snippet))
                    # or, result = "NA" to not use a feature-based scan.

                    # Save to log
                    log_f.write(
                        f"{file_path},{start_time},{end_time},{event_binary},{result}\n"
                    )
                    
                    if result is True:
                        folder = "pos"
                    elif result is False:
                        folder = "neg"
                    else:
                        folder = "output"
                        
                    # Save spectrogram
                    spectrogram = Snippet.spectrogram()

                    output_path = os.path.join(
                        output_folder,
                        "Results",
                        "Spectrograms",
                        folder
                    )

                    spectrogram.save_img(
                        source_name=snippet_name,
                        output_dir=output_path,
                        with_labels=False
                    )

                    # Save snippet                       
                    output_path = os.path.join(
                        output_folder,
                        "Results",
                        "Snippets",
                        folder
                    )

                    Snippet.save(
                        source_name=snippet_name,
                        output_dir=output_path
                    )

            #  --- Close loop and close log --- 

        except Exception as e:
            print("Error processing file:", file_path, e)

        print(f"Saving data from {file_path}")
        log_f.flush()

finally:
    log_f.close()
    print("All files have been processed. Please see the log for your results.")

# %%
"""
After the feature-based scan is completed, the resulting spectrograms can be used to train a CNN model. 
In this case, feature-based scanning serves as an initial fitering step. 
The results of the feature-based scan can then be refined manually to accurately reflect True and False events. 
A subset of the True and False spectrograms should then be used as training data. 
These should be collected in a folder, with one subfolder containing positives and one subfolder containing negatives. 
When creating spectrograms for CNN, make sure there are no axis labels (set with_labels = False when saving spectrograms).
The classify_spectrograms() function will output a log that can be used for statistcal analysis.
"""

from stridulant import train_model as tm, classify_spectrograms as cs

# Replace with your stridulant folder
train_dir = r"CNN\train"
test_folder = r"CNN\test"

## Train model
mod, hist = tm.train_model(train_dir,
                           target_size=(128, 128), 
                           batch_size=8, 
                           epochs=20, 
                           learning_rate=0.0001, 
                           class_weights = None
                           )
"""
Trains a Convolutional Neural Network (CNN) for binary classification of spectrogram images.

This function loads images from the specified directory, preprocesses them, and trains
a CNN model to classify whether each image represents a stridulation or not. The model
is saved as a .keras file, and the training history (including loss and accuracy) is
stored in a CSV file.

Args:
    train_dir (str): Path to the directory containing the training images. One subfolder should contain positives, 
        and one shoud contain negatives. 
    target_size (tuple): The target size to which each input image is resized (default is (128, 128)).
    batch_size (int): The number of images per batch used during training (default is 32).
    epochs (int): The number of epochs to train the model (default is 5).
    learning_rate (float): The learning rate for the optimizer (default is 0.001).
    class_weights (dict, optional): A dictionary of class weights for handling class imbalance. 
    If None, the class weights are computed based on the class distribution.

Returns:
    model (tf.keras.Model): The trained CNN model.
    history (History): The training history object containing loss and accuracy values.
    
"""

hist = pd.read_csv(os.path.join(train_dir, "training_history.csv"))
plt.figure(figsize=(10, 6))

# 
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

## Run model on the data to be tested
mod = os.path.join(train_dir, "stridulation_detection_model.keras")
cs.classify_spectrograms(test_folder, mod)

