# Stridulant

> ** Bioacoustic toolkit for analyzing novel sounds **

Stridulant is a comprehensive Python package for bioacoustic analysis, pre-loaded with functions to scan for stridulation-like insect sounds and ultrasonic acoustic emissions from plants. 

## Core functions

- **Event detection.** _Stridulant_ can automate the detection of acoustic events, eliminating empty audio by filtering for regions of sustained energy.
- **Feature extraction.** _Stridulant_ extracts a comprehensive set of acoustic features that can be used when reporting a description of newly studied or discovered sounds or to build feature-based scans. Currently, the features focus on non-vocalized sounds, including several pulse characteristics. These features may be expanded by users themselves or by getting in touch with the developers.
- **Feature-based scanning.** Acoustic features may be used to build feature-based scans, which can be used to loop through audio and filter acoustic events matching the target acoustic events. This may serve as a pre-filtering step for more complex (deep learning) models, but can be used as a stand-alone method if the target sounds have consistent feature values. _Stridulant_ comes pre-loaded with feature-based scans for insect stridulations and ultrasonic acoustic emissions from plants.
- **CNN training.** While not the main focus of _Stridulant_, the package includes a simple binary spectrogram classifier.
- **Audio processing tools.**: To prepare files for analysis, _Stridulant_ offer spectrogram generation, normalization and high- and low-pass filtering.

Use the pipeline to apply feature-based scans over whole folders or drives of audio.

See the manual for function usage and the accompanying publication for more context and case studies. Tutorials follow below. 

## Installation
Navigate to your _Stridulant_ folder and run:

```bash
pip install stridulant
```

### Dependencies

Stridulant requires:
- Python >= 3.7
- TensorFlow (optional, only needed for ML features)

### 0. Basic functionality

```python
"""
Created on Thu Feb 19 09:42:29 2026

"""
import stridulant as st
import os

# Replace with your stridulant folder #
os.chdir(r"C:\Users\tavn0004\stridulant\src\stridulant\Tutorials")
output_folder = r"C:\Users\tavn0004\stridulant\src\stridulant\Tutorials\Output"

"""
This tutorial assumes you have succesfully installed Stridulant and are now ready to use it.
Here you will learn how to work with single audio files and snippets. As a starting point,
we recommend you find one event of interest in your own audio, or use our example audio 
and work through each step.

"""
# %%
###########################################################################
## Snippets ##
###########################################################################

# Load an audio file #
audio, sr = st.load_audio(r"Worm_audios\A20_251010_007_Tr1_2.flac")

# Create a snippet #
Snippet_1 = st.create_snippet(audio, sr, 1452, 3)

# Play snippet #
Snippet_1.play()

# Save snippet #
Snippet_1.save("Test_1", output_dir = output_folder, verbose = True, metadata = True)

# %%
###########################################################################
## Find and extract events ##
###########################################################################

events = Snippet_1.find_events(min_event_duration = .3, env_smooth= 25, threshold_percentile= 50)
features = Snippet_1.extract_features(events[0], pulse_dist=1)
st.save_features(features, os.path.join(output_folder, "save_features.csv"))

# %%
###########################################################################
## Spectrograms  ##
###########################################################################

# Mel #
spectrogram_1 = Snippet_1.spectrogram(spec_type = "mel", n_mels = 30, hop_length = 10, window = "boxcar", env_smooth = 10)

# Hilbert with detected events #
events = Snippet_1.find_events(min_event_duration = .3, env_smooth= 25, threshold_percentile= 50)
spectrogram_2 = Snippet_1.spectrogram(spec_type = "hilbert", env_smooth = 10)

# Plot spectrogram #
spectrogram_1.plot()
spectrogram_2.plot()
spectrogram_2.plot_events(events = events)

# Save spectrogram object #
spectrogram_1.save_img("test", output_dir = output_folder, with_labels = True)
spectrogram_2.save_img("test", output_dir = output_folder, with_labels = True, events = events)

# %%
###########################################################################
## Normalize ##
###########################################################################

Snippet_1_normalize = st.create_snippet(audio, sr, 1452, 3)

# Normalize a single snippet #
Snippet_1_normalize.normalize()

# Normalize within one audio file #
audio_normalize = st.normalize_audio(audio = audio, target_max = 0.75)
#
# Global normalization (no example files) #
# st.normalize_global("your folder here", output_dir = "your folder here", target_max = 0.75)

# %%
###########################################################################
## Filtering ##
###########################################################################

Snippet_1_filtered = st.create_snippet(audio, sr, 1452, 3)

# Highpass filter #
Snippet_1_filtered.audio = st.highpass_filter(Snippet_1.audio, sr, cutoff = 3000, order = 8)

# Lowpass filter #
Snippet_1_filtered.audio = st.lowpass_filter(Snippet_1.audio, sr, cutoff = 20000, order = 8)
```

## Acoustic features details

The `extract_features()` method returns a dictionary with:

| Feature                  | Description                                                          | Unit                                           |
| ------------------------ | -------------------------------------------------------------------- | ---------------------------------------------- |
| `event_duration`         | Duration of the event                                                | Seconds                                        |
| `event_energy`           | Peak amplitude of the smoothed envelope                              | Typically unitless, unless audio is calibrated |
| `attack_rate`            | Rate at which amplitude rises in the first 20% of the event envelope | Amplitude units per sample                     |
| `duty_cycle`             | Fraction of event duration above an envelope threshold               | Fraction (0–1)                                 |
| `dynamic_range`          | Difference between the loudest and quietest spectral amplitudes      | dB                                             |
| `avg_pulse_duration`     | Mean pulse duration                                                  | Seconds                                        |
| `avg_pulse_interval`     | Mean time between pulses                                             | Seconds                                        |
| `pulse_count`            | Number of pulses                                                     | Count per events                               |
| `pulse_density`          | Pulses per second                                                    | Count per second                               |
| `pulse_regularity`       | Rhythm consistency                                                   | Standard deviation of intervals between pulses |
| `frequency_range`        | Bandwidth containing 90% of spectral energy                          | Hz                                             |
| `spectral_centroid_mean` | Dominant frequency                                                   | Hz                                             |
| `spectral_roll_off`      | Frequency at which 85% of energy falls below                         | Hz                                             |
| `tonal_variation`        | Frequency stability                                                  | Standard deviation of the mean of centroid     |

In `extract_features()`, if a number of mfcc dimensions is defined, the following statistics will be calculated for each dimension.

| Feature        | Description                                                   | Unit                           |
| -------------- | ------------------------------------------------------------- | ------------------------------ |
| `min.cc`       | Minimum value of the coefficient across frames               | Coefficient units              |
| `max.cc`       | Maximum value of the coefficient across frames               | Coefficient units              |
| `median.cc`    | Median value of the coefficient across frames                | Coefficient units              |
| `mean.cc`      | Mean value of the coefficient across frames                  | Coefficient units              |
| `var.cc`       | Variance of the coefficient across frames                    | Coefficient units²             |
| `skew.cc`      | Skewness (asymmetry) of the coefficient distribution         | Dimensionless                   |
| `kurt.cc`      | Kurtosis (tailedness/peakedness) of the coefficient distribution | Dimensionless               |
| `mean.d1.cc`   | Mean of the first derivative (velocity) of the coefficient   | Coefficient units/frame        |
| `var.d1.cc`    | Variance of the first derivative (velocity) of the coefficient | Coefficient units²/frame²    |
| `mean.d2.cc`   | Mean of the second derivative (acceleration) of the coefficient | Coefficient units/frame²     |
| `var.d2.cc`    | Variance of the second derivative (acceleration) of the coefficient | Coefficient units²/frame⁴ |

### 1. Prepare snippet sets

```python
import stridulant as st
import os

# Replace with your Tutorial folder #
os.chdir(r"C:\Users\tavn0004\stridulant\src\stridulant\Tutorials")
output_folder = r"C:\Users\tavn0004\stridulant\src\stridulant\Tutorials\Output"

"""
This tutorial will demonstrate how to create snippet sets from audio files. It is
also possible to import annotation tables, made in for example Raven.

"""

# %%
##########################################################################
## Making snippets and/or spectrograms from annotation tables ##
###########################################################################

"""
The example text files contain start and end times of positive (i.e., containing a target sound) 
and negative (i.e., containing a non-target sound) start and end times. 

Defining a snippet duration ensures all the output snippets are the same length,
even if the annotated sections are not. It does this by centering the snippet between the
specified start and end times. Having snippets of the same duration can make event 
detection in later steps more reliable.

Not defining a snippet length will simply use the start and end times in the table as
start and end times of the snippets.

"""

# Add kwargs for spectrogram() as needed. 
st.process_table(input_table = "raven_A20_251010_007_Tr1_2.Table.1.selections.txt",
              output_dir = output_folder,
              delim = "\t",
              audio = "Worm_audios\A20_251010_007_Tr1_2.flac",
              snippet_duration=3,
              ids = ["Selection", "Annotation"],
              spec_type = "hilbert"
              )     

st.process_table(input_table = "raven_C20_251016_005_Tr1_1.Table.1.selections.txt",
              output_dir = output_folder,
              delim = "\t",
              audio = "Worm_audios\C20_251016_005_Tr1_1.flac",
              snippet_duration=3,
              ids = ["Selection", "Annotation"],
              spec_type = "hilbert"
              )     

## Add kwargs for extract_features() as needed ##
st.feature_extractor(
          output_folder = output_folder,
          snippet_folder = "Output\Annotated_A20_251010_007_Tr1_2\Snippets",
          log_name="features_A20.csv",
          n_mfcc = 1)

st.feature_extractor(
          output_folder = output_folder,
          snippet_folder = "Output\Annotated_C20_251016_005_Tr1_1\Snippets",
          log_name="features_C20.csv",
          n_mfcc = 1)

# %%
##########################################################################
## Processing full audio files into snippets ##
###########################################################################

## Add kwargs for spectrogram generation as needed ##
## Cuts a whole audio file into snippets ##
st.process_audio_file(r"Worm_audios\A20_251010_007_Tr1_2.flac", output_folder = output_folder, spec_type = "hilbert")

## Subset event snippets ##
## Loops through all snippets and selects those with acoustic events ##
st.event_extractor(snippet_folder = "Output\A20_251010_007_Tr1_2\Audio_snippets",
                    spec_type = "hilbert",
                    metadata = False)

## The output can be manually sorted into Positive (containing a target sound) and Negative (containing a non-target sound) folders. ##
## Then, features can be extracted from both. ##
#st.feature_extractor(snippet_folder = "Folder_with_positives", log_name = "Positive_events_features.csv")
#st.feature_extractor(snippet_folder = "Folder_with_negatives", log_name = "Negative_event_features.csv")

#%%

## The resulting acoutic features dataframes can be used to explore differences in sounds. ##
## Here follows an example using seaborn. ##

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_A20 = pd.read_csv(os.path.join(output_folder, "features_A20.csv"))
data_C20 = pd.read_csv(os.path.join(output_folder, "features_C20.csv"))
data = pd.concat([data_A20, data_C20], ignore_index=True)

data["Result"] = data.iloc[:, 0].str.contains("Negative", na=False)\
                               .map({True: "Negative", False: "Positive"})

feature_columns = data.columns.drop([data.columns[0], "Result"])

data.shape[1]
n_cols = 7
n_rows = 4
plt.figure(figsize=(8*n_cols,6 *n_rows))

for i, col in enumerate(feature_columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x='Result', y=col, data=data)
    plt.title(f'{col}')
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()
```

### 2. Feature-based scans

```python
"""
Created on Thu Feb 19 12:05:23 2026

@author: Tamara van Steijn
"""
import stridulant as st
import os

# Replace with your Tutorial folder #
os.chdir(r"C:\Users\tavn0004\stridulant\src\stridulant\Tutorials")
folder = r"C:\Users\tavn0004\stridulant\src\stridulant\Tutorials"

# %%
"""
In this tutorial we will demonstrate how to detect stridulations and UAE on single snippets.
We will further demonstrate how to build your own function using worm sounds as an example.
Once the settings are confirmed to detect events in known positive snippet, the feature-based
scan can be applied in the pipeline.
 
"""
# %%
# Load an audio file on which to detect stridulations #
audio, sr = st.load_audio(r"Ant_stridulation.FLAC")

# Create a snippet #
Snippet_1 = st.create_snippet(audio, sr, 758, 3)
Snippet_1.play()

# Detect stridulation event #
Snippet_1.is_stridulation(min_event_duration=0.2, 
                threshold_percentile=30, 
                min_pulses=3, 
                min_regularity=10,
                uncoupled_min_duration=0.5, 
                max_duration=1, 
                sustain=0.5, 
                pulse_dist=20, 
                sp_range=(5500, 15000), 
                env_smooth = 10, 
                enable_coupled=True, 
                coupled_min_duration=0.22, 
                coupled_gap=0.6)

# %%
"""
is_stridulation evaluates if the audio snippet contains a promising stridulation signal based on 
acoustic features. Analyzes all detected events and returns features if ANY event 
meets all the stridulation criteria.

This function implements a two-stage detection strategy:
1. First looks for strong individual events that meet all criteria including normal duration
2. If no strong events found, looks for pairs of weak consecutive events that 
   together form a valid stridulation pattern, using relaxed duration criteria

Args:
    min_pulses (int): Minimum number of individual pulses required within the event.
                      Typical insect stridulations have 6+ distinct pulses.
                      
    min_regularity (float): Minimum pulse regularity score (1/standard_deviation of intervals).
                           Higher values indicate more consistent timing between pulses.
                           Values >10 suggest rhythmic, organized patterns.
                           
    min_duration (float): Minimum event duration in seconds for strong individual events.
                         Use 0 for no minimum.
                         
    max_duration (float/None): Maximum event duration in seconds. Use None for no maximum.
                              
    sustain (float): Minimum duty cycle ratio (0-1) indicating what fraction of the 
                    event duration contains actual sound.
                    
    pulse_dist (float): Minimum distance between pulses in milliseconds.
                        
    sp_range (tuple): Valid frequency range for spectral centroid in Hz (min, max).
    
    env_smooth (float): smooth factor for the Hilbert emvelope. It is a number in miliseconds that smooths 
    the peaks on that time range. Often numbers around 10 or so give a good trade-off betwee the smoothness
    of the line and the retention of features. Smaller numbers will give higher details, bigger numbers 
    smoother lines. These numbers operate with the sampling rate, for ultrasounds you probably want to go 
    smaller, like 1 or even 0.1. Just make sure that int(env_smooth/1000*sr)>0. You can check the sr of your
    audio when you load it, you'll get sr that you can print, or within the snippet, with snippet.sr
                     
    enable_coupled (bool): Whether to enable detection of coupled weak events.
                          
    coupled_min_duration (float): Minimum duration for events in coupled detection.
                                 Allows shorter events to be considered only when
                                 looking for coupled pairs.
                                 
    coupled_gap (float): Maximum time gap between consecutive weak events in seconds.

Returns:
    dict/False: Features dictionary if stridulation found, False otherwise.
"""
# %%
# Load an audio file on which to detect UAEs #
audio, sr = st.load_audio(r"Plant_sound.WAV") 

# Create a snippet #
Snippet_1 = st.create_snippet(audio, sr, 0, .01)
Snippet_1.audio = st.highpass_filter(Snippet_1.audio, sr, cutoff = 3000, order = 8)
spectrogram_1 = Snippet_1.spectrogram(spec_type = "fft", n_fft = 128, hop_length = 4, window = "hann", env_smooth = 1)
spectrogram_1.plot()

# Detect cavitation event #
Snippet_1.is_uae(spectral_rolloff_min = 23000, 
                       min_energy = 0.0015, 
                       min_event_duration=0.00005, 
                       max_event_duration = 0.002,
                       max_energy = 0.01, 
                       pulse_dist = 1, 
                       env_smooth = .1, 
                       threshold_percentile = 98)
# %%
"""
Evaluates if the audio snippet contains a promising cavitation signal based on 
acoustic features. Analyzes the most energetic event and return True if its classified as UAE.
         
This function is mostly based on what is known from tree UAE from the literature:
    a) UAE are short. Usually a few ms long.
    b) They have at least part of their energy in the ultrasonic range.
    c) They are likely to be quite faint, at least not perceivable by ear.
    
To use the function:
    a) you need relatively clean audio.
    b) audio cannot be normalized (if normalized the absolute energy becomes relative)

Args:
    spectral_rolloff_min: Minimum frequency at which at 85% of energy falls below.
            
    min_energy: Minimum event energy
                         
    min_event_duration (float): Minimun time for a sound to be considered an event.      
            
    max_event_duration (float): Minimun time for a sound to be considered an event.                         
                                 
    threshold_percentile (float): percentile for the events finding function. It is the percentile with 
    respect to the snippet.
            
    env_smooth (float): smooth factor for the Hilbert emvelope. It is a number in miliseconds that smooths 
    the peaks on that time range. Often numbers around 10 or so give a good trade-off betwee the smoothness
    of the line and the retention of features. Smaller numbers will give higher details, bigger numbers 
    smoother lines. These numbers operate with the sampling rate, for ultrasounds you probably want to go 
    smaller, like 1 or even 0.1. Just make sure that int(env_smooth/1000*sr)>0. You can check the sr of your
    audio when you load it, you'll get sr that you can print, or within the snippet, with snippet.sr
            
    pulse_dist (float): Minimum distance between pulses in milliseconds. This is legacy for the stridulations. 
    For cavitation you should keep it at 1 in principle.   
                      
Returns:
    dict/False: Features dictionary if stridulation found, False otherwise.
"""   

#######################################################
# %%
"""
EXAMPLE OF USER-DEFINED FUNCTION 

Note: Take care that throughout the process of developing your function, 
you use the same snippet_duration, env_smooh, threshold_percentile and pulse_dist values, 
as differences in these parameters can lead to different outcomes. 
"""
def is_worm_rumble(snippet, 
                      min_event_duration = .3,   # mandatory aurgument, choose your own values     
                      env_smooth = 25,           # mandatory aurgument, choose your own values            
                      threshold_percentile = 50, # mandatory aurgument, choose your own values    
                      pulse_dist = 1, # mandatory aurgument, choose your own values    
                      # From here, you can add the features that you think makes the sounds stand out
                      min_pulse_density = 160,
                      min_pulse_count = 100,
                      max_tonal_variation = .3,
                      min_spectral_rolloff = 14000, 
                      min_dynamic_range = 78
                      # Choose any name and value 
                      ):

        # Detect events
        events = snippet.find_events(
            min_event_duration,
            threshold_percentile,
            env_smooth
        )

        if not events:
            return False
        else:
            for event in events:
                features = snippet.extract_features(event, pulse_dist=pulse_dist)

                if (
                        # Define the arguments. After "features" should come a features name corresponding to the extract_features output (see below to create such output).
                        features["pulse_density"] > min_pulse_density 
                        and features["tonal_variation"] < max_tonal_variation
                        and features["spectral_rolloff"] > min_spectral_rolloff
                        and features["dynamic_range"] > min_dynamic_range
                        and features["pulse_count"] > min_pulse_count
                ):
                        return True

            return False

"""
Using feature_extractor() (see the tutorial Prepare snippet sets) we were able to establish the following differences
between the target and non-target sounds.
    - The event duration was most often higher than 0.2 seconds.
    - The minimum pulse count was around 15 (excluding some outliers).
    - The spectral roll-off was high.
"""

audio, sr = st.load_audio(r"Worm_audios\A20_251010_007_Tr1_2.flac")
target_sound_snippet = st.create_snippet(audio, sr, 1313, 3) 
events = target_sound_snippet.find_events(min_event_duration=0.5, threshold_percentile=25, env_smooth=25)
target_sound_snippet.extract_features(events[0])
target_sound_snippet.play()
is_worm_rumble(target_sound_snippet) # Target sound returns True

non_target_sound_snippet = st.create_snippet(audio, sr, 1303, 3) 
events = non_target_sound_snippet.find_events(min_event_duration=0.5, threshold_percentile=25, env_smooth=25)
non_target_sound_snippet.extract_features(events[0])
non_target_sound_snippet.play()
is_worm_rumble(non_target_sound_snippet) # Non-target sound returns False
```

### 3. CNN training and use

```python
"""
Created on Tue Feb 24 10:13:43 2026

"""

from stridulant import train_model as tm, classify_spectrograms as cs
import matplotlib.pyplot as plt
import pandas as pd
import os

# Replace with your stridulant folder
train_folder = r"C:\Users\tavn0004\stridulant\src\stridulant\Tutorials\CNN\train"
test_folder = r"C:\Users\tavn0004\stridulant\src\stridulant\Tutorials\CNN\test"

"""
In this tutorial we will use spectrograms of sounds from recordings of drying plants to train an CNN.
The training data consists of manually selected positive and negative spectrograms of 0.01 seconds in length.
When creating spectrograms for CNN training, make sure there are no axis labels (set with_labels = False).

"""

## Train model
mod, hist = tm.train_model(train_folder, 
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
    
"""

hist = pd.read_csv(os.path.join(train_folder, "training_history.csv"))
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

## Run model
mod = os.path.join(train_folder, "stridulation_detection_model.keras")
cs.classify_spectrograms(test_folder, mod)
```

## API Reference

### AudioSnippet Class

| Method | Description |
|--------|-------------|
| `save(source_name, output_dir)` | Save snippet as WAV |
| `play()` | Play audio (Esc to stop) |
| `spectrogram(spec_type, ...)` | Generate spectrogram |
| `normalize()` | Peak normalization (75% by default) |
| `find_events(...)` | Detect acoustic events |
| `extract_features(event, ...)` | Extract 13+ acoustic features |
| `is_stridulation(...)` | Classify as stridulation |
| `is_cavitation(...)` | Classify as cavitation |
| `add_gaussian_noise(...)` | Augmentation |
| `time_stretch(...)` | Augmentation |
| `pitch_shift(...)` | Augmentation |
| `shift(...)` | Augmentation |
| `clipping_distortion(...)` | Augmentation |
| `add_background_noise(...)` | Augmentation |
| `time_mask(...)` | Augmentation |

### Spectrogram Class

| Method | Description |
|--------|-------------|
| `plot(color)` | Display spectrogram |
| `plot_events(events)` | Plot Hilbert envelope with marked events |
| `save_img(output_path, with_labels, color, events)` | Save as PNG |
| `save_table(source_name, output_dir)` | Export as CSV |
| `compute_power_metrics()` | Calculate APD and PPD (dB) |

### Processing Functions

| Function | Description |
|----------|-------------|
| `process_audio_file(...)` | Full pipeline: split → spectrograms → save |
| `annotate_data(...)` | Separate positives/negatives using annotation CSV |
| `stridulation_scan(...)` | Scan entire audio for stridulations |
| `cavitation_scan(...)` | Scan entire audio for cavitation |
| `highpass_filter(audio, sr, cutoff)` | Butterworth high-pass filter |
| `lowpass_filter(audio, sr, cutoff)` | Butterworth low-pass filter |
| `normalize_global(input_dir, output_dir)` | Normalize folder to global peak |

### Utility Functions

| Function | Description |
|----------|-------------|
| `load_audio(path, normalize=False)` | Load audio with soundfile |
| `save_audio(audio, sr, path, overwrite=False)` | Save audio as WAV |
| `load_snippet(file_path)` | Load snippet from WAV (extracts start_time from filename) |

### ML Functions

| Function | Description |
|----------|-------------|
| `train_model(train_dir, ...)` | Train CNN classifier |
| `classify_spectrograms(input_dir, model_path)` | Classify with trained model |

## ⚙️ Configuration Parameters

### CNN Architecture

```
Input: (128, 128, 3)
├── Conv2D(32, 3x3, relu) + MaxPooling2D(2,2)
├── Conv2D(64, 3x3, relu) + MaxPooling2D(2,2)
├── Conv2D(128, 3x3, relu) + MaxPooling2D(2,2)
├── Flatten
├── Dense(512, relu) + Dropout(0.5)
└── Dense(1, sigmoid)
```

## 📁 Expected Directory Structures

### For `process_audio_file()`
```
output_folder/
├── Audio_snippets/
│   └── source_snippet_starttime.wav
└── Spectrograms/
    └── source_spectrogram_type_starttime_.png
```

### For `train_model()`
```
train_dir/
├── positive/
│   ├── spectrogram1.png
│   └── spectrogram2.png
└── negative/
│   ├── spectrogram3.png
│   └── spectrogram4.png
```



## Notes

- **TensorFlow is lazily loaded** - ML modules are not imported by default to keep the package lightweight. Import them explicitly when needed:
  ```python
  from stridulant import train_model, classify_spectrograms
  ```
- **Audio format** - WAV files recommended. Other formats supported via librosa.

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code follows existing patterns
2. Docstrings are complete (Google format)
3. Tests pass for core functionality

## 📄 License

**MIT + Commons Clause**

This software is freely available for non-commercial use (research, education, personal projects).

Commercial use, including selling the software or offering it as a paid service, is **not permitted** without explicit written permission from the author.

For commercial licensing inquiries, please contact: **saul.rguezm@gmail.com**

## 👤 Author

**Saúl Rodríguez Martínez**
- Email: saul.rguezm@gmail.com
- Bitbucket: [Saul_Rguez/stridulant](https://bitbucket.org/Saul_Rguez/stridulant)

## 🙏 Acknowledgements

Built with:
- [Librosa](https://librosa.org/) - Audio analysis
- [TensorFlow](https://tensorflow.org/) - Deep learning
- [Audiomentations](https://github.com/iver56/audiomentations) - Data augmentation
- [SoundDevice](https://python-sounddevice.readthedocs.io/) - Audio playback

## 📚 Citation

If you use Stridulant in your research, please cite:

```bibtex
@software{stridulant2025,
  author = {Rodríguez Martínez, Saúl},
  title = {Placeholder},
  year = {2025},
  url = {https://bitbucket.org/Saul_Rguez/stridulant}
}
```

---

**Stridulant** 
