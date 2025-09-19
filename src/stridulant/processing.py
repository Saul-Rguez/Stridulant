# -*- coding: utf-8 -*-
"""
Module: processing
Description:
    This module contains utility functions for processing audio data, including:
    - Segmenting audio files into snippets.
    - Generating spectrograms for analysis (Mel, FFT, Hilbert).
    - Saving audio snippets and spectrogram images to specified directories.
    - Playing and visualizing audio data.

    The module supports a range of audio file processing tasks for further analysis, 
    especially in the context of stridulation detection in ants. It uses the 
    AudioSnippet class to handle audio data and the Spectrogram class to visualize 
    frequency content.

Functions:
    - create_snippet: Splits the audio into smaller segments or snippets.
    - process_audio_file: Processes the entire audio file, generating snippets and spectrograms.
    - Additional helper functions for visualization and saving results.
    
Author: Saul Rodriguez Martinez
Date: 2025-02-15

"""

import os
from tqdm import tqdm  
import matplotlib
import matplotlib.pyplot as plt
from stridulant.audio_snippet import AudioSnippet
from stridulant.utils import load_audio
import shutil
import pandas as pd
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

def create_snippet(audio, sr, start_time, duration_sec):
    """
    Creates a single audio snippet from the provided audio data.

    Args:
    audio (np.ndarray): The full audio data.
    sr (int): The sampling rate of the audio.
    start_time (float): The start time in seconds of the snippet.
    duration_sec (float): The duration of the snippet in seconds.

    Returns:
    AudioSnippet: An instance of the AudioSnippet class containing the created snippet.
    
    Raises:
    ValueError: If the snippet duration exceeds the available length of the audio.
    """
    start_sample = int(start_time * sr)
    duration_samples = int(duration_sec * sr)
    if start_sample + duration_samples <= len(audio):
        segment = audio[start_sample:start_sample + duration_samples]
        return AudioSnippet(segment, sr, start_time)
    else:
        raise ValueError(f"Snippet duration exceeds available audio length at {start_time} seconds.")

def process_audio_file(audio_path, snippet_duration=2, with_labels=False, spec_type="mel",n_fft = 256, hop_length = 224, window = "boxcar", color = "inferno", output_folder=None, update_freq=10):
    """
    Processes the given audio file by splitting it into snippets, generating a mel spectrogram for each, 
    and saving them in appropriate directories.

    Args:
    audio_path (str): Path to the input audio file.
    snippet_duration (float): Duration of each snippet in seconds. Default is 2 seconds.
    output_folder (str): Base directory where the snippets and spectrograms will be saved. If None, uses the same directory as the audio file.
    with_labels (bool): toggles the axis and labels.
    spec_type (str): chooses between the 3 posible types, hil, mel or fft.
    color (str): choose a colormap for the image
    n_fft (int): size of the window for the fft (see spectrogram function).
    hop_length (int): overlap of the windows (see spectrogram funtion)
    window (str): window type (see spectrogram function)
    update_freq (int): Frequency of updates for the progress bar (every X snippets).
    """
    original_backend = matplotlib.get_backend()
    try:
        if output_folder is None:
            output_folder = os.path.splitext(audio_path)[0]
    
        audio, sr = load_audio(audio_path)
    
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        snippets_dir = os.path.join(output_folder, "Audio_snippets")
        spectrograms_dir = os.path.join(output_folder, "Spectrograms")
        os.makedirs(snippets_dir, exist_ok=True)
        os.makedirs(spectrograms_dir, exist_ok=True)
    
        start_time = 0
    
        total_snippets = int(len(audio) / (snippet_duration * sr))
    
        matplotlib.use('Agg')
        print(f"Processing file {audio_path}")
        with tqdm(total=total_snippets, desc="Processing snippets", unit="snippet", ncols=100, position=0, leave=True) as pbar:
            for i in range(total_snippets):
                try:
                    snippet = create_snippet(audio, sr, start_time, snippet_duration)
    
                    snippet.save(base_name, snippets_dir, verbose=False)
    
                    spec = snippet.spectrogram(spec_type=spec_type,n_fft = n_fft, hop_length = hop_length, window = window)
                    spec.save_img(base_name, spectrograms_dir, with_labels=with_labels, color=color, verbose=False)
    
                    start_time += snippet_duration
                    if i % update_freq == 0:
                        pbar.update(update_freq)
        
                except ValueError as e:
                    print(f"Error creating snippet: {e}")
                    break
    
                    plt.close('all')
    
            plt.close('all')  
        
    except KeyboardInterrupt:
        print ("\nProcess cancelled.")
    except Exception as e:
        print(f"Processing error: {e}")
    finally:
        matplotlib.use(original_backend)
        plt.close('all') 
        print("Graphic backend restored.")
        
def annotate_data(csv_path, snippets_dir, spectrograms_dir, snippet_duration=2.0, csv_delim = '\t'):
    """
    Separates audio snippets and spectrograms into positive and negative folders based on annotations.

    Args:
    csv_path (str): Path to the CSV file containing start and end times of stridulations.
    snippets_dir (str): Path to the directory containing audio snippets.
    spectrograms_dir (str): Path to the directory containing spectrograms.
    snippet_duration (float): Duration of each audio snippet in seconds.
    csv_delim (str): Delimiter used in the CSV file (default is tab).
    
    The CSV should have two columns: `start_time` and `end_time` indicating the time intervals of stridulations.
    All snippets overlapping these intervals are considered positive samples.
    """
    annotations = pd.read_csv(csv_path, delimiter=csv_delim)
    positive_snippets_dir = os.path.join(snippets_dir, 'positives')
    negative_snippets_dir = os.path.join(snippets_dir, 'negatives')
    positive_spectrograms_dir = os.path.join(spectrograms_dir, 'positives')
    negative_spectrograms_dir = os.path.join(spectrograms_dir, 'negatives')
    os.makedirs(positive_snippets_dir, exist_ok=True)
    os.makedirs(negative_snippets_dir, exist_ok=True)
    os.makedirs(positive_spectrograms_dir, exist_ok=True)
    os.makedirs(negative_spectrograms_dir, exist_ok=True)

    def is_positive(snippet_time):        
        for _, row in annotations.iterrows():
            if not (row['end_time'] <= snippet_time or row['start_time'] >= snippet_time + snippet_duration):
                return True
        return False

    for file in os.listdir(snippets_dir):
        if file.endswith('.wav') or file.endswith('.png'):
            snippet_time = float(file.split('_')[-1].split('.')[0])
            target_dir = positive_snippets_dir if is_positive(snippet_time) else negative_snippets_dir
            src = os.path.join(snippets_dir, file)
            dst = os.path.join(target_dir, file)
            shutil.move(src, dst)

    for file in os.listdir(spectrograms_dir):
        if file.endswith('.png'):
            snippet_time = float(file.split('_')[-1].split('.')[0])
            target_dir = positive_spectrograms_dir if is_positive(snippet_time) else negative_spectrograms_dir
            src = os.path.join(spectrograms_dir, file)
            dst = os.path.join(target_dir, file)
            shutil.move(src, dst)

    print(f"Separation completed successfully in {csv_path}.")
    

def normalize_audio(audio: np.ndarray, target_max: float = 0.75, global_max: float = None) -> np.ndarray:
    """
    Peak normalization of the audio signal.
    It rescales any given audio to a percentage of the maximum amplitude.
    Defaults to 75%.

    Args:
        audio (np.ndarray): The input audio signal.
        target_max (float): max percentage for the peak value (default 75%)
        global_max (float): only needed when normalizing several files to one 
        single peak. It is the absolute maximum amplitude across several files, 
        and it supersedes the need to calculate the maximum for each individual 
        file. It is calculated and used within the mormalize_global function.

    Returns:
        np.ndarray: The normalized audio signal.
    """
    if not global_max:
        current_max = np.max(np.abs(audio))
    else:
        current_max = global_max
  
    if current_max > 0:
       scaling_factor = target_max / current_max
       normalized_audio = audio * scaling_factor
       return normalized_audio
    else:
       return audio  # Return original if silent

def normalize_global(input_dir: str, output_dir: str, target_max: float = 0.75):
    """
    Normalizes all audio files in a folder using global maximum (one peak for 
    all files) computed from all files in the folder.

    Args:
        input_folder (str): Path to the folder containing input audio files.
        output_folder (str): Path to the folder where normalized audio files will be saved.
        target_max (float): max percentage for the peak value (default 75%)
    """
    os.makedirs(output_dir, exist_ok=True)
    file_paths = []    

    global_max = 0
    with tqdm(total=len(os.listdir(input_dir)), desc="Finding global max", unit="file", ncols=100, position=0, leave=True) as pbar:
        for filename in os.listdir(input_dir):
            if filename.endswith(".wav"):
                filepath = os.path.join(input_dir, filename)
                file_paths.append(filepath)
                audio, _ = librosa.load(filepath, sr = None)
                max_value = np.max(np.abs(audio))
                global_max = max(global_max, max_value)
            pbar.update(1)
            
    if global_max == 0:
        raise ValueError("All files are silent or empty. Normalization is not possible.")

    # Normalize and save each file
    with tqdm(total=len(file_paths), desc="Normalizing files", unit="file", ncols=100, position=0, leave=True) as pbar:
        for file in file_paths:
            audio, sr = librosa.load(file, sr = None)
            normalized_audio = normalize_audio(audio, target_max, global_max)
            filename = os.path.basename(file)
            output_filename = f"{os.path.splitext(filename)[0]}_normalized.wav"
            output_filepath = os.path.join(output_dir, output_filename)
            sf.write(output_filepath, normalized_audio, sr)
            
            pbar.set_postfix({"file": filename})  # Optional: display current filename
            pbar.update(1)  # Update progress bar after processing each file


def highpass_filter(audio, sr, cutoff=3000, order=8):
    """
    Applies a high-pass Butterworth filter to the input audio signal.

    Args:
        audio (np.ndarray): Input audio signal.
        sr (int): Sampling rate of the audio.
        cutoff (float): Cutoff frequency in Hz. Default is 3000 Hz.
        order (int): Filter order. Default is 8 (48 dB/octave roll-off).

    Returns:
        np.ndarray: Filtered audio signal.
    
    Raises:
        ValueError: If input parameters are invalid or filtering fails.
    """
    try:
        nyq = 0.5 * sr
        normal_cutoff = cutoff / nyq
        if normal_cutoff >= 1.0:
            raise ValueError("Cutoff frequency must be less than Nyquist rate.")

        b, a = butter(order, normal_cutoff, btype='highpass', analog=False)
        filtered_audio = filtfilt(b, a, audio, axis=0)
        return filtered_audio
    except Exception as e:
        raise ValueError(f"Error applying high-pass filter: {e}")