# -*- coding: utf-8 -*-
"""
Module that contains helper functions for loading and processing audio files.

Functions:
    - load_audio: Loads an audio file from the specified path and returns the audio along with the sampling rate.
    - load_snippet: Loads an audio snippet from a .wav file and returns an AudioSnippet object with the corresponding data.

Author: Saul Rodriguez Martinez
Creation date: 2025-02-15
"""

import librosa
from stridulant.audio_snippet import AudioSnippet


def load_audio(path, normalize = False):
    """
    Loads an audio file from the specified path.

    This function uses the librosa library to load an audio file from the given file path 
    and returns the audio data as a numpy array along with the sampling rate.

    Args:
        path (str): The full path of the audio file.

    Returns:
        tuple: A tuple containing the audio data as a numpy array and the sampling rate (sr).
        
    Raises:
        RuntimeError: If there is an issue loading the audio file, such as a file not found or format error.
    
    Example:
        audio, sr = load_audio('path/to/audio/file.wav')

    """
    from stridulant.processing import normalize_audio
    try:
        # Load audio file using librosa
        audio, sr = librosa.load(path, sr=None)        
        if normalize:
            audio = normalize_audio (audio)
        return audio, sr
    except Exception as e:
        # Raise an error if loading fails
        raise RuntimeError(f"Error loading the audio file {path}: {e}")


def load_snippet(file_path):
    """
    Loads an audio snippet from a file and returns an AudioSnippet object.

    This function loads a snippet from a .wav file and creates an AudioSnippet object 
    containing the audio data, sample rate, and start time (extracted from the file name).

    Args:
        file_path (str): The path to the .wav snippet file.

    Returns:
        AudioSnippet: An object containing audio data, sample rate, and start time.

    Example:
        snippet = load_snippet('path/to/snippet_123.wav')

    """
    # Load the audio snippet and its sample rate using librosa
    audio, sample_rate = librosa.load(file_path, sr=None)
    
    # Extract start time from the file name (assuming format 'someprefix_starttime.wav')
    start_time = file_path[0:-4].split("_")[-1]
    
    # Return an AudioSnippet object with the loaded data
    return AudioSnippet(audio, sample_rate, start_time)
