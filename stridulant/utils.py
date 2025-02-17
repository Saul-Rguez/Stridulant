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

def load_audio(path):
    """
    Loads an audio file from the specified path.

    Args:
        path (str): The full path of the audio file.

    Returns:
        tuple: A tuple containing the audio as a numpy array and the sampling rate (sr).
        
    Raises:
        RuntimeError: If there is an issue loading the audio file, such as a file not found or format error.
    """
    try:
        audio, sr = librosa.load(path, sr=None)  
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Error al cargar el archivo de audio {path}: {e}")


def load_snippet(file_path):
    """
    Loads an audio snippet from a file and returns an AudioSnippet object.

    Args:
        file_path (str): The path to the .wav snippet file.

    Returns:
        AudioSnippet: An object containing audio data, sample rate, and length.
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    length = len(audio) / sample_rate
    return AudioSnippet(audio, sample_rate, length)