# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:38:24 2025

@author: Saul
"""
import librosa
from audio_snippet import AudioSnippet

def load_audio(path):
    """
    Carga un archivo de audio desde la ruta especificada.

    Args:
        path (str): La ruta del archivo de audio.

    Returns:
        tuple: Una tupla que contiene el audio como un array de numpy y la frecuencia de muestreo (sr).
    
    Raises:
        RuntimeError: Si hay un problema al cargar el archivo de audio.
    """
    try:
        audio, sr = librosa.load(path, sr=None)  # Cargar el audio con la frecuencia de muestreo original
        return audio, sr
    except Exception as e:
        raise RuntimeError(f"Error al cargar el archivo de audio {path}: {e}")


def load_snippet(file_path):
    """
    Carga un snippet de audio desde un archivo y devuelve un objeto AudioSnippet.
    
    Args:
        file_path (str): Ruta al archivo .wav del snippet.
        
    Returns:
        AudioSnippet: Objeto cargado con datos de audio, sample_rate y longitud.
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    length = len(audio) / sample_rate
    return AudioSnippet(audio, sample_rate, length)