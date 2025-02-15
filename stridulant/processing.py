# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:39:19 2025

@author: Saul
"""

import os
from tqdm import tqdm  
import matplotlib
import matplotlib.pyplot as plt


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
    """
    start_sample = int(start_time * sr)
    duration_samples = int(duration_sec * sr)
    if start_sample + duration_samples <= len(audio):
        segment = audio[start_sample:start_sample + duration_samples]
        return AudioSnippet(segment, sr, start_time)
    else:
        raise ValueError(f"Snippet duration exceeds available audio length at {start_time} seconds.")

if __name__ == "__main__":
    audio_path = "test_audio.wav"
    start = 10
    duration = 5
    output_folder = "segments"

    audio, sr = load_audio(audio_path)

    # Create a single snippet
    snippet = create_snippet(audio, sr, start, duration)
    snippet.save(os.path.splitext(os.path.basename(audio_path))[0], output_folder)
    snippet.play()

    # Create and save different types of spectrograms
    for spec_type in ["mel", "fft", "hilbert"]:
        spec = snippet.spectrogram(spec_type=spec_type)
        spec.save(os.path.join(output_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_spectrogram_{snippet.start_time:.1f}_{spec_type}.png"))


def process_audio_file(audio_path, snippet_duration=2, output_folder=None, update_freq=10):
    """
    Processes the given audio file by splitting it into snippets, generating a mel spectrogram for each, 
    and saving them in appropriate directories.

    Args:
    audio_path (str): Path to the input audio file.
    snippet_duration (float): Duration of each snippet in seconds. Default is 2 seconds.
    output_folder (str): Base directory where the snippets and spectrograms will be saved. If None, uses el mismo directorio del archivo de audio.
    update_freq (int): Frequency of updates for the progress bar (every X snippets).
    """
    # Si no se pasa un output_folder, usamos el directorio del archivo de audio
    if output_folder is None:
        output_folder = os.path.dirname(audio_path)

    # Cargar el audio
    audio, sr = load_audio(audio_path)

    # Crear subdirectorios para los snippets y espectrogramas en el directorio del archivo original
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    snippets_dir = os.path.join(output_folder, "Audio_snippets")
    spectrograms_dir = os.path.join(output_folder, "Spectrograms")
    os.makedirs(snippets_dir, exist_ok=True)
    os.makedirs(spectrograms_dir, exist_ok=True)

    # Variables de inicio y duración del snippet
    start_time = 0

    # Obtener el número total de snippets para la barra de progreso
    total_snippets = int(len(audio) / (snippet_duration * sr))

    # Usar tqdm para la barra de progreso (configuración para que se actualice sin reimprimir)
    matplotlib.use('Agg')
    with tqdm(total=total_snippets, desc="Procesando snippets", unit="snippet", ncols=100, position=0, leave=True) as pbar:
        for i in range(total_snippets):
            try:
                # Crear un snippet
                snippet = create_snippet(audio, sr, start_time, snippet_duration)
    
                # Guardar el snippet
                snippet.save(base_name, snippets_dir, verbose=False)
    
                # Crear el espectrograma mel para este snippet
                spec = snippet.spectrogram(spec_type="mel")
                spec_filename = f"{base_name}_spectrogram_{start_time:.1f}_mel.png"
                spec.save(os.path.join(spectrograms_dir, spec_filename), with_labels=False, verbose=False)
    
                # Actualizar el tiempo de inicio para el siguiente snippet
                start_time += snippet_duration
    
                # Actualizar la barra de progreso después de cada `update_freq` iteraciones
                if i % update_freq == 0:
                    pbar.update(update_freq)
    
            except ValueError as e:
                print(f"Error al crear el snippet: {e}")
                break
    
                # Cerrar la figura después de cada iteración para evitar apilamiento
                plt.close('all')  # Aquí se usa `close('all')` ya que hay que cerrar todas las figuras abiertas
    
        # Asegurarse de que las figuras se cierren después de ser guardadas
        plt.close('all')  # Cerrar cualquier figura abierta al final
    matplotlib.use('TkAgg')
    
if __name__ == "__main__":
    audio_file_path = "test_audio.wav"  # Cambia esto con la ruta de tu archivo de audio
    process_audio_file(audio_file_path)