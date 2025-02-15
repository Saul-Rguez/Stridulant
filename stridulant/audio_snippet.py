# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:35:15 2025

@author: Saul
"""

import librosa
import librosa.display
import soundfile as sf
import sounddevice as sd
import os
import numpy as np
from scipy.signal import hilbert
from .spectrogram import Spectrogram

class AudioSnippet:
    """
    Represents a snippet (segment) of audio data with methods to manipulate,
    save, and play the snippet. This class is used to handle small chunks of audio
    for further analysis or processing.
    """

    def __init__(self, audio, sr, start_time):
        """
        Initializes the AudioSnippet instance.

        Args:
        audio (np.ndarray): Audio data for the snippet.
        sr (int): Sampling rate of the audio.
        start_time (float): Start time of the snippet in seconds.
        """
        self.audio = audio
        self.sr = sr
        self.start_time = start_time

    def save(self, source_name, output_dir, verbose = True):
        """
        Saves the current audio snippet as a .wav file in the specified output directory.

        Args:
        source_name (str): Base name for the saved file (typically the source file name).
        output_dir (str): Directory where the snippet will be saved.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_name = f"{source_name}_snippet_{self.start_time:.1f}.wav"
        output_path = os.path.join(output_dir, file_name)
        sf.write(output_path, self.audio, self.sr)
        if verbose:
            print(f"Saved snippet to '{output_path}'")

    def play(self):
        """
        Plays the audio snippet using the sounddevice library.

        Waits for the playback to finish before returning.
        """
        sd.play(self.audio, self.sr)
        sd.wait()

    def spectrogram(self, spec_type='mel'):
        """
        Creates and returns a Spectrogram instance based on the current snippet.

        Args:
        spec_type (str): Type of the spectrogram ('mel', 'fft', 'hilbert').

        Returns:
        Spectrogram: A Spectrogram instance containing the generated spectrogram.
        """
        if spec_type == 'hilbert':
            # Apply Hilbert Transform to get analytic signal
            analytic_signal = hilbert(self.audio)
            amplitude_envelope = np.abs(analytic_signal)

            # Return amplitude envelope and instantaneous frequency as spectrogram_data
            spectrogram_data = amplitude_envelope
            return Spectrogram(spectrogram_data, self.sr, spec_type='hilbert')
        else:
            # For Mel and FFT spectrograms, proceed as usual
            if spec_type == 'mel':
                max_freq = self.sr / 2  # Nyquist frequency
                S = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=128, fmax=max_freq)
            else:  # FFT
                S = np.abs(librosa.stft(self.audio))

            S_dB = librosa.power_to_db(S, ref=np.max)
            return Spectrogram(S_dB, self.sr, spec_type=spec_type)
