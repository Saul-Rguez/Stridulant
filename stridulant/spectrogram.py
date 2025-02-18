# -*- coding: utf-8 -*-
"""
Module for creating and manipulating audio spectrograms. It includes functionality
for generating and visualizing Mel spectrograms, FFT spectrograms, and Hilbert transforms.
The class also supports saving spectrograms as image files with customizable options.

Author: Saul Rodriguez Martinez
Date: 2025-02-15

"""

import librosa
import librosa.display
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

class Spectrogram:
    """
    Represents a spectrogram and provides methods to visualize and save it.
    This class can generate Mel spectrograms, FFT spectrograms, or Hilbert transforms.
    The spectrogram is computed from the audio data and can be displayed or saved in various formats.
    """
  
    def __init__(self, spectrogram_data, sr, spec_type='mel'):
        """
        Initializes the Spectrogram instance.

        Args:
            spectrogram_data (np.ndarray): The spectrogram data (e.g., Mel spectrogram, FFT result, or Hilbert transform result).
            sr (int): The sampling rate of the audio used to generate the spectrogram.
            spec_type (str): The type of the spectrogram ('mel', 'fft', or 'hilbert').

        """
        self.spectrogram_data = spectrogram_data
        self.sr = sr
        self.spec_type = spec_type
    
    def __repr__(self):
        """
        Returns a string representation of the Spectrogram instance that can
        be used to recreate the object. Useful for debugging and development.

        Returns:
            str: String representation of the Spectrogram object.
        """
        return f"Spectrogram(spectrogram_data={self.spectrogram_data.shape}, sr={self.sr}, spec_type='{self.spec_type}')"

    def __str__(self):
        """
        Returns a user-friendly string representation of the Spectrogram instance.

        Returns:
            str: A descriptive string with information about the spectrogram type, sampling rate, and data shape.
        """
        return f"Spectrogram with {self.spec_type} spectrogram, Sampling rate: {self.sr} Hz, Data shape: {self.spectrogram_data.shape}"

    def plot(self):
        """
        Displays the spectrogram as a plot with axes, color bar, and labels.
        Adjusts representation based on the type of spectrogram.
        
        For Hilbert spectrograms, it plots both the amplitude envelope and the instantaneous frequency.
        For Mel and FFT spectrograms, it uses librosa to display the spectrogram.

        """
        plt.figure(figsize=(6, 4))

        if self.spec_type == 'hilbert':
            # Plotting Hilbert transform with amplitude envelope and instantaneous frequency
            t = np.arange(len(self.spectrogram_data)) / self.sr
            plt.subplot(2, 1, 1)
            plt.title("Amplitude-modulated Signal (Hilbert Transform)")
            plt.ylabel("Amplitude")
            plt.plot(t, self.spectrogram_data, label='Amplitude Envelope', color='C0')

            analytic_signal = hilbert(self.spectrogram_data)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sr

            plt.subplot(2, 1, 2)
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.plot(t[1:], instantaneous_frequency, label='Instantaneous Frequency', color='C2')
            plt.legend()
            plt.tight_layout()
            plt.show()

        else:
            # Plotting Mel or FFT spectrogram
            librosa.display.specshow(self.spectrogram_data, sr=self.sr, cmap='inferno')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout(pad=0)
            plt.show()

    def save(self, output_path, with_labels=False, verbose=True):
        """
        Saves the spectrogram as an image file.

        Args:
            output_path (str): The path where the spectrogram image will be saved, including the file name and extension (e.g., 'path/to/file.png').
            with_labels (bool): If True, saves the spectrogram with axes and colorbar. If False, saves it without axes and colorbar. Defaults to False.
            verbose (bool): If True, prints a message confirming the save location. Defaults to True.

        """
        plt.figure(figsize=(4, 4))

        if self.spec_type == 'hilbert':
            # Saving Hilbert transform plot (amplitude envelope and instantaneous frequency)
            t = np.arange(len(self.spectrogram_data)) / self.sr
            plt.subplot(2, 1, 1)
            plt.title("Amplitude-modulated Signal (Hilbert Transform)")
            plt.ylabel("Amplitude")
            plt.plot(t, self.spectrogram_data, label='Amplitude Envelope', color='C0')

            analytic_signal = hilbert(self.spectrogram_data)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sr

            plt.subplot(2, 1, 2)
            plt.xlabel("Time (s)")
            plt.ylabel("Frequency (Hz)")
            plt.plot(t[1:], instantaneous_frequency, label='Instantaneous Frequency', color='C2')
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            if verbose:
                print(f"Saved spectrogram to '{output_path}'")
        
        else:
            # Saving Mel or FFT spectrogram plot
            librosa.display.specshow(self.spectrogram_data, sr=self.sr, cmap='inferno')
            if with_labels:
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.colorbar(format='%+2.0f dB')
            else:
                plt.axis('off')

            plt.tight_layout(pad=0)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            if verbose:
                print(f"Saved spectrogram to '{output_path}'")
