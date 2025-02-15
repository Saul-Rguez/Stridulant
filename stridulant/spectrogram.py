# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:36:33 2025

@author: Saul
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
        spectrogram_data (np.ndarray): The spectrogram data.
        sr (int): The sampling rate of the audio used to generate the spectrogram.
        spec_type (str): The type of the spectrogram ('mel', 'fft', or 'hilbert').
        """
        self.spectrogram_data = spectrogram_data
        self.sr = sr
        self.spec_type = spec_type  # Type of spectrogram ('mel', 'fft', 'hilbert')

    def plot(self):
        """
        Displays the spectrogram as a plot with axes, color bar, and labels.
        Adjusts representation based on the type of spectrogram.
        """
        plt.figure(figsize=(6, 4))

        if self.spec_type == 'hilbert':
            # If it's a Hilbert spectrogram, we assume spectrogram_data is the amplitude envelope
            # Plot the Amplitude Envelope
            t = np.arange(len(self.spectrogram_data)) / self.sr
            plt.subplot(2, 1, 1)
            plt.title("Amplitude-modulated Signal (Hilbert Transform)")
            plt.ylabel("Amplitude")
            plt.plot(t, self.spectrogram_data, label='Amplitude Envelope', color='C0')

            # If instantaneous frequency is needed, calculate from the envelope
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
            # For Mel and FFT spectrograms, proceed as usual
            librosa.display.specshow(self.spectrogram_data, sr=self.sr, cmap='inferno')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout(pad=0)
            plt.show()

    def save(self, output_path, with_labels=False, verbose = True):
        """
        Saves the spectrogram as an image file. 

        Args:
        output_path (str): The path where the spectrogram image will be saved.
        with_labels (bool): If True, saves the spectrogram with axes and colorbar.
                            If False, saves it without axes and colorbar. Defaults to False.
        """
        
        plt.figure(figsize=(4, 4))

        if self.spec_type == 'hilbert':
            # If it's a Hilbert spectrogram, we assume spectrogram_data is the amplitude envelope
            # Plot the Amplitude Envelope
            t = np.arange(len(self.spectrogram_data)) / self.sr
            plt.subplot(2, 1, 1)
            plt.title("Amplitude-modulated Signal (Hilbert Transform)")
            plt.ylabel("Amplitude")
            plt.plot(t, self.spectrogram_data, label='Amplitude Envelope', color='C0')

            # If instantaneous frequency is needed, calculate from the envelope
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