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
import os
import pandas as pd

class Spectrogram:
    """
    Represents a spectrogram and provides methods to visualize and save it.
    This class can generate Mel spectrograms, FFT spectrograms, or Hilbert transforms.
    The spectrogram is computed from the audio data and can be displayed or saved in various formats.
    """
  
    def __init__(self, spectrogram_data, sr, start_time, transformed, normalized, spec_type='mel'):
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
        self.start_time = start_time
        self.transformed = transformed
        self.normalized = normalized
        
        
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

    def save_img(self, source_name, output_dir, with_labels=False, verbose=True):
        """
        Saves the spectrogram as an image file.

        Args:
            output_path (str): The path where the spectrogram image will be saved, including the file name and extension (e.g., 'path/to/file.png').
            with_labels (bool): If True, saves the spectrogram with axes and colorbar. If False, saves it without axes and colorbar. Defaults to False.
            verbose (bool): If True, prints a message confirming the save location. Defaults to True.

        """
        os.makedirs(output_dir, exist_ok=True)
        transform_label = "_transformed" if self.transformed else ""
        norm_label = "_norm" if self.normalized else ""
        file_name = f"{source_name}_spectrogram_{self.spec_type}{norm_label}{transform_label}_{self.start_time}_.png"
        output_path = os.path.join(output_dir, file_name)
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
        
    def save_table(self, source_name, output_dir):
        """
        Saves the spectral information as a CSV table, following the same naming convention as the spectrogram image.
        
        Args:
            source_name (str): The base name of the audio file (without extension).
            output_dir (str): The directory where the CSV file will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        transform_label = "_transformed" if self.transformed else ""
        norm_label = "_norm" if self.normalized else ""
        file_name = f"{source_name}_spectrogram_{self.spec_type}{norm_label}{transform_label}_{self.start_time}.csv"
        output_path = os.path.join(output_dir, file_name)
        
        # Handle Mel spectrogram or FFT spectrogram
        if self.spec_type == 'mel':
            # In Mel spectrogram, the frequencies are the Mel bins
            freqs = librosa.mel_frequencies(n_mels=self.spectrogram_data.shape[0], fmin=0, fmax=self.sr // 2)
            times = librosa.times_like(self.spectrogram_data)
            # Transpose the spectrogram data to have rows as time and columns as frequencies
            spectrogram_values = self.spectrogram_data
            
            # Create the DataFrame with time as index and frequencies as columns
            df = pd.DataFrame(spectrogram_values, columns=times, index=freqs)
            
            # Save the DataFrame to a CSV file
            df.to_csv(output_path)
            print(f"Spectrogram table saved to '{output_path}'")
        
        elif self.spec_type == 'fft':
            # In FFT spectrogram, we calculate the frequencies using fft_frequencies
            freqs = librosa.fft_frequencies(sr=self.sr)
            times = librosa.times_like(self.spectrogram_data)
        
            # Transpose the spectrogram data to have rows as time and columns as frequencies
            spectrogram_values = self.spectrogram_data
            
            # Create the DataFrame with time as index and frequencies as columns
            df = pd.DataFrame(spectrogram_values, columns=times, index=freqs)
            
            # Save the DataFrame to a CSV file
            df.to_csv(output_path)
            print(f"Spectrogram table saved to '{output_path}'")
    
        elif self.spec_type == 'hilbert':
            # For Hilbert transform, save the amplitude envelope and instantaneous frequency
            t = np.arange(len(self.spectrogram_data)) / self.sr
    
            # Create a DataFrame for Hilbert data (amplitude envelope and instantaneous frequency)
            analytic_signal = hilbert(self.spectrogram_data)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sr
    
            # Create the DataFrame with time as index and the two types of data (amplitude and frequency) as columns
            df = pd.DataFrame({
                'Amplitude_Envelope': self.spectrogram_data,
                'Instantaneous_Frequency': np.concatenate(([0], instantaneous_frequency))  # Pad with 0 for consistency
            }, index=t)
    
            # Save the Hilbert transform DataFrame to CSV
            df.to_csv(output_path)
            print(f"Hilbert spectrogram table saved to '{output_path}'")
    
        else:
            print(f"Spectrogram type '{self.spec_type}' is not supported for table saving.")
            
            
    def compute_power_metrics(self):
        """
        Computes Average Power Density (APD) and Peak Power Density (PPD) for 
        fft or mel spectrograms, not for Hilbert
    
        Args:
            S_dB (np.ndarray): Spectrogram data already in dB scale.
    
        Returns:
            tuple: (APD, PPD) in dB.
        """
        if self.spec_type == 'hilbert':
            print ("This function cannot be applied to Hilbert spectrograms")
        else:
            APD = np.mean(self.spectrogram_data)
            PPD = np.max(self.spectrogram_data)   
    
        return APD, PPD