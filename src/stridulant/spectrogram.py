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
  
    def __init__(self, spectrogram_data, sr, start_time, transformed, normalized, spec_type='mel', fft_times = None):
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
        self.fft_times = fft_times
        
        
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

    def plot(self, color="inferno"):
        """
        Displays the spectrogram as a plot with axes, color bar, and labels.
        Adjusts representation based on the type of spectrogram.
        
        For Hilbert spectrograms, it plots both the amplitude envelope and the instantaneous frequency.
        For Mel and FFT spectrograms, it uses librosa to display the spectrogram.
        
        Args:
            color (str): sets the colormap. Use plt.colormaps() to learn about your options.

        """
        
        if self.spec_type == 'hilbert':
            # Plotting Hilbert transform with amplitude envelope and instantaneous frequency
            plt.figure(figsize=(20, 8)) 
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

        elif self.spec_type == 'mel':
            plt.figure(figsize=(20, 8))
            extent = [self.fft_times[0], self.fft_times[-1], 0, self.sr/2]
            plt.imshow(self.spectrogram_data, extent=extent, aspect='auto', origin='lower', cmap=color)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout(pad=0)
            plt.show()
        
        elif self.spec_type == 'fft':
            plt.figure(figsize=(20, 8))
            extent = [self.fft_times[0], self.fft_times[-1], 0, self.sr/2]
            plt.imshow(self.spectrogram_data, extent=extent, aspect='auto', origin='lower', cmap=color)
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout(pad=0)
            plt.show()

    def plot_events(self, detected_events=None):
        """
        Plots the Hilbert envelope with percentile thresholds for event detection.
        This method is only available for 'hilbert' type spectrograms.
    
        Args:
            percentiles (list): List of percentiles to display as thresholds (0-100).
            detected_events (list): List of events (start_time, end_time, energy) to mark on the plot.
        """
    
        # Create time axis
        t = np.arange(len(self.spectrogram_data)) / self.sr
    
        # Plot the smoothed envelope
        plt.plot(t, self.spectrogram_data, label='Smooth envelope', color='blue', alpha=0.7, linewidth=1)
        
        for i, (start, end, energy) in enumerate(detected_events):
            #plt.axvspan(start-0.01, end+0.01, alpha=0.2, color='red', label='Event' if i == 0 else "")
            plt.axvspan(start, end, alpha=0.2, color='red', label='Event' if i == 0 else "")

        plt.xlabel('Time (s)')
        plt.ylabel('Amplitud')
        plt.title('Hilbert envelope with events')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    

    def save_img(self, source_name, output_dir,color = "inferno", with_labels=False, verbose=True, events = False, metadata = True):
        """
        Saves the spectrogram as an image file.

        Args:
            output_path (str): The path where the spectrogram image will be saved, including the file name and extension (e.g., 'path/to/file.png').
            with_labels (bool): If True, saves the spectrogram with axes and colorbar. If False, saves it without axes and colorbar. Defaults to False.
            color (str): sets the color map for the plot. Use plt.colormaps() to know your options
            verbose (bool): If True, prints a message confirming the save location. Defaults to True.

        """
        os.makedirs(output_dir, exist_ok=True)
        transform_label = "_transformed" if self.transformed else ""
        norm_label = "_norm" if self.normalized else ""
        if metadata:
            file_name = f"{source_name}_spectrogram_{self.spec_type}{norm_label}{transform_label}_{self.start_time}_.png"
        else:
            file_name = f"{source_name}.png"
        output_path = os.path.join(output_dir, file_name)
        
        if events:
            plt.figure(figsize=(20, 8)) 
            # Create time axis
            t = np.arange(len(self.spectrogram_data)) / self.sr
        
            # Plot the smoothed envelope
            plt.plot(t, self.spectrogram_data, label='Smooth envelope', color='blue', alpha=0.7, linewidth=1)
            
            # Mark detected events if provided
            for i, (start, end, energy) in enumerate(events):
                plt.axvspan(start-0.01, end+0.01, alpha=0.2, color='red', label='Event' if i == 0 else "")
    
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitud')
            plt.title('Hilbert envelope with events')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
            plt.close()

        if self.spec_type == 'hilbert':
            if events:
                plt.figure(figsize=(20, 8)) 
                # Create time axis
                t = np.arange(len(self.spectrogram_data)) / self.sr
            
                # Plot the smoothed envelope
                plt.plot(t, self.spectrogram_data, label='Smooth envelope', color='blue', alpha=0.7, linewidth=1)
                
                # Mark detected events if provided
                for i, (start, end, energy) in enumerate(events):
                    plt.axvspan(start-0.01, end+0.01, alpha=0.2, color='red', label='Event' if i == 0 else "")
        
                plt.xlabel('Time (s)')
                plt.ylabel('Amplitud')
                plt.title('Hilbert envelope with events')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
                plt.close()
                
            else:
                plt.figure(figsize=(20, 8)) 
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
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.5)
                plt.close()

            if verbose:
                print(f"Saved spectrogram to '{output_path}'")
        
        elif self.spec_type == "mel":
            plt.figure(figsize=(20, 8)) 
            extent = [self.fft_times[0], self.fft_times[-1], 0, self.sr/2]
            plt.imshow(self.spectrogram_data, extent=extent, aspect='auto', origin='lower', cmap=color)
            if with_labels:
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.colorbar(format='%+2.0f dB')
                pad=0.5
            else:
                plt.axis('off')
                pad=0

            plt.tight_layout(pad=3.0)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=pad)
            plt.close()
            
        elif self.spec_type == "fft":
            plt.figure(figsize=(20, 8)) 
            extent = [self.fft_times[0], self.fft_times[-1], 0, self.sr/2]
            plt.imshow(self.spectrogram_data, extent=extent, aspect='auto', origin='lower', cmap=color)
            if with_labels:
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.colorbar(format='%+2.0f dB')
                pad = 0.5
            else:
                plt.axis('off')
                pad = 0

            plt.tight_layout(pad=3.0)
            plt.savefig(output_path, bbox_inches='tight', pad_inches=pad)
            plt.close()

            if verbose:
                print(f"Saved spectrogram to '{output_path}'")
        
    def save_table(self, source_name, output_dir, metadata = True):
        """
        Saves the spectral information as a CSV table, following the same naming convention as the spectrogram image.
        
        Args:
            source_name (str): The base name of the audio file (without extension).
            output_dir (str): The directory where the CSV file will be saved.
        """
        os.makedirs(output_dir, exist_ok=True)
        transform_label = "_transformed" if self.transformed else ""
        norm_label = "_norm" if self.normalized else ""
        if metadata:
            file_name = f"{source_name}_spectrogram_{self.spec_type}{norm_label}{transform_label}_{self.start_time}.csv"
        else:
            file_name = f"{source_name}.csv"
        output_path = os.path.join(output_dir, file_name)
        
        # Handle Mel spectrogram or FFT spectrogram
        if self.spec_type == 'mel':
            # In Mel spectrogram, the frequencies are the Mel bins
            freqs = librosa.mel_frequencies(n_mels=self.spectrogram_data.shape[0], fmin=0, fmax=self.sr // 2)
            times = librosa.times_like(self.spectrogram_data, sr=self.sr)
            # Transpose the spectrogram data to have rows as frequencies and columns as times
            spectrogram_values = self.spectrogram_data
            
            # Create the DataFrame with frequencies as index and times as columns
            df = pd.DataFrame(spectrogram_values, index=freqs, columns=times)
            
            # Save the DataFrame to a CSV file
            df.to_csv(output_path)
            print(f"Spectrogram table saved to '{output_path}'")
        
        elif self.spec_type == 'fft':
            # In FFT spectrogram, calculate frequencies based on actual data shape
            n_freq_bins = self.spectrogram_data.shape[0]
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2*(n_freq_bins-1))
            times = librosa.times_like(self.spectrogram_data, sr=self.sr, n_fft=2*(n_freq_bins-1))
            
            # Create the DataFrame with frequencies as index and times as columns
            df = pd.DataFrame(self.spectrogram_data, index=freqs, columns=times)
            
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
    
            # Create the DataFrame with time as index and the two types of data as columns
            df = pd.DataFrame({
                'Amplitude_Envelope': self.spectrogram_data,
                'Instantaneous_Frequency': np.concatenate(([0], instantaneous_frequency))
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
