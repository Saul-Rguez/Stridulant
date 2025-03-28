# -*- coding: utf-8 -*-
"""
Module: AudioSnippet
Description:
    This module defines the AudioSnippet class, which represents a snippet (segment) 
    of audio data. It provides methods to save the snippet to a .wav file, play the audio, 
    and generate spectrograms (Mel, FFT, Hilbert) for further analysis and processing. 
    It also supports data augmentation techniques for audio processing, useful in AI experiments 
    like audio classification.

Classes:
    AudioSnippet: Represents a segment of audio with methods to manipulate, save, 
                  and generate spectrograms.
                  Provides augmentation methods like Gaussian noise, time stretching, 
                  pitch shifting, etc.

Author: Saul Rodriguez Martinez
Date: 2025-02-15

"""

import librosa
import librosa.display
import soundfile as sf
import sounddevice as sd
import os
import numpy as np
from scipy.signal import hilbert
from audiomentations import AddGaussianNoise, TimeStretch, PitchShift, Shift, ClippingDistortion, AddBackgroundNoise, TimeMask


class AudioSnippet:
    """
    Represents a snippet (segment) of audio data with methods to manipulate,
    save, and play the snippet. This class is used to handle small chunks of audio
    for further analysis or processing.
    
    Attributes:
        audio (np.ndarray): Audio data for the snippet.
        sr (int): Sampling rate of the audio.
        start_time (float): Start time of the snippet in seconds.
        transformed (boolean): flag for transformed files.
        normalized (boolean): flag for normalized files
    """

    def __init__(self, audio, sr, start_time):
        """
        Initializes the AudioSnippet instance with the given audio data, sampling rate,
        and start time.

        Args:
            audio (np.ndarray): Audio data for the snippet.
            sr (int): Sampling rate of the audio.
            start_time (float): Start time of the snippet in seconds.
        """
        self.audio = audio
        self.sr = sr
        self.start_time = start_time
        self.transformed = False
        self.normalized = False
        
    def __repr__(self):
        """
        Returns a string representation of the AudioSnippet instance that can be used 
        to recreate the object. This is useful for debugging and development.

        Returns:
            str: A string representation of the AudioSnippet instance.
        """
        return f"AudioSnippet(audio_shape={self.audio.shape}, sr={self.sr}, start_time={self.start_time}s)"

    def __str__(self):
        """
        Returns a user-friendly string representation of the AudioSnippet instance.

        Returns:
            str: A descriptive string representing the audio snippet.
        """
        return f"Audio snippet starting at {self.start_time} seconds, sampling rate: {self.sr} Hz, audio data shape: {self.audio.shape}"
    

    def save(self, source_name, output_dir, verbose=True):
        """
        Saves the current audio snippet as a .wav file in the specified output directory.

        Args:
            source_name (str): Base name for the saved file (typically the source file name).
            output_dir (str): Directory where the snippet will be saved.
            verbose (bool): If True, prints a message when the snippet is saved. Default is True.
        
        Returns:
            None
        """
        os.makedirs(output_dir, exist_ok=True)
        
        transform_label = "_transformed" if self.transformed else ""
        norm_label = "_norm" if self.normalized else ""
        file_name = f"{source_name}_snippet{norm_label}{transform_label}_{self.start_time}.wav"
        output_path = os.path.join(output_dir, file_name)
        sf.write(output_path, self.audio, self.sr)
        if verbose:
            print(f"Saved snippet to '{output_path}'")

    def play(self):
        """
        Plays the audio snippet using the sounddevice library. Waits for the playback 
        to finish before returning.

        Returns:
            None
        """
        sd.play(self.audio, self.sr)
        sd.wait()
        

    def spectrogram(self, spec_type='mel',):
        """
        Creates and returns a Spectrogram instance based on the current snippet. The 
        spectrogram is generated using one of three types: Mel, FFT, or Hilbert.

        Args:
            spec_type (str): Type of the spectrogram ('mel', 'fft', 'hilbert'). Default is 'mel'.
        
        Returns:
            Spectrogram: A Spectrogram instance containing the generated spectrogram.
        """
        from stridulant.spectrogram import Spectrogram
        
        n_fft = 256  
        hop_length = n_fft // 2  
        window = 'boxcar' 
        
        if spec_type == 'hilbert':
            analytic_signal = hilbert(self.audio)
            amplitude_envelope = np.abs(analytic_signal)

            spectrogram_data = amplitude_envelope
            return Spectrogram(spectrogram_data, self.sr, self.start_time, self.transformed, self.normalized, spec_type='hilbert')
        else:
            if spec_type == 'mel':
                max_freq = self.sr / 2
                S = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=128, fmax=max_freq, center = False)
            else:  # FFT
                S = np.abs(librosa.stft(self.audio, center = False, n_fft = n_fft, hop_length = hop_length, window = window))**2     
           
            S_dB = librosa.power_to_db(S, ref = 1)
            
            return Spectrogram(S_dB, self.sr, self.start_time, self.transformed, self.normalized, spec_type=spec_type)
        
    def normalize(self):
        """
        Normalizes audio snippet.
        
        This method uses the normalize_audio function to normalize the snippet.
        
        Returns:
            None. The snippet itself is altered by the method, the nromalized 
            flag is set to true.
        """
        from stridulant.processing import normalize_audio
        
        self.audio = normalize_audio(self.audio)
        self.normalized = True
    
    # Augmentation Methods
    def add_gaussian_noise(self, min_amplitude=0.001, max_amplitude=0.015):
        """
        Adds Gaussian noise to the audio signal.
    
        This method applies Gaussian noise with random amplitude in the range specified by
        'min_amplitude' and 'max_amplitude'. The noise is added to the audio signal to simulate
        real-world disturbances, which can be useful for data augmentation in machine learning tasks.
    
        Args:
            min_amplitude (float, optional): The minimum amplitude for the Gaussian noise. 
                                              Default is 0.001.
            max_amplitude (float, optional): The maximum amplitude for the Gaussian noise. 
                                              Default is 0.015.
    
        The noise is generated using a uniform distribution within the range [min_amplitude, max_amplitude],
        and is added to the audio file.
    
        After applying the augmentation, the 'transformed' flag is set to True, indicating that the 
        audio has been modified.
    
        Example:
            Audio_snippet.add_gaussian_noise(min_amplitude=0.002, max_amplitude=0.01)
        """
        augmenter = AddGaussianNoise(min_amplitude, max_amplitude, p=1)
        self.audio = augmenter(self.audio, self.sr)  
        self.transformed = True  
        
    def time_stretch(self, min_rate=0.8, max_rate=1.25):
        """
        Applies time stretching to the audio signal.
    
        This method changes the speed of the audio signal by applying time stretching. The audio
        can be slowed down or sped up based on the rate chosen. The rate is determined randomly 
        within the range defined by 'min_rate' and 'max_rate', which are set by default to 
        0.8 (slow down) and 1.25 (speed up).
    
        Args:
            min_rate (float, optional): The minimum rate for time stretching. 
                                         A value below 1 will slow down the audio, while 
                                         a value above 1 will speed it up. Default is 0.8.
            max_rate (float, optional): The maximum rate for time stretching. 
                                         Similar to 'min_rate', but controls the upper limit. Default is 1.25.
    
        The rate is randomly chosen within the range [min_rate, max_rate], and the time-stretched
        audio is returned.
    
        Example:
            audio_snippet.time_stretch(min_rate=0.9, max_rate=1.1)
        """
        augmenter = TimeStretch(min_rate, max_rate, p=1)  
        self.audio = augmenter(self.audio, self.sr)  
        self.transformed = True  
        
    def pitch_shift(self, min_semitones=-4, max_semitones=4):
        """
        Applies pitch shifting to the audio signal.
    
        This method modifies the pitch of the audio by shifting it up or down randomly 
        within the range defined by 'min_semitones' and 'max_semitones'. The pitch is shifted
        by an amount determined within this range, effectively changing the perceived key 
        of the audio. A negative value shifts the pitch down, while a positive value shifts 
        it up.
    
        Args:
            min_semitones (int, optional): The minimum number of semitones to shift the pitch down. 
                                           Default is -4 (shift down by up to 4 semitones).
            max_semitones (int, optional): The maximum number of semitones to shift the pitch up. 
                                           Default is 4 (shift up by up to 4 semitones).
    
        The pitch is randomly shifted within the range [min_semitones, max_semitones], 
        and the pitch-shifted audio is returned.
    
        Example:
            audio_snippet.pitch_shift(min_semitones=-2, max_semitones=2)
        """
        augmenter = PitchShift(min_semitones, max_semitones, p=1)  
        self.audio = augmenter(self.audio, self.sr)  
        self.transformed = True  
    
    def shift(self, min_fraction=-0.5, max_fraction=0.5):
        """
        Shifts the audio signal in time by a random fraction.
    
        This method shifts the audio signal in time, either forward or backward, 
        based on a random fraction within the specified range of 'min_fraction' and 'max_fraction'.
        A negative value of the fraction shifts the audio backward, while a positive value shifts it forward.
    
        Args:
            min_fraction (float, optional): The minimum fraction of the signal to shift backward. 
                                             Default is -0.5, meaning the audio can be shifted back 
                                             by up to 50% of its length.
            max_fraction (float, optional): The maximum fraction of the signal to shift forward. 
                                             Default is 0.5, meaning the audio can be shifted forward 
                                             by up to 50% of its length.
    
        The shift is applied by randomly selecting a fraction in the range [min_fraction, max_fraction], 
        and the shifted audio is returned.
    
        Example:
            audio_snippet.shift(min_fraction=-0.3, max_fraction=0.3)
        """
        augmenter = Shift(min_fraction, max_fraction, p=1)  
        self.audio = augmenter(self.audio,self.sr)  
        self.transformed = True  
    
    def clipping_distortion(self, min_percent=10, max_percent=30):
        """
        Applies clipping distortion to the audio signal by limiting the amplitude to a given range.
    
        This method clips the audio signal by limiting its amplitude to a range determined by 
        a random percentage between 'min_percent' and 'max_percent'. The signal will be distorted 
        by setting values above or below this range to the corresponding limit.
    
        Args:
            min_percent (int, optional): The minimum percentage of the signal's amplitude 
                                           that will be clipped. Default is 10 meaning the 
                                           amplitude can be reduced by up to 10%.
            max_percent (int, optional): The maximum percentage of the signal's amplitude 
                                           that will be clipped. Default is 30, meaning the 
                                           amplitude can be reduced by up to 30%.
    
        The clipping distortion is applied with a random clipping percentage within the range 
        [min_percent, max_percent].
    
        Example:
            audio_snippet.clipping_distortion(min_percent=0.05, max_percent=0.2)
        """
        augmenter = ClippingDistortion(min_percent, max_percent, p=1)
        self.audio = augmenter(self.audio,self.sr)
        self.transformed = True
        
    def add_background_noise(self, background_data, min_background_influence=0.1, max_background_influence=0.3):
        """
        Adds background noise to the audio snippet by mixing it with a background audio file.
    
        This method randomly blends the audio with a background noise signal. The intensity 
        of the background noise is controlled by the 'min_background_influence' and 
        'max_background_influence' parameters, determining how much the background audio 
        affects the original signal.
    
        Args:
            background_data (array-like or None, optional): The background noise to add to the audio.
                                                             A folder of backgroud sounds must be specified.
                                                             Ideally, a folder with negative snippets o the same
                                                             legnth containing only noise.
            min_background_influence (float, optional): The minimum amount of influence the background 
                                                       noise will have on the audio. Default is 0.1.
            max_background_influence (float, optional): The maximum amount of influence the background 
                                                       noise will have on the audio. Default is 0.3.
    
        The background noise is applied with a random influence percentage within the range 
        [min_background_influence, max_background_influence].
    
        Example:
            audio_snippet.add_background_noise(background_data=my_background_noise, min_background_influence=0.05, max_background_influence=0.2)
        """
        augmenter = AddBackgroundNoise(background_data, min_background_influence, max_background_influence, p=1)
        self.audio = augmenter(self.audio, self.sr)
        self.transformed = True

    def time_mask(self, min_band_part=0.1, max_band_part=0.2):
        """
        Applies a time mask to the audio snippet by randomly masking a portion of the audio signal 
        along the time axis.
    
        This method randomly selects a segment of the audio and "masks" it by reducing its amplitude 
        to zero. The segment length is determined by the 'min_band_part' and 'max_band_part' parameters, 
        which specify the proportion of the total duration to be masked.
    
        Args:
            min_band_part (float, optional): The minimum proportion of the audio to be masked.
                                              Default is 0.1 (10% of the audio).
            max_band_part (float, optional): The maximum proportion of the audio to be masked.
                                              Default is 0.2 (20% of the audio).
                                              
        The mask's length is chosen randomly within the range defined by 'min_band_part' and 'max_band_part'.
    
        Example:
            audio_snippet.time_mask(min_band_part=0.05, max_band_part=0.15)
        """
        augmenter = TimeMask(min_band_part, max_band_part, p = 1)
        self.audio = augmenter(self.audio, self.sr)
        self.transformed = True
    
