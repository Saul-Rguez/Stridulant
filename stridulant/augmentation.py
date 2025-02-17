# -*- coding: utf-8 -*-
"""
Module for augmenting audio data using different transformations.

Created on Sat Feb 15 20:56:06 2025
@author: Saul

This module provides several audio augmentation functions, including:
- Time stretching
- Time shifting
- Pitch shifting
- Adding white noise
- Time warping

These transformations are applied randomly to create variations of a given audio snippet.
The module also includes functionality to generate and save these variations to a specified directory.

Usage:
- normalize_audio(y): Normalizes the audio between -1 and 1.
- time_stretch(y, rate_range=(0.8, 1.25)): Applies time stretching to the audio.
- time_shift(y, max_shift=0.1): Shifts the audio in time (rolls the audio).
- pitch_shift(y, sr, n_steps_range=(-3, 3)): Shifts the pitch of the audio.
- add_white_noise(y, noise_level=0.01): Adds white noise to the audio.
- time_warp(y, max_warp_factor=0.2): Applies time warping to the audio.
- generate_variations(audio_snippet, num_variations=10): Generates multiple variations of the audio.
- save_variations(audio_snippet, output_dir, num_variations=10): Saves the generated variations to a directory.

Autor: Saul Rodriguez Martinez
Fecha de creación: 2025-02-15

"""

import numpy as np
import librosa
import random
from scipy.signal import resample
from stridulant.audio_snippet import AudioSnippet

def normalize_audio(y):
    """
    Normalizes the audio, clipping values to the range -1 to 1.
    
    Parameters:
    y (array): The audio to normalize.
    
    Returns:
    array: The normalized audio.
    """
    return np.clip(y, -1, 1)

def time_stretch(y, rate_range=(0.8, 1.25)):
    """
    Applies random time stretching to the audio.
    
    Parameters:
    y (array): The audio to transform.
    rate_range (tuple): The range of random values for the stretching rate.
    
    Returns:
    array: The stretched audio.
    """
    rate = np.random.uniform(rate_range[0], rate_range[1])
    return normalize_audio(librosa.effects.time_stretch(y, rate=rate))

def time_shift(y, max_shift=0.1):
    """
    Shifts the audio in time (rolls the audio).
    
    Parameters:
    y (array): The audio to transform.
    max_shift (float): The maximum shift in fractions of the audio's length.
    
    Returns:
    array: The time-shifted audio.
    """
    shift = int(np.random.uniform(-max_shift, max_shift) * len(y))
    return normalize_audio(np.roll(y, shift))

def pitch_shift(y, sr, n_steps_range=(-3, 3)):
    """
    Randomly shifts the pitch of the audio.
    
    Parameters:
    y (array): The audio to transform.
    sr (int): The sample rate of the audio.
    n_steps_range (tuple): The range of random pitch shift steps.
    
    Returns:
    array: The audio with pitch shifted.
    """
    n_steps = np.random.uniform(n_steps_range[0], n_steps_range[1])
    return normalize_audio(librosa.effects.pitch_shift(y, sr, n_steps=n_steps))

def add_white_noise(y, noise_level=0.01):
    """
    Adds white noise to the audio.
    
    Parameters:
    y (array): The audio to transform.
    noise_level (float): The noise level to add.
    
    Returns:
    array: The audio with added white noise.
    """
    noise = np.random.normal(0, noise_level, y.shape)
    return normalize_audio(y + noise)

def time_warp(y, max_warp_factor=0.2):
    """
    Applies time warping to the audio.
    
    Parameters:
    y (array): The audio to transform.
    max_warp_factor (float): The maximum warping factor.
    
    Returns:
    array: The time-warped audio.
    """
    warp_factor = np.random.uniform(-max_warp_factor, max_warp_factor)
    new_length = int(len(y) * (1 + warp_factor))
    warped = resample(y, new_length)
    return normalize_audio(warped[:len(y)]) if len(warped) >= len(y) else normalize_audio(np.pad(warped, (0, len(y)-len(warped))))

def generate_variations(audio_snippet, num_variations=10):
    """
    Generates multiple variations of the audio using a random combination of transformations.
    
    Parameters:
    audio_snippet (AudioSnippet): The original audio object to transform.
    num_variations (int): The number of variations to generate.
    
    Returns:
    list: A list of AudioSnippet objects with the generated variations.
    """

    transformations = [time_stretch, 
                       time_shift, 
                       lambda y: pitch_shift(y, audio_snippet.sr), 
                       add_white_noise, 
                       time_warp]
    variations = []

    for _ in range(num_variations):
        y_var = audio_snippet.audio.copy() 
        selected = random.sample(transformations, k=random.randint(1, len(transformations))) 

        for transform in selected:
            y_var = transform(y_var)

        variations.append(AudioSnippet(y_var, audio_snippet.sr, audio_snippet.start_time))
    
    return variations

def create_variations(audio_snippet, output_dir, num_variations=10):
    """
    Generates and saves variations of an audio to an output directory.
    
    Parameters:
    audio_snippet (AudioSnippet): The original audio object to transform.
    output_dir (str): The directory where to save the variations.
    num_variations (int): The number of variations to generate and save.
    """
    variations = generate_variations(audio_snippet, num_variations)
    for idx, var in enumerate(variations):
        var.save(f"{output_dir}/{audio_snippet.name}_variation_{idx+1}.wav")
