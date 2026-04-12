# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 10:12:15 2026

"""
from stridulant.utils import load_audio
from stridulant.processing import highpass_filter, lowpass_filter, create_snippet
from stridulant.audio_snippet import AudioSnippet
import os
import pandas as pd
import glob

"""
This function uses find_events() and extract_features() from AudioSnippet and uses those 
functions to save all events of a sigular snippet or audio file (feature_finder()) or from 
a whole folder of audio files (batch_feature_finder()).

"""

def feature_finder(
          output_folder = None,
          output_name = "features",
          audio = None,
          highpass_filter_value = False,
          lowpass_filter_value = False,
          highpass_order = 8,
          lowpass_order = 8,
          min_event_duration=None,
          threshold_percentile=None,
          pulse_dist=20,
          env_smooth=None
          ):
    """

    Parameters
    ----------
    output_folder : where to store the csv file.
    
    output_name : name of the csv file. The default is "features".
    
    audio : the audio to scan for events. Can be either a snippet or a file path. 
        not recommended to be longer than 10 seconds.
    
    highpass_filter_value: Cutoff frequency in Hz. Default is 3000 Hz.
    
    highpass_order (int): Filter order. Default is 8 (48 dB/octave roll-off).
    
    lowpass_filter_value (float): Cutoff frequency in Hz. Default is 5000 Hz.
    
    lowpass_order (int): Filter order. Default is 8 (48 dB/octave roll-off).
    min_event_duration: the minimum amount of seconds an event needs to be to be detected.
    
    threshold_percentile (float): Percentile value for energy threshold (0-100) (default: 25)
    
    env_smooth (float): smooth factor for the Hilbert emvelope. It is a number in miliseconds that smooths 
        the peaks on that time range. Often numbers around 10 or so give a good trade-off between the smoothness
        of the line and the retention of features. Smaller numbers will give higher details, bigger numbers 
        smoother lines. These numbers operate with the sampling rate, for ultrasounds you probably want to go 
        smaller, like 1 or even 0.1. Just make sure that int(env_smooth/1000*sr)>0. You can check the sr of your
        audio when you load it, you'll get sr that you can print, or within the snippet, with snippet.sr
    
    pulse_dist (float): Minimum time between detectable pulses in milliseconds. 
                       Must be > 0. Controls pulse detection sensitivity:
                       - Lower values (5-10ms): Detect rapid pulses, risk false positives
                       - Higher values (20-30ms): More conservative, may miss fast sequences
                       Typical insect stridulations work well with 10-20ms.


    Returns
    -------
    A csv file with all events and their features.
    
    """

    if isinstance(audio, str):
        audio, sr = load_audio(audio)
        total_duration = len(audio) / sr
        snippet = create_snippet(audio, sr, 0, total_duration)
    else:
        snippet = audio
        sr = audio.sr

    if highpass_filter_value :
            snippet.audio = highpass_filter(snippet.audio, sr, cutoff = highpass_filter_value, order = highpass_order)
    if lowpass_filter_value :
            snippet.audio = lowpass_filter(snippet.audio, sr, cutoff = lowpass_filter_value, order = lowpass_order)
                
# Finding events #
    events = snippet.find_events(min_event_duration=min_event_duration, threshold_percentile=threshold_percentile, env_smooth = env_smooth)

# Extract features from all events #        
    features = pd.DataFrame()
    if events:
        for event in events:
            features_list = snippet.extract_features(event, pulse_dist, env_smooth)
            df = pd.DataFrame(features_list, index=["event"]).T

# Merge into main dataframe
            features = pd.concat([features, df], axis=1)
        
    else:
        print("No events in audio file found. Perhaps try different values for threshold percentile, minimum duration and env_smooth.")
    output_path = os.path.join(output_folder, f"{output_name}.csv")
    features.to_csv(output_path, index=True)
    print("CSV file with features exported.")


def batch_feature_finder(
          output_folder = None,
          snippet_folder = None,
          snippet_folder_list = None,
          highpass_filter_value = False,
          lowpass_filter_value = False,
          highpass_order = 8,
          lowpass_order = 8,
          min_event_duration=None,
          threshold_percentile=None,
          env_smooth=None,
          output_name = "features",
          df_format = "wide",
          pulse_dist=20):
    
    """
    
    Parameters
    ----------
    output_folder : where to store the csv file.
    
    output_name : name of the csv file. The default is "features".
    
    snippet_folder : file path with snippets to extract features from.
    
    snippet_folder_list : Input in case multiple folders need to be processed.
    
    highpass_filter_value: Cutoff frequency in Hz. Default is 3000 Hz.
    
    highpass_order (int): Filter order. Default is 8 (48 dB/octave roll-off).
    
    lowpass_filter_value (float): Cutoff frequency in Hz. Default is 5000 Hz.
    
    lowpass_order (int): Filter order. Default is 8 (48 dB/octave roll-off).
    min_event_duration: the minimum amount of seconds an event needs to be to be detected.
    
    threshold_percentile (float): Percentile value for energy threshold (0-100) (default: 25)
    
    env_smooth (float): smooth factor for the Hilbert emvelope. It is a number in miliseconds that smooths 
        the peaks on that time range. Often numbers around 10 or so give a good trade-off between the smoothness
        of the line and the retention of features. Smaller numbers will give higher details, bigger numbers 
        smoother lines. These numbers operate with the sampling rate, for ultrasounds you probably want to go 
        smaller, like 1 or even 0.1. Just make sure that int(env_smooth/1000*sr)>0. You can check the sr of your
        audio when you load it, you'll get sr that you can print, or within the snippet, with snippet.sr
    
    pulse_dist (float): Minimum time between detectable pulses in milliseconds. 
                               
    df_format: whether to make a "long" or "wide" dataframe.


    Returns
    -------
    A csv file with all events and their features.
    
    """

    if snippet_folder:
        all_audio_files = [
        f for f in glob.glob(os.path.join(snippet_folder, "**", "*"), recursive=True)
        if f.lower().endswith((".wav", ".flac"))
        ]
        
    if snippet_folder_list: 
        all_audio_files = []
        for folder in snippet_folder_list:
            all_audio_files.extend(
                f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
                if f.lower().endswith((".wav", ".flac"))
                )
            
    #print(snippet_folder)
    #print(all_audio_files)

    features = pd.DataFrame()

    for path in all_audio_files:
        audio, sr = load_audio(path)
        total_duration = len(audio) / sr
           
# Create a snippet #
        snippet = create_snippet(audio, sr, 0, total_duration)
        
        if highpass_filter_value :
            snippet.audio = highpass_filter(snippet.audio, sr, cutoff = highpass_filter_value, order = highpass_order)
        if lowpass_filter_value :
            snippet.audio = lowpass_filter(snippet.audio, sr, cutoff = lowpass_filter_value, order = lowpass_order)
        
# Finding events #
        events = snippet.find_events(min_event_duration=min_event_duration, threshold_percentile=threshold_percentile, env_smooth = env_smooth)

# Extract features from the highest energy event#
        events.sort(key=lambda x: x[2], reverse = True)
        if events:
            features_list = snippet.extract_features(events[0], pulse_dist, env_smooth)
            df = pd.DataFrame(features_list, index=[path]).T

    # Merge into main dataframe
            features = pd.concat([features, df], axis=1)
            
        else:
            continue
    if df_format == "long":
        features = features.T
    output_path = os.path.join(output_folder, f"{output_name}.csv")
    features.to_csv(output_path, index=True)
    print("All features exported to CSV.")


