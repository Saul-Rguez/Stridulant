# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 12:22:58 2026

"""

import os
from stridulant.processing import create_snippet
from stridulant.audio_snippet import AudioSnippet
from stridulant.spectrogram import Spectrogram
from stridulant.utils import load_audio
import pandas as pd

def process_table(input_table = None,
              output_dir = None, 
              delim = ",",
              audio = None,
              make_snippets = True,
              start_time_column = "Begin Time (s)",
              end_time_column = "End Time (s)",
              snippet_duration = None,
              ids = ["Selection"],
              
              make_spectrograms = True,
              spec_type='mel',
              n_fft = 256,
              hop_length = 224, 
              window = "boxcar", 
              n_mels=128, 
              env_smooth = 10,
              color = "inferno", 
              with_labels=True, 
              verbose=True, 
              metadata = True):
    """
    This function reads a selection or annotation table. The default is set for selection tables made in Raven.
    However, any table including a start and end time column plus up to 3 identifying columns can be imported.
    
    args: 
    input_table: path to a flat table in formats such as csv and txt
    output_dir: where to save snippets and spectrogram folders.
    delim: symbol used to seperate columns
    audio: path to audio
    make_snippets: if True, will save snippets
    start_time_column: the name of the column containing event start times
    end_time_clumn: the name of the column containing event end times.
    snippet_duration: if specefied, snippets will all be the same length. The start and end time columns will be
        used to find the center of the event, on which the snippet will be centered.
    ids: Identifying columns, such as treatment, channel, selection, etc.
    
    make_spectrograms: if True, spectrograms will be saved.
    spec_type (str): Type of the spectrogram ('mel', 'fft', 'hilbert'). Default is 'mel'.
    n_fft (int): this is the size of the window analised. In a nutshell,bigger numbers will
        increase resolution in frequency and smaller numbers increase resolution in time. This is a lot
        more efficient if the number is a power of 2.
    hop_length (int): overlap between windows. Bigger numbers (up tho the window size) will make
        the process faster at the expense of temporal resolution, but also will make windows more independent
        from each other. Big numbers are best for power analysis, smaller numbers are better for very short
        sounds and high temporal reolution (visualization and perhaps CNNs)
    window (str): shape of the window of the fft. Windows are related to leakage, or which is a "spill"
        of energy from a frequency bin to the adyacent one. The default is boxcar, which has a lot of leakage, 
        but also the best frequency reolution. We have been using that one for power analysis, but others with
        less leakage are possible. Check Librosa documentation to know your options.
    n_mels (int): number of mels for mel spectrograms. Higher numbers increase the resolution in the
        frequency domain. Best with powers of 2
    env_smooth (float): smooth factor for the Hilbert emvelope. It is a number in miliseconds that smooths 
        the peaks on that time range. Often numbers around 10 or so give a good trade-off betwee the smoothness
        of the line and the retention of features. Smaller numbers will give higher details, bigger numbers 
        smoother lines. These numbers operate with the sampling rate, for ultrasounds you probably want to go 
        smaller, like 1 or even 0.1. Just make sure that int(env_smooth/1000*sr)>0. You can check the sr of your
        audio when you load it, you'll get sr that you can print, or within the snippet, with snippet.sr
    
    """
    
# --- Loading table and audio. Extracting file name. --- #    
    table = pd.read_csv(input_table, delimiter=delim)
    file_name = os.path.basename(audio)
    file_name = file_name.rsplit( ".", 1 )[ 0 ] 
    audio, sr = load_audio(audio)
    print("Audio file found and loaded.")
    
# --- Looping over each row of the table --- #
    for index, row in table.iterrows():

# --- Setting name according to id columns that are specified --- #
        name = file_name
        for col in ids:
            if col:
                name += f"_{col}_{row[col]}"
                
# --- Deciding on where to start the snippet according to whether a snippet duration is defined --- #             
        if snippet_duration:
            duration = row[end_time_column]-row[start_time_column]
            middle = row[start_time_column]+duration/2
            start = middle - snippet_duration/2
            start = max(0, start)
            snippet = create_snippet(audio, sr, start, snippet_duration)
        else:
            snippet_duration = row[end_time_column]-row[start_time_column]
            snippet = create_snippet(audio, sr, row[start_time_column], snippet_duration)

# --- Make and save snippet --- #                
        if make_snippets:
            snippet.save(source_name = name, output_dir = os.path.join(output_dir, f"Snippets_{file_name}"), verbose=False)
            
# --- Make and save spectrogram --- #
        if make_spectrograms:
            spectrogram = snippet.spectrogram(spec_type = spec_type,
                                n_fft = n_fft,
                                hop_length = hop_length, 
                                window = window, 
                                n_mels = n_mels, 
                                env_smooth = env_smooth)
            spectrogram.save_img(
                                name,
                                output_dir=os.path.join(output_dir, f"Spectrograms_{file_name}"),
                                color=color,
                                with_labels=with_labels,
                                verbose=verbose,
                                metadata=metadata
                            )
                                        
    print("Done reading annotation table and saving snippets and/or spectrograms.")
        
        
