# -*- coding: utf-8 -*-
"""
Module: Batch process

Description:
    This is a module created to loop over a folder and all subfolders. 
    It creates a csv log with the results of the choosen search function (i.e. is_stridulation or is_cavitation, or user-defined).
    It is designed to handle interruption. After restart, the function will read the snippet log and start processing accordingly.

Date: 2026-01-07

"""
import os
import glob
from tqdm import tqdm
import librosa
import numpy as np
import random
from stridulant.utils import load_audio
from stridulant.processing import create_snippet
from stridulant.processing import highpass_filter
from stridulant.processing import lowpass_filter
from stridulant.processing import normalize_audio

def batch(method = None,
          snippet_duration = 10, 
          output_folder = None,
          folder_list = None,
          phrase_includes = [],
          phrase_excludes = [],

          # Event finder options
          min_event_duration=0.1,
          threshold_percentile=25,
          env_smooth=10,          
          
          # Normalization options
          normalize = None,
          target_max = 0.75,
          
          # Filter options
          highpass_filter_value = False,
          lowpass_filter_value = False,
          highpass_order = 8,
          lowpass_order = 8,         
          
          # Snippet log options
          log_name = "snippet_log",
          log_save = "all",
          
          # Spectrogram options
          make_spectrograms = False,
          with_labels= True,
          spec_type = "mel",
          n_fft = 128, 
          n_mels = 15,
          hop_length = 10, 
          window = "boxcar",
          
          # Options for saving snippets
          make_snippets = "none",
          metadata = True,
         
          **method_kwargs):
    """
    Creates a snippet log.
    
    Args:
        method: is_cavitation, is_stridulation or user defined (e.g., is_worm_rumble from the tutorials). 
            Can also be set to None to save all events without catagorizing them into postitives and negatives.
        snippet_duration: length of chunks used to detect events in.
        output_folder: where to store the log and optional snippets and spectrograms.
        folder_list: list of folders to gather audiofiles from.
        phrase_inludes/phrase_excludes: wether to in/exclude certain files based on phrases in their names. 
            For example, you might filter out files containing the phrase "test",
            or you might include only files included the phrase "control".
        
        min_event_duration: the minimum amount of seconds an event needs to be to be detected.
        threshold_percentile (float): Percentile value for energy threshold (0-100) (default: 25)
        env_smooth (float): smooth factor for the Hilbert emvelope. It is a number in miliseconds that smooths 
        the peaks on that time range. Often numbers around 10 or so give a good trade-off between the smoothness
        of the line and the retention of features. Smaller numbers will give higher details, bigger numbers 
        smoother lines. These numbers operate with the sampling rate, for ultrasounds you probably want to go 
        smaller, like 1 or even 0.1. Just make sure that int(env_smooth/1000*sr)>0. You can check the sr of your
        audio when you load it, you'll get sr that you can print, or within the snippet, with snippet.sr
        
        normalize: None for no normalization, "snippet" for normalizing within snippets,
            "file" for normalizing within each file and "global" for normalizing across folders and subfolders.
        target_max (float): max percentage for the peak value (default 75%)
        
        highpass_filter_value: Cutoff frequency in Hz. Default is 3000 Hz.
        highpass_order (int): Filter order. Default is 8 (48 dB/octave roll-off).
        lowpass_filter_value (float): Cutoff frequency in Hz. Default is 5000 Hz.
        lowpass_order (int): Filter order. Default is 8 (48 dB/octave roll-off).
        
        log_name: name of snippet log.
        log_save: "all" for positive and negative results, or "true_only" and "false_only" as the names imply.
            "true_only" does not add zeros for completly empty audiofiles.
        
        make_spectrograms: set to True if you wish to save spectrograms of each of the events that is found. Will be sorted in "positive" and "negative folders".
            If method is set to none, all spectrograms will be saved together in one folder.
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
            frequency domain. Best with powers of 2.
        
        make_snippets: Saves snippets of events. Options are `all' for both negative and positive snippets, 
            `true_only' and `false_only' for only saving positive or negative snippets, 
            and `random' for a random 1% of snippets. Default is False.
        metadata: adds labels to the snippets with event start time. 
            
        **method_kwargs collects any kwargs stemming from user-defined search functions.
    """
# --- Find or create log file ---

    log_file = os.path.join(output_folder, f"{log_name}.csv")

    resume_file = None
    resume_start = 0
        
    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Original_audio_path,Snippet_start_time, Snippet_end_time,Event,Result\n")
        print("New snippet log created.")
    else:
        print(f"Existing log found. Resuming from last entry: {log_file}")
        with open(log_file, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if len(lines) > 1:  
            last_line = lines[-1]
            parts = last_line.split(",")

            if len(parts) >= 5:
                resume_file, Snippet_last_start, Snippet_last_end, Event_binary, Result = parts
                try:
                    last_start = float(Snippet_last_start)
                    resume_start = last_start
                    print(f"Resuming from file {resume_file}, start = {resume_start} seconds")
                except Exception as e:
                    print(f"Failed to parse last log entry, starting fresh: {e}")
            else:
                print("Log file exists but wrong number of colums. Aborting.")
        else:
            print("Log file exists but empty (header only). Starting from scratch.")
            
# --- Gather files ending with wav or flac ---

    all_audio_files = []
    for folder in folder_list:
        all_audio_files.extend(
            f for f in glob.glob(os.path.join(folder, "**", "*"), recursive=True)
            if f.lower().endswith((".wav", ".flac"))
            )
        
# --- Filter files by including and excluding phrases ---
    filtered_files = [
        f for f in all_audio_files
        if any(t in f for t in phrase_includes)
        and not any(t in f for t in phrase_excludes)
    ]

    filtered_files.sort(key=str.lower, reverse=False)
    print(f"Remaining files to be processed: {filtered_files}")
    
    file_paths = []    
 
# Find global max #
    if normalize == "global":    
        global_max = 0
        with tqdm(total=len(filtered_files), desc="Finding global max", unit="file", ncols=100, position=0, leave=True) as pbar:
            for file in filtered_files:
                file_paths.append(file)
                audio, _ = librosa.load(file, sr = None)
                max_value = np.max(np.abs(audio))
                global_max = max(global_max, max_value)
                pbar.update(1)
                
        print(f"Global max is {global_max}.")        
        if global_max == 0:
            raise ValueError("All files are silent or empty. Normalization is not possible.")
             
 # --- Process files --- 
 
    if resume_file is None and resume_start == 0:
        start_index = 0
    else:
        start_index = filtered_files.index(resume_file)

# --- Start loop ---

    for file_path in filtered_files[start_index:]:
        print(f"Processing: {file_path}")
        log_f = open(log_file, "a") 
        try:
            
# --- Create a base name ---
            parts = os.path.normpath(file_path).split(os.sep)
            base_name = "_".join(parts)
            base_name = os.path.splitext(base_name)[0]
             
# --- Load audio, apply filters and normalization if defined ---       
            audio, sr = load_audio(file_path)

            if highpass_filter_value is not False:
                audio = highpass_filter(audio, sr, cutoff = highpass_filter_value, order = highpass_order)
                print("Highpass filter added")
            if lowpass_filter_value is not False:
                audio = lowpass_filter(audio, sr, cutoff = lowpass_filter_value, order = lowpass_order)
                print("Lowpass filter added")
            if normalize == "file":
                audio = normalize_audio(audio, target_max)
            if normalize == "global":
                audio = normalize_audio(audio, target_max, global_max)
                
# --- Calculate the number of snippets to be made in the file. Skips if (remainder of) the file is too short. ---
         
            total_duration = len(audio) / sr 
            num_snippets = int(total_duration // snippet_duration) # How many times the snippet duration fits in the audio file length.
            print(f"Duration: {total_duration:.1f}s ({num_snippets} snippets of {snippet_duration}s)")
            if total_duration < snippet_duration:
                print("Audio file too short (snippet duration does not fit in the length of the audio file).")
                log_f.write(f"{file_path},NA,NA,NA\n")
                continue
            if num_snippets == 0:
                print("Audio file too short (snippet duration does not fit in the length of the audio file).")        

# --- Calculate where in the file to resume ---

            snippet_start_index = 0
            if resume_file and os.path.normcase(file_path) == os.path.normcase(resume_file):
                snippet_start_index = int(resume_start // snippet_duration)
                if snippet_start_index >= num_snippets:
                    print(f"Resume point ({resume_start}s) beyond file duration, skipping {file_path}")
                    continue
                print(f"Resuming inside file at snippet index {snippet_start_index} ")
    
# --- Loop over (the remainder of) the file --- 
   
            for i in range(snippet_start_index, num_snippets):
                start_time = i * snippet_duration
                end_time = start_time + snippet_duration

# --- Create snippet name ---
         
                snippet_name = f"{base_name}_{start_time}_"

# --- Create snippet and find events in snippet, normalize if defined ---

                Snippet = create_snippet(audio, sr, start_time, snippet_duration)
                if normalize == "snippet":
                    Snippet.normalize()
                events = Snippet.find_events(
                    min_event_duration=min_event_duration,
                    threshold_percentile=threshold_percentile,
                    env_smooth=env_smooth)

# --- Create binary event results: "Yes" if event(s) are found, "No" if snippet is empty ---
                if not events:
                    event_binary = "No"
                    if log_save == "all":
                        log_f.write(f"{file_path},{start_time},{end_time},{event_binary},NA\n")
                        continue
                if events: 
                    event_binary = "Yes"

# --- Run method on snippet ---
         
                if method is None:
                    result = "NA"
                
                # Built-in snippet method passed as string
                if isinstance(method, str):

                    if not hasattr(Snippet, method):
                        raise ValueError(f"Stridulant has no method named '{method}'. Use 'is_stridulation' or 'is_cavitation'. If you are attempting to use a user-defined function, make sure it is loaded and to not use quotation marks.")

                    result = getattr(Snippet, method)(**method_kwargs)

                if callable(method):

                    result = method(Snippet, **method_kwargs)
         
# --- Save results ---

                if log_save == "all":
                    log_f.write(f"{file_path},{start_time},{end_time},{event_binary},{result}\n")
                
                if log_save == "true_only" and result:
                        log_f.write(f"{file_path},{start_time},{end_time},{event_binary},{result}\n")
                
                if log_save == "false_only" and not result:
                        log_f.write(f"{file_path},{start_time},{end_time},{event_binary},{result}\n")                    
            
                if make_snippets in ("all", "true_only") and result is True:
                        output_snippet_true = os.path.join(output_folder, "Results", "True_snippets")
                        os.makedirs(output_snippet_true, exist_ok=True)
                        Snippet.save(snippet_name, output_snippet_true, metadata=metadata)        
                       
                if make_snippets in ("all", "false_only") and result is False:
                        output_snippet_false = os.path.join(output_folder, "Results", "False_snippets")
                        os.makedirs(output_snippet_false, exist_ok=True)
                        Snippet.save(snippet_name, output_snippet_false, metadata=metadata)
                    
                if make_snippets == "all" and result == "NA":
                        output_snippet_false = os.path.join(output_folder, "Results", "Snippets")
                        os.makedirs(output_snippet_false, exist_ok=True)
                        Snippet.save(snippet_name, output_snippet_false, metadata=metadata)  
                    
                if make_snippets == "random" and result is True:
                        if random.random() < 0.01:
                            output_snippet_true = os.path.join(output_folder, "Results", "True_snippets")
                            os.makedirs(output_snippet_true, exist_ok=True)
                            Snippet.save(snippet_name, output_snippet_true, metadata=metadata)        

                if make_snippets == "random" and result is False:
                        if random.random() < 0.01:
                            output_snippet_true = os.path.join(output_folder, "Results", "False_snippets")
                            os.makedirs(output_snippet_true, exist_ok=True)
                            Snippet.save(snippet_name, output_snippet_true, metadata=metadata)        
                    
                if make_spectrograms:
                    if result is True:
                        output_img_true = os.path.join(output_folder, "Results", "True_spectrograms")
                        os.makedirs(output_img_true, exist_ok=True)
                        spectrogram = Snippet.spectrogram(spec_type=spec_type, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, window = window)
                        spectrogram.save_img(snippet_name, output_img_true, metadata=metadata, with_labels=with_labels)
                        
                    if result is False:
                        output_img_false = os.path.join(output_folder, "Results", "False_spectrograms")
                        os.makedirs(output_img_false, exist_ok=True)
                        spectrogram = Snippet.spectrogram(spec_type=spec_type, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, window = window)
                        spectrogram.save_img(snippet_name, output_img_false, metadata=metadata, with_labels=with_labels)
                        
                    if result == "NA":
                        output_img_true = os.path.join(output_folder, "Results", "Spectrograms")
                        os.makedirs(output_img_true, exist_ok=True)
                        spectrogram = Snippet.spectrogram(spec_type=spec_type, n_fft=n_fft, n_mels=n_mels, hop_length=hop_length, window = window)
                        spectrogram.save_img(snippet_name, output_img_true, metadata=metadata, with_labels=with_labels)                            
               
            log_f.flush()                            
        finally:
            log_f.close()
    print("All files have been processed. Please see the log for your results.")
