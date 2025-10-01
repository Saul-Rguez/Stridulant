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
from pynput import keyboard
import librosa
import librosa.display
import soundfile as sf
import sounddevice as sd
import os
import numpy as np
from scipy.signal import hilbert
from scipy import signal
from scipy.ndimage import uniform_filter1d
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
        Plays audio with keyboard interruption using pynput.
        """
        print("Press Esc to stop")
        
        stop_playback = False
        
        def on_press(key):
            nonlocal stop_playback
            try:
                if key == keyboard.Key.esc:
                    print("\nInterrupted")
                    sd.stop()
                    stop_playback = True
                    return False  
            except:
                pass
        
    
        listener = keyboard.Listener(on_press=on_press)
        listener.start()
        
        
        sd.play(self.audio, self.sr)
        
    
        while sd.get_stream().active and not stop_playback:
            sd.sleep(100)
        
        listener.stop()
        sd.wait()
            

    def spectrogram(self, spec_type='mel',n_fft = 256, hop_length = 224, window = "boxcar", n_mels=128, env_smooth = 10):
        """
        Creates and returns a Spectrogram instance based on the current snippet. The 
        spectrogram is generated using one of three types: Mel, FFT, or Hilbert.
        
        The default arguments are the ones used for power analysis as we conduct them. But
        I have made it more flexible because I think it is needed for certain ultrasounds.

        Args:
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
        more smooth lines.
            
        Returns:
            Spectrogram: A Spectrogram instance containing the generated spectrogram.
        """
        from stridulant.spectrogram import Spectrogram
        
        
        if spec_type == 'hilbert':
            analytic_signal = hilbert(self.audio)
            amplitude_envelope = np.abs(analytic_signal)
            envelope_smoothed = uniform_filter1d(amplitude_envelope, size = int(env_smooth/1000 * self.sr))

            spectrogram_data = envelope_smoothed
            return Spectrogram(spectrogram_data, self.sr, self.start_time, self.transformed, self.normalized, spec_type='hilbert')
        else:
            if spec_type == 'mel':
                max_freq = self.sr / 2
                S = librosa.feature.melspectrogram(y=self.audio, sr=self.sr, n_mels=n_mels, fmax=max_freq, center = False, n_fft = n_fft, hop_length = hop_length, window = window)
                spec_times = None
            else:  # FFT
                S = np.abs(librosa.stft(self.audio, center = False, n_fft = n_fft, hop_length = hop_length, window = window))**2
            spec_times = spec_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=self.sr, hop_length=hop_length, n_fft=n_fft)
            S_dB = librosa.power_to_db(S, ref = 1)
            return Spectrogram(S_dB, self.sr, self.start_time, self.transformed, self.normalized, spec_type=spec_type, fft_times = spec_times)
        
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
    

    
    def find_audio_events(self, min_event_duration=0.2, threshold_percentile=25):
        """
        Extracts stridulation features from a specific detected event.
        
        Args:
            pulse_dist (float): Minimum time between pulses in seconds (default: 0.02 = 20ms)
            event_index (int): Index of event to analyze (0 = most energetic)
            
        Returns:
            dict: Dictionary with extracted features
        """
            
        # Calculate energy envelope
        envelope = np.abs(signal.hilbert(self.audio))
        envelope_smoothed = uniform_filter1d(envelope, size=int(0.01 * self.sr))
        
        # Set threshold based on percentile of energy
        energy_threshold = np.percentile(envelope_smoothed, threshold_percentile)
        
        # Find regions above threshold
        above_threshold = envelope_smoothed > energy_threshold
        
        # Group consecutive samples above threshold into events
        events = []
        in_event = False
        event_start = 0
        min_event_samples = int(min_event_duration * self.sr)
        
        for i, is_above in enumerate(above_threshold):
            if is_above and not in_event:
                # Event starts
                in_event = True
                event_start = i
            elif not is_above and in_event:
                # Event ends
                in_event = False
                event_end = i
                
                # Only keep events longer than minimum duration
                if (event_end - event_start) >= min_event_samples:
                    event_energy = np.max(envelope_smoothed[event_start:event_end])
                    events.append((
                        event_start / self.sr,  # Start time in seconds
                        event_end / self.sr,    # End time in seconds  
                        event_energy            # Maximum energy in event
                    ))
        
        return events
    
    def extract_stridulation_features(self, pulse_dist = 0.02, event_index = 0):
        """
        Extracts stridulation features from a specific detected event within the snippet.
        If no event_index specified, analyzes the most prominent event.
        
        Args:
            event_index (int): Index of the event to analyze (default: 0 = most energetic)
            
        Returns:
            dict: Dictionary with extracted features from the specified event
        """

        
        # Detect events in the snippet
        events = self.find_audio_events()
        
        if not events:
            # No events detected, return default features
            return self._get_default_features()
        
        # Sort events by energy (most energetic first) and select the requested one
        events.sort(key=lambda x: x[2], reverse=True)
        
        if event_index >= len(events):
            event_index = 0  # Fall back to most energetic event
            
        event_start, event_end, event_energy = events[event_index]
        
        # Convert times to sample indices
        start_sample = int(event_start * self.sr)
        end_sample = int(event_end * self.sr)
        event_audio = self.audio[start_sample:end_sample]
        
        features = {}
        features['event_start_time'] = event_start
        features['event_duration'] = event_end - event_start
        features['event_energy'] = event_energy
        
        # 1. ENVELOPE ANALYSIS - within the detected event
        if len(event_audio) > 0:
            envelope = np.abs(signal.hilbert(event_audio))
            envelope_smoothed = uniform_filter1d(envelope, size=int(0.01 * self.sr))
            
            # Attack slope within the event (first 20% of event duration)
            attack_window = max(1, int(0.2 * len(envelope_smoothed)))
            if len(envelope_smoothed) > attack_window:
                attack_slope = (np.max(envelope_smoothed[:attack_window]) - 
                               envelope_smoothed[0]) / attack_window
            else:
                attack_slope = 0
            features['attack_slope'] = attack_slope
            
            # 2. PULSE DETECTION - within the event
            peaks, _ = signal.find_peaks(envelope_smoothed, 
                                        height=np.percentile(envelope_smoothed, 70),
                                        distance=int(pulse_dist * self.sr))  
            
            features['pulse_count'] = len(peaks)
            
            if len(peaks) > 1:
                intervals = np.diff(peaks) / self.sr
                features['pulse_regularity'] = 1.0 / (np.std(intervals) + 1e-6)
                features['avg_pulse_interval'] = np.mean(intervals)
                features['pulse_density'] = len(peaks) / features['event_duration']  # pulses per second
            else:
                features['pulse_regularity'] = 0
                features['avg_pulse_interval'] = 0
                features['pulse_density'] = 0
            
             # DUTY CYCLE: measures pulse sustain
            if len(peaks) > 0:
                
                pulse_threshold = np.percentile(envelope_smoothed, 30)
                
                samples_above_threshold = np.sum(envelope_smoothed > pulse_threshold)
                features['duty_cycle'] = samples_above_threshold / len(envelope_smoothed)
                
                pulse_durations = []
                above_pulse = envelope_smoothed > pulse_threshold
                
                in_pulse = False
                pulse_start = 0
                for i, is_above in enumerate(above_pulse):
                    if is_above and not in_pulse:
                        in_pulse = True
                        pulse_start = i
                    elif not is_above and in_pulse:
                        in_pulse = False
                        pulse_durations.append(i - pulse_start)
                
                if pulse_durations:
                    features['avg_pulse_duration'] = np.mean(pulse_durations) / self.sr  # en segundos
                else:
                    features['avg_pulse_duration'] = 0
            else:
                features['duty_cycle'] = 0
                features['avg_pulse_duration'] = 0
            
            # 3. SPECTRAL FEATURES - of the event
            S = np.abs(librosa.stft(event_audio, n_fft=512, hop_length=10, window="hann"))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            
            spectral_centroids = librosa.feature.spectral_centroid(S=S, sr=self.sr)[0]
            features['tonal_variation'] = np.std(spectral_centroids) / (np.mean(spectral_centroids) + 1e-6)
            
            features['dynamic_range'] = np.max(S_db) - np.min(S_db)
            
            # Spectral centroid mean (indicates dominant frequency range)
            features['spectral_centroid_mean'] = np.mean(spectral_centroids)
            
        else:
            # Event too short, return default values
            features.update(self._get_default_features())
        
        return features
    
    def _get_default_features(self):
        """
        Returns default feature values when no event is detected.
        
        Provides consistent output structure for downstream processing
        and prevents errors in feature analysis pipelines.
        """
        return {
            'event_start_time': 0,
            'event_duration': 0,
            'event_energy': 0,
            'attack_slope': 0,
            'pulse_count': 0,
            'pulse_regularity': 0,
            'avg_pulse_interval': 0,
            'pulse_density': 0,
            'duty_cycle': 0,
            'avg_pulse_duration': 0,
            'tonal_variation': 0,
            'dynamic_range': 0,
            'spectral_centroid_mean': 0
        }
    
    def is_promising_stridulation(self, min_pulses = 6, min_regularity = 10, min_duration = 0.5, max_duration = 0.8, sustain = 0.5, sp_range = (5500, 15000)):
        """
        Evaluates if the audio snippet contains a promising stridulation signal based on 
        acoustic features. Analyzes all detected events and returns True if ANY event 
        meets all the stridulation criteria.
        
        Args:
            min_pulses (int): Minimum number of individual pulses required within the event.
                              Typical insect stridulations have 6+ distinct pulses.
                              
            min_regularity (float): Minimum pulse regularity score (1/standard_deviation of intervals).
                                   Higher values indicate more consistent timing between pulses.
                                   Values >10 suggest rhythmic, organized patterns.
                                   
            min_duration (float): Minimum event duration in seconds. Filters out very brief 
                                 noises that are unlikely to be biological signals.
                                 
            max_duration (float): Maximum event duration in seconds. Filters out very long 
                                 continuous sounds that may be environmental noise.
                                 
            sustain (float): Minimum duty cycle ratio (0-1) indicating what fraction of the 
                            event duration contains actual sound vs silence. 
                            - Values near 0.2-0.4: Pulsed sounds with clear gaps
                            - Values near 0.6-0.8: More continuous, sustained sounds
                            - Helps distinguish true stridulations from isolated clicks
                            
            sp_range (tuple): Valid frequency range for spectral centroid in Hz (min, max).
                             Filters events based on dominant frequency content.
                             Typical insect stridulations fall within 5500-15000 Hz.
        
        Returns:
            bool: True if ANY detected event meets ALL stridulation criteria, False otherwise.
        """
        events = self.find_audio_events()
        
        if not events:
            return False
        
        for i, event in enumerate(events):
            features = self.extract_stridulation_features(event_index=i)

            if features['event_duration'] < min_duration or features['event_duration'] > max_duration:
                continue
            
            if (features['pulse_count'] >= min_pulses and
                features['pulse_regularity'] > min_regularity and  
                sp_range[0] < features['spectral_centroid_mean'] < sp_range [1] and
                features['duty_cycle'] > sustain):
                return True 
        
        return False
    
    
    


