# Stridulant 🦗🎵

> **Professional acoustic analysis toolkit for bioacoustics, cavitation detection, and machine learning**

Stridulant is a comprehensive Python package for audio analysis, specializing in **insect stridulation detection**, **ultrasonic cavitation monitoring**, and **spectrogram-based deep learning**. It provides a complete pipeline from audio loading to feature extraction, data augmentation, and CNN classification.

## 🎯 Key Applications

- **Bioacoustics**: Automatic detection of stridulation patterns (insects, birds, etc.)
- **Ultrasonic Engineering**: Cavitation event detection in liquids
- **Audio ML**: Complete data augmentation + feature extraction + training pipeline
- **Spectrogram Analysis**: Mel, FFT, and Hilbert envelope spectrograms

## ✨ Features

### Core Audio Processing
- **AudioSnippet class** - Segment-based audio manipulation
- **Event detection** - Energy-based event finding with Hilbert envelope
- **Feature extraction** - 13+ acoustic features (pulse count, duty cycle, spectral centroid, etc.)
- **Audio filtering** - High-pass and low-pass Butterworth filters
- **Normalization** - Peak normalization (individual or global across files)

### Data Augmentation (8 methods)
- Gaussian noise addition
- Time stretching
- Pitch shifting
- Time shifting
- Clipping distortion
- Background noise mixing
- Time masking

### Spectrogram Generation
- **Mel spectrograms** - Perceptually scaled frequency representation
- **FFT spectrograms** - Full spectral analysis
- **Hilbert transform** - Amplitude envelope + instantaneous frequency

### Machine Learning Pipeline
- **CNN training** - Binary classifier for stridulation detection
- **Model inference** - Classify spectrograms with trained model
- **Class imbalance handling** - Automatic class weight calculation
- **Training history logging** - CSV export of loss/accuracy

### High-Level Scanners
- `stridulation_scan()` - Scan entire audio files for stridulation events
- `cavitation_scan()` - Scan for cavitation bursts
- **Sliding window** - Configurable overlap and snippet duration
- **Automatic organization** - Creates folders with snippets and spectrograms of candidates

## 📦 Installation

```bash
pip install stridulant
```

### Dependencies

Stridulant requires:
- Python >= 3.7
- TensorFlow (optional, only needed for ML features)

## 🚀 Quick Start

### Basic audio loading and analysis

```python
from stridulant import load_audio, AudioSnippet

# Load an audio file
audio, sr = load_audio("cricket_recording.wav")

# Create a snippet (2 seconds starting at 5.0s)
from stridulant.processing import create_snippet
snippet = create_snippet(audio, sr, start_time=5.0, duration_sec=2.0)

# Detect stridulation
result = snippet.is_stridulation()
if result:
    print(f"Stridulation detected! {result['pulse_count']} pulses")
    print(f"Frequency range: {result['spectral_centroid_mean']:.0f} Hz")
```

### Spectrogram generation

```python
# Generate Mel spectrogram
mel_spec = snippet.spectrogram('mel', n_mels=128)
mel_spec.save_img("cricket", "./spectrograms", color="inferno")

# Generate Hilbert envelope
hilbert_spec = snippet.spectrogram('hilbert', env_smooth=10)
hilbert_spec.save_table("cricket", "./tables")  # Export as CSV
```

### Data augmentation for ML

```python
# Apply augmentations
snippet.add_gaussian_noise(min_amplitude=0.001, max_amplitude=0.015)
snippet.time_stretch(min_rate=0.8, max_rate=1.25)
snippet.pitch_shift(min_semitones=-4, max_semitones=4)

# Save augmented snippet (automatically flagged as transformed)
snippet.save("cricket", "./augmented", metadata=True)
```

### Full audio scanning

```python
from stridulant import stridulation_scan

# Scan entire audio file for stridulations
candidates, features = stridulation_scan(
    "long_recording.wav",
    snippet_duration=2.0,
    overlap=0.5,
    min_pulses=6,
    min_regularity=10,
    sp_range=(5500, 15000)  # Cricket frequency range
)

print(f"Found {len(candidates)} candidate events at: {candidates}")
```

### Training a CNN classifier

```python
from stridulant import train_model

# Directory structure:
# train_dir/
#   positive/  (stridulation spectrograms)
#   negative/  (non-stridulation spectrograms)

model, history = train_model(
    train_dir="./spectrograms",
    target_size=(128, 128),
    batch_size=32,
    epochs=10,
    learning_rate=0.001
)
```

### Classifying spectrograms with trained model

```python
from stridulant import classify_spectrograms

# Classify all images in a directory
classify_spectrograms(
    input_dir="./unlabeled_spectrograms",
    model_path="./spectrograms/stridulation_detection_model.keras"
)
# Files moved to ./unlabeled_spectrograms/positive/ or negative/
```

## 📖 API Reference

### AudioSnippet Class

| Method | Description |
|--------|-------------|
| `save(source_name, output_dir)` | Save snippet as WAV |
| `play()` | Play audio (Esc to stop) |
| `spectrogram(spec_type, ...)` | Generate spectrogram |
| `normalize()` | Peak normalization (75% by default) |
| `find_events(...)` | Detect acoustic events |
| `extract_features(event, ...)` | Extract 13+ acoustic features |
| `is_stridulation(...)` | Classify as stridulation |
| `is_cavitation(...)` | Classify as cavitation |
| `add_gaussian_noise(...)` | Augmentation |
| `time_stretch(...)` | Augmentation |
| `pitch_shift(...)` | Augmentation |
| `shift(...)` | Augmentation |
| `clipping_distortion(...)` | Augmentation |
| `add_background_noise(...)` | Augmentation |
| `time_mask(...)` | Augmentation |

### Spectrogram Class

| Method | Description |
|--------|-------------|
| `plot(color)` | Display spectrogram |
| `plot_events(events)` | Plot Hilbert envelope with marked events |
| `save_img(output_path, with_labels, color, events)` | Save as PNG |
| `save_table(source_name, output_dir)` | Export as CSV |
| `compute_power_metrics()` | Calculate APD and PPD (dB) |

### Processing Functions

| Function | Description |
|----------|-------------|
| `process_audio_file(...)` | Full pipeline: split → spectrograms → save |
| `annotate_data(...)` | Separate positives/negatives using annotation CSV |
| `stridulation_scan(...)` | Scan entire audio for stridulations |
| `cavitation_scan(...)` | Scan entire audio for cavitation |
| `highpass_filter(audio, sr, cutoff)` | Butterworth high-pass filter |
| `lowpass_filter(audio, sr, cutoff)` | Butterworth low-pass filter |
| `normalize_global(input_dir, output_dir)` | Normalize folder to global peak |

### Utility Functions

| Function | Description |
|----------|-------------|
| `load_audio(path, normalize=False)` | Load audio with soundfile |
| `save_audio(audio, sr, path, overwrite=False)` | Save audio as WAV |
| `load_snippet(file_path)` | Load snippet from WAV (extracts start_time from filename) |

### ML Functions

| Function | Description |
|----------|-------------|
| `train_model(train_dir, ...)` | Train CNN classifier |
| `classify_spectrograms(input_dir, model_path)` | Classify with trained model |

## ⚙️ Configuration Parameters

### Stridulation Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_pulses` | 6 | Minimum pulses per event |
| `min_regularity` | 10 | Rhythm consistency (1/std of intervals) |
| `min_duration` | 0.5s | Minimum event duration |
| `max_duration` | 1.0s | Maximum event duration |
| `sustain` | 0.5 | Duty cycle (fraction of event with sound) |
| `sp_range` | (5500, 15000) | Valid frequency range (Hz) |
| `pulse_dist` | 20ms | Minimum time between pulses |
| `env_smooth` | 10ms | Hilbert envelope smoothing |

### Cavitation Detection Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `energy_threshold` | 0.001 | Minimum energy for cavitation |
| `min_event_duration` | 0.5ms | Minimum event duration |
| `threshold_percentile` | 85 | Energy percentile for detection |

### CNN Architecture

```
Input: (128, 128, 3)
├── Conv2D(32, 3x3, relu) + MaxPooling2D(2,2)
├── Conv2D(64, 3x3, relu) + MaxPooling2D(2,2)
├── Conv2D(128, 3x3, relu) + MaxPooling2D(2,2)
├── Flatten
├── Dense(512, relu) + Dropout(0.5)
└── Dense(1, sigmoid)
```

## 📁 Expected Directory Structures

### For `process_audio_file()`
```
output_folder/
├── Audio_snippets/
│   └── source_snippet_starttime.wav
└── Spectrograms/
    └── source_spectrogram_type_starttime_.png
```

### For `train_model()`
```
train_dir/
├── positive/
│   ├── spectrogram1.png
│   └── spectrogram2.png
└── negative/
│   ├── spectrogram3.png
│   └── spectrogram4.png
```

### For `stridulation_scan()`
```
audio_name/
├── snippets/
│   └── audio_name_snippet_starttime.wav
└── spectrograms/
    ├── audio_name_spectrogram_fft_starttime_.png
    └── audio_name_spectrogram_hilbert_starttime_.png
```

## 🧠 Feature Extraction Details

The `extract_features()` method returns a dictionary with:

| Feature | Description |
|---------|-------------|
| `event_start_time` | Start time in seconds |
| `event_duration` | Duration in seconds |
| `event_energy` | Maximum energy in event |
| `pulse_count` | Number of detected pulses |
| `pulse_regularity` | 1 / std(intervals) |
| `avg_pulse_interval` | Mean time between pulses (s) |
| `pulse_density` | Pulses per second |
| `duty_cycle` | Fraction of event with sound (0-1) |
| `avg_pulse_duration` | Mean pulse duration (s) |
| `spectral_centroid_mean` | Dominant frequency (Hz) |
| `tonal_variation` | std(centroid) / mean(centroid) |
| `dynamic_range` | Max - min (dB) |
| `attack_slope` | Amplitude rise rate at onset |

## ⚠️ Notes

- **TensorFlow is lazily loaded** - ML modules are not imported by default to keep the package lightweight. Import them explicitly when needed:
  ```python
  from stridulant import train_model, classify_spectrograms
  ```

- **Cavitation detection** requires non-normalized audio (absolute energy matters)

- **Audio format** - WAV files recommended. Other formats supported via librosa.

- **Large files** - Use `stridulation_scan()` with overlap for long recordings

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. Code follows existing patterns
2. Docstrings are complete (Google format)
3. Tests pass for core functionality

## 📄 License

**MIT + Commons Clause**

This software is freely available for non-commercial use (research, education, personal projects).

Commercial use, including selling the software or offering it as a paid service, is **not permitted** without explicit written permission from the author.

For commercial licensing inquiries, please contact: **saul.rguezm@gmail.com**

## 👤 Author

**Saúl Rodríguez Martínez**
- Email: saul.rguezm@gmail.com
- Bitbucket: [Saul_Rguez/stridulant](https://bitbucket.org/Saul_Rguez/stridulant)

## 🙏 Acknowledgements

Built with:
- [Librosa](https://librosa.org/) - Audio analysis
- [TensorFlow](https://tensorflow.org/) - Deep learning
- [Audiomentations](https://github.com/iver56/audiomentations) - Data augmentation
- [SoundDevice](https://python-sounddevice.readthedocs.io/) - Audio playback

## 📚 Citation

If you use Stridulant in your research, please cite:

```bibtex
@software{stridulant2025,
  author = {Rodríguez Martínez, Saúl},
  title = {Placeholder},
  year = {2025},
  url = {https://bitbucket.org/Saul_Rguez/stridulant}
}
```

---

**Stridulant** 🦗