
# Stridulant

**Stridulant** is a powerful tool designed for efficient audio processing and analysis. It simplifies tasks such as cutting audio segments, augmentation, normalization, and the creation of spectrograms. Perfect for generating high-quality graphical material or preparing datasets for AI training.

## Features

- Easily cut audio files into segments (e.g., .wav).

- Apply augmentation techniques to improve audio datasets.

- Normalize audio levels to ensure consistent quality.

- Generate spectrograms and analyze frequency patterns.

- Seamlessly integrates with Python projects via the stridulant package.

- Supports popular audio processing libraries like librosa and soundfile.


## Installation

You can install **Stridulant** locally using `pip`:

```bash
pip install .
```

Alternatively, you can install the requirements directly from requirements.txt:

```bash
pip install -r requirements.txt
```

### Requirements
Make sure you have Python 3.6+ installed, along with the necessary dependencies.

1. Clone the repository:

```bash
git clone https://bitbucket.org/Saul_Rguez/stridulant.git
cd stridulant
```

2. Install the package and its dependencies:

```bash
pip install .
```

Alternatively, you can install the requirements directly from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

To get started, import the package and begin analyzing audio files:

```python
import stridulant as st
```

For more advanced functionality, check the provided documentation and explore the different methods to analyze the stridulations, visualize spectrograms, and more.

## Contributing

We welcome contributions to **Stridulant**! If you want to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
