"""
Stridulant - Data Preparation Module

This module provides functions to prepare audio data for training the stridulation classification model.
It reads annotations from a CSV file and separates audio snippets and spectrograms into positive and negative folders
based on the presence of stridulations within each snippet.

Functions:
- `annotate_data`: Separates audio snippets and spectrograms into positive and negative sets.

Author: Saul Rodriguez Martinez
Creation date: 2025-02-15
"""

import os
import shutil
import pandas as pd

def annotate_data(csv_path, snippets_dir, spectrograms_dir, output_dir, snippet_duration=2.0):
    """
    Separates audio snippets and spectrograms into positive and negative folders based on annotations.

    Args:
    csv_path (str): Path to the CSV file containing start and end times of stridulations.
    snippets_dir (str): Path to the directory containing audio snippets.
    spectrograms_dir (str): Path to the directory containing spectrograms.
    output_dir (str): Path to the output directory where positive and negative sets will be stored.
    snippet_duration (float): Duration of each audio snippet in seconds.

    The CSV should have two columns: `start_time` and `end_time` indicating the time intervals of stridulations.
    All snippets overlapping these intervals are considered positive samples.
    """
    annotations = pd.read_csv(csv_path)
    positives_dir = os.path.join(output_dir, 'positives')
    negatives_dir = os.path.join(output_dir, 'negatives')
    os.makedirs(positives_dir, exist_ok=True)
    os.makedirs(negatives_dir, exist_ok=True)

    def is_positive(snippet_name):
        snippet_time = float(snippet_name.split('_')[-2])
        for _, row in annotations.iterrows():
            if not (row['end_time'] < snippet_time or row['start_time'] > snippet_time + snippet_duration):
                return True
        return False

    for file in os.listdir(snippets_dir):
        if file.endswith('.wav') or file.endswith('.png'):
            target_dir = positives_dir if is_positive(file) else negatives_dir
            src = os.path.join(snippets_dir, file)
            dst = os.path.join(target_dir, file)
            shutil.move(src, dst)

    for file in os.listdir(spectrograms_dir):
        if file.endswith('.png'):
            target_dir = positives_dir if is_positive(file) else negatives_dir
            src = os.path.join(spectrograms_dir, file)
            dst = os.path.join(target_dir, file)
            shutil.move(src, dst)

    print("Separation completed successfully.")
