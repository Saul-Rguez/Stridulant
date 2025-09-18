# -*- coding: utf-8 -*-
"""
This module initializes the Stridulant package and exposes its functionality
for easy access to users.

Author: Saul Rodriguez Martinez
Date: 2025-02-15
"""


from stridulant.audio_snippet import *
from stridulant.spectrogram import *
from stridulant.processing import *
from stridulant.utils import *

# this functions will prevent heavy modules to be loaded on normal use case
# but seamlessly load them when needed.
def load_train_model():
    from stridulant import train_model
    return train_model

def load_classify_spectrograms():
    from stridulant import classify_spectrograms
    return classify_spectrograms