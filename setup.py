# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:23:13 2025

@author: Saul

!!WARNING this is unfinished and should be checked before distribution.
"""

from setuptools import setup, find_packages

setup(
    name='stridulant',
    version='0.1.0', 
    description='A multipurpose package to manipulate sound.',
    long_description=open('README.md').read(),  
    long_description_content_type='text/markdown',
    author='Saúl Rodríguez Martínez',  
    author_email='saul.rguezm@gmail.com', 
    url='https://bitbucket.org/Saul_Rguez/stridulant',  
    packages=find_packages(where="src"), 
    package_dir={'': 'src'},
    install_requires=[ 
        'librosa',
        'soundfile',
        'sounddevice',
        'matplotlib',
        'tqdm',
        'scipy',
        'numpy',
        'tensorflow',
        'pandas',
        'audiomentations'
        'pynput'
    ],
    classifiers=[ 
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  

    include_package_data=True)
