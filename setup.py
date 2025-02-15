# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 19:23:13 2025

@author: Saul

!!WARNING this is unfinished and should be checked before distribution.
"""

from setuptools import setup, find_packages

setup(
    name='stridulant',  # Nombre de tu paquete
    version='0.1.0',  # Versión de tu paquete
    description='An AI tool for detecting ant stridulations in audio data.',
    long_description=open('README.md').read(),  # Lee el contenido de tu README para mostrar en PyPi
    long_description_content_type='text/markdown',
    author='Tu Nombre',  # Tu nombre o el nombre del equipo de desarrollo
    author_email='tu_email@dominio.com',  # Tu correo electrónico
    url='https://github.com/tuusuario/stridulant',  # URL del repositorio
    packages=find_packages(),  # Encuentra todos los paquetes en el directorio actual
    install_requires=[  # Aquí van las dependencias necesarias
        'librosa',
        'soundfile',
        'sounddevice',
        'matplotlib',
        'tqdm',
        'scipy',
        'numpy',
    ],
    classifiers=[  # Clasificadores de PyPI
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Requiere Python 3.6 o superior
    entry_points={  # Si quieres definir un comando de consola para tu paquete
        'console_scripts': [
            'stridulant-cli=stridulant.cli:main',  # Esto es solo un ejemplo, se puede personalizar
        ],
    },
    include_package_data=True,  # Incluir archivos adicionales definidos en MANIFEST.in
)
