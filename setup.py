"""
DCLL library
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name = "decolle",
    version = "0.1",
    author = "Emre Neftci, Massi Iacono, Jacques Kaiser",
    author_email = "eneftci@uci.edu",
    description = ("Deep Continuous Local Learning"),
    keywords = "spiking neural networks learning",
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    test_suite = 'nose.collector',
    long_description=long_description,
    license='GPL 3.0',
    install_requires=[
        "torch>=0.4",
        "scipy>=1.0",
        "h5py",
        "tqdm",
        "tensorboardX",
        "torchneuromorphic>=0.2",
        "pyyaml"
    ]
)
