import setuptools
from setuptools import find_packages
from setuptools import setup, find_packages

setup(
    name='achilles',
    url='https://github.com/esteinig/achilles',
    author='Eike J. Steinig',
    author_email='eikejoachim.steinig@my.jcu.edu.au',
    packages=find_packages(),
    install_requires=[
      'tensorflow',
      'keras',
      'pandas',
      'numpy',
      'scipy',
      'scikit-image',
      'scikit-learn',
      'h5py',
      'tqdm',
      'seaborn',
      'matplotlib',
      'colorama',
      'mongoengine',
      'click',
      'pyyaml'
    ],
    version='0.2',
    license='MIT',
    description='Achilles is a platform to train and evaluate neural networks for nanopore signal classification.'
)