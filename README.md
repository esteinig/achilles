# Asclepius

Asclepius is a neural network model that distinguishes between host, e.g. human, and pathogen, e.g.*B. pseudomallei*, origin of raw nanopore signal. The model trains on limitied data and uses the distribution of class probabilities over several windows of sequence signal to predict taxonomy. We are currently reducing overfitting and deepenign the network for generalisation over the human genome and bacterial or viral pathogens. Patient sample simulations and sequencing runs of MRSA sequencing are planned for release.

### Data Generator

`asclepius gen --help`

### Training 

`asclepius train --help`

### Evaluation

`asclepius eval --help`

### Prediction

`asclepius predict --help`

### Plots

`asclepius plot --help`

### Util

`asclepius select --help`

### Documentation

[asclepius.readthedocs.io](https://asclepius.readthedocs.io)
