# Asclepius

Asclepius is a neural network model that distinguishes between host (human) and pathogen (bacterium *B. pseudomallei*) from raw nanopore signal. The model trains on limited reads (~ 3000). It will utilize the distribution of class probabilities over windows of sequence signal to predict outcome (human, pathogen). Issues are tracked on Github. Next steps are to reduce overfitting and deepening the network for generalisation over the human genome and more diverse bacterial or viral pathogens. Patient sample simulations and sequencing runs of MRSA genomes are planned for release and once a suitable architecture is found, we may consider extending the architecture to multi-label classifications (i.e. human, bacterial, viral) for rapid pathogen identification from complex mixtures.

### Data Generator

`asclepius make --help`

### Training 

`asclepius train --help`

### Evaluation

`asclepius evaluate --help`

### Prediction

`asclepius predict --help`

### Plots

`asclepius plot --help`

### Utils

`asclepius select --help`

### Documentation

[asclepius.readthedocs.io](https://asclepius.readthedocs.io)
