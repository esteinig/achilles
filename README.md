# Asclepius

Asclepius is a neural network model that distinguishes between host (human) and pathogen (bacterium *B. pseudomallei*) from raw nanopore signal. The model trains on limited reads (~ 3000). It will utilize the distribution of class probabilities over windows of sequence signal to predict outcome (human, pathogen). Issues are tracked on Github. Next steps are to reduce overfitting and deepening the network for generalisation over the human genome and more diverse bacterial or viral pathogens. Patient sample simulations and sequencing runs of MRSA genomes are planned for release and once a suitable architecture is found, we may consider extending the architecture to multi-label classifications (i.e. human, bacterial, viral) for rapid pathogen identification from complex mixtures.

This is a sort of minimal Keras implementation / adaptation of the open-source [deep neural net base-caller Chiron](https://github.com/haotianteng/Chiron), and all credit for the architecture of the model is due to Haotian Teng and Lachlan Coin, and the co-authors of [Chiron published in Gigascience (2018)](https://academic.oup.com/gigascience/article/7/5/giy037/4966989). It differs in several components, like current minimal model depth (1 residual block, one bidirectional LSTM) or using recurrent dropout instead of recurrent batch normalization, which was not readily available in Keras. It also gets rid of the CTC decoder and outputs with a simple fully-connected (binary) layer and softmax activation, to predict outcome (human, pathogen) in the current Asclepius model configuration.

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
