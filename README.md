<p align="left"><img src="logo/logo.png" height="115" width="110"></img></p>

# Achilles

Achilles is a neural network model that distinguishes between nanopore signals from hosts (e.g. human background) and pathogens (e.g. *Burkholderia pseudomallei*). The model currently trains on a small number of reads reads (~ 3,000 - 12,000 reads). Issues are tracked on Github.

This is a minimal Keras implementation / adaptation of the open-source [deep neural net base-caller Chiron](https://github.com/haotianteng/Chiron), and all credit for the architecture of the model is due to Haotian Teng and Lachlan Coin, and the co-authors of Chiron [published in Gigascience (2018)](https://academic.oup.com/gigascience/article/7/5/giy037/4966989).

### Models
---

Training regimes for architecture and hyperparameter exploration and models for prediction evalutaion of pathogen DNA  in host (human) background can be found in the following sections:

Pathogens:

* Zika virus detection from [ZIBRA](http://www.zibraproject.org/data/) project in Brazil - 2D R9 amplicons
* Biothreat agent and melioidosis bacterium *Burkholderia pseudomallei* + closely related *B. cepacia*, *B. mallei* - 1D Rapid R9.4
* XDR *Mycobacterium tuberculosis* from Papua New Guines - 1D Rapid R9.4
* *Klebsiella penumoniae* from Australia - 1D Rapid R9.4
* CA-MRSA from Pakistan (Bengal Bay clone) and Far North Queensland (Cape York Peninsula) - 1D Rapid R9.4

Hosts:

* [Human nanopore reference genome](https://github.com/nanopore-wgs-consortium/NA12878/blob/master/Genome.md) CEPH1463 (NA12878/GM12878, Ceph/Utah pedigree) - 1d Rapid R9 + R9.4

### Documentation
---

[achilles.readthedocs.io](https://achilles.readthedocs.io)
