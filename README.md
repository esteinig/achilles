<p align="left"><img src="logo/logo.png" height="115" width="110"></img></p>

# Achilles

![](https://img.shields.io/badge/tf--gpu-1.8-blue.svg)
![](https://img.shields.io/badge/keras-2.2.0-blue.svg)
![](https://img.shields.io/badge/docs-latest-green.svg)
![](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

 **`v0.3-alpha`**: `it's working, but there are no tests and the code is a mess` :bug: 

`Achilles` is a platform for training, evaluating and deploying neural network models that act as taxonomic classifiers of raw nanopore signal, for instance by distinguishing between nanopore signals from hosts (e.g. human background) and pathogens (e.g. *Burkholderia pseudomallei* or *Mycobacterium tuberculosis*). The neural networks are essentially a Keras implementation of the hybrid convolutional and recurrent architecture from [deep neural net base-caller Chiron](https://github.com/haotianteng/Chiron) [published in Gigascience (2018)](https://academic.oup.com/gigascience/article/7/5/giy037/4966989). We have replaced some of the regularization functions with those available in `Keras`, in particular we implemented internal and regular Dropout in the LSTM layer instead of Batch Normalization. Overall, the implementation is minimal, and replaces the bi-directional LSTM with a regular LSTM layer, as well as combining only a single resiudal block with a single LSTM, totalling around 600,00 learnable parameters, somewhere around the size of MobileNets. This necessitates longer training epochs, but keeps model prediction fast in the interest of mobile deployment or real-time / online learning from sequence streams.

### Install
---

`Tensorflow-GPU` and the associated `CUDA` driver on the GPU must be installed. We used `Achilles` on the JCU cluster that has two Tesla V100 with 16GB memory mounted. For some reason, installation of the interface environment did not work with higher `tensorflow-gpu` versions linked with `CUDA 9.2` or `CUDA 10`, so we installed `tensorflow-gpu v1.8.0` from `conda`, which worked on the previous setup, with the `cudnn` library version `v7.1.2`, which also installs the `cudatoolkit` version `v9.0` in `conda`. This environment works with the `CUDA 9.0` driver for the GPU on the cluster. 

```
pip install achilles==0.3-alpha  # does not install tensorflow-gpu
```

You know if the driver and `tensorflow-gpu` work when you call the main help interface of `Achilles`, which lists the avaialble tasks:

```
achilles --help
```

### Command line tasks
---

Alpha version is for testing the software with some pre-trained models. You can also train your own models, which relies on `Poremongo` also in alpha stage at the moment and subject to change, so the code is not so stable. 

Achilles is accessible through the CLI which summarizes some of the important tasks and exposes them to the user. Tasks like `achilles train` and `achilles create` have many parameters for setting the global parameters for signal sampling or the framework i nwhich the models are trained in `Keras`. 

Tasks:

``


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
