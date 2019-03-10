<p align="left"><img src="logo/logo.png" height="115" width="110"></img></p>

# Achilles

![](https://img.shields.io/badge/tf--gpu-1.8-blue.svg)
![](https://img.shields.io/badge/keras-2.2.0-blue.svg)
![](https://img.shields.io/badge/docs-latest-green.svg)
![](https://img.shields.io/badge/lifecycle-experimental-orange.svg)

 **`v0.3-alpha`**: `it's working, but there are no tests` :bug: 

`Achilles` is a platform for training, evaluating and deploying neural network models that act as taxonomic classifiers of raw nanopore signal, for instance by distinguishing between nanopore signals from hosts (e.g. human background) and pathogens (e.g. *Burkholderia pseudomallei* or *Mycobacterium tuberculosis*). The neural networks are essentially a Keras implementation of the hybrid convolutional and recurrent architecture from [deep neural net base-caller Chiron](https://github.com/haotianteng/Chiron) [published in Gigascience (2018)](https://academic.oup.com/gigascience/article/7/5/giy037/4966989). We have replaced some of the regularization functions with those available in `Keras`, in particular we implemented internal and regular Dropout in the LSTM layer instead of Batch Normalization. Overall, the implementation is minimal, and replaces the bi-directional LSTM with a regular LSTM layer, as well as combining only a single resiudal block with a single LSTM, totalling around 600,00 learnable parameters, somewhere around the size of MobileNets. This necessitates a higher number of epochs for training, but stil learns from limited signal data and keeps model prediction fast in the interest of mobile deployment or real-time / online learning from sequence streams.

### :snake: Install
---

`Tensorflow-GPU` and the associated `CUDA` driver on the GPU must be installed. We used `Achilles` on the JCU cluster that has two Tesla V100 with 16GB memory. For some reason, installation of the GPU environment did not work with higher `tensorflow-gpu` versions `> v1.8` that interface with `CUDA 9.2` or `CUDA 10`, so we installed `tensorflow-gpu v1.8.0` from `conda` with the `cudnn` library version `v7.1.2`, which also installs the `cudatoolkit` version `v9.0` in `conda`. This environment works with the `CUDA 9.0` driver for the GPUs on the cluster. 

```
pip install achilles==0.3-alpha  # does not install tensorflow-gpu
```

You know if the driver and `tensorflow-gpu` work when you call the main help interface of `Achilles`, which lists the available tasks:

```
achilles --help
```

### :whale: Command line interface (CLI)
---

Alpha version is for testing the software with some pre-trained models. You can also train your own models, which relies on `Poremongo` also in alpha stage at the moment and subject to change, so the code is not so stable. 

Achilles is accessible through the CLI which summarizes some of the important tasks and exposes them to the user. Tasks like `achilles train` and `achilles create` have many parameters for setting the global parameters for signal sampling or the framework in which the models are trained. 

#### Tasks
---

:sunflower: **`achilles create`**

Create a training or evaluation `HDF5` data set of singal slices from the `Fast5` files for input into training with `Achilles`. Uses `Poremongo` to sample labels from `MongoDB` database index of all read files on the system. Does not need GPU.

```
achilles create --help
```

***

:seedling: **`achilles train`**

Train a `HDF5` training dataset of signal slices and labels in `Keras` using the `Achilles` variant of `Chirons` hybrid convolutional and recurrent architecture. A simple fully connected layer predicts labels in the output. Absolutely needs GPU for training.

```
achilles train --help
```

***

:cactus: **`achilles eval`**

Evaluate a trained `HDF5` model (always best validation error from training run with `achilles train`) across a directory of evaluation datasets. Should run on GPU.

```
achilles eval --help
```

***

:deciduous_tree: **`achilles predict`**

Predict labels using a trained model and a directory of `.fast5` files. Can `--watch` a dirrectory for live `.fast5` files. Should run on GPU.

```
achilles predict --help
```


### :cat2: Pre-trained models
---

Currently all pretrained models are standardized to a lightweight `1 x 256-channel ResBlock + 1 x 200-unit LSTM` architecture with `dropout` in recurrent layers that predicts on overlapping slices of 400 signal values from `R9.4` pores; this creates a network model with around 660,000 trainable parameters, which we trianed over 500 epochs on a Tesla V100 GPU with 16GB memory over 8 hours with a batch size of 3000 bathes per forward pass. The model predicts from a terminal fully connected layer with `softmax` activation function over `n` labels. Training on the alpha version models is conducted on 100,000 signal slices extracted evenly over each subcategory of the label (pathogens, chromosomes) with a random sampling window on the read that extracts `50 x 400` slices with step 40. This equates to roughly 2000 reads per label and around 200 - 1000 reads per subcategory in the label depending on the number of subcategory mixtures that tags in the database are sampled from (e.g. pathogens or chromosome mixtures). Models are trained using `Adam` optimizer and the `binary crossentropy` loss function, which is selected due to the binary prediction of `pathogen` vs. `host`, depending on how we train the models with pathogen subcategories and human chromsomes.

In these pretrained models the human label is always trained from chromosomes 2, 4, 8, 16 and evaluated on chromsomes 5, 9, 15, 17 to make sure that the classifiers generalize over the whole human genome. Mixtures of pathogens on the other hand are useful to build generalized classifiers (bacteria vs. human) vs. specific classifiers (mrsa vs human).

:mouse2: **Generalists**:

  * Bacteria in human host (trained on Human reference genome mixture of chromosomes and *K. pneumoniae*, *M. tuberculosis*, *B.  pseudomallei*)  - `models/human.bacteria.alpha.1.hd5`
  * Bacteria in human host (trained on Human reference genome mixture of chromosomes and *E. coli*, *M. tuberculosis*, *B.  pseudomallei*)  - `models/human.bacteria.alpha.2.hd5`
  
:penguin: **Specialists**:

 * ... soon ...

### :turtle: Training data
---

**Pathogens**:

* Zika virus detection from [ZIBRA](http://www.zibraproject.org/data/) project in Brazil - 2D R9 amplicons
* Biothreat agent and melioidosis bacterium *Burkholderia pseudomallei* + closely related *B. cepacia*, *B. mallei* - 1D Rapid R9.4
* XDR *Mycobacterium tuberculosis* from Papua New Guines - 1D Rapid R9.4
* *Klebsiella penumoniae* from Australia - 1D Rapid R9.4
* CA-MRSA from Pakistan (Bengal Bay clone) and Far North Queensland (Cape York Peninsula) - 1D Rapid R9.4

**Hosts**:

* [Human nanopore reference genome](https://github.com/nanopore-wgs-consortium/NA12878/blob/master/Genome.md) CEPH1463 (NA12878/GM12878, Ceph/Utah pedigree) - 1d Rapid R9 + R9.4

### :fish: Documentation
---

[achilles.readthedocs.io](https://achilles.readthedocs.io)
