# Asclepius

Asclepius is a neural network model that distinguishes between host (human) and pathogen (bacterium *B. pseudomallei*) from raw nanopore signal. The model trains on limited reads (~ 3000). It will utilize the distribution of class probabilities over windows of sequence signal to predict outcome (human, pathogen). Issues are tracked on Github. Next steps are to reduce overfitting and deepening the network for generalisation over the human genome and more diverse bacterial or viral pathogens. Patient sample simulations and sequencing runs of MRSA genomes are planned for release and once a suitable architecture is found, we may consider extending the architecture to multi-label classifications (i.e. human, bacterial, viral) for rapid pathogen identification from complex mixtures.

This is a sort of minimal Keras implementation / adaptation of the open-source [deep neural net base-caller Chiron](https://github.com/haotianteng/Chiron), and all credit for the architecture of the model is due to Haotian Teng and Lachlan Coin, and the co-authors of [Chiron published in Gigascience (2018)](https://academic.oup.com/gigascience/article/7/5/giy037/4966989). It differs in several components, like current minimal model depth (1 residual block, one bidirectional LSTM) or using recurrent dropout instead of recurrent batch normalization, which was not readily available in Keras. It also gets rid of the CTC decoder and outputs with a simple fully-connected (binary) layer and softmax activation, to predict outcome (human, pathogen) in the current Asclepius model configuration.

### Documentation
---

[asclepius.readthedocs.io](https://asclepius.readthedocs.io)

### Performance for classification of human (chromosome 20) and *B. pseudomallei*
---

This is a proof-of-concept for a pathogen detector based on a scenario where we sequence the environmental bacterium and biothreat agent *B. pseudomallei* in a host contaminant background of human DNA (annd raw nanopore signal) on a MinION 1D R9.4. It also kind of makes an AI learn what a human genome looks like, so there is somethign to think about. This architecture can be modified for different pathogen detection scenarios (i.e. different host backgrounds or more compelex microbial communities) and by extending the open-source architecture based on Chiron, the detector can be trained de novo and on data suitable to specific local problems. This section on performance also looks at evaluating generalization of learning from a tiny fraction of chromosome 20 to accurately classify human signal from other chromosomes. In the evaluaton data, we also swap pathogen signals to assess generalization to other bacterial or viral pathogens and threshold the taxonomic distances, at which classification breaks down.

#### Architectures
---

| Run ID          | ResBlocks | LSTMs  | Windows   | Total     | Batch Size  | Epochs | Dropout   | Recurrent Dropout | Batch Norm |
| :-------------: | :-------: | :----: | :-------: | :-------: | :---------: | :----: | :-------: | :---------------: | :--------: |
| Baseline Drop1  |  1        | 1      | 400 x 400 | 237963    | 900         | 16/20  | 0.2       |  0.2              |  False     |
| Baseline Drop2  |  1        | 1      | 400 x 400 | 237963    | 900         | 39/40  | 0.3       |  0.3              |  False     |


#### Evaluations
---


| Run ID          | Train. Acc. | Val. Acc.  | Chr20 Acc. | Chr11 Acc.  |   
| :-------------: | :----------:| :--------: | :--------: | :---------: |
| Baseline Drop1  |  88.81%     | 88.97%     | 83.11%     | 87.6%       |
| Baseline Drop2  |  91.29%     | 90.43%     |            |             |


#### Training, validation and evaluation data sets
---

Training data sets

`Training`: 

* 133782 (burkholderia), 104181 (chromosome 20)
* 2800 Fast5 
* 70% training, 30% validation,
* 400 x 400, not normalized, random scanning

Evaluation data sets:

`Chromosome 20`: (same as training)

* 6731 (burkholderia), 5104 (chromosome 20)
* 2800 Fast5 
* 70% training, 30% validation,
* 400 x 400, not normalized, random select + scanning

`Chromosome 11`: (for generalization) 

* 6731 (burkholderia), 5104 (chromosome 20)
* 2800 Fast5 
* 70% training, 30% validation,
* 400 x 400, not normalized, random select + scanning

### Terminal
---

#### Data Generator

`asclepius make --help`

```
usage: asclepius.py make [-h] [--dirs DIRS] [--data_file DATA_FILE]
                         [--signal_max_per_class SIGNAL_MAX]
                         [--max_windows_per_read WINDOW_MAX]
                         [--random_windows_per_read]
                         [--signal_length SIGNAL_LENGTH]
                         [--signal_stride SIGNAL_STRIDE] [--normalize]
                         [--print]
```

#### Training 

`asclepius train --help`

```
usage: asclepius.py train [-h] [--data_file DATA_FILE]
                          [--output_file OUTPUT_FILE] [--run_id RUN_ID]
                          [--signal_length SIGNAL_LENGTH]
                          [--batch_size BATCH_SIZE] [--threads THREADS]
                          [--epochs EPOCHS] [--log_interval LOG_INTERVAL]
                          [--activation ACTIVATION] [--loss LOSS]
                          [--optimizer OPTIMIZER]
                          [--nb_residual_blocks NB_RESIDUAL_BLOCKS]
                          [--nb_channels NB_CHANNELS] [--nb_lstm NB_LSTM]
                          [--dropout DROPOUT] [--recurrent_dropout RC_DROPOUT]
                          [--batch_norm]
```

#### Evaluation

`asclepius evaluate --help`

```
usage: asclepius.py evaluate [-h] [--data_file DATA_FILE]
                             [--model_file MODEL_FILE]
                             [--batch_size BATCH_SIZE] [--threads THREADS]
                             [--data_path DATA_PATH]
```


#### Prediction

`asclepius predict --help`

```
usage: asclepius.py evaluate [-h] [--data_file DATA_FILE]
                             [--model_file MODEL_FILE]
                             [--batch_size BATCH_SIZE] [--threads THREADS]
                             [--data_path DATA_PATH]
```

#### Plots

`asclepius plot --help`

```
usage: asclepius.py plot [-h] [--log_file LOG_FILE] [--plot_file PLOT_FILE]
                         [--error]
```

#### Utils

`asclepius select --help`

```
usage: asclepius.py select [-h] [--input_dir INPUT_DIR]
                           [--output_dir OUTPUT_DIR] [--nb_fast5 N]
                           [--largest]
```
