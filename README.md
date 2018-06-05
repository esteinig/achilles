# Asclepius

Asclepius is a neural network model that distinguishes between host (human) and pathogen (bacterium *B. pseudomallei*) from raw nanopore signal. The model trains on limited reads (~ 3000). It will utilize the distribution of class probabilities over windows of sequence signal to predict outcome (human, pathogen). Issues are tracked on Github. Next steps are to reduce overfitting and deepening the network for generalisation over the human genome and more diverse bacterial or viral pathogens. Patient sample simulations and sequencing runs of MRSA genomes are planned for release and once a suitable architecture is found, we may consider extending the architecture to multi-label classifications (i.e. human, bacterial, viral) for rapid pathogen identification from complex mixtures.

This is a sort of minimal Keras implementation / adaptation of the open-source [deep neural net base-caller Chiron](https://github.com/haotianteng/Chiron), and all credit for the architecture of the model is due to Haotian Teng and Lachlan Coin, and the co-authors of [Chiron published in Gigascience (2018)](https://academic.oup.com/gigascience/article/7/5/giy037/4966989). It differs in several components, like current minimal model depth (1 residual block, one bidirectional LSTM) or using recurrent dropout instead of recurrent batch normalization, which was not readily available in Keras. It also gets rid of the CTC decoder and outputs with a simple fully-connected (binary) layer and softmax activation, to predict outcome (human, pathogen) in the current Asclepius model configuration.

### Documentation
---

[asclepius.readthedocs.io](https://asclepius.readthedocs.io)

### Performance for classification as human (chromosome 20) or *B. pseudomallei*
---

#### Architectures
---

| Run ID         | ResBlocks | LSTMs  | Windows   | Total     | Batch Size  | Epochs | Dropout   | Recurrent Dropout | Batch Norm |
| :------------: | :-------: | :----: | :-------: | :-------: | :---------: | :----: | :-------: | :---------------: | :--------: |
| Baseline Drop  |  1        | 1      | 400 x 400 | 237963    | 900         | 16/20  | 0.2       |  0.2              |  False     |


#### Evaluations
---


| Run ID         | Train. Acc. | Val. Acc.  | Chr20 Acc. | Chr11 Acc.  |   
| :------------: | :----------:| :--------: | :--------: | :---------: |
| Baseline Drop  |  88.81%     | 88.97%     | 83.11%     | 87.6%       |


#### Training, validation and evaluation data sets
---

Training data sets

* `Training`: 133782, 104181 (bp, chr20) from 2800 Fast5 (70%, validation 30%), 400 x 400, not normalized, random

Evaluation data sets:

* `Chromosome 20` (same as training) 6731, 5104 (bp, chr20) from 140 Fast5, 400 x 400, not normalized, random
* `Chromosome 11` (for generalization) - 6731, 5104 (bp, chr11) from 140 Fast5, 400 x 400, not normalized, random

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
