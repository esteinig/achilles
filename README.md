<p align="left"><img src="logo.png" height="115" width="110"></img></p>

# Achilles

Achilles is a neural network model that distinguishes between nanopore signals from hosts (e.g. human background) and pathogens (e.g. *B. pseudomallei*). The model currently trains on a small number of reads reads (~ 3000 reads). Issues are tracked on Github.

This is a minimal Keras implementation / adaptation of the open-source [deep neural net base-caller Chiron](https://github.com/haotianteng/Chiron), and all credit for the architecture of the model is due to Haotian Teng and Lachlan Coin, and the co-authors of Chiron [published in Gigascience (2018)](https://academic.oup.com/gigascience/article/7/5/giy037/4966989).

### Performance for classification of human (chromosome 20) and *B. pseudomallei*
---

This is a proof-of-concept for a pathogen detector based on raw nanopore signal from the [environmental bacterium and biothreat agent *B. pseudomallei*](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4746747/) in host contaminant background of human DNA on MinION 1D R9.4. This architecture can be modified for different pathogen detection scenarios i.e. different host backgrounds or more complex microbial communities. By extending the open-source architecture based on Chiron, the detector can be trained *de novo* and on data suitable to specific local problems. This section traces exploration of viable architectures and generalization over the human genome. This is kind of cool, because we are teaching an 'artificial intelligence' what ah uman in signal space looks like...

#### Architectures
---

| Run ID    | ResBlocks | Channels | RNN Layers | RNN Cell  | RNN cuDNN | RNN Units | RNN Dropout | Recurrent Dropout | FC Activation  | Classes  | Parameters  | Notes              | 
| :-------: | :-------: | :------: | :--------: | :-------: | :-------: | :-------: | :---------: | :---------------: | :------------: | :------: | :--------:  | :----------------: |
| minimal_1 |  1        | 256      | 1          | BLSTM     | no        | 200       | 0.2         | 0.2               | Softmax        | 2        |  999,778    | -                  |
| minimal_2 |  1        | 256      | 1          | BLSTM     | no        | 200       | 0.2         | 0.2               | Softmax        | 2        |  999,778    | -                  |
| minimal_3 |  1        | 256      | 1          | BLSTM     | no        | 200       | 0.3         | 0.3               | Softmax        | 2        |  999,778    | -                  | 
| minimal_4 |  1        | 256      | 1          | BLSTM     | no        | 200       | 0.2         | 0.2               | Softmax        | 2        |  999,778    | -                  |
| minimal_5 |  1        | 256      | 1          | BLSTM     | yes       | 200       | 0.0         | 0.0               | Softmax        | 2        |  1,001,378  | no dropout         |

#### Training 
---

| Run ID     | Dataset | Total   | Signal  | Windows  | Loss Func | Optimizer  | Batch Size | Epochs | Training  | Validation | 
| :--------: | :-----: | :-----: | :------:| :------: | :------:  | :--------: | :-------:  | :----: | :-------: | :--------: | 
| minimal_1  | Chr20   | 300000  | pA      | 400x400  | Binary CE | Adam       | 800        | 38/40  |  90.78%   | 90.59%     |
| minimal_2  | Chr14   | 300000  | pA      | 400x400  | Binary CE | Adam       | 800        | 38/40  |  91.78%   | 91.26%     |
| minimal_3  | Mixed   | 300000  | pA      | 400x400  | Binary CE | Adam       | 800        | 39/40  |  90.81%   | 90.56%     |
| minimal_4  | Mixed   | 300000  | DAC     | 400x400  | Binary CE | Adam       | 800        | 40/40  |  90.12%   | 89.70%     |
| minimal_5  | Mixed   | 300000  | DAC     | 400x400  | Binary CE | Adam       | 700        | 40/40  |  -        | -          |

*Notes*:

* minimal_1: trained baseline on chromosome 20 (terminal reads)
* minimal_2: prevent bias from potential telomere repeat regions, trained on chromosome 14 (central reads), might be biased from centromere reads?
* minimal_3: random mix of central and terminal reads of chromsomes 11 (terminal, similar to 20), 14 and 20, increased dropout to 0.3, slightly slower (more epochs) for training
* minimal_4: resampled mixed chromosome data without scaling raw DAC values into picoampere (pA) since this requires full read and prevents streaming analysis, reduced dropout to 0.2
* minimal_5: testing cuDNN enabled LSTM cells in Keras, about 2 - 3x faster training, but dropout not supported, looks like it is overfitting around 88% validation accuracy

#### Evaluations
---

| Run ID     | Signal   | Chr20    | Chr14   | Chr11   | Mixed  |
| :--------: | :------: | :------: | :-----: | :-----: | :----: |
| minimal_1  | pA       | 89.37%   | 87.04%  | 86.50%  | 87.97% |
| minimal_2  | pA       | 85.42%   | 88.30%  | 84.03%  | 86.27% |
| minimal_3  | pA       | 87.60%   | 87.61%  | 86.90%  | 90.47% |
| minimal_4  | DAC      |          |         |         |        |
| minimal_5  | DAC      |          |         |         |        |

#### Prediction Evalutations
---

pass

#### Training, validation and evaluation data sets
---

**Training data set for detection of *B. pseudomallei* in [human background](https://github.com/nanopore-wgs-consortium/NA12878/blob/master/Genome.md) DNA from the `nanopore-wgs-consortium/NA12878` genome project**:

* 150,000 (Burkholderia), 150,000 ([terminal chromosome 20](http://s3.amazonaws.com/nanopore-human-wgs/rel3-fast5-chr20.part05.tar), [central chromosome 14](http://s3.amazonaws.com/nanopore-human-wgs/rel3-fast5-chr14.part04.tar))
* 2762 Fast5
* 70% training, 30% validation
* 400 x 400, not normalized, random select + random consecutive scan

**Evaluation data sets for generalizing over human genome**:

* 150,000 (Burkholderia), 150,000 (Human)
* 400 x 400, not normalized, random consecutive scan

* random selection (same as training) of terminal [chromosome 20 (part5)](http://s3.amazonaws.com/nanopore-human-wgs/rel3-fast5-chr20.part05.tar) Fast5
* random selection  (same as training) of central [chromosome 14 (part4)](http://s3.amazonaws.com/nanopore-human-wgs/rel3-fast5-chr14.part04.tar) Fast5
* random selection of terminal [chromosome 11 (part9)](http://s3.amazonaws.com/nanopore-human-wgs/rel3-fast5-chr11.part09.tar) Fast5

* mixed random selection of chromosomes (11, 14, 20)

**Example command line task to generate training and evaluation data**:

`achilles make --dirs bp,human_chr14 --data_file training.chr14.h5 -l 400 -s 400 -m 150000`

### Documentation
---

[asclepius.readthedocs.io](https://asclepius.readthedocs.io)
