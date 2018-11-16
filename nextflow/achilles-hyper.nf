#!/usr/bin/env nextflow

/*
 *  This pipeline is part of the AI-based pathogen detection library 'Achilles'.
 *
 *  Pipeline            Achilles Hyper
 *  Version             0.1
 *  Description         Training and evaluation pipelines for exploration of architecture and hyperparameters with Achilles.
 *           
 *
 *  Container           Singularity GPU and Tensorflow-GPU 1.8 (--nv)
 *
 */

// Initialization before printing parameters:

import java.nio.file.*

workdir = System.getProperty("user.dir");

def get_print_dir(path_str) {  
    path = Paths.get(path_str).toAbsolutePath()
    return ".../" + path.getParent().getName() + "/" + path.getName()
}

print_dir = get_print_dir(params.dir)

if (params.labels instanceof Collection) {
    labels = params.labels.join(',')
} else {
    labels = params.labels
}

// Tokenize if pred_slices is a comma-separated string (command line Nextflow)

if (params.pred_slices instanceof String) {
    pred_slices = params.pred_slices.tokenize(',')
} else {
    pred_slices = params.pred_slices
}

log.info """
==================================================================
                      ACHILLES NEXTFLOW 
                  HYPERPARAMETER EXPLORATION
             (SAMPLE - TRAIN - EVALUATE - PREDICT)
                            v.0.1
==================================================================
 
dir           =    $print_dir
classes       =    $params.classes
labels        =    $params.labels
outdir        =    $params.outdir

                      Variable Parameters
==================================================================

window_max    =    $params.window_max
window_size   =    $params.window_size
window_step   =    $params.window_step
window_scan   =    $params.window_scan
nb_rb         =    $params.nb_rb
nb_lstm       =    $params.nb_lstm
blstm         =    $params.blstm

                      Constant Parameters
==================================================================

validation    =    $params.validation
chunck_size   =    $params.chunk_size
dropout       =    $params.dropout
threads       =    $params.threads
epochs        =    $params.epochs
batch_size    =    $params.batch_size
eval_max      =    $params.eval_max
pred_batch    =    $params.pred_batch
pred_slices   =    $params.pred_slices

                        Data Selection
==================================================================

to be continued...

==================================================================
==================================================================
"""

// Data wildcard directories in prefix directory for Channel.fromPath
def get_data_dirs(String dir) {
    if ( dir.endsWith("/") ) { return dir+"*" } else { return dir+"/*" }
}

// Fast5 class directory inputs for DataGenerator
def get_fast5_dirs(Path path) {
    class_paths = params.classes
                        .collect {  Paths.get(path.toString(), it) }
                        .join(",")
}

// Channels for feeding directory paths from Achilles selection:

training_input_channel    = Channel.fromPath(get_data_dirs(params.dir), type: 'dir')
                                 .map { [it.getName(), get_fast5_dirs(it)] }

// Evaluation data directories:

evaluation_input_channel  = Channel.fromPath(get_data_dirs(params.eval_dir), type: 'dir')
                                    .map { [it.getName(), get_fast5_dirs(it)] }

// Prediction data directories:

prediction_input_channel  = Channel.fromPath(get_data_dirs(params.pred_dir), type: 'dir')
                                   .map { [it.getName(), get_fast5_dirs(it)] }

// Channels for feeding hyperparameters:

window_max_channel  = Channel.from(params.window_max)
window_size_channel = Channel.from(params.window_size)
window_step_channel = Channel.from(params.window_step)
window_scan_channel = Channel.from(params.window_scan)

resblock_channel    = Channel.from(params.nb_rb)
lstm_channel        = Channel.from(params.nb_lstm)
blstm_channel       = Channel.from(params.blstm)

slice_channel       = Channel.from(params.pred_slices)

process DataGenerator {
   
    tag "Data: $prefix -- Processing: $max (windows/class) $size (size) $step (step) $scan (scan)"
    publishDir "$params.outdir/$prefix/data/training", mode: "copy", pattern: "*.h5"
    
    input:
    // Data directory path and string:
    set prefix, str_paths from training_input_channel

    // Dataset processing combination:
    each max from window_max_channel
    each size from window_size_channel
    each step from window_step_channel
    each scan from window_scan_channel

    output:
    set val("$prefix"), val("$size"), val("$step"), val("$scan"), val("${prefix}__${max}__${size}__${step}__${scan}") into training_parameter_channel
    set file("${prefix}__${max}__${size}__${step}__${scan}.training.h5"), file("${prefix}__${max}__${size}__${step}__${scan}.h5") into training_data_channel
    
    // Note that each make samples randomly from the class directories, therefore the same files may be present
    // across training data sets (but not evaluation data, see below)

    """
    python ~/code/achilles/achilles.py make --dirs $str_paths --out ${prefix}__${max}__${size}__${step}__${scan}.h5 \
    --validation $params.validation --window_size $size --window_max $max --window_step $step --window_scan $scan \
    --raw --chunk_size $params.chunk_size
    """

}

process ModelTraining {

    tag "Dataset: $data_id -- Architecture: $nb_resblock x RB, $nb_lstm x LSTM, BLSTM = $blstm"

    publishDir "$params.outdir/$prefix/training", mode: "copy", pattern: "*.checkpoint.*"
    publishDir "$params.outdir/$prefix/training", mode: "copy", pattern: "*.history"
    
    // All combinations of datasets (with prefix and window size)
    // and neural network architectures for training:

    input:
    set prefix, size, step, scan, data_id from training_parameter_channel
    set file(train_data), file(data) from training_data_channel

    each nb_resblock from resblock_channel
    each nb_lstm from lstm_channel
    each blstm from blstm_channel

    output:
    set val("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}"), file("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}.checkpoint.val_loss.h5") into evaluation_model_channel
    set val(prefix), val(size), val(step), val(scan), val(data_id), file(data) into evaluation_data_channel
    
    set val("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}"), file("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}.model.history") into training_summary
    
    set val("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}"), file("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}.checkpoint.val_loss.h5") into prediction_model_channel
    set val(prefix), val(size), val(step), val(scan), val(data_id), file(data) into prediction_data_channel

    """
    python ~/code/achilles/achilles.py train --file $train_data --run_id ${data_id}__${nb_resblock}__${nb_lstm}__${blstm} \
    --threads $params.threads --batch_size $params.batch_size --epochs $params.epochs --dropout $params.dropout \
    --rc_dropout $params.dropout --nb_residual_blocks $nb_resblock --nb_rnn $nb_lstm --activation softmax --window_size $size \
    --bi $blstm
    """

}

process EvaluationGenerator {
    
    /* Create independent evaluation datasets across training datasets for each evaluation directory, i.e.
       each evaluation directory is used to generate evaluation datasets for all models while excluding data 
       present in the training dataset used for training the model.

       If there a 8 parameter combinations and datasets of training data, and 2 evaluation directories, then 
       each evaluation directory generates 8 evaluation datasets with the same window settings as the channeled
       training dataset / model for a total of 16 evaluation datasets. 
    */

    tag "Evaluation: ${prefix_paths_tuple[0]} -- Model: $model_id -- Processing: $params.eval_max (windows/class)"
    publishDir "$params.outdir/$prefix/data/evaluation", mode: "copy", pattern: "*.eval.h5"

    // Unable to unpack prefix and paths tuple because set not compatible with each?

    input:
    each prefix_paths_tuple from evaluation_input_channel
    set model_id, file(model_file) from evaluation_model_channel // This is passed on to ModelEvaluation 
    set prefix, size, step, scan, data_id, file(data) from evaluation_data_channel

    output:
    set val(prefix), val("${prefix_paths_tuple[0]}__${model_id}"), file(model_file), file("${prefix_paths_tuple[0]}__${model_id}.eval.h5") into evaluation_channel

    """
    python ~/code/achilles/achilles.py make --dirs ${prefix_paths_tuple[1]} --out ${prefix_paths_tuple[0]}__${model_id}.eval.h5 \
    --validation $params.validation --window_size $size --window_max $params.eval_max --window_step $step --window_scan $scan \
    --raw --chunk_size $params.chunk_size
    """
}

process ModelEvaluation {

    tag "Evaluation ID: $eval_id"
    publishDir "$params.outdir/$prefix/evaluation", mode: "copy", pattern: "*.csv"

    input:
    set prefix, eval_id, file(eval_model), file(eval_data) from evaluation_channel

    output:
    set eval_id, file("${eval_id}.csv") into evaluation_summary

    """
    python ~/code/achilles/achilles.py evaluate --data_files $eval_data --model_files $eval_model \
    --batch_size $params.batch_size --threads $params.threads --data_path data --output_file ${eval_id}.csv
    """

}

process ModelPrediction {

    tag { "Prediction: ${prefix_paths_tuple[0]} -- Slices: $slice -- Model: $model_id" }
    
    publishDir "$params.outdir/$prefix/prediction", mode: 'copy', pattern: '*.pdf'
    publishDir "$params.outdir/$prefix/prediction", mode: 'copy', pattern: '*.csv'  
    publishDir "$params.outdir/$prefix/prediction/failed", mode: 'copy', pattern: '*.txt'

    input:
    each prefix_paths_tuple from prediction_input_channel
    each slice from slice_channel // Prediction for each slice size
    
    set model_id, file(model_file) from prediction_model_channel
    set prefix, size, step, scan, data_id, file(data) from prediction_data_channel

    output:
    file("${prefix_paths_tuple[0]}__${model_id}__slice${slice}.pdf")
    file("${prefix_paths_tuple[0]}__${model_id}__slice${slice}.csv")
    file("${prefix_paths_tuple[0]}__${model_id}__slice${slice}.fail.txt") // Files which failed prediction (unable to read)

    // Window step is same as window size for non-overlapping sequence of prediction windows.
    // Dirs takes all reads from the given directories (no maximum).

    """
    python ~/code/achilles/achilles.py --agg pevaluate --dirs ${prefix_paths_tuple[1]} --model_file $model_file \
    --windows $slice --window_size $size --window_step $size --prefix ${prefix_paths_tuple[0]}__${model_id}__slice${slice} \
    --batches $params.pred_batch --exclude $data --labels $labels --random --raw
    """

}
