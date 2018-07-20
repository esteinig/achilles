#!/usr/bin/env nextflow

/*
 *  This pipeline is part of the AI-based pathogen detection library 'Achilles'.
 *
 *  Pipeline            Achilles
 *  Version             0.1
 *  Description         Training and evaluation pipelines for architecture and hyperparameter performance with Achilles.
 *                      Specifically, this pipeline constructs a single dataset for training the model and uses the 
 *                      model with highest validation accuray on a distinct set of directories contaning Fast5 files for
 *                      evaluation and prediction. This workflow starts from the data set generation stage, so relies on
 *                      user-prepared directories containing selection of Fast5 files to use in training, evaluation and 
 *                      prediction tasks; best practice for random, recursive selection of the Fast5 libraries in storage 
 *                      is to use the Achilles command-line task 'select' with parameter input or config file.
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

def get_paths(dirs) {

    if (dirs instanceof Collection) {
        p_dirs = dirs.collect { Paths.get(it).toAbsolutePath() }
    } else {
        // If comma-separated string:
        t_dirs = dirs.tokenize(",")
        p_dirs = t_dirs.collect { Paths.get(it).toAbsolutePath() }
    }
    s_dirs = p_dirs.join(',')
    print_dirs = p_dirs.collect { get_print_dir( it.toString() ) }

    return [s_dirs, print_dirs]
}

// If directories from config files are collections: prepare these as strings for terminal input in Achilles

(dirs, print_dirs) = get_paths(params.dirs)
(eval_dirs, print_eval_dirs) = get_paths(params.eval_dirs)
(pred_dirs, print_pred_dirs) = get_paths(params.pred_dirs)

if (params.labels instanceof Collection) {
    labels = params.labels.join(',')
} else {
    labels = params.labels
}

// If pred_slices is a comma-separated string (command line Nextflow): tokenize

if (params.pred_slices instanceof String) {
    pred_slices = params.pred_slices.tokenize(',')
} else {
    pred_slices = params.pred_slices
}

log.info """
==================================================================
                      ACHILLES NEXTFLOW 
             (TRAINING - EVALUATION - PREDICTION)
                            v.0.1
==================================================================

id            =    $params.id
dirs          =    $print_dirs
labels        =    $params.labels
outdir        =    $params.outdir
window_max    =    $params.window_max
window_size   =    $params.window_size
window_step   =    $params.window_step
window_scan   =    $params.window_scan
validation    =    $params.validation
chunck_size   =    $params.chunk_size
nb_rb         =    $params.nb_rb
nb_lstm       =    $params.nb_lstm
blstm         =    $params.blstm
dropout       =    $params.dropout
epochs        =    $params.epochs
threads       =    $params.threads
batch_size    =    $params.batch_size
eval_dirs     =    $print_eval_dirs
eval_max      =    $params.eval_max
pred_dirs     =    $print_pred_dirs
pred_batch    =    $params.pred_batch
pred_slices   =    $params.pred_slices

==================================================================
==================================================================
"""

// Training

process TrainingGenerator {

    tag { id }
    publishDir "${params.outdir}/data/training", mode: "copy", pattern: "*.h5"

    input:
    val id from params.id

    output:
    set id, file("${id}.training.h5") into training_data
    set id, file("${id}.h5") into evaluation_generator, prediction_generator  // The non-split DataSet has path: data/files

    """
    python ~/code/achilles/achilles.py make --dirs $dirs --out ${id}.h5 --validation $params.validation \
    --window_size $params.window_size --window_max $params.window_max \
    --window_step $params.window_step --window_scan $params.window_scan --raw --chunk_size $params.chunk_size
    """
}

process ModelTraining {

    tag { id }
    publishDir "${params.outdir}/training", mode: "copy"
    publishDir "${params.outdir}/training", mode: "copy"

    input:
    set id, file(training) from training_data

    output:
    set id, file("${id}.checkpoint.val_loss.h5") into evaluation_model, prediction_model
    set id, file("${id}.model.history") into training_summary

    """
    python ~/code/achilles/achilles.py train --file $training --run_id $id --threads $params.threads --batch_size $params.batch_size \
    --epochs $params.epochs --dropout $params.dropout --rc_dropout $params.dropout --nb_residual_blocks $params.nb_rb \
    --nb_rnn $params.nb_lstm --activation softmax --window_size $params.window_size --bi $params.blstm
    """
}

// Evaluation of large independently sampled data set

process EvaluationGenerator {

    tag { id }
    publishDir "${params.outdir}/data/evaluation", mode: "copy", pattern: "*.eval.h5"

    input:
    set id, file(train_data) from evaluation_generator

    output:
    set id, file("${id}_eval.h5") into evaluation_data

    """
    python ~/code/achilles/achilles.py make --dirs $eval_dirs --out ${id}_eval.h5 --validation 0 \
    --window_size $params.window_size --window_max $params.eval_max  --window_step $params.window_step \
    --window_scan $params.window_scan --exclude $train_data --raw --chunk_size $params.chunk_size
    """
}

process ModelEvaluation {

    tag { id }
    publishDir "${params.outdir}/evaluation", mode: "copy"

    input:
    set id, file(eval_data) from evaluation_data
    set id, file(eval_model) from evaluation_model

    output:
    set id, file("eval.csv") into evaluation_summary

    """
    python ~/code/achilles/achilles.py evaluate --data_files $eval_data --model_files $eval_model \
    --batch_size $params.batch_size --threads $params.threads --data_path data --output_file eval.csv
    """

}

// Prediction evaluation by sampling consecutive slices from Fast5 files 
// for normalized product of output probabilities and class prediction
// over different slice sizes, produces confusion matrix

slices = Channel.from(params.pred_slices)

process ModelPrediction {

    tag { [id, slice].join(":") }
    publishDir "${params.outdir}/prediction", mode: "copy"

    input:
    val slice from slices // Prediction for each slice size

    set id, file(train_data) from prediction_generator  // For excluding files used in training
    set id, file(pred_model) from prediction_model
    
    output:
    file("${slice}.pdf")
    file("${slice}.csv")
    file("${slice}.fail.txt")

    // Window step is same as window size for non-overlapping sequence of prediction windows.
    // Dirs takes all reads from the given directories (no maximum).

    """
    python ~/code/achilles/achilles.py --agg pevaluate --dirs $pred_dirs --model_file $pred_model \
    --windows $slice --window_size $params.window_size --window_step $params.window_size \
    --prefix $slice --batches $params.pred_batch --exclude $train_data --labels $labels --random --raw
    """
}
