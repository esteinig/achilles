#!/usr/bin/env nextflow

/*
 *  This pipeline is part of the AI-based pathogen detection library 'Achilles'.
 *
 *  Pipeline            Achilles Training
 *  Version             0.1
 *  Description         Training and evaluation pipelines for architecture and hyperparameter performance with Achilles.
 *                      Specifically, this pipeline constructs a single dataset for training and uses the trained model
 *                      on a distinct set of evaluation and prediction directories .
 *
 *  Container           Singularity GPU and Tensorflow-GPU 1.8 (--nv)
 *
 */

log.info """
     ACHILLES-SINGLE v.0.1
==================================
id           = $params.id
dirs         = $params.dirs
labels       = $params.labels
outdir       = $params.outdir
window_max   = $params.window_max
window_size  = $params.window_size
window_step  = $params.window_step
window_scan  = $params.window_scan
validation   = $params.validation
chunck_size  = $params.chunk_size
nb_rb        = $params.nb_rb
nb_lstm      = $params.nb_lstm
blstm        = $params.blstm
dropout      = $params.dropout
epochs       = $params.epochs
threads      = $params.threads
batch_size   = $params.batch_size
eval_dirs    = $params.eval_dirs
eval_max     = $params.eval_max
pred_dirs    = $params.pred_dirs
pred_batch   = $params.pred_batch
pred_slices  = $params.pred_slices
==================================
"""

workdir = System.getProperty("user.dir");

// If directories from config files are collections, 
// prepare these as strings for terminal input in Achilles:

if (params.dirs instanceof Collection) {
    dirs = params.dirs.join(',')
} else {
    dirs = params.dirs
}

if (params.eval_dirs instanceof Collection) {
    eval_dirs = params.eval_dirs.join(',')
} else {
    eval_dirs = params.eval_dirs
}

if (params.pred_dirs instanceof Collection) {
    pred_dirs = params.pred_dirs.join(',')
} else {
    pred_dirs = params.pred_dirs
}

if (params.labels instanceof Collection) {
    labels = params.labels.join(',')
} else {
    labels = params.labels
}

if (params.pred_slices instanceof String) {
    pred_slices = params.pred_slices.tokenize(',')
} else {
    pred_slices = params.pred_slices
}

process TrainingGenerator {

    tag { id }
    publishDir "${params.outdir}/data/training", mode: 'copy'

    input:
    val id from params.id

    output:
    set id, file("${id}_training.h5") into training_data
    set id, file("${id}.h5") into evaluation_generator, prediction_generator  // The non-split DataSet has path: data/files

    """
    python ~/code/achilles/achilles.py make --dirs $dirs --out ${id}.h5 --validation $params.validation \
    --window_length $params.window_size --max_windows_per_class $params.window_max \
    --window_step $params.window_step --max_windows_per_read $params.window_scan --raw --chunk_size $params.chunk_size
    """
}

process Training {

    tag { id }
    publishDir "${params.outdir}/training", mode: 'copy'

    input:
    set id, file(training) from training_data

    output:
    set id, file("${id}/${id}.checkpoint.val_loss.h5") into evaluation_model, prediction_model
    set id, file("${id}/${id}.model.history") into training_summary

    """
    python ~/code/achilles/achilles.py train --file $training --run_id $id --threads $params.threads --batch_size $params.batch_size \
    --epochs $params.epochs --dropout $params.dropout --rc_dropout $params.dropout --nb_residual_blocks $params.nb_rb \
    --nb_rnn $params.nb_lstm --activation softmax --signal_length $params.window_size --output_file ${id}.h5 --no_bi
    """
}

process EvaluationGenerator {

    tag { id }
    publishDir "${params.outdir}/data/evaluation", mode: 'copy'

    input:
    set id, file(train_data) from evaluation_generator

    output:
    set id, file("${id}_eval.h5") into evaluation_data

    """
    python ~/code/achilles/achilles.py make --dirs $eval_dirs --out ${id}_eval.h5 --validation 0 \
    --window_length $params.window_size --max_windows_per_class $params.eval_max  --window_step $params.window_step \
    --max_windows_per_read $params.window_scan --exclude $train_data --raw --chunk_size $params.chunk_size
    """
}

process Evaluation {

    tag { id }
    publishDir "${params.outdir}/evaluation", mode: 'copy'

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

slices = Channel.from(params.pred_slices)

process PredictionEvaluation {

    tag { [id, slice].join(":") }
    publishDir "${params.outdir}/prediction", mode: 'copy'

    input:
    val slice from slices // Prediction for each slice size

    set id, file(train_data) from prediction_generator  // For excluding files used in training
    set id, file(pred_model) from prediction_model
    
    output:
    file("${slice}.pdf")
    file("${slice}.csv")

    // Window step is same as window size for non-overlapping sequence of prediction windows.
    // Dirs takes all reads from the given directories (no maximum).

    """
    python ~/code/achilles/achilles.py --agg pevaluate --dirs $pred_dirs --model_file $pred_model \
    --windows $slice --window_size $params.window_size --window_step $params.window_size \
    --random --prefix $slice --batches $params.pred_batch --exclude $train_data --labels $labels --raw
    """
}
