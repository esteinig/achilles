#!/usr/bin/env nextflow

/*
 *  This pipeline is part of the AI-based pathogen detection library 'Achilles'.
 *
 *  Pipeline            Achilles
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

                        Data Selection
==================================================================




                      Variable Parameters
==================================================================

window_max    =    $params.window_max
window_size   =    $params.window_size
window_step   =    $params.window_step
window_scan   =    $params.window_scan
nb_rb         =    $params.nb_rb
nb_lstm       =    $params.nb_lstm
blstm         =    $params.blstm
pred_slices   =    $params.pred_slices

                       Fixed Parameters
==================================================================

validation    =    $params.validation
chunck_size   =    $params.chunk_size
dropout       =    $params.dropout
threads       =    $params.threads
epochs        =    $params.epochs
batch_size    =    $params.batch_size
eval_max      =    $params.eval_max
pred_batch    =    $params.pred_batch

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

data_paths_channel   = Channel.fromPath(get_data_dirs(params.dir), type: 'dir')
                              .map { get_fast5_dirs(it)  }

data_prefix_channel = Channel.fromPath(get_data_dirs(params.dir), type: 'dir')
                             .map { it.getName() }

// Channels for feeding hyperparameters:

window_max_channel  = Channel.from(params.window_max)
window_size_channel = Channel.from(params.window_size)
window_step_channel = Channel.from(params.window_step)
window_scan_channel = Channel.from(params.window_scan)

resblock_channel    = Channel.from(params.nb_rb)
lstm_channel        = Channel.from(params.nb_lstm)
blstm_channel       = Channel.from(params.blstm)

process DataGenerator {
   
    tag { "Data: $prefix -- Dataset: $max (windows/class) $size (size) $step (step) $scan (scan)"  }
    publishDir { "$params.outdir/$prefix/data" }

    input:
    // Data directory path and string:
    val path_str from data_paths_channel
    val prefix from data_prefix_channel

    // Dataset processing combination:
    each max from window_max_channel
    each size from window_size_channel
    each step from window_step_channel
    each scan from window_scan_channel

    output:
    set val("$prefix"), val("$size"), val("${prefix}__${max}__${size}__${step}__${scan}"), file("${prefix}__${max}__${size}__${step}__${scan}.training.h5") into training_data
    set val("${prefix}__${max}__${size}__${step}__${scan}"), file("${prefix}__${max}__${size}__${step}__${scan}.h5") into evaluation_generator, prediction_generator
    
    // Note that each make samples randomly from the class directories, therefore the same files may be present
    // across training data sets (but not evaluation data, see below)

    """
    python ~/code/achilles/achilles.py make --dirs $path_str --out ${prefix}__${max}__${size}__${step}__${scan}.h5 \
    --validation $params.validation --window_size $size --window_max $max --window_step $step --window_scan $scan \
    --raw --chunk_size $params.chunk_size
    """

}



process Training {

    tag { "Dataset: $data_id -- Architecture: $nb_resblock x RB, $nb_lstm x LSTM, BLSTM = $blstm" }
    publishDir { "$params.outdir/$prefix/training" }

    // All combinations of datasets (with prefix and window size)
    // and neural network architectures for training:
    input:
    set prefix, window_size, data_id, file(training) from training_data

    each nb_resblock from resblock_channel
    each nb_lstm from lstm_channel
    each blstm from blstm_channel

    output:
    set val("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}"), file("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}/${data_id}__${nb_resblock}__${nb_lstm}__${blstm}.checkpoint.val_loss.h5") into evaluation_model, prediction_model
    set val("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}"), file("${data_id}__${nb_resblock}__${nb_lstm}__${blstm}/${data_id}__${nb_resblock}__${nb_lstm}__${blstm}.model.history") into training_summary

    """
    python ~/code/achilles/achilles.py train --file $training --run_id ${data_id}__${nb_resblock}__${nb_lstm}__${blstm} \
    --threads $params.threads --batch_size $params.batch_size \
    --epochs $params.epochs --dropout $params.dropout --rc_dropout $params.dropout --nb_residual_blocks $nb_resblock \
    --nb_rnn $nb_lstm --activation softmax --window_size $window_size --bi $blstm \
    --output_file ${data_id}__${nb_resblock}__${nb_lstm}__${blstm}.h5
    """

}


    
// // Training

// process TrainingGenerator {

//     tag "Prefix: $prefix -- Windows: $max $size $step $scan"
//     publishDir "${params.outdir}/data/training", mode: 'copy'

//     input:
//     val prefix from params.prefix
    // val max from window_max_channel
    // val size from window_size_channel
    // val step from window_step_channel
    // val scan from window_scan_channel
//     output:
//     set "${prefix}_${max}_${size}_${step}_${scan}", file([prefix, max, size, step, scan].join("_") + "_training.h5") into training_data
//     set "${prefix}_${max}_${size}_${step}_${scan}", file([prefix, max, size, step, scan].join("_") + ".h5") into evaluation_generator, prediction_generator  // The non-split DataSet has path: data/files

//     """
//     python ~/code/achilles/achilles.py make --dirs $dirs --out ${prefix}_${max}_${size}_${step}_${scan}.h5 \
//     --validation $params.validation --window_length $size --max_windows_per_class $max --window_step $step \
//     --max_windows_per_read $scan --raw --chunk_size $params.chunk_size
//     """
// }

    // process Training {

    //     tag { id }
    //     publishDir "${params.outdir}/training", mode: 'copy'

    //     input:
    //     set id, file(training) from training_data

    //     output:
    //     set id, file("${id}/${id}.checkpoint.val_loss.h5") into evaluation_model, prediction_model
    //     set id, file("${id}/${id}.model.history") into training_summary

    //     """
    //     python ~/code/achilles/achilles.py train --file $training --run_id $id --threads $params.threads --batch_size $params.batch_size \
    //     --epochs $params.epochs --dropout $params.dropout --rc_dropout $params.dropout --nb_residual_blocks $params.nb_rb \
    //     --nb_rnn $params.nb_lstm --activation softmax --signal_length $params.window_size --output_file ${id}.h5 --no_bi
    //     """
    // }

    // // Evaluation of large independently sampled data set

    // process EvaluationGenerator {

    //     tag { id }
    //     publishDir "${params.outdir}/data/evaluation", mode: 'copy'

    //     input:
    //     set id, file(train_data) from evaluation_generator

    //     output:
    //     set id, file("${id}_eval.h5") into evaluation_data

    //     """
    //     python ~/code/achilles/achilles.py make --dirs $eval_dirs --out ${id}_eval.h5 --validation 0 \
    //     --window_length $params.window_size --max_windows_per_class $params.eval_max  --window_step $params.window_step \
    //     --max_windows_per_read $params.window_scan --exclude $train_data --raw --chunk_size $params.chunk_size
    //     """
    // }

    // process Evaluation {

    //     tag { id }
    //     publishDir "${params.outdir}/evaluation", mode: 'copy'

    //     input:
    //     set id, file(eval_data) from evaluation_data
    //     set id, file(eval_model) from evaluation_model

    //     output:
    //     set id, file("eval.csv") into evaluation_summary

    //     """
    //     python ~/code/achilles/achilles.py evaluate --data_files $eval_data --model_files $eval_model \
    //     --batch_size $params.batch_size --threads $params.threads --data_path data --output_file eval.csv
    //     """

    // }

    // // Prediction evaluation by sampling consecutive slices from Fast5 files 
    // // for normalized product of output probabilities and class prediction
    // // over different slice sizes, produces confusion matrix

    // slices = Channel.from(params.pred_slices)

    // process PredictionEvaluation {

    //     tag { [id, slice].join(":") }
    //     publishDir "${params.outdir}/prediction", mode: 'copy'

    //     input:
    //     val slice from slices // Prediction for each slice size

    //     set id, file(train_data) from prediction_generator  // For excluding files used in training
    //     set id, file(pred_model) from prediction_model
        
    //     output:
    //     file("${slice}.pdf")
    //     file("${slice}.csv")

    //     // Window step is same as window size for non-overlapping sequence of prediction windows.
    //     // Dirs takes all reads from the given directories (no maximum).

    //     """
    //     python ~/code/achilles/achilles.py --agg pevaluate --dirs $pred_dirs --model_file $pred_model \
    //     --windows $slice --window_size $params.window_size --window_step $params.window_size \
    //     --prefix $slice --batches $params.pred_batch --exclude $train_data --labels $labels --random --raw
    //     """
    // }
