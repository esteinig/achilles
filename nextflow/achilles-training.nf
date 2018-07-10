#!/usr/bin/env nextflow

/*
 *  This pipeline is part of the AI-based pathogen detection library 'Achilles'.
 *
 *  Pipeline            Achilles Training
 *  Version             0.1
 *  Description         Training and evaluation pipelines for architecture and hyperparameter performance with Achilles.

 *  Container           Singularity GPU and Tensorflow-GPU 1.8 (--nv)
 *
 */

log.info """
ACHILLES-TRAINING NF v.0.1
===========================

id          = $params.id
dirs        = $params.dirs
labels      = $params.labels
outdir      = $params.outdir
window_max  = $params.window_max
window_size = $params.window_size
window_step = $params.window_step
window_scan = $params.window_scan
dac         = TRUE
med         = FALSE
pa          = FALSE
validation  = 0.3
nb_resblocks= $params.nb_resblocks
nb_lstm     = $params.nb_lstm
blstm       = $params.blstm
batch_size  = $params.batch_size
epochs      = $params.epochs
singularity = $params.singularity
profile     = $params.profile

"""

workdir = System.getProperty("user.dir");

process DataGenerator {

    tag { id }
    publishDir "${params.outdir}/data"

    output:
    set id, file("${id}_training.fastq") into training
    set id, file("${id}.h5") into data_summary

    """
    achilles --dirs $params.dirs --out "${id}.h5" --validation $params.validation
    --window_size$params.window_size} --max $params.window_max
    --window_step $params.window_step -mw $params.window_scan --raw --chunk_size 5000
    """
}

process Training {

    tag { id }
    publishDir "${params.outdir}/log"

    input:
    set id, file("${id}_training.fastq")

    output:
    set id, file("${id}_training.checkpoint.h5") into evaluation
    set id, file("${id}_training.log") into training_summary

    """
    
    """

}

process Evaluation {

    tag { id }
    publishDir "${params.outdir}/evaluation"

    input:
    set id, file("${id}_training.fastq")

    output:
    set id, file("${id}_training.checkpoint.h5") into prediction_evaluation
    set id, file("evaluation/${id}_eval.log") into evaluation_summary

    """
    """
    

}

process Prediction-Evaluation {

    tag { id }
    publishDir "${params.outdir}/prediction_evaluation"

    input:
    set id, file("${id}_training.fastq")

    output:
    set id, file("${id}_training.checkpoint.h5") into prediction_evaluation
    set id, file("evaluation/${id}_eval.log") into evaluation_summary

    """
    """

}