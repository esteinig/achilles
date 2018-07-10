#!/usr/bin/env nextflow

/*
 *  This pipeline is part of the AI-based pathogen detection library 'Achilles'.
 *
 *  Pipeline            Achilles Sample
 *  Version             0.1
 *  Description         Generate different random samples from host and pathogen read data for training with Achilles.

 *  Container           Singularity GPU and Tensorflow-GPU 1.8 GPU
 *
 */

log.info """
ACHILLES-SAMPLE NF v.0.1
=========================

id           = $params.id
dirs         = $params.dirs
labels       = $params.labels

sample        = $params.sample
sample_size   = $params.sample_size
sample_regime = $params.sample_regime

host_chr      = $params.host_chr

nb_eval       = $params.nb_eval
nb_peval      = $params.nb_peval

singularity   = $params.singularity
profile       = $params.profile

"""

workdir = System.getProperty("user.dir");

/* Prepare training / validation and independent evaluation data 
 * from human chromosomes and target pathogen.
*/

regime = {
    
    training: {
        exclude: "FAB3218",
        n: 10000
        pathogen: ["flowcell_1"]
        host: {
            "minimal": ["human_chr2", "human_chr4"],
            "diverse": ["human_chr2", "human_chr4", "human_chr8", "human_chr14", "human_chr20", "human_chrX"]
        },
    },

    evaluation: {
        exclude: "FAB3218",
        n: 5000
        pathogen: ["flow_cell_2", "flowcell_3", "flowcell_4", "flowcell_5"]
        host: {
            "minimal_same": ["human_chr2", "human_chr4"],
            "diverse_same": ["human_chr2", "human_chr4", "human_chr8", "human_chr14", "human_chr20", "human_chrX"]
            "minimal_general": ["human_chr3", "human_chr5"],
            "diverse_general": ["human_chr3", "human_chr5", "human_chr9", "human_chr15", "human_chr21", "human_chrY"]
        }
    },

    simulation:  {
        host: ["patient_sample_1"]
        pathogen: ["flow_cell_2", "flowcell_3", "flowcell_4", "flowcell_5"]
    },

}

process DataSampler {



}

process SimulationData {


}