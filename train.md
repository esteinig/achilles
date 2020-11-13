## Setting up on Spartan

This is for our partition on Spartan using `Singularity`: 

```
module load singularity/3.6.3
```

You can update the container by pulling the latest version from `DockerHub` but should not be necessary for now (unless you want to make some pull requests and update the repository code):

```
singularity pull achilles.sif docker://esteinig/achilles:latest
```

Then assign base directory and container execution alias for working with the container loading the data directory into the `/data/achilles` path inside the container:

```
BASE=/data/gpfs/projects/punim1384/achilles  # Achilles stuff on partition to load (outside container)
DATA=/data/achilles                          # Achilles loaded data (inside container)

alias achilles="singularity run -B ${BASE}/data:${DATA} ${BASE}/containers/achilles.sif"
```

Test `PoreMongo` and `Achilles` CLI - the latter may raise some `tensorflow-gpu` warnings because we have not passed the (non available) GPU drivers through to the `Singularity` container, which is fine for now:

```bash
achilles pm --help
achilles achilles --help
```

## MongoDB Service

Start a `MongoDB` service in background screen on login node - do not use for intensive tasks! It's a bit sneaky but we will not use a lot of memory or processors for samplimg from the database so it should be fine

```bash
screen -S mongo-service -d -m bash -c "singularity run -B ${BASE}/dbs/fuyi:/data/db ${BASE}/containers/mongo.sif"
```

This will open the user specific `MongoDB` database in `${BASE}/dbs` and serve on `localhost:27017` by default - you can open the screen with `screen -r mongo-service` to confirm the client is running and detach with `Ctrl + A + D`.

## PoreMongo Client

Now test a query against the database using `PoreMongo` CLI. This should give a connection success log and an empty tag table.

```bash
achilles pm display
```

Next step is to index some test files with the `PoreMongo `client for loading data into the DB. Each time reads are indexed from a (multi-)`Fast5` or directory of `Fast5` files, the objects are assigned user provided tags: usually they will be somethign like `R9.4`, `Staphylococcus aureus`, or `FAO92834` flow cell numbers to access sample tags later for generating the training sets; a database (`--db`) can be specified to separate data sets and must be specified in subsequent queries and sampling operations to change the default `poremongo` DB

```bash
achilles pm index -f ${DATA}/test_data/human.fast5 -t R9.4,Human,TestData
achilles pm index -f ${DATA}/test_data/saureus.fast5 -t R9.4,MRSA,TestData 
```

Now run the display task again to check out the new tagged reads in the DB which we can now sample from (using `--total` for a read total count query and `--quiet` to suppress connection log output)

```bash
achilles pm display -tq
```

Next we sample 20 signal reads with the `Human` tag randomly from the database using the `sample` task with display (`-d`) and suppressed log output (`-q`) so the results are printed cleanly into the console:

```bash
achilles pm sample -dq -s 20 -t Human
```

Repeat the command to confirm that reads with the `Human` tag are sampled randomly

```bash
achilles pm sample -dq -s 20 -t Human
```

Output columns are in order:

```
read_id   read_file   tags    read_uuid
```


## Achilles Training Dataset

Next we use the `Achilles` task `create` to sample and extract a set of tensors for training the nets with the `PoreMongo` client. We sample from tag or combinations of tags (`&`) in the command line interface, which are then used as classes for predictions (`0, 1, 2 ...` for each tag combination):

```
achilles achilles create \
  --tags TestData,Human TestData,MRSA \  # or use --global_tags
  --dataset ${DATA}/test_training.hd5 \
  --max_windows 100000 \
  --max_windows_per_read 50 \
  --window_size 200 \
  --window_step 0.1 \
  --sample 10000 \
  --proportion equal \
  --global_tags R9.4 \
  --validation 0.3
```

Here we sample to train a network on a binary class prediction of `Human` and `MRSA` signal from the `TestData` and `R9.4` tags applied to all indexed reads. We extract 100,000 raw data acquisition value windows per tag combination, where we sample a maximum of 50 windows in a continous sequence of windows of size 200 values with overlap of 10% from an equal sample of 10,000 reads for each tag combination (`--tags` & `--global_tags`).

```
If there are fewer than --sample reads in the database, the sample will contain non-unique reads!
```

Window size here pre-determines the tensor dimensions for input into the convolutional residual block layer of the nets with a total training size of `2 * 100000 * (1, 1, 200, 1)` total window samples (!) with `2 * 100000` labels for both tag combination / prediction classes used in the example. This then corresponds to `100000 / 50 = 2000` reads sampled for 50 consecutive windows per read per tag / label. Training and validation sets are split (`--validation`) - besides the initial random sample from the reads in the database, the order of blocks of overlapping windows (50 consecutive overlapping windows of size 200 `(50, 1, 200, 1)` that 'scan' the read signal) is randomised before written to the `--dataset {name}.hd5` (total data) and `{name}.training.hd5` (training-validation data) to the linked `Singularity` directory `$DATA`.

`HDF5` standard datasets are currently structured as follows:

* `/data/data`: `2 * 100,000` tensor arrays of shape `(50, 1, 200, 1)` shuffled
* `/data/labels`: vector of  one-hot encoded prediction labels, ordered as `data`
* `/data/decoded`: vector of numeric prediction labels, ordered as `data`
* `/data/reads`: vector of `read_id` of the signal reads used in this dataset, no specific order

In the training-validation set the data are in `/training/data` and `/validation/data` including label vectors in the corresponding paths.

## Achilles GPU Training

This section is fairly specific to the Spartan cluster. I wrote a `SLURM` job template here, at the moment the best way is to copy the template into the directory with the training file and run it from there, so in this case:

```
cd $DATA/test_data
cp $BASE/slurm/train.slurm .

# --> Edit the SLURM job file to modify training (see below) <--

sbatch train.slurm
```

Here is what the `SLURM` file currently does - note the different parameters. I think this default configuration on a 200 x window size with 10% overlap and batch size 2048 uses around 10 GB RAM - enough for the P100 GPUs on the cluster (12 GB). If you increase the window size for tensor sampling to e.g. 400 x window size withh 10% overlap it is recommended to halve the batch size as both parameters are major determinators of GPU memory required

```
#!/bin/bash
#SBATCH --job-name train_achilles
#SBATCH --nodes 1
#SBATCH --account punim1384
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:1
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=2
#SBATCH --qos gpgpuresplat

# ASSUME RUNNING IN TRAIN FILE DIRECTORY

module purge
module load singularity/3.6.3

TRAIN=test_training.training.hd5
OUTDIR=training_run

echo "Running training job in $PWD : $TRAIN --> $OUTDIR"

BASE=/data/gpfs/projects/punim1384/achilles

BATCH_SIZE=2048
EPOCHS=300
DROPOUT=0.2
LSTM=1
RESBLOCK=1

BASE_TRAIN=$(dirname $TRAIN)
TRAIN_NAME=$(basename $TRAIN)

echo "Directory of training $TRAIN : $BASE_TRAIN"
echo "Base name of $TRAIN : $TRAIN_NAME"


singularity run --nv -B $PWD:/data/achilles/training -B $BASE_TRAIN:/data/achilles/train_data \
     ${BASE}/containers/achilles.sif achilles train \
     --batch_size $BATCH_SIZE \
     --epochs $EPOCHS \
     --dropout $DROPOUT \
     --lstm $LSTM \
     --residual_block $RESBLOCK \
     --threads 2 \
     --outdir /data/achilles/training/${OUTDIR} \
     --run_id $OUTDIR \
     --file /data/achilles/train_data/$TRAIN_NAME \
     --verbose
```

Follow training in the `SLURM` log:

```
tail -f slurm-*.out
```

*Cleaning up after testing*

You can drop the entire database to remove all traces of the indexed reads:

```
achilles pm drop --force
```
