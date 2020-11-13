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
BASE=/data/gpfs/projects/punim1384/achilles  # <-- Achilles stuff on partition to load (outside container)
DATA=/data/achilles                          # <-- Achilles loaded data (inside container)

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
  --dataset test_training.hd5 \
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
If there are fewer than --max_windows reads in the database, the sample will contain repeat or non-unique reads!
```

Window size here pre-determines the tensor dimensions for input into the convolutional residual block layer of the nets with a total training size of `100,000 * (50, 1, 200, 1)` for each tag combination / prediction class. Training and validation sets are split (`--validation`) - besides the initial random sample from the reads in the database, the order of blocks of consecutive tensors deriving from a random starting point and sequence of overlapping windows (`(50, 1, 200, 1)` = 50 consecutive overlapping windows of size 200 that 'scan' the read signal) is radnomised before written to the `--dataset {name}.hd5` (total data) and `{name}.training.hd5` (training-validation data).

`HDF5` datasets are currently structured as follows:





You can drop the entire database to remove all traces of the indexed reads:

```
achilles pm drop --force
```
