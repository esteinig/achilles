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

```bash
achilles_base='/data/gpfs/projects/punim1384/achilles'
alias achilles="singularity run -B ${achilles_base}/data:/data/achilles ${achilles_base}/containers/achilles.sif"
```

Test `PoreMongo` and `Achilles` CLI - the latter may raise some `tensorflow-gpu` warnings because we have not passed the (non available) GPU drivers through to the `Singularity` container, which is fine for now:

```bash
achilles pm --help
achilles achilles --help
```

## MongoDB Service

Start a `MongoDB` service in background screen on login node - do not use for intensive tasks! It's a bit sneaky but we will not use a lot of memory or processors for samplimg from the database so it should be fine

```bash
screen -S mongo-service -d -m bash -c "singularity run -B ${achilles_base}/dbs/fuyi:/data/db ${achilles_base}/containers/mongo.sif"
```

This will open the user specific `MongoDB` database in `${achilles_dir}/dbs` and serve on `localhost:27017` by default - you can open the screen with `screen -r mongo-service` to confirm the client is running and detach with `Ctrl + A + D`.

## PoreMongo Client

Assign the `Achilles` directory loaded inside the container to a useful environmental variable:

```bash
achilles_work=/data/achilles
```

Now test a query against the database using `PoreMongo` CLI. This should give a connection success log and an empty tag table.

```bash
achilles pm display
```

Next step is to index some test files with the `PoreMongo `client for loading data into the DB. Each time reads are indexed from a (multi-)`Fast5` or directory of `Fast5` files, the objects are assigned user provided tags: usually they will be somethign like `R9.4`, `Staphylococcus aureus`, or `FAO92834` flow cell numbers to access sample tags later for generating the training sets; a database (`--db`) can be specified to separate data sets and must be specified in subsequent queries and sampling operations to change the default `poremongo` DB

```bash
achilles pm index -f ${achilles_work}/test_data/human.fast5 -t R9.4,Human,TestData
achilles pm index -f ${achilles_work}/test_data/saureus.fast5 -t R9.4,MRSA,TestData 
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

