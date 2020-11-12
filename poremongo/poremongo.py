""" Basically PoreDB for Fast5 management in MongoDB """

import os
import time
import math
import tqdm
import json
import signal
import random
import shutil
import pymongo
import pandas
import logging
import paramiko
import uuid

import multiprocessing as mp

from scp import SCPClient
from pathlib import Path
from functools import reduce
from operator import or_, and_
from datetime import datetime, timedelta
from deprecation import deprecated
from mongoengine import connect
from mongoengine.queryset.visitor import Q
from pymongo.errors import ServerSelectionTimeoutError
from ont_fast5_api.fast5_interface import get_fast5_file

from pyfastaq import sequences
from apscheduler.schedulers.background import BackgroundScheduler

from poremongo.poremodels import Read
from poremongo.utils import run_cmd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s]   %(message)s",
    datefmt='%H:%M:%S'
)

VERSION = '0.4'


class PoreMongo:

    """ API for PoreMongo """

    def __init__(
        self,
        uri: str = None,
        config: Path or dict = None,
        connect: bool = False,
        ssh: bool = False,
        mock: bool = False
    ):
        disallowed = ['local']
        self.db_name = Path(uri).stem

        if self.db_name in disallowed:
            raise ValueError(f"Database can not be named: "
                             f"{', '.join(disallowed)}")

        self.uri = uri
        self.verbose = True

        self.logger = logging.getLogger(__name__)

        if config:
            self._parse_config(config)
        else:
            self.config = dict()

        self.mock = mock

        self.local = True if "localhost:27017" in self.uri else False

        self.ssh = None
        self.scp = None

        self.client: pymongo.MongoClient = pymongo.MongoClient(None)
        self.connected: bool = False

        self.db = None  # Client DB
        self.fast5 = None  # Fast5 collection

        self.mongod_proc = None
        self.mongod_path: Path or None = None
        self.mongod_pid: int or None = None

        if connect:
            self.connect(ssh=ssh, is_mock=mock)

    def start_mongodb(
        self, dbpath: Path = Path('~/.poremongo/db'), port: int = 27017
    ):

        dbpath.mkdir(parents=True, exist_ok=True)

        self.logger.info(f'Running mongod DB client at {dbpath}')

        proc = run_cmd(
            f'mongod --dbpath {dbpath} --port {port}', shell=True
        )

        self.mongod_proc = proc
        self.mongod_path = dbpath
        self.mongod_pid = int(proc.pid)

        self.logger.info(f'Started MongoDB with pid: {proc.pid}')

    def terminate_mongodb(self):

        self.logger.info(f'Terminated MongoDB with pid: {self.mongod_pid}')

        os.killpg(self.mongod_proc.pid, signal.SIGTERM)

    def _parse_config(self, config: Path or dict):

        if isinstance(config, Path):
            with config.open('r') as cfg:
                config_dict = json.load(cfg)
                self.config = config_dict
                try:
                    self.uri = config_dict["uri"]
                except KeyError:
                    raise KeyError(
                        "Configuration dictionary must contain key 'uri' "
                        "to make the connection to MongoDB."
                    )
        elif isinstance(config, dict):
            try:
                self.uri = config["uri"]
                self.config = config
            except KeyError:
                raise KeyError(
                    "Configuration dictionary must contain key 'uri' "
                    "to make the connection to MongoDB."
                )
        else:
            raise ValueError(
                "Config must be string path to JSON file or dictionary."
            )

    def is_connected(self):

        return True if self.client else False

    def connect(self, ssh: bool = False, is_mock: bool = False, **kwargs):

        self.logger.debug(
            f'Attempting to connect to: {self.decompose_uri()}'
        )

        try:
            self.client = connect(
                host=self.uri,
                serverSelectionTimeoutMS=10000,
                is_mock=is_mock,
                **kwargs
            )

            self.client.server_info()

        except ServerSelectionTimeoutError:
            self.connected = False
            self.logger.info(
                f'Failed to connect to: {self.decompose_uri()}'
            )
            return self.connected

        self.connected = True
        self.logger.info(
            f'Success! Connected to: {self.decompose_uri()}'
        )

        self.db = self.client.db    # Database connected
        self.fast5 = self.db.fast5  # Fast5 collection

        self.logger.info(
            'Default collection for PoreMongo is "fast5"'
        )

        if ssh:
            self.logger.debug(
                'Attempting to open SSH and SCP'
            )
            self.open_ssh()
            self.open_scp()
            self.logger.info(
                'Success! Opened SSH and SCP to PoreMongo'
            )

        return self.connected

    def disconnect(self, ssh=False):

        self.logger.debug(
            f'Attempting to disconnect from: {self.decompose_uri()}'
        )

        self.client.close()

        self.client, self.db, self.fast5 = None, None, None

        self.logger.info(
            f'Disconnected from: {self.decompose_uri()}'
        )

        if ssh:
            self.logger.info(
                'Attempting to close SSH and SCP'
            )
            self.close_ssh()
            self.close_scp()
            self.logger.info(
                'Closed SSH and SCP'
            )

    def open_scp(self):

        self.scp = SCPClient(
            self.ssh.get_transport()
        )

        return self.scp

    def close_scp(self):

        self.scp.close()

    def close_ssh(self):

        self.ssh.close()

    # TODO: Exceptions for SSH configuration file
    def open_ssh(self, config_file=None):

        if config_file:
            with open(config_file, 'r') as infile:
                config = json.load(infile)
                ssh_config = config["ssh"]
        else:
            ssh_config = self.config["ssh"]

        self.ssh = self.create_ssh_client(
            server=ssh_config["server"],
            port=ssh_config["port"],
            user=ssh_config["user"],
            password=ssh_config["password"]
        )

        return self.ssh

    @staticmethod
    def create_ssh_client(server, port, user, password):

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(server, port, user, password)

        return client

    ##########################
    #     Fast5 Indexing     #
    ##########################

    def index_fast5(
        self, files: [Path] = None, tags: list = None, store_signal: bool = False
    ):

        """ Main access method to index Fast5 files into MongoDB """

        for file in files:
            self.logger.info(f"Index signal data [ {self.db_name} ]: {file}")

            reads = []
            with get_fast5_file(str(file), mode="r") as f5:
                for read in f5.get_reads():
                    unique_identifier = uuid.uuid4()
                    fast5_path = file.absolute()
                    read = Read(
                        fast5=str(fast5_path),
                        uuid=str(unique_identifier),
                        tags=tags,
                        read_id=read.read_id,
                        signal_data=read.get_raw_data() if store_signal else []
                    )
                    reads.append(read)

            try:
                Read.objects.insert(reads)
                self.logger.info(f'Inserted {len(reads)} signal reads with tags [ {", ".join(tags)} ]')
            except:
                raise

    ##########################
    #     Poremongo Tags     #
    ##########################

    def tag(self, tags, path_query=None, name_query=None, tag_query=None,
            raw_query=None, remove=False, recursive=True, not_in=False):

        """

        :param tags:
        :param path_query:
        :param name_query:
        :param tag_query:
        :param remove:
        :param recursive:
        :param not_in
        :param raw_query
        :return:
        """

        if isinstance(tags, str):
            tags = (tags,)

        if isinstance(path_query, Path):
            path_query = str(path_query)

        objects = self.query(
            raw_query=raw_query,
            path_query=path_query,
            name_query=name_query,
            tag_query=tag_query,
            recursive=recursive,
            not_in=not_in
        )

        if remove:
            self.logger.info(f'Remove tags from query reads: {", ".join(tags)}')
            objects.update(pull_all__tags=tags)
        else:
            self.logger.info(f'Add tags to query reads: {", ".join(tags)}')
            objects.update(add_to_set__tags=tags)

    ##########################
    #     DB Summaries       #
    ##########################

    @staticmethod
    def get_tag_counts(
        tags: list or None = None,
        limit: int or None = 100
    ) -> (list, list):

        if tags:
            match = {"$match": {"tags": {"$in": tags}}}
        else:
            match = {"$match": {"tags": {"$not": {"$size": 0}}}}

        counts = [
            match,
            {"$unwind": "$tags"},
            {"$group": {
                    "_id": "$tags",
                    # "latest": {"$last": "$exp_start_time"},
                    "count": {"$sum": 1}
            }},
            {"$match": {
                    "count": {"$gte": 1}
            }},
            {"$sort": {"count": -1}},
        ]

        if limit:
            counts += [
                {"$limit": limit},
            ]

        return list(Read.objects.aggregate(*counts))


    ##########################
    #     DB Queries         #
    ##########################

    def query(
        self,
        raw_query: dict = None,
        path_query: str or list = None,
        tag_query: str or list = None,
        query_logic: str = "AND",
        abspath: bool = False,
        recursive: bool = True,
        not_in: bool = False,
        model: Read = Read,
    ):

        """ API for querying file models using logic chains

        ... on raw queries, file path or tag queries.

        MongoEngine queries (Q) are chained by bitwise operator
        logic (query_logic).

        Path, tag and name queries can also be chained with each other
        if at least two parameters given (all same operator: query_logic).

        """

        # TODO implement nested lists as query objects and nested logic chains?

        if raw_query:
            return model.objects(__raw__=raw_query)

        if isinstance(path_query, str):
            path_query = [path_query, ]
        if isinstance(tag_query, str):
            tag_query = [tag_query, ]

        # Path filter should ask for absolute path by default:
        if abspath and path_query:
            path_query = [
                os.path.abspath(pq) for pq in path_query
            ]

        # Path filter for selection:
        if path_query:
            path_queries = self.get_path_query(
                path_query, recursive, not_in
            )
        else:
            path_queries = list()

        if tag_query:
            tag_queries = self.get_tag_query(
                tag_query, not_in
            )
        else:
            tag_queries = list()

        queries = path_queries + tag_queries

        if not queries:
            # If there are no queries, return all models:
            return model.objects

        # Chain all queries (within and between queries)
        # with the same bitwise operator | or &

        query = self.chain_logic(queries, query_logic)

        return model.objects(query)

    @staticmethod
    def get_tag_query(tag_query, not_in):

        if not_in:
            return []  # TODO
        else:
            return [
                Q(tags=tq) for tq in tag_query
            ]

    @staticmethod
    def get_name_query(name_query, not_in):

        if not_in:
            return [
                Q(__raw__={
                    "name": {'$regex': '^((?!{string}).)*$'.format(string=nq)}
                }) for nq in name_query]  # case sensitive regex (not contains)
        else:
            return [
                Q(name__contains=nq) for nq in name_query
            ]

    # TODO: Abspath - on Windows, UNIX

    @staticmethod
    def get_path_query(path_query, recursive, not_in):

        if recursive:
            if not_in:
                return [Q(__raw__={
                    "fast5": {'$regex': '^((?!{string}).)*$'.format(string=pq)}
                }) for pq in path_query]  # case sensitive regex (not contains)
            else:
                return [
                    Q(fast5__contains=pq) for pq in path_query
                ]
        else:
            return [
                Q(fast5__exact=pq) for pq in path_query
            ]

    @staticmethod
    def chain_logic(iterable, logic):
        if logic in ("OR", "or", "|"):
            chained = reduce(or_, iterable)
        elif logic in ("AND", "and", "&"):
            chained = reduce(and_, iterable)
        else:
            raise ValueError(
                "Logic parameter must be one of (AND, and, &) or (OR, or, |)"
            )

        return chained

    def filter(
        self,
        queryset,
        limit: int = None,
        shuffle: bool = False,
        unique: bool = True
    ) -> list:

        """ Filter where query sets are now living in memory """

        query_results = list(queryset)  # Lives happily ever after in memory.

        if unique:
            self.logger.info('Take the set of query results')
            query_results = list(
                set(query_results)
            )

        if shuffle:
            self.logger.info('Shuffle query results')
            random.shuffle(query_results)

        if limit:
            self.logger.info(f'Limit to first {limit} query results')
            query_results = query_results[:limit]

        return query_results

    #########################
    #   Cleaning DB + QC    #
    #########################

    # TODO: parse FASTQ for basecalled reads and attach Sequence model to Fast5

    @staticmethod
    def average_quality(quals):

        """
        Receive the integer quality scores of a read and return
        the average quality for that read

        First convert Phred scores to probabilities,
        calculate average error probability and
        convert average back to Phred scale.

        https://gigabaseorgigabyte.wordpress.com/2017/06/26/
        averaging-basecall-quality-scores-the-right-way/
        """

        return -10 * math.log(sum(
            [10 ** (q / -10) for q in quals]
        ) / len(quals), 10)

    # SELECTION AND MAPPING METHODS

    def schedule_run(
        self,
        fast5,
        outdir="run_sim_1",
        scale=1.0,
        timeout=None
    ):
        """Schedule a run extracted from sorted completion times
        for reads contained in Fast5 models. Scale adjusts the
        time intervals between reads. Use with group_runs to
        extract reads from the same sequencing run.

        Based on taeper - Michael Hall - https://github.com/mhall88/taeper

        :param fast5:
        :param scale:
        :param outdir:
        :param timeout:
        """
        # Will double copy of Fast5 for 2D reads as of now TODO
        reads = [(read, f5) for f5 in fast5 for read in f5.reads]

        # Sort by ascending read completion times first
        reads = sorted(
            reads, key=lambda x: x[0].end_time, reverse=False
        )

        read_end_times = [read[0].end_time for read in reads]

        # Compute difference between completion of reads
        time_delta = [0] + [
            delta/scale for delta in self._delta(read_end_times)
        ]

        scheduler = BackgroundScheduler()
        start = time.time()  # For callback
        run = datetime.now()  # For scheduler
        for i, delay in enumerate(time_delta):
            run += timedelta(seconds=delay)
            scheduler.add_job(
                self.copy_read, 'date',
                run_date=run,
                kwargs={
                    'read': reads[i][0],
                    'start': start,
                    'fast5': reads[i][1],
                    'outdir': outdir
                }
            )
        scheduler.start()
        if not timeout:
            print(f"Press Ctrl+{'Break' if os.name == 'nt' else 'C'} to exit")
            try:
                # Simulate application activity
                # which keeps the main thread alive
                while True:
                    time.sleep(2)
            except (KeyboardInterrupt, SystemExit):
                scheduler.shutdown()
        else:
            time.sleep(timeout)
            scheduler.shutdown()

    def copy_read(self, read, start, fast5, outdir):

        os.makedirs(os.path.abspath(outdir), exist_ok=True)

        shutil.copy(fast5.path, os.path.abspath(outdir))

        self._print_read(read.id, start)

    @staticmethod
    def _print_read(name, start):

        now = time.time()
        elapsed = round(float(now - start), 4)
        print(f"Read: {time.ctime(now)} elapsed={elapsed} name={name}")

        return now

    @staticmethod
    def _delta(times):

        return [times[n] - times[n - 1] for n in range(1, len(times))]

    @staticmethod
    def group_runs(fast5):

        pipeline = [
            {
                "$group": {
                    "_id": "$exp_start_time",
                    "fast5": {"$push": "$_id"}
                }
            }
        ]
        run_groups = list(
            fast5.aggregate(*pipeline, allowDiskUse=True)
        )

        runs = {}

        for run in run_groups:
            timestamp = int(
                run["_id"]
            )
            entry = {
                "run": datetime.fromtimestamp(timestamp),
                "fast5": run["fast5"]
            }
            runs[timestamp] = entry

        print(f"Extracted {len(runs)} {'run' if len(runs) == 1 else 'runs'}.")

        return runs

    def sample(
       self,
       objects,
       limit: int = 10,
       tags: str or list = None,
       proportion: str or list = None,
       unique: bool = False,
       include_tags: str or list = None,
       exclude_reads: list = None,
       return_documents: bool = True
    ):

        """ Add query to a queryset (file_objects)
        to sample a limited number of file objects;
        these can be sampled proportionally by tags.
        """

        if isinstance(tags, str):
            tags = [tags]

        if isinstance(include_tags, str):
            include_tags = [include_tags]

        if tags:
            if exclude_reads:
                query_pipeline = [
                    {"$match": {"read_id": {"$nin": exclude_reads}}}
                ]
            else:
                query_pipeline = []

            if include_tags:
                query_pipeline += [
                    {"$match": {"tags": {"$all": include_tags}}}
                ]

            # Random sample across given tags:
            if isinstance(proportion, list) and len(proportion) > 0:

                if not len(proportion) == len(tags):
                    raise ValueError(
                        "List of proportions must be the same length as list of tags."
                    )
                if not sum(proportion) == 1:
                    raise ValueError("List of proportions must sum to 1")

                self.logger.info(
                    f"Tags specified, list of proportions, sample tags"
                )

                results = []
                for i in range(len(tags)):
                    lim = int(limit * proportion[i])
                    query_pipeline += [
                        {"$match": {"tags": {"$in": [tags[i]]}}},
                        {"$sample": {"size": lim}}
                    ]
                    results += list(
                        objects.aggregate(
                            *query_pipeline, allowDiskUse=True
                        )
                    )

            # Equal size of random sample for each tag:
            else:
                if proportion == "equal":
                    self.logger.info(
                        f"Tags specified, equal proportions, sample {int(limit/len(tags))} Reads for each tag: {tags}"
                    )
                    results = []
                    for tag in tags:
                        query_pipeline += [
                            {"$match": {"tags": {"$in": [tag]}}},
                            {"$sample": {"size": int(limit/len(tags))}}
                        ]
                        results += list(
                            objects.aggregate(
                                *query_pipeline, allowDiskUse=True
                            )
                        )

                else:
                    self.logger.info(
                        f"Tags specified, but no proportions, sample {limit} Reads from all (&) tags: {' '.join(tags)}"
                    )
                    query_pipeline += [
                        {"$match": {"tags": {"$all": tags}}},
                        {"$sample": {"size": limit}}
                    ]
                    results = list(
                        objects.aggregate(*query_pipeline, allowDiskUse=True)
                    )

        else:
            self.logger.info(
                f"No tags specified, sample {limit} files over given file objects"
            )
            query_pipeline = [
                {"$sample": {"size": limit}}
            ]

            results = list(
                objects.aggregate(
                    *query_pipeline, allowDiskUse=True
                )
            )
        if unique:
            results = list(set(results))

        if return_documents:
            results = [Read(**result) for result in results]

        return results

    def objects_to_tsv(self, file_objects, out_file, labels=None, sep="\t"):

        self.logger.info(
            f"Writing file paths of Fast5 documents to {out_file}."
        )

        data = {"path": [obj.path for obj in file_objects], 'read_id': [obj.read_id for obj in file_objects]}

        if labels:
            data.update({"labels": labels})

        pandas.DataFrame(data).to_csv(
            out_file, header=None, index=None, sep=sep
        )

    def copy(
        self,
        file_objects,
        outdir,
        exist_ok: bool = True,
        symlink: bool = False,
        iterate: bool = False,
        ncpu: int = 1,
        chunk_size: int = 100,
        prefixes=None
    ):

        """ Copy or symlink into output directory

         ... use either generator (memory efficient, ncpu = 1) or
        list for memory dependent progbar (ncpu = 1) or multi-processing
        (speedup, ncpu > 1)
        """

        # If files are stored on remote server,
        # copy the files using Paramiko and SCP

        # Do this as iterator (ncpu = 1, if iterate)
        # or in memory (ncpu > 1, ncpu = 1 if not iterate, has progbar)

        if ncpu == 1:

            os.makedirs(outdir, exist_ok=exist_ok)

            if iterate:
                self.link_files(
                    file_objects,
                    outdir=outdir,
                    pbar=None,
                    symlink=symlink,
                    scp=self.scp,
                    prefixes=prefixes
                )
            else:
                file_objects = list(file_objects)
                with tqdm.tqdm(
                        total=len(file_objects)
                ) as pbar:
                    self.link_files(
                        file_objects,
                        outdir=outdir,
                        pbar=pbar,
                        symlink=symlink,
                        scp=self.scp,
                        prefixes=prefixes
                    )
        else:

            if self.scp:
                raise ValueError(
                    "You are trying to call the copy method with "
                    "multiprocessing options, while connected to "
                    "remote server via SHH. This is currently not "
                    "supported by PoreMongo."
                )

            os.makedirs(outdir, exist_ok=exist_ok)

            # Multiprocessing copy of file chunks, in memory:
            file_objects = list(file_objects)

            file_object_chunks = self._chunk_seq(file_objects, chunk_size)
            nb_chunks = len(file_object_chunks)

            if prefixes:
                prefix_chunks = self._chunk_seq(prefixes, chunk_size)
            else:
                prefix_chunks = [None for _ in range(nb_chunks)]

            self.logger.info(
                f"Linking file chunks across processors "
                f"(number of chunks = {nb_chunks}, ncpu = {ncpu})..."
            )

            # Does not work for multiprocessing

            pool = mp.Pool(processes=ncpu)
            for i in range(nb_chunks):
                pool.apply_async(
                    self.link_files,
                    args=(
                        file_object_chunks[i],
                        outdir,
                        None,
                        symlink,
                        self.scp,
                        prefix_chunks[i]
                    )
                )

            pool.close()
            pool.join()

    @staticmethod
    def _chunk_seq(seq, size):

        # Generator
        return [seq[pos:pos + size] for pos in range(0, len(seq), size)]

    @staticmethod
    def link_files(
        file_objects,
        outdir: str,
        symlink: bool = False,
        pbar=None,
        scp=None,
        prefixes=None
    ):

        for i, obj in enumerate(file_objects):

            if scp is not None:

                if prefixes:
                    prefix = prefixes[i]
                else:
                    prefix = None

                obj.get(scp, out_dir=outdir, prefix=prefix)

            else:
                # For results from aggregation (dicts)
                if isinstance(obj, dict):
                    obj_path = obj["path"]
                    obj_name = obj["name"]
                else:
                    obj_path = obj.path
                    obj_name = obj.name

                if prefixes:
                    obj_name = prefixes[i] + "_" + obj_name

                if symlink:
                    # If not copy, symlink:
                    target_link = os.path.join(outdir, obj_name)
                    os.symlink(obj_path, target_link)
                else:
                    # Copy files to target directory
                    shutil.copy(obj_path, outdir)

            if pbar:
                pbar.update(1)

    # Small helpers

    def decompose_uri(self):

        if "localhost" not in self.uri:
            user_split = self.uri.replace("mongodb://", "").split("@")
            return "mongodb://" + user_split.pop(0).split(":")[0] + \
                   "@" + "@".join(user_split)
        else:
            return self.uri

    def files_from_cache(self):

        # Cache is run summary file or index of file paths
        # Cache can be generated when doing a path search for indexing Fast5

        pass

    @staticmethod
    def files_from_list(path: Path, sep='\t') -> list:

        """ Read Fast5 paths from a summary file with one column """

        return pandas.read_csv(path, sep='\t').iloc[:, 0]

    @staticmethod
    def files_from_path(path: str, extension: str, recursive: bool):

        if not recursive:
            # Yielding files
            for file in os.listdir(path):
                if file.endswith(extension):
                    yield os.path.abspath(os.path.join(path, file))
        else:
            # Recursive search should be slightly faster in 3.6
            # is always a generator:
            for p, d, f in os.walk(path):
                for file in f:
                    if file.endswith(extension):
                        yield os.path.abspath(os.path.join(p, file))





