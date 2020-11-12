import os
import time
import numpy as np
import uuid

from pathlib import PosixPath
from mongoengine import *
from skimage.util import view_as_windows

from ont_fast5_api.fast5_interface import get_fast5_file

from datetime import datetime

# TODO: Paramiko Timeout


def timestamp_to_epoch(timestamp: float) -> float:
    """Auxiliary function to parse timestamp into epoch time."""

    epoch = datetime(1970, 1, 1)
    time_as_date = datetime.fromtimestamp(timestamp)
    return (time_as_date - epoch).total_seconds()


def epoch_to_timestamp(epoch_seconds: float) -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_seconds))


def get_read(file: str, read_id: str or list) -> np.array:

    with get_fast5_file(file, mode="r") as f5:
        if isinstance(read_id, str):
            for read in f5.get_reads():
                if read.read_id == read_id:
                    return read.get_raw_data()
            return None
        elif isinstance(read_id, list):
            reads = []
            for read in f5.get_reads():
                if read.read_id in read_id:
                    reads.append(
                        read.get_raw_data()
                    )
            return np.array(reads)
        else:
            raise ValueError("Argument read_id must be str or list of str")


class Read(Document):

    _id = ObjectIdField()  # state explicit for construction from dicts

    fast5 = StringField(required=True, unique=False)
    uuid = StringField(required=False, unique=True)
    tags = ListField(StringField())

    read_id = StringField()
    signal_data = ListField(IntField())

    meta = {"collection": "fast5"}

    def __str__(self):

        return f"{self.read_id}\t{self.fast5}\t{' '.join(self.tags)}\t{self.uuid}"

    def get_signal_windows(self, window_size: int, window_step: int):

        if len(self.signal_data) > 0:
            return view_as_windows(
                self.signal_data, window_shape=window_size, step=window_step
            )
        else:
            if not PosixPath(self.fast5).exists():
                raise ValueError(f'Could not get signal windows for file: {self.fast5}')

            return view_as_windows(
                get_read(file=self.fast5, read_id=self.read_id),
                window_shape=window_size, step=window_step
            )