"""
Code adapted from Pomoxis: https://github.com/nanoporetech/pomoxis

This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
(c) 2016 Oxford Nanopore Technologies Ltd.
"""

from pathlib import Path

import time
import os
from colorama import Fore

from watchdog.observers import Observer
from watchdog.events import RegexMatchingEventHandler
from watchdog.utils import has_attribute, unicode_paths
import logging

logger = logging.getLogger(__name__)

EVENT_TYPE_MOVED = 'moved'
EVENT_TYPE_DELETED = 'deleted'
EVENT_TYPE_CREATED = 'created'
EVENT_TYPE_MODIFIED = 'modified'

RE = Fore.RESET
R = Fore.RED
G = Fore.GREEN
Y = Fore.YELLOW
C = Fore.CYAN
M = Fore.MAGENTA


def wait_for_file(fname):
    """Block until a filesize remains constant."""
    size = None
    while True:
        try:
            newsize = os.path.getsize(fname)
        except:
            newsize = None
        else:
            if newsize is not None and size == newsize:
                break

        size = newsize


class StandardRegexMatchingEventHandler(RegexMatchingEventHandler):
    def __init__(self, callback, regexes, **kwargs):
        RegexMatchingEventHandler.__init__(self, regexes=regexes)

        self.callback = callback
        self.kwargs = kwargs  # Callback arguments

    def _process_file(self, event):
        """Process an event when a file is created (or moved).
        :param event: watchdog event.
        :returns: result of applying `callback` to watched file.
        """
        if event.event_type == EVENT_TYPE_CREATED:
            fname = event.src_path
        else:
            fname = event.dest_path
        # need to wait for file to be closed
        wait_for_file(fname)
        return self.callback(
            Path(fname), **self.kwargs
        )

    def dispatch(self, event):
        """Dispatch an event after filtering. We handle
        creation and move events only.

        :param event: watchdog event.
        :returns: None
        """
        if event.event_type not in (EVENT_TYPE_CREATED, EVENT_TYPE_MOVED):
            return
        if self.ignore_directories and event.is_directory:
            return

        paths = []
        if has_attribute(event, 'dest_path'):
            paths.append(unicode_paths.decode(event.dest_path))
        if event.src_path:
            paths.append(unicode_paths.decode(event.src_path))

        if any(r.match(p) for r in self.ignore_regexes for p in paths):
            return

        if any(r.match(p) for r in self.regexes for p in paths):
            self._process_file(event)


class Watcher(object):
    def __init__(self, path, event_handler, recursive=False):
        """Wrapper around common watchdog idiom.
        :param path: path to watch for new files.
        :param event_handler: subclass of watchdog.events.FileSystemEventHandler.
        :param recursive: watch path recursively?
        """
        self.observer = Observer()
        self.observer.schedule(event_handler, path, recursive)

    def start(self):
        """Start observing path."""
        self.observer.start()

    def stop(self):
        """Stop observing path."""
        self.observer.stop()
        self.observer.join()


def watch_path(
        path,
        callback,
        recursive=False,
        regexes: str or list = r'.*\.fast5$',
        **kwargs,
):

    """ Uses function callback(fname) """

    if isinstance(regexes, str):
        regexes = [regexes]

    handler = StandardRegexMatchingEventHandler(
        callback=callback, regexes=regexes, **kwargs
    )

    watch = Watcher(path, event_handler=handler, recursive=recursive)

    print(f'{Y}Starting to watch {R}{Path(path).absolute()}{Y} '
          f'for new {R}Fast5{Y} files:{RE}\n')
    watch.start()

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        try:
            print("\r  ")
            print(f'{Y}Press {R}Ctrl + C{Y} to exit.{RE}')
            watch.stop()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\r  ")
            exit(0)

