"""
=====================
Achilles Base Module
====================

"""

from pathlib import Path


class Achilles:
    """ Achilles base management class """

    def __init__(self):

        self.path: Path = Path().home() / '.achilles'
