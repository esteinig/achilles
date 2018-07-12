""" Test functions for utils.py

TODO:
    - Test for making output dir, linking and copying files
    - Test for minimum signal length in Fast5

"""


import os
import shutil
import tempfile
import achilles
import logging

from unittest import TestCase

import achilles.select as achilles_select_module


class AchillesTest:
    """ Functions and attributes available to individual TestCase classes specific to Achilles """

    def __init__(self):

        self.test_data_dir, self.test_data_sets, self.test_dir1, self.test_dir2 = self.get_test_data_dirs()

        self.read_dir1 = os.listdir(self.test_dir1)
        self.read_dir2 = os.listdir(self.test_dir2)

    @staticmethod
    def get_test_data_dirs(this: str = None):

        module_basedir = os.path.dirname(os.path.dirname(achilles.__file__))
        test_data_basedir = os.path.join(module_basedir, "tests", "test_data")

        if this:
            return os.path.join(module_basedir, "tests", this)
        else:
            return test_data_basedir, os.path.join(test_data_basedir, "data_sets"),\
                   os.path.join(test_data_basedir, "dir1"), os.path.join(test_data_basedir, "dir2")


class SelectTestCases(TestCase, AchillesTest):

    """ Test functions for task select in select.py """

    def __init__(self, *args, **kwargs):

        TestCase.__init__(self, *args, **kwargs)
        AchillesTest.__init__(self)

    def setUp(self):
        logging.disable(logging.CRITICAL)

        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        logging.disable(logging.NOTSET)

        shutil.rmtree(self.tmp)

    def test_default_select(self, nb_test_files=10):
        """Test selection of Fast5 from test_data_dir (dir1, dir2) with default parameters"""

        files_dir1 = achilles_select_module.select_fast5(input_dir=self.test_dir1, output_dir=None, limit=None,
                                                         min_signal=None, symlink=False, shuffle=True,
                                                         include=None, exclude=None)

        files_dir2 = achilles_select_module.select_fast5(input_dir=self.test_dir2, output_dir=None, limit=None,
                                                         min_signal=None, symlink=False, shuffle=True,
                                                         include=None, exclude=None)
        # Are there nb_test_files Fast5?
        self.assertEqual(len(files_dir1), nb_test_files)
        self.assertEqual(len(files_dir2), nb_test_files)

        # Do they contain the correct read_identifiers:
        file_names_dir1 = [os.path.basename(file) for file in files_dir1]
        file_names_dir2 = [os.path.basename(file) for file in files_dir2]

        # All contain correct read identifiers = True
        self.assertTrue(all([True if read in file_names_dir1 else False
                             for read in self.read_dir1]))

        self.assertTrue(all([True if read in file_names_dir2 else False
                             for read in self.read_dir2]))

    def test_default_select_limit(self, limit=3):

        files_dir1 = achilles_select_module.select_fast5(input_dir=self.test_dir1, output_dir=None, limit=limit,
                                                         min_signal=None, symlink=False, shuffle=True,
                                                         include=None, exclude=None)

        self.assertEqual(len(files_dir1), limit)

    def test_default_select_shuffle(self, shuffle=False):
        """Test shuffle off with default parameters on test_dir1"""

        files_dir1 = achilles_select_module.select_fast5(input_dir=self.test_dir1, output_dir=None, limit=None,
                                                         min_signal=None, symlink=False, shuffle=shuffle,
                                                         include=None, exclude=None)

        files_dir1 = [os.path.basename(file) for file in files_dir1]

        self.assertEqual(files_dir1, self.read_dir1)

    def test_default_select_include(self, n=6):
        """Test include with default parameters on test_dir1"""

        include = self.read_dir1[:n]

        files_dir1_include = achilles_select_module.select_fast5(input_dir=self.test_dir1, output_dir=None, limit=None,
                                                                 min_signal=None, symlink=False, shuffle=True,
                                                                 include=include, exclude=None)

        # Included only one Fast5 file:
        self.assertEqual(len(files_dir1_include), n)
        # This file has the correct read_id from var exclude:

    def test_default_select_exclude(self, n=6):
        """Test exclude with default parameters on test_dir1"""

        exclude = self.read_dir1[:n]
        files_dir1_exclude = achilles_select_module.select_fast5(input_dir=self.test_dir1, output_dir=None, limit=None,
                                                                 min_signal=None, symlink=False, shuffle=True,
                                                                 include=None, exclude=exclude)
        # Excluded 5 Fast5 file:
        self.assertEqual(len(files_dir1_exclude), len(self.read_dir1)-n)
        # This file has the correct read_id from var exclude:

    def test_default_select_include_dir(self):
        """Test exclude with default parameters on all reads from test_dir1"""

        include = self.test_dir1
        files_dir1_include = achilles_select_module.select_fast5(input_dir=self.test_dir1, output_dir=None, limit=None,
                                                                 min_signal=None, symlink=False, shuffle=True,
                                                                 include=include, exclude=None)
        # Excluded 5 Fast5 file:
        self.assertEqual(len(files_dir1_include), 10)
        # This file has the correct read_id from var exclude:

    def test_default_select_exclude_dir(self):
        """Test exclude with default parameters on all reads from test_dir1"""

        exclude = self.test_dir1
        files_dir1_exclude = achilles_select_module.select_fast5(input_dir=self.test_dir1, output_dir=None, limit=None,
                                                                 min_signal=None, symlink=False, shuffle=True,
                                                                 include=None, exclude=exclude)
        # Excluded 5 Fast5 file:
        self.assertEqual(len(files_dir1_exclude), 0)
        # This file has the correct read_id from var exclude:

    def test_default_select_exclude_dataset(self):

        """Test excluding files in training dataset (.h5), generated with:

        ds = Dataset(data_file="test_select.h5")

        ds.write_data(test_dir1, test_dir2, classes=2, max_windows_per_class=6,
                      window_size=200, window_step=20, normalize=False, max_windows_per_read=2,
                      window_random=True, window_recover=True, scale=False)

        --> max_windows_per_class = 6, max_windows_per_read = 2
        --> 6/2 = 3 files per class excluded, 7 retained

        """

        exclude = [os.path.join(self.test_data_sets, "test_select1.h5"),
                   os.path.join(self.test_data_sets, "test_select2.h5")]  # test_select2 is copy of test_select1

        files_dir1_exclude = achilles_select_module.select_fast5(input_dir=self.test_dir1, output_dir=None, limit=None,
                                                                 min_signal=None, symlink=False, shuffle=True,
                                                                 include=None, exclude=exclude)
        self.assertEqual(len(files_dir1_exclude), 7)

        files_dir2_exclude = achilles_select_module.select_fast5(input_dir=self.test_dir2, output_dir=None, limit=None,
                                                                 min_signal=None, symlink=False, shuffle=True,
                                                                 include=None, exclude=exclude)
        self.assertEqual(len(files_dir2_exclude), 7)

    def test_default_select_exclude_dataset(self):
        """Test excluding files in training dataset (.h5), generated with:

        ds = Dataset(data_file="test_select.h5")

        ds.write_data(test_dir1, test_dir2, classes=2, max_windows_per_class=6,
                      window_size=200, window_step=20, normalize=False, max_windows_per_read=2,
                      window_random=True, window_recover=True, scale=False)

        --> max_windows_per_class = 6, max_windows_per_read = 2
        --> 6/2 = 3 files per class excluded, 7 retained

        """

        include = [os.path.join(self.test_data_sets, "test_select1.h5"),
                   os.path.join(self.test_data_sets, "test_select2.h5")]  # test_select2 is copy of test_select1

        files_dir1_include = achilles_select_module.select_fast5(input_dir=self.test_dir1, output_dir=None, limit=None,
                                                                 min_signal=None, symlink=False, shuffle=True,
                                                                 include=include, exclude=None)

        self.assertEqual(len(files_dir1_include), 3)  # Includes all three files

        files_dir2_include = achilles_select_module.select_fast5(input_dir=self.test_dir2, output_dir=None, limit=None,
                                                                 min_signal=None, symlink=False, shuffle=True,
                                                                 include=include, exclude=None)
        
        self.assertEqual(len(files_dir2_include), 3)  # Includes all three files


# def make_test_dataset():
#
#     ds = Dataset(data_file="test_select.h5")
#
#     ds.write_data("../test_data/dir1", "../test_data/dir2", classes=2, max_windows_per_class=6,
#                   window_size=200, window_step=20, normalize=False, max_windows_per_read=2,
#                   window_random=True, window_recover=True, scale=False)
