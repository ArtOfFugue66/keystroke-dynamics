import unittest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

# from features.main import *
from common import utils


class TestComputeIndividualFeatures(unittest.TestCase):
    def test_compute_hold_latency_fcn(self):
        # Arrange: define test objects
        press_times = pd.Series(np.array([1, 2, 3]).astype(np.float64))
        release_times = pd.Series(np.array([4, 5, 6]).astype(np.float64))
        expected = np.array([3, 3, 3]).astype(np.float64)
        # Act: define actual result & expected result
        actual = utils.compute_hold_latency(press_times, release_times)
        # Assert: Check that the actual result matches the expected one
        assert_array_equal(actual, expected)

    def test_compute_interkey_latency_fcn(self):
        # Arrange
        press_times = pd.Series(np.array([4, 5, 6]).astype(np.float64))
        release_times = pd.Series(np.array([1, 2, 3]).astype(np.float64))
        expected = np.array([0, 4, 4]).astype(np.float64)
        # Act
        actual = utils.compute_interkey_latency(press_times, release_times)
        # Assert
        assert_array_equal(actual, expected)

    def test_compute_press_latency_fcn(self):
        # Arrange
        press_times = pd.Series(np.array([4, 5, 6]).astype(np.float64))
        expected = np.array([0, 1, 1]).astype(np.float64)
        # Act
        actual = utils.compute_press_latency(press_times)
        # Assert
        assert_array_equal(actual, expected)

    def test_compute_release_latency_fcn(self):
        # Arrange
        release_times = pd.Series(np.array([2, 4, 6]).astype(np.float64))
        expected = np.array([0, 2, 2]).astype(np.float64)
        # Act
        actual = utils.compute_release_latency(release_times)
        # Assert
        assert_array_equal(actual, expected)

        # NOTE: This is pretty useless considering the existence of the other 4 tests_pairs in this file.
        #       It would have made more sense to just write this test, given the code's structure.

    def test_compute_test_section_features_fcn(self):
        # Arrange
        test_df = pd.DataFrame(
            columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME'],
            data=[[10001, 1, 1472052236111.0, 1472052236162.0],
                  [10001, 1, 1472052236442.0, 1472052236634.0],
                  [10001, 1, 1472052236570.0, 1472052236635.0],
                  [10001, 1, 1472052236778.0, 1472052236834.0]]
        )
        expected = pd.DataFrame(
            columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME',
                     'HOLD_LATENCY', 'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
            data=[[10001, 1, 1472052236111.0, 1472052236162.0, 51.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 1472052236442.0, 1472052236634.0, 192.00000, 280.00000, 331.00000, 472.00000],
                  [10001, 1, 1472052236570.0, 1472052236635.0, 65.00000, 64.00000, 128.00000, 1.00000],
                  [10001, 1, 1472052236778.0, 1472052236834.0, 56.00000, 143.00000, 208.00000, 199.00000]]
        )
        # Act
        # compute_sequence_features.__test__ = False
        actual = utils.compute_sequence_features(sequence_df=test_df)
        # Assert
        assert_frame_equal(actual, expected)
