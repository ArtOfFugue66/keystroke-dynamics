import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from pairs.pairing import make_pairs, make_pair_batches
import pairs.conf


class TestReadDataset(unittest.TestCase):
    def test_make_pairs_fcn_genuine_pairs(self):
        """
        Test make_pairs() functionality for GENUINE pairs.
        NOTE: Expected list is 3 x 2 / 2 = 3 DataFrames long
        """
        # Arrange
        test_genuine_sequences = [
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 2, 0.05900, 0.00000, 0.00000, 0.00000],
                      [10001, 2, 0.19500, 0.26000, 0.31300, 0.00000]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 3, 0.05900, 0.00000, 0.00000, 0.00000],
                      [10001, 3, 0.19500, 0.26000, 0.31300, 0.00000]]
            )
        ]
        expected = [
            pd.DataFrame(
                columns=['PARTICIPANT_ID_1', 'TEST_SECTION_ID_1', 'HOLD_LATENCY_1',
                         'INTERKEY_LATENCY_1', 'PRESS_LATENCY_1', 'RELEASE_LATENCY_1',
                         'PARTICIPANT_ID_2', 'TEST_SECTION_ID_2', 'HOLD_LATENCY_2',
                         'INTERKEY_LATENCY_2', 'PRESS_LATENCY_2', 'RELEASE_LATENCY_2',
                                                                  'TARGET_DISTANCE'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000, 10001, 2, 0.05900, 0.00000, 0.00000, 0.00000, 0.0],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000, 10001, 2, 0.19500, 0.26000, 0.31300, 0.00000, 0.0]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID_1', 'TEST_SECTION_ID_1', 'HOLD_LATENCY_1',
                         'INTERKEY_LATENCY_1', 'PRESS_LATENCY_1', 'RELEASE_LATENCY_1',
                         'PARTICIPANT_ID_2', 'TEST_SECTION_ID_2', 'HOLD_LATENCY_2',
                         'INTERKEY_LATENCY_2', 'PRESS_LATENCY_2', 'RELEASE_LATENCY_2',
                                                                  'TARGET_DISTANCE'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000, 10001, 3, 0.05900, 0.00000, 0.00000, 0.00000, 0.0],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000, 10001, 3, 0.19500, 0.26000, 0.31300, 0.00000, 0.0]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID_1', 'TEST_SECTION_ID_1', 'HOLD_LATENCY_1',
                         'INTERKEY_LATENCY_1', 'PRESS_LATENCY_1', 'RELEASE_LATENCY_1',
                         'PARTICIPANT_ID_2', 'TEST_SECTION_ID_2', 'HOLD_LATENCY_2',
                         'INTERKEY_LATENCY_2', 'PRESS_LATENCY_2', 'RELEASE_LATENCY_2',
                                                                  'TARGET_DISTANCE'],
                data=[[10001, 2, 0.05900, 0.00000, 0.00000, 0.00000, 10001, 3, 0.05900, 0.00000, 0.00000, 0.00000, 0.0],
                      [10001, 2, 0.19500, 0.26000, 0.31300, 0.00000, 10001, 3, 0.19500, 0.26000, 0.31300, 0.00000, 0.0]]
            )
        ]
        # Act
        actual = make_pairs(pair_type_flag=True,
                            sequences_user_1=test_genuine_sequences,
                            sequences_user_2=None)
        # Assert
        self.assertEqual(len(actual), len(expected))
        for actual_pair, expected_pair in zip(actual, expected):
            assert_frame_equal(actual_pair, expected_pair)

    def test_make_impostor_pairs_2_sequences_each(self):
        """
        Test make_pairs() functionality for IMPOSTOR pairs.
        NOTE: Expected list of pairs is 2 x (2 users - 1) x 2 = 4 DataFrames long
        """
        # Arrange
        margin = np.float64(pairs.conf.MARGIN)
        test_sequences_1 = [
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 2, 0.05900, 0.00000, 0.00000, 0.00000],
                      [10001, 2, 0.19500, 0.26000, 0.31300, 0.00000]]
            )
        ]
        test_sequences_2 = [
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10003, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                      [10003, 1, 0.19200, 0.28000, 0.33100, 0.00000]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10003, 2, 0.05900, 0.00000, 0.00000, 0.00000],
                      [10003, 2, 0.19500, 0.26000, 0.31300, 0.00000]]
            )
        ]
        # 2 genuine sequences & 2 impostor sequences => 4 total impostor pairs
        expected = [
            pd.DataFrame(
                columns=['PARTICIPANT_ID_1', 'TEST_SECTION_ID_1', 'HOLD_LATENCY_1',
                         'INTERKEY_LATENCY_1', 'PRESS_LATENCY_1', 'RELEASE_LATENCY_1',
                         'PARTICIPANT_ID_2', 'TEST_SECTION_ID_2', 'HOLD_LATENCY_2',
                         'INTERKEY_LATENCY_2', 'PRESS_LATENCY_2', 'RELEASE_LATENCY_2',
                                                                  'TARGET_DISTANCE'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000, 10003, 1, 0.05100, 0.00000, 0.00000, 0.00000, margin],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000, 10003, 1, 0.19200, 0.28000, 0.33100, 0.00000, margin]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID_1', 'TEST_SECTION_ID_1', 'HOLD_LATENCY_1',
                         'INTERKEY_LATENCY_1', 'PRESS_LATENCY_1', 'RELEASE_LATENCY_1',
                         'PARTICIPANT_ID_2', 'TEST_SECTION_ID_2', 'HOLD_LATENCY_2',
                         'INTERKEY_LATENCY_2', 'PRESS_LATENCY_2', 'RELEASE_LATENCY_2',
                                                                  'TARGET_DISTANCE'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000, 10003, 2, 0.05900, 0.00000, 0.00000, 0.00000, margin],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000, 10003, 2, 0.19500, 0.26000, 0.31300, 0.00000, margin]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID_1', 'TEST_SECTION_ID_1', 'HOLD_LATENCY_1',
                         'INTERKEY_LATENCY_1', 'PRESS_LATENCY_1', 'RELEASE_LATENCY_1',
                         'PARTICIPANT_ID_2', 'TEST_SECTION_ID_2', 'HOLD_LATENCY_2',
                         'INTERKEY_LATENCY_2', 'PRESS_LATENCY_2', 'RELEASE_LATENCY_2',
                                                                  'TARGET_DISTANCE'],
                data=[[10001, 2, 0.05900, 0.00000, 0.00000, 0.00000, 10003, 1, 0.05100, 0.00000, 0.00000, 0.00000, margin],
                      [10001, 2, 0.19500, 0.26000, 0.31300, 0.00000, 10003, 1, 0.19200, 0.28000, 0.33100, 0.00000, margin]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID_1', 'TEST_SECTION_ID_1', 'HOLD_LATENCY_1',
                         'INTERKEY_LATENCY_1', 'PRESS_LATENCY_1', 'RELEASE_LATENCY_1',
                         'PARTICIPANT_ID_2', 'TEST_SECTION_ID_2', 'HOLD_LATENCY_2',
                         'INTERKEY_LATENCY_2', 'PRESS_LATENCY_2', 'RELEASE_LATENCY_2',
                                                                  'TARGET_DISTANCE'],
                data=[[10001, 2, 0.05900, 0.00000, 0.00000, 0.00000, 10003, 2, 0.05900, 0.00000, 0.00000, 0.00000, margin],
                      [10001, 2, 0.19500, 0.26000, 0.31300, 0.00000, 10003, 2, 0.19500, 0.26000, 0.31300, 0.00000, margin]]
            )
        ]
        # Act
        actual = make_pairs(pair_type_flag=False,
                            sequences_user_1=test_sequences_1,
                            sequences_user_2=test_sequences_2)
        # Assert

        self.assertEqual(len(actual), len(expected))
        for actual_pair, expected_pair in zip(actual, expected):
            assert_frame_equal(actual_pair, expected_pair)
