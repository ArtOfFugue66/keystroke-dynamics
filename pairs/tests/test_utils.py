import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from utils.general import split_by_section_id


class TestReadDataset(unittest.TestCase):
    def test_split_by_section_id_fcn_single_df(self):
        # Arrange
        test_df = pd.DataFrame(
            columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                     'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
            data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000],
                  [10001, 2, 0.00000, 0.06400, 0.12800, 0.00100],
                  [10001, 2, 0.05600, 0.00000, 0.20800, 0.19900]]
        )
        expected = [
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 2, 0.00000, 0.06400, 0.12800, 0.00100],
                      [10001, 2, 0.05600, 0.00000, 0.20800, 0.19900]]
            )
        ]
        # Act
        actual = split_by_section_id(dfs=test_df)
        # Assert
        self.assertEqual(len(actual), len(expected))  # Check that the fcn returns the correct # of DataFrames
        for i in range(len(actual)):
            assert_frame_equal(actual[i], expected[i])  # Check that each DataFrame in the list has the correct data

    def test_split_by_section_id_fcn_df_list(self):  # TODO
        # Arrange
        test_dfs = [
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000],
                      [10001, 2, 0.00000, 0.06400, 0.12800, 0.00100],
                      [10001, 2, 0.05600, 0.00000, 0.20800, 0.19900]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000],
                      [10001, 2, 0.00000, 0.06400, 0.12800, 0.00100],
                      [10001, 2, 0.05600, 0.00000, 0.20800, 0.19900]]
            )
        ]
        expected = [
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 2, 0.00000, 0.06400, 0.12800, 0.00100],
                      [10001, 2, 0.05600, 0.00000, 0.20800, 0.19900]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                      [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000]]
            ),
            pd.DataFrame(
                columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY',
                         'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
                data=[[10001, 2, 0.00000, 0.06400, 0.12800, 0.00100],
                      [10001, 2, 0.05600, 0.00000, 0.20800, 0.19900]]
            )
        ]
        # Act
        actual = split_by_section_id(dfs=test_dfs)
        # Assert
        self.assertEqual(len(actual), len(expected))  # Check that the fcn returns the correct # of DataFrames
        for i in range(len(actual)):
            assert_frame_equal(actual[i], expected[i])  # Check that each DataFrame in the list has the correct data
