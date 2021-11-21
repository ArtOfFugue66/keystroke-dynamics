import unittest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from features.main import preprocess_df, pad_or_trim_df


class TestDataFrameOperations(unittest.TestCase):
    def test_preprocess_df_fcn(self):
        # Arrange
        df = pd.DataFrame(
            columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY', 'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
            data=[[10001, 1, 51.00000, 0.00000, np.nan, 0.00000],
                  [10001, 1, 192.00000, 280.00000, 331.00000, np.nan],
                  [10001, 2, np.nan, 64.00000, 128.00000, 1.00000],
                  [10001, 2, 56.00000, np.nan, 208.00000, 199.00000]]
        )
        expected = pd.DataFrame(
            columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY', 'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
            data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000],
                  [10001, 2, 0.00000, 0.06400, 0.12800, 0.00100],
                  [10001, 2, 0.05600, 0.00000, 0.20800, 0.19900]]
        )
        # Act
        actual = preprocess_df(df)
        # Assert
        assert_frame_equal(actual, expected)

    def test_pad_or_trim_df_fcn_long_df(self):
        # Arrange
        df = pd.DataFrame(
            columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY', 'INTERKEY_LATENCY', 'PRESS_LATENCY',
                     'RELEASE_LATENCY'],
            data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000],
                  [10001, 1, 0.00000, 0.06400, 0.12800, 0.00100],
                  [10001, 1, 0.05600, 0.00000, 0.20800, 0.19900],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000]]
        )  # Has a length of 10
        sequence_length = 4
        expected = pd.DataFrame(
            columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY', 'INTERKEY_LATENCY', 'PRESS_LATENCY',
                     'RELEASE_LATENCY'],
            data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000],
                  [10001, 1, 0.00000, 0.06400, 0.12800, 0.00100],
                  [10001, 1, 0.05600, 0.00000, 0.20800, 0.19900]]
        )  # Has length of 4
        # Act
        actual = pad_or_trim_df(df, sequence_length=sequence_length)
        # Assert
        assert_frame_equal(actual, expected)

    def test_pad_or_trim_df_fcn_short_df(self):
        # Arrange
        df = pd.DataFrame(
            columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY', 'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY'],
            data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000],
                  [10001, 1, 0.00000, 0.06400, 0.12800, 0.00100],
                  [10001, 1, 0.05600, 0.00000, 0.20800, 0.19900]]
        )  # Has same TEST_SECTION_ID throughout, since it represents a sequence sub-frame
        sequence_length = 10
        expected = pd.DataFrame(
            columns=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'HOLD_LATENCY', 'INTERKEY_LATENCY', 'PRESS_LATENCY','RELEASE_LATENCY'],
            data=[[10001, 1, 0.05100, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.19200, 0.28000, 0.33100, 0.00000],
                  [10001, 1, 0.00000, 0.06400, 0.12800, 0.00100],
                  [10001, 1, 0.05600, 0.00000, 0.20800, 0.19900],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000],
                  [10001, 1, 0.00000, 0.00000, 0.00000, 0.00000]]
        )
        # Act
        actual = pad_or_trim_df(df, sequence_length=sequence_length)
        # Assert
        assert_frame_equal(actual, expected, check_dtype=False)  # Don't check dtype, not relevant
