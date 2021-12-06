import numpy as np
import pandas as pd
import csv
from typing import List


def read_files_from_original_dataset(filenames: List[str]) -> List[pd.DataFrame]:
    """
    Function that uses the 'read_csv()' Pandas function to read a
    tab-separated text file into a Pandas DataFrame.

    :param filenames: List of filenames to read from the original
    (PRESS_TIME & RELEASE_TIME) dataset
    :return: List of DataFrames, one per each file in the original
    dataset
    """
    file_name = ''
    data_frames = []
    try:
        for file_name in filenames:
            data_frames.append(
                pd.read_csv(file_name,
                            delimiter='\t',
                            encoding="ISO-8859-1",
                            engine="python",
                            quoting=csv.QUOTE_NONE,
                            dtype={'PARTICIPANT_ID': np.int32, 'TEST_SECTION_ID': np.int32, 'SENTENCE': 'string',
                                   'USER_INPUT': 'string', 'KEYSTROKE_ID': np.int32, 'PRESS_TIME': np.float64,
                                   'RELEASE_TIME': np.float64, 'LETTER': 'string', 'KEYCODE': np.int32},
                            usecols=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME'])
            )
    except Exception as e:
        print(f"\n[ERROR] Skipping file {file_name}: {e}")

    return data_frames

def compute_hold_latency(press_times: pd.Series, release_times: pd.Series) -> np.ndarray:
    """
    Compute HOLD_LATENCY feature for a keystroke sequence (sentence),
    using two Pandas Series: one contains a timestamp corresponding
    to the moment when each key in the sequence was pressed, the
    other contains a timestamp corresponding to the moment when each
    key in the sequence was released.

    HOLD_LATENCY is defined as the time duration a key is held down
    (or time difference between the moment a key is pressed and the moment it is released).

    :param press_times: Timestamps of keystroke presses
    :param release_times: Timestamps of keystroke releases
    :return: Numpy array containing HOLD_LATENCY feature values.
    """
    press_times = press_times.to_numpy()
    release_times = release_times.to_numpy()
    hold_latencies = np.array(release_times - press_times)
    return hold_latencies

def compute_interkey_latency(press_times: pd.Series, release_times: pd.Series) -> np.ndarray:
    """
    The same as with the 'compute_hold_latency()' function, except
    this function computes the INTERKEY_LATENCY feature.

    INTERKEY_LATENCY is defined as the difference in time between
    the moment a key is released and the moment the next key is pressed.

    :param press_times: Timestamps of keystroke presses
    :param release_times: Timestamps of keystroke releases
    :return: Numpy array containing INTERKEY_LATENCY feature values
    """
    # We offset the two Series objects by 1 element to compute the feature
    # more easily.
    press_times = press_times[1:].to_numpy()
    release_times = release_times[:-1].to_numpy()
    # We start the Numpy array with a 0-value because we cannot compute
    # interkey latency for the first key press, since we do not have a
    # previous key release.
    interkey_latencies = np.append(np.float64(0), press_times - release_times)
    return interkey_latencies

def compute_press_latency(press_times: pd.Series) -> np.ndarray:
    """
    Compute PRESS_LATENCY feature for a given keystroke sequence (sentence).

    PRESS_LATENCY is defined as the difference in time between the moments
    of two consecutive key press events.

    :param press_times: Timestamps of keystroke presses
    :return: Numpy array containing PRESS_LATENCY feature values
    """
    press_times_1 = press_times[1:].to_numpy()
    press_times_2 = press_times[:-1].to_numpy()
    press_latencies = np.append(np.float64(0), press_times_1 - press_times_2)
    return press_latencies

def compute_release_latency(release_times: pd.Series) -> np.ndarray:
    """
    Compute RELEASE_LATENCY feature for a given keystroke sequence (sentence).

    RELEASE_LATENCY is defined as the difference in time between the moments
    of two consecutive key release events.

    :param release_times: Timestamps of keystroke releases
    :return: Numpy array containing RELEASE_LATENCY feature values
    """
    release_times_1 = release_times[1:].to_numpy()
    release_times_2 = release_times[:-1].to_numpy()
    release_latencies = np.append(np.float64(0), release_times_1 - release_times_2)
    return release_latencies

def compute_sequence_features(sequence_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract timestamps of key presses & key releases from a DataFrame
    containing a keystroke sequence's PRESS_TIMEs & RELEASE_TIMEs.

    :param sequence_df: DataFrame containing keystroke sequence data
    :return: Initial DataFrame object with new columns, one for each temporal
    feature
    """
    all_press_times = sequence_df['PRESS_TIME']      # Get values from raw data columns
    all_release_times = sequence_df['RELEASE_TIME']

    # Compute each feature column, compute its absolute value & add it to the keystroke sequence DataFrame
    sequence_df['HOLD_LATENCY'] = np.abs(compute_hold_latency(press_times=all_press_times, release_times=all_release_times))
    sequence_df['INTERKEY_LATENCY'] = np.abs(compute_interkey_latency(press_times=all_press_times, release_times=all_release_times))
    sequence_df['PRESS_LATENCY'] = np.abs(compute_press_latency(press_times=all_press_times))
    sequence_df['RELEASE_LATENCY'] = np.abs(compute_release_latency(release_times=all_release_times))

    # Return updated keystroke sequence DataFrame
    return sequence_df

def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess a DataFrame:
    - Replace rogue np.inf values with np.nan,
    - For each feature column, scale the values in the column to (0, 1) interval by converting them from milliseconds to seconds,
    - Replace np.nan values with np.float64(0).

    :param df: DataFrame object containing temporal features.
    :return: Processed DataFrame object.
    """
    # Replace infinite values with NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Make list of all feature columns in a keystroke sequence DataFrame
    cols = ['HOLD_LATENCY', 'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY']
    # For each feature column,
    for col in cols:
        # convert its values from milliseconds to seconds  (e.g., 261.0 to 0.261)
        df[col] = np.float64((df[col] / 1000) % 60)
    df = df.replace(np.nan, np.float64(0))  # Replace NaNs with 0s of the same dtype as the feature columns

    return df

def pad_df(df: pd.DataFrame, pad_length: int) -> pd.DataFrame:
    """
    Add padding to a DataFrame object representing a keystroke sequence (sentence).
    We first extract PARTICIPANT_ID & TEST_SECTION_ID to create a padding row,
    then append the row to 'df', 'pad_length' times.

    :param df: Keystroke sequence as a DataFrame object
    :param pad_length: How many rows to add to 'df'
    :return: 'df', padded up to the necessary length
    """
    # Using 'item()' to convert 1-length ndarrays (unique()) to their scalar equivalents
    participant_id = df['PARTICIPANT_ID'].unique().item()
    test_section_id = df['TEST_SECTION_ID'].unique().item()
    columns = df.columns             # Get df columns
    zero_value = np.float64(0)       # Define zero-value for padding
    padding_row = {                  # Define row with padding values
        columns[0]: participant_id,
        columns[1]: test_section_id,
        columns[2]: zero_value,
        columns[3]: zero_value,
        columns[4]: zero_value,
        columns[5]: zero_value
    }
    # Append the padding row at the end of df for the required number of times
    for i in range(pad_length):
        df = df.append(padding_row, ignore_index=True)
    # Convert non-feature columns back to original dtype
    df['PARTICIPANT_ID'] = df['PARTICIPANT_ID'].astype(np.int32)
    df['TEST_SECTION_ID'] = df['TEST_SECTION_ID'].astype(np.int32)
    # Return the padded DataFrame
    return df
