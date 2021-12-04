import numpy as np
import pandas as pd
import csv
from typing import List


def read_files_from_original_dataset(filenames: List[str]) -> List[pd.DataFrame]:
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
    press_times = press_times.to_numpy()
    release_times = release_times.to_numpy()
    hold_latencies = np.array(release_times - press_times)
    return hold_latencies

def compute_interkey_latency(press_times: pd.Series, release_times: pd.Series) -> np.ndarray:
    press_times = press_times[1:].to_numpy()
    release_times = release_times[:-1].to_numpy()
    interkey_latencies = np.append(np.float64(0), press_times - release_times)
    return interkey_latencies

def compute_press_latency(press_times: pd.Series) -> np.ndarray:
    press_times_1 = press_times[1:].to_numpy()
    press_times_2 = press_times[:-1].to_numpy()
    press_latencies = np.append(np.float64(0), press_times_1 - press_times_2)
    return press_latencies

def compute_release_latency(release_times: pd.Series) -> np.ndarray:
    release_times_1 = release_times[1:].to_numpy()
    release_times_2 = release_times[:-1].to_numpy()
    release_latencies = np.append(np.float64(0), release_times_1 - release_times_2)
    return release_latencies

def compute_sequence_features(sequence_df: pd.DataFrame) -> pd.DataFrame:
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
    # Replace infinite values with NaNs
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # Make list of all feature columns in a keystroke sequence DataFrame
    cols = ['HOLD_LATENCY', 'INTERKEY_LATENCY', 'PRESS_LATENCY', 'RELEASE_LATENCY']
    # For each feature column,
    for col in cols:
        # convert its values from milliseconds to seconds  (e.g., 261.0 to 0.261)
        df[col] = np.float64((df[col] / 1000) % 60)
    # Update
    df = df.replace(np.nan, np.float64(0))  # Replace NaNs with 0s of the same dtype as the feature columns

    return df

def pad_df(df: pd.DataFrame, pad_length: int) -> pd.DataFrame:
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
    # Append the padding row at the end of df for the required # of times
    for i in range(pad_length):
        df = df.append(padding_row, ignore_index=True)
    # Convert non-feature columns back to original dtype
    df['PARTICIPANT_ID'] = df['PARTICIPANT_ID'].astype(np.int32)
    df['TEST_SECTION_ID'] = df['TEST_SECTION_ID'].astype(np.int32)
    # Return the padded DataFrame
    return df
