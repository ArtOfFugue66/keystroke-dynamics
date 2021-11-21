import pandas as pd
import numpy as np
from typing import List, Dict
import warnings

from features import conf
from features.utils import read_file_list_from_dataset

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


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

def pad_or_trim_df(df: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
    df_len = len(df)
    if df_len > sequence_length:                                        # If df is too long,
        df.drop(df.tail(df_len - sequence_length).index, inplace=True)  # trim it.
    elif df_len < sequence_length:                  # If df is too short,
        pad_length = sequence_length - df_len
        df = pad_df(df, pad_length=pad_length)      # pad it.
    return df                                       # If df_len == sequence_length, return df as-is.


def process_dataset_chunk(user_dfs: List[pd.DataFrame], sequence_length: int) -> Dict[int, pd.DataFrame]:
    chunk_users_sequences = {}  # { PARTICIPANT_ID: [sequence_list] } dict

    for user_df in user_dfs:
        # Get list of test section IDs for current user
        test_section_ids = user_df['TEST_SECTION_ID'].unique()
        # Get participant ID of current user, as scalar using 'item()'
        participant_id = user_df['PARTICIPANT_ID'].unique().item()
        # Split user's frame into typing sequence sub-frames (keystroke sequences)
        user_sequences = [user_df[user_df['TEST_SECTION_ID'] == sub_frame_id] for sub_frame_id in test_section_ids]
        # Compute timing features for current keystroke sequence
        sequences_w_features = [compute_sequence_features(df) for df in user_sequences]
        # Drop raw data columns from all keystroke sequences
        sequences_w_features = [df.drop(['PRESS_TIME', 'RELEASE_TIME'], axis=1) for df in sequences_w_features]
        # Preprocess sequence (scale to [0, 1], handle NaNs etc)
        preprocessed_sequences = [preprocess_df(df) for df in sequences_w_features]
        # To consider a sequence stand-alone, reset its indices
        preprocessed_sequences = [df.reset_index(drop=True) for df in preprocessed_sequences]
        # Bring keystroke sequences to 'sequence_length', either by trimming or padding each sequence
        sized_sequences = [pad_or_trim_df(df, sequence_length) for df in preprocessed_sequences]  # Pad/trim sequence to conf.SEQUENCE_LENGTH
        # Concatenate all keystroke sequences into one DataFrame
        final_single_df = pd.concat(sized_sequences)
        # Update dict with processed keystroke sequences of all users
        chunk_users_sequences[participant_id] = final_single_df

    return chunk_users_sequences


def process_write_features(filename_chunk: List[str], sequence_length: int):
    chunk_of_dataframes = read_file_list_from_dataset(filename_chunk)
    chunk_of_dataframes_with_features = process_dataset_chunk(chunk_of_dataframes, sequence_length)

    features_output_dir = f"{conf.OUTPUT_DIR}"

    for participant_id, all_user_sequences in chunk_of_dataframes_with_features.items():
        file_to_write = open(f"{features_output_dir}/{participant_id}_features.txt", "w+")
        # noinspection PyTypeChecker
        all_user_sequences.to_csv(file_to_write, sep='\t', encoding='ISO-8859-1', line_terminator='\n', index=False)


def compute_features_dataset():
    """
    This function will run if this Python file is run directly.
    It reads files from the raw dataset into Pandas DataFrames in a chunk-by-chunk fashion,
    computes the timing features for all users & writes the resulting DataFrames to a new folder.
    :return: None
    """
    from utils import list_to_chunks_by_size
    import os
    from tqdm import tqdm
    from multiprocessing import Process
    import time

    os.chdir(conf.SMALL_DATASET_DIR)
    dataset_filenames = os.listdir(".")

    all_dataset_chunks = list(list_to_chunks_by_size(dataset_filenames, conf.CHUNK_SIZE))  # 40 chunks of size 100

    start_time = time.time()  # ----------- Capture timestamp before CPU-intensive code ----------- #

    for outer_chunk_index, outer_chunk in tqdm(enumerate(all_dataset_chunks), total=len(all_dataset_chunks), desc="[INFO] Processing dataset chunks"):  # 40 chunks of size 100
        thread_chunks = list(list_to_chunks_by_size(outer_chunk, conf.THREAD_CHUNK_SIZE))  # 10 chunks of size 10
        process_list = []
        for inner_chunk_index, inner_chunk in enumerate(thread_chunks):  # For each 10 files
            # Create a process that handles the file,
            process = Process(target=process_write_features, args=(inner_chunk, conf.SEQUENCE_LENGTH), name=f"process-{inner_chunk_index}")
            # append in to the process list
            process_list.append(process)
            # and start it
            process.start()

        for process in process_list:
            # Wait for each process in the list to finish execution
            process.join()

    end_time = time.time()    # ----------- Capture timestamp after CPU-intensive code ----------- #
    print(f"Files were processed in {end_time - start_time} seconds!")


if __name__ == '__main__':
    compute_features_dataset()
