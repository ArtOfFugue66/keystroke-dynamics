import pandas as pd
from typing import List, Dict
import warnings

import features.conf
from features.utils import pad_df, compute_sequence_features, preprocess_df, read_files_from_original_dataset
from common.utils import list_to_chunks_by_size
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')


def pad_or_trim_df(df: pd.DataFrame, sequence_length: int) -> pd.DataFrame:
    """
    Function that pads/trims a DataFrame to a fixed length.
    - for trimming, the DataFrame object's 'drop()' is used,
    - for padding, the 'features.utils.pad_df()' utility function is called.

    :param df: DataFrame object to pad/trim to 'sequence_length' size
    :param sequence_length: Fixed length to pad/trim 'df' to
    :return: The padded/trimmed or unaffected DataFrame.
    """
    df_len = len(df)
    if df_len > sequence_length:                                        # If df is too long,
        df.drop(df.tail(df_len - sequence_length).index, inplace=True)  # trim it.
    elif df_len < sequence_length:                  # If df is too short,
        pad_length = sequence_length - df_len
        df = pad_df(df, pad_length=pad_length)      # pad it.
    return df                                       # If df_len == sequence_length, return df as-is.


def process_dataset_chunk(user_dfs: List[pd.DataFrame], sequence_length: int) -> Dict[int, pd.DataFrame]:
    """
    Function that processes the keystroke timing data of a number of users. By 'processing' we mean:
    - Splitting a user's DataFrame into multiple DataFrames, one for each sentence (TEST_SECTION_ID),
    - Computing the 4 features (HOLD_LATENCY, INTERKEY_LATENCY etc.) for each sequence DataFrame,
    - Dropping unnecessary columns (PRESS_TIME, RELEASE_TIME) from the resulting DataFrames,
    - Scaling all values between 0 and 1 and replacing NaN values with 0.0,
    - Padding/trimming all sequence DataFrames to 'sequence_length' length,
    - Concatenating all sequence DataFrames into a final DataFrame associated with the same user,
    - Adding this final DataFrame to a dictionary.

    :param user_dfs: List of DataFrames from the original dataset, belonging to multiple users
    :param sequence_length: Fixed length of a keystroke sequence (sentence) to pad/trim to
    :return: Dictionary of form {user's participant id: DataFrame with all processed sequences for the user}
    """
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
    """
    Function run by each Process created in 'compute_features_dataset()'.
    It reads a subset of the files from the original dataset, processes
    each file (computes features, normalizes data, pads/trims sequences),
    and writes the resulting DataFrames to a file named with the
    '_features.txt' suffix.

    :param filename_chunk: Subset of filenames from the whole dataset.
    :param sequence_length: Nr. of keystrokes in a sequences (used for padding/trimming sequences)
    :return: None
    """
    chunk_of_dataframes = read_files_from_original_dataset(filename_chunk)
    chunk_of_dataframes_with_features = process_dataset_chunk(chunk_of_dataframes, sequence_length)

    features_output_dir = f"{features.conf.OUTPUT_DIR}"

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

    import os
    from tqdm import tqdm
    from multiprocessing import Process
    import time

    os.chdir(features.conf.SMALL_DATASET_DIR)
    dataset_filenames = os.listdir(".")

    all_dataset_chunks = list(list_to_chunks_by_size(dataset_filenames, features.conf.CHUNK_SIZE))  # 40 chunks of size 100

    start_time = time.time()  # ----------- Capture timestamp before CPU-intensive code ----------- #

    for outer_chunk_index, outer_chunk in tqdm(enumerate(all_dataset_chunks), total=len(all_dataset_chunks), desc="[INFO] Processing dataset chunks"):  # 40 chunks of size 100
        process_chunks = list(list_to_chunks_by_size(outer_chunk, features.conf.PROCESS_CHUNK_SIZE))  # 10 chunks of size 10
        process_list = []
        for inner_chunk_index, inner_chunk in enumerate(process_chunks):  # For each 10 files
            # Create a process that handles the files,
            process = Process(target=process_write_features, args=(inner_chunk, features.conf.SEQUENCE_LENGTH), name=f"process-{inner_chunk_index}")
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
