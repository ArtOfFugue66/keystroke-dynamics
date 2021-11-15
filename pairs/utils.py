from __future__ import annotations

import pandas as pd
import numpy as np
from typing import List
import csv
from tqdm import tqdm


def read_file_list_from_dataset(filenames: List[str]) -> pd.DataFrame | List[pd.DataFrame]:
    """
    NOTE: Using two cases to avoid consecutive calls to this function if a list of files has to be read
    :param filenames:
    :return:
    """
    dataFrames = []
    try:
        for filename in tqdm(filenames, total=len(filenames), desc="Reading files from dataset"):
            dataFrames.append(
                pd.read_csv(filename,
                            delimiter='\t',
                            encoding="ISO-8859-1",
                            engine="python",
                            quoting=csv.QUOTE_NONE,
                            dtype={'PARTICIPANT_ID': np.int32, 'TEST_SECTION_ID': np.int32,
                                   'HOLD_LATENCY': np.float64, 'INTERKEY_LATENCY': np.float64,
                                   'PRESS_LATENCY': np.float64, 'RELEASE_LATENCY': np.float64})
            )
    except Exception as e:
        print(f"\n[ERROR] Skipping file {filename}: {e}")

    if len(dataFrames) > 1:
        return filenames, dataFrames
    elif len(dataFrames) == 1:  # If a single file has been read
        return filenames[0], dataFrames[0]


def split_by_section_id(dfs: List[pd.DataFrame] | pd.DataFrame) -> List[pd.DataFrame]:
    """
    If df is a single DataFrame, split it into sub-frames by test section ID, reset index for each & return.
    If df is a list of DataFrame objects, do all these operations but return a single list containing all sub-frames.
    :param dfs: DataFrame containing all keystroke data for a single user
    :return: List of test section sub-frames as standalone DataFrames
    """
    if type(dfs) is pd.DataFrame:
        # Get list of test section IDs for current user
        test_section_ids = dfs['TEST_SECTION_ID'].unique()
        # Split user's frame into typing sequence sub-frames (keystroke sequences)
        user_sequences = [dfs[dfs['TEST_SECTION_ID'] == sub_frame_id] for sub_frame_id in test_section_ids]
        # To consider a sequence stand-alone, reset its indices
        user_sequences = [df.reset_index(drop=True) for df in user_sequences]
        return user_sequences
    elif type(dfs) is list:
        all_users_sequences = []
        for df in tqdm(dfs, total=len(dfs), desc="Splitting dfs into sub-frames"):
            test_section_ids = df['TEST_SECTION_ID'].unique()
            user_sequences = [df[df['TEST_SECTION_ID'] == sub_frame_id] for sub_frame_id in test_section_ids]
            user_sequences = [df.reset_index(drop=True) for df in user_sequences]
            all_users_sequences.extend(user_sequences)
        return all_users_sequences
