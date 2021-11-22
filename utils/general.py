from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


def list_to_chunks_by_size(file_list: List, chunk_size: int) -> List:
    """
    Generator function; Internal state of function persists across calls.
    Split 'file_list' list into multiple sub-lists of a given size (chunks).
    @param: file_list: list to be split
    @param: chunk_size: size of created sub-lists
    """
    dataset_length = len(file_list)

    for i in range(0, dataset_length, chunk_size):
        yield file_list[i: i + chunk_size]


def list_to_chunks_by_count(file_list: List, no_chunks: int) -> List[List]:
    """
    Generator function; Internal state of function persists across calls.
    Split 'file_list' list into 'no_chunks' chunks of same size.
    :param file_list: list to be split
    :param no_chunks: number of sub-lists to split into
    :return:
    """
    for chunk in np.array_split(file_list, no_chunks):
        yield list(chunk)  # Converting to list for ease of use


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