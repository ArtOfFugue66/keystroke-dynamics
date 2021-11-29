from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def split_list_by_percentages(list_to_split: List, *percentages) -> List[List]:
    """
    Take a list and split it into sub-lists by percentages (e.g., 80%-10%-10%).

    :param list_to_split: List that should be split into sub-lists.
    :param percentages: An arbitrary number of floats that specify the number of sublists & the size for each resulting sublist.
    """
    pass


def unravel_batches(batch_list: List[Tuple[pd.DataFrame, pd.DataFrame, np.float64]]) \
        -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[np.float64]]:
    """
    Retrieve elements of the pair tuples and place them into individual lists.

    :param batch_list: List of all the batches to be unraveled.
    :return: Tuple containing (first sequences, second sequences, target distances).
    """
    first_sequence_features = []
    second_sequence_features = []
    target_distances = []

    for batch_index, batch in tqdm(enumerate(batch_list), total=len(batch_list),
                                   desc=f"[BATCHES] Unraveling sequences from batches"):
        for seq1, seq2, target_distance in batch:
            first_sequence_features.append(seq1)
            second_sequence_features.append(seq2)
            target_distances.append(target_distance)

    return first_sequence_features, second_sequence_features, target_distances


def list_to_chunks_by_size(file_list: List, chunk_size: int) -> List:
    """
    Generator function; Internal state of function persists across calls.
    Split a list into multiple sub-lists (chunks) of the specified size.

    @param: file_list: list to be split
    @param: chunk_size: size of created sub-lists
    """
    dataset_length = len(file_list)

    for i in range(0, dataset_length, chunk_size):
        yield file_list[i: i + chunk_size]


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
