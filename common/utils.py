from __future__ import annotations

from typing import List, Tuple, NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from pairs import conf


def test_train_val_split(list_to_split: List, percentages: Tuple) -> Tuple[List, List, List]:
    """
    Take a list and split it into sub-lists by percentages (e.g., 80%-10%-10%).

    @param list_to_split: List that should be split into sub-lists.
    @param percentages: An arbitrary number of floats that specify the number of sublists & the size for each resulting sublist.
    """
    train_pct, test_pct, val_pct = percentages
    no_elements = len(list_to_split)

    return list_to_split[:int(no_elements * train_pct)], \
           list_to_split[int(no_elements * train_pct):int(no_elements * test_pct)], \
           list_to_split[int(no_elements * test_pct):]

def unravel_batches(batch_list: List[Tuple[pd.DataFrame, pd.DataFrame, np.float64]]) \
        -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[np.float64]]:
    """
    Retrieve elements of pair tuples in a batch and place them into individual lists.

    @param batch_list: List of all the batches to be unraveled.
    @return: Tuple containing (first sequences, second sequences, target distances).
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
        chunk = file_list[i: i + chunk_size]
        if len(chunk) == chunk_size:  # This check is meant to ensure that the LAST chunk is exactly the necessary size and not smaller
            yield chunk

def split_by_section_id(dfs: List[pd.DataFrame] | pd.DataFrame) -> List[pd.DataFrame]:
    """
    If df is a single DataFrame, split it into sub-frames by test section ID, reset index for each & return.
    If df is a list of DataFrame objects, do all these operations but return a single list containing all sub-frames.

    @param dfs: DataFrame containing all keystroke data for a single user
    @return: List of test section sub-frames as standalone DataFrames
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


def make_pair_batches(genuine_pairs: List[pd.DataFrame],
                      impostor_pairs: List[pd.DataFrame],
                      batch_size: int = conf.BATCH_SIZE,
                      shuffle_batches_flag: bool = True) -> List[List[pd.DataFrame]]:
    """
    Groups a number of genuine pairs & impostor pairs into batches
    Each batch contains half genuine pairs, half impostor pairs

    :param shuffle_batches_flag: Specify whether to shuffle batches; Needed for unit testing
    :param genuine_pairs: A list of Pandas DataFrames representing positive keystroke sequence
    pairs (i.e., belonging to the same user)
    :param impostor_pairs: A list of Pandas DataFrames representing negative keystroke sequence
    pairs (i.e., belonging to different users)
    :param batch_size: The size of each batch that should be created
    :return: A list containing all batches created using 'genuine_pairs' and 'impostor_pairs' param values
    """

    batches = []

    # Split genuine pairs & impostor pairs lists into halves
    genuine_half_batches = list_to_chunks_by_size(genuine_pairs, batch_size // 2)
    impostor_half_batches = list_to_chunks_by_size(impostor_pairs, batch_size // 2)

    # Iterate through both lists of batch halves at once
    for genuine_half, impostor_half in tqdm(zip(genuine_half_batches, impostor_half_batches),
                                            total=len(genuine_pairs),
                                            desc="[BATCHES] Making batches from chunk pairs"):
        # Concatenate the two halves -> 'genuine_half' becomes a batch
        genuine_half.extend(impostor_half)

        # Randomly shuffle the batch before appending it to the list
        # NOTE: This is to avoid batches always starting with genuine pairs
        if shuffle_batches_flag:
            np.random.shuffle(genuine_half)

        # Add the batch to the list
        batches.append(genuine_half)

    return batches
