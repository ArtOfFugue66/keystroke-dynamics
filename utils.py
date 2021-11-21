from typing import List, Tuple
import pandas as pd
import numpy as np


def pairs_to_sequences_and_target(pair_tuples: List[Tuple[pd.DataFrame, pd.DataFrame, np.float64]]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[np.float64]]:
    """
    TODO: Function description
    :param pair_tuples:
    :return:
    """
    first_sequences, second_sequences, target_distances = [], [], []

    for pair in pair_tuples:
        first_sequences.append(pair[0])
        second_sequences.append(pair[1])
        target_distances.append(pair[2])

    return first_sequences, second_sequences, target_distances


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
