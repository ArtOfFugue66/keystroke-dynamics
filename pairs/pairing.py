import itertools
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

import pairs.conf
from pairs.utils import split_by_section_id


def make_pairs(pair_type_flag: bool, sequences_user_1: List[pd.DataFrame],
               sequences_user_2: List[pd.DataFrame] = None) -> List:
    """
    TODO: Fcn description
    :param sequences_user_1: List of sequences belonging to 1st user.
    :param sequences_user_2: List of sequences belonging to 2nd user
                             (same user in the case of genuine pairs).
    :param pair_type_flag:   Switch that determines the type of pairs to be created
                             (True -> genuine pairs; False -> impostor pairs)
    :return: List of genuine/impostor pairs, as joined DataFrames.
    """
    df_pairs_list = []

    if pair_type_flag is True:  # Genuine pairs
        df_tuples = list(itertools.combinations(sequences_user_1, 2))  # target distance is 0 for genuine pair
        target = np.float64(0)
    else:  # Impostor pairs
        df_tuples = list(itertools.product(sequences_user_1, sequences_user_2))
        target = np.float64(pairs.conf.MARGIN)  # target distance is 'alpha' for impostor pair

    for seq1, seq2 in df_tuples:
        sequence_pair = seq1.join(seq2, lsuffix='_1', rsuffix='_2')
        sequence_pair["TARGET_DISTANCE"] = target
        df_pairs_list.append(sequence_pair)

    return df_pairs_list


def read_and_make_pairs(filenames, chunk_size):
    from features.utils import list_to_chunks_by_size
    from features import utils

    filename_chunks = list_to_chunks_by_size(filenames, chunk_size)

    for chunk_index, chunk in enumerate(filename_chunks):  # For current chunk of filenames,
        chunk_genuine_pairs = []
        chunk_impostor_pairs = []

        filenames, chunk_dfs = utils.read_file_list_from_dataset(chunk)  # read DataFrame for each user,
        try:
            for filename, user_df in tqdm(zip(filenames, chunk_dfs), total=len(chunk),
                                          desc=f"Making pairs for dataset chunk {chunk_index}"):
                user_sequences = split_by_section_id(user_df)  # split it into sequences,
                chunk_genuine_pairs.extend(make_pairs(True, user_sequences))

                impostor_list = [x for x in chunk_dfs if
                                 not x.equals(user_df)]  # Iterate through all users in the chunk except the current one

                for impostor_df in impostor_list:
                    impostor_sequences = split_by_section_id(impostor_df)
                    impostor_pairs = make_pairs(False, user_sequences, impostor_sequences)
                    chunk_impostor_pairs.extend(impostor_pairs)
        except Exception as e:
            print(f"\n[ERROR] Skipping file {filename}: {e}")
            raise ValueError

        yield chunk_genuine_pairs, chunk_impostor_pairs


def make_pair_batches(genuine_pairs: List[pd.DataFrame],
                      impostor_pairs: List[pd.DataFrame]) -> List[List[pd.DataFrame]]:
    """

    :param genuine_pairs:
    :param impostor_pairs:
    :return:
    """
    from features.utils import list_to_chunks_by_size
    batches = []

    batch_size = pairs.conf.BATCH_SIZE

    # Make half-batches from the current dataset chunk
    genuine_half_batches = list_to_chunks_by_size(genuine_pairs, batch_size / 2)
    impostor_half_batches = list_to_chunks_by_size(impostor_pairs, batch_size / 2)

    for genuine_half, impostor_half in zip(genuine_half_batches, impostor_half_batches, strict=True):
        batch = genuine_half.append(impostor_half)
        batches.append(batch)  # TODO: maybe use 'extend' here

    return batches
