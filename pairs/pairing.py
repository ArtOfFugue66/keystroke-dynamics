import itertools
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import conf.pairs as conf
from utils.pairs import read_features_from_dataset
from utils.general import list_to_chunks_by_size


def make_pairs_for_user(pair_type_flag: bool, sequences_user_1: List[pd.DataFrame],
                        sequences_user_2: List[pd.DataFrame] = None) -> List:
    """
    Make genuine or impostor pairs for a certain user using
    his sequences and, optionally, those of another user.
    :param sequences_user_1: List of sequences belonging to 1st user.
    :param sequences_user_2: List of sequences belonging to 2nd user
                             (same user in the case of genuine pairs).
    :param pair_type_flag:   Switch that determines the type of pairs to be created
                             (True -> genuine pairs; False -> impostor pairs)
    :return: List of genuine/impostor pairs, as joined DataFrames.
    """
    from collections import namedtuple

    KeystrokePair = namedtuple('KeystrokePair', 'seq1 seq2 target')
    df_pairs_list = []

    if pair_type_flag is True:  # Genuine pairs
        df_tuples = list(itertools.combinations(sequences_user_1, 2))  # target distance is 0 for genuine pair
        target = np.float64(0)
    else:  # Impostor pairs
        df_tuples = list(itertools.product(sequences_user_1, sequences_user_2))
        target = np.float64(conf.MARGIN)  # target distance is 'alpha' for impostor pair

    for (seq1, seq2) in df_tuples:
        # if 'PARTICIPANT_ID' in seq1.columns and 'TEST_SECTION_ID' in seq1.columns:
        seq1 = seq1.drop(['PARTICIPANT_ID', 'TEST_SECTION_ID'], axis=1)
        # if 'PARTICIPANT_ID' in seq2.columns and 'TEST_SECTION_ID' in seq2.columns:
        seq2 = seq2.drop(['PARTICIPANT_ID', 'TEST_SECTION_ID'], axis=1)

        df_pairs_list.append(KeystrokePair(seq1, seq2, target))

    return df_pairs_list


def make_pairs_from_features_dfs(dfs: List[pd.DataFrame], index_of_chunk: int) \
        -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Reads files from features dataset chunk-by-chunk &
    makes genuine pairs & impostor pairs for each chunk.

    :param dfs: Pandas DataFrames that are to be used for pair creation.
    :param index_of_chunk: Index of the dataset chunk being processed in
    the function call.
    :return: Tuple (genuine pairs, impostor pairs)
    """
    from utils.general import split_by_section_id

    chunk_genuine_pairs = []
    chunk_impostor_pairs = []

    try:
        for user_df in tqdm(dfs, total=len(dfs), desc=f"[PAIRS] Making pairs for chunk #{index_of_chunk}"):
            user_sequences = split_by_section_id(user_df)  # split it into sequences,
            chunk_genuine_pairs.extend(
                make_pairs_for_user(True, user_sequences))  # add the user's genuine pairs to this chunk's genuine pairs list.

            impostor_list = [x for x in dfs if
                             not x.equals(user_df)]  # Iterate through all users in the chunk except the current one;

            for impostor_df in impostor_list:  # For each "impostor"
                # (if we don't have enough impostor pairs for the current chunk),
                if len(chunk_impostor_pairs) <= len(chunk_genuine_pairs):
                    impostor_sequences = split_by_section_id(impostor_df)  # split his DataFrame into sequences,
                    impostor_pairs = make_pairs_for_user(False, user_sequences,
                                                         impostor_sequences)  # make impostor pairs using this impostor's sequences
                    chunk_impostor_pairs.extend(impostor_pairs)  # and add them to the list.
    except Exception as e:
        print(f"\n[ERROR] Skipping user {user_df['PARTICIPANT_ID'][0]}'s file: {e}")
        raise ValueError  # TODO: Remove this?

    return chunk_genuine_pairs, chunk_impostor_pairs


def make_pair_batches(genuine_pairs: List[pd.DataFrame],
                      impostor_pairs: List[pd.DataFrame],
                      batch_size: int = None,
                      shuffle_batches_flag: bool = True) -> List[List[pd.DataFrame]]:
    """
    Groups a number of genuine pairs & impostor pairs into batches.
    Each batch contains half genuine pairs, half impostor pairs.

    :param shuffle_batches_flag: Specify whether to shuffle batches; Needed for unit testing.
    :param genuine_pairs: A list of Pandas DataFrames representing positive keystroke sequence
    pairs (i.e., belonging to the same user).
    :param impostor_pairs: A list of Pandas DataFrames representing negative keystroke sequence
    pairs (i.e., belonging to different users).
    :param batch_size: The size of each batch that should be created.
    :return: A list containing all batches created using 'genuine_pairs' and 'impostor_pairs' param values.
    """

    batches = []
    if not batch_size:
        batch_size = conf.BATCH_SIZE

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
