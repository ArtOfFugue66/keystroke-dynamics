from __future__ import annotations
import csv
import itertools
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from tqdm import tqdm

from pairs import conf


def read_features_from_dataset(filenames: List[str], index_of_chunk: int) -> pd.DataFrame | List[pd.DataFrame]:
    """
    Reads files from the dataset of features into Pandas DataFrames.

    :param filenames: List of filenames to be read from the features dataset.
    :param index_of_chunk: Index of the dataset chunk being processed in
    the function call.
    :return: Either a single DataFrame or a list of DataFrames, depending on length of 'filenames' param.
    """
    import features.conf

    dataFrames = []
    try:
        for filename in tqdm(filenames, total=len(filenames),
                             desc=f"[DATA] Reading features dataset for chunk #{index_of_chunk}"):
            dataFrames.append(
                pd.read_csv(f"{features.conf.OUTPUT_DIR}/{filename}",
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

    # NOTE: Using two cases to avoid consecutive calls to this function if a list of files has to be read.
    if len(dataFrames) > 1:
        return dataFrames
    elif len(dataFrames) == 1:  # If a single file has been read
        return dataFrames[0]


def make_pairs_for_user(pair_type_flag: bool, sequences_user_1: List[pd.DataFrame],
                        sequences_user_2: List[pd.DataFrame] = None) -> List:
    """
    Make genuine pairs for a certain user using his sequences,
    and impostor pairs using his sequences & those of another user.

    :param sequences_user_1: List of sequences belonging to 1st user
    :param sequences_user_2: List of sequences belonging to 2nd user
                             (same user in the case of genuine pairs)
    :param pair_type_flag:   Switch that determines the type of pairs to be created
                             (True -> genuine pairs; False -> impostor pairs)
    :return: List of genuine/impostor pairs, as joined DataFrames
    """
    from collections import namedtuple
    import tensorflow as tf

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

        # NOTE: Converting DataFrames to Tensors results in errors when calling '.fit()'
        # seq1_tensor = tf.convert_to_tensor(seq1, tf.float64)
        # seq2_tensor = tf.convert_to_tensor(seq2, tf.float64)
        # # target_tensor = tf.convert_to_tensor(target, tf.float64)
        # # target_tensor = tf.expand_dims(target, axis=-1)

        df_pairs_list.append(KeystrokePair(seq1, seq2, target))

    return df_pairs_list


def make_pairs_from_features_dfs(dfs: Dict[List[str], List[pd.DataFrame]], index_of_chunk: int) \
        -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """
    Read files from features dataset chunk-by-chunk &
    make genuine pairs & impostor pairs for each chunk

    :param dfs: List of dataset filenames and their corresponding Pandas DataFrames,
    which will be used for pair creation
    :param index_of_chunk: Index of the dataset chunk being processed in
    the function call (for progress bar & error handling)
    :return: Tuple (genuine pairs, impostor pairs) for given dataset chunk
    """
    from common.utils import split_by_section_id

    chunk_genuine_pairs = []
    chunk_impostor_pairs = []

    try:
        for filename, df in tqdm(dfs.items(), total=len(dfs), desc=f"[PAIRS] Making pairs for chunk #{index_of_chunk}"):
            user_sequences = split_by_section_id(df)  # split it into sequences,
            chunk_genuine_pairs.extend(
                make_pairs_for_user(pair_type_flag=True,
                                    sequences_user_1=user_sequences))  # add the user's genuine pairs to this chunk's genuine pairs list.

            impostor_list = [x for x in dfs.values() if not x.equals(df)]  # Iterate through all users in the chunk except the current one;

            for impostor_df in impostor_list:  # For each "impostor"
                # (if we don't have enough impostor pairs for the current chunk),
                if len(chunk_impostor_pairs) <= len(chunk_genuine_pairs):
                    impostor_sequences = split_by_section_id(impostor_df)  # split his DataFrame into sequences,
                    impostor_pairs = make_pairs_for_user(pair_type_flag=False,
                                                         sequences_user_1=user_sequences,
                                                         sequences_user_2=impostor_sequences)  # make impostor pairs using this impostor's sequences
                    chunk_impostor_pairs.extend(impostor_pairs)  # and add them to the list.
    except Exception as e:
        print(f"\n[ERROR] Skipping {filename} file: {e}")

    return chunk_genuine_pairs, chunk_impostor_pairs
