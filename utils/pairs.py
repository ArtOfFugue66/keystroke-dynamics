from __future__ import annotations

import csv
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def pairs_to_sequences_and_target(pair_dfs: List[Tuple[pd.DataFrame, pd.DataFrame, np.float64]]) -> Tuple[
    List[pd.DataFrame], List[pd.DataFrame], List[np.float64]]:
    """
    TODO: Function description
    :param pair_dfs:
    :return:
    """
    first_sequences, second_sequences, target_distances = [], [], []

    for pair in pair_dfs:
        first_sequences.append(pair[0])
        second_sequences.append(pair[1])
        target_distances.append(pair[2])

    return first_sequences, second_sequences, target_distances


def read_features_from_dataset(filenames: List[str], index_of_chunk: int) -> pd.DataFrame | List[pd.DataFrame]:
    """
    Reads files from the features dataset into Pandas DataFrames.

    :param filenames: List of filenames to be read from the features dataset.
    :param index_of_chunk: Index of the dataset chunk being processed in
    the function call.
    :return: Either a single DataFrame or a list of DataFrames, depending on length of 'filenames' param.
    """
    import conf.features

    dataFrames = []
    try:
        for filename in tqdm(filenames, total=len(filenames), desc=f"[DATA] Reading features dataset for chunk #{index_of_chunk}"):
            dataFrames.append(
                pd.read_csv(f"{conf.features.OUTPUT_DIR}/{filename}",
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
