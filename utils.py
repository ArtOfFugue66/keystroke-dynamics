from typing import List, Tuple
import pandas as pd
import numpy as np


def pair_tuples_to_elements(pair_tuples: List[Tuple[pd.DataFrame, pd.DataFrame, np.float64]]) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[np.float64]]:
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
