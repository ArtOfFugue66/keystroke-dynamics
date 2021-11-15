import csv
import numpy as np
import pandas as pd
from typing import List


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


def read_file_list_from_dataset(filenames: List[str]) -> List[pd.DataFrame]:
    file_name = ''
    data_frames = []
    try:
        for file_name in filenames:
            data_frames.append(
                pd.read_csv(file_name,
                            delimiter='\t',
                            encoding="ISO-8859-1",
                            engine="python",
                            quoting=csv.QUOTE_NONE,
                            dtype={'PARTICIPANT_ID': np.int32, 'TEST_SECTION_ID': np.int32, 'SENTENCE': 'string',
                                   'USER_INPUT': 'string', 'KEYSTROKE_ID': np.int32, 'PRESS_TIME': np.float64,
                                   'RELEASE_TIME': np.float64, 'LETTER': 'string', 'KEYCODE': np.int32},
                            usecols=['PARTICIPANT_ID', 'TEST_SECTION_ID', 'PRESS_TIME', 'RELEASE_TIME'])
            )
    except Exception as e:
        print(f"\n[ERROR] Skipping file {file_name}: {e}")

    return data_frames
