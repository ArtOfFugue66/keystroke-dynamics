import csv
from typing import List

import numpy as np
import pandas as pd


def read_files_from_original_dataset(filenames: List[str]) -> List[pd.DataFrame]:
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