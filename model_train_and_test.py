import os

import numpy as np
from numpy import floor
from tqdm import tqdm

import conf
import pairs.conf
import pairs.utils
from pairs.pairing import make_pairs, make_pair_batches
from features.utils import list_to_chunks_by_size
from loguru import logger



@logger.catch()
def main():
    chunk_size = conf.CHUNK_SIZE

    ### Read files from processed dataset directory (genuine user data)
    os.chdir(pairs.conf.GENUINE_DATA_PATH)
    genuine_filename = os.listdir('.')
    genuine_df = pairs.utils.read_file_list_from_dataset(genuine_filename)

    ### Read files from processed dataset directory (impostor user data)
    os.chdir(pairs.conf.IMPOSTOR_DATA_PATH)
    impostor_filenames = os.listdir('.')
    impostor_dfs = pairs.utils.read_file_list_from_dataset(impostor_filenames[:100])

    ### Split genuine & impostor data into one DataFrame / individual keystroke sequence
    genuine_sequences = pairs.utils.split_by_section_id(genuine_df)
    impostor_sequences = pairs.utils.split_by_section_id(impostor_dfs)

    ### Make genuine pairs
    genuine_pairs = make_pairs(genuine_sequences, )
    # Make impostor pairs
    impostor_pairs = make_impostor_pairs(genuine_sequences, impostor_sequences)

    ### NOTE: "The pairs were chosen randomly in each training batch...
    ### Shuffle lists of genuine & impostor pairs
    np.random.shuffle(genuine_pairs)
    np.random.shuffle(impostor_pairs)

    ### Make batches
    no_total_batches = int(floor(len(genuine_pairs) / pairs.conf.GENUINE_PAIRS_PER_BATCH))  # Each batch should have at least 1 genuine pair
    no_total_pairs = len(genuine_pairs) + len(impostor_pairs)
    batch_size = int(floor(no_total_pairs // no_total_batches))

    # Split list of all impostor pairs into a fixed number of sub-lists
    impostor_batches = list_to_chunks_by_size(impostor_pairs, batch_size)
    # genuine_pairs ...

    batches = []
    for i in range(0, no_total_batches):
        # Make batch containing the genuine pair and all corresponding impostor pairs
        batch = impostor_batches[i]
        batch.append(genuine_pairs[i])
        batches.append(batch)




    # TODO: Perform train_test_validation split on the list of batches
    pass


if __name__ == "__main__":
    main()
