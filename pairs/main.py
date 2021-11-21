from __future__ import annotations

import os

import conf
import pairs
from pairs.pairing import read_and_make_pairs, make_pair_batches


def compute_pairs_dataset():
    """
    This function will run if this Python file is run directly.
    It reads files from the features dataset into Pandas DataFrames in a chunk-by-chunk fashion,

    :return: None
    """
    # Read files from processed dataset directory (genuine user data)
    os.chdir(pairs.conf.DATA_DIR)
    features_filenames = os.listdir('.')
    # Get chunk size from conf
    size_of_chunk = conf.CHUNK_SIZE

    i = 0
    for chunk_genuine_pairs, chunk_impostor_pairs in read_and_make_pairs(features_filenames, size_of_chunk):
        # Trim list of impostor pairs to the size of the genuine pairs list
        chunk_impostor_pairs = chunk_impostor_pairs[:len(chunk_genuine_pairs)]

        chunk_batches = make_pair_batches(chunk_genuine_pairs, chunk_impostor_pairs)

        # TODO: model.fit(x=(first_sequences, second_sequences), y=target_distances etc.)
        # TODO: !!! Think about how you'll randomly shuffle the pairs & about setting aside a number of pairs
        #       for validation & for testing

        print(f'Chunk #{i + 1}')


if __name__ == "__main__":
    compute_pairs_dataset()
