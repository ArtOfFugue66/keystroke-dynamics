from __future__ import annotations

import os

from tqdm import tqdm
from loguru import logger

import conf
import pairs.conf
import utils
from pairs.pairing import make_pairs, read_and_make_pairs
from features.utils import list_to_chunks_by_size


if __name__ == "__main__":
    # Read files from processed dataset directory (genuine user data)
    os.chdir(conf.DATA_DIR)
    features_filenames = os.listdir('.')
    # Get chunk size from conf
    size_of_chunk = conf.CHUNK_SIZE
    # Get batch size from conf
    size_of_batch = conf.BATCH_SIZE

    i = 0
    for chunk_genuine_pairs, chunk_impostor_pairs in read_and_make_pairs(features_filenames, size_of_chunk):
        # Trim list of impostor pairs to the size of the genuine pairs list
        chunk_impostor_pairs = chunk_impostor_pairs[:len(chunk_genuine_pairs)]

        # Make half-batches from the current dataset chunk
        genuine_half_batches = list_to_chunks_by_size(chunk_genuine_pairs, size_of_batch / 2)
        impostor_half_batches = list_to_chunks_by_size(chunk_impostor_pairs, size_of_batch / 2)

        ### Group half-batches into batches
        batches = []
        # Iterate through both lists at once
        for genuine_half, impostor_half in zip(genuine_half_batches, impostor_half_batches, strict=True):
            batch = genuine_half.append(impostor_half)

            first_sequences, second_sequences, target_distances = [], [], []

            for pair in batch:
                first_sequences.append(pair[0])
                second_sequences.append(pair[1])
                target_distances.append(pair[2])

            # TODO: model.fit(x=(first_sequences, second_sequences), y=target_distances etc.)
            # TODO: !!! Think about how you'll randomly shuffle the pairs & about setting aside a number of pairs
            #       for validation & for testing

        print(f'Chunk #{i+1}')

