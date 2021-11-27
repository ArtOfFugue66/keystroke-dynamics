import os
import numpy as np
from loguru import logger

import conf.features
import conf.siamese
import pairs.pairing
import utils.general
import utils.pairs


@logger.catch()
def main():
    from tqdm import tqdm

    chunk_size = conf.features.CHUNK_SIZE  # Get size of a dataset chunk from features conf file
    features_filenames = os.listdir(conf.features.OUTPUT_DIR)  # List the features dataset directory
    np.random.shuffle(features_filenames)  # Randomly shuffle the items in the
    features_filenames_chunks = utils.general.list_to_chunks_by_size(features_filenames,
                                                                     chunk_size)  # Split the list of filenames in chunks

    all_chunk_batches = []
    first_sequence_features = []
    second_sequence_features = []
    target_distances = []

    for chunk_index, chunk in enumerate(features_filenames_chunks):  # Iterate through the chunks of filenames
        chunk_of_features = utils.pairs.read_features_from_dataset(chunk, chunk_index)  # Read each file in DataFrames
        chunk_genuine_pairs, chunk_impostor_pairs = pairs.pairing.make_pairs_from_features_dfs(chunk_of_features,
                                                                                               chunk_index)  # Make genuine & impostor pairs for all users in the chunk
        all_chunk_batches.append(pairs.pairing.make_pair_batches(chunk_genuine_pairs,
                                                                 chunk_impostor_pairs,
                                                                 batch_size=conf.siamese.BATCH_SIZE,
                                                                 shuffle_batches_flag=True))  # Make batches using the genuine & impostor pairs of this chunk
    # Get total number of batches
    total_no_batches = len(all_chunk_batches)
    # TODO: Perform train-test-validation split on list of batches
    # train_batches, test_batches, validation_batches = np.split(all_chunk_batches, [int(len())])

    for batch_index, batch in tqdm(enumerate(all_chunk_batches), total=len(all_chunk_batches),
                                   desc=f"[BATCHES] Unraveling sequences from batches"):
        for seq1, seq2, target_distance in batch:
            first_sequence_features.append(seq1)
            second_sequence_features.append(seq2)
            target_distances.append(target_distance)

        # TODO: Perform train_test_validation split on the list of batches
        all_batches_data = (first_sequence_features, second_sequence_features, target_distances)


if __name__ == "__main__":
    main()
