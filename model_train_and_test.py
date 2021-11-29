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
    from siamese.siamese import make_siamese

    chunk_size = conf.features.CHUNK_SIZE  # Get size of a dataset chunk from features conf file
    features_filenames = os.listdir(conf.features.OUTPUT_DIR)  # List the features dataset directory
    np.random.shuffle(features_filenames)  # Randomly shuffle the items in the
    features_filenames_chunks = utils.general.list_to_chunks_by_size(features_filenames[:chunk_size*2],
                                                                     chunk_size)  # Split the list of filenames in chunks

    all_chunk_batches = []

    for chunk_index, chunk in enumerate(features_filenames_chunks):  # Iterate through the chunks of filenames
        chunk_of_features = utils.pairs.read_features_from_dataset(chunk, chunk_index)  # Read each file in DataFrames
        chunk_genuine_pairs, chunk_impostor_pairs = pairs.pairing.make_pairs_from_features_dfs(chunk_of_features,
                                                                                               chunk_index)  # Make genuine & impostor pairs for all users in the chunk
        all_chunk_batches.extend(pairs.pairing.make_pair_batches(chunk_genuine_pairs,
                                                                 chunk_impostor_pairs,
                                                                 batch_size=conf.siamese.BATCH_SIZE,
                                                                 shuffle_batches_flag=True))  # Make batches using the genuine & impostor pairs of this chunk
    # Get total number of batches
    total_no_batches = len(all_chunk_batches)
    # Perform train-test-validation split on list of batches
    train_batches, test_batches, validation_batches = all_chunk_batches[:int(total_no_batches*0.6)], \
                                                      all_chunk_batches[int(total_no_batches*0.6):int(total_no_batches*0.8)], \
                                                      all_chunk_batches[int(total_no_batches * 0.8):]
    # Unravel keystroke pairs from all batches
    # TODO: len(train_first_sequences) & len(validation_first_sequences) not multiples of 512,
    #       BECAUSE the last batch in the 'all_chunk_batches' list is not 512 pairs long. Fix this in pairs.pairing.make_pair_batches()
    train_first_sequences, train_second_sequences, train_target_distances = utils.general.unravel_batches(train_batches)
    test_first_sequences, test_second_sequences, test_target_distances = utils.general.unravel_batches(test_batches)
    validation_first_sequences, validation_second_sequences, validation_target_distances = utils.general.unravel_batches(validation_batches)
    # Get a siamese RNN model object
    batch_size, input_shape, emb_dims = conf.siamese.BATCH_SIZE, conf.siamese.INPUT_SHAPE, conf.siamese.EMBEDDING_DIMENSIONS
    siamese_model = make_siamese(batch_size, input_shape, emb_dims)


if __name__ == "__main__":
    main()
