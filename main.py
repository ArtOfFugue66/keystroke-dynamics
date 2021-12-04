import os
import numpy as np
from loguru import logger

from features import conf as features_conf
from siamese import conf as siamese_conf
from common import utils as common_utils
from pairs import utils as pairing_utils


@logger.catch()
def main():
    from siamese.utils import make_siamese

    chunk_size = features_conf.CHUNK_SIZE  # Get size of a dataset chunk from features conf file
    features_filenames = os.listdir(features_conf.OUTPUT_DIR)  # List the features dataset directory
    np.random.shuffle(features_filenames)  # Randomly shuffle the dataset filenames list
    features_filenames_chunks = common_utils.list_to_chunks_by_size(features_filenames[:chunk_size * 2],
                                                                    chunk_size)  # Split the list of filenames in chunks
    # features_filenames_chunks = utils.general.list_to_chunks_by_size(features_filenames, chunk_size)  # Split the list of filenames in chunks

    all_chunk_batches = []

    for chunk_index, chunk in enumerate(features_filenames_chunks):  # Iterate through the chunks of filenames
        chunk_of_features = pairing_utils.read_features_from_dataset(chunk, chunk_index)  # Read each file in DataFrames
        chunk_genuine_pairs, chunk_impostor_pairs = pairing_utils.make_pairs_from_features_dfs(chunk_of_features,
                                                                                               chunk_index)  # Make genuine & impostor pairs for all users in the chunk
        all_chunk_batches.extend(common_utils.make_pair_batches(chunk_genuine_pairs,
                                                                chunk_impostor_pairs,
                                                                batch_size=siamese_conf.BATCH_SIZE,
                                                                shuffle_batches_flag=True))  # Make batches using the genuine & impostor pairs of this chunk
    # Get total number of batches
    total_no_batches = len(all_chunk_batches)
    # Perform train-test-validation split on list of batches
    train_batches, test_batches, validation_batches = all_chunk_batches[:int(total_no_batches * 0.6)], \
                                                      all_chunk_batches[
                                                      int(total_no_batches * 0.6):int(total_no_batches * 0.8)], \
                                                      all_chunk_batches[int(total_no_batches * 0.8):]
    # Unravel keystroke pairs from all batches
    train_first_sequences, train_second_sequences, train_target_distances = common_utils.unravel_batches(train_batches)
    test_first_sequences, test_second_sequences, test_target_distances = common_utils.unravel_batches(test_batches)
    validation_first_sequences, validation_second_sequences, validation_target_distances = common_utils.unravel_batches(
        validation_batches)
    # Get a siamese RNN model object
    batch_size, input_shape, emb_dims = siamese_conf.BATCH_SIZE, siamese_conf.INPUT_SHAPE, siamese_conf.EMBEDDING_DIMENSIONS
    siamese_model = make_siamese(batch_size, input_shape, emb_dims)

    train_first_sequences = np.asarray(train_first_sequences)
    train_second_sequences = np.asarray(train_second_sequences)
    train_target_distances = np.asarray(train_target_distances)

    history = siamese_model.fit(
        x=(train_first_sequences, train_second_sequences),
        y=train_target_distances,
        batch_size=siamese_conf.BATCH_SIZE,
        epochs=10,
        shuffle=False
    )


if __name__ == "__main__":
    main()
