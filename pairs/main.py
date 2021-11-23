from __future__ import annotations
import os
from typing import List

import conf.pairs as conf
from pairs.pairing import make_pairs_from_features_dfs


def process_write_pairs(features_chunk: List[str], index_of_chunk: int):
    """
    Writes positive & negative keystroke sequence pairs to the disk,
    using HDF5 format (1 HDF5 file per chunk of features dataset).

    :param features_chunk: Chunk of features DataFrames to be used for making pairs.
    :param index_of_chunk: Index of chunk in the whole features dataset.
    :return:
    """
    from utils.pairs import read_features_from_dataset

    # Read files belonging to the current chunk
    chunk_dfs = read_features_from_dataset(features_chunk)  # read DataFrame for each user,
    # Make positive & negative pairs for the current chunk
    chunk_genuine_pairs, chunk_impostor_pairs = make_pairs_from_features_dfs(chunk_dfs, index_of_chunk)
    # Trim list of impostor pairs to the size of the genuine pairs list
    chunk_impostor_pairs = chunk_impostor_pairs[:len(chunk_genuine_pairs)]
    # Define name of HDF5 file for the current chunk
    hdf_filename = f"chunk_{index_of_chunk}_data.h5"
    # Write genuine & impostor pairs of the current chunk to appropriate directories
    for pair_index, (genuine_pair, impostor_pair) in enumerate(zip(chunk_genuine_pairs, chunk_impostor_pairs)):
        genuine_pair.to_hdf(f"{conf.PAIRS_OUTPUT_DIR}/{hdf_filename}", key=f"genuine_{pair_index}", mode="a")
        impostor_pair.to_hdf(f"{conf.PAIRS_OUTPUT_DIR}/{hdf_filename}", key=f"impostor_{pair_index}", mode="a")

def compute_pairs_dataset():
    """
    This function will run if this Python file is run directly.
    It reads files from the features dataset into Pandas DataFrames in a chunk-by-chunk fashion,
    makes the genuine & impostor pairs for each user in the chunk and groups them proportionally
    into batches.

    :return: None
    """
    from utils.general import list_to_chunks_by_size
    from multiprocessing import Process
    from tqdm import tqdm

    # Read files from processed dataset directory (genuine user data)
    os.chdir(conf.FEATURES_INPUT_DIR)
    features_filenames = os.listdir('.')
    # Get chunk size from conf
    size_of_chunk = conf.CHUNK_SIZE

    all_features_chunks = list(list_to_chunks_by_size(features_filenames, size_of_chunk))

    for outer_chunk_index, outer_chunk in tqdm(enumerate(all_features_chunks), total=len(all_features_chunks), desc="[INFO] Processing features dataset chunks"):  # For current chunk of filenames,
        thread_chunks = list(list_to_chunks_by_size(outer_chunk, conf.THREAD_CHUNK_SIZE))
        process_list = []
        for inner_chunk_index, inner_chunk in enumerate(thread_chunks):
            # Create a process that handles the files,
            process = Process(target=process_write_pairs, args=(inner_chunk, inner_chunk_index), name=f"process-{inner_chunk_index}")
            # append in to the process list
            process_list.append(process)
            # and start it
            process.start()

        for process in process_list:
            # Wait for each process in the list to finish execution
            process.join()


if __name__ == "__main__":
    compute_pairs_dataset()
