import features.conf


FEATURES_INPUT_DIR = features.conf.OUTPUT_DIR
CHUNK_SIZE = 100
THREAD_CHUNK_SIZE = 10
MARGIN = 0.1  # The 'alpha' in the Contrastive Loss formula TODO: Experiment with different values for this
# Batch size: TODO: Determine the value for max accuracy
BATCH_SIZE = 512
