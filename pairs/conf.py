import features.conf

# Directory containing features dataset files
FEATURES_INPUT_DIR = '../features-dataset'
# FEATURES_INPUT_DIR = features.conf.OUTPUT_DIR
# Size of a dataset chunk
CHUNK_SIZE = 100
# The 'alpha' in the Contrastive Loss formula TODO: Experiment with different values for this
MARGIN = 0.1
# Batch size: TODO: Determine the value for max accuracy
BATCH_SIZE = 512
