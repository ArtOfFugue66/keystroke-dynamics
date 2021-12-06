# Directory containing original (raw) dataset
BASE_DATA_DIR = "C:/WORK/MASTER/DISERTATIE/Datasets"
# Directory containing 4.000-files (small) dataset
SMALL_DATASET_DIR = f"{BASE_DATA_DIR}/136m-keystrokes-small"
# Directory containing 168.000-files (full) dataset
WHOLE_DATASET_DIR = f"{BASE_DATA_DIR}/136m-keystrokes-full/Keystrokes/files"
# Directory to write files containing the 4 computed features for each user
OUTPUT_DIR = f"{BASE_DATA_DIR}/features"
# Fixed length of a keystroke sequence
SEQUENCE_LENGTH = 70
# Size of a dataset chunk
CHUNK_SIZE = 100
# Number of files to be handled by a single Process object
PROCESS_CHUNK_SIZE = int(CHUNK_SIZE / 10)
