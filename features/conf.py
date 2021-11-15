BASE_DATA_DIR = "C:/WORK/MASTER/DISERTATIE/Datasets"
SMALL_DATASET_DIR = f"{BASE_DATA_DIR}/136m-keystrokes-small"
WHOLE_DATASET_DIR = f"{BASE_DATA_DIR}/136m-keystrokes-full/Keystrokes/files"

OUTPUT_DIR = f"{BASE_DATA_DIR}/features"
# GENUINE_DATA_PATH = f"{BASE_OUTPUT_DIR}/genuine"
# IMPOSTOR_DATA_PATH = f"{BASE_OUTPUT_DIR}/impostor"

# ----------- #

SEQUENCE_LENGTH = 70
NO_THREADS = 10

CHUNK_SIZE = 100
THREAD_CHUNK_SIZE = int(CHUNK_SIZE / 10)