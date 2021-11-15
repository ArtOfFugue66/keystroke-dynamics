DATA_DIR = "C:/WORK/MASTER/DISERTATIE/Datasets/features"

MARGIN = 0.1  # The 'alpha' in the Contrastive Loss formula

CHUNK_SIZE = 100

BASE_DATA_DIR = "C:/WORK/MASTER/DISERTATIE/Datasets"
OUTPUT_DIR = f"{BASE_DATA_DIR}/features"
GENUINE_DATA_PATH = f"{OUTPUT_DIR}/positive"
IMPOSTOR_DATA_PATH = f"{OUTPUT_DIR}/negative"

# Batch size: TODO: Determine the value for max accuracy
# Timesteps: number of keystrokes in a sequence
# Number of features: 4 (HOLD_LATENCY, INTERKEY_LATENCY, PRESS_LATENCY, RELEASE_LATENCY)
BATCH_SIZE, TIMESTEPS, NO_FEATURES = 512, 70, 4
