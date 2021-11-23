BASE_DATA_DIR = "C:/WORK/MASTER/DISERTATIE/Datasets"
FEATURES_DATA_DIR = f"{BASE_DATA_DIR}/features"
OUTPUT_DIR = f"{BASE_DATA_DIR}/pairs"
GENUINE_DATA_PATH = f"{OUTPUT_DIR}/positive"
IMPOSTOR_DATA_PATH = f"{OUTPUT_DIR}/negative"
CHUNK_SIZE = 10
MARGIN = 0.1  # The 'alpha' in the Contrastive Loss formula

# Batch size: TODO: Determine the value for max accuracy
BATCH_SIZE = 512
