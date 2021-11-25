# Batch size: number of samples in a training/test/validation batch
# Timesteps: number of keystrokes in a sequence
# Number of features: 4 (HOLD_LATENCY, INTERKEY_LATENCY, PRESS_LATENCY, RELEASE_LATENCY)
BATCH_SIZE, TIMESTEPS, NO_FEATURES = 512, 70, 4
INPUT_SHAPE = (TIMESTEPS, NO_FEATURES)
EMBEDDING_DIMENSIONS = 128

SIAMESE_NAME_PREFIX = "Siamese_RNN"
