from tensorflow.keras.layers import Input, BatchNormalization, Masking, LSTM, Dropout
from tensorflow.keras.models import Model


def make_lstm(input_shape, embedding_dimensions):
    visible = Input(shape=input_shape, name="LSTM_Input")
    masking = Masking(0.0, name="LSTM_Masking")(visible)
    batch_norm_1 = BatchNormalization(name="LSTM_Batch_Norm_1")(masking)
    lstm_1 = LSTM(embedding_dimensions, name="LSTM_1", activation="tanh", return_sequences=True)(batch_norm_1)
    dropout = Dropout(0.5, name="LSTM_Dropout")(lstm_1)
    batch_norm_2 = BatchNormalization(name="LSTM_Batch_Norm_2")(dropout)
    lstm_2 = LSTM(embedding_dimensions, name="LSTM_2", activation="tanh", return_sequences=True)(batch_norm_2)

    return Model(inputs=visible, outputs=lstm_2, name="Sister_network")
