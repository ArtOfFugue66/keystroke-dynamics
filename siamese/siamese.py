import tensorflow_addons as tfa
from tensorflow.keras.layers import Input, BatchNormalization, Lambda, Dense
from tensorflow.keras.models import Model

import utils.siamese as utils
from rnn import make_lstm
import conf.siamese as conf


def make_siamese(batch_size, input_shape, embedding_dimensions=128):
    # Get an instance of the LSTM model that will compute embeddings from input keystroke sequences
    lstm_model = make_lstm(input_shape, embedding_dimensions)
    # Define inputs to the Siamese RNN (keystroke sequences)
    xi_input = Input(batch_size=batch_size, shape=input_shape, name="Input_sequence_1")
    xj_input = Input(batch_size=batch_size, shape=input_shape, name="Input_sequence_2")

    # Define embeddings of the two input sequences
    xi_embedded, xj_embedded = lstm_model(xi_input), lstm_model(xj_input)

    # "Merge" the embeddings of the two input sequences by computing their Euclidean distance
    merge_layer = Lambda(utils.euclidean_distance, name="Euclidean_distance")([xi_embedded, xj_embedded])

    # Add another layer of normalization
    batch_norm_layer = BatchNormalization(name="Batch_Norm")(merge_layer)

    # Define the output layer; 'softmax' activation because the output should be interpreted as
    # the probability that the two input sequences are similar
    output_layer = Dense(1, activation="softmax")(batch_norm_layer)

    # Define the 'full' Siamese RNN model that takes as input two keystroke sequences and outputs
    # the probability that they are similar (they belong to the same user)
    model = Model(inputs=[xi_input, xj_input], outputs=output_layer)

    # Compile the model with the Contrastive loss
    # TODO: Add optimizer parameter & tweak the optimizer-related parameters
    model.compile(loss=tfa.losses.ContrastiveLoss, metrics=['accuracy'])

    # TODO: Generate the name for the Siamese model before returning it,
    #       based on values such as embedding dimensions etc. (see Sentdex videos)
    # TODO: Assign the generated name to the model
    # Generate a unique name for the model using relevant parameters; The purpose of this is to
    # make performance comparisons of different model configurations easier
    model_name = f"{conf.SIAMESE_NAME_PREFIX}_{batch_size}_{embedding_dimensions}"
    return model
