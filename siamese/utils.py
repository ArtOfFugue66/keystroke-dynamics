from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Lambda, BatchNormalization, Dense, Masking, LSTM, Dropout

from . import conf as conf
from pairs import conf as pairs_conf

def euclidean_distance(sequence_embeddings: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
    """
    Function that computes and returns the Euclidean distance between two vectors.
    The parameter should be a tuple of two embedding vectors output by the LSTM tower network.

    :param sequence_embeddings: Embedding vectors
    :return: Euclidean distance
    """
    xi_vect, xj_vect = sequence_embeddings

    # # Option 1: Using tensor operations
    eucl_dist = tf.norm(tf.math.subtract(xi_vect, xj_vect), ord='euclidean', keepdims=True)
    return eucl_dist

    # Option 2: Using numpy operations
    # eucl_dist = np.linalg.norm(xi_vect - xj_vect)
    # return eucl_dist

def contrastive_loss(y_true, y_pred):
    """
    TODO: Implement & try using this as the loss function, in a separate git branch

    :param y_true:
    :param y_pred:
    :return:
    """
    pass

def make_lstm(input_shape: Tuple[int, int],
              embedding_dimensions: int) -> tf.keras.Model:
    """
    Define the LSTM sister network structure.
    This network is used in the Siamese model.

    :param input_shape: Shape of the input for the LSTM network
    :param embedding_dimensions: Desired number of dimensions for embedding vectors
    computed by the LSTM network
    :return: LSTM sister network object
    """
    # Input ("visible") layer of the LSTM network
    visible = Input(shape=input_shape, name="LSTM_Input")

    # Masking layer results in rows (or "timesteps") containing only
    # 'mask_value' values to be skipped from calculations.
    masking = Masking(mask_value=0.0, name="LSTM_Masking")(visible)

    # Batch normalization layer re-centers and re-scales data to make the network faster and more stable
    batch_norm_1 = BatchNormalization(name="LSTM_Batch_Norm_1")(masking)

    # LSTM layers carry information specific to early timesteps (i.e., keypresses) into later timesteps.
    # Setting 'return_sequences' parameter to True instructs this layer to return output features for
    # all timesteps in the input sequence, as opposed to only returning the output features for the last timestep.
    lstm_1 = LSTM(embedding_dimensions, name="LSTM_1", activation="tanh", return_sequences=True)(batch_norm_1)

    # The dropout layer aids in preventing over-fitting on the data presented to the final model.
    # To achieve this, visible & hidden units (with associated weights) are chosen at random and dropped.
    dropout = Dropout(0.5, name="LSTM_Dropout")(lstm_1)

    batch_norm_2 = BatchNormalization(name="LSTM_Batch_Norm_2")(dropout)

    # Not setting 'return_sequences' to True because we only need the output features for the FINAL
    # timestep in the sequence (i.e., output shape should be 1 x 128 as opposed to SEQUENCE_LENGTH x 128)
    lstm_2 = LSTM(embedding_dimensions, name="LSTM_2", activation="tanh")(batch_norm_2)

    # NOTE: No need to add a loss to this model since the weights should update based
    #       on the value of the contrastive loss, used in the Siamese model
    # Return a tf.python.keras.Model object with specified input layers & output layers
    return Model(inputs=visible, outputs=lstm_2, name="Sister_network")

def name_model(loss, batch_size: int, embedding_dims, optimizer=None):
    """
    Function that generates a uniquely identifying name for a model architecture
    based on the relevant objects that are used to compile the model.

    :param loss: Loss function object
    :param batch_size: Number of samples processed before the model weights are updated
    :param embedding_dims: Dimension count of vectors output by the sister network
    :param optimizer: Optimizer object used to increase training efficiency
    :return: A uniquely identifying name for the model that includes info from all params
    """
    prefix = conf.SIAMESE_NAME_PREFIX
    loss_str = 'TFAddonsCL' if type(loss) == tfa.losses.ContrastiveLoss else 'CustomCL'
    optimizer_str = 'NOOPT' if optimizer is None else str(optimizer)  # TODO: Debug this when an optimizer is added
    batch_size_str = str(batch_size) + "BS"
    embedding_dims_str = str(embedding_dims) + "ED"

    return f"{prefix}_{loss_str}_{optimizer_str}_{batch_size_str}_{embedding_dims_str}"

def make_siamese(batch_size: int,
                 input_shape: Tuple[int, int],
                 embedding_dimensions: int = conf.EMBEDDING_DIMENSIONS) -> tf.keras.Model:
    """
    Define the Siamese network structure with a LSTM network as the tower (sister) network.

    :param batch_size: Number of samples (sequences) to embed before
    updating the model's weights
    :param input_shape: Shape of a single sample
    :param embedding_dimensions: The number of dimensions of the output
    features (embedding vector) of a single sample
    :return: Compiled Siamese model
    """
    # Get an instance of the LSTM model that will compute embeddings from input keystroke sequences
    lstm_model = make_lstm(input_shape, embedding_dimensions)

    # Define inputs to the Siamese RNN (keystroke sequences)
    xi_input = Input(batch_size=batch_size, shape=input_shape, name="Input_sequence_1")
    xj_input = Input(batch_size=batch_size, shape=input_shape, name="Input_sequence_2")

    # Define embeddings of the two input sequences
    xi_embedded, xj_embedded = lstm_model(xi_input), lstm_model(xj_input)

    # Use the embeddings of the two input sequences in a layer
    # that computes the Euclidean distance between them
    merge_layer = Lambda(euclidean_distance, name="Euclidean_distance")([xi_embedded, xj_embedded])

    # Define the output layer; 'softmax' activation because the output should
    # be interpreted as the probability that the two input sequences are similar
    output_layer = Dense(1, activation="softmax")(merge_layer)

    # Set up parameters required for creating & compiling the model
    model_loss = tfa.losses.ContrastiveLoss(margin=pairs_conf.MARGIN,
                                            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                                            name="contrastive_loss")
    # model_loss = tf.losses.BinaryCrossentropy()
    model_optimizer = None  # TODO after you get the model to train
    model_metrics = ['accuracy']
    model_name = name_model(model_loss, batch_size, embedding_dimensions, model_optimizer)

    # Instantiate a Siamese model that takes as input two keystroke sequences and outputs
    # the confidence (probability) that they are similar (belong to the same user)
    model = Model(inputs=[xi_input, xj_input], outputs=output_layer, name=model_name)

    # Compile the model with the Contrastive loss
    model.compile(loss=model_loss, metrics=model_metrics)

    return model
