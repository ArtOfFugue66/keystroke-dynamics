import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.layers import Lambda, BatchNormalization, Dense, Masking, LSTM, Dropout

from . import conf as conf
from pairs import conf as pairs_conf

def euclidean_distance(sequence_embeddings):
    xi_vect, xj_vect = sequence_embeddings

    eucl_dist = tf.norm(tf.math.subtract(xi_vect, xj_vect), ord='euclidean', keepdims=True)
    # Using tf.reshape() to avoid the 'ValueError: Input has undefined rank.'
    # when this value is passed to a BatchNormalization layer in siamese.siamese.make_siamese()
    # return tf.reshape(eucl_dist, shape=(1, 1))
    return eucl_dist

    # eucl_dist = np.linalg.norm(xi_vect - xj_vect)
    # return eucl_dist

def contrastive_loss(y_true, y_pred):
    """
    TODO if you cannot obtain good results using tfa.losses.ContrastiveLoss

    :param y_true:
    :param y_pred:
    :return:
    """
    pass

def make_lstm(input_shape, embedding_dimensions):
    visible = Input(shape=input_shape, name="LSTM_Input")
    masking = Masking(0.0, name="LSTM_Masking")(visible)
    batch_norm_1 = BatchNormalization(name="LSTM_Batch_Norm_1")(masking)
    lstm_1 = LSTM(embedding_dimensions, name="LSTM_1", activation="tanh", return_sequences=True)(batch_norm_1)
    dropout = Dropout(0.5, name="LSTM_Dropout")(lstm_1)
    batch_norm_2 = BatchNormalization(name="LSTM_Batch_Norm_2")(dropout)
    lstm_2 = LSTM(embedding_dimensions, name="LSTM_2", activation="tanh", return_sequences=True)(batch_norm_2)

    return Model(inputs=visible, outputs=lstm_2, name="Sister_network")

def make_siamese(batch_size, input_shape, embedding_dimensions=128):
    # Get an instance of the LSTM model that will compute embeddings from input keystroke sequences
    lstm_model = make_lstm(input_shape, embedding_dimensions)
    # Define inputs to the Siamese RNN (keystroke sequences)
    xi_input = Input(batch_size=batch_size, shape=input_shape, name="Input_sequence_1")
    xj_input = Input(batch_size=batch_size, shape=input_shape, name="Input_sequence_2")

    # Define embeddings of the two input sequences
    xi_embedded, xj_embedded = lstm_model(xi_input), lstm_model(xj_input)

    # "Merge" the embeddings of the two input sequences by computing their Euclidean distance
    merge_layer = Lambda(euclidean_distance, name="Euclidean_distance", output_shape=(1, 1))([xi_embedded, xj_embedded])

    # Add another layer of normalization
    # batch_norm_layer = BatchNormalization(name="Batch_Norm")(merge_layer)

    # Define the output layer; 'softmax' activation because the output should be interpreted as
    # the probability that the two input sequences are similar
    # output_layer = Dense(1, activation="softmax")(batch_norm_layer)
    output_layer = Dense(1, activation="softmax")(merge_layer)

    # Set up parameters required for creating & compiling the model
    # TODO: Add optimizer parameter & tweak the optimizer-related parameters
    model_loss = tfa.losses.ContrastiveLoss(margin=pairs_conf.MARGIN,
                                            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                                            name="contrastive_loss")
    model_metrics = ['accuracy']
    # Use a unique name for the model using relevant parameters; The purpose of this is to
    # make performance comparisons of different model configurations easier
    model_name = f"{conf.SIAMESE_NAME_PREFIX}_" \
                 f"{'TFAddonsCL' if type(model_loss) == tfa.losses.ContrastiveLoss else 'SelfCL'}_" \
                 f"{batch_size}BS_" \
                 f"{embedding_dimensions}ED"  # TODO: Add info on optimizer & optimizer parameters in the name

    # Define the 'full' Siamese RNN model that takes as input two keystroke sequences and outputs
    # the probability that they are similar (they belong to the same user)
    model = Model(inputs=[xi_input, xj_input], outputs=output_layer, name=model_name)

    # Compile the model with the Contrastive loss
    model.compile(loss=model_loss, metrics=model_metrics)

    return model
