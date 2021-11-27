import tensorflow as tf


def euclidean_distance(sequences):
    xi_vect, xj_vect = sequences
    eucl_dist = tf.linalg.norm(xi_vect - xj_vect)
    return eucl_dist

def contrastive_loss(y_true, y_pred):
    pass  # TODO if you cannot obtain results using tfa.losses.ContrastiveLoss
