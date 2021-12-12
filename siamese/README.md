# DNN definitions package  

- This package provides functions that define the architectures of
relevant DNNs and return objects that inherit from the `tf.keras.Model` 
class.
- The `siamese.utils.euclidean_distance()` function computes
the Euclidean distance between two (embedding) vectors.
- The `siamese.utils.make_lstm()` function defines the architecture
of the tower (sister) network that serves as the base of the Siamese
model. The purpose of this tower network is to compute embedding
vectors (output features) using input features of keystroke sequences.
- The `siamese.utils.make_siamese()` function defines the architecture
of the full Siamese network, which uses the tower network to compute 
embeddings of the input sequences and updates the tower network's 
weights based on the loss (i.e., **_contrastive loss_**) value computed
using the two embedding vectors.

---
