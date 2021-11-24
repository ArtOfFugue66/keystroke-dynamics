# Keystroke dynamics biometrics using Recurrent Neural Networks

---

- To-be sub-repository of Master's Thesis project, 'TypeFace'
- Based on the paper **TypeNet: Deep Learning Keystroke Biometrics** (Acien et al.)
- Using the dataset **136M Keystrokes**; Publication: **Observations on Typing from 136 Million Keystrokes** (Dhakal et
  al.)

---

- Python version: 3.8.12 (Anaconda 3 venv)
- For data preprocessing & preparation: Pandas, Numpy
- For training of the neural network: Tensorflow/Keras

---

### TODO

- [x] Refactor tests in `pairs/tests/test_pairing.py` & `pairs/pairing.py:make_pairs()` until all tests pass
- [x] Finish implementation of `pairs/pairing.py:make_pair_batches()`(start off with unit tests for it)
- [x] Analyze & decide on a method to store the keystroke sequence pairs created on disk
    - [ ] ~~Option 1: Create 1 file for each pair (either .csv or .pickle)~~  
      + ~~Easier to store pairs since there is one pair / file~~  
      - ~~Lower performance when making the batches since there will be many calls to file reading function~~  
      - ~~Lots of resulting files~~
    - [x] Option 2: Concatenate all pairs of a chunk (positive & negative ones separately) into one DataFrame & write it
      to HDF5 file  
      + Less resulting files (as many files as there are threads processing chunks of the features dataset)
- [ ] Write `siamese` module: Should contain everything related to the Siamese RNN necessary for the user authentication
  - [ ] Use [Image similarity estimation using a Siamese Network with a contrastive loss](https://keras.io/examples/vision/siamese_contrastive/) tutorial as reference
- [ ] Build main module: Should read raw data, compute features, make pairs, do training-validation-testing split, call training method on Siamese RNN model