# Project tasks 

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
- [x] Refactor project & code structure to be easier to debug and refactor
- [x] Solve error handling in `pairs.make_pairs_from_features_dfs()`; Also print error filename in `except` statement
- [x] Add comments to all lines in all files in all modules
- [x] Add READMEs to each package in the project
- [ ] Main program workflow documentation (step-by-step explanation of the data manipulation process)
- [ ] Write `siamese` module: Should contain everything related to the Siamese RNN necessary for the user authentication
  - [x] Use [Image similarity estimation using a Siamese Network with a contrastive loss](https://keras.io/examples/vision/siamese_contrastive/) tutorial as reference
  - [x] Write function that builds "tower" network (LSTM embedding network)
  - [x] Write function that builds Siamese network
  - [ ] Migrate code that builds the NAME for the Siamese network to a separate function & call it in `siamese.utils.make_lstm()`
  - [ ] Write function that validates & tests a Siamese model
  - [ ] Write unit tests for the whole module
  - [ ] Write code that calls Tensorboard to plot metrics of model training/validation/testing
- [x] Build main module: Should read raw data, compute features, make pairs, do training-validation-testing split, call training method on Siamese RNN model
- [ ] Tune the `siamese` module until you confirm, using your own typing data, that the model works as intended
  - [ ] Add an optimizer in `siamese.utils.make_siamese()` & add its name to the model name
- [ ] (Optional) Refactor all data processing operations to use the `tf.data` APIf
