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
- [x] Main program workflow documentation (step-by-step explanation of the data manipulation process)
- [ ] Write `capture` module: Should contain everything related to capturing timestamps of key presses & releases
  - [x] Use two lists, one for **keydown** events & one for **keyup** events, to implement event pairing more easily
  - [x] Disregard events captured after the `2 * conf.SEQUENCE_LENGTH` **keyup**
  - [x] Identify & fix error that occasionally swaps places of two consecutive keypresses (see line 53-54 in .txt)
  - [ ] Implement additional logic into the module
    - [x] Increment conf.PARTICIPANT_ID on program run
    - [x] Capture more sequences (repeatedly capture keystrokes from e.g., 5 different sequences)
    - [x] Keep track of and increment test section ID of a given user
    - [ ] Fine tune the newly-written capture code
      - [ ] Reduce processing time for a single keyboard event; Else we lose keystroke timing data
      - [ ] Stop program for aprox. 2s when typing for a section is done and prompt for him/her to write another sentence
    - [ ] Tidy up the capture code
    - [ ] Write unit tests
    - [ ] Manually test things like how the data is affected if the last event is a keydown event
  - [ ] Migrate functions into separate file(s) inside the module
- [ ] Write `siamese` module: Should contain everything related to the Siamese RNN necessary for the user authentication
  - [x] Use [Image similarity estimation using a Siamese Network with a contrastive loss](https://keras.io/examples/vision/siamese_contrastive/) tutorial as reference
  - [x] Write function that builds "tower" network (LSTM embedding network)
  - [x] Write function that builds Siamese network
  - [x] Migrate code that builds the NAME for the Siamese network to a separate function & call it in `siamese.utils.make_lstm()`
  - [ ] Write function that validates & tests a Siamese model
  - [ ] Write unit tests for the whole module
  - [ ] Write code that calls Tensorboard to plot metrics of model training/validation/testing
- [x] Build main module: Should read raw data, compute features, make pairs, do training-validation-testing split, call training method on Siamese RNN model
  - [ ] Write function that reads keystroke info of enrolled participant into DataFrame object(s) (`main.py`)
- [ ] Tune the `siamese` module until you confirm, using your own typing data, that the model works as intended
  - [ ] Add an optimizer in `siamese.utils.make_siamese()` & add its name to the model name
- [ ] (Optional) Refactor all data processing operations to use the `tf.data` API
