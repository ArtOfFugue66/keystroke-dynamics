# Keystroke sequence pairing package  

- This package provides utility functions for reading data from 
files created using the _features_ module and making positive 
(genuine) and negative (impostor) pairs from feature DataFrames.
- The dataset files are read from the features dataset into 
Pandas DataFrame objects using `pandas.DataFrame.read_csv()`.
- After reading the data:

|     * | PARTICIPANT_ID* | TEST_SECTION_ID* | HOLD_LATENCY** | INTERKEY_LATENCY** | PRESS_LATENCY** | RELEASE_LATENCY** |
|------:|----------------:|-----------------:|---------------:|-------------------:|----------------:|------------------:|
|     0 |           10322 |           109736 |          0.051 |                  0 |               0 |                 0 |
|     1 |           10322 |           109736 |          0.192 |               0.28 |           0.331 |             0.472 |
|     2 |           10322 |           109736 |          0.065 |              0.064 |           0.128 |             0.001 |
|     3 |           10322 |           109736 |          0.056 |              0.143 |           0.208 |             0.199 |
|     4 |           10322 |           109736 |          0.064 |               0.08 |           0.136 |             0.144 |
|     5 |           10322 |           109736 |          0.056 |              0.432 |           0.496 |             0.488 |
|     6 |           10322 |           109736 |          0.096 |                1.2 |           1.256 |             1.296 |
|     7 |           10322 |           109736 |          0.128 |              0.024 |            0.12 |             0.152 |
|     8 |           10322 |           109736 |          0.064 |              0.144 |           0.272 |             0.208 |
|     9 |           10322 |           109736 |          0.072 |              0.064 |           0.128 |             0.136 |
| [...] |           [...] |            [...] |          [...] |              [...] |           [...] |             [...] |

\* A feature DataFrame; Identical to the expected output DataFrames of `features.main()`

- By using `pairs.utils.make_pairs_from_feature_dfs()`, which calls
`pairs.utils.make_pairs_for_user()`, feature DataFrames in a
`pairs.conf.CHUNK_SIZE`-sized pool are used to generate all possible 
positive pairs for a single user (15 x 14 / 2 = 105) and negative pairs 
for that same user, using the feature DataFrames of all the other users 
in the pool  
  
- A keystroke sequence pair is defined as a tuple containing the _1st
keystroke sequence_, the _2nd keystroke sequence_ (either from the same
user or another user based on the pair type) and the _target distance_
(0 for _positive_ pairs, `pairs.conf.MARGIN` for _negative_ pairs)

- The tuples representing keystroke sequence pairs can be used in a data
pipeline: they can be batched and fed to a neural network in order to 
obtain a trained model, validate it and test it

---
