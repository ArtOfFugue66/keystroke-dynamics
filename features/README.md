# Temporal features computing package  

- This package is responsible with reading the data from the 
**136M Keystrokes** dataset and computing temporal features
from the raw timing data (_key press_ and _key release_ timestamps)
of a user  

- The dataset files are read into Pandas DataFrame objects using 
`pandas.DataFrame.read_csv()`. Only the `PARTICIPANT_ID`, 
`TEST_SECTION_ID`, `PRESS_TIME`, and `RELEASE_TIME` columns 
are selected. 
- After reading the data:
  
 |       | PARTICIPANT_ID* | TEST_SECTION_ID* | PRESS_TIME** | RELEASE_TIME** |
|------:|----------------:|-----------------:|-------------:|---------------:|
|     0 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
|     1 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
|     2 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
|     3 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
|     4 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
|     5 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
|     6 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
|     7 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
|     8 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
|     9 |           10322 |           109736 |  1.47205e+12 |    1.47205e+12 |
| [...] |           [...] |            [...] |        [...] |          [...] |

- The operations performed on the data read, in order:
  - The temporal features (`HOLD_LATENCY`, `INTERKEY_LATENCY`, `PRESS_LATENCY`, 
  `RELEASE_LATENCY`) are computed for each sequence of keystrokes, 
  - The temporal feature values are scaled to the [0, 1] interval,
  - Each sequence is padded or trimmed to a fixed length (number of timesteps),
  - All sequences corresponding to a user are concatenated into a single DataFrame.
- After these operations are performed:

|       | PARTICIPANT_ID* | TEST_SECTION_ID* | HOLD_LATENCY** | INTERKEY_LATENCY** | PRESS_LATENCY** | RELEASE_LATENCY** |
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

  
\* __np.int32__ datatype  
\** __np.float64__ datatype  

- The DataFrame objects containing the processed data of all the users 
selected from the dataset can then be input into `pairs.utils` functions
in order to make positive (genuine) & negative (impostor) pairs, to be
used for training, testing and validating Siamese model architectures

---
