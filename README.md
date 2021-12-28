# Keystroke dynamics biometrics using Recurrent Neural Networks

---

- To-be sub-repository of Master's Thesis project, **TypeFace**
- Based on the paper **TypeNet: Deep Learning Keystroke Biometrics** (Acien et al.)
- Using the dataset **136M Keystrokes**; Publication: **Observations on Typing from 136 Million Keystrokes** (Dhakal et
  al.)

---

- **Python version**: 3.8.12 (Anaconda 3 venv)
- **IDE**: PyCharm 2021.3 Professional Edition
- **Data preprocessing & preparation**: Pandas, Numpy
- **DNN training**: Tensorflow/Keras
- **DNN evaluation**: TensorBoard 

---
- To install required modules for the Python/Conda virtual environment used, run  
`pip install -r requirements.txt`
- To run existing unit tests, install `rednose` and run  
    ``` 
    pip install nosetests
    nosetests --rednose --verbose tests/ 
    ```
---

- This (main) package implements a full workflow, from reading and processing **temporal features** 
data to training, testing and validating a working Siamese model for keystroke pattern recognition.

- The operations performed inside this package:
1. The **features** dataset files are listed.
2. The files are read into Pandas DataFrames, on a chunk-by-chunk basis in order
to avoid memory starvation.
3. Each file (containing a single user's temporal features) is read and used to 
make **positive** (genuine) and **negative** (impostor) pairs, stored as named
tuple objects.
4. The resulting positive and negative pairs are grouped together into batches.
5. The resulting batches are split into three sub-groups: **training** batches, 
**testing** batches and **validation** batches.
6. The batches are "unraveled" (every pair in each batch is unpacked; The 
temporal feature data corresponding to the sequences that comprise the pair 
are stored in separate lists: one for the first sequences, one for the second
sequences and one for the target distances, depending on the pair's type - 
positive or negative)
7. A Siamese RNN model object is obtained.
8. The model is fed the training sequences and target distances.
9. **TODO**: The model is tested and validated.
10. **TODO**: The model's training, testing and validation metrics are visualized 
using TensorBoard.

---