# Keystroke dynamics biometrics using Recurrent Neural Networks

- To-be sub-repository of Master's Thesis project, 'TypeFace'

### TODO
- [ ] Refactor tests in `pairs/tests/test_pairing.py` & `pairs/pairing.py:make_pairs()` until all tests pass  
- [ ] Analyze & decide on a method to store the keystroke sequence pairs created on disk  
	- [ ] Option 1: Create 1 file for each pair (either .csv or .pickle)  
					+ Easier to store pairs since there is one pair / file  
		 			- Lower performance when making the batches since there will be many calls to file reading function  
		 			- Lots of resulting files
	- [ ] Option 2: Concatenate all pairs of a chunk (positive & negative ones separately) into one DataFrame & write it to HDF5 file
					+ Less resulting files (for a dataset of 4,000 files: 40 chunks, 2 dfs (positive & negative) / chunk => 80 HDF5 files)  
