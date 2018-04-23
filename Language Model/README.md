**N-Gram Implementation of Character Language Model**

This is an implementation of a probabilistic language model at the character level. A probabilistic language model maps sequences of symbols to discrete probabilities based on the context. The vocabulary is the set of all valid UTF-8 encodings of Unicode 10 in the basic multilingual plane, thus V = 65,424. This is an N-gram model where the probability of a character depends on the previous n-1 characters. It is based on Markov chain as seen below. In this case, w refers to a single Unicode character.
                                
**PROCESS:**

The main python file is ngram_language_model.py which takes as input a random seed and a mode. Random seed is used to sample from the probability distribution of candidate ngrams when generating new characters. The mode can have three values – TRAIN, TEST and RUN. TRAIN mode is used to train the model using files stored in the train_data folder. TEST mode can be used to calculate perplexity score on test files stored in test_data folder. The TRAIN mode trains the model on the training files and generates the absolute frequency and probability distribution of every k-gram encountered. The probability is calculated using Maximum Liklihood Estimation.
Two lists of dictionaries are used to store the frequencies and probabilities of the ngrams. Each list has one dictionary for each of the k-grams where k = 1 to n. These dictionaries are serialized, zipped and stored in the bin folder to be used for querying the model. The frequencies are stored to calculate the probabilities of unseen sequences on the fly by referring to the (n-1)th gram which represents the context. The RUN mode is the mode where we can pass input from stdin as a sequence of characters starting with one of the 3 modes – o for observe, q for query and g for generate. The python script is wrapped in a Unix bash script which takes random seed as the only input parameter.

**SMOOTHING:**
Since the language model needs to be able to generate text and predict probability of characters from any natural language, the training data plays a very important role. Exposing the model to all 65424 Unicode characters which covers hundreds of natural languages is a challenging task. It is very likely that the model has not seen many sequences and will give a probability of zero for valid sequences from languages it has not been exposed to.
To handle overfitting as seen above, I have introduced bias in the model using Laplace smoothing. We assume that each sequence was encountered at least once in the training set.  The below formula is used to calculate the adjusted probabilities where alpha is set to 1. This method ensures that perplexity scores do not shoot up for unseen sequences. 
 
**TRAINING:**
The model is trained on a mixture of different languages like English, Portugese, Spanish, Dutch, Finnish, etc. 
