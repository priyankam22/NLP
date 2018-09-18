#!/usr/bin/env python


import math
import numpy as np
import os
import pickle
import sys
import gzip
from datetime import datetime

# Initialize variables
input_seed = int(sys.argv[1])
mode = sys.argv[2]

# Define a 4 gram model
n = 4
seq_length = n-1
V = 65424
lamb = 1 #Laplace smoothing


# Function to load input file
def load_doc(filename):
    ''' This function takes a filename as input and returns the loaded file in a string object
    '''
    file = open(filename,'r', encoding='utf-8')
    text = file.read()
    file.close()
    return text

# Function to save a file
def sav_doc(lines, filename):
    ''' This function takes a list as input and saves the data in a file on the disk
    '''
    seq_data = '\n'.join(lines)
    file = open(filename, 'w', encoding='utf-8')
    file.write(seq_data)
    file.close()
    
# Tokenize - Create tokens of size n from data
def create_tokens(data, n):
    '''Creates tokens of size n from data
    '''
    sequences = list()
    for i in range(n-1,len(data)):
        seq = train_data[i-n+1:i+1]
        sequences.append(seq)
    
    return sequences

def seq_freq_dist(sequences, n):
    ''' Input: sequences - a list of n-grams
               n - length of each sequence
        Returns: A list of dictionaries
        This funtion extracts the counts for all kgrams from 1 to n and stores in a list of dictionaries in increasing order.
        1-gram is stored at position 0 and n-gram is stored in position n-1.
    '''
    gram_freq = list()
    for i in range(n):
        gram_freq.append(dict())

    for i in range(len(sequences)):
        for g in range(len(gram_freq)):     
            if sequences[i][0:g+1] not in gram_freq[g].keys():
                gram_freq[g][sequences[i][0:g+1]] = 1
            else:
                gram_freq[g][sequences[i][0:g+1]] += 1
                
    return gram_freq

def seq_prob_dist(sequences, n):
    ''' Input: sequences - a list of n-grams
           n - length of each sequence
        Returns: A list of dictionaries
        This funtion calculates the relative frequencies all kgrams from 1 to n and stores in a list of dictionaries in increasing order.
        1-gram is stored at position 0 and n-gram is stored in position n-1.
    '''
    gram_prob = list()
    for i in range(n):
        gram_prob.append(dict())
        
    #gram_freq = seq_freq_dist(sequences, n)

    for i in gram_freq[0].keys():
        gram_prob[0][i] = (gram_freq[0][i]+ lamb)/(sum(gram_freq[0].values()) + V*lamb)

    for g in range(1,len(gram_prob)):
        for i in gram_freq[g].keys():
            gram_prob[g][i] = (gram_freq[g][i] + lamb)/(gram_freq[g-1][i[0:g]] + V*lamb)
            
    return gram_prob


def query_probability(history, char):
    ''' Return the log base 2 probability of the input character given the history'''
    # Get the sequence to search
    if len(history) >= seq_length:
        query_seq = history[len(history)-seq_length:len(history)] + char
    else:
        query_seq = history + char
    
    # Extract the probability
    # Kgram is stored in gram_prob[k-1]
    order = len(query_seq)-1
    
    if query_seq in gram_prob[order].keys():
        return math.log(gram_prob[order][query_seq],2)
    elif order > 0:
        # Default to Laplace smoothing of unseen characters
        context_freq = 0
        if query_seq[0:order] in gram_freq[order-1]:
            context_freq = gram_freq[order-1][query_seq[0:order]]
        return math.log((1/(context_freq + V*lamb)),2)
    else:
        return math.log(1/(sum(gram_freq[0].values()) + V*lamb),2)
    
def generate_char(history):
    '''This function generates a chracter given a context'''
    if (len(history)) >= seq_length:
        ngram_hist = history[len(history)-seq_length:len(history)]
    else:
        ngram_hist = history
               
    candidate_ngrams = dict()
    
    for i in gram_prob[len(ngram_hist)].keys():
        if len(ngram_hist) == 0:
            candidate_ngrams = gram_prob[0]
        elif i[0:len(ngram_hist)] == ngram_hist:
            candidate_ngrams[i] = gram_prob[len(ngram_hist)][i]
            
    #Never seen this sequence before
    if len(candidate_ngrams) == 0:
        candidate_ngrams = gram_prob[0]
           
    np.random.seed(input_seed)
    pred_seq = np.random.choice(list(candidate_ngrams.keys()), p=np.array(list(candidate_ngrams.values()))/sum(candidate_ngrams.values()))   
    pred_y = pred_seq[-1]
    pred_prob = math.log(candidate_ngrams[pred_seq],2)

    return (pred_y, pred_prob)


def run_lang_model(user_input):
    ''' This function gets user input, interprets the characters in the input and prints appropriate output.'''

    # Initialize variables
    curr_hist = ''
    i = 0

    # Keep processing the user input characters till you encounter a 'x' or end of input
    while i < len(user_input):
        # Observerthe next character and add to history
        if user_input[i] == 'o':
            # Check whether user inputed next character else through error
            if i+1 < len(user_input):
                # if user_input[i+1] is stop symbol then clear history
                c = user_input[i+1]
                # Check for the stop symbol
                if c.encode('utf-8') == b'\x03':
                    #End of passage.Clear history
                    curr_hist = ''
                    print('//Cleared the history!')
                else:
                    estimated_prob = query_probability(curr_hist, c)
                    estimated_char, prob = generate_char(curr_hist)
                    print('//Estimated probability is %f. Character added to history!' % (estimated_prob))
                    curr_hist = curr_hist + c
                i+=2

        # Query the input character and return the log 2 probability
        elif user_input[i] == 'q':
            # Check whether user inputed next character else through error
            if i+1 < len(user_input):             
                c = user_input[i+1]                                                                                                      
                estimated_prob = query_probability(curr_hist, c)
                print(estimated_prob)
                i+=2

        # Generate the next character and append to history
        elif user_input[i] == 'g':
            generated_char, prob = generate_char(curr_hist)
            curr_hist = curr_hist + generated_char
            print("%s// generated with probability %f!" % (generated_char, prob))
            i+=1

        # Exit
        elif user_input[i] == 'x':
            break
        
        else: #Some random input.Ignore and proceed.
            i+=1
               
def gen_seq(seed, nchar):
    '''Creates a sequence of nchar using seed'''
    history = seed
    for i in range(nchar):
            c = generate_char(history)[0]
            history = history + c
    print(history)
    
def calc_perplexity(test_file):
    '''Calculates the perplexity of the language model for a given test set'''

    test_data = load_doc(test_file)
    test_seq = test_data.split('\n')
    
    total_chars = sum([len(seq) for seq in test_seq])
   
    l = (1/total_chars)*sum([query_probability(seq[:-1], seq[-1]) for seq in test_seq])
    perplexity = 2**(-l)
    return(perplexity)    
            
#################################################################################################################
#                                               MAIN
#################################################################################################################

if __name__ == '__main__':
    
    # Train the model on training data
    if mode == 'TRAIN': 
   
        # Initialize variables
        train_data = ''
        i = 0
        training_files_limit = 20
        train_dir = 'train_data'

        # Loop through files under directory and append to train_data
        for file in os.listdir(train_dir):
            if i > training_files_limit:
                break
            print("[%s] Loading file: %s" % (str(datetime.now()), file)) 
            file_data = load_doc(train_dir + '/' + file)
            train_data += file_data
            print("Length of training data:", len(train_data))
            i+=1
	
        # Create tokens of size n
        print("[%s] Creating tokens from training data" % (str(datetime.now())))
        sequences = create_tokens(train_data, n)
        print("Number of sequences: %d" % len(sequences))
    
        # Train the model and create the frequency and probability distribution of all kgrams
        print("[%s] Generating frequency distribution" % (str(datetime.now())))
        gram_freq = seq_freq_dist(sequences, n)
        gram_prob = seq_prob_dist(sequences, n)

        # Serialize the lanuguage model and store on disk
        print("[%s] Pickling the language model dictionaries" % (str(datetime.now())))

        pickle.dump(gram_freq, gzip.open('bin/gram_freq.p','wb'))
        pickle.dump(gram_prob, gzip.open('bin/gram_prob.p','wb'))
        
        print("Length of all kgram dictionaries:")
        for i in range(n):
            print(i+1, len(gram_prob[i]))
	
        # Check the sum of probabilities over unigrams.Should be 1.
        print("Sum of Unigram probabilities:", sum(gram_prob[0].values()) + (V - len(gram_freq[0]))/ (sum(gram_freq[0].values()) + V*lamb))

    # Test the perplexity of test data
    elif mode == 'TEST':
        
        test_dir = 'test_data'

        # Unpickle the language model
        print("[%s] Unpickling the language model..." % (str(datetime.now())))
        gram_freq = pickle.load(gzip.open('bin/gram_freq.p','rb'), encoding='utf-8')
        gram_prob = pickle.load(gzip.open('bin/gram_prob.p','rb'), encoding='utf-8')

        for file in os.listdir(test_dir):
            print("Perplexity for file %s: %f" % (file, calc_perplexity(test_dir + '/' + file)))

    # Run the model to take charaacter level input and output probabilities
    elif mode == 'RUN':

        user_input = sys.stdin.readlines()

        gram_freq = pickle.load(gzip.open('bin/gram_freq.p','rb'), encoding='utf-8')
        gram_prob = pickle.load(gzip.open('bin/gram_prob.p','rb'), encoding='utf-8')

        # Load the frequency dictionaries
        run_lang_model(''.join(user_input))