# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 17:31:12 2019

@author: AndresArciniegas
"""

# Methods to perform the pre processing of the text for word segmentation in Chinese.

import tensorflow.keras as K
from tensorflow.keras.utils import to_categorical
import RebuildHelper
import numpy as np

class ChinesePreprocess:
    """ Contains the used methods to convert simplified chinese text to N-Grams with padding,
    which is the inputs of the training model
    
    Usage::
    
        >>> import ChinesePreprocessPredict as CP
        >>> data = CP(FILE_PATH,)
    
    :param train_data: A list of tuples of the form ``(color, label)``.
    :rtype: A :class:`Classifier <Classifier>`
    """
    def __init__(self, FILE_PATH , vocabulary={}, maxlength=50, verbose=True):
        if verbose: print("\n[MAIN] Data preprocessing  starting...")
        
        self.sents_orig = read_sentences(FILE_PATH, space_join=False,verbose=verbose)
        
        self.sents_ready, self.sents_cut, self.cuts = RebuildHelper.build_sentences_to_predict(self.sents_orig, maxlength)
        
        if verbose: print("\n[MAIN] Processing sentences into unigrams and bigrams with padding...")
                
        unigrams_num, bigrams_num = self.get_formated_ngrams(self.sents_ready, vocabulary, verbose)
        
        self.unigrams_pad = apply_padding(unigrams_num, maxlength, verbose)
        self.bigrams_pad  = apply_padding(bigrams_num, maxlength, verbose)
      
    def get_formated_ngrams(self, sents_nospaces, vocabulary = {}, verbose=True): 
        """ Makes the conversion of the provided sentences into unigrams and bigrams with padding """
        # Generate vocabulary from ngrams
        self.unigrams = to_ngrams(sents_nospaces, 1, verbose)
        self.bigrams = to_ngrams(sents_nospaces, 2, verbose)
        merged_ngrams = merge_list_content([self.unigrams, self.bigrams])
        
        # Use preset vocabulary in case of having it
        # If not defined, a vocabulary is build
        if vocabulary == {}: 
            self.vocab = generate_vocabulary(merged_ngrams, verbose)    
        else:
            print("[INFO] Using preset vocabulary. No. of elements: " , str(len(vocabulary)))
            self.vocab = vocabulary
        self.vocab_inv = {value:key for key,value in self.vocab.items()}
        unigrams_num = apply_vocabulary(self.vocab, self.vocab_inv, self.unigrams, verbose)
        bigrams_num = apply_vocabulary(self.vocab, self.vocab_inv, self.bigrams, verbose)
        
        return unigrams_num, bigrams_num

## Auxiliary methods:
## These are defined in this file, to format the preprocessed object while building it.
#--------------------------------------------------------------------------------------
        
#----------------------------------------------------
################### File Reading ####################
#----------------------------------------------------

def read_sentences(file_path, space_join=False , num_samples=0, verbose=True):
    
    """ Read the sentences contained in a text file, separated by newline character (\n)
    
    :param file_path: path of the file to read.
    :param num_samples: Define a number of elements to return from the total. If not defined, all the elements are returned.
    :space_join: Used in training stage, to return also the sentences without spaces.
        
    :return: sents, sents_nospace 
    :rtype: list, list
    """
    verboseprint = print if verbose else lambda *a, **k: None

    verboseprint("\n[INFO] Reading data file...")
    with open(file_path, encoding='utf8') as file:
        sents = file.read().splitlines()
        
        verboseprint("[INFO] Read file: %s" % file_path)
        verboseprint("[INFO] Total number of sentences: ",len(sents))
        verboseprint("[INFO] Sample of the file: \n", sents[1:3])
        
    if num_samples != 0:
        if num_samples > len(sents):
            verboseprint('[Warning] The given number of samples is greater than the total.\n All samples will be included.\n')
        else:
            sents = sents[0:num_samples] 
    
    # Used for training. Removes the spaces from the sentences.
    if space_join: 
        sents_nospace = [sent.replace(' ' ,'').replace('\u3000' ,'') for sent in sents] # Remove spaces
        return sents, sents_nospace
    
    return sents


#-----------------------------------------------
################### Padding ####################
#-----------------------------------------------

def padding_labels(sentences_bies, MAXLENGTH = 50, verbose=True):
    ''' Pads the sentences to the provided max_length'''' 
    labels_pad = apply_padding(sentences_bies, MAXLENGTH, verbose) 
    # Converts to One-hot encoding
    if verbose: print("\n[INFO] Converting to One-hot encoding representation")
    labels_one_hot = to_categorical(labels_pad) 
    if verbose: 
        print("[INFO] Sample from labels: ")
        print(sentences_bies[0][0:10])
        print(labels_one_hot[0][0:10])
    return labels_one_hot

def apply_padding(elements, maxlength=50, verbose=True):
    """ Applies padding to a provided maximum length. 
    By default the value of the padding is zero, and the truncation and padding is done at the end ('post')
    
    :Example: 
        >>> elements = [[3, 3, 1, 1, 3, 3],
                        [0, 1, 1],
                        [3, 3, 1, 1, 3, 3, 4, 5, 1, 3, 4, 1, 3, 7, 1]]
        >>> apply_padding(elements, maxlength=10)
        Out: array([[3, 3, 1, 1, 3, 3, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [3, 3, 1, 1, 3, 3, 4, 5, 1, 3]])
    """
    return K.preprocessing.sequence.pad_sequences(elements, padding='post', truncating='post', maxlen=maxlength)
        
#-----------------------------------------------
################## Vocabulary  #################
#-----------------------------------------------
    
def generate_vocabulary(sentence, verbose=True):
    """ Extracts the vocabulary of different words present in the input sentence. 
    
    :param sentence: A single string of text (not a list) containing the words to become a dictionary. 
    :return: Dictionary with unique words in the sentence associated to an integer index. Includes as first element the expression
    for unknown words {1: "UNK"}
    
    Example: {1:"UNK", 2:"word2", 3:"word3"}
    """
    if verbose: print("\n[INFO] Generating vocabulary from sentence...")
    unique_words = set(sentence)
    vocab = {}
    vocab = {w:i+2 for (i,w) in enumerate(unique_words)} # +2 since no zero is desired,and 1 is for <UNK>
    vocab["UNK"]=1 
    if verbose:
        print("[INFO] Vocabulary generated from sentence successfully. \n Number of elements: %s" % str(len(vocab)))
        print("[INFO]Sample of vocabulary: \n")
        c = 0
        print('UNK',":", str(vocab['UNK']))
        for k,v in vocab.items():
            print(str(k),":", str(v))
            c += 1
            if c> 3: break
    return vocab
        

def apply_vocabulary(vocabulary, vocabulary_inv, sentences, verbose = True, check_element = 1):
    """ Takes a list of sentences and translate them into their equivalence in the provided vocabulary """
    sentences_vocab = []
    for sent in sentences:
        word_vocab = []
        for word in sent:
            if word in vocabulary:
                word_vocab.append(vocabulary[word])
            else:
                word_vocab.append(vocabulary['UNK']) # The term is unknown. This is usually 1
        sentences_vocab.append(word_vocab)                
    if verbose:
        print("[INFO] Conversion to translated sentences with vocabulary complete.")
        print("--- Check: ---")
        print("Element: ", str(check_element))
        #print("Returned element: \n", sentences_vocab[check_element])
        #print("Converted using inverse vocabulary:\n",[vocabulary_inv[char] for char in sentences_vocab[check_element]])
        #print("Original element:\n",sentences[check_element], '\n\n')
    return sentences_vocab

#-----------------------------------------------------------
################### NGram - Conversion #####################
#-----------------------------------------------------------

def to_ngrams(sentences, n=1, verbose=True):
    """ Returns a list of sentences into ngram-splitted sentences """
    
    ngrams_sentences = []
    
    if n==1: # Unigrams
        if verbose: print("[INFO] Converting to unigrams...")
        ngrams_sentences = [list(sent) for sent in sentences]

    if n==2: # Bigrams
        if verbose: print("[INFO] Converting to bigrams...")
        for sentence in sentences:
            bigrams = []
            for i in range(len(sentence)-1): # -1 to avoid the las element alone
                #i = i+1 # Avoid the first element
                bigrams.append(sentence[i:i+2]) # Takes two elements
            ngrams_sentences.append(bigrams)
    
    return ngrams_sentences
        
#%%
#----------------------------------------------------
################### BIES format #####################
#----------------------------------------------------
def num2BIES(sent):
  #''.join(num2BIES(['0', '2', '0', '2', '3', '3', '0', '2', '0', '2', '3', '0', '2']))
  #Out: BEBESSBEBESBE
  tags = {'0':'B',
          '1':'I',
          '2':'E',
          '3':'S'
         }
  bies = [tags[s] for s in sent]
  return bies


           
#%%
#----------------------------------------------------
################ Formating prediction ###############
#----------------------------------------------------
def format_prediction(ypred, sents_input):
    ypredf = [] # Predicted output (formatted)
    for y in ypred:
        ypredf.append([str(np.argmax(i)) for i in y])
    
    # Convert to BIES letter format, and remove zero-padding applied at the beggining
    ybies = [''.join(num2BIES(y)[:len(sents_input[i])]) for i, y in enumerate(ypredf)]
    return ybies
        
def merge_list_content(lists):
    """ From a list of lists, returns a single merged list with the content of all the input lists 
    
    """
    list_concat = []
    for l in lists:
        list_concat = list_concat + l
    merged_content = []
    [merged_content.extend(l) for l in list_concat]         
    return merged_content
