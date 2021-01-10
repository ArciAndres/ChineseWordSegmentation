# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:28:25 2019

@author: AndresArciniegas
"""

# For splitting and rebulding sentence
#%%

import re
import math
#%%

#with open('../sample_files/pku_test.utf8', 'r', encoding='utf-8') as file:
#    sents_orig = file.readlines()
#sents_orig = [x.split('\n')[0] for x in sents_orig] # Remove the new line character

#=============================================
#-------- Build sentences to input------------
#=============================================

def build_sentences_to_predict(sents, maxlength=50):
    """
        Reorganizes a list of sentences according to a provided maximum length, 
        and splitting by known single chars. 
        
        Parameters:
            sents: list
                List of sentences with no spaces in between
            maxlength: int
                Maximum length allowed per sentence. If longer, it gets splitted
    """    
    sents_cut, cuts_in_sents = cut_sentences(sents)
    # Each sentence is splitted in parts with a maximum length
    sents_cut_max = [get_sent_parts(x, maxlength) for x in sents_cut]
    
    sents_cut_max_ready = []
    for sent in sents_cut_max:
        for sent_part in sent:
            sents_cut_max_ready.append(sent_part)
    
    return sents_cut_max_ready, sents_cut_max, cuts_in_sents

def cut_sentences(sents_orig):
    """Cuts the sentences of string in a list into kown single chars
        Returns
        -------
        sents_cut: list of divided sentences
        cuts_in_sents: position of the cuts per sentence, to rebuild later.
    """
    
    # Define a splitter with the characters that we know for sure they are single chars, labeled as 'S'
    cutChar = re.compile(u'[。，、？！\.\?,!；：”“]')
    sents_cut = []
    cuts_in_sents = [] # Saves the position of the cuts, to rebuild later
    is_cut=False
    for sent in sents_orig:
        cuts = []
        j = 0
        for i in cutChar.finditer(sent):
            cuts.append(i.start()) # Appends the position of the cut
            is_cut=True
            sents_cut.append(sent[j:i.start()+1]) # Append the sentence until the charm including it
            j = i.end()
        if is_cut:
            if j <= len(sent)-1:
                sents_cut.append(sent[j:len(sent)])
                cuts.append(j)
        if not is_cut:
            sents_cut.append(sent)
        is_cut = False
        cuts_in_sents.append(cuts)
    return sents_cut, cuts_in_sents

def get_sent_parts(sent, maxlength):
    # Check max length
    sent_parts = []
    j = 0
    if len(sent) > maxlength:
        for i in range(math.ceil(len(sent)/maxlength)): # number of sentences longer than maxlen
            sent_parts.append(sent[j:(i+1)*maxlength])
            j = (i+1)*maxlength
    else: sent_parts.append(sent) # It has to be inside a list
    return sent_parts
#%%

#=============================================
#----------- Rebuild sentences ---------------
#=============================================
    
def rebuild_sentences_from_bies(ybies, sents_cut_max, cuts_in_sents):  
    number_of_parts = [len(x) for x in sents_cut_max]
    sents_rebuild = []
    accum = 0
    for v in number_of_parts:
        sents_rebuild.append(''.join(ybies[accum:accum+v]))
        accum += v
    result = rebuild_sentences(sents_rebuild, cuts_in_sents)      
    return result

# Rebuild the sentences from bies.
def rebuild_sentences(sents_to_rebuild, cuts_in_sents):
    sents_rebuilt = []
    accum=0
    cuts_per_line = [len(x) for x in cuts_in_sents]
    for i, cuts_num in enumerate(cuts_per_line):
        sent_reb = []
        if cuts_num == 0:
            sent_reb = sents_to_rebuild[accum]
            accum += 1
        else:
            for j in range(cuts_num):            
                sent_reb.append(sents_to_rebuild[accum+j])
            sent_reb = "".join(sent_reb)
            accum += cuts_num
        sents_rebuilt.append(sent_reb)
        
    return sents_rebuilt


#%%

