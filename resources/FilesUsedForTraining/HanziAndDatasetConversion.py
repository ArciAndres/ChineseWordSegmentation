# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 18:18:11 2019

@author: AndresArciniegas
"""

from hanziconv import HanziConv

# AS (Traditional Chinese)
# CITYU (Traditional Chinese)
# MSR (Simplified Chinese)
# PKU (Simplified Chinese)

# The task is to convert all of them to simplified chinese. So, AS and CITYU must be processed

#%%
def reorder_elements_in_list_by_words(sentences, length=50): 
    """
    Reordering (reshaping) the words in the sentences allows to reduce the number of them, avoiding useless computations when padding.
    
    :param sentences: is a list of strings to be reordered by words
    :param length: size of each new sentence (no. of words).
    """
    sents_single = ' '.join(sentences) #The space is necessary to avoid involuntary merging of words
    sents_simp_split = sents_single.split(' ')
    sents_reordered = []
    num_of_elems = round(len(sents_simp_split)/length)
    for i in range(num_of_elems):
        newsent = sents_simp_split[i*length:(i+1)*length]
        sents_reordered.append(' '.join(newsent))
    print('\n[INFO] Number of sentences in original file: ', len(sentences))
    print('[INFO] Number of sentences in reordered file: ', len(sents_reordered))
    print("[INFO] Porcentage of length respect to input:", round(len(sents_reordered)/len(sentences)*100,2),'%')

    return sents_reordered    
    
def reorder_elements_in_list_by_chars(sentences, length=100):
    """
    Reordering (reshaping) the words in the sentences allows to reduce the number of them, avoiding useless computations when padding.
    
    :param sentences: is a list of strings to be reordered by words
    :param length: size of each new sentence (no. of words).
    """
    sents_single = ' '.join(sentences)
    sents_r = []
    spaces = -1
    p0 = 0; pos_buf = 0
    for pos,char in enumerate(sents_single):
        if char == " ":
            spaces += 1 #  # Spaces accumulator
            #print(sent[p0:pos], p0, pos, pos-p0,  maxlen + spaces, pos_buf)
            #if p0==0: 
            if pos-p0 > length + spaces:
                #print("taken!")
                # Takes since the range of characters until the previous space position (not the current), 
                # so the length is kept always below the 'length' parameter
                sents_r.append(sents_single[p0:pos_buf]) 
                p0 = pos_buf+1; 
                spaces=0;
            pos_buf = pos 
        if pos == len(sents_single)-1: #Include the last sentence, no matter how long it is
            sents_r.append(sents_single[p0:pos+1])    
    print('\n[INFO] Number of sentences in original file: ', len(sentences))
    print('[INFO] Number of sentences in reordered file: ', len(sents_r))
    print("[INFO] Porcentage of length respect to input:", round(len(sents_r)/len(sentences)*100,2),'%')
    #    lsentsr = [len(x.replace(' ','')) for x in sents_r]
#    print(lsentsr)

    return sents_r       
    
def save_into_file(lines, file_path):
    with open(file_path, 'w', encoding='utf8') as file:
        file.writelines('\n'.join(lines))

def to_simplified_chinese(sentences):
    return [HanziConv.toSimplified(s) for s in sentences]

def file_to_simplified_chinese(filePath,save=True,save_file=""):
    with open(file_path, encoding='utf8') as file:
        sents = file.read().splitlines()
    sents = [x.replace('  ',' ').replace('\u3000' ,' ') for x in sents]
    sents_simp = to_simplified_chinese(sents)    
    return sents, sents_simp

#%%

#%%
length = 30   

#%%
# ----------------------------------------------
################# ALL MERGED  ##################
# ----------------------------------------------
print("[INFO] Processing ALL Datasets merged (Test)")
file_path = "../dataset/icwb2-data/gold/ALL_test_gold_simp_reordered_shuf.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path,length)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
save_into_file(sents_reorder , "../dataset/icwb2-data/gold/as_test_gold_simp_reordered_30.utf8")
#%%
print("[INFO] Processing ALL Datasets merged (Train)")
file_path = "../dataset/icwb2-data/training/ALL_training_simp_reordered_shuf.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path,length)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
save_into_file(sents_reorder, "../dataset/icwb2-data/training/ALL_training_simp_reordered_shuf_30.utf8")
 
#%%
# ----------------------------------------------
###################### AS ######################
# ----------------------------------------------
print("[INFO] Processing AS Dataset (Gold)")
file_path = "../dataset/icwb2-data/gold/as_test_gold.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
save_into_file(sents_reorder , "../dataset/icwb2-data/gold/as_test_gold_simp_reordered.utf8")
#save_into_file(sents_simp , "../dataset/icwb2-data/gold/as_test_gold_simp.utf8")

print("[INFO] Processing AS Dataset (Training)")
file_path = "../dataset/icwb2-data/training/as_training.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
#save_into_file(sents_reorder , "../dataset/icwb2-data/training/as_training_simp_reordered.utf8")
save_into_file(sents_reorder , "../dataset/icwb2-data/training/as_training_simp_reordered.utf8")
#save_into_file(sents_simp , "../dataset/icwb2-data/training/as_training_simp.utf8")

#%%
# ----------------------------------------------
##################### CITYU ####################
# ----------------------------------------------
print("[INFO] Processing CITYU Dataset (Gold)")

file_path = "../dataset/icwb2-data/gold/cityu_test_gold.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
save_into_file(sents_reorder , "../dataset/icwb2-data/gold/cityu_test_gold_simp_reordered.utf8")

print("[INFO] Processing CITYU Dataset (Training)")

file_path = "../dataset/icwb2-data/training/cityu_training.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
save_into_file(sents_reorder , "../dataset/icwb2-data/training/cityu_training_simp_reordered.utf8")

#%%
# Reorder also the ones that were already in simplified chinese
# If processed by the simplified-chinese converted, they remain the same

# ----------------------------------------------
##################### MSR ######################
# ----------------------------------------------
print("[INFO] Processing MSR Dataset (Training)")

file_path = "../dataset/icwb2-data/training/msr_training.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
save_into_file(sents_reorder , "../dataset/icwb2-data/training/msr_training_simp_reordered.utf8")

print("[INFO] Processing MSR Dataset (Gold)")

file_path = "../dataset/icwb2-data/gold/msr_test_gold.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
save_into_file(sents_reorder , "../dataset/icwb2-data/gold/msr_test_gold_simp_reordered.utf8")

#%%
# ----------------------------------------------
##################### PKU ######################
# ----------------------------------------------
print("[INFO] Processing PKU Dataset (Training)")

file_path = "../dataset/icwb2-data/training/pku_training.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
save_into_file(sents_reorder , "../dataset/icwb2-data/training/pku_training_simp_reordered.utf8")

print("[INFO] Processing PKU Dataset (Gold)")

file_path = "../dataset/icwb2-data/gold/pku_test_gold.utf8"
sents, sents_simp = file_to_simplified_chinese(file_path)
sents_reorder = reorder_elements_in_list_by_chars(sents_simp, length)
save_into_file(sents_reorder , "../dataset/icwb2-data/gold/pku_test_gold_simp_reordered.utf8")


       