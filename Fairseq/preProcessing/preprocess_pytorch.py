#!/usr/bin/env python
# coding: utf-8

# # STEP 1: Cleaning the data

# In[1]:


import sentencepiece as spm
import os
from tokenizers import SentencePieceBPETokenizer
import string
import re
import sys
from unicodedata import normalize


# **Load data**

# In[2]:


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# **Split data into sentences**

# In[3]:


# split a loaded document into sentences
def to_sentences(doc):
    return doc.strip().split('\n')


# **Clean the data**
# 
# *Tibetan*
# 
# * Remove shey "།" (end of the line punctuation) 
# * Remove any numbers 
# * Remove any charcter that isn't Tibetan alphabet example  ༼༄༸ or non-Tibetan characters 
# * There is no upper or lower case in Tibetan so no need to normalize 
# * Make sure that there is no space between words (seperated by tsek "་")
# * Remove words that contain non Tibetan alphabets example: བས྄ྟན༼་
# 

# **Create Tibetan unicode array for easy access**
# 
# *Tibetan Unicode Array:*
# 
# * Tibetan Vowel : (ུ): 3956 ( ི) : 3954 ( ེ) : 3962  ( ོ) : 3964
# * Consonants : 3904 - 3946 
# * Subjoined Consonants : 3984 - 4028 
# * Numbers : 3872 - 3881 
# * punctuation: Tsek (་) : 3851 ; shey (།) : 3853 
# 

# In[4]:


tib_unicode = []
# adding consonants 
for i in range(3904, 3947):
    tib_unicode.append(i)
# adding subjoined consonants 
for i in range(3984, 4029):
    tib_unicode.append(i)
# adding numbers 
for i in range(3872, 3881):
    tib_unicode.append(i)
# adding punctuations 
tib_unicode.append(3851)
tib_unicode.append(3853)
# adding vowels
tib_unicode.append(3956)
tib_unicode.append(3954)
tib_unicode.append(3962)
tib_unicode.append(3964)

tib_str = "" # Contains all Tibetan alphabets, numbers, and special characters.
tib_alph_str = "" # Contains only Tibetan alphabets
tib_num = "" # Contains only Tibetan numbers 

for i in range(3904, 3947):
    tib_str += chr(i)
    tib_alph_str += chr(i)
# adding subjoined consonants 
for i in range(3984, 4029):
    tib_str += chr(i)
    tib_alph_str += chr(i)
# adding numbers 
for i in range(3872, 3881):
    tib_str += chr(i)
    tib_num += chr(i)
# adding punctuations 
tib_str += chr(3851)
tib_str += chr(3853)
# adding vowels
tib_str += chr(3956)
tib_str += chr(3954)
tib_str += chr(3962)
tib_str += chr(3964)

tib_alph_str += chr(3956)
tib_alph_str += chr(3954)
tib_alph_str += chr(3962)
tib_alph_str += chr(3964)


# In[5]:



# checks if the word contains non Tibetan alphabets
def isalpha(word):
    for w in word:
        if w not in tib_alph_str:
            return False
    return True

# clean a list of lines (Tibetan)
def clean_lines_bo(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(tib_str))

    for line in lines:
        
        #  remove strings between [] that was not translated into English (this is for this specific data)
        line = re.sub("[\(\[].*?[\)\]]", "", line)

        # tokenize on tsek and shek
        line = re.split("་|།", line)

        # remove non-printable chars form each token
        line = [re_print.sub('', w) for w in line]

        # remove tokens with numbers in them
        line = [word for word in line if isalpha(word)]

        line = '་'.join(line)

        # remove any empty line or white spaces at the end of the line
        if line.rstrip():
            
            # store as string (removed shek)
            cleaned.append(line)

    return cleaned


# *English*
# 
# * Remove punctuation 
# * Remove any numbers 
# * Remove any character that is not an English alphabet 
# * Normalize everything to lower letters 
# * Remove words that contain non English alphabets 

# In[6]:


# clean a list of lines (English)
def clean_lines_en(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(string.printable))

    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)

    for line in lines:
        # normalize unicode characters
        line = normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('UTF-8')

        # tokenize on white space
        line = line.split()

        # convert to lower case
        line = [word.lower() for word in line]

        # remove punctuation from each token
        line = [word.translate(table) for word in line]

        # remove non-printable chars form each token
        line = [re_print.sub('', w) for w in line]
        
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
    
        # # store as string
        if not line == []:    
            cleaned.append(' '.join(line))

    return cleaned


# **Save the clean sentences to a file** 

# In[7]:


# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
    with open(filename, 'w') as filehandle:
        filehandle.writelines("%s\n" % sentence for sentence in sentences)

    print('Saved: %s' % filename)


# In[9]:


def save_clean_sentences_binary(sentences, filename):
    with open(filename, 'wb') as filehandle:
        filehandle.writelines("%s\n" % sentence for sentence in sentences)

    print('Saved: %s' % filename)


# **Get information on the shortest and longest sentences in the two data**

# *English*

# In[10]:


# shortest and longest sentence lengths
def sentence_lengths_en(sentences):
	lengths = [len(s.split()) for s in sentences]
	return min(lengths), max(lengths)


# *Tibetan*

# In[11]:


def sentence_lengths_bo(sentences):
	lengths = [len(s.split("་")) for s in sentences]
	return min(lengths), max(lengths)


# In[12]:


if __name__ == '__main__':
    bo_text = "../data/bo.txt"
    en_text = "../data/en.txt"
    clean_bo = "../data/cleanBo.txt"
    clean_en = "../data/cleanEn.txt"

    # Tibetan
    doc = load_doc(bo_text)
    sentences = to_sentences(doc)
    sentences = clean_lines_bo(sentences)
    minlen, maxlen = sentence_lengths_bo(sentences)
    print('Tibetan data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))
    save_clean_sentences(sentences, clean_bo) 
    # spot check
    for i in range(5):
        print(sentences[i])
    print()
    # English
    doc = load_doc(en_text)
    sentences = to_sentences(doc)
    sentences = clean_lines_en(sentences)
    minlen, maxlen = sentence_lengths_en(sentences)
    print('English data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))

    save_clean_sentences(sentences, clean_en)
    
    # spot check
    for i in range(5):
        print(sentences[i])


# # Step 2 : Tokenize the training data 

# **Generate a Tibetan tokenizer using sentencepiece and monolingual Tibetan data**
# 
# Using 32000 vocabulary size .. just because (also it was used by the author who wrote sentencepiece)

# **Model Training** 

# *Tibetan*

# In[ ]:


spm.SentencePieceTrainer.train(
        input='../data/boTokenData.txt', 
        model_prefix='bo', 
        vocab_size=32000)
# sp = spm.SentencePieceProcessor(model_file='train.model')
# print(sp.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་'], out_type=str))


# *English*

# In[7]:


spm.SentencePieceTrainer.train(
        input='../data/enTokenData.txt', 
        model_prefix='en', 
        vocab_size=25000)


# **Segmentation**

# In[8]:


sp = spm.SentencePieceProcessor(model_file='bo.model')
print(sp.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་'], out_type=str))
print(sp.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་', 'བཀ྄ྲ་ཤིས་བདེ་ལེགས།'], out_type=int))
print(sp.decode([4149, 306, 6, 245, 4660, 748]))
print(sp.decode(['▁ངའི་', 'མིང་', 'ལ་', 'བསྟན་', 'སྒྲོལ་མ་', 'ཟེར་']))
sp.get_piece_size()


# In[9]:


sp = spm.SentencePieceProcessor(model_file='en.model')
print(sp.encode(["My name isn't Tenzin Dolma Gyalpo"], out_type=str))
print(sp.encode(['My name is Tenzin Dolma Gyalpo', 'Hello'], out_type=int))
print(sp.decode([[8803, 180, 12, 5519, 15171, 17894], [887, 21491]]))
sp.get_piece_size()


# **Tokenizing training data**

# *Tibetan*

# In[16]:


sp = spm.SentencePieceProcessor(model_file='bo.model')
doc = load_doc("../data/train.bo")
sentences = to_sentences(doc)
bo_token = sp.encode(sentences, out_type=str)
save_clean_sentences_binary(bo_token, "../data-bin/data.tokenized.bo-en/train.bo-en.bo.bin")
# spot check
for i in range(5):
    print(bo_token[i])


# *English*

# In[12]:


sp = spm.SentencePieceProcessor(model_file='en.model')
doc = load_doc("../data/train.en")
sentences = to_sentences(doc)
en_token = sp.encode(sentences, out_type=str)
save_clean_sentences_binary(en_token, "../data-bin/data.tokenized.bo-en/train.bo-en.en.bin")
# spot check
for i in range(5):
    print(en_token[i])


# In[ ]:



