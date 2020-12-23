
import string
import re
import sys
from unicodedata import normalize
import os.path


# Tibetan Unicode Array: 

# Tibetan Vowel : (ུ): 3956 ( ི) : 3954 ( ེ) : 3962  ( ོ) : 3964
# Consonants : 3904 - 3946 
# Subjoined Consonants : 3984 - 4028 
# Numbers : 3872 - 3881 
# punctuation: Tsek (་) : 3851 ; shey (།) : 3853 
    
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

tib_str = ""
tib_alph_str = ""
tib_num = ""

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

# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

# split a loaded document into sentences
def to_sentences(doc):
    return doc.strip().split('\n')
        
def sentence_lengths(sentences):
	lengths = [len(s.split("་")) for s in sentences]
	return min(lengths), max(lengths)
    
def isalpha(word):
    for w in word:
        if w not in tib_alph_str:
            return False
    return True

# clean a list of lines
def clean_lines(lines):
    cleaned = list()
    # prepare regex for char filtering
    re_print = re.compile('[^%s]' % re.escape(tib_str))
    # # prepare translation table for removing punctuation

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

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
    with open(filename, 'a') as filehandle:
        filehandle.writelines("%s\n" % sentence for sentence in sentences)

    print('Saved: %s' % filename)


if __name__ == '__main__':
    if(len(sys.argv) < 2):
        filename = input("enter the file name: ")
        output = input("enter the output file name: ")
    else:
        filename = sys.argv[1]
        output = sys.argv[2]


    doc = load_doc(filename)
    sentences = to_sentences(doc)
    sentences = clean_lines(sentences)
    minlen, maxlen = sentence_lengths(sentences)
    print('Tibetan data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))

    save_clean_sentences(sentences, output)
    # spot check
    for i in range(1):
        print(sentences[i])
