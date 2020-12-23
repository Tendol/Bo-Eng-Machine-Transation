
import string
import re
import sys
from unicodedata import normalize


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

# shortest and longest sentence lengths
def sentence_lengths(sentences):
	lengths = [len(s.split()) for s in sentences]
	return min(lengths), max(lengths)

# clean a list of lines
def clean_lines(lines):
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

# save a list of clean sentences to file
def save_clean_sentences(sentences, filename):
    with open(filename, 'w') as filehandle:
        filehandle.writelines("%s\n" % sentence for sentence in sentences)
    print('Saved: %s' % filename)


if __name__ == '__main__':
    if(len(sys.argv) < 3):
        filename = input("enter the file name: ")
        output = input("enter the output file name: ")
    else:
        filename = sys.argv[1]
        output = sys.argv[2]

    doc = load_doc(filename)
    sentences = to_sentences(doc)
    sentences = clean_lines(sentences)
    minlen, maxlen = sentence_lengths(sentences)
    print('English data: sentences=%d, min=%d, max=%d' % (len(sentences), minlen, maxlen))

    save_clean_sentences(sentences, output)
    # spot check
    for i in range(10):
        print(sentences[i])
