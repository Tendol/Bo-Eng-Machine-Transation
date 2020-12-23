import sys
import re


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

def to_sentence(doc):
    return re.split('།', doc)

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
    sentences = to_sentence(doc)
    save_clean_sentences(sentences, output)

    # for i in range(10):
    #     print(sentences[i])

