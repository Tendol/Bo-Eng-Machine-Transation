import os 
import sys 

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

def save_clean_sentences(lines, filename):
    with open(filename, 'w') as filehandle:
        filehandle.writelines("%s\n" % line for line in lines)
    print('Saved: %s' % filename)

def clean_lines(lines):
    cleaned = list()

    for line in lines:
        token, cnt = line.split("\t")
        cleaned.append(token + " " + cnt[1:])
    return cleaned


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
    save_clean_sentences(sentences, output)