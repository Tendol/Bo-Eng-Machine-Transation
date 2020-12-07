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

def to_sentences(doc):
    return doc.strip().split('\n')

if __name__ == '__main__':
    doc = load_doc("train.bo")
    sentences = to_sentences(doc)
    for i in range(0, 1000):
        print(sentences[i])
    
