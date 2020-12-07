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

def seperate_into_two_files(lines):
    bo_append = []
    en_append = []
    for line in lines: 
        bo, en = line.split("\t")
        if bo and en: 
            bo_append.append(bo)
            en_append.append(en)
    
    return bo_append, en_append


if __name__ == '__main__':
    doc = load_doc("../data/bo-en.txt")
    sentences = to_sentences(doc)
    bo, en = seperate_into_two_files(sentences)


