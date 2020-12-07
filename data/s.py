

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

def cal_longest_sentence(lines):
    longest = 0
    for line in lines:
        a = line.split("་")
        if longest < len(a):
            longest = len(a)
    return longest

if __name__ == '__main__':
    doc = load_doc("train.en")
    sentences = to_sentences(doc)
    l = cal_longest_sentence(sentences)
    print(l) 


