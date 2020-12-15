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


doc = load_doc("boMonoData.txt")
sentences = to_sentences(doc)

for i in range(100000, 110000):
    print(sentences[i])