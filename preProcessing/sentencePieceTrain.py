import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file='m.model')
print(sp.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་'], out_type=str))