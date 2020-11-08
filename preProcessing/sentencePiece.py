import sentencepiece as spm
import os
from tokenizers import SentencePieceBPETokenizer

spm.SentencePieceTrainer.train(
        input='../data/boMonoData.txt', 
        model_prefix='m', 
        vocab_size=32000)

sp = spm.SentencePieceProcessor(model_file='m.model')
print(sp.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་'], out_type=str))
