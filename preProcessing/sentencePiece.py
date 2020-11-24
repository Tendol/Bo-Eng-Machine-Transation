import sentencepiece as spm
import os
from tokenizers import SentencePieceBPETokenizer

spm.SentencePieceTrainer.train(
        input='../data/boTokenData.txt', 
        model_prefix='m3', 
        vocab_size=35000)

sp = spm.SentencePieceProcessor(model_file='m3.model')
print(sp.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་'], out_type=str))
