import sentencepiece as spm
import pandas as pd
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    T5ForConditionalGeneration, 
    T5Config,
    AdamW,
    get_cosine_with_hard_restarts_schedule_with_warmup
)
import time
from datetime import datetime
import math

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print(f'device = {device}')

srcDataPath = '../data/train.bo'
tgtDataPath = '../data/train.en'

srcTokenizerPath = '../preProcessing/bo.model'
tgtTokenizerPath = '../preProcessing/en.model'

sampleOutPath = './T5_sample_results.txt'



## Load data

srcFile = open(srcDataPath, 'r', encoding = 'utf-8')
tgtFile = open(tgtDataPath, 'r', encoding = 'utf-8')

dataMatrix = []

while True: 
    srcLine = srcFile.readline().strip()
    tgtLine = tgtFile.readline().strip()
    if not srcLine or not tgtLine: 
        break 
    dataMatrix.append([srcLine, tgtLine])
  
# Create pandas dataframe for showing examples in jupyter notebook
df = pd.DataFrame(dataMatrix, columns = ['src', 'tgt'])

# Store source and target texts as lists 
srcTextsAll = df['src'].tolist()
tgtTextsAll = df['tgt'].tolist()


## Tokenizers 
srcTokenizer = spm.SentencePieceProcessor(model_file=srcTokenizerPath)
tgtTokenizer = spm.SentencePieceProcessor(model_file=tgtTokenizerPath)
tgt_eos_id = tgtTokenizer.piece_to_id('</s>')
tgt_pad_id = tgtTokenizer.piece_to_id('<pad>')


## Load the trained model with the lowest validation loss 
print('Loading model...')
state_dict = torch.load('T5_checkpoint_best_epoch=44.pt', map_location = device)
T5model = T5ForConditionalGeneration.from_pretrained(
    't5-small', 
    return_dict = True, 
    eos_token_id = tgt_eos_id, 
    pad_token_id = tgt_pad_id, 
    decoder_start_token_id = tgt_pad_id,   # If I don't add this line, then all predictions start with <unk>
    dropout_rate = 0.2, 
    max_length = 100, 
    state_dict = state_dict
).to(device)
print('Model loading is complete')

## Function for generating translation 
def generate_translation(model, src_text): 
    model.eval()
    
    src_ids = srcTokenizer.encode(src_text)
    src_ids = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    outs = model.generate(
        src_ids, 
        num_beams = 8, 
        repetition_penalty = 2.5, 
        length_penalty = 0.6, 
        early_stopping = True, 
    )
    
    # If any token beyond vocab size, make it pad
    outs = outs.masked_fill(outs >= tgtTokenizer.get_piece_size(), tgt_pad_id)
    ids = outs[0].tolist()
    
    pred_text = tgtTokenizer.decode(ids)
    return pred_text


# Pick selected examples, generate translation, and compare 
selected = [0, 1, 2, 13, 24, 41]
sample_writer = open(sampleOutPath, 'w', encoding='utf-8')
print('Generating translations for selected sentences...')
for idx in selected: 
    translated_sentence = generate_translation(T5model, srcTextsAll[idx])
    sample_writer.write(f'Origianl source text: {srcTextsAll[idx]}\n\n')
    sample_writer.write(f'Original target text: {tgtTextsAll[idx]}\n\n')
    sample_writer.write(f'Predicted target text: {translated_sentence}\n\n')
    sample_writer.write('-' * 50 + '\n\n')
print('Generation complete. See', sampleOutPath)
sample_writer.close()