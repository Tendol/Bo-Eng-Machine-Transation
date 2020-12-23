import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup
import sentencepiece as spm
import pandas as pd
from typing import Optional
import math
import time
import datetime


device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)

print(f'device = {device}')

srcDataPath = '../data/train.bo'
tgtDataPath = '../data/train.en'

srcTokenizerPath = '../preProcessing/bo.model'
tgtTokenizerPath = '../preProcessing/en.model'

sampleOutPath = './Scratch_sample_results.txt'


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
tgt_bos_id = tgtTokenizer.piece_to_id('<s>')
tgt_eos_id = tgtTokenizer.piece_to_id('</s>')
tgt_pad_id = tgtTokenizer.piece_to_id('<pad>')


## Define model class for loading state_dict later 
class PositionalEncoding(nn.Module):    # What PositionalEncoding for? 
    def __init__(self, hparams): 
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = hparams['dropout'])
        self.d_model = hparams['d_model']
        pe = torch.zeros(hparams['max_len'], self.d_model)    # What pe mean? 
        position = torch.arange(0, hparams['max_len']).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (
                -math.log(10000.0) / self.d_model
            )
        )    # What for? 
        pe[:, 0::2] = torch.sin(position * div_term)    # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)    # odd dimensions
        pe = pe.unsqueeze(0).transpose(0, 1)    # Unsqueeze turns a matrix to a 3D tensor. Transpose 0th and 1st dim? 
        self.register_buffer('pe', pe)
        
    def forward(self, x): 
        x = x * math.sqrt(self.d_model)    # What for        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    
class MyTransformer(nn.Module): 
    def __init__(self, hparams) -> None: 
        super(MyTransformer, self).__init__()
        
        self.source_embedding = nn.Embedding(
            hparams['source_vocab_length'], hparams['d_model']
        )
        self.pos_encoder = PositionalEncoding(hparams)
        encoder_layer = nn.TransformerEncoderLayer(
            hparams['d_model'], hparams['nhead'], 
            hparams['dim_feedforward'], hparams['dropout'], 
            hparams['activation']
        )
        encoder_norm = nn.LayerNorm(hparams['d_model'])    # What for? 
        self.encoder = nn.TransformerEncoder(
            encoder_layer, hparams['num_encoder_layers'], encoder_norm
        )
        
        self.target_embedding = nn.Embedding(
            hparams['target_vocab_length'], hparams['d_model']
        )
        decoder_layer = nn.TransformerDecoderLayer(
            hparams['d_model'], hparams['nhead'], 
            hparams['dim_feedforward'], hparams['dropout'], 
            hparams['activation']
        )
        decoder_norm = nn.LayerNorm(hparams['d_model'])
        self.decoder = nn.TransformerDecoder(
            decoder_layer, hparams['num_decoder_layers'], decoder_norm
        )
        
        self.out = nn.Linear(hparams['d_model'], hparams['target_vocab_length'])   # The original examples wrote nn.Linear(512, target_vocab_length). I suspect this is a typo as hard-coding numbers is not really cool 
        
        self._reset_parameters()
        self.d_model = hparams['d_model']
        self.nhead = hparams['nhead']
        
        
    def forward(self, src: Tensor, tgt: Tensor,
                src_mask: Optional[Tensor] = None, 
                tgt_mask: Optional[Tensor] = None, 
                memory_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None, 
                tgt_key_padding_mask: Optional[Tensor] = None, 
                memory_key_padding_mask: Optional[Tensor] = None
               ) -> Tensor: 
        # Why batch size is the number of columns instead of rows? 
        if src.size(1) != tgt.size(1): 
            raise RuntimeError('The batch number of src and tgt must be equal')
            
        src = self.source_embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src, mask = src_mask, src_key_padding_mask = src_key_padding_mask)
        
        tgt = self.target_embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(
            tgt, memory, tgt_mask = tgt_mask, 
            memory_mask = memory_mask, 
            tgt_key_padding_mask = tgt_key_padding_mask, 
            memory_key_padding_mask = memory_key_padding_mask
        )
        output = self.out(output)
        return output
        
    
    def _reset_parameters(self): 
        r'''Initiate parameters in the transformer model'''
        # How work? 
        for p in self.parameters(): 
            if p.dim() > 1: 
                torch.nn.init.xavier_uniform_(p)


hparams = dict(
    d_model = 512, 
    dropout = 0.3, 
    max_len = 5000,    # I don't know what this maxlen is for 
    nhead = 8,    # Little understand what for 
    num_encoder_layers = 6, 
    num_decoder_layers = 6, 
    dim_feedforward = 2048, 
    activation = 'relu', 
    source_vocab_length = srcTokenizer.get_piece_size(),    # Consider increase
    target_vocab_length = tgtTokenizer.get_piece_size(),    # Consider increase 
    num_epochs = 50, 
    train_batch_size = 8, 
    val_batch_size = 1,     # For minimal padding or avoiding padding 
    lr = 1e-4, 
    adam_betas = (0.9, 0.98), 
    # adam_eps = 1e-9, 
    weight_decay = 1e-4, 
    warmup_steps = 4000, 
    train_percentage = 0.95, 
    val_percentage = 0.02, 
    checkpoint_at = [9, 19, 29, 39], 
)


# Load state dictionary
print('Loading model...')
state_dict = torch.load('Scratch_checkpoint_best_epoch=34.pt', map_location = device)
model = MyTransformer(hparams).to(device)
model.load_state_dict(state_dict)
print('Model loading complete')


## Function for generating translation 
# Use greedy decoding 
def greedy_decode_sentence(model, sentence, max_len = 100): # Restrict translation up to 100 words 
    model.eval()
    src = torch.LongTensor([srcTokenizer.encode(sentence)]).to(device)    #  !! Caution! Datatype for autograd 
    tgt_init_tok = tgt_bos_id
    tgt = torch.LongTensor([[tgt_init_tok]]).to(device)    # For forward() purpose, stored as a column vector
    translated_sentence = ''
    
    for i in range(max_len): 
        size = tgt.size(0)
        np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1).to(device).float()
        np_mask = np_mask.masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0))
        
        pred = model(src.transpose(0, 1), tgt, tgt_mask = np_mask)
        generated_id = pred.argmax(dim = 2)[-1]    # Not sure the mechanism behind
        generated_word = tgtTokenizer.decode([generated_id.item()])
        translated_sentence += (' ' + generated_word)

        # Append the new token to tgt
        tgt = torch.cat((tgt, torch.LongTensor([[generated_id]]).to(device)))
        
        # Stop generation when </s> is generated
        if generated_id == tgt_eos_id: 
            break 
            
    return translated_sentence


## Pick selected examples, generate translation, and compare 
selected = [0, 1, 2, 13, 24, 41]
sample_writer = open(sampleOutPath, 'w', encoding='utf-8')
print('Generating translations for selected sentences...')
for idx in selected: 
    translated_sentence = greedy_decode_sentence(model, srcTextsAll[idx])
    sample_writer.write(f'Origianl source text: {srcTextsAll[idx]}\n\n')
    sample_writer.write(f'Original target text: {tgtTextsAll[idx]}\n\n')
    sample_writer.write(f'Predicted target text: {translated_sentence}\n\n')
    sample_writer.write('-' * 50 + '\n\n')
print('Generation complete. See', sampleOutPath)
sample_writer.close()