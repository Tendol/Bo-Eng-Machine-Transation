# =======================================
##### Build a transformer from scratch
# =======================================


# Import models and set path of data 

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



# --------------------------
#### Section 1: Load data
# --------------------------

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



# --------------------------
#### Section 2: Prepare for tokenization of text data
# --------------------------
''' 
# Turn a string into a list of numerical token ids 
# In section 2, we only import the tokenizer and define the necessary helper functions. The actual tokenization will happen in MyBatchIterator, a class that grabs text data by batch and tokenize the texts. 
'''

# Load tokenizers that are already trained
srcTokenizer = spm.SentencePieceProcessor(model_file=srcTokenizerPath)
tgtTokenizer = spm.SentencePieceProcessor(model_file=tgtTokenizerPath)

# Get the ids for special tokens 
# <s> -- begin of sentence (bos)
# </s> -- end of sentence (eos)
# <pad> -- pad token that fill a token vector to s specific length 
src_bos_id = srcTokenizer.piece_to_id('<s>')
src_eos_id = srcTokenizer.piece_to_id('</s>')
src_pad_id = srcTokenizer.piece_to_id('<pad>')
tgt_bos_id = tgtTokenizer.piece_to_id('<s>')
tgt_eos_id = tgtTokenizer.piece_to_id('</s>')
tgt_pad_id = tgtTokenizer.piece_to_id('<pad>')

'''
# For a transformer to work, target token ids must be wrapping by <s></s>
# The token vectors in the same training batch must have the same length. 
# We thus define helper functions for truncation, padding, and adding special tokens
'''


def truncate(sentvec, maxlen, enable_bos_eos, **kwargs): 
    '''
    Truncate a sentence vector to maxlen by deleting the trailing ids. 
    Args
    -- sentvec. List. Vector of tokenization of a sentence 
    -- maxlen. Int. The max length of tokenization. Must >=3 
    -- pad_id. Int. The id for <pad>
    -- enable_bos_eos. Bool. Indicate whether to wrap a sentence with <s> and </s> 
    -- kwargs['bos_id']. Int. The id for <s>
    -- kwargs['eos_id']. Int. The id for </s> 
    '''
    
    # No error checking for now

    # Reserve two places for <s></s> if enable_bos_eos is set to True 
    if enable_bos_eos: 
        maxlen = maxlen - 2    # Need to reserve two positions for <s></s>
        bos_id = kwargs['bos_id']
        eos_id = kwargs['eos_id']
        
    # Remove trailing token ids if needed 
    if len(sentvec) > maxlen: 
        newvec = sentvec[:maxlen].copy()
    else: 
        newvec = sentvec.copy()
        
    # Return the new token vector
    if enable_bos_eos: 
        return [bos_id] + newvec + [eos_id]
    else: 
        return newvec


def pad(sentvec, maxlen, pad_id): 
    ''' 
    # If a token list is shorter than tolen, then add <pad> until `tolen` and get the attention mask where 0--><pad> and 1-->non-pad characters 
    '''
    sentlen = len(sentvec)
    
    # No need to pad if the sentence is long enough 
    if len(sentvec) >= maxlen: 
        return sentvec
    
    else: 
        return sentvec + [pad_id] * (maxlen - sentlen)


def trim(sentvec, maxlen, pad_id, enable_bos_eos, **kwargs): 
    '''truncate and then pad a sentence. '''
    
    ids = truncate(sentvec, maxlen, enable_bos_eos, **kwargs)
    ids= pad(ids, maxlen, pad_id)
    return ids



# --------------------------
#### Section 3: Helper classes: MyBatchIterator and Timer
# --------------------------

'''
# Deep learning models usually process a large piece of data by small batches. 
# The class `MyBatchIterator` is an iterator for grabbing text data by specified batch size, and then tokenize the text into numerical token ids. 
'''

class MyBatchIterator: 
    def __init__(self, srcTexts, tgtTexts, 
                 srcTokenizer, tgtTokenizer,
                 start_idx, end_idx, batch_size, 
                 src_pad_id, tgt_pad_id, 
                 src_bos_id = None, tgt_bos_id = None, 
                 src_eos_id = None, tgt_eos_id = None
                ): 
        self.srcTexts = srcTexts
        self.tgtTexts = tgtTexts
        self.srcTokenizer = srcTokenizer 
        self.tgtTokenizer = tgtTokenizer
        self.start_idx = start_idx    # Starting index of original dataset, inclusive
        self.end_idx = end_idx    # Ending index of original dataset, exclusive 
        self.batch_size = batch_size    # batch_size specified by user s
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.src_bos_id = src_bos_id
        self.tgt_bos_id = tgt_bos_id 
        self.src_eos_id = src_eos_id
        self.tgt_eos_id = tgt_eos_id 
    
    
    # Tokenize a batch of texts and trim with special tokens
    def tokenize_batch_and_trim(self, text_batch, tokenizer, pad_id, enable_bos_eos, **kwargs):
        ids_batch = []
        maxlen = 0

        # Add <s></s> if needed 
        # Get the maximum vector length in the current batch 
        for text in text_batch: 
            ids = tokenizer.encode(text)
            # Add <s></s> if needed
            ids = truncate(ids, len(ids) + 10, enable_bos_eos, **kwargs)
            ids_batch.append(ids)
            # Update maxlen 
            if len(ids) > maxlen: 
                maxlen = len(ids)
    
        # Pad all vectors to the maximum length
        padded_ids_batch = [pad(ids, maxlen, pad_id) for ids in ids_batch]
        return torch.tensor(padded_ids_batch).to(device)

    
    def __iter__(self):
        self.curr_idx = self.start_idx
        return self
    
    
    # Defines what happends when next() is called on the iterator 
    # When next() is called, grab the next batch of texts, tokenize them, and return token ids
    def __next__(self): 
        if self.curr_idx >= self.end_idx: 
            raise StopIteration  
        
        # Get text batch
        if self.curr_idx + self.batch_size < self.end_idx: 
            text_batch_dict = {
                'src': self.srcTexts[self.curr_idx : self.curr_idx + self.batch_size],
                'tgt': self.tgtTexts[self.curr_idx : self.curr_idx + self.batch_size], 
            }
            self.curr_idx += self.batch_size
        else:
            text_batch_dict = {
                'src': self.srcTexts[self.curr_idx : self.end_idx], 
                'tgt': self.tgtTexts[self.curr_idx : self.end_idx],
            }
            self.curr_idx = self.end_idx
        
        # Tokenize text batch
        return {
            # No special token except for <pad> for source tokenization
            'src': self.tokenize_batch_and_trim(text_batch_dict['src'], self.srcTokenizer, self.src_pad_id, enable_bos_eos = False), 
            # Add <s></s><pad> for target tokenization
            'tgt': self.tokenize_batch_and_trim(text_batch_dict['tgt'], self.tgtTokenizer, self.tgt_pad_id, enable_bos_eos = True, bos_id = self.tgt_bos_id, eos_id = self.tgt_eos_id)
        }
    

    # The length of iterator
    # i.e. The total number of batches 
    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)
        

'''
# The `Timer` class is for estimating the remaining time for processing a batch 
'''
class Timer:
    def __init__(self, num_total_units):
        # num_total_units: How many units of tasks need to be done
        self.start = datetime.datetime.now()
        self.num_total_units = num_total_units

    def remains(self, num_done_units):
        # num_done_units: How many units of tasks are done
        now  = datetime.datetime.now()
        time_taken = now - self.start
        sec_taken = int(time_taken.total_seconds())
        time_left = (self.num_total_units - num_done_units) * (now - self.start) / num_done_units
        sec_left = int(time_left.total_seconds())
        return f"Time taken {sec_taken // 60:02d}:{sec_taken % 60:02d}, Estimated time left {sec_left // 60:02d}:{sec_left % 60:02d}"



# --------------------------
#### Section 4: Encoder and Model class
# --------------------------

class PositionalEncoding(nn.Module):   
    def __init__(self, hparams): 
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = hparams['dropout'])
        self.d_model = hparams['d_model']
        pe = torch.zeros(hparams['max_len'], self.d_model)    # positional encoding 
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



# --------------------------
#### Section 5: Training routine
# --------------------------

def train(train_iter, val_iter, model, optim, scheduler, hparams): 
    train_losses = []    # For storing averages losses durinig each train epoch
    val_losses = []      # For storing averages losses during each val epoch
    train_step_counter = 0
    val_step_counter = 0
    tb_refresh_rate = 60    # Flush tensorboard log every ~ seconds
    msg_refresh_rate = 10    # Flush message log every ~ seconds
    best_epoch = 0
    
    msg_writer = open('message.log', 'w')    # For logging training progress
    tb_writer = SummaryWriter(flush_secs=tb_refresh_rate)    # Tensorboard writer 
    sample_writer = open('sample.log', 'w', encoding = 'utf-8')    # For logging example sentences
    
    for epoch in range(hparams['num_epochs']):      
        torch.cuda.empty_cache()   
        msg_writer.write(f'Epoch {epoch}/{hparams["num_epochs"]}\n')
        sample_writer.write(f'Epoch {epoch}/{hparams["num_epochs"]}\n')
        msg_writer.flush() 
        
        ''' Part I: Training loop '''
        model.train()    # Flip to train mode 
        train_loss = 0
        
        msg_offset = msg_writer.tell()    # Will overwrite progress info at this offset 
        refresh_timer_start = time.time()    # Count time until refreshing the message log (refresh rate = `msg_refresh_rate`)
        myTimer = Timer(len(train_iter))    # For estimating remaining time for training each epoch
        
        for idx, batch in enumerate(train_iter): 
            # Get token ids 
            src = batch['src'].to(device)    # batch_size * maxlen(src)
            tgt = batch['tgt'].to(device)    # batch_size * maxlen(tgt)
            
            tgt_input = tgt[:, :-1]    # Remove the last column, intended EOS 
            targets = tgt[:, 1:].contiguous().view(-1)    # Remove the first column (BOS should not be used for computing loss)
            
            # Create attention masks 
            src_mask = (src != 0).float().to(device)
            src_mask = src_mask.masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0))    # map 0-->(-inf), 1-->0. What for? 
            tgt_mask = (tgt_input != 0).float().to(device)
            tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0))    # map 0-->(-inf), 1-->0. What for? 
            
            size = tgt_input.size(1)    # size of target len with final token removed 
            np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1).to(device)    # This mask looks like Fig.3(b) (causal mask) in T5 paper. What np means? 
            np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0))    # map 0-->(-inf), 1-->0. What for? 
            
            # Forward, backprop, optimizer 
            optim.zero_grad()
            preds = model(
                src.transpose(0, 1), 
                tgt_input.transpose(0, 1), 
                tgt_mask = np_mask, 
                # src_mask = src_mask, 
                # tgt_key_padding_mask = tgt_mask
                # I have no idea why these two args are commented out
            )
            preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))    # Why transpose back? Then convert to 2D tensor reserving column number 
            loss = F.cross_entropy(preds, targets, ignore_index = 0, reduction = 'sum')
            loss.backward()
            optim.step()
            scheduler.step()
            train_loss += loss.item() / src.size(0)    # Tutorial uses the constant BATCH_SIZE as denominator, but since the final batch may have a smaller size, I decided to use current batch size 
            
            # Tensorboard logging 
                # Which epoch are we at 
                # Loss of current training batch 
            tb_writer.add_scalar('Epoch/train', epoch, train_step_counter)
            tb_writer.add_scalar('Loss(step)/train', loss, train_step_counter)
            train_step_counter += 1
            
            # Message logging 
                # Show how many batches are completed
                # Show time elapsed and expected remaining time 
            refresh_timer_end = time.time()
            if (refresh_timer_end - refresh_timer_start > msg_refresh_rate): 
                refresh_timer_start = time.time()    # reset refresh_time
                msg_writer.seek(msg_offset)
                msg_writer.write(f'Train batches {idx}/{len(train_iter)} completed. ')
                msg_writer.write(myTimer.remains(num_done_units = idx))
                msg_writer.flush()
                
        # Training epoch end 
        msg_writer.seek(msg_offset)    # Will overwrite previous progress log
        msg_writer.write(f'Train batches {len(train_iter)}/{len(train_iter)} completed. ')
        msg_writer.write(myTimer.remains(num_done_units = len(train_iter)))
        msg_writer.write('\n')
        msg_writer.flush() 
       
    
        ''' Part II: Eval loop '''
        model.eval()    # Flip to eval mode 
        val_loss = 0
        
        msg_offset = msg_writer.tell()    # Overwrite progress info at this offset 
        refresh_timer_start = time.time()    # Start counting until refreshing the message log (refresh rate = 10s)
        myTimer = Timer(len(val_iter))    # For estimating remaining time for cross-validating each epoch 
        
        with torch.no_grad(): 
            for idx, batch in enumerate(val_iter): 
                # Get token ids
                src = batch['src'].to(device)    # batch_size * maxlen(src)
                tgt = batch['tgt'].to(device)    # batch_size * maxlen(tgt)
               
                tgt_input = tgt[:, :-1]    # Remove the last column, intended EOS  
                targets = tgt[:, 1:].contiguous().view(-1)    # Remove the first column (BOS should not be used for computing loss)
                
                # Create attention masks 
                src_mask = (src != 0).float().to(device)
                src_mask = src_mask.masked_fill(src_mask == 0, float('-inf')).masked_fill(src_mask == 1, float(0))    # map 0-->(-inf), 1-->0. What for? 
                tgt_mask = (tgt_input != 0).float().to(device)
                tgt_mask = tgt_mask.masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0))    # map 0-->(-inf), 1-->0. What for? 

                size = tgt_input.size(1)    # size of target len with final token removed 
                np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1).to(device)    # This mask looks like Fig.3(b) (causal mask) in T5 paper. What np means? 
                np_mask = np_mask.float().masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0))    # map 0-->(-inf), 1-->0. What for? 
                
                # Forward 
                preds = model(
                    src.transpose(0, 1), 
                    tgt_input.transpose(0, 1), 
                    tgt_mask = np_mask, 
                    # src_mask = src_mask, 
                    # tgt_key_padding_mask = tgt_mask
                    # I have no idea why these two args are commented out
                )
                preds = preds.transpose(0, 1).contiguous().view(-1, preds.size(-1))    # Why transpose back? Then convert to 2D tensor reserving column number 
                loss = F.cross_entropy(preds, targets, ignore_index = 0, reduction = 'sum')
                val_loss += loss.item() / src.size(0)
                
                # Tensorboard logging 
                    # Which epoch are we at 
                    # Loss of current validation batch 
                tb_writer.add_scalar('Epoch/val', epoch, val_step_counter)
                tb_writer.add_scalar('Loss(step)/val', loss, val_step_counter)
                val_step_counter += 1
                
                # Message logging 
                    # Show how many batches are completed
                    # Show time elapsed and expected remaining time 
                refresh_timer_end = time.time()
                if (refresh_timer_end - refresh_timer_start > msg_refresh_rate): 
                    refresh_timer_start = time.time()    # reset refresh_time
                    msg_writer.seek(msg_offset)
                    msg_writer.write(f'Val batches {idx}/{len(val_iter)} completed. ')
                    msg_writer.write(myTimer.remains(num_done_units = idx))
                    msg_writer.flush()
                
            # Val epoch end 
            msg_writer.seek(msg_offset)
            msg_writer.write(f'Val batches {len(val_iter)}/{len(val_iter)} completed. ')
            msg_writer.write(myTimer.remains(num_done_units = len(val_iter)))
            msg_writer.write('\n')
            msg_writer.flush() 
            

            ## Epoch end 
            
            # Extra logs
            # Average train loss and val loss during this epoch
            print(f'Epoch {epoch}/{hparams["num_epochs"]} completed. Train_loss: {train_loss / len(train_iter):.3f}. Val_loss: {val_loss / len(val_iter):.3f}')
            msg_writer.write(f'Epoch {epoch}/{hparams["num_epochs"]} completed. Train_loss: {train_loss / len(train_iter):.3f}. Val_loss: {val_loss / len(val_iter):.3f}')
            tb_writer.add_scalar('Loss(epoch)/train', train_loss / len(train_iter), epoch)
            tb_writer.add_scalar('Loss(epoch)/val', val_loss / len(val_iter), epoch)
            
            # Save best model till now 
            if val_loss / len(val_iter) < min(val_losses, default = 1e9): 
                best_epoch = epoch
                msg_writer.write(f'Saving state_dict...\n')
                torch.save(model.state_dict(), 'checkpoint_best_epoch.pt')

            # Save checkpoint model 
            if epoch in hparams['checkpoint_at']: 
                print(f'Saving checkpoint state_dict...')
                torch.save(model.state_dict(), f'checkpoint_epoch={epoch}.pt')
                
            train_losses.append(train_loss / len(train_iter))
            val_losses.append(val_loss / len(val_iter))

            # Check sentence examples after each epoch
            example_sent_idx = [0, 1, 2, 127, 214, 377, 277, 206]
            for idx in example_sent_idx: 
                translated_sentence = greedy_decode_sentence(model, srcTextsAll[idx])
                sample_writer.write(f'Origianl source text: {srcTextsAll[idx]}\n')
                sample_writer.write(f'Original target text: {tgtTextsAll[idx]}\n')
                sample_writer.write(f'Predicted target text: {translated_sentence}\n\n')
            
            sample_writer.flush()
            msg_writer.write('\n' + '=' * 50 + '\n\n')
            sample_writer.write('=' * 50 + '\n\n')
            
    # Wrap up the training routine 
    msg_writer.write(f'Best epoch idx = {best_epoch}')
    torch.save(model.state_dict(), 'checkpoint_final_epoch.pt')
    msg_writer.close()
    tb_writer.close()
    sample_writer.close()



'''
# Define the helper function for generating translation for a source text
# Use "greedy-decoding" algorithm 
'''

def greedy_decode_sentence(model, sentence, max_len = 100): # Restrict translation up to 100 words 
    model.eval()
    src = torch.LongTensor([srcTokenizer.encode(sentence)]).to(device) 
    tgt_init_tok = tgt_bos_id
    tgt = torch.LongTensor([[tgt_init_tok]]).to(device)    # For forward() purpose, stored as a column vector
    translated_sentence = ''
    
    for i in range(max_len): 
        size = tgt.size(0)
        np_mask = torch.triu(torch.ones(size, size) == 1).transpose(0, 1).to(device).float()
        np_mask = np_mask.masked_fill(np_mask == 0, float('-inf')).masked_fill(np_mask == 1, float(0))
        
        # Predict the next word based on previous words 
        pred = model(src.transpose(0, 1), tgt, tgt_mask = np_mask)
        generated_id = pred.argmax(dim = 2)[-1]    # Not sure the mechanism behind
        generated_word = tgtTokenizer.decode([generated_id.item()])
        translated_sentence += (' ' + generated_word)
        
        # Stop generation when </s> is generated
        if generated_id == tgt_eos_id: 
            break 
        
        # Append the new token to tgt
        tgt = torch.cat((tgt, torch.LongTensor([[generated_id]]).to(device)))
        
    return translated_sentence



# --------------------------
#### Section 6: Instantiate and train! 
# --------------------------

'''
# In this final section, we
    # Specify hyperparameters
    # Instantiate model 
    # Instantiate optimizer and scheduler 
    # Instantiate the batch iterator 
    # Start training 
'''

hparams = dict(
    d_model = 512, 
    dropout = 0.3, 
    max_len = 5000,    
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


model = MyTransformer(hparams).to(device)

optim = torch.optim.Adam(model.parameters(), lr = hparams['lr'], betas = hparams['adam_betas'], weight_decay = hparams['weight_decay'])

# The scheduler first warm up to the target learning rate and then decay according to a cosine function
scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optim, 
    num_warmup_steps = hparams['warmup_steps'], 
    num_training_steps = hparams['num_epochs'] * math.ceil(len(srcTextsAll) / hparams['train_batch_size']), 
    num_cycles = 3
)

train_mbi = MyBatchIterator(
    srcTextsAll, tgtTextsAll, srcTokenizer, tgtTokenizer,
    start_idx = 0, 
    end_idx = int(hparams['train_percentage'] * len(srcTextsAll)), 
    batch_size = hparams['train_batch_size'], 
    src_pad_id = src_pad_id, tgt_pad_id = tgt_pad_id, 
    tgt_bos_id = tgt_bos_id, tgt_eos_id = tgt_eos_id)

val_mbi = MyBatchIterator(
    srcTextsAll, tgtTextsAll, srcTokenizer, tgtTokenizer,
    start_idx = int(hparams['train_percentage'] * len(srcTextsAll)),
    end_idx = int((hparams['train_percentage'] + hparams['val_percentage']) * len(srcTextsAll)), 
    batch_size = hparams['val_batch_size'], 
    src_pad_id = src_pad_id, tgt_pad_id = tgt_pad_id, 
    tgt_bos_id = tgt_bos_id, tgt_eos_id = tgt_eos_id)

train(iter(train_mbi), iter(val_mbi), model, optim, scheduler, hparams)


