#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import pytorch_lightning as pl
import time
from datetime import datetime
import math

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print(f'device = {device}')


# In[2]:


srcDataPath = '../data/train.bo'
tgtDataPath = '../data/train.en'

srcTokenizerPath = '../preProcessing/bo.model'
tgtTokenizerPath = '../preProcessing/en.model'


# Load data

# In[3]:


srcFile = open(srcDataPath, 'r', encoding = 'utf-8')
tgtFile = open(tgtDataPath, 'r', encoding = 'utf-8')

dataMatrix = []

while True: 
    srcLine = srcFile.readline().strip()
    tgtLine = tgtFile.readline().strip()
    if not srcLine or not tgtLine: 
        break 
    dataMatrix.append([srcLine, tgtLine])
  
# Create pandas dataframe 
df = pd.DataFrame(dataMatrix, columns = ['src', 'tgt'])
df


# In[4]:


srcTextsAll = df['src'].tolist()
tgtTextsAll = df['tgt'].tolist()


# ## Tokenizers for Tibetan and English
# 
# The code cell below uses Google SentencePiece tokenizer. 

# In[5]:


# Load tokenizers that are already trained
srcTokenizer = spm.SentencePieceProcessor(model_file=srcTokenizerPath)
tgtTokenizer = spm.SentencePieceProcessor(model_file=tgtTokenizerPath)

# Verify for Tibetan
print(srcTokenizer.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་'], out_type=str))
print(srcTokenizer.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་', 'བཀ྄ྲ་ཤིས་བདེ་ལེགས།'], out_type=int))
print(srcTokenizer.decode([4149, 306, 6, 245, 4660, 748]))
print(srcTokenizer.decode(['▁ངའི་', 'མིང་', 'ལ་', 'བསྟན་', 'སྒྲོལ་མ་', 'ཟེར་']))
print('Vocab size of Tibetan Tokenizer:', srcTokenizer.get_piece_size())

# Verify for English
print(tgtTokenizer.encode(["My name isn't Tenzin Dolma Gyalpo"], out_type=str))
print(tgtTokenizer.encode(['My name is Tenzin Dolma Gyalpo', 'Hello'], out_type=int))
print(tgtTokenizer.decode([[8804, 181, 13, 5520, 15172, 17895], [888, 21492]]))
print('Vocab size of English Tokenizer:', tgtTokenizer.get_piece_size())


# In[6]:


src_bos_id = srcTokenizer.piece_to_id('<s>')
src_eos_id = srcTokenizer.piece_to_id('</s>')
src_pad_id = srcTokenizer.piece_to_id('<pad>')
tgt_bos_id = tgtTokenizer.piece_to_id('<s>')
tgt_eos_id = tgtTokenizer.piece_to_id('</s>')
tgt_pad_id = tgtTokenizer.piece_to_id('<pad>')

print(src_bos_id, src_eos_id, src_pad_id, tgt_bos_id, tgt_eos_id, tgt_pad_id)


# The vectors of tokenization must have the same length. We thus define several helper functions for truncation and padding. 

# In[7]:


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
    ## For a transformer model, the target sentences have to be wrapped by <s> and </s>, but the source sentences don't have to 
    ## For T5 model, an ids vector is automatically shifted to the right, 
    
    if enable_bos_eos: 
        maxlen = maxlen - 2    # Need to reserve two positions for <s></s>
        bos_id = kwargs['bos_id']
        eos_id = kwargs['eos_id']
        
    # Truncate the sentence if needed 
    if len(sentvec) > maxlen: 
        newvec = sentvec[:maxlen].copy()
    else: 
        newvec = sentvec.copy()
        
    # Return the new vector
    if enable_bos_eos: 
        return [bos_id] + newvec + [eos_id]
    else: 
        return newvec


# In[8]:


def pad_and_get_attention_mask(sentvec, tolen, pad_id): 
    ''' 
    If a token list is shorter than tolen, then add <pad> until `tolen` and get the attention mask where 0--><pad> and 1-->non-pad characters 
    '''
    sentlen = len(sentvec)
    
    # No need to pad if the sentence is long enough 
    if len(sentvec) >= tolen: 
        return sentvec, [1] * sentlen
    
    else: 
        return sentvec + [pad_id] * (tolen - sentlen), [1] * sentlen + [0] * (tolen - sentlen)


# In[9]:


def trim(sentvec, tolen, pad_id, enable_bos_eos, **kwargs): 
    '''truncate and then pad a sentence. Return a tuple with ids and attention mask'''
    
    ids = truncate(sentvec, tolen, enable_bos_eos, **kwargs)
    ids, attention_mask = pad_and_get_attention_mask(ids, tolen, pad_id)
    return ids, attention_mask


# Show some examples to verify that our `trim()` function works. 

# In[10]:


ids, attention_mask = trim([100, 200, 300, 400, 500], tolen = 4, pad_id = tgt_pad_id, enable_bos_eos = False)
print(ids, attention_mask)


# In[11]:


ids, attention_mask = trim([100, 200, 300, 400, 500], tolen = 9, pad_id = tgt_pad_id, enable_bos_eos = False)
print(ids, attention_mask)


# In[12]:


ids, attention_mask = trim([100, 200, 300, 400, 500], tolen = 4, pad_id = tgt_pad_id, enable_bos_eos = True, bos_id = tgt_pad_id, eos_id = tgt_eos_id)
print(ids, attention_mask)


# In[13]:


ids, attention_mask = trim([100, 200, 300, 400, 500], tolen = 9, pad_id = tgt_pad_id, enable_bos_eos = True, bos_id = tgt_pad_id, eos_id = tgt_eos_id)
print(ids, attention_mask)


# ## Batch iterator
# 
# Returns a batch of token ids as torch tensors upon each call of `__next__()`. 

# In[14]:


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
        
    
    # Tokenize a list of texts and trim with special tokens
    # Return a tuple (list of [ids], list of [masks])
    def tokenize_batch_and_trim(self, text_batch, tokenizer, pad_id, enable_bos_eos, **kwargs):
        ids_batch = []
        maxlen = 0
        res_ids, res_attention_mask = [], []
        
        # Add <s></s> if needed and get maxlen 
        for text in text_batch: 
            ids = tokenizer.encode(text)
            # Add <s></s> if needed
            ids = truncate(ids, len(ids) + 10, enable_bos_eos, **kwargs)
            ids_batch.append(ids)
            # Update maxlen 
            if len(ids) > maxlen: 
                maxlen = len(ids)
        
        # Pad to the current maxlen in the batch 
        for ids in ids_batch: 
            padded_ids, attention_mask = pad_and_get_attention_mask(ids, maxlen, pad_id)
            res_ids.append(padded_ids)
            res_attention_mask.append(attention_mask)
        
        return res_ids, res_attention_mask
    
    
    def __iter__(self): 
        self.curr_idx = self.start_idx 
        return self 
    
    
    def __next__(self): 
        if self.curr_idx >= self.end_idx: 
            raise StopIteration 
            
        # Take care of indices for correct iteration 
        if self.curr_idx + self.batch_size < self.end_idx: 
            head, tail = self.curr_idx, self.curr_idx + self.batch_size
            self.curr_idx += self.batch_size
        else:
            head, tail = self.curr_idx, self.end_idx
            self.curr_idx = self.end_idx 
            
        # Get source and target texts 
        src_texts = self.srcTexts[head:tail]
        tgt_texts = self.tgtTexts[head:tail]
        
        # Tokenize
        src_ids, src_mask = self.tokenize_batch_and_trim(src_texts, self.srcTokenizer, self.src_pad_id, enable_bos_eos = False)
        tgt_ids, tgt_mask = self.tokenize_batch_and_trim(tgt_texts, self.tgtTokenizer, self.tgt_pad_id, enable_bos_eos = True, bos_id = self.tgt_bos_id, eos_id = self.tgt_eos_id)
        
        # Return the results as dictionaries of torch tensors 
        return {
            'src_ids': torch.LongTensor(src_ids).to(device),
            'src_mask': torch.FloatTensor(src_mask).to(device),
            'tgt_ids': torch.LongTensor(tgt_ids).to(device),
            'tgt_mask': torch.FloatTensor(tgt_mask).to(device),
        }
    
    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)
        


# Here is an example of how batch iterator works. 

# In[15]:


mbi = MyBatchIterator(
    srcTextsAll, tgtTextsAll, 
    srcTokenizer, tgtTokenizer,
    start_idx = 10, end_idx = 20, batch_size = 8, 
    src_pad_id = src_pad_id, tgt_pad_id = tgt_pad_id, 
    src_bos_id = src_bos_id, tgt_bos_id = tgt_pad_id, 
    src_eos_id = src_eos_id, tgt_eos_id = tgt_eos_id
)

mbi = iter(mbi)

print('length of iterator:', len(mbi))

for idx, batch in enumerate(mbi): 
    print(f"batch index: {idx}, src size: {batch['src_ids'].size()} = {batch['src_mask'].size()}; tgt size: {batch['tgt_ids'].size()} = {batch['tgt_mask'].size()}")
    print(f"sample src ids: {batch['src_ids'][0]}")
    print(f"sample src mask: {batch['src_mask'][0]}")
    print(f"sample tgt ids: {batch['tgt_ids'][0]}")
    print(f"sample tgt mask: {batch['tgt_mask'][0]}")
    print('='*50)


# ## Helper classes and functions
# 
# We define a `Timer` class for estimating remaining time for an epoch. 

# In[16]:


class Timer:
    def __init__(self, num_total_units):
        # num_total_units: How many units of tasks need to be done
        self.start = datetime.now()
        self.num_total_units = num_total_units

    def remains(self, num_done_units):
        # num_done_units: How many units of tasks are done
        now  = datetime.now()
        time_taken = now - self.start
        sec_taken = int(time_taken.total_seconds())
        time_left = (self.num_total_units - num_done_units) * (now - self.start) / num_done_units
        sec_left = int(time_left.total_seconds())
        return f"Time taken {sec_taken // 60:02d}:{sec_taken % 60:02d}, Estimated time left {sec_left // 60:02d}:{sec_left % 60:02d}"


# ## Define the training routine 

# In[17]:


def train(train_iter, val_iter, model, optimizer, scheduler, hparams): 
    train_losses = []
    val_losses = []
    train_step_counter = 0
    val_step_counter = 0
    tb_refresh_rate = 60    # Flush tensorboard log every ? sec
    msg_refresh_rate = 10    # Flush message log every ? sec 
    save_model_every = 10    # Save model every ? epoch 
    best_epoch = 0
    
    msg_writer = open('message.log', 'w')    # For logging training progress 
    tb_writer = SummaryWriter(flush_secs = tb_refresh_rate)    # Tensorboard writer 
    sample_writer = open('sample.log', 'w', encoding = 'utf-8')    # For logging example sentences 
    
    for epoch in range(hparams['num_epochs']): 
        print('Begin epoch', epoch)
        msg_writer.write(f'Epoch {epoch}/{hparams["num_epochs"]}\n')
        sample_writer.write(f'Epoch {epoch}/{hparams["num_epochs"]}\n\n')
        msg_writer.flush() 
        
        
        ''' Part I: Training loop '''
        model.train()    # Flip to train mode
        train_loss = 0
        
        msg_offset = msg_writer.tell()    # Overwrite progress info at this offset 
        refresh_timer_start = time.time()    # Start counting until refreshing the message log (refresh rate = 10s)
        myTimer = Timer(len(train_iter))    # For estimating remaining time for training
        
        for idx, batch in enumerate(train_iter): 
            src_ids = batch['src_ids']
            src_mask = batch['src_mask']
            tgt_ids = batch['tgt_ids']
            tgt_mask = batch['tgt_mask']
            
            decoder_input_ids = tgt_ids[:, :-1]    # Remove the last column, intended EOS
            labels = tgt_ids[:, 1:]    # Remove the first column (starting token)
            
            # Forward, backprop, optimizer, scheduler 
            optimizer.zero_grad()
            loss = T5model.forward(
                input_ids = src_ids, 
                attention_mask = src_mask, 
                decoder_input_ids = decoder_input_ids, 
                # decoder_attention_mask = tgt_mask,  # According to T5 doc, decoder attention mask is generated automatically so I won't define it myself. 
                labels = labels.masked_fill(labels == tgt_pad_id, -100)    # -100 means not to compute loss at this token 
            ).loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() / src_ids.size(0)    # For computing average later 
            
            # Tensorboard logging
            tb_writer.add_scalar('Epoch/train', epoch, train_step_counter)
            tb_writer.add_scalar('Loss(step)/train', loss, train_step_counter)
            for param_group in optimizer.param_groups:
                if param_group['lr']:
                    tb_writer.add_scalar('learning_rate*e-5', param_group['lr'] * 1e5, train_step_counter)
            train_step_counter += 1
    
            # Message logging 
            refresh_timer_end = time.time()
            if (refresh_timer_end - refresh_timer_start > msg_refresh_rate): 
                refresh_timer_start = time.time()    # reset refresh_time
                msg_writer.seek(msg_offset)
                msg_writer.write(f'Train batches {idx}/{len(train_iter)} completed. ')
                msg_writer.write(myTimer.remains(num_done_units = idx))
                msg_writer.flush()
                
        # Training epoch end 
        msg_writer.seek(msg_offset)
        msg_writer.write(f'Train batches {len(train_iter)}/{len(train_iter)} completed. ')
        msg_writer.write(myTimer.remains(num_done_units = len(train_iter)))
        msg_writer.write('\n')
        msg_writer.flush()
        
        
        
        ''' Part II: Eval loop '''
        model.eval()    # Flip to eval mode 
        val_loss = 0
        
        msg_offset = msg_writer.tell()    # Overwrite progress info at this offset 
        refresh_timer_start = time.time()    # Start counting until refreshing the message log (refresh rate = 10s)
        myTimer = Timer(len(train_iter))    # For estimating remaining time for training
        
        with torch.no_grad(): 
            for idx, batch in enumerate(val_iter): 
                src_ids = batch['src_ids']
                src_mask = batch['src_mask']
                tgt_ids = batch['tgt_ids']
                tgt_mask = batch['tgt_mask']

                decoder_input_ids = tgt_ids[:, :-1]    # Remove the last column, intended EOS
                labels = tgt_ids[:, 1:]    # Remove the first column (starting token)
                
                loss = T5model.forward(
                    input_ids = src_ids, 
                    attention_mask = src_mask, 
                    decoder_input_ids = decoder_input_ids, 
                    # decoder_attention_mask = tgt_mask,  # According to T5 doc, decoder attention mask is generated automatically so I won't define it myself. 
                    labels = labels.masked_fill(labels == tgt_pad_id, -100)    # -100 means not to compute loss at this token 
                ).loss
                val_loss += loss.item() / src_ids.size(0)    # For computing average later 
                
                # Tensorboard logging 
                tb_writer.add_scalar('Epoch/val', epoch, val_step_counter)
                tb_writer.add_scalar('Loss(step)/val', loss, val_step_counter)
                val_step_counter += 1
                
                # Message logging 
                refresh_timer_end = time.time()
                if (refresh_timer_end - refresh_timer_start > msg_refresh_rate): 
                    refresh_timer_start = time.time()    # reset refresh_time
                    msg_writer.seek(msg_offset)
                    msg_writer.write(f'Val batches {idx}/{len(train_iter)} completed. ')
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
        print(f'Epoch {epoch}/{hparams["num_epochs"]} completed. Train_loss: {train_loss / len(train_iter):.3f}. Val_loss: {val_loss / len(val_iter):.3f}')
        msg_writer.write(f'Epoch {epoch}/{hparams["num_epochs"]} completed. Train_loss: {train_loss / len(train_iter):.3f}. Val_loss: {val_loss / len(val_iter):.3f}')
        tb_writer.add_scalar('Loss(epoch)/train', train_loss / len(train_iter), epoch)
        tb_writer.add_scalar('Loss(epoch)/val', val_loss / len(val_iter), epoch)

        # Save best model till now
        if val_loss / len(val_iter) < min(val_losses, default = 1e9): 
            best_epoch = epoch
            print(f'Saving best state_dict...')
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
            translated_sentence = generate_translation(model, srcTextsAll[idx])
            sample_writer.write(f'Origianl source text: {srcTextsAll[idx]}\n\n')
            sample_writer.write(f'Original target text: {tgtTextsAll[idx]}\n\n')
            sample_writer.write(f'Predicted target text: {translated_sentence}\n\n')
            sample_writer.write('-' * 50 + '\n\n')
        
        
        sample_writer.flush()
        msg_writer.write('\n' + '=' * 70 + '\n\n')
        sample_writer.write('=' * 50 + '\n\n')
        
    # Wrap up
    msg_writer.write(f'Best epoch idx = {best_epoch}')
    torch.save(model.state_dict(), 'checkpoint_final_epoch.pt')
    msg_writer.close()
    tb_writer.close()
    sample_writer.close()


# ## Function for generating translation

# In[18]:


def generate_translation(model, src_text): 
    model.eval()
    
    src_ids = srcTokenizer.encode(src_text)
    src_ids = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    outs = model.generate(
        src_ids, 
        max_length = hparams['max_length'], 
        bos_token_id = None, 
        eos_token_id = tgt_eos_id, 
        pad_token_id = tgt_pad_id,
        num_beams = 4, 
        repetition_penalty = 2.5, 
        length_penalty = 0.6, 
        early_stopping = True, 
    )
    
    # If any token beyond vocab size, make it pad
    outs = outs.masked_fill(outs >= tgtTokenizer.get_piece_size(), tgt_pad_id)
    
    pred_text = tgtTokenizer.decode(outs[0].tolist())
    return pred_text
    
    


# ## Instantiate and train
# 
# After reading the documentation for `PretrainedConfig` and `PretrainedModel`, I got the idea that hyperparameters shall be passed as `**kwargs` when calling `from_pretrained()`. To see what hyperparameters are available to configure, 
# 
# ```
# T5model = T5ForConditionalGeneration.from_pretrained('t5-small')
# T5model.config_class().to_dict()
# ```
# 
# <b style="color:red;">Caution!</b> Set tgt_bos_id as pad id when instantiate iterator

# In[19]:


hparams = dict(
    num_epochs = 50, 
    train_batch_size = 8, 
    val_batch_size = 1, 
    train_percentage = 0.95, 
    val_percentage = 0.03, 
    checkpoint_at = [9, 19, 29, 39],   # At which intermediate epoch do we save model
    # --------------------------------------------------
    weight_decay = 1e-4, 
    warmup_steps = 4000, 
    dropout = 0.2,
    target_lr = 1e-4,     # max learning rate achieved by scheduler 
    adam_betas = (0.9, 0.98), 
    # adam_eps = 1e-9, 
    max_length = 100,    # max length of sequence to be generated 
)

T5model = T5ForConditionalGeneration.from_pretrained(
    't5-small', 
    return_dict = True, 
    # bos_token_id = tgt_pad_id,    # T5 starts generation with <pad> token, so I delete this line to avoid disruption
    eos_token_id = tgt_eos_id, 
    pad_token_id = tgt_pad_id, 
    decoder_start_token_id = tgt_pad_id,   # If I don't add this line, then all predictions start with <unk>
    dropout_rate = hparams['dropout'], 
    max_length = hparams['max_length'], 
).to(device)

optimizer_grouped_parameters = [
    {
        # parameters with weight decay 
        'params': [param for name, param in T5model.named_parameters() if ('bias' not in name and 'layer_norm.weight' not in name)], 
        'weight_decay': hparams['weight_decay'], 
    }, 
    {
        # parameters without weight decay
        'params': [param for name, param in T5model.named_parameters() if ('bias' in name or 'layer_norm.weight' in name)], 
        'weight_decay': 0.0, 
    }
]

optimizer = AdamW(
    optimizer_grouped_parameters, 
    lr = hparams['target_lr'], 
    betas = hparams['adam_betas'],
)

scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, 
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
    tgt_bos_id = tgt_pad_id, tgt_eos_id = tgt_eos_id
)

val_mbi = MyBatchIterator(
    srcTextsAll, tgtTextsAll, srcTokenizer, tgtTokenizer,
    start_idx = int(hparams['train_percentage'] * len(srcTextsAll)),
    end_idx = int((hparams['train_percentage'] + hparams['val_percentage']) * len(srcTextsAll)), 
    batch_size = hparams['val_batch_size'], 
    src_pad_id = src_pad_id, tgt_pad_id = tgt_pad_id, 
    tgt_bos_id = tgt_pad_id, tgt_eos_id = tgt_eos_id
)

train(iter(train_mbi), iter(val_mbi), T5model, optimizer, scheduler, hparams)


# For MistGPU only
import os
os.system('sh ~/shutdown.sh')




