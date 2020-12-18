#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sentencepiece as spm
import pandas as pd
from tokenizers import SentencePieceBPETokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import SparseAdam
from transformers import (
    T5Model, 
    T5ForConditionalGeneration, 
    AdamW,
    get_linear_schedule_with_warmup
)
import pytorch_lightning as pl
import time
from datetime import datetime
import textwrap

device = torch.device(
    'cuda:0' if torch.cuda.is_available() else 'cpu'
)
print(f'device = {device}')


# In[2]:


boDataPath = '../data/train.bo'
enDataPath = '../data/train.en'

boTokenizerPath = '../preProcessing/bo.model'
enTokenizerPath = '../preProcessing/en.model'


# ## Load data 
# 

# In[3]:


boFile = open(boDataPath, 'r', encoding = 'utf-8')
enFile = open(enDataPath, 'r', encoding = 'utf-8')

dataMatrix = []

while True: 
    boLine = boFile.readline().strip()
    enLine = enFile.readline().strip()
    if not boLine or not enLine: 
        break 
    dataMatrix.append([boLine, enLine])
  
# Create pandas dataframe 
df = pd.DataFrame(dataMatrix, columns = ['bo', 'en'])
df


# In[4]:


boTextsAll = df['bo'].tolist()
enTextsAll = df['en'].tolist()


# ## Tokenizers for Tibetan and English
# 
# The code cell below uses Google SentencePiece tokenizer. 

# In[5]:


# Load tokenizers that are already trained
boTokenizer = spm.SentencePieceProcessor(model_file=boTokenizerPath)
enTokenizer = spm.SentencePieceProcessor(model_file=enTokenizerPath)

# Verify for Tibetan
print(boTokenizer.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་'], out_type=str))
print(boTokenizer.encode(['ངའི་མིང་ལ་བསྟན་སྒྲོལ་མ་ཟེར་', 'བཀ྄ྲ་ཤིས་བདེ་ལེགས།'], out_type=int))
print(boTokenizer.decode([4149, 306, 6, 245, 4660, 748]))
print(boTokenizer.decode(['▁ངའི་', 'མིང་', 'ལ་', 'བསྟན་', 'སྒྲོལ་མ་', 'ཟེར་']))
print('Vocab size of Tibetan Tokenizer:', boTokenizer.get_piece_size())

# Verify for English
print(enTokenizer.encode(["My name isn't Tenzin Dolma Gyalpo"], out_type=str))
print(enTokenizer.encode(['My name is Tenzin Dolma Gyalpo', 'Hello'], out_type=int))
print(enTokenizer.decode([[8804, 181, 13, 5520, 15172, 17895], [888, 21492]]))
print('Vocab size of English Tokenizer:', enTokenizer.get_piece_size())


# We need to get the ids for our special tokens `<s>`, `</s>`, `<pad>`. 

# In[6]:


bo_bos_id = boTokenizer.piece_to_id('<s>')
bo_eos_id = boTokenizer.piece_to_id('</s>')
bo_pad_id = boTokenizer.piece_to_id('<pad>')
en_bos_id = enTokenizer.piece_to_id('<s>')
en_eos_id = enTokenizer.piece_to_id('</s>')
en_pad_id = enTokenizer.piece_to_id('<pad>')

print(bo_bos_id, bo_eos_id, bo_pad_id, en_bos_id, en_eos_id, en_pad_id)


# The vectors of tokenization must have the same length. We thus define several helper functions for truncation and padding

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


def pad_and_get_attention_mask(sentvec, maxlen, pad_id): 
    ''' 
    Pad a sentence to maxlen and get the attention mask where 0--><pad> and 1-->non-pad characters 
    '''
    
    sentlen = len(sentvec)
    
    # No need to pad if the sentence is long enough 
    if len(sentvec) >= maxlen: 
        return sentvec, [1] * sentlen
    
    else: 
        return sentvec + [pad_id] * (maxlen - sentlen), [1] * sentlen + [0] * (maxlen - sentlen)


# In[9]:


def trim(sentvec, maxlen, pad_id, enable_bos_eos, **kwargs): 
    '''truncate and then pad a sentence. Return a tuple with ids and attention mask'''
    
    ids = truncate(sentvec, maxlen, enable_bos_eos, **kwargs)
    ids, attention_mask = pad_and_get_attention_mask(ids, maxlen, pad_id)
    return ids, attention_mask


# Show some examples to verify that our `trim()` function works. 

# In[10]:


ids, attention_mask = trim([100, 200, 300, 400, 500], maxlen = 4, pad_id = en_pad_id, enable_bos_eos = False)
print(ids, attention_mask)


# In[11]:


ids, attention_mask = trim([100, 200, 300, 400, 500], maxlen = 9, pad_id = en_pad_id, enable_bos_eos = False)
print(ids, attention_mask)


# In[12]:


ids, attention_mask = trim([100, 200, 300, 400, 500], maxlen = 4, pad_id = en_pad_id, enable_bos_eos = True, bos_id = en_pad_id, eos_id = en_eos_id)
print(ids, attention_mask)


# In[13]:


ids, attention_mask = trim([100, 200, 300, 400, 500], maxlen = 9, pad_id = en_pad_id, enable_bos_eos = True, bos_id = en_pad_id, eos_id = en_eos_id)
print(ids, attention_mask)


# Finally, we find out appropriate max_len for our Tibetan and English data. 

# In[14]:


maxBoLen, maxEnLen = 0, 0
boOver100, enOver100 = 0, 0 
boOver50, enOver70 = 0, 0 

for bo in boTextsAll: 
    x = boTokenizer.encode(bo)
    if len(x) > maxBoLen:
        maxBoLen = len(x)
    if len(x) > 100: 
        boOver100 += 1
    if len(x) > 50: 
        boOver50 += 1
        
for en in enTextsAll: 
    x = enTokenizer.encode(en)
    if len(x) > maxEnLen:
        maxEnLen = len(x)  
    if len(x) > 100: 
        enOver100 += 1
    if len(x) > 70:
        enOver70 += 1
        
print('maxBoLen:', maxBoLen)
print('maxEnLen:', maxEnLen)
print('Number bo-text longer than 100:', boOver100)
print('Number en-text longer than 100:', enOver100)
print('Number bo-text longer than 50:', boOver50)
print('Number en-text longer than 70:', enOver70)


# ## Pytorch `Dataset`

# In[15]:


class MyDataset(Dataset): 
    def __init__(self, boTexts, enTexts, boTokenizer, enTokenizer, boMaxLen, enMaxLen): 
        super().__init__()
        self.boTexts = boTexts
        self.enTexts = enTexts
        self.boTokenizer = boTokenizer
        self.enTokenizer = enTokenizer
        self.boMaxLen = boMaxLen
        self.enMaxLen = enMaxLen
        
    ''' Return the size of dataset '''
    def __len__(self): 
        return len(self.boTexts)
    
    '''
    -- The routine for querying one data entry 
    -- The index of must be specified as an argument
    -- Return a dictionary 
    '''
    def __getitem__(self, idx): 
        # Apply tokenizer
        boOutputs = self.boTokenizer.encode(self.boTexts[idx])
        enOutputs = self.enTokenizer.encode(self.enTexts[idx])
        
        # Truncation and padding 
        boIds, boMask = trim(
            boOutputs, 
            maxlen = self.boMaxLen, 
            pad_id = bo_pad_id, 
            enable_bos_eos = False
        )
        
        enIds, enMask = trim(
            enOutputs, 
            maxlen = self.enMaxLen, 
            pad_id = en_pad_id, 
            enable_bos_eos = True, 
            bos_id = en_pad_id,    # According to huggingface doc, target sequence is prepended by <pad> for T5  
            eos_id = en_eos_id
        )
        
        return {
            'source_ids': torch.tensor(boIds), 
            'source_mask': torch.tensor(boMask), 
            'target_ids': torch.tensor(enIds), 
            'target_mask': torch.tensor(enMask)
        }


# ## Define model class

# In[16]:


class T5FineTuner(pl.LightningModule): 
    ''' Part 1: Define the architecture of model in init '''
    def __init__(self, hparams):
        super(T5FineTuner, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(
            hparams['pretrainedModelName'], 
            return_dict = True    # I set return_dict true so that outputs  are presented as dictionaries
        )
        self.boTokenizer = hparams['boTokenizer']
        self.enTokenizer = hparams['enTokenizer']
        self.hparams = hparams
        self.scheduler_is_created = False
        self.epoch_counter = 0
        
        
    ''' Part 2: Define the forward propagation '''
    def forward(self, input_ids, attention_mask = None, decoder_input_ids = None, decoder_attention_mask = None, labels = None):  
        return self.model(
            input_ids, 
            attention_mask = attention_mask, 
            decoder_input_ids = decoder_input_ids, 
            decoder_attention_mask = decoder_attention_mask, 
            labels = labels
        )
    
    
    ''' Part 3: Configure optimizer and scheduler '''
    def configure_optimizers(self): 
        # Optimizer
        # I have no idea why to configure parameter this way 
        optimizer_grouped_parameters = [
            {
                # parameter with weight decay 
                'params': [param for name, param in model.named_parameters() if ('bias' not in name and 'LayerNorm.weight' not in name)], 
                'weight_decay': self.hparams['weight_decay'], 
            }, 
            {
                'params': [param for name, param in model.named_parameters() if ('bias' in name or 'LayerNorm.weight' in name)], 
                'weight_decay': 0.0, 
            }
        ]
        
        self.optimizer = AdamW(optimizer_grouped_parameters, lr = self.hparams['learning_rate'])
        
        # Scheduler
        # To create a scheduler with linear decay, we need to manually compute the number of training steps and pass it as an argument for the schduler 
        train_size = int(self.hparams['train_percentage'] * len(boTextsAll))
        batch_size = self.hparams['batch_size']
        num_processor = max(1, self.hparams['num_gpu'])
        num_epoch = self.hparams['num_train_epochs']
        total_training_steps = train_size // (batch_size * num_processor) * num_epoch
        
        # Create a scheduler for adjusting learning rate 
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer = self.optimizer, 
            num_warmup_steps = self.hparams['warmup_steps'], 
            num_training_steps = total_training_steps
        )
        
        self.lr_dict = {
            'scheduler': self.lr_scheduler, # The LR schduler
            'interval': 'step', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
        }
        
        # Do constant rate this time
        return [self.optimizer]# , [self.lr_dict]

    
    ''' Part 4.1: Training logic '''
    def training_step(self, batch, batch_idx):         
        loss = self._step(batch)
        self.log('train_loss', loss)
        # For monitoring purpose, log learning rate 
        for param_group in self.optimizer.param_groups:
            if param_group['lr']:
                self.log('learning_rate*e-4', param_group['lr'] * 1e4)
        return loss
    
    
    def _step(self, batch): 
        labels = batch['target_ids']
        # labels[labels[:, :] == en_pad_id] = -100
        # Explanation in huggingface doc: All labels set to -100 are ignored (masked), the loss is only computed for labels in [0, ..., config.vocab_size]
        
        outputs = self(
            input_ids = batch['source_ids'], 
            attention_mask = batch['source_mask'], 
            # decoder_input_ids = batch['target_ids'], 
            decoder_attention_mask = batch['target_mask'], 
            labels = labels
        )
        
        return outputs.loss

    
    ''' Part 4.2: Validation logic '''
    def validation_step(self, batch, batch_idx):        
        loss = self._step(batch)
        self.log('val_loss', loss)
        
        
    ''' Part 4.3: Test logic '''
    def test_step(self, batch, batch_idx): 
        loss = self._step(batch)
        self.log('test_loss', loss)
    
    
    ''' Part 5: Data loaders '''
    def _get_dataloader(self, start_idx, end_idx): 
        dataset = MyDataset(
            boTexts = boTextsAll[start_idx:end_idx], 
            enTexts = enTextsAll[start_idx:end_idx], 
            boTokenizer = self.hparams['boTokenizer'], 
            enTokenizer = self.hparams['enTokenizer'], 
            boMaxLen = self.hparams['max_input_len'], 
            enMaxLen = self.hparams['max_output_len']
        )
        
        return DataLoader(dataset, batch_size = hparams['batch_size'])
    
    
    def train_dataloader(self): 
        start_idx = 0
        end_idx = int(self.hparams['train_percentage'] * len(boTextsAll))
        return self._get_dataloader(start_idx, end_idx)
    
    
    def val_dataloader(self): 
        start_idx = int(self.hparams['train_percentage'] * len(boTextsAll))
        end_idx = int((self.hparams['train_percentage'] + self.hparams['val_percentage']) * len(boTextsAll))
        return self._get_dataloader(start_idx, end_idx)
    
    
    def test_dataloader(self): 
        start_idx = int((self.hparams['train_percentage'] + self.hparams['val_percentage']) * len(boTextsAll))
        end_idx = len(boTextsAll)
        return self._get_dataloader(start_idx, end_idx)
    
    
    ''' Part 6: hooks and callbacks '''
    def on_train_epoch_end(self, outputs): 
        start_idx = 0
        end_idx = 8

        testset = MyDataset(
            boTexts = boTextsAll[start_idx:end_idx], 
            enTexts = enTextsAll[start_idx:end_idx], 
            boTokenizer = self.hparams['boTokenizer'], 
            enTokenizer = self.hparams['enTokenizer'], 
            boMaxLen = self.hparams['max_input_len'], 
            enMaxLen = self.hparams['max_output_len']
        )

        test_dataloader = DataLoader(testset, batch_size = self.hparams['batch_size'])
        testit = iter(test_dataloader)

        # Take one batch from testset 
        batch = next(testit)

        # Generate target ids
        outs = self.model.generate(
            batch['source_ids'].cuda(), 
            attention_mask = batch['source_mask'].cuda(), 
            use_cache = True, 
            decoder_attention_mask = batch['target_mask'], 
            max_length = self.hparams['max_output_len'], 
            num_beams = 4, 
            repetition_penalty = 2.5, 
            length_penalty = 0.6, 
            early_stopping = True, 
        )
        
        pred_texts = [self.enTokenizer.decode(ids) for ids in (outs % enTokenizer.get_piece_size()).tolist()]    # temporary solution %
        source_texts = [self.boTokenizer.decode(ids) for ids in batch['source_ids'].tolist()]
        target_texts = [self.enTokenizer.decode(ids) for ids in batch['target_ids'].tolist()]
        
        file = open(f'./sample_outputs/epoch_{self.epoch_counter:2d}.txt', 'w', encoding = 'utf-8')

        for i in range(len(pred_texts)): 
            lines = textwrap.wrap("Tibetan Text:\n%s\n" % source_texts[i], width=100)
            file.write("\n".join(lines) + '\n')
            file.write(("\nActual translation: %s" % target_texts[i]) + '\n')
            file.write(("\nPredicted translation: %s" % pred_texts[i]) + '\n')
            file.write('=' * 50 + '\n' * 2)
            
        file.close()
        self.epoch_counter += 1


# In[17]:


hparams = {
    'boTokenizer': boTokenizer,
    'enTokenizer': enTokenizer,
    'pretrainedModelName': 't5-small', 
    'train_percentage': 0.95, 
    'val_percentage': 0.04, 
    'learning_rate': 1e-4, 
    'max_input_len': 50, 
    'max_output_len': 70, 
    'batch_size': 8, 
    'num_train_epochs': 10, 
    'num_gpu': 1, 
    'weight_decay': 0, 
    'warmup_steps': 0,  # For scheduler 
}


# ## Training

# In[ ]:


torch.cuda.empty_cache()

train_params = dict(
    gpus = hparams['num_gpu'], 
    max_epochs = hparams['num_train_epochs'], 
    progress_bar_refresh_rate = 20, 
)

model = T5FineTuner(hparams)

trainer = pl.Trainer(**train_params)

trainer.fit(model)

# Save model for later use
now = datetime.now()
trainer.save_checkpoint('04_t5simple_bo_en_' + now.strftime("%Y-%m-%d--%H=%M=%S") + '.ckpt')

trainer.test()


# ## Testing

# In[18]:


# Load a previously saved model

torch.cuda.empty_cache()

modelLoaded = T5FineTuner.load_from_checkpoint(checkpoint_path='__05_t5simple_bo_en_2020-12-16--02=06=12.ckpt').to(device)


# In[20]:


start_idx = 2000
end_idx = 2008

testset = MyDataset(
    boTexts = boTextsAll[start_idx:end_idx], 
    enTexts = enTextsAll[start_idx:end_idx], 
    boTokenizer = hparams['boTokenizer'], 
    enTokenizer = hparams['enTokenizer'], 
    boMaxLen = hparams['max_input_len'], 
    enMaxLen = hparams['max_output_len']
)

test_dataloader = DataLoader(testset, batch_size = hparams['batch_size'])
testit = iter(test_dataloader)

# Take one batch from testset 
batch = next(testit)

# Generate target ids
outs = modelLoaded.model.generate(
    batch['source_ids'].cuda(), 
    attention_mask = batch['source_mask'].cuda(), 
    use_cache = True, 
    decoder_attention_mask = batch['target_mask'], 
    max_length = hparams['max_output_len'], 
    bos_token_id = en_bos_id,
    eos_token_id = en_eos_id,
    pad_token_id = en_pad_id,
    num_beams = 4, 
    repetition_penalty = 2.5, 
    length_penalty = 0.6, 
    early_stopping = True, 
)

pred_texts = [enTokenizer.decode(ids) for ids in outs.tolist()]
source_texts = [boTokenizer.decode(ids) for ids in batch['source_ids'].tolist()]
target_texts = [enTokenizer.decode(ids) for ids in batch['target_ids'].tolist()]

for i in range(len(pred_texts)): 
    lines = textwrap.wrap("Tibetan Text:\n%s\n" % source_texts[i], width=100)
    print("\n".join(lines))
    print("\nActual translation: %s" % target_texts[i])
    print("\nPredicted translation: %s" % pred_texts[i])
    print('=' * 50 + '\n')


# In[ ]:


# %tensorboard --logdir lightning_logs/


# In[ ]:


x, y = trim(enTokenizer.encode('who would give these shavenheaded ascetics alms or think to help them'), maxlen = 100, pad_id = en_pad_id, enable_bos_eos=True, bos_id = en_bos_id, eos_id = en_eos_id)
enTokenizer.decode(x)


# In[ ]:




