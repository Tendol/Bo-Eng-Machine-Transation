import torch
import math
device = torch.device('cuda:0')

x = [[i] * 200 for i in range(100000)]

class MyBatchIterator: 
    def __init__(self, x, start_idx, end_idx, batch_size): 
        self.x = x 
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.batch_size = batch_size
    
    def __iter__(self):
        self.curr_idx = self.start_idx
        return self
    
    
    def __next__(self): 
        if self.curr_idx >= self.end_idx: 
            raise StopIteration  
        
        # Get text batch
        if self.curr_idx + self.batch_size < self.end_idx: 
            batch = self.x[self.curr_idx : self.curr_idx + self.batch_size]
            self.curr_idx += self.batch_size

        else:
            batch = self.x[self.curr_idx : self.end_idx]
            self.curr_idx = self.end_idx
        
        return torch.tensor(batch).to(device)
    

    def __len__(self):
        return math.ceil((self.end_idx - self.start_idx) / self.batch_size)
        


mbi = MyBatchIterator(x, start_idx = 0, end_idx = 20, batch_size = 8)
print('length of iterator:', len(mbi))

for idx, batch in enumerate(mbi): 
    print('size of current batch:', batch.size())