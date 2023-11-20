import numpy as np 
import pandas as pd 

def batch_data(dataset, batch_size, drop_last, shuffle=False):
    batches = []
    index = 0
    for i in range(len(dataset)):
        x_val = np.squeeze(dataset[i:i+batch_size])
        batches.append(x_val)
    if drop_last == True and len(batches[-1]) < batch_size:
        batches = batches[:-1]
    print(len(batches))
    batches = np.array(batches)
    print(batches.shape)
    return batches

class DataLoader:
    def __init__ (self, dataset, batch_size = 1, shuffle  = False, num_workers = 0, drop_last = False):
        self.dataset = dataset 
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.dataloader = batch_data(dataset=dataset,batch_size=batch_size,shuffle=shuffle,
                                     drop_last=drop_last)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        if self.drop_last: 
            return len(self.dataloader) - 1 if len(self.dataloader) > 0 else 0 
        else: 
            return len(self.dataloader)
        
