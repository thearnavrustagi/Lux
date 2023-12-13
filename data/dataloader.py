from ToyTorch import Tensor

import numpy as np
import pandas as pd

from math import ceil, floor


class DataLoader:
    def __init__(
        self, dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False
    ):
        if num_workers < 0:
            raise ValueError("Num workers should be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last

        self.idx = 0
        self.dset_len = len(self.dataset)
        self.len = self.dset_len/batch_size
        self.len = floor(self.len) if self.drop_last else ceil(self.len)
        self.dtype = type(self.dataset[0])

    def __iter__(self):
        idx = self.idx
        while True:
            if idx == self.len: break
            yield self[idx]
            idx += 1

    def __getitem__(self,idx):
        if self.dtype == tuple:
            rets = [[] for _ in range(len(self.dataset[0]))]
            for i in range(self.batch_size):

                didx = (idx*self.batch_size + i)
                # if the dataloader reaches an immature end
                if didx >= self.dset_len:
                    break
                
                dataset_item = self.dataset[didx]
                for i,(item,data) in enumerate(zip(rets, dataset_item)):
                    rets[i].append(data)
            return tuple(np.array(i) for i in rets)

    def __len__(self):
        return self.len

