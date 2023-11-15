def DataLoaderFunc(dataset, batch_size, drop_last, shuffle=False):
    new_arr = np.ndarray()
    index = 0
    for x in ((len(dataset)//batch_size) + 1):
        for i in batch_size:
            a = np.ndarray()
            np.append(a, dataset[index])
            index += 1
        np.append(new_arr, a)
    if drop_last == True and len(new_arr[-1]) < batch_size:
        new_arr = new_arr[:-1]
    return new_arr

class CustomDataLoader:
    def __init__ (self, dataset, batch_size = 1, shuffle  = False, num_workers = 0, drop_last = False):
        self.dataset = dataset 
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.dataloader = DataLoaderFunc(dataset=dataset,batch_size=batch_size,shuffle=shuffle,
        num_workers=num_workers,drop_last=drop_last)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        if self.drop_last: 
            return len(self.dataloader) - 1 if len(self.dataloader) > 0 else 0 
        else: 
            return len(self.dataloader)
