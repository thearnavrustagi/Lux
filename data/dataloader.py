class DataLoader (object):
     def __iter__(self):
        raise NotImplementedError("Subclasses of Dataloader should implement __iter__.")

    def __len__(self):
        raise NotImplementedError("Subclasses of Dataloader should implement __len__.")
