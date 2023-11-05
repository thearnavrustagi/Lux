class Dataset(object):
    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self):
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")
