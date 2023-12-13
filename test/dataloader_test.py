from ToyTorch.data import DataLoader
from dataset_test import StockDataset

if __name__ == "__main__":
    x = StockDataset()
    def mdl (x):
        return DataLoader(x, batch_size=64, shuffle=True)
    train_dataloader = mdl(x)
    for x in train_dataloader:
        print(x)
