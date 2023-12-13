from ToyTorch.data import DataLoader
from dataset_test import StockDataset

if __name__ == "__main__":
    print("="*80)
    print("Testing DataLoaders")
    print("expected output :")
    print("14")
    print("64, 60, 60) (64,)")
    x = StockDataset()
    def mdl (x):
        return DataLoader(x, batch_size=64, shuffle=True)
    train_dataloader = mdl(x)
    print(len(train_dataloader))
    print(train_dataloader[0][0].shape, train_dataloader[1][1].shape)
