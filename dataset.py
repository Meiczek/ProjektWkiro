import torch
from torch.utils.data import Dataset

class MocapDataset(Dataset):
    def __init__(self, data_file):
        import pickle
        with open(data_file, 'rb') as f:
            self.X = pickle.load(f)
        self.X = torch.tensor(self.X, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
