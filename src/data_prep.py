import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import glob
import os
from typing import Optional

class Dataset4DFlowNet(Dataset):
    # constructor
    def __init__(self,
                 data_dir: str = './data/demo_geo_model.npy',
                 ):
        self.data = np.load(data_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        pos, vol = self.data[idx, :4], self.data[idx, 4:]
        pos = torch.from_numpy(pos)
        vol = torch.from_numpy(vol)
        pos = pos.type(torch.FloatTensor)
        vol = vol.type(torch.FloatTensor)

        return (pos, vol)

if __name__ == "__main__":
    dataset = Dataset4DFlowNet()
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=len(dataset))
    print(len(dataloader))
    
