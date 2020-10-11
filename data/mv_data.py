import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import make_column_transformer

ROOT = '/content/gdrive/'
MY_GOOGLE_DRIVE_PATH = 'My Drive/Capstone Project'

data_dir = ROOT + MY_GOOGLE_DRIVE_PATH + '/Edge/data/'
class MVDataset(Dataset):
    """ Pytorch dataset for Telescope """

    def __init__(self, csv_file = 'mv.dat', path = 'data/'):
        """
        constructor to load a csv, preprocess it into torch Dataset
        """

        self.dataset = pd.read_table(path + csv_file,skiprows=15,delimiter=",",header=None)
        self.dataset.columns = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6',
                     'X7', 'X8', 'X9', 'X10', 'Y']


        scaler = MinMaxScaler()
        data = scaler.fit_transform(self.dataset.iloc[:, :-1])
        target = np.asarray(self.dataset.Y)

        self.x = data.astype(np.float32)
        self.y = target.astype(np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

      sample_x = self.x[idx]
      sample_y = np.array([self.y[idx]])
      return (torch.from_numpy(sample_x), torch.from_numpy(sample_y))
