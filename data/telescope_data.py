import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import make_column_transformer

ROOT = '/content/gdrive/'
MY_GOOGLE_DRIVE_PATH = 'My Drive/Capstone Project'


class TelescopeDataset(Dataset):
    """ Pytorch dataset for Telescope """

    def __init__(self, csv_file = 'telescope.dat', path = 'data/'):
        """
        constructor to load a csv, preprocess it into torch Dataset
        """

        self.dataset = pd.read_table(path + csv_file, header=None, delimiter=',')
        self.dataset.columns = ['FLength', 'FWidth', 'FSize', 'FConc', 'FConc1', 'FAsym',
                     'FM3Long', 'FM3Trans', 'FAlpha', 'FDist', 'Class']

        scaler = StandardScaler()
        data = scaler.fit_transform(self.dataset.iloc[:, :-1])
        target = LabelEncoder().fit_transform(self.dataset.Class)
        self.x = data.astype(np.float32)
        self.y = target.astype(np.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

      sample_x = self.x[idx]
      sample_y = np.array([self.y[idx]])
      return (torch.from_numpy(sample_x), torch.from_numpy(sample_y))
