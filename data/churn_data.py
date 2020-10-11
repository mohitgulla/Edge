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


class ChurnDataset(Dataset):
    """ Pytorch dataset for Churn """

    def __init__(self, csv_file = 'churn.csv', path = 'data/'):
        """
        constructor to load a csv, preprocess it into torch Dataset
        """
        self.dataset = pd.read_csv(os.path.join(path, csv_file))
        self.dataset.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

        preprocess = make_column_transformer((OneHotEncoder(), ['Geography', 'Gender']), remainder = StandardScaler())
        data = preprocess.fit_transform(self.dataset.iloc[:, :-1])
        target = np.array(self.dataset.iloc[:, -1])
        self.x = data.astype(np.float32)
        self.y = target.astype(np.long)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

      sample_x = self.x[idx]
      sample_y = np.array([self.y[idx]])
      # return (sample_x, sample_y)
      return (torch.from_numpy(sample_x), torch.from_numpy(sample_y))
