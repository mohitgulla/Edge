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


class CaliforniaDataset(Dataset):
    """ Pytorch dataset for California """

    def __init__(self, csv_file = 'california.dat', path = 'data/'):
        """
        constructor to load a csv, preprocess it into torch Dataset
        """

        self.dataset = pd.read_table(path + csv_file, skiprows=13,delimiter=",",header=None)
        self.dataset.columns =  ['Longitude', 'Latitude', 'HousingMedianAge', 'TotalRooms', 'TotalBedrooms', 'Population',
                     'Households', 'MedianIncome', 'MedianHouseValue']




        self.dataset.MedianHouseValue/=(1000)
        low = .05
        high = .95
        quant_california = self.dataset.MedianHouseValue.quantile([low, high])
        print(quant_california)
        self.dataset = self.dataset[(self.dataset.MedianHouseValue>=quant_california[0.05]) & (self.dataset.MedianHouseValue<=quant_california[0.95] )]
        scaler = MinMaxScaler()

        data = scaler.fit_transform(self.dataset.iloc[:, 2:-1])
        target = np.asarray(self.dataset.MedianHouseValue)
        self.x = data.astype(np.float32)
        self.y = target.astype(np.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

      sample_x = self.x[idx]
      sample_y = np.array([self.y[idx]])
      return (torch.from_numpy(sample_x), torch.from_numpy(sample_y))
