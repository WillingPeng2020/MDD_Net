# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:18:45 2021

@author: Administrator
"""

import argparse
import numpy as np
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import h5py


class DatasetFromFolder(data.Dataset):
    def __init__(self, train_set):
        super(DatasetFromFolder, self).__init__()
        self.train = train_set['data'][:]

    def __getitem__(self, index):
        return self.train[index, :] 

    def __len__(self):
        return self.train.shape[0]

parser = argparse.ArgumentParser()
parser.add_argument('--test_set_path', type=str, required=True, help="Path to testing source measured data")
parser.add_argument('--model', type=str, required=True, help="Path to saved model parameters")
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.switch_backend('agg')

data_file = h5py.File(args.test_set_path, 'r')
test_data = DatasetFromFolder(data_file)
test_dataLoader = data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
model = torch.load(args.model).to(device)

with torch.no_grad():
    for step, test_data in enumerate(test_dataLoader):
        test_data= test_data.to(device)
        test_data = test_data.unsqueeze(2)
        fre, time = model(test_data)
        result = time.squeeze(2).cpu()
        if step == 0:
            time_result = np.array(result)
        else:
            time_result = np.append(time_result, np.array(result), axis=0)

    
