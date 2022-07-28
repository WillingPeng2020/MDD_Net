# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 09:18:45 2021

@author: Administrator
"""
### test
import argparse
import numpy as np
import torch
from torch.utils import data
from train import DatasetFromFolder
from math import log10
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--test_set_path', type=str, required=False, default="data_5k\measured\test.h5")
parser.add_argument('--model', type=str, default="data_5k_model/model12_all.pth")
parser.add_argument('--batch_size', type=int, default=1)
args = parser.parse_args()

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

data_file = h5py.File(args.test_set_path, 'r')

test_data = DatasetFromFolder(data_file)
test_dataLoader = data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
model12 = torch.load(args.model).to(device)

with torch.no_grad():
    for step, (test_data, test_label) in enumerate(test_dataLoader):
        test_data, test_label= test_data.to(device), test_label.to(device)
        test_data = test_data.unsqueeze(2)
        test_label = test_label.unsqueeze(2)
        fre, time = model12(test_data)
        result = time.squeeze(2).cpu()
        if step == 0:
            time_result = np.array(result)
        else:
            time_result = np.append(time_result, np.array(result), axis=0)

test_data = data_file['data'][:]
label_data = data_file['label'][:]

snr_result = []
rmse_result = []
prd_result = []

# caculate SNR
for i in range(len(test_data)):

    noise_test = time_result[i] - label_data[i]
    temp = 10 * log10((np.sum(label_data[i] **2)) / ((np.sum(noise_test**2))))
    snr_result.append(temp)
    temp = 1e4 * np.linalg.norm(label_data[i] - time_result[i], ord=2)/(len(label_data[i])**0.5)
    rmse_result.append(temp)
    temp = 100 * (np.sum(np.linalg.norm(label_data[i] - time_result[i], ord=2)**2) / np.sum(np.linalg.norm(label_data[i], ord=2)**2))**0.5
    prd_result.append(temp)

print("  Average SNR of model: {:.4f} dB".format(np.sum(snr_result) / len(snr_result)))
print("  Average PRD of model: {:.4f} ".format(np.sum(prd_result) / len(prd_result)))
print("  Average RMSE of model: {:.4f} ".format(np.sum(rmse_result) / len(rmse_result)))
