# -*- coding: utf-8 -*-

import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from model12 import net
from os.path import join
from torch.utils import data
import os
import h5py

class DatasetFromFolder(data.Dataset):
    def __init__(self, train_set):
        super(DatasetFromFolder, self).__init__()
        self.train = train_set['data'][:]
        self.label = train_set['label'][:]

    def __getitem__(self, index):
        return self.train[index, :], self.label[index, :] 

    def __len__(self):
        return self.train.shape[0]

class Trainer(object):
    def __init__(self, args, train_dataLoader, val_dataLoader):
        super(Trainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = torch.device('cuda:2' if self.CUDA else 'cpu')
        self.lr = args.lr
        self.epochs = args.epochs
        self.train_dataLoader = train_dataLoader
        self.val_dataLoader = val_dataLoader
        self.val_best_loss = 500 * 1e-8

    def build_model(self):
        model_path = "model.pth"

        if os.path.exists(model_path):
            self.model = torch.load(model_path).to(self.device)
            print("Model loaded!")
        else:
            self.model = net(200).to(self.device)
            self.model = self.model.double()
            self.model.weight_init(0.0, 0.01)
        
        self.criterion = nn.MSELoss()

        if self.CUDA:
            cudnn.benchmark = True
            self.criterion.cuda()
    
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay=1e-9)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200, 400, 600], gamma=0.1)
        
    def save_model_val(self):
        model_out_path = join("model.pth")
        torch.save(self.model, model_out_path)
        print("model saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0.0
        alpha1 = 0.7
        alpha2 = 0.3
        for batch_num, (data, target) in enumerate((self.train_dataLoader)):
            data, target = data.to(self.device), target.to(self.device)
            data = data.unsqueeze(2)
            target = target.unsqueeze(2)

            fre_result, time_result = self.model(data)

            time_loss = alpha1 * self.criterion(time_result, target)
            fre_loss = alpha2 * self.criterion(fre_result, target)

            loss = time_loss + fre_loss
            self.optimizer.zero_grad()
            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        print("Average Loss: {:.4f}".format(train_loss / len(self.train_dataLoader)*1e8))
        return train_loss / len(self.train_dataLoader)

    def val(self):
        self.model.eval()
        val_loss = 0.0
        alpha1 = 1
        alpha2 = 0
   
        with torch.no_grad():
            for batch_num, (data, target) in enumerate(self.val_dataLoader):
                data, target = data.to(self.device), target.to(self.device)
                data = data.unsqueeze(2)
                target = target.unsqueeze(2)
                fre_result, time_result = self.model(data)

                loss = alpha1 * self.criterion(time_result, target) + alpha2 * self.criterion(fre_result, target)
                val_loss += loss.item()
        print("val Average Loss: {:.4f}".format(val_loss/len(self.val_dataLoader)*1e8))
        return val_loss / len(self.val_dataLoader)

    def run(self):
        self.build_model()
        print('==> model built!')
        for epoch in range(1, self.epochs + 1):
            print("\n===> model Epoch {} starts:".format(epoch))
            _ = self.train()
            val_loss = self.val()
            if val_loss < self.val_best_loss:
                self.val_best_loss = val_loss
                self.save_model_val() 
            self.scheduler.step()
        print("best val Average Loss: {:.8f}".format(self.val_best_loss*1e8))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set_path', type=str, required=True, help="Path to training source data")
    parser.add_argument('--val_set_path', type=str, required=True, help="Path to validating source data")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    data_file = h5py.File(args.train_set_path, 'r')
    train_data = DatasetFromFolder(data_file)
    train_dataLoader = data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)

    data_file = h5py.File(args.val_set_path, 'r')    
    val_data = DatasetFromFolder(data_file)
    val_dataLoader = data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)

    model = Trainer(args, train_dataLoader, val_dataLoader)
    model.run()
