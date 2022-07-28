# MDD_Net
## Title
MDD-Net for Nonlinear Magnetization Signal Filtering in MPI
## Introduction
This is a simple version of testing code for MPI signal filtering task.
## Prepare test Dataset
Download test dataset to folder "./data/"
Dataset is provided in http:
## Prepare model.pth
Download well-trained model to folder "./path/"
well-trained model is provided in http:
## Test
set the 'args.test_set_path' and corresponding 'args.model' in test.py
e.g. test_set_path: data/gauss_5dB/test.h5  --  model: path/gauss_5.pth
run test.py
