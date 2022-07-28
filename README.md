# MDD_Net
## Title
MDD-Net for Nonlinear Magnetization Signal Filtering in MPI
## Introduction
This is a simple version of testing code for MPI signal filtering task.
## Prepare test Dataset
Download test dataset to folder "./Dataset/"  
Dataset is provided in http:
## Prepare model.pth
Download well-trained model to folder "./Path/"  
well-trained model is provided in http:
## Test
1. simulated data  
set the 'args.test_set_path' and corresponding 'args.model' in test.py  
e.g. test_set_path: ./Dataset/gauss_5dB/test.h5  --  model: ./Path/gauss_5.pth  
run test.py  
2. measured data  
set the 'args.test_set_path' point to './Dataset/measured/test.h5' and corresponding 'args.model' point to './Path/fusion.pth' in test_measured.py
