import os
from os import path
import glob
import math
import shutil
import random
import pandas as pd
import numpy as np
from scipy import signal
from PIL import Image
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms as T
import Huy.transforms as imgT

#------------------------------------------------------------------------------ 
#Initial Dataset
#------------------------------------------------------------------------------ 

def is_file(filename, exts):
    return any(filename.endswith(extension) for extension in exts)

def convert_label(trainYn, nb_classes):
    a,b,c,_= np.shape(trainYn)
    trainY = np.zeros((a,b,c,nb_classes))
    for i in range(0,a):
        trainY[i,:,:,0:4]= trainYn[i,:,:,0:4]
        trainY[i,:,:,4] = (trainYn[i,:,:,4] + trainYn[i,:,:,5] + trainYn[i,:,:,6] + trainYn[i,:,:,7])
    return trainY

def load_data(filelist):
    nb_classes = 5
    inputData = np.load(filelist[0])
    targetData = np.load(filelist[1])
    targetData = convert_label(targetData, nb_classes)
    return inputData, targetData

class DatasetFromFile(data.Dataset):
    """Maniplulate dataset before loading
    """
    def __init__(self, filelist=['input','target'], transform=None, islabel=True):
        super(DatasetFromFile, self).__init__()
        for filepath in filelist:
            if not path.isfile(filepath):
                raise ValueError(filepath, ' is not a file')
        self.filelist = filelist
        self.transform = transform
        self.islabel = islabel #often use for prediction
        self.inputData, self.targetData = load_data(filelist)
        self.len = self.inputData.shape[0]
            
    def __getitem__(self, index):
        input = self.inputData[index]
        input = input/255
        if self.islabel:
            target = self.targetData[index]
            if self.transform is not None:
                input, target = self.transform([input, target])
        else:
            target = -1
            input = self.transform(input)
        
        return input, target, index
        
    def __len__(self):
        return self.len
        
#------------------------------------------------------------------------------ 
#Do transforms
#------------------------------------------------------------------------------ 
def create_dark_mask():
    """create dark mask for adding black box 
    """
    black_dx = 60
    black_dy = 20
    dark_mask = np.zeros((black_dx, black_dy))
    for k in range(black_dy):
        dark_mask[:,k] = (np.abs(k-black_dy//2) / (black_dy/2.))**2
    return dark_mask

def create_elastic_indices():
    """create indices for elastic deformation
    used once at the start epoch
    """
    #initial values
    alpha, alpha2, sigma = 10, 15, 50
    shape = (480, 352) #same as shape of input images
    x_mesh, y_mesh = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    
    #below is used once per epoch for the elastic deformation
    g_1d = signal.gaussian(300, sigma)
    kernel_deform = np.outer(g_1d, g_1d)
    dx = signal.fftconvolve(np.random.rand(*shape) * 2 - 1, kernel_deform, mode='same')
    dy = signal.fftconvolve(np.random.rand(*shape) * 2 - 1, kernel_deform, mode='same')
    dx = alpha * (dx - np.mean(dx)) / np.std(dx)
    dy = alpha2 * (dy - np.mean(dy))/ np.std(dy)
    indices_x, indices_y = x_mesh+dx, y_mesh+dy
    indices_x_clipped = np.clip(indices_x, a_min=0, a_max=shape[1]-1)
    indices_y_clipped = np.clip(indices_y, a_min=0, a_max=shape[0]-1)
    return indices_x_clipped, indices_y_clipped

def train_transform(dark_mask, indices_x_clipped, indices_y_clipped):
    return imgT.EnhancedCompose([
        #random flip
        T.Lambda(imgT.randomlyFlip),
        #intensity nonlinear
        [T.Lambda(imgT.intensityNonliearShift), None],
        #add blackbox
        [imgT.AddBlackBox(dark_mask), None],
        #imgT.RandomRotate(),
        #elastic deformation
        imgT.ElasticDeformation(indices_x_clipped, indices_y_clipped),
        #multiplicative gauss
        [imgT.AddGaussian(ismulti=True), None],
        #additive gauss
        [imgT.AddGaussian(ismulti=False), None],
        # for non-pytorch usage, remove to_tensor conversion
        [T.Lambda(imgT.to_tensor), T.Lambda(imgT.to_tensor_target)]
        ])

def val_transform():
    return imgT.EnhancedCompose([
            # for non-pytorch usage, remove to_tensor conversion
            [T.Lambda(imgT.to_tensor), T.Lambda(imgT.to_tensor_target)]
        ])

#------------------------------------------------------------------------------ 
#Get method (used for DataLoader)
#------------------------------------------------------------------------------ 
def get_trainSet(train_filelist):
    dark_mask = create_dark_mask()
    indices_x_clipped, indices_y_clipped = create_elastic_indices()
    return DatasetFromFile(train_filelist, transform=train_transform(dark_mask, indices_x_clipped, indices_y_clipped))

def get_valSet(val_filelist):
    dark_mask = create_dark_mask()
    indices_x_clipped, indices_y_clipped = create_elastic_indices()
    return DatasetFromFile(val_filelist, transform= val_transform())#train_transform(dark_mask, indices_x_clipped, indices_y_clipped))

