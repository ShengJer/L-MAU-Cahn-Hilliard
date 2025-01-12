import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random

class PhaseDataset(Dataset):
    def __init__(self, data_filepath, config, number_of_samples):
        self.data_filepath = data_filepath # the data set path to data file
        self.time = config.time # data time
        self.width = config.width # data size
        self.height = config.height # data size
        self.number_of_samples = number_of_samples
        self.all_filelist = self.get_all_filelist() # get the filelist from the directory
        self.samples = len(self.all_filelist) # the length of file list
        
    def get_all_filelist(self):
        filelist = []
        # get the video ids based on training or testing data
        filelist = os.listdir(self.data_filepath)
        filelist = random.sample(filelist, self.number_of_samples)
        return filelist
     
    def __len__(self):
        return len(self.all_filelist)
    
    def __getitem__(self, idx):
        filename = os.path.join(self.data_filepath, self.all_filelist[idx])
        all_seq = np.load(filename)['data'] # (time, 256, 256) # the data need to be autocorrelation data
        
        return all_seq

        # output shape by data loader (B, time, 256, 256)
