import numpy as np
import os
import random
import logging
import re
# create a logger for custom information
logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, input_param, configs, flag):
       self.paths = input_param['paths']
       self.name = input_param['name']
       self.input_data_type = input_param.get('input_data_type', 'float32')
       self.minibatch_size = input_param['minibatch_size']
       self.current_input_length = input_param['seq_length']
       self.current_position = 0
       self.current_batch_indices = []
       self.current_batch_size = 0
       self.dim = input_param['dim'] # need to be [PCs]
       self.configs = configs
       self.flag = flag
       if  flag =='valid' or flag == 'test':
           self.ms_path = input_param['ms_data_path']
       self.load()
       ## the paths would specify the numpy array file
    def load(self):
        self.data = np.load(self.paths[0])['data']
        
        if self.flag == 'test' or self.flag =='valid':
            # Identify and sort filenames dynamically
            filenames = os.listdir(self.ms_path[0])
            pattern = r'iter=(\d+)\.npz'  # Regular expression to match "iter=xxx.npz"
            
            # Extract numeric parts and sort
            title = sorted(
                int(re.search(pattern, f).group(1))
                for f in filenames
                if re.match(pattern, f)
            )

            # Generate the sorted list of file paths
            self.ms_data = [
                os.path.join(self.ms_path[0], f"iter={num}.npz")
                for num in title
            ]
            
        # get the un-separate separate list
            
    def total(self):
        return len(self.data) # 20 batchs
    
    
    def begin(self, do_shuffle = True):
        self.indices = np.arange(0, len(self.data))# index for each batch
        
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else: # divide into batch even total length can not divide into batch
            self.current_batch_size = self.total() - self.current_position
            
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        
        
        
    def next(self):
        self.current_position += self.current_batch_size
        if self.no_batch_left():
            return None
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
            
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]
        

    def no_batch_left(self):
        if self.current_position > self.total() - self.current_batch_size:
            return True
        else:
            return False
        
    def get_batch(self):
        if self.flag != 'test':
            if self.no_batch_left():
                logger.error(
                    "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
                return None
        input_batch = np.zeros(
            (self.current_batch_size, self.current_input_length) +
            tuple(self.dim)).astype(self.input_data_type)
        ## (B, TL, PCs)
        # initialize a zero batch tensor
        if self.flag == 'train' or self.flag =='valid':
            for i in range(self.current_batch_size):
                batch_ind = self.current_batch_indices[i]
                data_slice = self.data[batch_ind, :, :] # (TL, PCs)
                input_batch[i, :self.current_input_length, :] = data_slice
            input_batch = input_batch.astype(self.input_data_type)
            return input_batch
        
        elif self.flag =='test':
            
            input_ms = np.zeros(
                (self.current_batch_size, self.current_input_length) +
                (1, 256, 256)).astype(self.input_data_type)
            
            ### in this case, the batch would always be 1
            for i in range(self.current_batch_size):
                batch_ind = self.current_batch_indices[i]
                data_slice = self.data[batch_ind, :, :] # (TL, PCs)
                input_batch[i, :self.current_input_length, :] = data_slice
                batch_ms_path = self.ms_data[batch_ind]
                ms_slice = np.load(batch_ms_path)['data']
                input_ms[i, :self.current_input_length, :, :, :] = ms_slice
            input_ms = input_ms.astype(self.input_data_type)
            input_batch = input_batch.astype(self.input_data_type)
            return input_batch, input_ms
            
    
    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))
