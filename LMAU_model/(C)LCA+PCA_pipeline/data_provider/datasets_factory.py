import sys
import os
from data_provider import phase_field
import numpy as np

datasets_map = {
    'phase_field': phase_field}

def data_provider(dataset_name, train_data_paths, valid_data_paths, ms_data_path, batch_size,
                  img_width, seq_length, configs, is_training=True, PCs=64):
    
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    ## The split() method splits a string into a list.
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    ms_data_list = ms_data_path.split(',')
    if dataset_name == 'phase_field':
        dim = np.array([PCs])
        test_input_param = {'paths': valid_data_list,
                            'ms_data_path' : ms_data_list,
                            'minibatch_size': batch_size,
                            'input_data_type': 'float32',
                            'seq_length': seq_length,
                            'dim': dim,
                            'name': dataset_name + 'test iterator'}
        if is_training:
            flag ='valid'
        else:
            flag='test'
        test_input_handle = datasets_map[dataset_name].InputHandle(test_input_param, configs, flag=flag)
        test_input_handle.begin(do_shuffle=False)
        if is_training:
            train_input_param = {'paths': train_data_list,
                                 'minibatch_size': batch_size,
                                 'input_data_type': 'float32',
                                 'seq_length': seq_length,
                                 'dim': dim,
                                 'name': dataset_name + ' train iterator'}
            train_input_handle = datasets_map[dataset_name].InputHandle(train_input_param, configs, flag='train')
            train_input_handle.begin(do_shuffle=True)
            return train_input_handle, test_input_handle
        else:
            return test_input_handle
        
