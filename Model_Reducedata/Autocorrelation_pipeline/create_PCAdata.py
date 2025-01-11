import numpy as np
import pickle as pk
import os
import argparse




parser = argparse.ArgumentParser(description='Generate training, validation, testing data from  PCA model')
parser.add_argument('-train_filepath', type=str, default='../../High_Dimension_data/autocorrelation_data/train')
parser.add_argument('-valid_filepath', type=str, default='../../High_Dimension_data/autocorrelation_data/valid')
parser.add_argument('-test_filepath', type=str, default='../../High_Dimension_data/autocorrelation_data/test')
parser.add_argument('-time', type=int, default=80)
parser.add_argument('-PCA_components', type=int, default=500)
parser.add_argument('-PCA_path', type=str, default='PCA_model_PC=50')
parser.add_argument('-result_path', type=str, default='PCA_data_PC=50')


train_filelist = os.listdir(args.train_filepath)
valid_filelist = os.listdir(args.valid_filepath)
test_filelist = os.listdir(args.test_filepath)

modelname = os.path.join(args.PCA_path, 'pca_{}.pkl'.format(args.PCA_components)) 
pca_model = pk.load(open(modelname,"rb"))

PCA_train_data = np.zeros((len(train_filelist), args.time, args.PCA_components))
PCA_valid_data = np.zeros((len(valid_filelist), args.time, args.PCA_components))
PCA_test_data = np.zeros((len(test_filelist), args.time, args.PCA_components))

for i in range(len(train_filelist)):
    filename = os.path.join(args.train_filepath, train_filelist[i])
    ims=np.load(filename)['data'].astype(np.float32) 
    ims = ims.reshape((args.time,-1))
    ims=pca_model.transform(ims)
    PCA_train_data[i, :, :] = ims
    
arrayname = os.path.join(args.result_path, 'PCA_train_data.npz')
np.savez_compressed(arrayname, data=PCA_train_data)
del PCA_train_data

for i in range(len(valid_filelist)):
    filename = os.path.join(args.valid_filepath, valid_filelist[i])
    ims=np.load(filename)['data'].astype(np.float32) # (TL, 1, 64, 64)
    ims = ims.reshape((args.time,-1))
    ims=pca_model.transform(ims) #(TL, 25)
    PCA_valid_data[i, :, :] = ims
    
arrayname = os.path.join(args.result_path, 'PCA_valid_data.npz')
np.savez_compressed(arrayname, data=PCA_valid_data)
del PCA_valid_data

for i in range(len(test_filelist)):
    filename = os.path.join(args.test_filepath, test_filelist[i])
    ims=np.load(filename)['data'].astype(np.float32) # (TL, 1, 64, 64)
    ims = ims.reshape((args.time,-1))
    ims=pca_model.transform(ims) #(TL, 25)
    PCA_test_data[i, :, :] = ims

arrayname = os.path.join(args.result_path, 'PCA_test_data.npz')
np.savez_compressed(arrayname, data=PCA_test_data)

