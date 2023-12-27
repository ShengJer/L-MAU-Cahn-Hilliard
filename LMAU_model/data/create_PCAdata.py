import numpy as np
import pickle as pk
import os

############## those parameters can be change
Time = 80
PCs = 50
train_directory = r'./auto_train_data'
valid_directory = r'./auto_valid_data'
test_directory = r'./auto_test_data'
PCA_dir = r'./PCA_model'
PCA_data_dir = r'./PCA_data'
##############

train_filelist = os.listdir(train_directory)
valid_filelist = os.listdir(valid_directory)
test_filelist = os.listdir(test_directory)

modelname = os.path.join(PCA_dir, 'pca_{}.pkl'.format(PCs)) 
pca_model = pk.load(open(modelname,"rb"))

PCA_train_data = np.zeros((len(train_filelist), Time, PCs))
PCA_valid_data = np.zeros((len(valid_filelist), Time, PCs))
PCA_test_data = np.zeros((len(test_filelist), Time, PCs))

for i in range(len(train_filelist)):
    filename = os.path.join(train_directory, train_filelist[i])
    ims=np.load(filename)['data'].astype(np.float32) 
    ims = ims.reshape((Time,-1))
    ims=pca_model.transform(ims)
    PCA_train_data[i, :, :] = ims
    
arrayname = os.path.join(PCA_data_dir, 'PCA_train_data.npz')
np.savez_compressed(arrayname, data=PCA_train_data)
del PCA_train_data

for i in range(len(valid_filelist)):
    filename = os.path.join(valid_directory, valid_filelist[i])
    ims=np.load(filename)['data'].astype(np.float32) # (TL, 1, 64, 64)
    ims = ims.reshape((Time,-1))
    ims=pca_model.transform(ims) #(TL, 25)
    PCA_valid_data[i, :, :] = ims
    
arrayname = os.path.join(PCA_data_dir, 'PCA_valid_data.npz')
np.savez_compressed(arrayname, data=PCA_valid_data)
del PCA_valid_data

for i in range(len(test_filelist)):
    filename = os.path.join(test_directory, test_filelist[i])
    ims=np.load(filename)['data'].astype(np.float32) # (TL, 1, 64, 64)
    ims = ims.reshape((Time,-1))
    ims=pca_model.transform(ims) #(TL, 25)
    PCA_test_data[i, :, :] = ims

arrayname = os.path.join(PCA_data_dir, 'PCA_test_data.npz')
np.savez_compressed(arrayname, data=PCA_test_data)

