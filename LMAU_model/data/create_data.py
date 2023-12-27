import numpy as np
import pickle as pk
import os

train_directory = r'/home/r10524004/auto_data/auto_train_data_store'
valid_directory = r'/home/r10524004/auto_data/auto_valid_data_store'
test_directory = r'/home/r10524004/auto_data/auto_test_data_store'
PCA_dir = r'/home/r10524004/sMAU/Autocorrelation/PCA_model'
PCA_data_dir = r'/home/r10524004/sMAU/Autocorrelation/PCA_data'

train_filelist = os.listdir(train_directory)
valid_filelist = os.listdir(valid_directory)

modelname = os.path.join(PCA_dir, 'pca_50.pkl') 
pca_model = pk.load(open(modelname,"rb"))

Time = 80
PCs = 50

PCA_train_data = np.zeros((len(train_filelist), Time, PCs))
PCA_valid_data = np.zeros((len(valid_filelist), Time, PCs))
PCA_test_data = np.zeros((10, Time, PCs))

for i in range(len(train_filelist)):
    filename = os.path.join(train_directory, train_filelist[i])
    ims=np.load(filename)['data'].astype(np.float32) # (TL, 1, 64, 64)
    ims = ims.reshape((80,-1))
    ims=pca_model.transform(ims) #(TL, 50)
    PCA_train_data[i, :, :] = ims
    
arrayname = os.path.join(PCA_data_dir, 'PCA_train_data.npz')
np.savez_compressed(arrayname, data=PCA_train_data)
del PCA_train_data
print("train_data finish !!")

for i in range(len(valid_filelist)):
    filename = os.path.join(valid_directory, valid_filelist[i])
    ims=np.load(filename)['data'].astype(np.float32) # (TL, 1, 64, 64)
    ims = ims.reshape((80,-1))
    ims=pca_model.transform(ims) #(TL, 25)
    PCA_valid_data[i, :, :] = ims
    
arrayname = os.path.join(PCA_data_dir, 'PCA_valid_data.npz')
np.savez_compressed(arrayname, data=PCA_valid_data)
del PCA_valid_data

for idx, name in enumerate(range(4919, 4929)):
    filename = os.path.join(test_directory, 'auto_iter={}'.format(name)+'.npz')
    ims=np.load(filename)['data'].astype(np.float32)
    ims = ims.reshape((80,-1))
    ims=pca_model.transform(ims)
    PCA_test_data[idx, :, :] = ims

arrayname = os.path.join(PCA_data_dir, 'PCA_test_data.npz')
np.savez_compressed(arrayname, data=PCA_test_data)

