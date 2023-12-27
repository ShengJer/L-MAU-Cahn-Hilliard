import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pk
from numpy import linalg as LA
from sklearn.decomposition import PCA

######### Please change the parameters for here
samples=4500 # Total number of sequences in training dataset
Time = 80 # number of frames in a sequence
width = 256 # The 2D data Nx
height = 256 # the 2D data Ny
PCA_components = 50 # desired number of principal components
##########

auto_shape = (samples,
              Time,
              width,
              height)

def save_direct(direct_name):
    Current_directory = os.getcwd()
    directory = os.path.join(Current_directory, direct_name)
    if not os.path.exists(directory):
          os.makedirs(directory)
    return directory

current = os.getcwd()
datapath = os.path.join(current, 'auto_data')
train_datapath = os.path.join(datapath, r'auto_train_data')
train_filelist = os.listdir(train_datapath)



auto = np.zeros(auto_shape, dtype=np.float32)
for i in range(1, samples+1):
    filename = os.path.join(train_datapath, train_filelist[i-1])
    data = np.load(filename)['data']
    auto[i-1, :, :, :]=data.astype(np.float32)        


pca = PCA(
    svd_solver='full',
    n_components=PCA_components)

PCA_trans = pca.fit_transform(auto.reshape((samples)*Time,width*height))

PC_plot = PCA_trans.reshape((samples, Time, PCA_components))

exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
print("variance={}".format(cum_sum_eigenvalues[-1]))

model_direct=save_direct(direct_name='PCA_model')
filename = os.path.join(model_direct, 'PCA_variance.png')
plt.plot(range(1, len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, marker='.')
plt.xlabel("principle components")
plt.ylabel("variance")
plt.savefig(filename)
plt.close()


modelname = os.path.join(model_direct, 'pca_{}.pkl'.format(PCA_components))
pk.dump(pca, open(modelname,"wb"))
var_name = os.path.join(model_direct, 'cum_var.npy')
np.save(var_name, cum_sum_eigenvalues)


