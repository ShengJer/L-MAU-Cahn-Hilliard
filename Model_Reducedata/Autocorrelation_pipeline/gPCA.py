import numpy as np
import os
import matplotlib.pyplot as plt
import pickle as pk
from numpy import linalg as LA
from sklearn.decomposition import PCA
import argparse

parser = argparse.ArgumentParser(description='Generate PCA model from Autocorrelation data')

parser.add_argument('-train_filepath', type=str, default='../../High_Dimension_data/autocorrelation_data/train')
parser.add_argument('-PCA_components', type=int, default=50)
parser.add_argument('-time', type=int, default=80)
parser.add_argument('-width', type=int, default=256)
parser.add_argument('-height', type=int, default=256)
parser.add_argument('-PCA_path', type=str, default='PCA_model_PC=50')
args = parser.parse_args()



samples=len(os.listdir(args.train_filepath))
auto_shape = (samples,
              args.time,
              args.width,
              args.height)

def save_direct(direct_name):
    Current_directory = os.getcwd()
    directory = os.path.join(Current_directory, direct_name)
    if not os.path.exists(directory):
          os.makedirs(directory)

print("start building PCA model")

train_filelist = os.listdir(args.train_filepath)



auto = np.zeros(auto_shape, dtype=np.float32)
for i in range(samples):
    filename = os.path.join(args.train_filepath, train_filelist[i])
    data = np.load(filename)['data']
    auto[i, :, :, :]=data.astype(np.float32)        


pca = PCA(
    #svd_solver='full',
    n_components=args.PCA_components)

PCA_trans = pca.fit_transform(auto.reshape((samples)*args.time,args.width*args.height))

PC_plot = PCA_trans.reshape((samples, args.time, args.PCA_components))

exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
print("variance={}".format(cum_sum_eigenvalues[-1]))

save_direct(direct_name=args.PCA_path)
filename = os.path.join(args.PCA_path, 'PCA_variance.png')
plt.plot(range(1, len(cum_sum_eigenvalues)+1), cum_sum_eigenvalues, marker='.')
plt.xlabel("principle components")
plt.ylabel("variance")
plt.savefig(filename)
plt.close()


modelname = os.path.join(args.PCA_path, 'pca_{}.pkl'.format(args.PCA_components))
pk.dump(pca, open(modelname,"wb"))
var_name = os.path.join(args.PCA_path, 'cum_var.npy')
np.save(var_name, cum_sum_eigenvalues)


