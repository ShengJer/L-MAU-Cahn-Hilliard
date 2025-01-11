import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append('../../Autoencoder')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from model_autoencoder import LCA
from torch.utils.data import DataLoader
from auto_dataset import PhaseDataset
from utils import *
import pickle as pk
import os
from numpy import linalg as LA
import re
import argparse


parser = argparse.ArgumentParser(description='Generate PCA model from trained Autoencoder')

parser.add_argument('-train_filepath', type=str, default='../../High_Dimension_data/microstructure_data/train')
parser.add_argument('-test_filepath', type=str, default='../../High_Dimension_data/microstructure_data/test')
parser.add_argument('-PCA_components', type=int, default=500)
parser.add_argument('-Autoencoder_dir', type=str, default='./data/Autoencoder_model')
parser.add_argument('-Autoencoder_name', type=str, default='EncoderDecoder.pt.tar')
parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument('-time', type=int, default=80)
parser.add_argument('-width', type=int, default=256)
parser.add_argument('-height', type=int, default=256)
parser.add_argument('-channels', type=int, default=1)
parser.add_argument('-latent_width', type=int, default=8)
parser.add_argument('-latent_height', type=int, default=8)
parser.add_argument('-latent_channel', type=int, default=128)
parser.add_argument('-batch_size', type=int, default=10) # batchsize * time = true sample to the autoencoder
parser.add_argument('-num_workers', type=int, default=8)
parser.add_argument('-PCA_path', type=str, default='Autoencoder_PCA_model')
parser.add_argument('-graph_path', type=str, default='Reconstruction')
args = parser.parse_args()


train_samples = len(os.listdir(args.train_filepath))
valid_samples = len(os.listdir(args.valid_filepath))
test_samples = len(os.listdir(args.test_filepath))

# Load training and validation dataset
train_dataset = PhaseDataset(data_filepath=args.train_filepath, config=args, number_of_samples=train_samples)
valid_dataset = PhaseDataset(data_filepath=args.valid_filepath, config=args, number_of_samples=valid_samples)


# Load testing dataset in order
# Identify and sort filenames dynamically
filenames = os.listdir(args.test_filepath)
pattern = r'iter=(\d+)\.npz'  # Regular expression to match "iter=xxx.npz"
# Extract numeric parts and sort
title = sorted(
    int(re.search(pattern, f).group(1))
    for f in filenames
    if re.match(pattern, f)
)

data_np = np.zeros((test_samples, args.time, args.channels, args.width, args.height))
for i in range(len(title)):
    filename = os.path.join(args.test_filepath, 'iter='+str(title[i])+'.npz')
    data_np[i,:,:,:,:]=np.load(filename)['data']




# create dataloader
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.num_workers)

model = LCA(in_channels=args.channels)
device = torch.device(args.device if torch.cuda.is_available() else 'cpu')


# load the model:
model.to(device)
filename = os.path.join(args.Autoencoder_dir, args.Autoencoder_name)
print("load model from"+filename)
stats = torch.load(filename, map_location=device)
model.load_state_dict(stats['model_param'])

data_latent = []
model.eval()
with torch.no_grad():
    for batchidx, inp in enumerate(train_loader):
        ## reshape the input tensor into desire shape
        inp = inp.reshape((-1, args.channels, args.height, args.width))
        inp=inp.to(torch.float32).to(device)
        pred, latent = model(inp)            
        
        inp = inp.cpu().numpy()
        pred = pred.cpu().numpy()
        latent =latent.cpu().numpy()
        
        data_latent.append(latent) # [(BxT, 128, 8, 8)]
    
    data_latent = np.concatenate(data_latent) # [SxT, 128, 8, 8]
    shape = data_latent.shape 
    data_latent=data_latent.reshape((shape[0], -1)).astype(np.float32) # [SxT, 128x8x8]
    
    ## construct the PCA model
    pca = PCA(
        svd_solver='full',
        n_components=args.PCA_components)
    
    PCA_trans = pca.fit_transform(data_latent) # (SxT, PCA)
    PC_plot = PCA_trans.reshape((-1, args.time, args.PCA_components)) # (S, T, PCA)
    
    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    print("variance={}".format(cum_sum_eigenvalues[-1]))
    dir_name = os.path.join(args.results_path)
    result_dir = save_direct(dir_name)
    modelname = os.path.join(result_dir, 'pca_{}.pkl'.format(args.PCA_components))
    pk.dump(pca, open(modelname,"wb"))
    var_name = os.path.join(result_dir, 'cum_var_ms.npy')
    np.save(var_name, cum_sum_eigenvalues)

### plot the 3D plot
for i in np.arange(0, 360, 30):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    
    for j in range(len(PC_plot)):
        ax.scatter(PC_plot[j, :, 0], PC_plot[j, :, 1], PC_plot[j, :, 2], marker='.')
    
    ax.set_xlabel(r"$\varphi_{1}$", style='italic', fontsize=12)
    ax.set_ylabel(r"$\varphi_{2}$", style='italic', fontsize=12)
    ax.set_zlabel(r"$\varphi_{3}$", style='italic', fontsize=12)
    ax.tick_params(axis='both', which='both', labelsize=12)
    
    ax.view_init(azim=i, elev=23)
    
    filename = os.path.join(result_dir, "3D_Auto_azim={}_ms.png".format(i))
    fig.savefig(filename)
    plt.close()

dir_name = os.path.join(args.results_path)
result_dir = save_direct(dir_name)
modelname = os.path.join(result_dir, 'pca_{}.pkl'.format(args.PCA_components))

## load PCA model
pca_reload = pk.load(open(modelname,"rb"))
reconstruction_loss = 0


### iterate the test dataset to see if the Reconstruction work well
inp = data_np
inp = inp.reshape((-1, args.channels, args.height, args.width))
inp=torch.from_numpy(inp).to(torch.float32).to(device)

dir_decode, latent = model(inp)
latent =latent.cpu().numpy() # (SxT, 128, 8, 8)
shape = latent.shape
latent = latent.reshape((shape[0], -1)).astype(np.float32) #(SxT, 128x8x8)

pc_trans = pca_reload.transform(latent) #(SxT, PCA)
pc_projected = pca_reload.inverse_transform(pc_trans) #(SxT, 128x8x8)

PD_data = pc_projected.reshape(latent.shape) #(SxT, 128,8,8)

total_loss=LA.norm(latent-pc_projected, None) 

reconstruction_loss += total_loss
PD_data = PD_data.reshape((-1, args.latent_channel, args.width, args.height))
PD_data=torch.from_numpy(PD_data).to(torch.float32).to(device)
pred = model.decoder(PD_data) # (SxT, 1, 256, 256)

inp = inp.reshape((len(data_np), args.time, args.height, args.width))
pred = pred.reshape(((len(data_np), args.time, args.height, args.width)))
dir_decode = dir_decode.reshape(((len(data_np), args.time, args.height, args.width)))
inp = inp.cpu().numpy()
pred = pred.cpu().numpy()
dir_decode =dir_decode.cpu().numpy()


for i in range(len(data_np)):
    dir_name = os.path.join(args.graph_path, str(i+1))
    graph_dir = save_direct(dir_name)
    x = np.arange(0, inp.shape[2])
    y = np.arange(0, inp.shape[3])
    X, Y = np.meshgrid(x, y)
    rangeGT = [10, 20 ,30 ,40 ,50, 60, 70, 79]
    for j in rangeGT:
        name = 'gt' + str(j) + '.png'
        file_name = os.path.join(graph_dir, name)
        counter_set = plt.contourf(X, Y, inp[i, j, :, :], levels=np.linspace(0, 1, 30))
        plt.colorbar(counter_set, label='$\phi_{p}$')
        plt.savefig(file_name)
        plt.close()
    for j in rangeGT:
        name = 'PCA+decode' + str(j) + '.png'
        file_name = os.path.join(graph_dir, name)
        counter_set = plt.contourf(X, Y, pred[i, j, :, :], levels=np.linspace(0, 1, 30))
        plt.colorbar(counter_set, label='$\phi_{p}$')
        plt.savefig(file_name)
        plt.close()
    for j in rangeGT:
        name = 'Dir-decode' + str(j) + '.png'
        file_name = os.path.join(graph_dir, name)
        counter_set = plt.contourf(X, Y, dir_decode[i, j, :, :], levels=np.linspace(0, 1, 30))
        plt.colorbar(counter_set, label='$\phi_{p}$')
        plt.savefig(file_name)
        plt.close()
        
## save the data
data_filename = os.path.join(args.graph_path, 'Reconstruct_data_compressed.npz')
np.savez_compressed(data_filename, inp=inp, dir_decode=dir_decode, PCA_decode=pred)

















