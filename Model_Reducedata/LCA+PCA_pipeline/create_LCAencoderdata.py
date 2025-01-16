import os
import torch
import numpy as np
import sys
sys.path.append('../../Autoencoder')
from model_autoencoder import LCA
from torch.utils.data import DataLoader
from auto_dataset import PhaseDataset
import argparse
import pickle as pk
import re
from utils import *
import fnmatch
parser = argparse.ArgumentParser(description='Generate training, validation, testing data from trained PCA and Autoencoder data (C-)LCA')
parser.add_argument('-train_filepath', type=str, default='../../High_Dimension_data/microstructure_data/train')
parser.add_argument('-valid_filepath', type=str, default='../../High_Dimension_data/microstructure_data/valid')
parser.add_argument('-test_filepath', type=str, default='../../High_Dimension_data/microstructure_data/test')
parser.add_argument('-PCA_path', type=str, default='Autoencoder_PCA_model')
parser.add_argument('-PCA_components', type=int, default=300)
parser.add_argument('-Autoencoder_dir', type=str, default='./LCA_model')
parser.add_argument('-Autoencoder_name', type=str, default='EncoderDecoder.pt.tar')
parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument('-time', type=int, default=80)
parser.add_argument('-width', type=int, default=256)
parser.add_argument('-height', type=int, default=256)
parser.add_argument('-channels', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=20)
parser.add_argument('-num_workers', type=int, default=2)
parser.add_argument('-result_path', type=str, default='C-LCA_PCA_data')

def get_files(filepath, pattern="iter=*.npz"):
    valid_files = [file for file in os.listdir(filepath) if fnmatch.fnmatch(file, pattern)]
    # Sort by the numeric part after "iter="
    return sorted(valid_files, key=lambda x: int(x.split('=')[1].split('.')[0]))

args = parser.parse_args()


train_samples = len(get_files(args.train_filepath))
valid_samples = len(get_files(args.valid_filepath))
test_samples = len(get_files(args.test_filepath))

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


train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.num_workers)

valid_loader = DataLoader(valid_dataset,
                          batch_size=args.batch_size,
                          shuffle=False,
                          num_workers=args.num_workers)

device = torch.device((args.device if torch.cuda.is_available() else 'cpu'))


model = LCA(in_channels=args.channels)
model.to(device)
filename = os.path.join(args.Autoencoder_dir, args.Autoencoder_name)
print("load model from"+filename)
stats = torch.load(filename, map_location=device)
model.load_state_dict(stats['model_param'])

### load the PCA model
modelname = os.path.join(args.PCA_path, 'pca_{}.pkl'.format(args.PCA_components))
pca_reload = pk.load(open(modelname,"rb"))

store = []
model.eval()
with torch.no_grad():
    for batchidx, inp in enumerate(train_loader):
        inp = inp.reshape((-1, args.channels, args.height, args.width))
        inp=inp.to(torch.float32).to(device)
        _, latent = model(inp) #input (BxT, C, H, W)
        # latent shape (BxT, 128, 8, 8)
        latent = latent.cpu().numpy()
        shape = latent.shape
        latent = latent.reshape((shape[0], -1)).astype(np.float32) #(BxT, 128x8x8)
        pc_trans = pca_reload.transform(latent) # (BxT, PCs)
        pc_trans = pc_trans.reshape((-1, args.time, args.PCA_components)) #(B,T,PCs)

        store.append(pc_trans)
        
store = np.concatenate(store) # (Total, T, Pcs)
save_direct(args.result_path)
train_data_name = os.path.join(args.result_path, 'train_data_{}.npz'.format(args.PCA_components))
np.savez(train_data_name, data=store)

print("train_data finish, begin with valid data")

store = []
with torch.no_grad():
    for batchidx, inp in enumerate(valid_loader):
        inp = inp.reshape((-1, args.channels, args.height, args.width))
        inp=inp.to(torch.float32).to(device)
        _, latent = model(inp) #input (BxT, C, H, W)
        # latent shape (BxT, 128, 8, 8)
        latent = latent.cpu().numpy()
        shape = latent.shape
        latent = latent.reshape((shape[0], -1)).astype(np.float32) #(BxT, 128x8x8)
        pc_trans = pca_reload.transform(latent) # (BxT, PCs)
        pc_trans = pc_trans.reshape((-1, args.time, args.PCA_components)) #(B,T,PCs)
        store.append(pc_trans)
        
store = np.concatenate(store)

valid_data_name = os.path.join(args.result_path, 'valid_data_{}.npz'.format(args.PCA_components))
np.savez(valid_data_name, data=store)

store = []
with torch.no_grad():
    inp = data_np
    inp = inp.reshape((-1, args.channels, args.height, args.width))
    inp=torch.from_numpy(inp).to(torch.float32).to(device)
    _, latent = model(inp) #input (BxT, C, H, W)
    # latent shape (BxT, 128, 8, 8)
    latent = latent.cpu().numpy()
    shape = latent.shape
    latent = latent.reshape((shape[0], -1)).astype(np.float32) #(BxT, 128x8x8)
    pc_trans = pca_reload.transform(latent) # (BxT, PCs)
    pc_trans = pc_trans.reshape((-1, args.time, args.PCA_components)) #(B,T,PCs)
    store=pc_trans
    
test_data_name = os.path.join(args.result_path, 'test_data_{}.npz'.format(args.PCA_components))
np.savez(test_data_name, data=store)

