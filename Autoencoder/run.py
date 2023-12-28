import os
import torch
import logging
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from model_autoencoder import HCA, LCA
from torch.utils.data import DataLoader
from auto_dataset import PhaseDataset
import argparse
from utils import *
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s %(message)s", datefmt="[%Y-%m-%d %H:%M:%S]")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='PyTorch - Autoencoder')

parser.add_argument('-train_filepath', type=str, default='./data/train_data')

parser.add_argument('-ckpt_path', type=str, default='ckpt_path')
parser.add_argument('-graph_path', type=str, default='graph_path')

parser.add_argument('-load_model', type=int, default=0)
parser.add_argument('-time', type=int, default=80)
parser.add_argument('-width', type=int, default=256)
parser.add_argument('-height', type=int, default=256)
parser.add_argument('-channels', type=int, default=1)
parser.add_argument('-batch_size', type=int, default=3) 
parser.add_argument('-num_workers', type=int, default=8) 
parser.add_argument('-model_name', type=str, default='LCA')
parser.add_argument('-num_epoch', type=int, default=1000)
parser.add_argument('-lr', type=float, default=0.001)
parser.add_argument('-step_size', type=int, default=50) 
parser.add_argument('-gamma', type=float, default=0.8)
parser.add_argument('-alpha', type=float, default=10.0)

parser.add_argument('-device', type=str, default='cuda:0')
parser.add_argument('-train_samples', type=int, default=4500)
parser.add_argument('-display_epoch', type=int, default=20)
parser.add_argument('-valid_epoch', type=int, default=50)

args = parser.parse_args()

# Load dataset
train_dataset = PhaseDataset(data_filepath=args.train_filepath, config=args)

# create dataloader
train_loader = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          shuffle=True,
                          num_workers=args.num_workers)

if args.model_name == 'LCA':
    model = LCA(in_channels=args.channels)
elif args.model_name == 'HCA':
    model = HCA(in_channels=args.channels)
    
# optimizer, scheduler and criterion
optimizer = Adam(model.parameters(), lr=args.lr)

scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

criterionL1 = nn.L1Loss()

criterionL2 = nn.MSELoss()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

def train(args, model):

    model.to(device)
    
    # training:
    if args.load_model == 1:
        if args.model_name == 'LCA':
            Current_directory = os.getcwd()
            checkpoint_path = os.path.join(Current_directory, args.ckpt_path)
            filename = os.path.join(checkpoint_path, 'LCA.pt.tar')
            print("load model from"+filename)
            stats = torch.load(filename)
            model.load_state_dict(stats['model_param'])
        elif args.model_name == 'HCA':
            Current_directory = os.getcwd()
            checkpoint_path = os.path.join(Current_directory, args.ckpt_path)
            filename = os.path.join(checkpoint_path, 'HCA.pt.tar')
            print("load model from"+filename)
            stats = torch.load(filename)
            model.load_state_dict(stats['model_param'])
    
    best_valid_loss = 1e+09
    epoch=0
    train_hist = []
    valid_hist = []
    lr_hist = []
    for epoch in range(1, args.num_epoch+1):
        step_loss=0
        for batchidx, inp in enumerate(train_loader):
            # the output would be (B, time, H, W)
            inp = inp.reshape((-1, args.channels, args.height, args.width)) # inp(BxT, 1, H, W)
            
            ####If you want to insert mass conservation constraint, please uncomment the line below:
                # inp_avg = torch.mean(inp, dim=(2,3)).to(torch.float32).to(device)
            
            model.train()
            optimizer.zero_grad()
            inp=inp.to(torch.float32).to(device)
            pred, _ = model(inp) # would be two output one is reconstruction other is latent space
            
            
            ####If you want to insert mass conservation constraint, please uncomment the two lines below:
                # pred_avg = torch.mean(pred, dim=(2,3))
                # mse_loss_avg = criterionL2(pred_avg, inp_avg)
            
            mae_loss = criterionL1(pred, inp)
            
            ####If you want to insert mass conservation constraint, please change the loss function to this:
                # total_loss = mae_loss + args.alpha*mse_loss_avg

            total_loss = mae_loss
            step_loss += total_loss.detach().cpu().item()
            total_loss.backward()
            optimizer.step()
            lr_hist.append(optimizer.param_groups[0]["lr"])
                    
        scheduler.step()
        avg_epoch_loss = step_loss / (batchidx+1)
        print_epoch_img_loss = mae_loss.detach().cpu().item()
        train_hist.append(avg_epoch_loss)
        if epoch % args.display_epoch == 0:
            logger.info(f"Training: Epoch: {epoch}, step img loss: {print_epoch_img_loss},  Training epoch loss: {avg_epoch_loss}")
        
        if epoch % args.valid_epoch == 0:
            avg_valid_loss = validation(args, epoch, model)
            valid_hist.append(avg_valid_loss)
            
            if avg_valid_loss < best_valid_loss:
                checkpoint_path = save_direct(args.ckpt_path)
                if args.model_name =='LCA':
                    stats = {}
                    stats['model_param'] = model.state_dict()  
                    model_file = os.path.join(checkpoint_path, 'LCA.pt.tar')
                    torch.save(stats, model_file)
                    print("save model to %s" % model_file)
                elif args.model_name =='HCA':
                    stats = {}
                    stats['model_param'] = model.state_dict()  
                    model_file = os.path.join(checkpoint_path, 'HCA.pt.tar')
                    torch.save(stats, model_file)
                    print("save model to %s" % model_file)
                best_valid_loss = avg_valid_loss
                
            
    hist = {
        "train_hist": train_hist,
        "valid_hist": valid_hist,
        "lr_hist": lr_hist
    }
    
    Graph(args, hist)
        


def validation(args, epoch, model):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for batchidx, inp in enumerate(valid_loader):
            
            inp = inp.reshape((-1, args.channels, args.height, args.width))
            ####If you want to insert mass conservation constraint, please uncomment the line below:
                # inp_avg = torch.mean(inp, dim=(2,3)).to(torch.float32).to(device)
            inp=inp.to(torch.float32).to(device)
            pred, _ = model(inp)
            ####If you want to insert mass conservation constraint, please uncomment the two lines below:
                # pred_avg = torch.mean(pred, dim=(2,3))
                # mse_loss_avg = criterionL2(pred_avg, inp_avg)
                
            mae_loss = criterionL1(pred, inp)
            ####If you want to insert mass conservation constraint, please change the loss function to this:
                # total_loss = mae_loss + args.alpha*mse_loss_avg
            total_loss = mae_loss
            valid_loss += total_loss.cpu().item()
     
    ave_valid_loss = valid_loss / (batchidx + 1)
    logger.info(f"Validation: avg Loss per batch : {ave_valid_loss}")

    return  ave_valid_loss 



def Graph(args, loss_hist):
    graph_directory = save_direct(args.graph_path)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(1, args.num_epoch+1), loss_hist['train_hist'], marker=".")
    ax.set_xlabel('epochs')
    ax.set_ylabel('average_epoch_loss')
    ax.set_title('training loss')
    plt.savefig(os.path.join(graph_directory, r'train_epoch_loss.png'))
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(args.valid_epoch, args.num_epoch+args.valid_epoch, args.valid_epoch), loss_hist['valid_hist'], marker=".")
    ax.set_xlabel('epochs')
    ax.set_ylabel('average_epoch_loss')
    ax.set_title('validation loss')
    plt.savefig(os.path.join(graph_directory, r'validation_epoch_loss.png'))
    plt.close()
    
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(range(1, len(loss_hist['lr_hist'])+1), loss_hist['lr_hist'], marker=".")
    ax.set_xlabel('itrs')
    ax.set_ylabel('Learning_rate')
    ax.set_title('Learning_rate')
    plt.savefig(os.path.join(graph_directory, r'Learning_rate.png'))
    plt.close()
            

        
if __name__ == "__main__":
    print("train stage")
    clean_directory(args)
    train(args, model)
        
        
    

          




