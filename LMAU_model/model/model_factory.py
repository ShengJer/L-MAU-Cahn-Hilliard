import sys
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
sys.path.append('./model')
import LMAU

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.in_features = configs.in_features # C
        self.num_layers = configs.num_layers 
        self.sampling_stop_iter = configs.sampling_stop_iter
        networks_map = {
            'lmau': LMAU.RNN,
        }
        num_hidden = []
        for i in range(configs.num_layers):
            num_hidden.append(configs.num_hidden) #[64 ,64, 64, 64]
        self.num_hidden = num_hidden
        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device) # initialize the network 
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)
        
        self.optimizer = Adam(self.network.parameters(), lr=configs.lr)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=configs.step_size, gamma=configs.lr_decay)
        self.MSE_criterion = nn.MSELoss() # mean square error (MSE) 
        self.L1_loss = nn.L1Loss() # mean absolute error (MAE)
        self.device = configs.device
    def save(self, itr):
        stats = {'net_param': self.network.state_dict()}
        current = os.getcwd()
        save_path = os.path.join(current, self.configs.save_dir)
        filelist = os.listdir(save_path)
        if filelist == []:
            checkpoint_path = os.path.join(save_path, 'model.pt.tar' + '-' + str(itr))
            torch.save(stats, checkpoint_path)
        else:
            for i in range(len(filelist)):
                filename = os.path.join(save_path, filelist[i])
                os.remove(filename)
            checkpoint_path = os.path.join(save_path, 'model.pt.tar' + '-' + str(itr))
            torch.save(stats, checkpoint_path)
        print("save predictive model to %s" % checkpoint_path)
        
    def load(self, pm_checkpoint_path):
        print('load predictive model:', pm_checkpoint_path)
        stats = torch.load(pm_checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])
        
    def train(self, data, mask, itr):
        # data = [B, TL, Fin] np.float32
        # mask = (B, TL-IL-1, Fin) np.float32
        self.network.train()
        
        frames = data
        frames_tensor = torch.FloatTensor(frames).to(self.device)
        mask_tensor = torch.FloatTensor(mask).to(self.device)

        next_frames = self.network(frames_tensor, mask_tensor)
        # next_frames (B, TL-1, Fin)
        ground_truth = frames_tensor

        self.optimizer.zero_grad()
        loss_l1 = self.L1_loss(next_frames,
                               ground_truth[:, 1:])
        loss_l2 = self.MSE_criterion(next_frames,
                                     ground_truth[:, 1:])
        # do we minimize l1 + l2 or l2 only
        if self.configs.loss_type == 'L1':
            loss_gen = loss_l1
        elif self.configs.loss_type == 'L2':
            loss_gen = loss_l2
        elif self.configs.loss_type == 'L1+L2' or self.configs.loss_type =='L2+L1':
            loss_gen = loss_l2 + loss_l1
        #loss_gen = loss_l1
        loss_gen.backward()
        self.optimizer.step()
        self.scheduler.step()
       # print('Lr decay to:%.8f', self.optimizer.param_groups[0]['lr'])
        
        return next_frames, loss_l1.detach().cpu().numpy(), loss_l2.detach().cpu().numpy()

    def test(self, data, mask): # this also means validation
        frames = data
        with torch.no_grad():
            self.network.eval()
            frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
            mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
            next_frames = self.network(frames_tensor, mask_tensor)
            ground_truth = frames_tensor
            loss_l1 = self.L1_loss(next_frames,
                                   ground_truth[:, 1:])
            loss_l2 = self.MSE_criterion(next_frames,
                                         ground_truth[:, 1:])
        return next_frames.cpu().numpy(), loss_l1.cpu().numpy(), loss_l2.cpu().numpy()
