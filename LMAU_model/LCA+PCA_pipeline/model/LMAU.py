import sys
import torch
import torch.nn as nn
sys.path.append('./model/layers')
from layers.LMAUcell import LMAUCell
import math

class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()
        self.configs = configs
        self.in_features = configs.in_features
        self.num_layers = num_layers # 4
        self.num_hidden = num_hidden # [64, 64, 64, 64]
        self.tau = configs.tau # 5
        self.cell_mode = configs.cell_mode
        self.states = ['recall', 'normal']
        if not configs.model_mode in self.states:
            raise AssertionError
        cell_list = []
        
        for i in range(num_layers):
            in_features = num_hidden[i - 1] 
            
            '''
            i   Cin    | Cout
            0   64      64
            1   64      64
            2   64      64
            3   64      64
            '''
            cell_list.append(
                LMAUCell(in_features, num_hidden[i], self.tau, self.cell_mode)
            )
        
        self.cell_list = nn.ModuleList(cell_list)
        self.Lin_start = nn.Sequential(
            nn.Linear(self.in_features, num_hidden[0], bias=False),
            nn.LayerNorm([num_hidden[0]]))
        self.Lin_last = nn.Linear(num_hidden[num_layers - 1], self.in_features, bias=False)
        self.cell_list.apply(self._init_weights)

    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    def forward(self, frames, mask_true):
        # frames = (B, TL, Fin)
        # mask_true = (B, TL-IL-1, Fin)
        batch_size = frames.shape[0]
        next_frames = []
        T_t = []
        T_pre = []
        S_pre = []
        x_gen = None
        # initialize the memory of temporal and spatial memory
        for layer_idx in range(self.num_layers):
            tmp_t = []
            tmp_s = []
            if layer_idx == 0:
                in_features = self.num_hidden[layer_idx]
            else:
                in_features = self.num_hidden[layer_idx - 1]
            for i in range(self.tau):
                tmp_t.append(torch.zeros([batch_size, in_features]).to(self.configs.device))
                tmp_s.append(torch.zeros([batch_size, in_features]).to(self.configs.device))
            T_pre.append(tmp_t) #[[size tau tensor],[size tau tensor],[size tau tensor],[size tau tensor]]
            S_pre.append(tmp_s) # #[[size tau tensor],[size tau tensor],[size tau tensor],[size tau tensor]]
        for t in range(self.configs.total_length - 1): # t=0~TL-2
            if t < self.configs.input_length: # before time=input_length, take out the true frame
                net = frames[:, t] #(B, Fin)
            else:
                time_diff = t - self.configs.input_length
                net = mask_true[:, time_diff] * frames[:, t] + (1 - mask_true[:, time_diff]) * x_gen
                # probability of taking out true or predicted frame
            frames_feature = net
            frames_feature_encoded = []
            frames_feature = self.Lin_start(frames_feature)
            frames_feature_encoded = frames_feature
            if t == 0:
                for i in range(self.num_layers):
                    zeros = torch.zeros([batch_size, self.num_hidden[i]]).to(self.configs.device)
                    T_t.append(zeros) # initialize the T_t state for each layers #(B, F_hidden)
            S_t = frames_feature
            for i in range(self.num_layers):
                t_att = T_pre[i][-self.tau:] # a list contains previous size tau tensor with shape=(B,C_hidden)
                t_att = torch.stack(t_att, dim=0) # (tau, B, F)
                s_att = S_pre[i][-self.tau:]
                s_att = torch.stack(s_att, dim=0) # (tau, B, F)
                T_t[i], S_t = self.cell_list[i](T_t[i], S_t, t_att, s_att)
                T_pre[i].append(T_t[i])
            out = S_t
            if self.configs.model_mode == 'recall':
                out = out + frames_feature_encoded
            x_gen = self.Lin_last(out)
            next_frames.append(x_gen)
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 2).contiguous()
        # next_frames (B, TL-1, Fin)
        return next_frames

    
    
    
    