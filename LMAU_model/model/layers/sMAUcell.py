import torch
import torch.nn as nn
import math

class sMAUCell(nn.Module):
    def __init__(self, in_features, num_hidden, tau, cell_mode):
        super(sMAUCell, self).__init__()
        self.num_hidden = num_hidden
        self.cell_mode = cell_mode # residual or normal in states
        self.tau = tau # how much previous time should be considered
        self.states = ['residual', 'normal']
        if not self.cell_mode in self.states:
            raise AssertionError
        self.Lin_t_next = nn.Sequential(
            nn.Linear(in_features, num_hidden),
            nn.LayerNorm([num_hidden])
        )
        self.Lin_s_next = nn.Sequential(
            nn.Linear(num_hidden, num_hidden),
            nn.LayerNorm([num_hidden])
        )
        
        self.Lin_s = nn.Sequential(
            nn.Linear(num_hidden, 3* num_hidden),
            nn.LayerNorm([3 * num_hidden])
        )
        self.Lin_t = nn.Sequential(
            nn.Linear(num_hidden, 3* num_hidden),
            nn.LayerNorm([3 * num_hidden])
        )
        self.softmax = nn.Softmax(dim=0)
    def forward(self, T_t, S_t, t_att, s_att):
        # t_att  
        # s_att  (tau, B, hidden)
        # T_t  #( B, hidden)
        # S_t  #(B, hidden)
        s_next = self.Lin_s_next(S_t)  #(B, hidden)
        t_next = self.Lin_t_next(T_t) #(B, hidden)
        weights_list = []
        for i in range(self.tau):
            weights_list.append((s_att[i] * s_next).sum(dim=1) / math.sqrt(self.num_hidden))
        weights_list = torch.stack(weights_list, dim=0) # (tau, B)
        weights_list = torch.reshape(weights_list, (*weights_list.shape, 1)) # (tau, B, 1,)
        weights_list = self.softmax(weights_list) # (tau, B, 1)
        T_trend = t_att * weights_list
        T_trend = T_trend.sum(dim=0) # (B, hidden)
        t_att_gate = torch.sigmoid(t_next) #(B, hidden)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend #(B, hidden)
        
        T_concat = self.Lin_t(T_fusion)  #(B, C_hidden*3)
        S_concat = self.Lin_s(S_t) #(B, C_hidden*3)
        t_g, t_t, t_s = torch.split(T_concat, self.num_hidden, dim=1) #(B, C_hidden, W/p/(2^n), H/p/(2^n))
        s_g, s_t, s_s = torch.split(S_concat, self.num_hidden, dim=1) #(B, C_hidden, W/p/(2^n), H/p/(2^n))
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s
        if self.cell_mode == 'residual':
            S_new = S_new + S_t
        return T_new, S_new
        
        
        
        
        
        
        
        
        
        
        
        
        
        

