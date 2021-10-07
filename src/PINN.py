"""
Author: Haobo Tan
This is the basic structure of the Physical-Informed Neural Network
"""
import numpy as np
import typing
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable


class pinn(nn.Module):
    def __init__(self, in_chn, out_chn, hid_layer, hid_par) -> None:
        super(pinn, self).__init__()
        net_dict = OrderedDict()
        net_dict["mlp_1"] = nn.Linear(in_chn, hid_par)
        net_dict["sig_1"] = nn.Sigmoid()
        for i in range(2, hid_layer):
            
            net_dict[f'mlp_{i}'] = nn.Linear(hid_par, hid_par)
            net_dict[f'sig_{i}'] = nn.Sigmoid()
        net_dict[f'mlp_{hid_layer}'] = nn.Linear(hid_par, out_chn)
        net_dict[f'sig_{hid_layer}'] = nn.Sigmoid()

        self.net = nn.Sequential(net_dict)
    
    def forward(self, x):
        out = self.net(x)
        
        return out

if __name__ == "__main__":
    net = pinn(3, 2, 5, 5)
    print(net)