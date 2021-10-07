import numpy as np
import typing
import torch
from torch import nn, optim
from collections import OrderedDict
from torch.autograd import Variable
import parser
import os
import os.path as osp
import argparse
# import sys
# sys.path.insert('./src')
from torch.utils.data.dataloader import DataLoader
from src.data_prep import DatasetPINN
from src import *
from src.PINN import *
from src.loss import *
import torch.multiprocessing as mp
import torch.distributed as dist
# from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description="PINN")
    parser.add_argument("--data", type=str, default="./data/demo_geo_model.npy")
    parser.add_argument("--in_chn", type=int, default=4)
    parser.add_argument("--out_chn", type=int, default=5)
    parser.add_argument("--hid_par", type=int, default=25)
    parser.add_argument("--hid_lay", type=int, default=5)
    parser.add_argument("--function", type=str, default="NS")
    # parser.add_argument("--multi_gpu", action="store_False")
    # parser.add_argument("--DDP", action="store_True")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1e3)
    args = parser.parse_args()
    
    net = pinn(in_chn=args.in_chn, out_chn = args.out_chn, hid_layer=args.hid_lay, hid_par=args.hid_par)
    # if args.DDP:
    #     dist.init_process_group(
    #         backend='nccl',
    #         init_method='env://',
    #     )
    loss_ns = Navier_Stokes_3D(Rey=1e2, Pec=1e2)
    loss_sim = veolicity_loss()
    optimizer = optim.Adam(net.parameters(), args.lr)

    trainset = DatasetPINN(data_dir=args.data)
    trainloader = DataLoader(trainset, batch_size=len(trainset))

    for epoch in range(1, int(args.epochs)):
        net.train()
        for i, (pos, vol) in enumerate(trainloader, 0):
            optimizer.zero_grad()
            
            pos = Variable(pos).requires_grad_()
            pred = net(pos)
            loss_flu = loss_ns(pred, pos)
            loss_vol = loss_sim(pred, vol)
            loss = loss_flu +  loss_vol
            loss.backward()
            optimizer.step()
            print(epoch, i)


if __name__ == "__main__":
    main()    

