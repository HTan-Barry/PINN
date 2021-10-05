import numpy as np
import typing
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.autograd import Variable
import parser
import os
import os.path as osp
import argparse
from src import *
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

def main():
    parser = argparse.ArgumentParser(description="PINN")
    parser.add_argument("--data", type=str)
    parser.add_argument("--in_chn", type=int)
    parser.add_argument("--out_chn", type=int)
    parser.add_argument("--hid_par", type=int)
    parser.add_argument("--hid_lay", type=int)
    parser.add_argument("--function", type=str, default="NS")
    parser.add_argument("--multi_gpu", action="store_False")
    parser.add_argument("--DDP", action="store_True")
    args = parser.parse_args()
    
    net = pinn(in_chn=args.in_chn, out_chn = args.out, hid_layer=args.hid_lay, hid_par=args.hid_par)
    if args.DDP:
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
        )
    

