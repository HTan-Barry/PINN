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

loss = nn.MSELoss()

def relative_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.sqrt( mean_squared_error(pred, exact)/np.mean(np.square(exact - np.mean(exact))))
    return torch.sqrt( loss(pred, exact) / torch.mean(torch.square(exact - torch.mean(exact))) )


def mean_squared_error(pred, exact):
    if type(pred) is np.ndarray:
        return np.mean(np.square(pred - exact))
    return loss(pred, exact)


def fwd_gradients(Y, x):
    dummy = torch.ones_like(Y)
    G = torch.autograd.grad(Y, x, dummy, create_graph= True)
    return G

class Navier_Stokes_3D(nn.Module):
    def __init__(self, Rey, Pec):
        super(Navier_Stokes_3D, self).__init__()
        self.Pec = Pec
        self.Rey = Rey


    def forward(self, pred, data):
        c, u, v, w, p = pred[:,0], pred[:,1], pred[:,2], pred[:,3], pred[:,4]
        
        

        # c_txyz = fwd_gradients(c, data)
        # u_txyz = fwd_gradients(u, data)
        # v_txyz = fwd_gradients(v, data)
        # w_txyz = fwd_gradients(w, data)
        # p_txyz = fwd_gradients(p, data)
        print(fwd_gradients(c, data[1]))
        
        c_t = c_txyz[:,0:1]
        c_x = c_txyz[:,1:2]
        c_y = c_txyz[:,2:3]
        c_z = c_txyz[:,3:4]
        
        u_t = u_txyz[:,0:1]
        u_x = u_txyz[:,1:2]
        u_y = u_txyz[:,2:3]
        u_z = u_txyz[:,3:4]
                            
        v_t = v_txyz[:,0:1]
        v_x = v_txyz[:,1:2]
        v_y = v_txyz[:,2:3]
        v_z = v_txyz[:,3:4]
        
        w_t = w_txyz[:,0:1]
        w_x = w_txyz[:,1:2]
        w_y = w_txyz[:,2:3]
        w_z = w_txyz[:,3:4]
                            
        p_x = p_txyz[:,1:2]
        p_y = p_txyz[:,2:3]
        p_z = p_txyz[:,3:4]

        # second gradient
        
        c_x_txyz = fwd_gradients(c_x, data)
        c_y_txyz = fwd_gradients(c_y, data)
        c_z_txyz = fwd_gradients(c_z, data)
        c_xx = c_x_txyz[:,1:2] #wanted
        c_yy = c_y_txyz[:,2:3] #wanted
        c_zz = c_z_txyz[:,3:4] #wanted
                            
                    
        u_x_txyz = fwd_gradients(u_x, data)
        u_y_txyz = fwd_gradients(u_y, data)
        u_z_txyz = fwd_gradients(u_z, data)
        u_xx = u_x_txyz[:,1:2] #wanted
        u_yy = u_y_txyz[:,2:3] #wanted
        u_zz = u_z_txyz[:,3:4] #wanted
        
        v_x_txyz = fwd_gradients(v_x, data)
        v_y_txyz = fwd_gradients(v_y, data)
        v_z_txyz = fwd_gradients(v_z, data)
        v_xx = v_x_txyz[:,1:2] #wanted
        v_yy = v_y_txyz[:,2:3] #wanted
        v_zz = v_z_txyz[:,3:4] #wanted
                            
        w_x_txyz = fwd_gradients(w_x, data)
        w_y_txyz = fwd_gradients(w_y, data)
        w_z_txyz = fwd_gradients(w_z, data)
        w_xx = w_x_txyz[:,1:2] #wanted
        w_yy = w_y_txyz[:,2:3] #wanted
        w_zz = w_z_txyz[:,3:4] #wanted
        
        e1 = c_t + (u*c_x + v*c_y + w*c_z) - (1.0/self.Pec)*(c_xx + c_yy + c_zz)
        e2 = u_t + (u*u_x + v*u_y + w*u_z) + p_x - (1.0/self.Rey)*(u_xx + u_yy + u_zz)
        e3 = v_t + (u*v_x + v*v_y + w*v_z) + p_y - (1.0/self.Rey)*(v_xx + v_yy + v_zz)
        e4 = w_t + (u*w_x + v*w_y + w*w_z) + p_z - (1.0/self.Rey)*(w_xx + w_yy + w_zz)
        e5 = u_x + v_y + w_z
        
        return e1, e2, e3, e4, e5
class veolicity_loss(nn.Module):
    def __init__(self):
        super(veolicity_loss, self).__init__()
        
    def forward(self, pred, data):
        pred_speed = pred[:, -4:-1]
        return loss(pred_speed, data)