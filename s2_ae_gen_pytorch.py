import os
import time
import sys

import numpy as np

import torch
import torch.optim as optim
from torch import nn as nn
from torch.nn.utils import parametrizations as params
import kornia.geometry.transform as T
from tqdm import tqdm

#from skimage.transform import resize, rotate
#from scipy.ndimage import rotate

from misc_fns import *
#from s2_common import *

class WLearner(nn.Module):
    def __init__(self, ip_n, op_n, t):
        super(WLearner, self).__init__()
        self.ip_n = ip_n * ip_n
        self.op_n = op_n
        self.t = t
        self.type_0_mask = self.t == 0
        self.type_0_idx = (t == 0).nonzero().flatten()
        self.type_0 = len(self.type_0_idx)

        self.W = nn.Parameter(torch.empty(self.ip_n, self.op_n, dtype=torch.cfloat))
        nn.init.xavier_uniform_(self.W.real)
        nn.init.zeros_(self.W.imag[:, :self.type_0])
        nn.init.xavier_uniform_(self.W.imag[:, self.type_0:])
        print("W.shape: ", self.W.shape)

    def forward(self, x, ang):
       
        self.W[:, self.type_0_mask].imag = 0

        scale_f = torch.exp(-1j*self.t * ang * torch.pi / 180.)
        sh = x.shape
        #print("x.shape: ", sh)
        x = x.reshape(-1, self.ip_n)
        x = x @ self.W.conj()
        x = x * scale_f
        x = x @ self.W.T
        x = x.reshape(sh)

        #sh = y.shape
        #print("y.shape: ", sh)
        #y = y.reshape(-1, self.ip_n)
        #y = y @ self.W
        #y = y.reshape(sh)

        return x.real, self.W.T @ self.W.conj()

def fill_t(cnt): 
    t = np.array([0] * cnt[0])
    for i in range(1, cnt.shape[0]):
        mult = cnt[i]
        pos_t = [i] * mult
        t = np.concatenate((t, pos_t))
    return t

def CplxNorm(x):
    norm = torch.mean(x * x.conj(), axis=0)
    return norm.real.sum()

    r = {}
    n =28
    xv, yv = np.meshgrid(np.linspace(-(n-1)/2, (n-1)/2, n), np.linspace(-(n-1)/2, (n-1)/2, n))
    r_mat = np.zeros((n, n))
    tot = 0
    for i in range(n):
        for j in range(n):
            rad = int(2*(xv[i, j]**2 + yv[i, j]**2))
            if rad > 2 * (n//2)**2:
                continue
            if rad not in r:
                r[rad] = []
            r[rad].append(float(np.arctan2(yv[i, j], xv[i, j]) * 180 / np.pi))
            r_mat[i, j] = rad
            tot += 1
    print("Total : ", tot)

    cnt = []
    tot = 0
    for k, v in r.items():
        print (k, len(v))
        if len(v) % 2 == 0:
            cnt = cnt + list(range((len(v)+1)//2))
        else:
            cnt = cnt + list(range(len(v) / 2 + 1))
        tot += len(v)



if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

 
    jo = read_json(sys.argv[1])

#############
    
    #Consts
    s_size = jo.s_size
    b_size = jo.b_size
    epochs = jo.epochs
   
    if b_size == 9999:
        b_size = s_size / 10
   
    #valid_size = 
    LEARNING_RATE = jo.l_rate
    ip_n = jo.scale_shape 
    op_n = jo.act_shape
 
    R = jo.act_recons
    B = jo.act_reg
    
    BETA = 0.1
    if hasattr(jo, "beta"):
        BETA = jo.beta
    BETA *= (s_size / 10000.)

    log_tm = time.strftime('%Y-%m-%d_%H_%M_%S')
    #log_dir = 'logs/ae_known_ang-{}/'.format(log_tm)
    #log_dir = 'logs/ae_known_ang/'
    log_dir = '{}_ae_{}_{}'.format(jo.log_dir_prefix, jo.aug_count, s_size)
    print('**************************')
    print(log_dir)
    print('**************************')
    
    #data = np.load(jo.train_data)    #np.load('mnist_train.npz')
    #utr = data['data'][:s_size]
    utr = torch.rand(s_size, 1, ip_n, ip_n, device=device)

    # TODO = add support for rgb images
    # utr = utr.reshape(-1, op_n**2, order='F')
    #data.close()
 
    #if ip_n != op_n:
    #    utr = resize(utr.reshape(-1, op_n, op_n), (utr.shape[0], ip_n, ip_n)).reshape(-1, ip_n**2)
    
    print('utr.shape ', utr.shape)

    X_train = utr
    y_train = utr.clone().detach()
 
    cnt_ip = np.repeat(jo.act_cnt, jo.act_cnt_rep)
    cnt_int = np.repeat(jo.scale_cnt, jo.scale_cnt_rep)
    cnt_op_1 = np.repeat(jo.act_cnt, jo.act_cnt_rep)
    cnt_op = np.repeat(jo.act_cnt, jo.act_cnt_rep)
    print('cnt_op : ', cnt_op)
    cnt_op = np.int64(np.hstack([cnt_op[:1], cnt_op[1:]/2]))

    print (jo.act_cnt)
    print('cnt_ip : ', cnt_ip)
    print('sum(cnt_ip) : ', sum(cnt_ip))
    print('cnt_int : ', cnt_int)
    print('sum(cnt_int) : ', sum(cnt_int))
    print('cnt_op : ', cnt_op)
    print('sum(cnt_op) : ', sum(cnt_op))
    print('cnt_op_1 : ', cnt_op_1)
    print('sum(cnt_op_1) : ', sum(cnt_op_1))

    t_ip  = fill_t(cnt_ip) 
    t_op  = fill_t(cnt_op) 
    t_int = fill_t(cnt_int) 
    
    op_sh = t_op.shape[0]

    print(cnt_ip)
    print(cnt_op)
    print(cnt_int)

    print('t_ip: ', t_ip)
    print('t_op: ', t_op)

    model = WLearner(ip_n, op_sh, torch.tensor(t_op))
    model.to(device)
    model = params.orthogonal(model, "W")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    start_epoch = 0
    for epoch in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(len(X_train))
        pbar = tqdm(range(0, len(X_train), b_size), desc=f"Epoch {epoch+1}/{epochs}", ncols=120)
        for i in pbar:
            idx = perm[i:i+b_size]
            ang = torch.rand(b_size) * 360.0
            xb, yb = X_train[idx], y_train[idx]
            #xb = torch.rand(b_size, 1, ip_n, ip_n)
            #yb = xb.clone().detach()
            xb = T.rotate(xb, ang)
            optimizer.zero_grad()
            y_hat, wtw = model(xb.to(dtype=torch.cfloat), ang.unsqueeze(1))
            reg = torch.norm(wtw - torch.eye(op_sh))
            #recon_err = CplxNorm(y_hat-yb) + B * reg
            recon_err = criterion(y_hat, yb)
            loss = recon_err + B * reg
            loss.backward()
            optimizer.step()
            train_err = loss # compute_error(model, X_train, y_train, batch_size)
            pbar.set_postfix({"recon rr": f"{recon_err:.4f}", " reg:": f"{reg:.4f}"})

        if (epoch + 1) % 5 == 0:
            W_cplx = model.W.detach().numpy()
            save_model(W_cplx, epoch+1)
    
    state_dict = model.state_dict()
    print('state dict: ', state_dict)
    #torch.save(state_dict, '{}/state_dict_{}x{}'.format(log_dir, ip_n ** 2, sum(cnt_op_1)))
    print (model.W)
    W_cplx = model.W.detach().numpy()
    save_model(W_cplx, "xx")
    # #W_cplx = state_dict['W']
    # #print(state_dict['W'])
    # #print(state_dict['W'].shape)
    # W = np.zeros((ip_n ** 2, sum(cnt_op_1)))
    # W[:, :cnt_ip[0]] = W_cplx[:, :cnt_ip[0]].real
    # W[:, cnt_ip[0]::2] = W_cplx[:, cnt_ip[0]:].real
    # W[:, cnt_ip[0]+1::2] = W_cplx[:, cnt_ip[0]:].imag
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)
    # np.save('{}/W_{}x{}_xx'.format(log_dir, ip_n ** 2, sum(cnt_op_1)), W)
    # np.save('{}/cnt_{}x{}_xx'.format(log_dir, ip_n ** 2, sum(cnt_op_1)), cnt_op_1)

def save_model(W_cplx, epoch):
    W_cplx = model.W.detach().numpy()
    #W_cplx = state_dict['W']
    #print(state_dict['W'])
    #print(state_dict['W'].shape)
    W = np.zeros((ip_n ** 2, sum(cnt_op_1)))
    W[:, :cnt_ip[0]] = W_cplx[:, :cnt_ip[0]].real
    W[:, cnt_ip[0]::2] = W_cplx[:, cnt_ip[0]:].real
    W[:, cnt_ip[0]+1::2] = W_cplx[:, cnt_ip[0]:].imag
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    np.save('{}/W_{}x{}_{}'.format(log_dir, ip_n ** 2, sum(cnt_op_1)), W, epoch)
    np.save('{}/cnt_{}x{}_{}'.format(log_dir, ip_n ** 2, sum(cnt_op_1)), cnt_op_1, epoch)


    
