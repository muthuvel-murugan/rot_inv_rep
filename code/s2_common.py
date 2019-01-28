import os
import time
import sys

import numpy as np
import tensorflow as tf

#from skimage.transform import resize, rotate
from scipy.ndimage import rotate

from gen_R1 import *

import tf_rot_inv_cust

def init_tf(gpu_percent=0.5, grow=True, thrds=10):
    config = tf.ConfigProto()
    
    config.intra_op_parallelism_threads = thrds
    config.inter_op_parallelism_threads = thrds

    config.gpu_options.per_process_gpu_memory_fraction = gpu_percent
    config.gpu_options.allow_growth = grow

    session = tf.Session(config=config)
    #tf.set_random_seed(1)
    #np.random.seed(1)

def l1_loss(params):
    return tf.abs(tf.reshape(params, [-1]))
    
def l2_loss(params):
    return tf.square(tf.reshape(params, [-1]))

def full_quad_tensor_n(inputs, t, keep_d1=False):
    # input.shape = (#samp, t.shape[0], 1)
    fname = sys._getframe().f_code.co_name
    print '{} - inputs shape : {}'.format(fname, inputs.shape)
    w_tot = t.shape[0]
    w_0 = np.sum(t==0)
    w_rest = np.sum(t>0)
    print '{} - w_tot : {}'.format(fname, w_tot)
    print '{} - w_0 : {}'.format(fname, w_0)
    print '{} - w_rest : {}'.format(fname, w_rest)

    ind_0 = np.arange(w_0, dtype=np.int32)
    ind_1 = np.arange(w_0, w_tot, dtype=np.int32)

    print '{} - ind0 : {}'.format(fname, ind_0)
    print '{} - ind0.shape : {}'.format(fname, ind_0.shape)
    print '{} - ind1 : {}'.format(fname, ind_1)
    print '{} - ind1.shape : {}'.format(fname, ind_1.shape)
   
    # bring the proj coeffs to the front
    inputs = tf.transpose(inputs, [1, 0, 2])
    print '{} - inputs shape : {}'.format(fname, inputs.shape)
    ip_0 = tf.gather(inputs, ind_0)
    ip_1 = tf.gather(inputs, ind_1)
    # bring #samps to the front
    inputs = tf.transpose(inputs, [1, 0, 2])
    ip_0 = tf.transpose(ip_0, [1, 0, 2])
    ip_1 = tf.transpose(ip_1, [1, 0, 2])

    # deg 1 tensor (i.e. the input itself)
    op_all_deg_1 = inputs
    t_all_deg_1 = t
    print '{} - op_all_deg_1 shape : {}'.format(fname, op_all_deg_1.shape)

    # TODO dont tensor with type 0
    # TODO
    w_0 = 0

    # deg 2, w = 0 case
    if w_0 > 0:
        op_0_deg_2 = tf.matmul(ip_0, inputs, transpose_b=True)
        #print '{} - op_0_deg_2 shape : {}'.format(fname, op_0_deg_2.shape)
        op_0_deg_2 = tf.reshape(op_0_deg_2, [-1, w_0 * w_tot, 1])
        t_0_deg_2 = np.tile(t, w_0)
        #print '{} - op_0_deg_2 shape : {}'.format(fname, op_0_deg_2.shape)

    # TODO For higher degree tensors, normalize one part
    # deg 2, w > 0 case
    if w_rest > 0:
        r = tf.sqrt(ip_1[:, ::2] ** 2 + ip_1[:, 1::2] ** 2)
        r = tf.stack([r, r], axis=-1)
        r = tf.reshape(r, [-1, w_rest, 1])
        r = tf.maximum(r, 1e-7)
        ip_1_n = ip_1 / r
        ip_1_tens = tf.matmul(ip_1, ip_1_n, transpose_b=True)
        re1 = ip_1_tens[:, ::2, ::2]
        re2 = ip_1_tens[:, 1::2, 1::2]
        im1 = ip_1_tens[:, ::2, 1::2]
        im2 = ip_1_tens[:, 1::2, ::2]
        a = re1 - re2
        b = im1 + im2
        c = re1 + re2
        d = -im1 + im2

        op_1_deg_2 = tf.stack([a, b, c, d], axis=-1)
        #print '{} - op_1_deg_2 shape : {}'.format(fname, op_1_deg_2.shape)

        op_1_deg_2 = tf.reshape(op_1_deg_2, [-1, w_rest * w_rest, 1])

        def _t_add(t1, t2):
            tt = np.zeros((t1.shape[0], t2.shape[0]), dtype=np.int32)
            for i in range(t1.shape[0]):
                tt[i] = t1[i] + t2
            return tt
        
        t_rest = t[ind_1]
        ta = _t_add(t_rest[::2], t_rest[::2])
        tb = _t_add(-t_rest[1::2], -t_rest[1::2])
        tc = _t_add(t_rest[::2], -t_rest[1::2])
        td = _t_add(-t_rest[1::2], t_rest[::2])
        t_1_deg_2 = np.stack([ta, ta, tc, tc], axis=-1)
        t_1_deg_2 = t_1_deg_2.flatten()
        mask_1_deg_2 = np.zeros_like(t_1_deg_2, dtype=np.bool)
        mask_1_deg_2[::2] = (t_1_deg_2[::2] >= 0)
        mask_1_deg_2[1::2] = mask_1_deg_2[::2]

        op_1_deg_2 = tf.boolean_mask(op_1_deg_2, mask_1_deg_2, axis=1)
        t_1_deg_2 = t_1_deg_2[mask_1_deg_2]
        #print '{} - op_1_deg_2 shape : {}'.format(fname, op_1_deg_2.shape)

    if keep_d1:
        outputs = op_all_deg_1
        t_op = t_all_deg_1

    # TODO Forcing not to take op_0_deg_2
    #w_0 = 0

    if w_0 > 0:
        if keep_d1:
            outputs = tf.concat([outputs, op_0_deg_2], axis=1)
            t_op = np.concatenate([t_op, t_0_deg_2])
        else:
            outputs = op_0_deg_2
            t_op = t_0_deg_2
        
        #outputs = tf.concat([outputs, op_0_deg_2], axis=1)
        #t_op = np.concatenate([t_op, t_0_deg_2])
    
    # TODO Forcing not to take op_0_deg_2
    #w_0 = 0
    
    if w_rest > 0:
        if w_0 == 0:
            if keep_d1:
                outputs = tf.concat([outputs, op_1_deg_2], axis=1)
                t_op = np.concatenate([t_op, t_1_deg_2])
            else:
                outputs = op_1_deg_2
                t_op = t_1_deg_2
        else:
            outputs = tf.concat([outputs, op_1_deg_2], axis=1)
            t_op = np.concatenate([t_op, t_1_deg_2])

    #outputs = op_1_deg_2
    #t_op = t_1_deg_2

    t_op_order = np.argsort(t_op, kind='mergesort')
    t_op = t_op[t_op_order]
    outputs = tf.gather(outputs, t_op_order, axis=1)

    print '{} - outputs shape : {}'.format(fname, outputs.shape)
    return outputs, t_op


def full_quad_tensor(inputs, t, keep_d1=False):
    # input.shape = (#samp, t.shape[0], 1)
    fname = sys._getframe().f_code.co_name
    print '{} - inputs shape : {}'.format(fname, inputs.shape)
    w_tot = t.shape[0]
    w_0 = np.sum(t==0)
    w_rest = np.sum(t>0)
    print '{} - w_tot : {}'.format(fname, w_tot)
    print '{} - w_0 : {}'.format(fname, w_0)
    print '{} - w_rest : {}'.format(fname, w_rest)

    ind_0 = np.arange(w_0, dtype=np.int32)
    ind_1 = np.arange(w_0, w_tot, dtype=np.int32)

    print '{} - ind0 : {}'.format(fname, ind_0)
    print '{} - ind0.shape : {}'.format(fname, ind_0.shape)
    print '{} - ind1 : {}'.format(fname, ind_1)
    print '{} - ind1.shape : {}'.format(fname, ind_1.shape)
   
    # bring the proj coeffs to the front
    inputs = tf.transpose(inputs, [1, 0, 2])
    print '{} - inputs shape : {}'.format(fname, inputs.shape)
    ip_0 = tf.gather(inputs, ind_0)
    ip_1 = tf.gather(inputs, ind_1)
    # bring #samps to the front
    inputs = tf.transpose(inputs, [1, 0, 2])
    ip_0 = tf.transpose(ip_0, [1, 0, 2])
    ip_1 = tf.transpose(ip_1, [1, 0, 2])

    # deg 1 tensor (i.e. the input itself)
    op_all_deg_1 = inputs
    t_all_deg_1 = t
    print '{} - op_all_deg_1 shape : {}'.format(fname, op_all_deg_1.shape)

    # deg 2, w = 0 case
    if w_0 > 0:
        op_0_deg_2 = tf.matmul(ip_0, inputs, transpose_b=True)
        print '{} - op_0_deg_2 shape : {}'.format(fname, op_0_deg_2.shape)
        mask = np.ones((w_0, w_tot), dtype=np.bool)
        for ii in range(1, w_0):
            for jj in range(ii+1):
                mask[ii, :jj] = 0
        mask = mask.flatten()
        op_0_deg_2 = tf.reshape(op_0_deg_2, [-1, w_0 * w_tot, 1])
        op_0_deg_2 = tf.boolean_mask(op_0_deg_2, mask, axis=1)
        t_0_deg_2 = np.tile(t, w_0)
        t_0_deg_2 = t_0_deg_2[mask]

        print '{} - op_0_deg_2 shape : {}'.format(fname, op_0_deg_2.shape)

    # deg 2, w > 0 case
    if w_rest > 0:
        ip_1_tens = tf.matmul(ip_1, ip_1, transpose_b=True)
        re1 = ip_1_tens[:, ::2, ::2]
        re2 = ip_1_tens[:, 1::2, 1::2]
        im1 = ip_1_tens[:, ::2, 1::2]
        im2 = ip_1_tens[:, 1::2, ::2]
        a = re1 - re2
        b = im1 + im2
        c = re1 + re2
        d = -im1 + im2

        op_1_deg_2 = tf.stack([a, b, c, d], axis=-1)
        #print '{} - op_1_deg_2 shape : {}'.format(fname, op_1_deg_2.shape)

        op_1_deg_2 = tf.reshape(op_1_deg_2, [-1, w_rest * w_rest, 1])

        def _t_add(t1, t2):
            tt = np.zeros((t1.shape[0], t2.shape[0]), dtype=np.int32)
            for i in range(t1.shape[0]):
                tt[i] = t1[i] + t2
            return tt
        
        t_rest = t[ind_1]
        ta = _t_add(t_rest[::2], t_rest[::2])
        tb = _t_add(-t_rest[1::2], -t_rest[1::2])
        tc = _t_add(t_rest[::2], -t_rest[1::2])
        td = _t_add(-t_rest[1::2], t_rest[::2])
        t_1_deg_2 = np.stack([ta, ta, tc, tc], axis=-1)
        t_1_deg_2 = t_1_deg_2.flatten()
        mask_1_deg_2 = np.zeros_like(t_1_deg_2, dtype=np.bool)
        mask_1_deg_2[::2] = (t_1_deg_2[::2] >= 0)
        mask_1_deg_2[1::2] = mask_1_deg_2[::2]

        op_1_deg_2 = tf.boolean_mask(op_1_deg_2, mask_1_deg_2, axis=1)
        t_1_deg_2 = t_1_deg_2[mask_1_deg_2]
        #print '{} - op_1_deg_2 shape : {}'.format(fname, op_1_deg_2.shape)

    if keep_d1:
        outputs = op_all_deg_1
        t_op = t_all_deg_1

    if w_0 > 0:
        if keep_d1:
            outputs = tf.concat([outputs, op_0_deg_2], axis=1)
            t_op = np.concatenate([t_op, t_0_deg_2])
        else:
            outputs = op_0_deg_2
            t_op = t_0_deg_2
        
        #outputs = tf.concat([outputs, op_0_deg_2], axis=1)
        #t_op = np.concatenate([t_op, t_0_deg_2])
    
    if w_rest > 0:
        if w_0 == 0:
            if keep_d1:
                outputs = tf.concat([outputs, op_1_deg_2], axis=1)
                t_op = np.concatenate([t_op, t_1_deg_2])
            else:
                outputs = op_1_deg_2
                t_op = t_1_deg_2
        else:
            outputs = tf.concat([outputs, op_1_deg_2], axis=1)
            t_op = np.concatenate([t_op, t_1_deg_2])

    #outputs = op_1_deg_2
    #t_op = t_1_deg_2

    t_op_order = np.argsort(t_op, kind='mergesort')
    t_op = t_op[t_op_order]
    outputs = tf.gather(outputs, t_op_order, axis=1)

    print '{} - outputs shape : {}'.format(fname, outputs.shape)
    return outputs, t_op

def create_node(name, x_ip, t_ip, t_op, tens=True):
    fname = sys._getframe().f_code.co_name

    if tens:
        x_int, t_int = full_quad_tensor(x_ip, t_ip, keep_d1=True)
    else:
        x_int, t_int = x_ip, t_ip


    nop = 0

    max_w = min(t_int[-1], t_op[-1])

    wt_list = []
    op_list = []
    for i in range(max_w+1):
        ip_sh = np.sum(t_int == i)
        op_sh = np.sum(t_op == i)
        if i == 0:
            wt = tf.get_variable('weights_{}_{}'.format(name, i), [ip_sh, op_sh], \
                                initializer=tf.truncated_normal_initializer(stddev=0.1, mean=-1e-4),
                                regularizer=l1_loss)
                                #initializer=tf.contrib.layers.xavier_initializer(),
                                #initializer=tf.truncated_normal_initializer(stddev=0.01),
            wt_list.append(wt)
            nop += ip_sh * op_sh
            x_tmp = tf.boolean_mask(x_int, t_int==i, axis=1)
            x_tmp = tf.reshape(x_tmp, [-1, ip_sh])
            x_tmp = tf.matmul(x_tmp, wt)
            x_op = x_tmp
        else:
            #print '{} - i = {}, ip_sh : {}'.format(fname, i, ip_sh)
            #print '{} - i = {}, op_sh : {}'.format(fname, i, op_sh)
            ip_sh_h = ip_sh / 2
            op_sh_h = op_sh / 2
            wt = tf.get_variable('weights_{}_{}_i'.format(name, i), [ip_sh_h, op_sh_h], \
                                initializer=tf.truncated_normal_initializer(stddev=0.1, mean=-1e-4),
                                regularizer=l1_loss)
                                #initializer=tf.contrib.layers.xavier_initializer(),
                                #initializer=tf.truncated_normal_initializer(stddev=0.01),
            #print '{} - i = {}, wt shape : {}'.format(fname, i, wt.shape)
            wt_11 = tf.reshape(tf.stack([wt, tf.zeros_like(wt)], axis=1), [ip_sh, op_sh_h])
            wt_12 = tf.reshape(tf.stack([tf.zeros_like(wt), wt], axis=1), [ip_sh, op_sh_h])
            #print '{} - i = {}, wt1 shape : {}'.format(fname, i, wt_1.shape)
            wt_2 = tf.reshape(tf.stack([wt_11, wt_12], axis=-1), [ip_sh, op_sh], \
                name='weights_{}_{}'.format(name, i))
            #print '{} - i = {}, wt2 shape : {}'.format(fname, i, wt_2.shape)
            wt_list.append(wt_2)
            nop += ip_sh_h * op_sh_h
            x_tmp = tf.boolean_mask(x_int, t_int==i, axis=1)
            x_tmp = tf.reshape(x_tmp, [-1, ip_sh])
            x_tmp = tf.matmul(x_tmp, wt_2)
            x_op = tf.concat([x_op, x_tmp], axis = 1)

    x_op = tf.reshape(x_op, [-1, np.sum(t_op <= max_w), 1])

    print '{} - x op shape : {}'.format(fname, x_op.shape)
    print '{} - t op shape : {}'.format(fname, t_op.shape)
    print '{} - t total shape : {}'.format(fname, np.sum(t_op <= max_w))

    print '{} - t_op[t_op <= {}] : {}'.format(fname, max_w, t_op[t_op <= max_w])
    print '{} - ************************************ Number of Params : {}'.format(fname, nop)

    return x_op, t_op[t_op <= max_w], nop, wt_list

#def build_R1_for_rot(ang, cnt, n):
#    fname = sys._getframe().f_code.co_name
#
#    tot_dim = sum(cnt)
#    R = np.zeros((tot_dim, tot_dim))
#    o = 1.                                                                                                                           
#    cur_r = cnt[0] 
#    R[:cnt[0], :cnt[0]] = np.eye(cnt[0])
#    for c in cnt[1:]:         
#        r = np.zeros((2, 2))                                      
#        cc = np.cos(o * np.deg2rad(ang))
#        ss = np.sin(o * np.deg2rad(ang))
#        r[0, 0] = r[1, 1] = cc
#        r[0, 1] = -ss
#        r[1, 0] = ss
#        for i in range(c/2):
#            if o == 1. and i == 0:
#                R[cur_r:cur_r+2, cur_r:cur_r+2] = np.eye(2)
#            #print cur_r                                                                                                             
#            else:
#                R[cur_r:cur_r+2, cur_r:cur_r+2] = r
#            cur_r += 2
#        o += 1
#    return R

def rot_bat(X, Y, ip_n, op_n, mult=1):
    n_samp = X.shape[0]
    X1 = np.zeros_like(X)
    Y1 = np.zeros_like(Y)
    for j in range(mult):
        ang = np.random.rand(n_samp) * 360.
        for i in range(n_samp):
            #X1[i, :, 0] = build_R1_for_rot(ang[i], cnt_ip, ip_n).dot(X[i, :, 0])
            X1[i, :] = rotate(X[i, :].reshape(ip_n, ip_n, order='F'), ang[i], reshape=False).reshape(-1, order='F')
            Y1[i, :] = rotate(Y[i, :].reshape(op_n, op_n, order='F'), ang[i], reshape=False).reshape(-1, order='F')
        X = np.concatenate([X, X1], axis=0)
        Y = np.concatenate([Y, Y1], axis=0)
    return (X, Y)

def rot_bat_rgb(X, Y, ip_n, op_n, mult=1):
    n_samp = X.shape[0]
    X1 = np.zeros_like(X)
    Y1 = np.zeros_like(Y)
    for j in range(mult):
        ang = np.random.rand(n_samp) * 360.
        for i in range(n_samp):
            #X1[i, :, 0] = build_R1_for_rot(ang[i], cnt_ip, ip_n).dot(X[i, :, 0])
            X1[i] = rotate(X[i], ang[i], reshape=False)
            Y1[i] = rotate(Y[i], ang[i], reshape=False)
        X = np.concatenate([X, X1], axis=0)
        Y = np.concatenate([Y, Y1], axis=0)
    return (X, Y)

    
def fill_t(cnt): 
    t = np.array([0] * cnt[0])
    for i in range(1, cnt.shape[0]):
        mult = cnt[i]
        pos_t = [i] * mult
        t = np.concatenate((t, pos_t))
    return t

def gs_on(A):
    cols = A.shape[1]                                                                                
    B = np.zeros_like(A) 
    B[:, 0] = A[:, 0] / np.linalg.norm(A[:, 0]) 
    for i in range(1, cols):
        b = A[:, i]
        for j in range(i):
            b -= np.dot(B[:, j], b) * B[:, j]
        b /= np.linalg.norm(b)
        B[:, i] = b
    return B


