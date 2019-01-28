import os
import time
import sys
import json

#from __future__ import print_function
from argparse import Namespace

import numpy as np
import tensorflow as tf

from skimage.transform import resize, rotate

#from gen_R1 import *
#from t3_schur import full_quad_tensor
#from t3_schur import create_node

from s2_common import create_node, l1_loss, l2_loss
from s2_common import full_quad_tensor as full_quad_tensor
#from t3_schur import full_quad_tensor, create_node

# This file is used to define projW & supporting functions

def init_tf(gpu_percent=0.5, grow=True, thrds=15, seed=-1):
    config = tf.ConfigProto()
    
    config.intra_op_parallelism_threads = thrds
    config.inter_op_parallelism_threads = thrds

    config.gpu_options.per_process_gpu_memory_fraction = gpu_percent
    config.gpu_options.allow_growth = grow

    session = tf.Session(config=config)
    if seed != -1:
        tf.set_random_seed(seed)
        np.random.seed(seed)



def read_json(json_fname):
    
    with open(json_fname) as fp:
        jo = json.load(fp, object_hook=lambda d: Namespace(**d))
    
    with open(json_fname) as fp:
        print fp.read()
    
    if type(jo.test_data) != list:
        jo.test_data = [jo.test_data]
    if type(jo.op_fname) != list:
        jo.op_fname = [jo.op_fname]

    
    return jo

def proj_so2_W(X, W_fname, cnt_fname):
    w = np.load(W_fname)
    cnt = np.load(cnt_fname)
    wtx = X.dot(W)
    return wtx, cnt

def bn(inputs, t, training):
    fname = sys._getframe().f_code.co_name
    print '{} - inputs shape : {}'.format(fname, inputs.shape)
   
    w_0 = np.sum(t==0)

    ip_0 = inputs[:, :w_0]
    ip_1 = inputs[:, w_0:]
    
    print '{} - w_0 shape : {}'.format(fname, ip_0.shape)
    print '{} - rest shape : {}'.format(fname, ip_1.shape)
    
    op_0 = tf.layers.batch_normalization(ip_0, training=training, axis=1)

    outputs = tf.concat([op_0, ip_1], axis=1)
    print '{} - inputs shape : {}'.format(fname, outputs.shape)

    return outputs

def main():
    # Read json file into object
    # jo is the json object..

    jo = read_json(sys.argv[1])
    
    init_tf(thrds=10, seed=jo.seed)
    
    
    #Consts
    b_size = jo.b_size      #200 
    #epochs = 1000 / 4
    epochs = jo.epochs      #100
   
    valid_size = jo.v_size
    LEARNING_RATE = jo.l_rate
    KEEP_PROB = 1. - jo.dropout
    lr_alt_interval = jo.lr_alt_interval
    lr_decay_rate = jo.lr_decay_rate
    log_interval = jo.log_interval
    test_log_interval = jo.test_log_interval
    #BETA = 0
    
    ip_n = jo.ip_shape      #28
    op_n = jo.op_shape      #14

    tens_first = False
    if hasattr(jo, "tensor_first") and jo.tensor_first == True:
        tens_first = True
    scale_const = 1.
    if hasattr(jo, "scale_const"):
        scale_const = jo.scale_const
    BETA = 0.1
    if hasattr(jo, "beta"):
        BETA = jo.beta
    
    log_tm = time.strftime('%Y-%m-%d_%H_%M_%S')
    #log_dir = 'logs/ae_run-{}/'.format(log_tm)
    log_dir = jo.log_dir    #'logs/classify_14/'
    print '**************************'
    print log_dir
    print '**************************'

    
    # Step 2: Define paramaters for the model
   
    data = np.load(jo.train_data)    #np.load('mnist_train.npz')
    utr = data['data']
    tr_idx = data['labels']
    print 'utr.shape ', utr.shape
    data.close()
  
    # Add provision to add noise 
    # At training, and / or test

    if ip_n != op_n:
        ut_op = resize(utr.reshape(-1, ip_n, ip_n), (utr.shape[0], op_n, op_n)).reshape(-1, op_n**2)
    else:
        ut_op = utr.reshape(-1, op_n**2)

    X_train = np.float32(ut_op) / scale_const 
    y_train = np.identity(10)[tr_idx]
    
    data = np.load(jo.test_data[0])     #np.load('mnist_test.npz')
    uts = data['data']
    ts_idx = data['labels']
    data.close()

    #uts += np.random.randn(*uts.shape) * 0.2
    print np.max(uts), np.min(uts)
    print uts.dtype
    
    if ip_n != op_n:
        ut_op = resize(uts.reshape(-1, ip_n, ip_n), (uts.shape[0], op_n, op_n)).reshape(-1, op_n**2)
    else:
        ut_op = uts.reshape(-1, op_n**2)

    X_test = np.float32(ut_op) / scale_const
    y_test = np.identity(10)[ts_idx]

    #X_test = X_train[:]
    #y_test = y_train[:]
    
    #######################################
    #mn = np.mean(X_train)
    #sd = np.std(X_train)
    #X_train = (X_train - mn) / sd
    #X_test = (X_test - mn) / sd
    #######################################

    n_classes = np.unique(tr_idx).shape[0]
    
    def _fill_t(cnt): 
        t = np.array([0] * cnt[0])
        for i in range(1, cnt.shape[0]):
            mult = cnt[i]
            pos_t = [i] * mult
            t = np.concatenate((t, pos_t))
        return t

    # Depending upon which group we work on, we need to get
    # the appropriate fourier transform of the input
    # TODO

    w_ip = np.float32(np.load(jo.W_fname))
    cnt_ip = np.load(jo.cnt_fname)
    t_ip = _fill_t(cnt_ip)

    cnt_in = jo.cnt_in

    #print cnt_ip1
    #cnt_in1 = np.repeat([12, 10], 4)
    ##cnt_in1[0] = 30
    #cnt_in2 = np.repeat([14, 12, 10], 2)
    ##cnt_in2[0] = 40
    #cnt_in3 = np.repeat([14, 12], 2)
    ##cnt_in3[0] = 40
    #cnt_in4 = np.repeat([30], 1)
   
    n_layers = len(cnt_in)
    t_in = [0] * n_layers
    for i in range(n_layers):
        cnt_in[i] = np.array(cnt_in[i])
        t_in[i] = _fill_t(cnt_in[i])

    #cnt_op = np.array(jo.cnt_op)
    #t_op = _fill_t(cnt_op)
   
    print 'Tot input dims : ', np.sum(cnt_ip)
    print 'Input multiplicities : ', cnt_ip
    #print 'Tot output dims : ', np.sum(cnt_op)
    #print 'Output multiplicities : ', cnt_op

    ip_sh = X_train.shape[1]
    op_sh = y_train.shape[1]
    x = tf.placeholder(tf.float32, [None, ip_sh], name="X_placeholder")
    y_ = tf.placeholder(tf.float32, [None, op_sh], name="Y_placeholder")
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    train_phase = tf.placeholder(tf.bool, name="is_training")
    
    ################
    ## Project x onto the variable W_in
    ################

    # TODO
    x_1 = tf.matmul(x, w_ip)


    x_1 = tf.expand_dims(x_1, axis=-1)
    nop = 0

    wt = []

    if tens_first:
        x_i, t_i = full_quad_tensor(x_1, t_ip, keep_d1=True)
    else:
        x_i = x_1
        t_i = t_ip

    for i in range(n_layers):
        xx, tt, nop1, ww = create_node('LC{}'.format(i), x_i, t_i, t_in[i], tens=False)
        x_i, t_i = full_quad_tensor(xx, tt, keep_d1=True)
        x_i = bn(x_i, t_i, train_phase)
        wt.append(ww)
        nop += nop1
        print '****************************** No. of Params : {}'.format(nop1)

    
    #xx, tt, nop1, ww = create_node('LC{}'.format(i+1), x_i, t_i, t_op)
    #wt.append(ww)
    #nop += nop1
    #print '****************************** No. of Params : {}'.format(nop1)

    xx = tf.boolean_mask(x_i, t_i == 0, axis = 1)
    input_features = np.sum(t_i == 0)
    print 'Features at last level : {}'.format(input_features)
    #xx = x_i
    #input_features = t_i.shape[0]

    print xx.shape

    fc = tf.squeeze(xx, axis=-1)
    fc = tf.nn.dropout(fc, keep_prob)
    
    #with tf.variable_scope('softmax_linear') as scope:
    #input_features = np.prod(fc.get_shape().as_list()[1:])
    w = tf.get_variable('weights', [input_features, n_classes], \
                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                        regularizer=l1_loss)
                        #initializer=tf.contrib.layers.xavier_initializer(),
                        #initializer=tf.truncated_normal_initializer(stddev=0.01),
    b = tf.get_variable('biases', [n_classes], \
                        initializer=tf.constant_initializer(value=0.1))
    y_conv = tf.matmul(fc, w) + b
    nop += b.get_shape().as_list()[0] + np.prod(w.get_shape().as_list())

    print '****************************** Total No. of Params : {}  **********'.format(nop)

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    loss = cross_entropy
    b_reg = tf.reduce_mean(tf.concat(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), axis=0))
    loss += BETA * b_reg 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops): 
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    #train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
    #train_step = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cross_entropy)
    #train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    prediction = tf.argmax(y_conv, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    prob = tf.nn.softmax(y_conv, axis=1)
    
    X_valid, y_valid = X_test[-valid_size:], y_test[-valid_size:]
    #X_valid, y_valid = X_test[:], y_test[:]
    #X_test, y_test = X_test[:-valid_size], y_test[:-valid_size]

    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
#########################
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(epochs):
        idx_all = np.random.permutation(X_train.shape[0])
        tot_batches = X_train.shape[0] / b_size
        t1 = time.time()
        acc_tot = 0.
        cl = 0.
        for bb in range(tot_batches):
          idx = idx_all[bb*b_size:(bb+1)*b_size]
          X_batch, y_batch = X_train[idx], y_train[idx]
          #X_batch = ip_rot_bat(X_train[idx], 28)
          _, p_count, l = sess.run([train_step, correct_prediction, loss], \
                feed_dict = {x: X_batch, y_: y_batch, keep_prob : KEEP_PROB, learning_rate : LEARNING_RATE, train_phase : True})
          acc_tot += np.sum(p_count)
          cl += l
        # TODO
        acc_tot /= tot_batches * b_size  #X_train.shape[0]
        cl /= tot_batches
        #fc_vals = sess.run([fc_ip], feed_dict={x: X_valid[:1], y_: y_valid[:1], keep_prob: 1.})
        t2 = time.time()
        tr_acc = acc_tot
        
        if (i+1) % lr_alt_interval == 0:
            LEARNING_RATE = LEARNING_RATE * lr_decay_rate

        if i % test_log_interval == 0:
            print_test = '*** TEST\n'
            X_valid, y_valid = X_test[:], y_test[:]
        else:
            print_test = ''
            X_valid, y_valid = X_test[-valid_size:], y_test[-valid_size:]

        if i % log_interval == 0: 
            # Eval validation set
            idx_all = np.arange(X_valid.shape[0])
            tot_batches = X_valid.shape[0] / b_size
            va_acc = 0.
            cl_v = 0.
            for bb in range(tot_batches):
              idx = idx_all[bb*b_size:(bb+1)*b_size]
              X_batch, y_batch = X_valid[idx], y_valid[idx]
              #X_batch += np.random.randn(*X_batch.shape) * 0.1
              p_count, l_v  = sess.run([correct_prediction, loss], \
                    feed_dict = {x: X_batch, y_: y_batch, keep_prob: 1.0, train_phase : False})
              va_acc += np.sum(p_count)
              cl_v += l_v
            # TODO
            va_acc /= tot_batches * b_size  #X_valid.shape[0]
            cl_v /= tot_batches
            br = sess.run(b_reg)
            #va_acc = get_accuracy(X_valid, y_valid, b_size, correct_prediction)
            t3 = time.time()
            print 'Epoch: {:4d}    Accuracy: {:0.4f}, ' \
                  ' {:0.4f} Loss: {:0.4f}, {:0.4f}, '  \
                  'Time: {:0.2f}, {:0.2f} BReg: {:0.4f} LR : {:.2e} {}'.format(\
                  i+1, tr_acc, va_acc, cl, cl_v, t2-t1, t3-t2, br, LEARNING_RATE, print_test)
            #print 'Epoch: {:4d}    Accuracy: Train {:0.4f}, ' \
            #      'Valid {:0.4f}    Time: Train {:0.2f} Validate {:0.2f}'.format(\
            #      i+1, tr_acc, va_acc, t2-t1, t3-t2)

        #w_learnt_r = sess.run(w_ip_r)
        #w_learnt_i = sess.run(w_ip_i)
        #np.save('{}/Learnt_W_r_{}'.format(log_dir, op_n ** 2), w_learnt_r)
        #np.save('{}/Learnt_W_i_{}'.format(log_dir, op_n ** 2), w_learnt_i)
      # Eval test set
      #for tst_fname, op_fname in zip(jo.test_data, jo.op_fname):

      b1 = sess.run(b)
      print 'b'
      print b1

      for tst_fname in jo.test_data:
        data = np.load(tst_fname)
        uts = data['data']
        ts_idx = data['labels']
        data.close()
        if ip_n != op_n:
            ut_op = resize(uts.reshape(-1, ip_n, ip_n), (uts.shape[0], op_n, op_n)).reshape(-1, op_n**2)
        else:
            ut_op = uts.reshape(-1, op_n**2)
    
        X_test = np.float32(ut_op) / scale_const
        y_test = np.identity(10)[ts_idx]

        idx_all = np.arange(X_test.shape[0])
        tot_batches = X_test.shape[0] / b_size
        acc = 0.
        cl = 0.
        pred = np.array([100], dtype=np.int32)
        ts_data = np.zeros((X_test.shape[0], 12))
        for bb in range(tot_batches):
          idx = idx_all[bb*b_size:(bb+1)*b_size]
          X_batch, y_batch = X_test[idx], y_test[idx]
          #X_batch += np.random.randn(*X_batch.shape) * 0.1
          p_count, pp, l, pvec = sess.run([correct_prediction, prediction, loss, prob], \
                feed_dict = {x: X_batch, y_: y_batch, keep_prob: 1.0, train_phase : False})
          acc += np.sum(p_count)
          cl += l
          pred = np.concatenate([pred, pp])
          ts_data[idx, :10] = pvec
          ts_data[idx, 10] = pp
          ts_data[idx, 11] = ts_idx[idx]
        
        # TODO
        acc /= tot_batches * b_size  #X_test.shape[0]
        cl /= tot_batches
        pred = pred[1:]
        #va_acc = get_accuracy(X_valid, y_valid, b_size, correct_prediction)
        #acc = get_accuracy(X_test, y_test, b_size, correct_prediction)
        print '{} accuracy {}'.format(tst_fname, acc)
        print '{} loss {}'.format(tst_fname, cl)
        #np.save(op_fname, ts_data) 
     
      #wt_list1 = sess.run(wt_1)
      #wt_list2 = sess.run(wt_2)
      #wt_list3 = sess.run(wt_3)
      #wt_list4 = sess.run(wt_4)

      #print '{2}/wt_1_{0}x{0}_{1}'.format(op_n, i, log_dir)
      #for i in range(len(wt_list1)):
      #    np.save('{2}/wt_1_{0}x{0}_{1}'.format(op_n, i, log_dir), wt_list1[i])
      #for i in range(len(wt_list2)):
      #    np.save('{2}/wt_2_{0}x{0}_{1}'.format(op_n, i, log_dir), wt_list2[i])
      #for i in range(len(wt_list3)):
      #    np.save('{2}/wt_3_{0}x{0}_{1}'.format(op_n, i, log_dir), wt_list3[i])
      #for i in range(len(wt_list4)):
      #    np.save('{2}/wt_4_{0}x{0}_{1}'.format(op_n, i, log_dir), wt_list4[i])

#########################
#    with tf.Session() as sess:
#      sess.run(tf.global_variables_initializer())
#      t1 = time.time()
#      for i in range(epochs):
#        if rot_flag:
#            idx_all = np.random.permutation(X_train.shape[0] / 2)
#            tot_batches = X_train.shape[0] / (2 * b_size)
#        else:
#            idx_all = np.random.permutation(X_train.shape[0])
#            tot_batches = X_train.shape[0] / b_size
#        cum_loss = 0.
#        for b in range(tot_batches):
#          idx = idx_all[b*b_size:(b+1)*b_size]
#          #aa = np.random.rand(b_size) * 2 * np.pi
#          #aa = aa.reshape(-1, 1, 1, 1)
#          #aa = np.tile(aa, (1, 1, 1, sum(cnt_op[1:])))
#          #aa = np.float32(aa)
#
#          if rot_flag:
#            offset = X_train.shape[0] / 2
#            idx1 = idx_all[b*b_size:(b+1)*b_size] + offset
#            idx = np.concatenate([idx, idx1])
#
#          X_batch, y_batch = X_train[idx], y_train[idx]
#          _, l, r_r, r_i, r_o, ol = sess.run([train_step, loss, reg_r, reg_i, reg_op,
#                            old_loss], \
#                    feed_dict={x: X_batch, y_: y_batch})
#          cum_loss += l
#        if (i+1) % 1 == 0:
#            t2 = time.time()
#            print 'Epoch: {:4d}    Loss: {:0.4f}, reg_r: {:0.4f}, ' \
#                  'reg_i: {:0.4f}, reg_op: {:04f} old_loss: {:0.4f} Time: {:0.2f}'.format(\
#                i+1, l, r_r, r_i, r_o, ol, t2-t1)
#                #i+1, cum_loss, r1, t2-t1)
#            t1 = time.time()
#            w_learnt_r = sess.run(w_ip_r)
#            w_learnt_i = sess.run(w_ip_i)
#            np.save('{}/Learnt_W_r_{}'.format(log_dir, op_n ** 2), w_learnt_r)
#            np.save('{}/Learnt_W_i_{}'.format(log_dir, op_n ** 2), w_learnt_i)
#
#      w_learnt_r = sess.run(w_ip_r)
#      w_learnt_i = sess.run(w_ip_i)
#      np.save('{}/Learnt_W_r_{}'.format(log_dir, op_n ** 2), w_learnt_r)
#      np.save('{}/Learnt_W_i_{}'.format(log_dir, op_n ** 2), w_learnt_i)
#      
#      regen_img = sess.run(y_r, \
#                    feed_dict={x: X_batch, y_: y_batch, keep_prob: 1.0})
#
#      np.savez('{1}/regen_imgs_{0}x{0}'.format(op_n, log_dir), ip=y_batch, op=regen_img)
#

if __name__ == '__main__':
    main()
