import os
import time
import sys

import numpy as np
import tensorflow as tf

from skimage.transform import resize, rotate
from scipy.ndimage import rotate

from misc_fns import *
from s2_common import *

if __name__ == '__main__':

    jo = read_json(sys.argv[1])
    init_tf()
   
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
    print '**************************'
    print log_dir
    print '**************************'
    
    data = np.load(jo.train_data)    #np.load('mnist_train.npz')
    utr = data['data'][:s_size]

    # TODO = add support for rgb images
    utr = utr.reshape(-1, op_n**2, order='F')
    data.close()
 
    if ip_n != op_n:
        utr = resize(utr.reshape(-1, op_n, op_n), (utr.shape[0], ip_n, ip_n)).reshape(-1, ip_n**2)
    
    print 'utr.shape ', utr.shape

    X_train = np.float32(utr)
    y_train = np.float32(utr.copy())
 
    cnt_ip = np.repeat(jo.act_cnt, jo.act_cnt_rep)
    cnt_int = np.repeat(jo.scale_cnt, jo.scale_cnt_rep)
    cnt_op = np.repeat(jo.act_cnt, jo.act_cnt_rep)

    print 'cnt_ip : ', cnt_ip
    print 'sum(cnt_ip) : ', sum(cnt_ip)
    print 'cnt_int : ', cnt_int
    print 'sum(cnt_int) : ', sum(cnt_int)

    if jo.aug_count > 0:
        print 'Using generated rotations in input ********************************'
        X_train, y_train = rot_bat(X_train, y_train, ip_n, ip_n, jo.aug_count)
    rot_flag = False
    
    t_ip  = fill_t(cnt_ip) 
    t_op  = fill_t(cnt_op) 
    t_int = fill_t(cnt_int) 
    
    print cnt_ip
    print cnt_op
    print cnt_int

    ip_sh = X_train.shape[1]
    op_sh = y_train.shape[1]
    x = tf.placeholder(tf.float32, [None, ip_sh], name="X_placeholder")
    y_ = tf.placeholder(tf.float32, [None, op_sh], name="Y_placeholder")
    ang = tf.placeholder(tf.float32, [None, 1, 1, sum(cnt_op[1:])], name="ang_placeholder")
    
    ################
    ## Project x onto the variable W_in
    ################

    op_sh = t_ip.shape[0]
    wmat = np.random.randn(ip_n ** 2, op_sh)
    print 'wmat.shape ', wmat.shape
    w_init, _, _ = np.linalg.svd(wmat, full_matrices=False)
    print 'w_init.shape ', w_init.shape
    w_init = np.float32(w_init)
    w_ip = tf.get_variable('weights_ip', [ip_n ** 2, op_sh], \
                        initializer=tf.constant_initializer(w_init))
                        #initializer=tf.constant_initializer(w_init))
                        #initializer=tf.truncated_normal_initializer(stddev=0.1))
    #x_1 = tf.matmul(x, w_ip)
    #################
    # Rotate the image before projecting
    #################
    a_rot = ang[:, 0, 0, 0]
    x_r1 = tf.reshape(x, [-1, ip_n, ip_n, 1])
    x_r1 = tf.contrib.image.rotate(x_r1, a_rot)
    x_r1 = tf.reshape(x_r1, [-1, ip_n ** 2])

    x_1 = tf.matmul(x_r1, w_ip)
   
    x_1 = tf.expand_dims(x_1, axis=-1)
    wtw = tf.matmul(w_ip, w_ip, transpose_a=True)
    print wtw.shape
    id_wt = np.eye(op_sh)
    reg2 = tf.reduce_mean(tf.abs(id_wt - wtw))
    
    x_int2, t_int2, wt_int2, _ = create_node('node2', x_1, t_ip, t_int, tens=False)
    
    x_int1, t_int1, wt_int1, _ = create_node('node1', x_int2, t_int2, t_op)
    
    fc = tf.reshape(x_int1, [-1, t_int1.shape[0]])
    
    op_sh = t_int1.shape[0]
    
    ##############################
    #  Apply unrotate by doint WR to fc

    fc1 = tf.reshape(fc, [-1, 1, 1, op_sh])
    a1 = tf.transpose(ang, [3, 0, 1, 2])
    rfc = tf_rot_inv_cust.useRinv(fc1, ip_n, cnt=cnt_op, ang=a1)
    rfc = tf.reshape(rfc, [-1, op_sh])

    y_conv = tf.matmul(rfc, w_ip, transpose_b=True)

    #recons_err = tf.reduce_mean(((y_ - y_conv) ** 2))
    recons_err = tf.reduce_mean(l1_loss(y_ - y_conv))
    loss = R * recons_err + B * reg2
    b_reg = tf.reduce_mean(tf.concat(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), axis=0))
    loss += BETA * b_reg 
    #loss = 2000 * old_loss + BETA2 * reg2
    #loss = -tf.log(tf.minimum(loss, 1e-7))
    #loss = 100 * old_loss + BETA1 * reg1
            #+ tf.reduce_mean(tf.abs(y_conv) - y_conv)
            
    #loss = tf.reduce_mean(tf.abs(y_ - y_conv)) \
    #        + BETA * reg1
    
    #loss = tf.reduce_mean(
    #    tf.nn.l2_loss(y_ - y_conv)) + BETA * reg1
    #train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(loss)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    #train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    #train_step = tf.train.AdagradOptimizer(1).minimize(loss)

    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      t1 = time.time()
      for i in range(epochs):
        idx_all = np.random.permutation(X_train.shape[0])
        tot_batches = X_train.shape[0] / b_size
        cum_loss = 0.
        for b in range(tot_batches):
          idx = idx_all[b*b_size:(b+1)*b_size]
          aa = np.random.rand(b_size) * 2 * np.pi
          aa = aa.reshape(-1, 1, 1, 1)
          aa = np.tile(aa, (1, 1, 1, sum(cnt_op[1:])))
          aa = np.float32(aa)

          X_batch, y_batch = X_train[idx], y_train[idx]
          _, l, r1, ol, br = sess.run([train_step, loss, reg2, recons_err, b_reg], \
                    feed_dict={x: X_batch, y_: y_batch, ang: aa})
          cum_loss += l
        if (i+1) % 1 == 0:
            t2 = time.time()
            print 'Epoch: {:4d}   Recons: {:0.4f}, Reg: {:0.4f}, ' \
                  ' Breg: {:0.4f} Time: {:0.2f}'.format(\
                i+1, ol, r1, br, t2-t1)
                #i+1, cum_loss, r1, t2-t1)
            t1 = time.time()
            w_learnt = sess.run(w_ip)
            np.save('{}/Learnt_W_{}'.format(log_dir, ip_n ** 2), w_learnt)
      w_learnt = sess.run(w_ip)
      np.save('{}/Learnt_W_{}'.format(log_dir, ip_n ** 2), w_learnt)
      print w_learnt.shape
      W = gs_on(w_learnt) 
      np.save('{}/W_{}x{}'.format(log_dir, ip_n ** 2, op_sh), W)
      np.save('{}/cnt_{}x{}'.format(log_dir, ip_n ** 2, op_sh), cnt_op)
      #regen_img = sess.run(y_conv, \
      #              feed_dict={x: X_batch, y_: y_batch, keep_prob: 1.0})
      #wt_list1 = sess.run(wt_int1) #, \
      #              feed_dict={x: X_train[reg_idx], y_: y_train[reg_idx]})
      #wt_list2 = sess.run(wt_int2, \
      #              feed_dict={x: X_train[reg_idx], y_: y_train[reg_idx]})
      #np.savez('{1}/regen_imgs_{0}x{0}'.format(op_n, log_dir), ip=y_batch, op=regen_img)
      #for i in range(len(wt_list1)):
      #    np.save('{2}/wt_int1_{0}x{0}_{1}'.format(op_n, i, log_dir), wt_list1[i])
      #for i in range(len(wt_list2)):
      #    np.save('wt_int2_{}'.format(i), wt_list2[i])
      #saver.save(sess, '{}my_model.ckpt'.format(log_dir))


