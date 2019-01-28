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
 
    R14 = jo.scale_recons
    R28 = jo.act_recons
    B14 = jo.scale_reg
    B28 = jo.act_reg

    BETA = 0.1
    if hasattr(jo, "beta"):
        BETA = jo.beta
    BETA *= (s_size / 10000.)

    log_tm = time.strftime('%Y-%m-%d_%H_%M_%S')
    #log_dir = 'logs/ae_known_ang-{}/'.format(log_tm)
    #log_dir = 'logs/ae_known_ang/'
    log_dir = '{}_cae_{}_{}'.format(jo.log_dir_prefix, jo.aug_count, s_size)
    print '**************************'
    print log_dir
    print '**************************'
    
    data = np.load(jo.train_data)    #np.load('mnist_train.npz')
    utr_op = data['data'][:s_size]

    # TODO = add support for rgb images
    utr_op = utr_op.reshape(-1, op_n**2, order='F')
    data.close()

    if op_n != ip_n:
        utr_ip = resize(utr_op.reshape(-1, op_n, op_n), (utr_op.shape[0], ip_n, ip_n)).reshape(-1, ip_n**2)
    else:
        utr_ip = utr_op.copy()
    
    print 'utr_act.shape ', utr_op.shape
    print 'utr_scale_shape ', utr_ip.shape

    X_train = np.float32(utr_ip)
    y_train = np.float32(utr_op)
 
    cnt_ip = np.repeat(jo.scale_cnt, jo.scale_cnt_rep)
    cnt_op = np.repeat(jo.act_cnt, jo.act_cnt_rep)

    print 'cnt_ip : ', cnt_ip
    print 'sum(cnt_ip) : ', sum(cnt_ip)
    print 'cnt_op : ', cnt_op
    print 'sum(cnt_op) : ', sum(cnt_op)

    if jo.aug_count > 0:
        print 'Using generated rotations in input ********************************'
        #utr_ip, utr_op = rot_bat(utr_ip, utr_op, mult)
        X_train, y_train = rot_bat(X_train, y_train, ip_n, op_n, jo.aug_count)
    rot_flag = False
    
    t_ip  = fill_t(cnt_ip)
    t_op  = fill_t(cnt_op)

    print cnt_ip
    print cnt_op

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
    #w_init = np.load('W_196.npy')
    #ip_sh = np.sum(t_int <= max_w)
    w_ip = tf.get_variable('weights_ip', [ip_n ** 2, op_sh], \
                        initializer=tf.constant_initializer(w_init))
                        #initializer=tf.constant_initializer(w_init))
                        #initializer=tf.truncated_normal_initializer(stddev=0.1))
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
    reg_w_14 = tf.reduce_mean(tf.abs(id_wt - wtw))
    
    #x_2, t_2, wt_2 = create_node('lc1', x_1, t_ip, t_int, tens=False)
    
    x_3, t_3, wt_3, wt_14_28 = create_node('tens_lc1', x_1, t_ip, t_op)
    
    # Cut down the extra dim added for tensoring from x
    x_3 = tf.squeeze(x_3, axis=-1)

    #fc = tf.reshape(x_3, [-1, t_3.shape[0]])
    
    op_sh = t_3.shape[0]
    wmat = np.random.randn(op_sh, op_n ** 2)
    print 'wmat.shape ', wmat.shape
    _, _, w_init = np.linalg.svd(wmat, full_matrices=False)
    print 'w_init.shape ', w_init.shape
    w_init = np.float32(w_init)
    w_op = tf.get_variable('weights_op', [op_sh, op_n ** 2], \
                         initializer=tf.constant_initializer(w_init))
    #                    #initializer=tf.constant_initializer(w_init))
    #                    #initializer=tf.truncated_normal_initializer(stddev=0.1))
        
    ##############################
    #  Apply unrotate by doint WR to fc

    x_3 = tf.reshape(x_3, [-1, 1, 1, op_sh])
    a1 = tf.transpose(ang, [3, 0, 1, 2])
    x_4 = tf_rot_inv_cust.useRinv(x_3, op_n, cnt=cnt_op, ang=a1)
    x_4 = tf.reshape(x_4, [-1, op_sh])

    y_conv = tf.matmul(x_4, w_op)
    wtw_o = tf.matmul(w_op, w_op, transpose_b=True)
    print wtw_o.shape
    id_wt = np.eye(op_sh)
    reg_w_28 = tf.reduce_mean(tf.abs(id_wt - wtw_o))

    ###############################
    # Create a reverse network.. 28 to 14
    
    #################
    # Rotate the image before projecting
    #################
    y_r1 = tf.reshape(y_, [-1, op_n, op_n, 1])
    y_r1 = tf.contrib.image.rotate(y_r1, a_rot)
    y_r1 = tf.reshape(y_r1, [-1, op_n ** 2])

    y_1 = tf.matmul(y_r1, w_op, transpose_b=True)
   
    y_1 = tf.expand_dims(y_1, axis=-1)
    
    #y_2, t_y_2, wt_y_2 = create_node('lc_y_1', y_1, t_op, t_int, tens=False)
    y_2, t_y_2, wt_y_2, wt_28_14 = create_node('lc_y_1', y_1, t_op, t_ip, tens=False)
    
    y_3, t_y_3 = y_2, t_y_2
    #y_3, t_y_3, wt_y_3 = create_node('tens_lc_y_1', y_2, t_y_2, t_ip)
    
    # Cut down the extra dim added for tensoring from x
    y_3 = tf.squeeze(y_3, axis=-1)

    #fc = tf.reshape(x_3, [-1, t_3.shape[0]])
    
    ip_sh = t_y_3.shape[0]
    ##############################
    #  Apply unrotate by doint WR to fc

    y_3 = tf.reshape(y_3, [-1, 1, 1, ip_sh])
    a1 = tf.transpose(ang[:, :, :, :sum(cnt_ip[1:])], [3, 0, 1, 2])
    print 'a1 shape : ', a1.shape
    y_4 = tf_rot_inv_cust.useRinv(y_3, ip_n, cnt=cnt_ip, ang=a1)
    y_4 = tf.reshape(y_4, [-1, ip_sh])

    x_conv = tf.matmul(y_4, w_ip, transpose_b=True)

    recon_err_14 = tf.reduce_mean(((x - x_conv) ** 2))
    #### TODO
    #recon_err_14 = tf.reduce_mean(tf.abs(x - x_conv))


    ###############################

    recon_err_28 = tf.reduce_mean(((y_ - y_conv) ** 2))
    #### TODO
    #recon_err_28 = tf.reduce_mean(tf.abs(y_ - y_conv))

    print 'x conv ', x_conv.shape
    print 'y conv ', y_conv.shape
    print ' w_ip ', w_ip.shape
    print ' w_op ', w_op.shape

    loss = R28 * recon_err_28 + B28 * reg_w_28 + B14 * reg_w_14
    loss = loss + R14 * recon_err_14
    b_reg = tf.reduce_mean(tf.concat(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), axis=0))
    loss += BETA * b_reg 
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
          # Random angles for each batch
          aa = np.random.rand(b_size) * 2 * np.pi
          aa = aa.reshape(-1, 1, 1, 1)
          aa = np.tile(aa, (1, 1, 1, sum(cnt_op[1:])))
          aa = np.float32(aa)

          X_batch, y_batch = X_train[idx], y_train[idx]
          _, l, rw14, rw28, r14, r28, br = sess.run([train_step, loss, reg_w_14, reg_w_28, recon_err_14, recon_err_28, b_reg], \
                    feed_dict={x: X_batch, y_: y_batch, ang: aa})
          cum_loss += l
        if (i+1) % 1 == 0:
            t2 = time.time()
            print 'Epoch: {:4d}  Recons 14,28: {:0.4f} {:0.04f}, ' \
                  'Reg 14,28: {:0.4f} {:0.4f}, BReg: {:0.4f} Time: {:0.2f}'.format(\
                i+1, r14, r28, rw14, rw28, br, t2-t1)
                #i+1, cum_loss, r1, t2-t1)
            t1 = time.time()
            w_ip_learnt = sess.run(w_ip)
            w_op_learnt = sess.run(w_op)
            w_op_learnt = w_op_learnt.T
            np.save('{}/Learnt_W_{}'.format(log_dir, op_n ** 2), w_op_learnt)
            np.save('{}/Learnt_W_{}'.format(log_dir, ip_n ** 2), w_ip_learnt)


      # Save the weights (phi and psi maps)

      wt_14_28_learnt = sess.run(wt_14_28)
      wt_28_14_learnt = sess.run(wt_28_14)
      np.save('{}/map_{}_to_{}'.format(log_dir, ip_n, op_n), wt_14_28_learnt)
      np.save('{}/map_{}_to_{}'.format(log_dir, op_n, ip_n), wt_28_14_learnt)

      w_ip_learnt = sess.run(w_ip)
      w_op_learnt = sess.run(w_op)
      w_op_learnt = w_op_learnt.T

      np.save('{}/Learnt_W_{}'.format(log_dir, op_n ** 2), w_op_learnt)
      np.save('{}/Learnt_W_{}'.format(log_dir, ip_n ** 2), w_ip_learnt)
      W1 = gs_on(w_ip_learnt) 
      np.save('{}/W_{}x{}'.format(log_dir, ip_n ** 2, t_ip.shape[0]), W1)
      np.save('{}/cnt_{}x{}'.format(log_dir, ip_n ** 2, t_ip.shape[0]), cnt_ip)
      W2 = gs_on(w_op_learnt) 
      np.save('{}/W_{}x{}'.format(log_dir, op_n ** 2, t_op.shape[0]), W2)
      np.save('{}/cnt_{}x{}'.format(log_dir, op_n ** 2, t_op.shape[0]), cnt_op)
      #w_learnt = sess.run(w)
      #np.save('{}/Learnt_W_{}'.format(log_dir, op_n ** 2), w_learnt)
      #print w_learnt.shape
      #regen_img = sess.run(y_conv, \
      #              feed_dict={x: X_batch, y_: y_batch, ang:aa, keep_prob: 1.0})
      #np.savez('{1}/regen_imgs_{0}x{0}'.format(op_n, log_dir), ip=y_batch, op=regen_img)
      #regen_img = sess.run(x_conv, \
      #              feed_dict={x: X_batch, y_: y_batch, ang:aa, keep_prob: 1.0})
      #np.savez('{1}/regen_imgs_{0}x{0}'.format(ip_n, log_dir), ip=X_batch, op=regen_img)
      #wt_list1, wt_list2 = sess.run(wt_1, wt_2)
      #for i in range(len(wt_list1)):
      #    np.save('{2}/wt_int1_{0}x{0}_{1}'.format(op_n, i, log_dir), wt_list1[i])
      #for i in range(len(wt_list2)):
      #    np.save('wt_int2_{}'.format(i), wt_list2[i])
      #saver.save(sess, '{}my_model.ckpt'.format(log_dir))


