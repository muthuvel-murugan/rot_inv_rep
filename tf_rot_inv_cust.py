import os
import time
import sys

import numpy as np
import tensorflow as tf
#from SO3Features_opt_keras import Y_m_l_const_n, load_cg
#from SO3Features_opt_keras import SO3Features_opt, Y_m_l_const

# This file is used to define projW & supporting functions

def init_tf(gpu_percent=0.5, grow=True):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_percent
    config.gpu_options.allow_growth = grow
    session = tf.Session(config=config)
    tf.set_random_seed(1)
    np.random.seed(1)

def block_diagonal(matrices, dtype=tf.float32):
  r"""Constructs block-diagonal matrices from a list of batched 2D tensors.

  Args:
    matrices: A list of Tensors with shape [..., N_i, M_i] (i.e. a list of
      matrices with the same batch dimension).
    dtype: Data type to use. The Tensors in `matrices` must match this dtype.
  Returns:
    A matrix with the input matrices stacked along its main diagonal, having
    shape [..., \sum_i N_i, \sum_i M_i].

  """
  matrices = [tf.convert_to_tensor(matrix, dtype=dtype) for matrix in matrices]
  blocked_rows = tf.Dimension(0)
  blocked_cols = tf.Dimension(0)
  batch_shape = tf.TensorShape(None)
  for matrix in matrices:
    full_matrix_shape = matrix.get_shape().with_rank_at_least(2)
    batch_shape = batch_shape.merge_with(full_matrix_shape[:-2])
    blocked_rows += full_matrix_shape[-2]
    blocked_cols += full_matrix_shape[-1]
  ret_columns_list = []
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    ret_columns_list.append(matrix_shape[-1])
  ret_columns = tf.add_n(ret_columns_list)
  row_blocks = []
  current_column = 0
  for matrix in matrices:
    matrix_shape = tf.shape(matrix)
    row_before_length = current_column
    current_column += matrix_shape[-1]
    row_after_length = ret_columns - current_column
    row_blocks.append(tf.pad(
        tensor=matrix,
        paddings=tf.concat(
            [tf.zeros([tf.rank(matrix) - 1, 2], dtype=tf.int32),
             [(row_before_length, row_after_length)]],
            axis=0)))
  blocked = tf.concat(row_blocks, -2)
  blocked.set_shape(batch_shape.concatenate((blocked_rows, blocked_cols)))
  return blocked


def projW(inputs, n, max_o=-1, padding='VALID', strides=[1, 1, 1, 1]):
    fname = sys._getframe().f_code.co_name
    nsq = n*n
    ch = int(inputs.get_shape()[-1])
    print '{} - inputs shape : {}'.format(fname, inputs.shape)

    W = np.load('W_{}.npy'.format(nsq))
    cnt = np.load('cnt_{}.npy'.format(nsq))

    if max_o != -1:
        cnt = cnt[:max_o+1]
    
    tot_cnt = np.sum(cnt)
    W = W[:, :tot_cnt]
    
    print '{} - ch : {}'.format(fname, ch)
    print '{} - W shape : {}'.format(fname, W.shape)

    W = W.reshape(n, n, 1, -1)
    W = np.float32(W)

    W_t = tf.tile(W, [1, 1, ch, 1])

    print '{} - W_t shape : {}'.format(fname, W_t.shape)

    outputs = tf.nn.depthwise_conv2d(inputs, W_t, strides, padding)
    print '{} - outputs shape : {}'.format(fname, outputs.shape)

    return outputs


def onlyOmega0(inputs, n, max_o=-1):
    fname = sys._getframe().f_code.co_name
    nsq = n*n
    print '{} - inputs shape : {}'.format(fname, inputs.shape)

    cnt = np.load('cnt_{}.npy'.format(nsq))
    if max_o != -1:
        cnt = cnt[:max_o+1]
    
    tot_cnt = np.sum(cnt)
    tot_ch = int(inputs.get_shape()[-1])
    ch = tot_ch / tot_cnt

    print '{} - ch : {}'.format(fname, ch)

    w_0_ind = np.arange(cnt[0], dtype=np.int32)
    ind = w_0_ind.copy()
    print '{} - ind : {}'.format(fname, ind)
    for i in range(1, ch):
        ind = np.concatenate((ind, ((i * int(np.sum(cnt))) + w_0_ind)))
    
    print '{} - ind : {}'.format(fname, ind)
    print '{} - ind.shape : {}'.format(fname, ind.shape)

    inputs = tf.transpose(inputs, [3, 0, 1, 2])
    print '{} - inputs shape : {}'.format(fname, inputs.shape)
    outputs = tf.gather(inputs, ind)
    outputs = tf.transpose(outputs, [1, 2, 3, 0])

    return outputs


def useNorm(inputs, n, max_o=-1):
    fname = sys._getframe().f_code.co_name
    nsq = n*n
    print '{} - inputs shape : {}'.format(fname, inputs.shape)

    cnt = np.load('cnt_{}.npy'.format(nsq))

    if max_o != -1:
        cnt = cnt[:max_o+1]
   
    tot_cnt = np.sum(cnt)
    tot_ch = int(inputs.get_shape()[-1])
    ch = tot_ch / tot_cnt

    print '{} - ch : {}'.format(fname, ch)

    w_0_ind = np.arange(cnt[0], dtype=np.int32)
    ind_0 = w_0_ind.copy()
    for i in range(1, ch):
        ind_0 = np.concatenate((ind_0, ((i * int(np.sum(cnt))) + w_0_ind)))

    mask_1 = np.ones(tot_ch, dtype=np.bool)
    mask_1[ind_0] = False
    ind_1 = np.arange(tot_ch, dtype=np.int32)[mask_1]

    print '{} - ind0 : {}'.format(fname, ind_0)
    print '{} - ind0.shape : {}'.format(fname, ind_0.shape)
    print '{} - ind1 : {}'.format(fname, ind_1)
    print '{} - ind1.shape : {}'.format(fname, ind_1.shape)
    
    inputs = tf.transpose(inputs, [3, 0, 1, 2])
    print '{} - inputs shape : {}'.format(fname, inputs.shape)
    op_0 = tf.gather(inputs, ind_0)

    op_1 = tf.gather(inputs, ind_1)
    op_1 = op_1 * op_1
    op_1 = op_1[::2] + op_1[1::2]
    op_1 = tf.sqrt(op_1)

    outputs = tf.concat([op_0, op_1], axis=0)
    outputs = tf.transpose(outputs, [1, 2, 3, 0])
    print '{} - outputs shape : {}'.format(fname, outputs.shape)

    return outputs

def useRinv(inputs, n, max_o=-1, cnt=None, ang=None):
    fname = sys._getframe().f_code.co_name
    nsq = n*n
    print '{} - inputs shape : {}'.format(fname, inputs.shape)

    #if cnt == None:
    #    cnt = np.load('cnt_{}.npy'.format(nsq))

    if max_o != -1:
        cnt = cnt[:max_o+1]
   
    tot_cnt = np.sum(cnt)
    tot_ch = int(inputs.get_shape()[-1])
    ch = tot_ch / tot_cnt

    print '{} - ch : {}'.format(fname, ch)

    c = cnt[1:]
    
    ##x_mask = np.zeros(tot_cnt, dtype=np.bool)
    ##y_mask = np.zeros(tot_cnt, dtype=np.bool)
    #mask = [1] * tot_cnt
    #cc = cnt[0]
    #for o in cnt[1:]:
    #    #x_mask[cc] = True
    #    #y_mask[cc+1] = True
    #    mask[cc+1] = 0
    #    cc += o
    #mask = mask * ch
    #mask = np.arange(tot_ch)[np.array(mask, dtype=np.bool)]

    #print '{} - mask : {}'.format(fname, mask)


    w_0_ind = np.arange(cnt[0], dtype=np.int32)
    ind_0 = w_0_ind.copy()
    for i in range(1, ch):
        ind_0 = np.concatenate((ind_0, ((i * int(np.sum(cnt))) + w_0_ind)))

    #cc = [0] * c[0]
    #cind = c[0]
    #for i in c[1:]:
    #    cc.extend([cind] * i)
    #    cind += i

    use_o = 1

    cc = [np.sum(c[:use_o-1])] * sum(c)
    cc = np.array(cc)

    cc1 = np.array(cc, dtype=np.int32)
    for i in range(1, ch):
        cc1 = np.concatenate((cc1, i*sum(c)+cc))
    
    scale = [1. / use_o] * c[0]
    cind = c[0]
    j = 2
    for i in c[1:]:
        scale.extend([np.float32(j) / use_o] * i)
        cind += i
        j += 1
    
    scale = scale * ch
    scale = np.array(scale, dtype=np.float32)

    print '{} - cc1 : {}'.format(fname, cc1)
    print '{} - scale : {}'.format(fname, scale)

    mask_1 = np.ones(tot_ch, dtype=np.bool)
    mask_1[ind_0] = False
    ind_1 = np.arange(tot_ch, dtype=np.int32)[mask_1]

    print '{} - ind0 : {}'.format(fname, ind_0)
    print '{} - ind0.shape : {}'.format(fname, ind_0.shape)
    print '{} - ind1 : {}'.format(fname, ind_1)
    print '{} - ind1.shape : {}'.format(fname, ind_1.shape)
    
    inputs = tf.transpose(inputs, [3, 0, 1, 2])
    print '{} - inputs shape : {}'.format(fname, inputs.shape)
    op_0 = tf.gather(inputs, ind_0)

    ip_1 = tf.gather(inputs, ind_1)   
    ip_1_x = tf.gather(ip_1, cc1)
    ip_1_y = tf.gather(ip_1, cc1+1)
    #if ang == None:
    #ang = tf.atan2(ip_1_y, ip_1_x)
    print '{} - ang shape : {}'.format(fname, ang.shape)
    ang = tf.transpose(ang, [1, 2, 3, 0])
    print '{} - ang shape : {}'.format(fname, ang.shape)
    print '{} - scale shape : {}'.format(fname, scale.shape)
    ang = scale * ang
    ang = tf.transpose(ang, [3, 0, 1, 2])
    cs = tf.cos(ang)
    ss = tf.sin(ang)
    ss = tf.transpose(ss, [1, 2, 3, 0])
    sg = np.ones(sum(c) * ch, dtype=np.float32)
    print '{} - sg shape : {}'.format(fname, sg.shape)
    print '{} - ss shape : {}'.format(fname, ss.shape)
    sg[1::2] = -1.
    ss = sg * ss
    ss = tf.transpose(ss, [3, 0, 1, 2])
    ind1 = np.arange(sum(c) * ch, dtype=np.int32)
    ind2 = np.ones_like(ind1)
    ind2[::2] = ind1[1::2]
    ind2[1::2] = ind1[::2]
    ip_2 = tf.gather(ip_1, ind2)
    op_1 = cs * ip_1 + ss * ip_2

    outputs = tf.concat([op_0, op_1], axis=0)
    #outputs = tf.gather(outputs, mask)
    outputs = tf.transpose(outputs, [1, 2, 3, 0])
    print '{} - outputs shape : {}'.format(fname, outputs.shape)

    #outputs = tf.nn.l2_normalize(outputs, -1)

    return outputs

def _useRinv(inputs, n, max_o=-1):
    fname = sys._getframe().f_code.co_name
    nsq = n*n
    print '{} - inputs shape : {}'.format(fname, inputs.shape)

    cnt = np.load('cnt_{}.npy'.format(nsq))

    if max_o != -1:
        cnt = cnt[:max_o+1]
   
    tot_cnt = np.sum(cnt)
    tot_ch = int(inputs.get_shape()[-1])
    ch = tot_ch / tot_cnt

    print '{} - ch : {}'.format(fname, ch)
    
    x_mask = np.zeros(tot_cnt, dtype=np.bool)
    y_mask = np.zeros(tot_cnt, dtype=np.bool)
    c = cnt[0]
    for o in cnt[1:]:
        x_mask[c] = True
        y_mask[c+1] = True
        c += o
    mask = np.logical_not(y_mask)

    def _eachCh(ip):
        fname = sys._getframe().f_code.co_name
        print '{} - Input Shape : {}'.format(fname, ip.get_shape())
    
        tot_cols = tot_cnt
        #R = np.zeros((tot_cols, tot_cols))
        #ang = np.zeros_like(cnt, dtype=np.float32)
        ang = -1. * tf.atan2(tf.boolean_mask(ip, y_mask), tf.boolean_mask(ip, x_mask))
        print '{} - cnt shape : {}'.format(fname, cnt)
        print '{} - ang shape : {}'.format(fname, ang.shape)

        blks = []
        blks.append(tf.eye(cnt[0]))

        j = 0
        for o in cnt[1:]:
            o /= 2
            c_m = np.array([1.,  0, 0, 1])
            s_m = np.array([0., -1, 1, 0])
            blk = c_m * tf.cos(ang[j]) + s_m * tf.sin(ang[j])
            blk = tf.reshape(blk, [2, 2])
            j += 1
            for i in range(o):
                blks.append(blk)

        R = block_diagonal(blks) 
        ip = tf.reshape(ip, [-1, 1])
        print '{} - ip Shape : {}'.format(fname, ip.get_shape())
        op = tf.matmul(R, ip)
        print '{} - op Shape : {}'.format(fname, op.get_shape())
        op = tf.reshape(op, [-1])
        print '{} - op Shape : {}'.format(fname, op.get_shape())
        #op = tf.boolean_mask(op, mask)
        #print '{} - op Shape : {}'.format(fname, op.get_shape())
        return op

    def _eachPatch(ip):
        fname = sys._getframe().f_code.co_name
        print '{} - Input Shape : {}'.format(fname, ip.get_shape())
        ip = tf.reshape(ip, [-1, tot_cnt])
        op = tf.map_fn(_eachCh, ip)
        print '{} - op Shape : {}'.format(fname, op.get_shape())
        op = tf.reshape(op, [-1])
        print '{} - op Shape : {}'.format(fname, op.get_shape())
        return op

    def _eachSamp(ip):
        fname = sys._getframe().f_code.co_name
        print '{} - Input Shape : {}'.format(fname, ip.get_shape())
        sh = ip.get_shape().as_list()
        sh[1] = sh[0] * sh[1]
        ip = tf.reshape(ip, sh[1:])
        op = tf.map_fn(_eachPatch, ip)
        print '{} - op Shape : {}'.format(fname, op.get_shape())
        sh[1] = sh[1] / sh[0]
        op = tf.reshape(op, sh)
        print '{} - op Shape : {}'.format(fname, op.get_shape())
        return op

    
    outputs = tf.map_fn(_eachSamp, inputs)
    print '{} - outputs Shape : {}'.format(fname, outputs.get_shape())
    return outputs


def get_accuracy(data, lbl, b_size, pred_var):
    tot_bat = data.shape[0] / b_size 
    cum_pred = 0.
    for i in range(tot_bat):
        idx = np.arange(b_size, dtype=np.int) + i*b_size
        X_batch, y_batch = data[idx], lbl[idx]
        cum_pred +=  np.sum(pred_var.eval(feed_dict={  \
            x: X_batch, y_: y_batch, keep_prob: 1.0}))
        if data.shape[0] % b_size != 0:
            X_batch, y_batch = data[i*b_size:], lbl[i*b_size:]
            cum_pred +=  np.sum(pred_var.eval(feed_dict={
                x: X_batch, y_: y_batch, keep_prob: 1.0}))

    return cum_pred / data.shape[0]


if __name__ == '__main__':
# Main routine
# Code skeleton taken from
# https://github.com/chiphuyen/stanford-tensorflow-tutorials/blob/master/examples/07_convnet_mnist.py
    
    init_tf()
    
    
    #Consts
    b_size = 1000 
    epochs = 200
    n = 9
   
    valid_size = 1000
    LEARNING_RATE = 0.01
    DROPOUT = 0.5
    N_EPOCHS = 10
    
    log_tm = time.strftime('%Y-%m-%d %H:%m:%s')
    log_dir = 'logs/run-{}/'.format(log_tm)
    
    # Step 2: Define paramaters for the model
    
    # ROTATED
    utr = np.load('data/mnist_rot_train_data.npy') 
    # STRAIGHT
    #utr = np.load('data/mnist_unrot_train_data.npy') 
    tr_idx = np.load('data/mnist_train_label.npy')
    X_train = np.float32(utr)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    y_train = np.identity(10)[tr_idx]
    
    # ROTATED
    uts = np.load('data/mnist_rot_test_data.npy')
    # STRAIGHT
    #uts = np.load('data/mnist_unrot_test_data.npy')
    
    ts_idx = np.load('data/mnist_test_label.npy')
    X_test = np.float32(uts)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    y_test = np.identity(10)[ts_idx]
    
    n_classes = np.unique(tr_idx).shape[0]
    
    
    with tf.name_scope('data'):
        x = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X_placeholder")
        y_ = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")
        keep_prob = tf.placeholder(tf.float32)
    
    #dropout = tf.placeholder(tf.float32, name='dropout')
    
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    with tf.variable_scope('conv1') as scope:
        
        max_o = 6
        op_ch = 128
        Wtx1 = projW(x, n, max_o)
        #Wtx1_0 = onlyOmega0(Wtx1, n)
        #Wtx1_0 = useNorm(Wtx1, n, max_o)
        Wtx1_0 = useRinv(Wtx1, n, max_o)
        #Wtx1_0 = Wtx1
        
        ip_ch = Wtx1_0.get_shape()[-1]    
        
        kernel1 = tf.get_variable('kernel', [1, 1, ip_ch, op_ch], \
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases1 = tf.get_variable('biases', [op_ch], \
                            initializer=tf.constant_initializer(value=0.1))
        conv = tf.nn.conv2d(Wtx1_0, kernel1, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv + biases1, name=scope.name)
    
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
    
    with tf.variable_scope('conv2') as scope:
        
        max_o = 6    
        op_ch = 256
        Wtx2 = projW(pool1, n, max_o)
        #Wtx2_0 = onlyOmega0(Wtx2, n)
        #Wtx2_0 = useNorm(Wtx2, n, max_o)
        Wtx2_0 = useRinv(Wtx2, n, max_o)
        #Wtx2_0 = Wtx2
        
        ip_ch = Wtx2_0.get_shape()[-1]    
        
        kernel2 = tf.get_variable('kernel', [1, 1, ip_ch, op_ch], \
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases2 = tf.get_variable('biases', [op_ch], \
                            initializer=tf.constant_initializer(value=0.1))
        conv = tf.nn.conv2d(Wtx2_0, kernel2, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(conv + biases2, name=scope.name)
    
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
    
    fc_ip = pool2
    
    with tf.variable_scope('fc1') as scope:
        input_features = np.prod(fc_ip.get_shape().as_list()[1:])
        w = tf.get_variable('weights', [input_features, 5000], \
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        #b = tf.get_variable('biases', [1024], \
        #                    initializer=tf.constant_initializer(value=0.1))
        #b = tf.get_variable('biases', [1], \
        #                    initializer=tf.constant_initializer(value=0.1))
    
        # reshape pool2 to 2 dimensional
        pooln = tf.reshape(fc_ip, [-1, input_features])
        #fc1 = tf.nn.relu(tf.matmul(pooln, w) + b, name='relu')
        fc1 = tf.nn.relu(tf.matmul(pooln, w), name='relu')
        fc1 = tf.nn.dropout(fc1, keep_prob, name='relu_dropout')
    
    fc_ip = fc1
    with tf.variable_scope('fc2') as scope:
        input_features = np.prod(fc_ip.get_shape().as_list()[1:])
        w = tf.get_variable('weights', [input_features, 500], \
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        #b = tf.get_variable('biases', [1024], \
        #                    initializer=tf.constant_initializer(value=0.1))
        #b = tf.get_variable('biases', [1], \
        #                    initializer=tf.constant_initializer(value=0.1))
    
        # reshape pool2 to 2 dimensional
        pooln = tf.reshape(fc_ip, [-1, input_features])
        #fc2 = tf.nn.relu(tf.matmul(pooln, w) + b, name='relu')
        fc2 = tf.nn.relu(tf.matmul(pooln, w), name='relu')
        fc2 = tf.nn.dropout(fc2, keep_prob, name='relu_dropout')
    

    fc = fc2
    
    with tf.variable_scope('softmax_linear') as scope:
        input_features = np.prod(fc.get_shape().as_list()[1:])
        w = tf.get_variable('weights', [input_features, n_classes], \
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        #b = tf.get_variable('biases', [n_classes], \
        #                    initializer=tf.constant_initializer(value=0.1))
        #b = tf.get_variable('biases', [1], \
        #                    initializer=tf.constant_initializer(value=0.1))
        #logits = tf.matmul(fc, w) + b
        #y_conv = tf.matmul(fc, w) + b
        y_conv = tf.matmul(fc, w)
    
#    with tf.variable_scope('fc') as scope:
#        input_features = np.prod(fc_ip.get_shape().as_list()[1:])
#        w = tf.get_variable('weights', [input_features, 1024], \
#                            initializer=tf.truncated_normal_initializer(stddev=0.1))
#        #b = tf.get_variable('biases', [1024], \
#        #                    initializer=tf.constant_initializer(value=0.1))
#        b = tf.get_variable('biases', [1], \
#                            initializer=tf.constant_initializer(value=0.1))
#    
#        # reshape pool2 to 2 dimensional
#        pooln = tf.reshape(fc_ip, [-1, input_features])
#        fc = tf.nn.relu(tf.matmul(pooln, w) + b, name='relu')
#        fc = tf.nn.dropout(fc, keep_prob, name='relu_dropout')
#    
#    
#    with tf.variable_scope('softmax_linear') as scope:
#        w = tf.get_variable('weights', [1024, n_classes], \
#                            initializer=tf.truncated_normal_initializer(stddev=0.1))
#        b = tf.get_variable('biases', [n_classes], \
#                            initializer=tf.constant_initializer(value=0.1))
#        #b = tf.get_variable('biases', [1], \
#        #                    initializer=tf.constant_initializer(value=0.1))
#        #logits = tf.matmul(fc, w) + b
#        y_conv = tf.matmul(fc, w) + b
    

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
    #train_step = tf.train.AdagradOptimizer(LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    X_valid, y_valid = X_test[-valid_size:], y_test[-valid_size:]
    X_test, y_test = X_test[:-valid_size], y_test[:-valid_size]

    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      for i in range(epochs):
        idx_all = np.random.permutation(X_train.shape[0])
        tot_batches = X_train.shape[0] / b_size
        t1 = time.time()
        for b in range(tot_batches):
          idx = idx_all[b*b_size:(b+1)*b_size]
          X_batch, y_batch = X_train[idx], y_train[idx]
          train_step.run(feed_dict={x: X_batch, y_: y_batch, keep_prob: DROPOUT})
        #fc_vals = sess.run([fc_ip], feed_dict={x: X_valid[:1], y_: y_valid[:1], keep_prob: 1.})
        t2 = time.time()
        tr_acc = get_accuracy(X_train, y_train, b_size, correct_prediction)
        va_acc = get_accuracy(X_valid, y_valid, b_size, correct_prediction)
        t3 = time.time()
        print 'Epoch: {:4d}    Accuracy: Train {:0.4f}, ' \
              'Valid {:0.4f}    Time: Train {:0.2f} Validate {:0.2f}'.format(\
              i+1, tr_acc, va_acc, t2-t1, t3-t2)
        #print fc_vals[0].shape
        #print fc_vals[0][0, 0, 0, :5]
        #tr_acc_summary = tf.summary.scalar('train_acc', tf.Constant(tr_acc))
        #print tr_acc_summary
        #va_acc_summary = tf.summary.scalar('valid_acc', va_acc)
        #file_writer.add_summary(tr_acc_summary, i+1)
        #file_writer.add_summary(va_acc_summary, i+1)
        save_path = saver.save(sess, 'ckpt/my_model', i+1)
    
      save_path = saver.save(sess, 'ckpt/my_model_final.ckpt')
      acc = get_accuracy(X_test, y_test, b_size, correct_prediction)
      print 'test accuracy {0}'.format(acc)
