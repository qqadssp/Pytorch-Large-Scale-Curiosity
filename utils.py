import os
import platform
from functools import partial

import torch
import numpy as np

'''
def bcast_tf_vars_from_root(sess, vars):
    """
    Send the root node's parameters to every worker.

    Arguments:
      sess: the TensorFlow session.
      vars: all parameter variables including optimizer's
    """
    rank = MPI.COMM_WORLD.Get_rank()
    for var in vars:
        if rank == 0:
            MPI.COMM_WORLD.bcast(sess.run(var))
        else:
            sess.run(tf.assign(var, MPI.COMM_WORLD.bcast(None)))
'''

def get_mean_and_std(array):
    mean = np.array(np.mean(array)) # local_mean

    n_array = array - mean
    sqs = n_array ** 2
    var = np.array(np.mean(sqs)) # local_mean
    std = var ** 0.5
    return mean, std

'''
def guess_available_gpus(n_gpus=None):
    if n_gpus is not None:
        return list(range(n_gpus))
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        cuda_visible_divices = os.environ['CUDA_VISIBLE_DEVICES']
        cuda_visible_divices = cuda_visible_divices.split(',')
        return [int(n) for n in cuda_visible_divices]
    nvidia_dir = '/proc/driver/nvidia/gpus/'
    if os.path.exists(nvidia_dir):
        n_gpus = len(os.listdir(nvidia_dir))
        return list(range(n_gpus))
    raise Exception("Couldn't guess the available gpus on this machine")


def setup_mpi_gpus():
    """
    Set CUDA_VISIBLE_DEVICES using MPI.
    """
    available_gpus = guess_available_gpus()

    node_id = platform.node()
    nodes_ordered_by_rank = MPI.COMM_WORLD.allgather(node_id)
    processes_outranked_on_this_node = [n for n in nodes_ordered_by_rank[:MPI.COMM_WORLD.Get_rank()] if n == node_id]
    local_rank = len(processes_outranked_on_this_node)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(available_gpus[local_rank])


def guess_available_cpus():
    return int(multiprocessing.cpu_count())


def setup_tensorflow_session():
#    num_cpu = guess_available_cpus()

    tf_config = tf.ConfigProto(allow_soft_placement=True
#        inter_op_parallelism_threads=num_cpu,
#        intra_op_parallelism_threads=num_cpu
    )
    return tf.Session(config=tf_config)
'''

def random_agent_ob_mean_std(env, nsteps=1000):
    ob = np.asarray(env.reset())
    obs = [ob]
    for _ in range(nsteps):
        ac = env.action_space.sample()
        ob, _, done, _ = env.step(ac)
        if done:
            ob = env.reset()
        obs.append(np.asarray(ob))
    mean = np.mean(obs, 0).astype(np.float32)
    std = np.std(obs, 0).mean().astype(np.float32)
    return mean, std

def flatten_dims(x, dim):
    if dim == 0:
        return x.reshape((-1,))
    else:
        return x.reshape((-1,) + x.shape[-dim:])


def unflatten_first_dim(x, sh):
    assert x.shape[0] // sh[0] * sh[0] == x.shape[0] # whether x.shape[0] is N_integer times of sh[0]
    return x.view((sh[0],) + (x.shape[0] // sh[0],) + x.shape[1:])

'''
def add_pos_bias(x):
    with tf.variable_scope(name_or_scope=None, default_name="pos_bias"):
        b = tf.get_variable(name="pos_bias", shape=[1] + x.get_shape().as_list()[1:], dtype=tf.float32,
                            initializer=tf.zeros_initializer())
        return x + b
'''

class small_convnet(torch.nn.Module):
    def __init__(self, ob_space, nl, feat_dim, last_nl, layernormalize, batchnorm=False):
        super(small_convnet, self).__init__()
        self.H = ob_space.shape[0]
        self.W = ob_space.shape[1]
        self.C = ob_space.shape[2]

        feat_list = [[self.C, 32, 8, (4, 4)], [32, 64, 8, (2, 2)], [64, 64, 3, (1, 1)]]
        self.conv = torch.nn.Sequential()
        oH = self.H
        oW = self.W
        for idx, f in enumerate(feat_list):
            self.conv.add_module('conv_%i' % idx, torch.nn.Conv2d(f[0], f[1], kernel_size=f[2], stride=f[3]))
            self.conv.add_module('nl_%i' % idx, nl())
            if batchnorm: 
                self.conv.add_module('bn_%i' % idx, torch.nn.BatchNorm2d(f[1]))
            oH = (oH - f[2])/f[3][0] + 1
            oW = (oW - f[2])/f[3][1] + 1

        assert oH == int(oH) # whether oH is a .0 float ?
        assert oW == int(oW)
        self.flatten_dim = int(oH * oW * feat_list[-1][1])
        self.fc = torch.nn.Linear(self.flatten_dim, feat_dim)

        self.last_nl = last_nl
        self.layernormalize = layernormalize
        self.init_weight()

    def init_weight(self):
        for m in self.conv:
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)
        torch.nn.init.constant_(self.fc.bias.data, 0.0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, self.flatten_dim) # dims is calculated manually, 84*84 -> 20*20 -> 9*9 ->7*7
        x = self.fc(x)
        if self.last_nl is not None:
            x = self.last_nl(x)
        if self.layernormalize:
            x = self.layernorm(x)
        return x

    def layernorm(self, x):
        m = torch.mean(x, -1, keepdim=True)
        v = torch.std(x, -1, keepdim=True)
        return (x - m) / (v + 1e-8)

'''
def small_convnet(x, nl, feat_dim, last_nl, layernormalize, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    x = bn(tf.layers.conv2d(x, filters=32, kernel_size=8, strides=(4, 4), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=4, strides=(2, 2), activation=nl))
    x = bn(tf.layers.conv2d(x, filters=64, kernel_size=3, strides=(1, 1), activation=nl))
    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = bn(fc(x, units=feat_dim, activation=None))
    if last_nl is not None:
        x = last_nl(x)
    if layernormalize:
        x = layernorm(x)
    return x
'''

class small_deconvnet(torch.nn.Module):
    def __init__(self, ob_space, feat_dim, nl, ch, positional_bias):
        super(small_deconvnet, self).__init__()
        self.H = ob_space.shape[0]
        self.W = ob_space.shape[1]
        self.C = ob_space.shape[2]

        self.feat_dim = feat_dim
        self.nl = nl
        self.ch = ch
        self.positional_bias = positional_bias

        self.sh = (64, 8, 8)
        self.fc = torch.nn.Sequential(torch.nn.Linear(feat_dim, np.prod(self.sh)), nl())

        # the last kernel_size is 7 not 8 compare to the origin implementation, to make the output shape be [96, 96]
        feat_list = [[self.sh[0], 128, 4, (2, 2), (1, 1)], [128, 64, 8, (2, 2), (3, 3)], [64, ch, 7, (3, 3), (2, 2)]] 
        self.deconv = torch.nn.Sequential()
        for i, f in enumerate(feat_list):
            self.deconv.add_module('deconv_%i' % i, torch.nn.ConvTranspose2d(f[0], f[1], kernel_size=f[2], stride=f[3], padding=f[4]))
            if i != len(feat_list) - 1:
                self.deconv.add_module('nl_%i' % i, nl())

        self.bias = torch.nn.Parameter(torch.zeros(self.ch, self.H, self.W), requires_grad=True) if self.positional_bias else None

        self.init_weight()

    def init_weight(self):
        for m in self.fc:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
        for m in self.deconv:
            if isinstance(m, torch.nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, features):
        x = self.fc(features)
        x = x.view((-1,) + self.sh)
        x = self.deconv(x)
        x = x[:, :, 6:-6, 6:-6]
        assert x.shape[-2:] == (84, 84)
        if self.positional_bias:
            x = x + self.bias
        return x

'''
def small_deconvnet(z, nl, ch, positional_bias):
    sh = (8, 8, 64)
    z = fc(z, np.prod(sh), activation=nl)
    z = tf.reshape(z, (-1, *sh))
    z = tf.layers.conv2d_transpose(z, 128, kernel_size=4, strides=(2, 2), activation=nl, padding='same')
    assert z.get_shape().as_list()[1:3] == [16, 16]
    z = tf.layers.conv2d_transpose(z, 64, kernel_size=8, strides=(2, 2), activation=nl, padding='same')
    assert z.get_shape().as_list()[1:3] == [32, 32]
    z = tf.layers.conv2d_transpose(z, ch, kernel_size=8, strides=(3, 3), activation=None, padding='same')
    assert z.get_shape().as_list()[1:3] == [96, 96]
    z = z[:, 6:-6, 6:-6]
    assert z.get_shape().as_list()[1:3] == [84, 84]
    if positional_bias:
        z = add_pos_bias(z)
    return z
'''

def unet(x, nl, feat_dim, cond, batchnorm=False):
    bn = tf.layers.batch_normalization if batchnorm else lambda x: x
    layers = []
    x = tf.pad(x, [[0, 0], [6, 6], [6, 6], [0, 0]])
    x = bn(tf.layers.conv2d(cond(x), filters=32, kernel_size=8, strides=(3, 3), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [32, 32]
    layers.append(x)
    x = bn(tf.layers.conv2d(cond(x), filters=64, kernel_size=8, strides=(2, 2), activation=nl, padding='same'))
    layers.append(x)
    assert x.get_shape().as_list()[1:3] == [16, 16]
    x = bn(tf.layers.conv2d(cond(x), filters=64, kernel_size=4, strides=(2, 2), activation=nl, padding='same'))
    layers.append(x)
    assert x.get_shape().as_list()[1:3] == [8, 8]

    x = tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))
    x = fc(cond(x), units=feat_dim, activation=nl)

    def residual(x):
        res = bn(tf.layers.dense(cond(x), feat_dim, activation=tf.nn.leaky_relu))
        res = tf.layers.dense(cond(res), feat_dim, activation=None)
        return x + res

    for _ in range(4):
        x = residual(x)

    sh = (8, 8, 64)
    x = fc(cond(x), np.prod(sh), activation=nl)
    x = tf.reshape(x, (-1, *sh))
    x += layers.pop()
    x = bn(tf.layers.conv2d_transpose(cond(x), 64, kernel_size=4, strides=(2, 2), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [16, 16]
    x += layers.pop()
    x = bn(tf.layers.conv2d_transpose(cond(x), 32, kernel_size=8, strides=(2, 2), activation=nl, padding='same'))
    assert x.get_shape().as_list()[1:3] == [32, 32]
    x += layers.pop()
    x = tf.layers.conv2d_transpose(cond(x), 4, kernel_size=8, strides=(3, 3), activation=None, padding='same')
    assert x.get_shape().as_list()[1:3] == [96, 96]
    x = x[:, 6:-6, 6:-6]
    assert x.get_shape().as_list()[1:3] == [84, 84]
    assert layers == []
    return x


def tile_images(array, n_cols=None, max_images=None, div=1):
    if max_images is not None:
        array = array[:max_images]
    if len(array.shape) == 4 and array.shape[3] == 1:
        array = array[:, :, :, 0]
    assert len(array.shape) in [3, 4], "wrong number of dimensions - shape {}".format(array.shape)
    if len(array.shape) == 4:
        assert array.shape[3] == 3, "wrong number of channels- shape {}".format(array.shape)
    if n_cols is None:
        n_cols = max(int(np.sqrt(array.shape[0])) // div * div, div)
    n_rows = int(np.ceil(float(array.shape[0]) / n_cols))

    def cell(i, j):
        ind = i * n_cols + j
        return array[ind] if ind < array.shape[0] else np.zeros(array[0].shape)

    def row(i):
        return np.concatenate([cell(i, j) for j in range(n_cols)], axis=1)

    return np.concatenate([row(i) for i in range(n_rows)], axis=0)

