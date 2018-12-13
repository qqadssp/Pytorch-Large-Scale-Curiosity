import numpy as np
import torch

from auxiliary_tasks import JustPixels
from utils import small_convnet, flatten_dims, unflatten_first_dim, unet


class Dynamics(object):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='dynamics'):
        self.scope = scope
        self.auxiliary_task = auxiliary_task
        self.hidsize = self.auxiliary_task.hidsize
        self.feat_dim = feat_dim
        self.ac_space = self.auxiliary_task.ac_space
        self.ob_mean = self.auxiliary_task.ob_mean
        self.ob_std = self.auxiliary_task.ob_std
        self.predict_from_pixels = predict_from_pixels
        self.param_list = []
        if predict_from_pixels:
            self.features_model = small_convnet(self.ob_space, nl=torch.nn.LeakyReLU, feat_dim=self.feat_dim, last_nl=torch.nn.LeakyReLU, layernormalize=False)
            self.param_list = self.param_list + [dict(params = self.features_model.parameters())]
        else:
            self.features_model = None

        # not understand why we need a net in the origin implementation
        self.loss_net = loss_net(nblocks=4, feat_dim=self.feat_dim, ac_dim=self.ac_space.n, out_feat_dim=self.feat_dim, hidsize=self.hidsize)
        self.param_list = self.param_list + [dict(params=self.loss_net.parameters())]

        self.features = None
        self.next_features = None
        self.ac = None
        self.ob = None

    def update_features(self, obs, last_obs):
        if not self.predict_from_pixels:
            self.features = self.auxiliary_task.features.detach() # I'm not sure if there need a .detach for better performence, just keep it.
            self.next_features = self.auxiliary_task.next_features.detach()
        else:
            self.features = self.get_features(obs)
            last_features = self.get_features(last_ob)
            self.next_features = torch.cat([self.features[:, 1:], last_features], 1)
        self.ac = self.auxiliary_task.ac
        self.ob = self.auxiliary_task.ob

    def get_features(self, x):
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = x.shape
            x = flatten_dims(x, self.ob_space.n)
        x = np.transpose(x, [i for i in range(len(x.shape)-3)] + [-1, -3, -2])
        x = (x - self.ob_mean) / self.ob_std
        x = self.features_model(x)
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self):
        ac = self.ac
        sh = ac.shape
        ac = flatten_dims(ac, len(self.ac_space.shape))
        ac = torch.zeros(ac.shape + (self.ac_space.n,)).scatter_(1, torch.tensor(ac).unsqueeze(1), 1) # one_hot(self.ac, self.ac_space.n, axis=2)
        ac = unflatten_first_dim(ac, sh)

        features = self.features
        next_features = self.next_features
        assert features.shape[:-1] == ac.shape[:-1]
        sh = features.shape
        x = flatten_dims(features, 1)
        ac = flatten_dims(ac, 1)
        x = self.loss_net(x, ac)
        x = unflatten_first_dim(x, sh)
        return torch.mean((x - next_features) ** 2, -1)

    def calculate_loss(self, obs, last_obs, acs):
        n_chunks = 4
        n = obs.shape[0]
        chunk_size = n // n_chunks
        assert n % n_chunks == 0
        sli = lambda i: slice(i * chunk_size, (i + 1) * chunk_size)
        losses = None
        for i in range(n_chunks):
            ob = obs[sli(i)]
            last_ob = last_obs[sli(i)]
            ac = acs[sli(i)]
            self.auxiliary_task.policy.update_features(ob, ac)
            self.auxiliary_task.update_features(ob, last_ob)
            self.update_features(ob, last_ob)
            loss = self.get_loss()
            if losses is None:
                losses = loss
            else:
                losses = torch.cat((losses, loss), 0)
        return losses.data.numpy()

class loss_net(torch.nn.Module):
    def __init__(self, nblocks, feat_dim, ac_dim, out_feat_dim, hidsize, activation=torch.nn.LeakyReLU):
        super(loss_net, self).__init__()
        self.nblocks = nblocks
        self.feat_dim = feat_dim
        self.ac_dim = ac_dim
        self.out_feat_dim = out_feat_dim
        self.activation = activation
        model_list = [torch.nn.Linear(feat_dim+ac_dim, hidsize), activation()]
        for _ in range(self.nblocks):
            model_list.append(torch.nn.Linear(hidsize+ac_dim, hidsize))
            model_list.append(activation())
            model_list.append(torch.nn.Linear(hidsize+ac_dim, hidsize))
        model_list.append(torch.nn.Linear(hidsize+ac_dim, out_feat_dim))
        self.model_list = model_list
        self.init_weight()

    def init_weight(self):
        for m in self.model_list:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def forward(self, features, ac):
        idx = 0
        x = torch.cat((features, ac), dim=-1)
        for _ in range(2):
            x = self.model_list[idx](x)
            idx += 1
        for _ in range(self.nblocks):
            x0 = x
            for _ in range(3):
                if isinstance(self.model_list[idx], torch.nn.Linear): x = torch.cat((x, ac), dim=-1)
                x = self.model_list[idx](x)
                idx += 1
            x = x + x0
        x = torch.cat((x, ac), dim=-1)
        x = self.model_list[idx](x)
        assert idx == len(self.model_list) - 1
        assert x.shape[-1] == self.out_feat_dim
        return x

class UNet(Dynamics):
    def __init__(self, auxiliary_task, predict_from_pixels, feat_dim=None, scope='pixel_dynamics'):
        assert isinstance(auxiliary_task, JustPixels)
        assert not predict_from_pixels, "predict from pixels must be False, it's set up to predict from features that are normalized pixels."
        super(UNet, self).__init__(auxiliary_task=auxiliary_task,
                                   predict_from_pixels=predict_from_pixels,
                                   feat_dim=feat_dim,
                                   scope=scope)

    def get_features(self, x, reuse):
        raise NotImplementedError

    def get_loss(self):
        nl = tf.nn.leaky_relu
        ac = tf.one_hot(self.ac, self.ac_space.n, axis=2)
        sh = tf.shape(ac)
        ac = flatten_two_dims(ac)
        ac_four_dim = tf.expand_dims(tf.expand_dims(ac, 1), 1)

        def add_ac(x):
            if x.get_shape().ndims == 2:
                return tf.concat([x, ac], axis=-1)
            elif x.get_shape().ndims == 4:
                sh = tf.shape(x)
                return tf.concat(
                    [x, ac_four_dim + tf.zeros([sh[0], sh[1], sh[2], ac_four_dim.get_shape()[3].value], tf.float32)],
                    axis=-1)

        with tf.variable_scope(self.scope):
            x = flatten_two_dims(self.features)
            x = unet(x, nl=nl, feat_dim=self.feat_dim, cond=add_ac)
            x = unflatten_first_dim(x, sh)
        self.prediction_pixels = x * self.ob_std + self.ob_mean
        return tf.reduce_mean((x - tf.stop_gradient(self.out_features)) ** 2, [2, 3, 4])
