import numpy as np
import torch
from utils import small_convnet, flatten_dims, unflatten_first_dim, small_deconvnet


class FeatureExtractor(object):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None,
                 scope='feature_extractor'):
        self.scope = scope
        self.features_shared_with_policy = features_shared_with_policy
        self.feat_dim = feat_dim
        self.layernormalize = layernormalize
        self.policy = policy
        self.hidsize = policy.hidsize
        self.ob_space = policy.ob_space
        self.ac_space = policy.ac_space
        self.ob_mean = self.policy.ob_mean
        self.ob_std = self.policy.ob_std

        self.features_shared_with_policy = features_shared_with_policy
        self.param_list = []
        if features_shared_with_policy:
            self.features_model = None
        else:
            self.features_model = small_convnet(self.ob_space, nl=torch.nn.LeakyReLU, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)
            self.param_list = self.param_list + [dict(params=self.features_model.parameters())]

        self.scope = scope

        self.features = None
        self.next_features = None
        self.ac = None
        self.ob = None

    def update_features(self, obs, last_obs):
        if self.features_shared_with_policy:
            self.features = self.policy.flat_features
            last_features = self.policy.get_features(last_obs)
        else:
            self.features = self.get_features(obs)
            last_features = self.get_features(last_obs)
        self.next_features = torch.cat([self.features[:, 1:], last_features], 1)
        self.ac = self.policy.ac
        self.ob = self.policy.ob

    def get_features(self, x):
        x_has_timesteps = (len(x.shape) == 5)
        if x_has_timesteps:
            sh = x.shape
            x = flatten_dims(x, len(self.ob_space.shape))
        x = (x - self.ob_mean) / self.ob_std
        x = np.transpose(x, [i for i in range(len(x.shape)-3)] + [-1, -3, -2]) # transpose channel axis
        x = self.features_model(torch.tensor(x))
        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_loss(self, *arg, **kwarg):
        return torch.tensor(0.0)


class InverseDynamics(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None):
        super(InverseDynamics, self).__init__(scope="inverse_dynamics", policy=policy,
                                              features_shared_with_policy=features_shared_with_policy,
                                              feat_dim=feat_dim, layernormalize=layernormalize)

        self.fc = torch.nn.Sequential(torch.nn.Linear(self.feat_dim * 2, self.policy.hidsize),
                                   torch.nn.ReLU(),
                                   torch.nn.Linear(self.policy.hidsize, self.ac_space.n)
                                  )
        self.param_list = self.param_list + [dict(params=self.fc.parameters())]
        self.init_weight()

    def init_weight(self):
        for m in self.fc:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def get_loss(self):
        x = torch.cat([self.features, self.next_features], 2)
        sh = x.shape
        x = flatten_dims(x, 1)
        param = self.fc(x)
        idfpd = self.policy.ac_pdtype.pdfromflat(param)
        ac = flatten_dims(self.ac, len(self.ac_space.shape))
        return idfpd.neglogp(torch.tensor(ac))


class VAE(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=False, spherical_obs=False):
        assert not layernormalize, "VAE features should already have reasonable size, no need to layer normalize them"
        super(VAE, self).__init__(scope="vae", policy=policy,
                                  features_shared_with_policy=features_shared_with_policy,
                                  feat_dim=feat_dim, layernormalize=False)

        self.features_model = small_convnet(self.ob_space, nl=torch.nn.LeakyReLU, feat_dim=2 * self.feat_dim, last_nl=None, layernormalize=False)
        self.decoder_model = small_deconvnet(self.ob_space, feat_dim=self.feat_dim, nl=torch.nn.LeakyReLU, ch = 4 if spherical_obs else 8, positional_bias=True)

        self.param_list = [dict(params=self.features_model.parameters()), dict(params=self.decoder_model.parameters())]

        self.features_std = None
        self.next_features_std = None

        self.spherical_obs = spherical_obs
        if self.spherical_obs:
            self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
            self.param_list = self.param_list + [dict(params=[self.scale])]

    def update_features(self, obs, last_obs):
        features = self.get_features(obs)
        last_features = self.get_features(last_obs)
        next_features = torch.cat([features[:, 1:], last_features], 1)

        self.features, self.features_std= torch.split(features, [self.feat_dim, self.feat_dim], -1) # use means only for features exposed to dynamics
        self.next_features, self.next_features_std = torch.split(next_features, [self.feat_dim, self.feat_dim], -1)
        self.ac = self.policy.ac
        self.ob = self.policy.ob

#    def get_features(self, x):
#        nl = tf.nn.leaky_relu
#        x_has_timesteps = (x.get_shape().ndims == 5)
#        if x_has_timesteps:
#            sh = tf.shape(x)
#            x = flatten_two_dims(x)
#        with tf.variable_scope(self.scope + "_features", reuse=reuse):
#            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
#            x = small_convnet(x, nl=nl, feat_dim=2 * self.feat_dim, last_nl=None, layernormalize=False)
#        if x_has_timesteps:
#            x = unflatten_first_dim(x, sh)
#        return x

    def get_loss(self):
        posterior_mean = self.features
        posterior_scale = torch.nn.functional.softplus(self.features_std) 
        posterior_distribution = torch.distributions.normal.Normal(posterior_mean, posterior_scale) 

        sh = posterior_mean.shape
        prior = torch.distributions.normal.Normal(torch.zeros(sh), torch.ones(sh)) 

        posterior_kl = torch.distributions.kl.kl_divergence(posterior_distribution, prior) 

        posterior_kl = torch.sum(posterior_kl, -1)
        assert len(posterior_kl.shape) == 2

        posterior_sample = posterior_distribution.rsample() # do we need to let the gradient pass through the sample process?
        reconstruction_distribution = self.decoder(posterior_sample)
        norm_obs = self.add_noise_and_normalize(self.ob)
        norm_obs = np.transpose(norm_obs, [i for i in range(len(norm_obs.shape) - 3)] + [-1, -3, -2]) # transpose channel axis
        reconstruction_likelihood = reconstruction_distribution.log_prob(torch.tensor(norm_obs).float())
        assert reconstruction_likelihood.shape[-3:] == (4, 84, 84)
        reconstruction_likelihood = torch.sum(reconstruction_likelihood, [-3, -2, -1])

        likelihood_lower_bound = reconstruction_likelihood - posterior_kl
        return - likelihood_lower_bound

    def add_noise_and_normalize(self, x):
        x = x + np.random.normal(0.0, 1.0, size=x.shape) # no bias
        x = (x - self.ob_mean) / self.ob_std
        return x

    def decoder(self, z):
        z_has_timesteps = (len(z.shape) == 3)
        if z_has_timesteps:
            sh = z.shape
            z = flatten_dims(z, 1)
        z = self.decoder_model(z)
        if z_has_timesteps:
            z = unflatten_first_dim(z, sh)
        if self.spherical_obs:
            scale = torch.max(self.scale, torch.tensor(-4.0))
            scale = torch.nn.functional.softplus(scale)
            scale = scale * torch.ones(z.shape)
        else:
            z, scale = torch.split(z, [4, 4], -3)
            scale = torch.nn.functional.softplus(scale)
        return torch.distributions.normal.Normal(z, scale)


class JustPixels(FeatureExtractor):
    def __init__(self, policy, features_shared_with_policy, feat_dim=None, layernormalize=None,
                 scope='just_pixels'):
        assert not layernormalize
        assert not features_shared_with_policy
        super(JustPixels, self).__init__(scope=scope, policy=policy,
                                         features_shared_with_policy=False,
                                         feat_dim=None, layernormalize=None)

    def get_features(self, x, reuse):
        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
        return x

    def get_loss(self):
        return tf.zeros((), dtype=tf.float32)
