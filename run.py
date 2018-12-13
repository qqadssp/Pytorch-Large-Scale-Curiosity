#!/usr/bin/env python
import functools
import os.path as osp
from functools import partial

import gym
import torch
import logger
from monitor import Monitor
from atari_wrappers import NoopResetEnv, FrameStack

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit


def start_experiment(**args):
    make_env = partial(make_env_all_params, add_monitor=True, args=args)

    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'])
    log = get_experiment_environment(**args)
    with log:
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        trainer.train()


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()

        self.policy = CnnPolicy(
            scope='pol',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            hidsize=512,
            feat_dim=512,
            ob_mean=self.ob_mean,
            ob_std=self.ob_std,
            layernormalize=False,
            nl=torch.nn.LeakyReLU)

        self.feature_extractor = {"none": FeatureExtractor,
                                  "idf": InverseDynamics,
                                  "vaesph": partial(VAE, spherical_obs=True),
                                  "vaenonsph": partial(VAE, spherical_obs=False),
                                  "pix2pix": JustPixels}[hps['feat_learning']]
        self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                        # if we use VAE, 'features_shared_with_policy' should be set to False,
                                                        # because the shape of output_features of VAE.get_features is feat_dims * 2, including means and stds,
                                                        # but the shape of out_features of policy.get_features is feat_dims,
                                                        # only means is used as features exposed to dynamics 
                                                        features_shared_with_policy=False, 
                                                        feat_dim=512,
                                                        layernormalize=hps['layernorm'])

        self.dynamics = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet
        self.dynamics = self.dynamics(auxiliary_task=self.feature_extractor,
                                      predict_from_pixels=hps['dyn_from_pixels'],
                                      feat_dim=512)

        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics= self.dynamics
        )

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def train(self):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.dynamics)
        while True:
            info = self.agent.step()
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break

        self.agent.stop_interaction()


def make_env_all_params(rank, add_monitor, args):
    if args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        if 'Montezuma' in args['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    from misc_util import set_global_seeds
    process_seed = args["seed"]
    set_global_seeds(process_seed)

    logger_context = logger.scoped_configure(dir='tmp', format_strs=['stdout', 'log', 'csv'])
    return logger_context


def add_environments_params(parser):
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4', type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=4500, type=int)
    parser.add_argument('--env_kind', type=str, default="atari")
    parser.add_argument('--noop_max', type=int, default=30)


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=4)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=int(1e8))


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=16)
    # nsegs_per_env should always be 1, because last_feature is directly concatenated to features to form next_features, no special process for nsegs,
    # see FeatureExtractor.next_features in auxiliary_task.py in origin implementation, here just keep it.
    parser.add_argument('--nsegs_per_env', type=int, default=1)  
    parser.add_argument('--envs_per_process', type=int, default=8)
    parser.add_argument('--nlumps', type=int, default=2)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
    add_optimization_params(parser)
    add_rollout_params(parser)

    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    # why dyn_from_pixels? not very clear about the difference between 'dyn_from_pixels' and 'features_shared_with_policy'
    # I doubt that 'dyn_from_pixels' should always be zero, because out_features = auxiliary_task.next_features in Dynamics in dynamics.py, just keep it 
    parser.add_argument('--dyn_from_pixels', type=int, default=0) 
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default="vaenonsph",
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])

    args = parser.parse_args()

    start_experiment(**args.__dict__)
