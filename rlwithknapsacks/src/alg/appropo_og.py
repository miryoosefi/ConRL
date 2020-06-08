import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import gym
from gym import spaces
import time
import argparse
import operator
import pandas as pd
import scipy
from numpy import linalg as LA
from datetime import datetime
from scipy.stats import entropy
import uuid
from collections import defaultdict

#from ApproPO.util import init_cache, CacheItem
#from ApproPO.envs.gym_frozenmarsrover.envs.maps import MAPS
#from ApproPO.policy import MixturePolicy
#from ApproPO.util import calc_dist_uni
#from rlwithknapsacks.src.util import init_cache, CacheItem

class MixturePolicy:
    def __init__(self):
        self.loss_vec = []
        self.reward = None
        self.constraint = None

    def add_response(self, best_exp_rtn=None):
        self.loss_vec.append(best_exp_rtn)
        self.exp_rtn_of_avg_policy = np.average(np.stack(self.loss_vec, axis=0), axis=0)
        self.reward = self.exp_rtn_of_avg_policy[0]
        self.constraint = self.exp_rtn_of_avg_policy[1]

class Appropo:
    def __init__(self, proj_oracle=None, M=None, args=None, solver=None):
        #self.args = args
        self.mx_size = args.mx_size
        self.proj_oracle = proj_oracle

        self.policy = MixturePolicy()
        self.theta = proj_oracle.get_theta()

        #if solver == 'a2c':
        #    self.cache = init_cache(rl_oracle_generator=solver, args=args)
        self.cache = []

        self.M = M

        self.best_exp_rtn = None
        self.best_exp_params = None
        self.value = float('inf')

    def __call__(self, rl_solver=None):
        _lambda = self.theta[:-1]

        if np.linalg.norm(_lambda) != 0:
            _lambda /= np.linalg.norm(_lambda)
        pseudo_reward = _lambda[0]*self.M.R+_lambda[1]*self.M.C
        # we want solver to minimize lambda dot Z
        pseudo_reward *= -1.0

        #for rtn in self.cache:
        if self.value <= 0:
            for item in self.cache:
                (rtn, params) = item
                if np.dot(self.theta, np.append(self.best_exp_rtn, self.mx_size))>=np.dot(self.theta, np.append(rtn, self.mx_size)):
                    self.best_exp_rtn = rtn
                    self.best_exp_params = params
        else:
            self.best_exp_rtn = None
            self.best_exp_params = None

        if self.best_exp_rtn is not None:# or \
        #        np.dot(self.theta, np.append(self.best_exp_rtn, self.mx_size)) > 0:
            rl_solver.reset_oracle(self.best_exp_params)

            #result = rl_solver(R=pseudo_reward, theta=(-1)*_lambda)
        else:
            #rl_solver.reset_oracle(params)
            result = rl_solver(R=pseudo_reward, theta=(-1)*_lambda)

            π_list = result['pi_list']
            values = rl_solver.value_evaluation(π_list)
            self.best_exp_rtn = [values['reward'], values['constraint']]
            self.cache.append((self.best_exp_rtn, rl_solver.policy.state_dict()))

        self.value = np.dot(self.theta, np.append(self.best_exp_rtn, self.mx_size))


        if self.value <= 0:
            self.proj_oracle.update(self.best_exp_rtn.copy()) # Update OLO
            self.theta = self.proj_oracle.get_theta()
            self.policy.add_response(best_exp_rtn=self.best_exp_rtn)

            dist_to_target = np.linalg.norm(self.policy.exp_rtn_of_avg_policy\
                                      - self.proj_oracle.proj(self.policy.exp_rtn_of_avg_policy))

            status = f' New theta: {_lambda[:2]}\n'
            status += f'  value: {self.value}\n'
            status += f'  exp_rtn_of_avg_policy: {self.policy.exp_rtn_of_avg_policy[:2]}\n'
            status += f'  best_exp_rtn: {self.best_exp_rtn[:2]}\n'
            status += f'  dist-to-target: {dist_to_target}\n'
        else:
            status = f'  Old theta: {_lambda}\n'
            status += f'  best_exp_rtn: {self.best_exp_rtn[:2]}\n'

        #metrics = {'reward': self.policy.reward, 'constraint': self.policy.constraint}
        metrics = {'mixture_reward': self.policy.reward,
                   'mixture_constraint': self.policy.constraint,
                   'current_reward': self.best_exp_rtn[0],
                   'current_constraint': self.best_exp_rtn[1],
                   'alg': 'appropo',
                   'num_trajs': rl_solver.stats['num_trajs'],
                   'expected_consumption': rl_solver.stats['expected_consumption'],
                   'training_consumpution': rl_solver.stats['training_consumpution']}
        return (metrics, status)
