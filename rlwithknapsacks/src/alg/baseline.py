import matplotlib
matplotlib.use('Agg')

import pylab as pl
import numpy as np
import copy
import argparse
from itertools import count
from collections import defaultdict

import matplotlib.pyplot as plt
from rlwithknapsacks.src.policy import *
from rlwithknapsacks.src.policy import SoftmaxPolicy
from collections import deque

class RCPOPolicy:
    def __init__(self, shadow=False):
        self.shadow = shadow
        self.queue = deque(maxlen=10)

    def add_response(self, best_exp_rtn=None):
        self.queue.append(best_exp_rtn)
        rtn = np.average(self.queue, axis=0)

        self.reward = rtn[0]
        self.constraint = rtn[1]

class Baseline:
    def __init__(self, M=None, args=None):
        self.M = M
        self.theta = [1, 0]
        self.budget = M.budget
        self._lambda_lr = args.baseline_lambda_lr
        #self.policy = MixturePolicy()
        self.policy = RCPOPolicy()

    def __call__(self, rl_solver):
        pseudo_reward = self.theta[0]*self.M.R+self.theta[1]*self.M.C
        result = rl_solver(R=pseudo_reward, theta=self.theta)

        values = rl_solver.value_evaluation(result['pi_list'])
        self.policy.add_response([values['reward'], values['constraint']])

        self.theta[1] = min(0, self.theta[1] - self._lambda_lr * (values['constraint'] - self.budget))

        # Message printed to screen
        status = f'  _lambda: {self.theta[1]}\n'
        status += f' current_reward: {values["reward"]}\n'
        status += f' current_constraint: {values["constraint"]}\n'
        status += f' mixture_reward: {self.policy.reward}\n'
        status += f' mixture_constraint: {self.policy.constraint}\n'
        status += f'  last state: {result["last_state"]}'

        # Metrics to save
        metrics = {'mixture_reward': self.policy.reward,
                   'mixture_constraint': self.policy.constraint,
                   'current_reward': values['reward'],
                   'current_constraint': values['constraint'],
                   'alg': 'baseline',
                   'num_trajs': rl_solver.stats['num_trajs'],
                   'expected_consumption': rl_solver.stats['expected_consumption'],
                   'training_consumpution': rl_solver.stats['training_consumpution']}
        return (metrics, status)
