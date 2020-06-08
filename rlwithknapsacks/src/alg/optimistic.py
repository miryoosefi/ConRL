import matplotlib
matplotlib.use('Agg')

import pylab as pl
import numpy as np
import copy
import argparse
from itertools import count

import matplotlib.pyplot as plt
from rlwithknapsacks.src.policy import *
from rlwithknapsacks.src.policy import SoftmaxPolicy
from rlwithknapsacks.src.rl_solver import A2C
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

class Optimistic:
    def __init__(self, G=None, M=None, args=None, planner=None):
        self.num_states = len(G.states)
        self.num_actions = len(G.A)
        self.bonus_coef = args.bonus_coef
        self.H = G.H
        self.budget = M.budget
        self.conplanner_iter = args.conplanner_iter
        self.M = M
        self.policy = RCPOPolicy()
        self._lambda_lr = args.optimistic_lambda_lr
        self.planner = planner
        self.optomistic_reset = args.optomistic_reset

        self.π_list = np.zeros((self.H, self.num_states, self.num_actions))
        self.p_sum = np.zeros((self.num_states, self.num_actions, self.num_states))
        self.r_sum = np.zeros((self.num_states, self.num_actions))
        self.c_sum = np.zeros((self.num_states, self.num_actions))
        self.visitation_sum = np.zeros((self.num_states, self.num_actions))

        self.R = np.einsum('sap,sap->sa', self.M.P, self.M.R) if len(np.shape(self.M.R)) == 3 else self.M.R
        self.C = np.einsum('sap,sap->sa', self.M.P, self.M.C) if len(np.shape(self.M.C)) == 3 else self.M.C
        self._lambda = 0

    def conplanner(self, p_hat, r_hat, c_hat):
        #d = M.d
        #budget = M.budget

        #_lambda = 0
        _lambda_lr = self._lambda_lr
        _lambda_avg = 0

        for _ in range(self.conplanner_iter):
            pseudo_reward = r_hat+self._lambda*c_hat
            result = self.planner(P=p_hat, R=pseudo_reward, fic=True)
            π_list = result['pi_list']
            c = self.planner.value_evaluation(π_list, P=p_hat, R=pseudo_reward, C=c_hat)['constraint']
            self._lambda = min(0, self._lambda - _lambda_lr * (c - self.budget))
            _lambda_avg += self._lambda

        _lambda_avg /= self.conplanner_iter
        pseudo_reward = r_hat + _lambda_avg * c_hat
        result = self.planner(P=p_hat, R=pseudo_reward)


        if isinstance(self.planner, A2C):
            if self.optomistic_reset == 'warm-start':
                self.planner.reset_oracle(params=self.planner.policy.state_dict())
            elif self.optomistic_reset == 'scratch':
                self.planner.reset_oracle()
            elif self.optomistic_reset == 'continue':
                pass
            else:
                raise Exception("Unknown  optomistic_reset criteria")
        return result['pi_list'], result['V']

    def __call__(self, rl_solver):
        #rl_solver = self.planner
        #for round in count(start=1, step=1):
        p_hat = np.zeros((self.num_states, self.num_actions, self.num_states))
        r_hat = np.zeros((self.num_states, self.num_actions))
        c_hat = np.zeros((self.num_states, self.num_actions))
        bonus = np.zeros((self.num_states, self.num_actions))

        shape = (self.num_states,self.num_actions,self.num_states)
        i = np.identity(shape[0])
        tmp_p_hat = np.repeat(i, shape[1], axis=0).reshape(shape)

        tmp_r_hat = np.zeros((self.num_states, self.num_actions))
        tmp_c_hat = np.zeros((self.num_states, self.num_actions))
        tmp_bonus = np.zeros((self.num_states, self.num_actions))

        for s in range(self.num_states):
            for a in range(self.num_actions):
                visitation = self.p_sum[s,a,:].sum()
                if visitation == 0:
                    p_hat[s, a, s] = 1
                    bonus[s,a] = 1
                else:
                    p_hat[s, a, :] = self.p_sum[s,a,:]/visitation
                    r_hat[s, a] = self.r_sum[s,a]/visitation
                    c_hat[s,a] = self.c_sum[s,a]/visitation
                    bonus[s,a] = np.sqrt(1.0 / visitation) * self.bonus_coef


        π_list, v_π_list = self.conplanner(p_hat, r_hat + bonus, c_hat - bonus)

        (p, r, c, v, last_state) = rl_solver.run(π_list)
        # Update Counts
        self.p_sum += p
        self.r_sum += r
        self.c_sum += c
        self.visitation_sum += v




        values = rl_solver.value_evaluation(π_list)
        self.policy.add_response([values['reward'], values['constraint']])
        status = f' current_reward: {values["reward"]}\n'
        status += f' current_constraint: {values["constraint"]}\n'
        status += f' mixture_reward: {self.policy.reward}\n'
        status += f' mixture_constraint: {self.policy.constraint}\n'
        status += f' last_state: {last_state}'

        metrics = {'mixture_reward': self.policy.reward,
                   'mixture_constraint': self.policy.constraint,
                   'current_reward': values['reward'],
                   'current_constraint': values['constraint'],
                   'alg': 'optimistic',
                   'num_trajs': rl_solver.stats['num_trajs'],
                   'expected_consumption': rl_solver.stats['expected_consumption'],
                   'training_consumpution': rl_solver.stats['training_consumpution']}
        return (metrics, status)

