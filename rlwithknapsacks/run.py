import matplotlib
matplotlib.use('Agg')

import pylab as pl
import numpy as np
import copy
import argparse
from itertools import count
import pandas as pd
import random
from datetime import datetime
import torch

import matplotlib.pyplot as plt
from rlwithknapsacks.src.mdp import FiniteHorizonCMDP, DiscountedMDP
from rlwithknapsacks.src.envs.gridworld import GridWorld
from rlwithknapsacks.src.envs.box_gridworld import BoxGridWorld
from rlwithknapsacks.src.envs.whisky_gridworld import WhiskyGridWorld
from rlwithknapsacks.src.rl_solver import ValueIteration, UCBVI, PolicyGradient, A2C #, A2C_Planner
from rlwithknapsacks.src.alg import Appropo, Optimistic, Baseline
from rlwithknapsacks.src.alg.noregret_learner import ProjOrthant, NoRegretLearner
from rlwithknapsacks.src.args import get_args, set_gridworld_defaults, set_boxgridworld_defaults
from rlwithknapsacks.src.alg.cheungs.train_Toc_UCRL import train_Toc_UCRL,plot_tuning,plot_Toc_UCRL

parser = argparse.ArgumentParser()
get_args(parser)
args = parser.parse_args()

if args.env == 'gridworld':
    set_gridworld_defaults(args)
elif args.env == 'box_gridworld':
    set_boxgridworld_defaults(args)

RANDOM_SEED = None
if args.seed.isdigit():
    RANDOM_SEED = int(args.seed)
elif args.seed.lower() == 'random':
    RANDOM_SEED = random.randint(0,1e+5)
else:
    raise Exception("Error: Unknown seed type specified")
args.seed = RANDOM_SEED

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


default_maps = {'box_gridworld': '6x6box.txt' ,\
                'gridworld': '8x8rcpo.txt',\
                'whisky_gridworld': '6x8whisky.txt'}


if args.map == 'default':
    args.map = default_maps[args.env]

def train(solver, alg, env, args):
    results = []
    for round in range(args.rounds):
        [metrics, status_str] = alg(solver)
        metrics['round'] = round
        results.append(metrics)
        print(f'Round: {round}/{args.rounds}')
        print(f'{status_str}')
        print(f'----------------------------')

        df = pd.DataFrame(results,\
              columns=np.hstack(['round', 'mixture_reward', 'mixture_constraint',\
                                 'current_reward', 'current_constraint', 'alg', 'num_trajs',\
                                 'expected_consumption', 'training_consumpution']))
        now = datetime.now()
        date = now.strftime("%Y%m%d%H%M%S")
        output_filename =  f'{args.output_dir}/{args.env}_{args.alg}_{args.map}_{args.solver}_'
        output_filename += f'{args.num_episodes}_{args.proj_lr}_{args.seed}_{args.baseline_lambda_lr}_'
        output_filename += f'{args.value_coef}_{args.entropy_coef}_{args.optimistic_lambda_lr}_'
        output_filename += f'{args.conplanner_iter}_{args.num_fic_episodes}_{args.planner}_'
        output_filename += f'{args.optomistic_reset}.csv'
        # df.to_csv(output_filename)

def main():
    if 'gridworld' in args.env:
        if args.env == 'gridworld':
            G = GridWorld(args=args)
            budget = [0.3]
        elif args.env == 'box_gridworld':
            G = BoxGridWorld(args=args)
            budget = [0.1]
        elif args.env == 'whisky_gridworld':
            G = WhiskyGridWorld(args=args)
        [mdp_values, Si, Ai] = G.encode() # [MDP, State-lookups, Action-lookups]
        args.num_states = G.num_states
        args.rows = G.rows
        args.cols = G.cols
        args.initial_state = G.initial_state

        #budget = args.budget
        d = len(budget)

        env = FiniteHorizonCMDP(*mdp_values, d, budget, G.H, Si, G.terminals)
        #D = M.S * M.A + M.S + M.A

    if args.solver =='value_iteration':
       solver = ValueIteration(M=env, args=args)
    elif args.solver == "ucbvi":
        solver = UCBVI(M=env,args=args)
    elif args.solver == 'policy_gradient' or args.solver == 'reinforce' :
        env = DiscountedMDP(*mdp_values, d, budget, G.H, Si, G.terminals, gamma=0.99)
        solver = PolicyGradient(M=env, args=args)
    elif args.solver == 'a2c' :
        solver = A2C(M=env,args=args,G=G)
    #elif args.solver == 'a2c_planner':
    #    solver = A2C_Planner(M=env, args=args, G=G)

    # Basic example of Value iteration ----------------
    if args.alg == 'baseline':
        alg = Baseline(M=env, args=args)
    # Optimistic Reinforcement Learning --------------
    elif args.alg == 'optimistic':
        if args.planner == "value_iteration":
            planner = ValueIteration(M=env, args=args)
        elif args.planner == "a2c":
            planner = A2C(M=env , args=args , G=G)
        else:
            raise Exception("Error Unknown Planner")
        alg = Optimistic(G=G, M=env, args=args, planner=planner)
    # Appropoj ----------------------------------------
    elif args.alg == 'appropo':
        dim = len(env.budget)+1
        if args.env == 'gridworld':
            reward_threshold = 1.4
        elif args.env == 'box_gridworld':
            reward_threshold =1.58

        proj_oracle = NoRegretLearner(dim=dim, proj=ProjOrthant(env.budget,reward_threshold), args=args)
        alg = Appropo(proj_oracle=proj_oracle, M=env, args=args)

    if args.alg != "Toc_UCRL":
        train(solver, alg, env, args)
    else:
        args.budget = budget
        train_Toc_UCRL(env, args, tuning=True)
        #plot_tuning(args)
        #plot_Toc_UCRL(args)

if __name__ == '__main__':
    main()
