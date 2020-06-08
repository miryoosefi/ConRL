import pylab as pl
import numpy as np
import copy
import argparse
from itertools import count
import pandas as pd
import random
from datetime import datetime
import torch

from rlwithknapsacks.src.alg.cheungs.mdp_q.gen_mdp import *
from rlwithknapsacks.src.alg.cheungs.mdp_q.optimization import *

from rlwithknapsacks.src.mdp import FiniteHorizonCMDP, DiscountedMDP
from rlwithknapsacks.src.envs.gridworld import GridWorld
from rlwithknapsacks.src.args import get_args, set_gridworld_defaults, set_boxgridworld_defaults


def create_instant(env,budget):
    R = np.einsum('sap,sap->sa', env.P, env.R) if len(env.R.shape) == 3 else env.R
    C = np.einsum('sap,sap->sa', env.P, env.C) if len(env.C.shape) == 3 else env.C

    instant = dict()
    instant["S"] = list(range(env.S))
    instant["A"] = {i: np.arange(env.A) for i in range(env.S)}
    instant["p"] = dict()
    for s in range(env.S):
        for a in range(env.A):
            instant["p"][(s, a)] = np.array([env.P[s,a,s_prime] for s_prime in range(env.S)])
    instant["v_mean"]= dict()
    for s in range(env.S):
        for a in range(env.A):
            instant["v_mean"][(s, a)] = np.array([R[s,a],C[s,a]])
    instant["v_mat"]=np.array([[0,0] for s in range(env.S)])
    instant["reward_type"] = "KPI"
    instant["target"] = [1, budget/env.H]
    return instant

def make_loop(inst,env):
    for terminal in env.terminals:
        indx = env.Si[terminal]
        for a in inst["A"][indx]:
            inst["p"][(indx, a)] = env.s0



def train_Toc_UCRL(env, args, tuning=False):
    print(f"running with value_coef = {args.UCRL_coeff[0]}")
    print(f"running with seed = {args.seed}")
    my_instant = create_instant(env, budget=args.budget[0])
    make_loop(my_instant, env)
    res, metrics = Toc_UCRL2_q(my_instant, coeff=args.UCRL_coeff, args=args)
    print(res["bar_V"])

    df = pd.DataFrame(metrics, \
                      columns=np.hstack(['round', 'mixture_reward', 'mixture_constraint', \
                                         'current_reward', 'current_constraint', 'alg', 'num_trajs', \
                                         'expected_consumption', 'training_consumpution']))

    res_path = "./results/"
    res_path = res_path + args.env+"/"
    if tuning:
        res_path += "tuning/"
    res_path = res_path + f"UCRL_reward_coeff_{args.UCRL_coeff[0]}_random_seed_{args.seed}.csv"
    df.to_csv(res_path)



def plot_tuning(args):
    import os
    import matplotlib.pyplot as plt
    f = plt.figure(constrained_layout=True, figsize=(20, 10))
    widths = [1]
    heights = [1, 1]
    spec = f.add_gridspec(ncols=1, nrows=2, width_ratios=widths, height_ratios=heights)
    spec.update(wspace=0.025, hspace=0.05)
    res_path = "./results/"
    res_path = res_path+args.env+"/"+"tuning/"
    for row,measure in enumerate(["mixture_reward",'mixture_constraint']):
        ax = f.add_subplot(spec[row, 0])
        for filename in os.listdir(res_path):
            if ".csv" not in filename:
                continue
            df = pd.read_csv(res_path+filename)
            x = df['num_trajs'].values
            y = df[measure].values
            width = 2
            label = "Toc_"+filename.split("random")[0][:-1]
            ax.plot(x, y, linewidth=width, label=f'{label}', markersize=15)
            ax.set(xlim=(0, 33000))
            if row == 0:
                ax.legend()

            if "constraint" in measure:
                ax.set_ylabel('constraint satisfaction', fontsize=20)
                ax.set(ylim=(0, 2))
                ax.set_xlabel("number of trajectories")
                ax.hlines(args.budget[0], 0, 33000, linestyles='dashed', linewidth=width, label='constraint')
            elif "reward" in measure:
                ax.set_ylabel("reward", fontsize=20)
                ax.set(ylim=(0,2))
                if args.env == "gridworld":
                    ax.hlines(1.4, 0, 33000, linestyles='dashed', linewidth=width, label='constraint')
                elif args.env == "box_gridworld":
                    ax.hlines(1.58, 0, 33000, linestyles='dashed', linewidth=width, label='constraint')

    plt.tight_layout()
    plt.savefig(res_path+"UCRL_tuning.pdf")


def plot_Toc_UCRL(args):
    import os
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    x_union = set()
    dfs = []
    res_path = "./results/"
    res_path = res_path + args.env + "/"
    for filename in os.listdir(res_path):
        if ".csv" not in filename:
            continue
        df = pd.read_csv(res_path + filename)
        dfs.append(df)
        x = df['num_trajs'].values
        for elem in x:
            x_union.add(elem)

    x_union = sorted(list(x_union))
    ret = dict()
    for row, measure in enumerate(["mixture_reward", 'mixture_constraint']):
        ys = []
        for df in dfs:
            x = df["num_trajs"].values
            y = df[measure].values
            f = interp1d(x, y, fill_value="extrapolate")
            y_new = [f(x_new) for x_new in x_union]
            ys.append(y_new)

        ys = np.array(ys)
        x = x_union
        y = np.mean(ys, axis=0)
        error = np.std(ys, axis=0)

        ret[measure] = (x, y, error)
    # return ret

    f = plt.figure(constrained_layout=True, figsize=(10, 10))
    widths = [1]
    heights = [1, 1]
    spec = f.add_gridspec(ncols=1, nrows=2, width_ratios=widths, height_ratios=heights)
    spec.update(wspace=0.025, hspace=0.05)
    for row, measure in enumerate(["mixture_reward", 'mixture_constraint']):
        ax = f.add_subplot(spec[row, 0])

        (x, y, error) = ret[measure]
        if np.isnan(y):
            continue

        width = 2
        alpha = .3
        label = "Toc_UCRL2"
        color = "b"
        ax.plot(x, y, linewidth=width, label=f'{label}', markersize=15,color=color)
        ax.fill_between(x, y + error, y - error,alpha=alpha, color=color)


        ax.set(xlim=(0, 33000))
        if row == 0:
            ax.legend()

        if "constraint" in measure:
            ax.set_ylabel('constraint satisfaction', fontsize=20)
            ax.set(ylim=(0, 2))
            ax.set_xlabel("number of trajectories")
            ax.hlines(args.budget[0], 0, 33000, linestyles='dashed', linewidth=width, label='constraint')
        elif "reward" in measure:
            ax.set_ylabel("reward", fontsize=20)
            ax.set(ylim=(0, 2))
            if args.env == "gridworld":
                ax.hlines(1.4, 0, 33000, linestyles='dashed', linewidth=width, label='constraint')
            elif args.env == "box_gridworld":
                ax.hlines(1.58, 0, 33000, linestyles='dashed', linewidth=width, label='constraint')
    plt.tight_layout()
    plt.savefig(res_path+"UCRL.pdf")