import numpy as np


def gen_g(inst, coeff=None):
    K = inst["v_mat"].shape[1]
    target = inst["target"]
    reward_type = inst["reward_type"] 


    if reward_type == "exploration":
        def g(w):
            diff = w - target
            return - (0.5 / K) * np.dot(diff, diff)
    elif reward_type == "KPI":
        def g(w):
            diff = w - target
            diff = np.maximum(diff, 0.)
            return - (0.5 / K) * np.dot(diff, diff) + (1. / K) * np.dot(coeff, w)
    return g