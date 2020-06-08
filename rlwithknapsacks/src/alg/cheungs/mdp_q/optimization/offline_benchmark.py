import cvxpy as cp
import numpy as np
from rlwithknapsacks.src.alg.cheungs.mdp_q.gen_mdp import gen_g

def offline_benchmark(inst, round_thres = 0.000001):
    S = inst["S"]
    A = inst["A"]
    v_mean = inst["v_mean"]
    K = inst["v_mat"].shape[1]
    p = inst["p"]
    
    len_S = len(S)
    len_A = 0
    for s in S:
        if len(A[s]) > len_A:
            len_A = len(A[s])
            
    x = dict()
    for s in S:
        for a in A[s]:
            x[(s, a)] = cp.Variable()    
            
    # define constraints
    constraints = []
    # sum to one constraint, non_neg constraints
    cum_sum = 0
    for s in S:
        for a in A[s]:
            cum_sum = cum_sum + x[(s, a)]
            constraints.append(x[(s, a)] >= 0)
    constraints.append(cum_sum == 1.)
    
    # conservation of flow constraints
    for s in S:
        lhs = 0.
        for a in A[s]: 
            lhs = lhs + x[(s, a)]
        rhs = 0.
        for s_prime in S:
            for a_prime in A[s_prime]:
                rhs = rhs + p[(s_prime, a_prime)][s] * x[(s_prime, a_prime)]
        constraints.append(lhs == rhs)
        
    # define objective
    target = inst["target"]
    reward_type = inst["reward_type"]
    obj_arg = dict()
    for k in range(K):
        obj_arg[k] = 0.
        for s in S:
            for a in A[s]:
                obj_arg[k] = obj_arg[k] + v_mean[(s, a)][k] * x[(s, a)]
    
    if reward_type == "exploration":
        obj = 0.
        for k in range(K):
            obj = obj - (target[k] - obj_arg[k]) ** 2
        obj = obj / (2. * K)
        objective = cp.Maximize(obj)
    elif reward_type == "KPI":
        linear_obj = 0.
        quad_obj = 0.
        aux_var = dict()
        for k in range(K):
            aux_var[k] = cp.Variable()
            constraints.append(aux_var[k] >= 0)
            constraints.append(aux_var[k] >= target[k] - obj_arg[k])
            linear_obj = linear_obj + obj_arg[k]
            quad_obj = quad_obj - aux_var[k] ** 2
        linear_obj = linear_obj / (1. * K)
        quad_obj = quad_obj / (2. * K)
        objective = cp.Maximize(linear_obj + quad_obj)
    else:
        raise ValueError("reward_type has to be exploration or KPI")
        # to do: more reward types.
    
    # define the optimization problem
    opt_prob = cp.Problem(objective, constraints)
    opt_prob.solve()  # Returns the optimal value.
    res = dict()
    res["status"] = opt_prob.status
    #res["opt_val"] = opt_prob.value
    soln_dict = dict()
    norm_sum = 0.
    for s in S:
        for a in A[s]:
            soln_dict[(s, a)] = x[(s, a)].value
            if soln_dict[(s, a)] <= round_thres:
                soln_dict[(s, a)] = 0.
            norm_sum = norm_sum + soln_dict[(s, a)]
    g = gen_g(inst)
    round_v = np.zeros(K)    
    for s in S:
        for a in A[s]:
            dummy = soln_dict[(s, a)]
            soln_dict[(s, a)] = dummy / norm_sum
            round_v = round_v + soln_dict[(s, a)] * v_mean[(s, a)]   
    res["opt_v_arg"] = round_v
    res["opt_val"] = g(round_v)
    if abs(res["opt_val"] - opt_prob.value) > round_thres:
        raise ValueError("Rounding goes wrong, please lower round_thres")
    res["opt_soln"] = soln_dict
    return res
