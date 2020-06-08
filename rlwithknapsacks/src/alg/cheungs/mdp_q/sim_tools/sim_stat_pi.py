import numpy as np
from . import sim_one_step

def sim_stat_pi(pi, inst, s1 = 0, nb_steps = 1):
    #simulate a stationary policy pi, with starting state s1, for nb_steps many time steps
    # in instance inst. Return the sequence of V, s'
    
    # pi should be a dictionary, pi[s] = a distribution over A[s]
    S = inst["S"]
    A = inst["A"] 
    #inst["SA_dict"] = SA_dict
    SA_list = inst["SA_list"] 
    p = inst["p"] 
    v_mat = inst["v_mat"] 
    v_mean = inst["v_mean"] 
    reward_type = inst["reward_type"] 
    target = inst["target"]        

    a_list = list()
    s_list = list()
    s_cur = s1
    s_list.append(s_cur)
    V_list = list()
    
    for t in range(nb_steps):
        np.random.seed()
        a = np.random.choice(A[s_cur], size = None, p=pi[s_cur])
        s, V = sim_one_step(s_cur, a, inst)
        s_cur = s
        a_list.append(a)
        s_list.append(s)
        V_list.append(V)
    traj = dict()
    traj["s_list"] = s_list
    traj["a_list"] = a_list
    traj["V_list"] = V_list
    return traj