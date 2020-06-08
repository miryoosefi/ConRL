import numpy as np
def sim_one_step(s, a, inst):
    # It simulate, given a state s and action a, for the outocmes:
    # V(s, a), s' \sim p(\cdot |s, a)
    S = inst["S"]
    #A = inst["A"] 
    #inst["SA_dict"] = SA_dict
    #SA_list = inst["SA_list"] 
    p = inst["p"] 
    v_mat = inst["v_mat"] 
    v_mean = inst["v_mean"] 
    reward_type = inst["reward_type"] 
    target = inst["target"]
    
    K = v_mat.shape[1]
    s_next = np.random.choice(S, size = None, p = p[(s,a)])
    if reward_type == "exploration":
        V = v_mat[s]
        return s_next, V
    elif reward_type == "KPI":
        if s != 0:
            V = v_mean[(s,a)]
        else:
            V = np.zeros(K)
        return s_next, V
    else:
        raise ValueError('reward type should either be "exploration" or "KPI".')