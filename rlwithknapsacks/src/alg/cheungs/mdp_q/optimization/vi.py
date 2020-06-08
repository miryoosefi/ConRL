import numpy as np
def vi(rew_dict, inst, epsilon, max_iters = 100000, verbose = True):
    S = inst["S"]
    A = inst["A"] 
    p = inst["p"] 
    v_mat = inst["v_mat"] 
    v_mean = inst["v_mean"] 
    reward_type = inst["reward_type"] 
    target = inst["target"] 
    
    # some state vectors
    len_S = len(S)
    u_prev = np.zeros(len_S)
    u_next = np.zeros(len_S)
    #Upsilon = dict()
    pi = dict()
    gap = 1
    t = 1
    while gap > epsilon and t <= max_iters:
        for s in S:
            inc_Upsilon = -10000.
            inc_a = -1
            for a in A[s]:
                Upsilon = rew_dict[(s, a)] + np.dot(p[(s, a)], u_prev)
                if Upsilon > inc_Upsilon:
                    inc_Upsilon = Upsilon
                    inc_a = a
            u_next[s] = inc_Upsilon
            pi[s] = np.eye(len(A[s]))[inc_a]
        gap = np.max(u_next - u_prev) - np.min(u_next - u_prev)
        if verbose:
            print("Iteration " + str(t) + ": the gap is " + str(gap))
        u_min = np.min(u_next)
        for s in S:
            u_prev[s] = u_next[s] - u_min
        t = t+1
    if gap <= epsilon:
        if verbose:
            print("Converges at iteration " + str(t))
            print("An upper bound to ave-opt is :" + str(np.max(u_next - u_prev) ))
        return pi
    else:
        print("The gap is still " + str(gap))
        raise Exception('VI fails to converge')
        
                
    