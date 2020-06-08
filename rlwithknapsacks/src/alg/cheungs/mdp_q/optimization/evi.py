import numpy as np

def evi(rew_dict, inst,p_UCB_dict, p_LCB_dict,
       epsilon, max_iters = 10000, verbose = True):
    
    S = inst["S"]
    A = inst["A"] 
    reward_type = inst["reward_type"] 
    target = inst["target"] 
    
    # some state vectors
    len_S = len(S)
    u_prev = np.zeros(len_S)
    u_next = np.zeros(len_S)
    tilde_p = dict()
    #Upsilon = dict()
    pi = dict()
    gap = 1
    t = 1
    while gap > epsilon and t <= max_iters:
        # sort u in non-increasing order 
        for s in S:
            for a in A[s]:
                tilde_p[(s, a)] = np.zeros(len_S)
                for s_prime in S:
                    tilde_p[(s, a)][s_prime] = p_LCB_dict[(s, a)][s_prime]
        order_S = np.argsort((-1.) * u_prev)
        for s in S:
            for a in A[s]:
                track_sum = np.sum(tilde_p[(s, a)])
                track_ind = 0
                while (track_sum <= 1.) and (track_ind < len_S):
                    cur_s = order_S[track_ind]
                    increment = p_UCB_dict[(s, a)][cur_s] - p_LCB_dict[(s, a)][cur_s]
                    old_track_sum = track_sum
                    tilde_p[(s, a)][cur_s] = p_UCB_dict[(s, a)][cur_s]
                    track_sum = track_sum + increment
                    track_ind = track_ind + 1
                tilde_p[(s, a)][cur_s] = 1. - old_track_sum + p_LCB_dict[(s, a)][cur_s]
                if np.sum(tilde_p[(s, a)]) > 1.00001 or np.sum(tilde_p[(s, a)]) < 0.99999:
                    raise ValueError("Something is wrong with the tilde p updating.")
        for s in S:
            inc_Upsilon = -10000000.
            inc_a = -1
            for a in A[s]:
                Upsilon = rew_dict[(s, a)] + np.dot(tilde_p[(s, a)], u_prev)
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

