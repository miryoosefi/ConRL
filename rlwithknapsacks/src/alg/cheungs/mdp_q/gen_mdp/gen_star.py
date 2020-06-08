import numpy as np
def gen_star(K = 3, 
             branches = 5,
             center_gd = 1,
             branch_gd = 1,
             center_cf = 2,
             branch_cf = 2,
             prob_c = 0.9, # prob  should be \geq 0.5 higher prob => easier problem
             prob_b = 0.9,
             reward_type = "KPI", #reward_type = "exploration"  
             target = None):
# return

    # set up the state space
    S = list()
    for i in range(branches + 1):
        S.append(i)
    # State is indexed as 0, 1, ..., branches. State 0 is the center, state i is the ith branch.

    # set up the collection of actions. Expressed as a dictionary
    A = dict()
    # actions associated with the center state
    center_action_size = branches * center_gd + center_cf
    branch_action_size = 2 * branch_gd + branch_cf

    A[0] = np.arange(center_action_size)
    for s in 1 + np.arange(branches):
        A[s] = np.arange(branch_action_size)

    # make the list and dict of state-action pairs. 
    SA_list = list()
    SA_dict = dict()
    for s in S:
        SA_dict[s] = list()
        for a in A[s]:
            SA_list.append((s, a))
            SA_dict[s].append((s, a))



    # make the dictionary of transition kernal p
    p = dict()
    for (s, a) in SA_list:
        p[(s, a)] = np.zeros(branches + 1)

    # create permutations to mask the optimal actions
    np.random.seed()
    center_perm = np.random.permutation( center_action_size )
    ### prob for center state
    # for i \in [0, ..., branches - 1]:
    #     p( 0 | 0, center_perm[i]) = 1 - prob
    #     p( s | 0, center_perm[i]) = prob
    for i in range(branches * center_gd):
        rand_a = center_perm[i]
        # rand_a goes to branch i + 1 with high prob
        s_prime = (i // center_gd) + 1
        p[(0, rand_a)][s_prime] = prob_c
        p[(0, rand_a)][0] = 1. - prob_c

    for i in branches * center_gd + np.arange(center_cf):
        rand_a = center_perm[i]
        p[(0, rand_a)][0] = 1. / (1. + branches)
        for s in 1 + np.arange(branches):
            p[(0, rand_a)][s] = 1. / (1. + branches)

    ### prob for branch state

    # for i \in [branches, ..., branches + branch_cf - 1]:
    #     p( 0 | s, branch_perm[i]) = 1/2
    #     p( s | s, branch_perm[i]) = 1/2    
    # for i in 0:
    #     p( 0 | s, branch_perm[i]) = 1 - prob
    #     p( s | s, branch_perm[i]) = prob
    for s in 1 + np.arange(branches):
        # create permutations to mask the optimal actions
        np.random.seed()
        branch_perm = np.random.permutation( branch_action_size )
        for i in np.arange(branch_gd):
            rand_a = branch_perm[i]
            p[(s, rand_a)][0] = 1. - prob_b
            p[(s, rand_a)][s] = prob_b
        for i in branch_gd + np.arange(branch_gd):
            rand_a = branch_perm[i]
            p[(s, rand_a)][0] = prob_b
            p[(s, rand_a)][s] = 1. - prob_b
        for i in branch_gd * 2 + np.arange(branch_cf):
            rand_a = branch_perm[i]
            p[(s, rand_a)][0] = 0.5
            p[(s, rand_a)][s] = 0.5

    # make the vectorial feedback v
    v_mean = dict()
    if reward_type == "exploration":
        # define v_mat
        # define index v_index
        # each state visit frequency is a KPI. So "K" = 1 + branches.
        
        # define v_mat:
        v_mat = np.eye(branches + 1)
        
        # define v_mean
        for s in S:
            for a in A[s]:
                v_mean[(s, a)] = v_mat[s]
        # output target
        if target == None:
            target = np.ones(1 + branches) / (1. * branches)
            target[0] = 0.

    elif reward_type == "KPI":
        # associate each state with a subset of the K dimension
        # each action create frac values in the selected dimension.

        # define v_mat S \times K
        v_mat = np.zeros((1, K))
        for s in range(branches):
            #num_places = min(K, (2*K / 3) + 1)
            row_ind = np.random.choice(K, K // 2, replace = False)
            row = np.zeros(K)
            for row_i in row_ind:
                row[row_i] = np.random.uniform(low = 0.3, high = 1)
            row = row / np.sum(row)
            row = row.reshape((1, K))
            v_mat = np.concatenate((v_mat, row))        
        # define v_mean
        for s in S:
            if s != 0:
                for a in A[s]:
                    v_mean[(s, a)] = np.dot(p[(s, a)], v_mat)
            else:
                for a in A[s]:
                    v_mean[(s, a)] = np.zeros(K)
        # define target
        weight_coeff = np.random.uniform(low = 0., high = 1., size = branches)
        weight_coeff_norm = weight_coeff / np.sum(weight_coeff)
        target = np.zeros(K)
        for i in range(branches):
            target = target + weight_coeff_norm[i] * v_mat[i+1]
        
        #target = np.dot(np.ones(branches + 1), v_mat) / (1.2 * K) 
    else:
        raise ValueError('reward type should either be "exploration" or "KPI".')

    inst = dict()
    inst["S"] = S
    inst["A"] = A
    #inst["SA_dict"] = SA_dict
    #inst["SA_list"] = SA_list
    inst["p"] = p
    inst["v_mat"] = v_mat
    inst["v_mean"] = v_mean
    inst["reward_type"] = reward_type
    inst["target"] = target
    
    return inst
