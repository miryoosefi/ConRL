from rlwithknapsacks.src.alg.cheungs.mdp_q.gen_mdp import gen_g
from rlwithknapsacks.src.alg.cheungs.mdp_q.sim_tools import sim_one_step
from rlwithknapsacks.src.alg.cheungs.mdp_q.optimization import vi, evi
import numpy as np

def Toc_UCRL2_q(inst,
                T = 1000000,
                Q = None,
                delta = 0.1,
                tune_rad = True,
                uncertain_model = True,
                start_s = 0,
                verbose = False, 
                min_threshold = 0.0000001, # prevent evi from running forever
                coeff=None,
                args=None):

    metrics = []
    horizon = args.horizon
    S = inst["S"]
    A = inst["A"]
    len_S = len(S)
    len_A = 0
    for s in S:
        if len(A[s]) > len_A:
            len_A = len(A[s])
    reward_type = inst["reward_type"]
    K = inst["v_mat"].shape[1]
    target = inst["target"]

    if Q == None:
        if reward_type == "exploration":
            Q = 1. / np.sqrt(K)
        elif reward_type == "KPI":
            Q = 1. / np.sqrt(K)

    t = 1
    m = 1
    cur_s = start_s
    tau = list()
    tau.append(0) # place holder

    N_dict = dict()
    inc_N_dict = dict()
    v_hat_dict = dict()
    v_UCB_dict = dict()
    v_LCB_dict = dict()
    p_hat_dict = dict()
    p_UCB_dict = dict()
    p_LCB_dict = dict()

    traj = dict()
    traj["s_list"] = list()
    traj["s_list"].append(start_s)
    traj["a_list"] = list()
    traj["V_list"] = list()
    #record the start time of an episode caused by Psi or nu
    traj["Psi_list"] = list()
    traj["nu_list"] = list()
    #record the time intervales of the episodes stopped by Psi or nu
    traj["Psi_interval_list"] = list()
    traj["nu_interval_list"] = list()

    pi_list = list()
    bar_V = np.zeros(K)

    if reward_type == "exploration":
        v_mean = inst["v_mean"]
        for s in S:
            for a in A[s]:
                N_dict[(s, a)] = 0
                v_hat_dict[(s, a)] = v_mean[(s, a)]
                p_hat_dict[(s, a)] = np.ones(len_S) / (len_S * 1.)
        def grad_g(w):
            return (1. / K) * (target - w) 
    elif reward_type == "KPI":
        for s in S:
            for a in A[s]:
                N_dict[(s, a)] = 0
                v_hat_dict[(s, a)] = np.zeros(K)
                p_hat_dict[(s, a)] = np.ones(len_S) / (len_S * 1.)
        def grad_g(w):
            return (1. / K) * (coeff - np.maximum(w - target, 0))

    # run 
    #interval_list_to_put = "nu_interval_list"
    start_time = 1
    #traj["nu_list"].append(1)
    for m in range(1, T + 1):
        if t > T:
                break
        tau.append(t)
        print("Epoch " + str(m) + ", starting at time " + str(t))
        print(f"bar_V = {horizon*bar_V}")
        ################################
        # making a record for plotting #
        ################################
        record = {'mixture_reward': bar_V[0]*horizon,
         'mixture_constraint': bar_V[1]*horizon,
         'current_reward': bar_V[0]*horizon,
         'current_constraint': bar_V[1]*horizon,
         'alg': 'Toc_UCRL',
         'num_trajs': t//horizon,
         'expected_consumption':  bar_V[1]*t,
         'training_consumpution': bar_V[1]*t,
         'round': m}

        metrics.append(record)

        #
        # make Hvm
        if (uncertain_model == True) and (reward_type != "exploration"):
            for s in S:
                for a in A[s]:
                    v_UCB_dict[(s, a)] = np.zeros(K)
                    v_LCB_dict[(s, a)] = np.zeros(K)
                    N_plus = max(N_dict[(s, a)], 1 ) * 1.
                    if tune_rad == False:
                        log_v = np.log( 12. * K * len_S * len_A * t * t / delta )
                        for k in range(K):
                            v_rad = np.sqrt( 2 * v_hat_dict[(s, a)][k] * log_v / N_plus )
                            v_rad = v_rad + (3. * log_v / N_plus)
                            v_UCB_dict[(s, a)][k] = min( v_hat_dict[(s, a)][k] + v_rad , 1)
                            v_LCB_dict[(s, a)][k] = max( v_hat_dict[(s, a)][k] - v_rad , 0)
                    else:
                        for k in range(K):
                            v_rad_1 = np.sqrt( v_hat_dict[(s, a)][k] * np.log(t + 1) / N_plus)
                            v_rad_2 = np.log(t + 1) / N_plus
                            v_rad = max( v_rad_1 , v_rad_2  )
                            v_UCB_dict[(s, a)][k] = min( v_hat_dict[(s, a)][k] + v_rad , 1)
                            v_LCB_dict[(s, a)][k] = max( v_hat_dict[(s, a)][k] - v_rad , 0)

        # compute optimistic rewards
        cur_theta = grad_g(bar_V)
        opt_rew_dict = dict()
        if (uncertain_model == True) and (reward_type != "exploration"):
            for s in S:
                for a in A[s]:
                    opt_rew_dict[(s, a)] = 0.
                    for k in range(K):
                        if cur_theta[k] <= 0:
                            opt_rew_dict[(s, a)] = opt_rew_dict[(s, a)] + cur_theta[k] * v_LCB_dict[(s, a)][k]
                        else:
                            opt_rew_dict[(s, a)] = opt_rew_dict[(s, a)] + cur_theta[k] * v_UCB_dict[(s, a)][k]
        else:
            v_mean = inst["v_mean"]
            for s in S:
                for a in A[s]:
                    opt_rew_dict[(s, a)] = np.dot(cur_theta, v_mean[(s, a)])

        # make Hpm
        if uncertain_model == True:
            for s in S:
                for a in A[s]:
                    p_UCB_dict[(s, a)] = np.zeros(len_S)
                    p_LCB_dict[(s, a)] = np.zeros(len_S)
                    N_plus = max(N_dict[(s, a)], 1 ) * 1.
                    if tune_rad == False:
                        log_p = np.log( 12. * K * len_S * len_S * len_A * t * t / delta )
                        for s_prime in range(len_S):
                            p_rad = np.sqrt( 2 * p_hat_dict[(s, a)][s_prime] * log_p / N_plus )
                            p_rad = p_rad + (3. * log_p / N_plus)
                            p_UCB_dict[(s, a)][s_prime] =\
                            min( p_hat_dict[(s, a)][s_prime] + p_rad , 1)
                            p_LCB_dict[(s, a)][s_prime] =\
                            max( p_hat_dict[(s, a)][s_prime] - p_rad , 0)
                    else:
                        for s_prime in range(len_S):
                            p_rad_1 = np.sqrt( p_hat_dict[(s, a)][s_prime] * np.log(t + 1) / N_plus)
                            p_rad_2 = np.log(t + 1) / N_plus
                            p_rad = max( p_rad_1 , p_rad_2  )
                            p_UCB_dict[(s, a)][s_prime] = min( p_hat_dict[(s, a)][s_prime] + p_rad , 1)
                            p_LCB_dict[(s, a)][s_prime] = max( p_hat_dict[(s, a)][s_prime] - p_rad , 0)

        # solve for tilde pi 
        if uncertain_model == True:
            epsilon = min(0.001 /t, min_threshold)
            pi = evi(opt_rew_dict, 
                     inst, 
                     p_UCB_dict, p_LCB_dict, 
                     epsilon, verbose = verbose)
            pi_list.append(pi)
        else:
            # vi
            epsilon = min(0.001 / t, min_threshold)
            pi = vi(opt_rew_dict,
                    inst,  
                    epsilon, verbose = verbose)
            pi_list.append(pi)

        nu = dict()
        for s in S:
            for a in A[s]:
                nu[(s, a)] = 0
                inc_N_dict[(s, a)] = max(N_dict[(s, a)], 1)

        ref_theta = grad_g(bar_V)
        Psi = 0.

        test_a = np.random.choice(A[cur_s], p = pi[cur_s])
        while ( Psi <= Q ) and ( nu[(cur_s, test_a)] < inc_N_dict[(cur_s, test_a)] ):
            cur_a = test_a
            next_s, cur_V = sim_one_step(cur_s, cur_a, inst)
            # update statistics
            N_dummy = N_dict[(cur_s, cur_a)]
            N_dict[(cur_s, cur_a)] = N_dummy + 1

            v_dummy = v_hat_dict[(cur_s, cur_a)]
            v_hat_dict[(cur_s, cur_a)] = \
            (1. / (N_dummy + 1.)) * cur_V + (N_dummy / (N_dummy + 1.)) * v_dummy

            p_dummy = p_hat_dict[(cur_s, cur_a)]
            p_hat_dict[(cur_s, cur_a)] = \
            (1. / (N_dummy + 1.)) * np.eye(len_S)[next_s] + \
            (N_dummy / (N_dummy + 1.)) * p_dummy

            traj["s_list"].append(next_s)
            traj["a_list"].append(cur_a)
            traj["V_list"].append(cur_V)
            bar_V = (1. / t) * cur_V + (t - 1.) * bar_V / t

            # update markers
            cur_theta = grad_g(bar_V)
            Psi = Psi + np.linalg.norm(cur_theta - ref_theta)
            nu_dummy = nu[(cur_s, cur_a)]
            nu[(cur_s, cur_a)] = nu_dummy + 1

            # update current state and action
            cur_s = next_s
            test_a = np.random.choice(A[cur_s], p = pi[cur_s])
            # some print out
            #print str(Psi)
            if Psi > Q:
                print ("Break because of Psi > Q")
                traj["Psi_list"].append(t+1)
                traj["Psi_interval_list"].append(range(start_time, t + 1))
                start_time = t + 1
            elif nu[(cur_s, test_a)] >= inc_N_dict[(cur_s, test_a)]:
                print ("Break because of nu")
                traj["nu_list"].append(t+1)
                traj["nu_interval_list"].append(range(start_time, t + 1))
                start_time = t + 1
            # update time :)
            t = t + 1

    res = dict()
    res["s_list"] = traj["s_list"][:T+1]
    res["a_list"] = traj["a_list"][:T]
    res["V_list"] = traj["V_list"][:T]
    res["bar_V"] = np.mean(res["V_list"], axis = 0)
    g = gen_g(inst,coeff)
    res["obj_val"] = g(res["bar_V"])
    res["Psi_list"] = traj["Psi_list"]
    res["nu_list"] = traj["nu_list"]
    res["Psi_interval_list"] = traj["Psi_interval_list"]
    res["nu_interval_list"] = traj["nu_interval_list"]
    return res, metrics
