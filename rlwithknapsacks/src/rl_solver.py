import numpy as np
from arsenal.maths import sample
import warnings
from collections import defaultdict
from arsenal import Alphabet
from rlwithknapsacks.src.policy import SoftmaxPolicy
from rlwithknapsacks.src.baseline import Averaging, EWMA, SimpleMovingAveraging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class RL_Solver:
    def __init__(self, M=None, args=None):
        self.M = M
        self.P = M.P
        self.A = M.A
        self.H = M.H
        self.args=args

        self.S = M.S
        self.R = np.einsum('sap,sap->sa', M.P, M.R) if len(M.R.shape) == 3 else M.R
        self.C = np.einsum('sap,sap->sa', M.P, M.C) if len(M.C.shape) == 3 else M.C
        self.s0 = M.s0

        self.stats = defaultdict(int)

        self.d = M.d
        self.Si = M.Si
        self.budget = M.budget

        self.num_episodes = 1

    def __call__(self):
        pass

    def run(self, π_list): #, P=None, R=None, C=None):
        (S, A, P, R, C, H) = (self.S, self.A, self.P, self.R, self.C, self.H)

        p_traj = np.zeros((S,A,S))
        r_traj = np.zeros((S,A))
        c_traj = np.zeros((S,A))
        visitation = np.zeros((S,A))

        if len(np.shape(R)) == 3:
            R = np.einsum("sap,sap->sa",P,R)
        if len(np.shape(C)) == 3:
            C = np.einsum("sap,sap->sa",P,C)
            #C = np.einsum("sap,dsap->dsa",P,C)

        s = sample(self.s0)
        for h in range(H):
            #a = sample(π_list[h,s,:])
            try:
                a = π_list[h,s,:].argmax()
            except:
                a = π_list(s)

            s_next = sample(P[s,a,:])
            p_traj[s,a,s_next] += 1.0
            r_traj[s,a] += R[s,a]
            c_traj[s,a] += C[s,a]
            visitation[s,a]+= 1.0
            s = s_next
        self.stats['training_consumpution'] += c_traj.sum()
        self.stats['num_trajs'] += 1
        return p_traj,r_traj,c_traj,visitation, self.Si.lookup(s)

    def value_evaluation(self, π_list, P=None, R=None, C=None):
        """Compute `H`-step value functions for `policy`."""
        #S = self.S;A = self.A;H = self.H;d = self.d;
        update_consumption = False
        if P is None and R is None and C is None:
            update_consumption = True

        (S, A, H, d) = (self.S, self.A, self.H, self.d)

        C = self.C if C is None else C
        P = self.P if P is None else P
        R = self.R if R is None else R

        Q = np.zeros((H+1,S,A))
        V = np.zeros((H+1,S))
        V_C = np.zeros((H+1,S))
        Q_C = np.zeros((H+1,S,A))
        for h in reversed(range(H)):
            for s in range(S):
                for a in range(A):
                    Q[h,s,a] = R[s,a]+ P[s,a,:].dot(V[h+1,:])
                    Q_C[h,s,a,]= C[s,a] + P[s,a,:].dot(V_C[h+1,:])
                V_C[h,s]=π_list[h,s,:].dot(Q_C[h,s,:])
                V[h,s] = π_list[h,s,:].dot(Q[h,s,:])

        if update_consumption:
            self.stats['expected_consumption'] += self.s0.dot(V_C[0,:])*self.num_episodes
        return {
            'V': V,
            'V_C': V_C,
            'Q':   Q,
            'Q_C': Q_C,
            'reward': self.s0.dot(V[0,:]),
            'constraint': self.s0.dot(V_C[0,:]),
        }

    def reset(self):
        pass

    def reset_oracle(self):
        pass


class ValueIteration(RL_Solver):

    def __init__(self, M=None, args=None):
        super(ValueIteration, self).__init__(M=M, args=args)

    def __call__(self, P=None, R=None, theta=None, fic=None):
    #def value_iteration(self, P=None, R=None):
        """Compute optimal `H`-step Q-functions and policy"""
        S = self.S; A = self.A; H = self.H
        Q = np.zeros((H+1,S,A))
        V = np.zeros((H+1,S))

        R = self.R if R is None else R
        P = self.P if P is None else P
        R = np.einsum('sap,sap->sa', P, R) if len(R.shape) == 3 else R
        π_list = np.zeros((H+1,S,A))
        for h in reversed(range(H)):
            Q[h,:,:] = R + P.dot(V[h+1,:])
            π_list[h][range(S), Q[h].argmax(axis=1)] = 1
            V[h] = Q[h].max(axis=1)

        return {
            'V': V,
            'Q': Q,
            'pi_list': π_list,
            'last_state': (0,0)
         }


class UCBVI(RL_Solver):
    def __init__(self, M=None, args=None):
        super(UCBVI, self).__init__(M=M, args=args)

        self.p_sum = np.zeros((self.S, self.A, self.S))
        self.r_sum = np.zeros((self.S, self.A))
        self.c_sum = np.zeros((self.S, self.A))
        self.visitation_sum = np.zeros((self.S, self.A))

    def __call__(self, P=None, R=None, theta=None, n_iter_ucbvi=1):
        p_hat = np.zeros((self.S, self.A, self.S))
        r_hat = np.zeros((self.S, self.A))
        c_hat = np.zeros((self.S, self.A))
        bonus = np.zeros((self.S, self.A))

        if np.linalg.norm(theta) != 0:
            theta = theta/np.linalg.norm(theta)

        for _ in range(n_iter_ucbvi):
            for s in range(self.S):
                for a in range(self.A):
                    visitation = self.p_sum[s, a, :].sum()
                    if visitation == 0:
                        p_hat[s, a, s] = 1
                        bonus[s, a] = 1 #* len(theta)
                    else:
                        p_hat[s, a, :] = self.p_sum[s, a, :] / visitation
                        r_hat[s, a] = self.r_sum[s, a] / visitation
                        c_hat[s, a] = self.c_sum[s, a] / visitation
                        bonus[s, a] = np.sqrt(1.0 / visitation) * self.args.bonus_coef #* len(theta)

            helper_solver = ValueIteration(M=self.M, args=self.args)
            values = helper_solver(P=p_hat, R=theta[0]*r_hat+theta[1]*c_hat+bonus)
            π_list = values['pi_list']

            (p, r, c, v, last_state) = self.run(π_list)
            # Update Counts
            self.p_sum += p
            self.r_sum += r
            self.c_sum += c
            self.visitation_sum += v

            results = {
                'V': values['V'],
                'Q': values['Q'],
                'pi_list': π_list,
                'last_state': last_state
            }
        return results

class PolicyGradient(RL_Solver):
    def __init__(self, M=None, args=None):
        super(PolicyGradient, self).__init__(M=M, args=args)

        self.D = M.S * M.A + M.S + M.A
        self.baseline = [EWMA() for _ in range(self.S)]

        alphabet = Alphabet()
        State = Alphabet()
        def features(s,a):
            f = np.zeros(self.D)
            (r,c) = self.Si.lookup(s)
            #f[alphabet['action-bias',a]] = 1
            #f[alphabet['state-bias',s]] = 1
            #f[alphabet[s,a]] = 1      # no feature aliasing
            state = State[(r,c)]
            f[alphabet[state,a]] = 1
            #f[alphabet[z,a]] = 1
            return f

        self.π = SoftmaxPolicy(self.A, self.D, self.S, features, baseline=EWMA)
        self.initial_state = args.initial_state
        self.pg_type = args.solver

    def __call__(self, P=None, R=None, theta=None, n_iter_ucbvi=1):
        R = self.R if R is None else R
        (C, H, Si, P, S, A) = (self.C, self.H, self.Si, self.P, self.S, self.A)

        num_episodes = 10
        for _ in range(num_episodes):
          [_, g] = self.π.policy_gradient(self.M, R)
          self.π.update(-g, learning_rate=.01)

        (_, _, _, _, last_state) = self.run(self.π)

        π_table = np.repeat(self.π.to_table(self.M)[np.newaxis, :, :], H, axis=0)
        results = {
            'V': None,
            'Q': None,
            'pi_list': π_table,
            'last_state': last_state
        }
        return results

def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim, grid_size):
        super(Policy, self).__init__()
        #self.affine1 = nn.Linear(input_dim,128)
        channels = 16
        self.grid_size = grid_size
        size = self.grid_size
        self.conv = layer_init(nn.Conv2d(2, channels, kernel_size=3, stride=1, padding=1))
        self.affine1 = layer_init(nn.Linear(channels * size * size, 64))
        self.affine2 = layer_init(nn.Linear(64,output_dim))
        self.value_head = layer_init(nn.Linear(64,1))

        self.saved_log_probs =[]
        self.values = []
        self.rewards = []
        self.entropies = []

    def forward(self, x):
        x = F.relu(self.conv(x))
        # self.linear(x.view(x.shape[0], -1))
        x = F.relu(self.affine1(x.view(x.shape[0], -1)))
        #x = F.relu(x)
        action_scores = self.affine2(x)
        value = self.value_head(x)
        return F.softmax(action_scores, dim=1), value


class A2C(RL_Solver):
    def __init__(self, G=None, M=None, args=None):
        super(A2C, self).__init__(M=M, args=args)

        self.G = G
        self.input_size = len(G.get_representation(s=0, Si=self.Si))
        self.output_size = self.A
        self.grid_size = np.shape(G.get_representation(s=0, Si=self.Si))[1]
        self.discount_factor = args.discount_factor
        self.learning_rate = args.actor_critic_lr
        self.policy = Policy(self.input_size, self.output_size, self.grid_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.eps = np.finfo(np.float32).eps.item()

        self.value_coef = args.value_coef
        self.entropy_coef = args.entropy_coef

        self.num_episodes = args.num_episodes
        self.num_fic_episodes = args.num_fic_episodes

    def reset_oracle(self, params=None):
        if params is None:
            self.policy = Policy(self.input_size, self.output_size, self.grid_size)
        else:
            self.policy.load_state_dict(params)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

    def select_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, value = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        self.policy.values.append(value)
        self.policy.entropies.append(m.entropy())
        return action.item()

    def finish_episode(self):
        R = 0
        policy_loss = []
        value_loss = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r + self.discount_factor * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        #returns = (returns - returns.mean()) / (returns.std() + self.eps)
        for log_prob,value, R in zip(self.policy.saved_log_probs,self.policy.values, returns):
            advantage = R - value.item()
            policy_loss.append(-log_prob * advantage)
            R = torch.tensor([R]).float()
            value_loss.append(F.smooth_l1_loss(value, R.reshape(-1, 1)))
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).mean() \
               + (self.value_coef * torch.stack(value_loss).mean()) \
               - (self.entropy_coef * torch.stack(self.policy.entropies).mean())
        #policy_loss = torch.cat(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]
        del self.policy.values[:]
        del self.policy.entropies[:]

    def to_table(self):
        π = np.zeros((self.S,self.A))
        for s in range(self.S):
            rep = self.G.get_representation(s=s, Si=self.Si)
            rep = np.asarray(rep)
            rep = torch.from_numpy(rep).float().unsqueeze(0)
            with torch.no_grad():
                probs, _ = self.policy(rep)
            π[s,:] = probs.cpu().detach().numpy()
        return π

    def __call__(self, P=None, R=None, theta=None, fic=False):
        if fic:
            R = np.einsum('sap,sap->sa', P, R) if len(R.shape) == 3 else R
            P = P
            num_episodes = self.num_fic_episodes
        else:
            R = self.R if R is None else R
            P = self.P if P is None else P
            num_episodes = self.num_episodes

        (C, H, Si, S, A) = (self.C, self.H, self.Si, self.S, self.A)

        s = None
        for t in range(num_episodes):
            s = sample(self.s0)
            c = 0
            for h in range(H):
                rep = self.G.get_representation(s=s, Si=self.Si)
                #rep = np.asarray(rep)
                a = self.select_action(rep)
                sp = sample(P[s, a])
                try:
                    r = R[s, a, sp]
                except:
                    r = R[s,a]
                c += C[s,a]
                self.policy.rewards.append(r)
                s = sp
            self.finish_episode()
            self.stats['training_consumpution'] += c
        self.stats['num_trajs'] += self.num_episodes
        π_list = np.repeat(self.to_table()[np.newaxis, :, :], H, axis=0)
        results = {
            'V': None,
            'Q': None,
            'pi_list': π_list,
            'last_state': self.Si.lookup(s)
        }
        return results
