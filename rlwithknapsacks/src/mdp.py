# -*- coding: utf-8 -*-
import numpy as np
from arsenal.maths import sample
import warnings

class CMDP(object):
    def __init__(self, s0, P, R, C, d, budget, Si, terminals):
        # P: Probability distribution p(S' | A S) stored as an array S x A x S'
        # R: Reward function r(S, A, S) -> Reals stored as an array S x A x S'
        # s0: Distribution over the initial state.
        self.s0 = s0
        [self.S, self.A, _] = P.shape
        # has a shape of (S,A,S)
        self.P = P
        # reward, has a shape of (S,A)
        self.R = R
        # constraints, has a shape of (d,S,A)
        self.C = C
        # number of constraints
        self.d = d
        # budget of constraints in shape (d)
        self.budget = budget
        self.Si = Si

        self.terminals = terminals

class FiniteHorizonCMDP(CMDP):
    """Finite-horizon MDP."""
    def __init__(self, s0, P, R, C ,d, budget, H, Si, terminals):
        super(FiniteHorizonCMDP, self).__init__(s0, P, R, C, d, budget, Si, terminals)
        self.H = H

class MarkovChain():
    "γ-discounted Markov chain."
    def __init__(self, s0, P, gamma):
        self.s0 = s0
        self.S = None
        self.P = P
        self.gamma = gamma

    def successor_representation(self):
        "Dayan's successor representation."
        return np.linalg.solve(np.eye(self.S) - self.gamma * self.P,
                               np.eye(self.S))
    def stationary(self):
        "Stationary distribution."
        return np.linalg.solve(np.eye(self.S) - self.gamma * self.P.T,  # note the transpose
                               (1-self.gamma) * self.s0)

    def eigenvalue_stationary(self):
        "Stationary distribution Eigen Values"
        pi = np.random.rand(13,1)
        for _ in range(100000): pi = pi.T.dot(self.P)
        return pi

class MRP(MarkovChain):
    "Markov reward process."
    def __init__(self, s0, P, R, gamma):
        super(MRP, self).__init__(s0, P, gamma)
        self.R = R
        self.gamma = gamma
        [self.S, _] = P.shape
        assert R.ndim == 1 and R.shape[0] == P.shape[0] == P.shape[1]

    def V(self):
        "Value function"
        return np.linalg.solve(np.eye(self.S) - self.gamma * self.P, self.R)

class DiscountedMDP(CMDP):
    "γ-discounted, infinite-horizon Markov decision process."
    #def __init__(self, s0, P, R, C, Si, gamma=None):
    def __init__(self, s0, P, R, C, d, budget, H, Si, terminals, gamma=None):
        # γ: Temporal discount factor
        super(DiscountedMDP, self).__init__(s0, P, R, C, d, budget, Si, terminals)
        self.gamma = gamma
        self.Si = Si
        self.H = H

    def run(self, learner):
        s = sample(self.s0)
        while True:
            a = learner(s)
            if np.random.uniform() <= (1-self.gamma):
                sp = sample(self.s0)
                r = 0
            else:
                sp = sample(self.P[s,a,:])
                r = self.R[s,a,sp]
            if not learner.update(s, a, r, sp):
                break
            s = sp

    def mrp(self, policy, R=None):
        "MDP becomes an `MRP` when we condition on `policy`."
        R = R if R is not None else self.R
        return MRP(self.s0,
                   np.einsum('sa,sap->sp', policy, self.P),
                   np.einsum('sa,sap,sap->s', policy, self.P, R),
                   self.gamma)

      #def J(self, policy):
      #    "Expected value of `policy`."
      #    return self.mrp(policy).J()

    def V(self, policy):
        "Value function for `policy`."
        return self.mrp(policy).V()

    def successor_representation(self, policy):
        "Dayan's successor representation."
        return self.mrp(policy).successor_representation()

    def dvisit(self, policy, R=None):
        "γ-discounted stationary distribution over states conditioned `policy`."
        return self.mrp(policy, R).stationary()

    def Q(self, policy, R=None):
        "Compute the action-value function `Q(s,a)` for a policy."
        R = R if R is not None else self.R

        v = self.V(policy)
        r = np.einsum('sap,sap->sa', self.P, R)
        Q = np.zeros((self.S, self.A))
        P = self.P

        for s in range(self.S):
            for a in range(self.A):
                Q[s,a] = r[s,a] + self.gamma*self.P[s,a,:].dot(v)
        return Q

    def Advantage(self, policy):
        "Advantage function for policy."
        return self.Q(policy) - self.V(policy)[None].T  # handles broadcast

    def B(self, V):
        "Bellman operator."
        # Act greedily according to one-step lookahead on V.
        Q = self.Q_from_V(V)
        pi = np.zeros((self.S, self.A))
        pi[range(self.S), Q.argmax(axis=1)] = 1
        v = Q.max(axis=1)
        return v, pi

    def Q_from_V(self, V):
          "Lookahead by a single action from value function estimate `V`."
          return (self.P * (self.R + self.gamma * V[None,None,:])).sum(axis=2)
