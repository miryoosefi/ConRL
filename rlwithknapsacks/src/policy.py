import pylab as pl
import numpy as np
from arsenal.maths.checkgrad import fdcheck
from arsenal.maths.stepsize import adam
from arsenal.maths import softmax, onehot, sample
#from collections import defaultdict
from arsenal import iterview, Alphabet
from arsenal import viz
import copy
from rlwithknapsacks.src.sample import categorical, random_choice

class LinearPolicy(object):

    def __init__(self, A, D, S, features, baseline=None):
        self.A = A
        self.D = D
        self.S = S
        self.weights = np.zeros(D)
        self.update = adam(self.weights)
        self.features = features
        self.baseline = [baseline() for _ in range(S)]

    def __call__(self, s):
        return categorical(self.scores(s))
        #return random_choice(self.p(s))

    def to_table(self, M):
        pi = np.zeros((M.S, M.A))
        for s in range(M.S):
            pi[s,:] = self.p(s)
        return pi

    def scores(self, s):
        return np.array([self.weights.dot(self.features(s,a)) for a in range(self.A)])

    def solve_csc(self, M, rollin, rollout):
        """
        Solve the cost-sensitive classification method induced by (rollin,rollout).
        where `rollin` weights each state's importance and `rollout` assess
        the reward of each action in context.
        """
        from scipy.optimize import minimize
        return minimize(lambda w: self.csc_objective(w, M, rollin, rollout),
                        self.weights, jac=True).x

    def csc_objective(self, w, M, rollin, rollout):
        raise NotImplementedError

class SoftmaxPolicy(LinearPolicy):

    def p(self, s):
        "conditional probability of each action in state s."
        return softmax(self.scores(s))

    def dlogp(self, s, a):
        "Compute ∇ log p(a | s)"
        d = np.zeros(self.D)
        d = self.features(s,a)
        p = self.p(s)
        for ap in range(self.A):
            d -= p[ap] * self.features(s, ap)
        return d

    def r(self, t, r, b, γ, avg=False):
        """Reward function."""
        total = 0.0
        count = 1.0
        for count, tau in enumerate(range(t, len(r)), 1):
            total += (γ**(tau-t)*r[tau]) - b
        return total/float(count) if avg else total

    def reinforce(self, sa, r, M):
        "Reinforce"
        #A = M.Advantage(pi)
        g = []
        γ = .99
        b = self.baseline[0]()
        for t, traj in enumerate(sa):
            s,a = traj
            b = self.baseline[s]()
            g.append(self.dlogp(s,a) * self.r(t, r, 0, γ))
            self.baseline[s].update(self.r(0, r, 0, γ, avg=True))
        return None, g

    def reinforce(self, sa, r, M):
        "Reinforce"
        #A = M.Advantage(pi)
        g = []
        γ = .99
        b = self.baseline[0]()
        for t, traj in enumerate(sa):
            s,a = traj
            b = self.baseline[s]()
            g.append(self.dlogp(s,a) * self.r(t, r, 0, γ))
            self.baseline[s].update(self.r(0, r, 0, γ, avg=True))
        return None, g



    def policy_gradient(self, M, R):
        "Policy gradient"
        pi = self.to_table(M)
        d = M.dvisit(pi, R)
        Q = M.Q(pi, R)
        g = np.zeros_like(self.weights)
        for s in range(M.S):
            for a in range(M.A):
                g += d[s] * pi[s, a] * Q[s, a] * self.dlogp(s, a)
        return None,g
