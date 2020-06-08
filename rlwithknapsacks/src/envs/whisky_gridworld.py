# -*- coding: utf-8 -*-
"""
Grid world environment
"""

import pylab as pl
import numpy as np
from matplotlib.table import Table
import os

from arsenal import Alphabet
from arsenal.viz import update_ax

class Action(object):
    def __init__(self, name, dx, dy):
        self.name = name
        self.dx = dx
        self.dy = dy
    def __iter__(self):
        return iter((self.dx, self.dy))
    def __repr__(self):
        return self.name


class WhiskyGridWorld(object):
    """A two-dimensional grid MDP.  All you have to do is specify the grid as a list
    of lists of rewards; use None for an obstacle. Also, you should specify the
    terminal states.  An action is an (x, y) unit vector; e.g. (1, 0) means move
    east.

    """
    EMPTY, WALL, START, GOAL, PIT, WHISKY = np.array(['0','W','S','G', 'P', 'Y'], dtype='|S1')

    def __init__(self, args=None):
        self.mapname = args.map
        self.H = args.horizon
        self.randomness = args.randomness

        # Load Map
        __location__ = os.path.dirname((os.path.abspath(__file__)))
        default_map_dir = os.path.join(__location__, "maps")

        self.ax = None
        self.A = [
            Action('⬆',0,-1), # up     b/c 0,0 is upper left
            Action('⬇',0,1),  # down
            Action('⬅',-1,0), # left
            Action('➡',1,0),  # right
        ]

        self.states = set()
        self.reward = {}
        self.constraint = {}
        self.walls = set()

        self.grid = grid = np.loadtxt(os.path.join(default_map_dir, self.mapname), dtype='|S1')
        [self.rows, self.cols] = self.grid.shape
        self.num_states = grid.size

        # Parse Map
        self.initial_state = np.argwhere(self.grid == self.START)[0]
        #self.box = np.argwhere(self.grid == self.BOX)[0]
        self.initial_state = tuple(self.initial_state.tolist() + [0])# + self.box.tolist())

        self.whisky_state = set()

        #self.terminals = np.argwhere(self.grid == self.GOAL)[0]
        #self.terminals = [tuple(self.terminals)]
        self.terminals = []

        # Convert Numpy bytes to int
        new_grid = np.zeros(self.grid.shape)

        params = [(r, c, d)
                    for r in range(self.rows)
                    for c in range(self.cols)
                    for d in [0,1]]

        for (r, c, d) in params:
            # Agent cant be in wall location
            # Block cant be in wall location
            if grid[r, c] != self.WALL:
                s = (r, c, d)
                self.states.add(s)

                if grid[r,c] == self.GOAL:
                    self.terminals.append(tuple(s))
                    r_value = 1.0
                elif grid[r,c] == self.START: r_value = 0.0
                elif grid[r,c] == self.EMPTY: r_value = 0.0
                elif grid[r,c] == self.WHISKY:
                    self.whisky_state.add(tuple(s))
                    r_value = 0.0
                else: raise Exception("Unknown grid attribute: ", grid[r, c])

                self.reward[s] = r_value
            elif grid[r, c] == self.WALL:
                self.walls.add((r,c))


        #new_grid = np.where(np.isnan(new_grid), None, new_grid)
        #self.grid = new_grid.copy()

    def encode(self):
        # Encode problem into an MDP so we can call its solver.
        Si = Alphabet()
        Ai = Alphabet()
        s0 = np.zeros(len(self.states))
        s0[Si[self.initial_state]] = 1
        P = np.zeros((len(self.states), len(self.A), len(self.states)))
        R = np.zeros((len(self.states), len(self.A), len(self.states)))
        C = np.zeros((len(self.states), len(self.A), len(self.states)))

        for s in self.states:
            si = Si[s]
            if s[2] == 1:
                self.randomness = 0.9
            else:
                self.randomness = 0.0
            for a in self.A:
                ai = Ai[a]
                sp, r, c = self.simulate(s, a)
                spi = Si[sp]
                P[si,ai,spi] += 1 * (1-self.randomness)
                R[si,ai,spi] = r
                C[si,ai,spi] = 0.0

                for a_r in self.A:
                    ai_r = Ai[a_r]
                    sp, r, c = self.simulate(s, a_r)
                    spi = Si[sp]
                    P[si, ai, spi] += 1 * self.randomness/(len(self.A))
                    R[si, ai, spi] = r
                    C[si, ai, spi] = 0.0

        Si.freeze(); Ai.freeze()
        return (s0, P, R, C), Si, Ai

    def simulate(self, s, a):
        if s in self.terminals:
            return s, self.reward[s], 0.0

        # Transition the agent to next state
        dx, dy = a
        sp = [s[0] + dy, s[1] + dx, s[2]]

        if tuple(sp) in self.whisky_state:
            sp[2] = max(s[2], 1.0)
        sp = tuple(sp)

        if sp in self.states:
            sp = sp
            r = self.reward[sp]
        else:
            sp = s # stay in same state if we hit a wall
            r = 0.0
        c = 0.0
        return sp, r, c
