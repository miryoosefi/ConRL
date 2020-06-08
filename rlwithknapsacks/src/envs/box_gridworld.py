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


class BoxGridWorld(object):
    """A two-dimensional grid MDP.  All you have to do is specify the grid as a list
    of lists of rewards; use None for an obstacle. Also, you should specify the
    terminal states.  An action is an (x, y) unit vector; e.g. (1, 0) means move
    east.

    """
    EMPTY, WALL, START, GOAL, PIT, BOX = np.array(['0','W','S','G', 'P', 'X'], dtype='|S1')

    def __init__(self, args=None):
        self.mapname = args.map
        self.H = args.horizon
        self.steps = 1.0/self.H
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
        self.box = np.argwhere(self.grid == self.BOX)[0]
        self.initial_state = tuple(self.initial_state.tolist() + self.box.tolist())

        #self.terminals = np.argwhere(self.grid == self.GOAL)[0]
        #self.terminals = [tuple(self.terminals)]
        self.terminals = []

        # Convert Numpy bytes to int
        new_grid = np.zeros(self.grid.shape)

        params = [(ra, ca, rb, cb)
                    for ra in range(self.rows)
                    for ca in range(self.cols)
                    for rb in range(self.rows)
                    for cb in range(self.cols) ]

        for (ra, ca, rb, cb) in params:
            # Agent cant be in wall location
            # Block cant be in wall location
            if grid[ra, ca] != self.WALL and grid[rb, cb] != self.WALL:
                s = (ra, ca, rb, cb)
                self.states.add(s)

                if grid[ra,ca] == self.GOAL:
                    self.terminals.append(tuple(s))
                    r_value = 1.0
                elif grid[ra,ca] == self.START: r_value = 0.0
                elif grid[ra,ca] == self.EMPTY: r_value = 0.0
                elif grid[ra,ca] == self.BOX: r_value = 0.0
                else: raise Exception("Unknown grid attribute: ", grid[ra, ca])

                self.reward[s] = r_value
                #new_grid[ra, ca] = r_value
            elif grid[ra, ca] == self.WALL:
                self.walls.add((ra,ca))
                #new_grid[ra, ca] = None
                #new_grid[ra, ca] = None


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
            #row, col = s
            si = Si[s]
            for a in self.A:
                ai = Ai[a]
                sp, r, c = self.simulate(s, a)
                spi = Si[sp]
                P[si,ai,spi] += 1 * (1-self.randomness)
                R[si,ai,spi] = r
                C[si,ai,spi] = c

                for a_r in self.A:
                    ai_r = Ai[a_r]
                    sp, r, c = self.simulate(s, a_r)
                    spi = Si[sp]
                    P[si, ai, spi] += 1 * self.randomness/(len(self.A))
                    R[si, ai, spi] = r
                    C[si, ai, spi] = c

        Si.freeze(); Ai.freeze()
        return (s0, P, R, C), Si, Ai

    def check_box_location(self, box_location):
        open_area = []
        for a in self.A:
            dx, dy = a
            sp = (box_location[0] + dy, box_location[1] + dx)
            open_area.append( int(sp in self.walls) )
        return 1 if sum(open_area) >= 2 else 0


    def simulate(self, s, a):
        if s in self.terminals:
            return s, self.reward[s]*self.steps, self.check_box_location(s[2:])*self.steps

        # Transition the agent to next state
        dx, dy = a
        sp = [s[0] + dy, s[1] + dx, s[2], s[3]]

        # If the agent next position is the box location
        # move the box in the same direction
        if tuple(sp[:2]) == tuple(s[2:]):
            sp[2:] = [(s[2] + dy), (s[3] + dx)]
        sp = tuple(sp)

        if sp in self.states:
            sp = sp
            r = self.reward[sp]
        else:
            sp = s # stay in same state if we hit a wall
            r = 0
        c = self.check_box_location(sp[2:])*self.steps
        return sp, r, c

    def get_representation(self, s=None, Si=None):
        grid = np.zeros((2, self.grid.shape[0], self.grid.shape[1]))
        (x1, y1, x2, y2) = Si.lookup(s)
        grid[0][x1][y1] = 1
        grid[1][x2][y2] = 1
        #(x, y) = Si.lookup(s)
        #rep = list()
        #for x_loc, y_loc in self.terminals:
        #    rep.append(x_loc-x)
        #    rep.append(y_loc-y)
        #    if self.constraint[x_loc, y_loc] !=0:
        #        rep.append(-1)
        #    else:
        #        rep.append(1)
        #return rep
        return grid

