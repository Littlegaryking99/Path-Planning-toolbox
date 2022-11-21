import sys
import time
import numpy as np
import matplotlib
import quadprog
from quadprog import solve_qp
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dynamics import Dyn
from agent import Agent
from obstacle import Sphere, Ellipsoid, Wall
from gurobipy import *
from goal import Goal
from sim_plots import Cbf_data, Clf_data, ColorManager
from params import *
from scipy.sparse import csr_matrix, lil_matrix
import math

class Quadmodel(object):
    def __init__(self, numofa = 0, numofo = 0):
        self.params = Params('configs.ini')
        self.P = []
        self.q = []
        self.G = []
        self.h = []
        self.numofa = numofa
        self.numofo = numofo
    
    def buildModel(self, dyn):
        #Initialize P, q, G, h for a stardard form of quadratic programs
        #min            1/2x^TPx + q^Tx
        #subject to     Gx <= h
        # x: 3 * numofa, 1   
        if dyn is Dyn.UNICYCLE:
            self.P = np.zeros((3 * self.numofa, 3 * self.numofa))
            self.q = np.zeros(3 * self.numofa)
            for i in range(self.numofa):
                self.P[3 * i, 3 * i] = self.params.vel_penalty
                self.P[3 * i + 1, 3 * i + 1] = self.params.steer_penalty
                self.P[3 * i + 2, 3 * i + 2] = self.params.p

    def add_var(self, lb, ub, name, agt):
        if agt.dyn_enum == Dyn.UNICYCLE:
            index = agt.id
            newconstraint_lb = np.zeros(3 * self.numofa)
            newconstraint_ub = np.zeros(3 * self.numofa)
            if name[0] == 'v':
                newconstraint_lb[3 * agt.id] = -1
                newconstraint_ub[3 * agt.id] = 1
            elif name[0] == 'w':
                newconstraint_lb[3 * agt.id + 1] = -1
                newconstraint_ub[3 * agt.id + 1] = 1
            elif name[0] == 'd':
                newconstraint_lb[3 * agt.id + 2] = -1
                newconstraint_ub[3 * agt.id + 2] = 1
            else:
                return 
            self.G.append(newconstraint_lb)
            self.G.append(newconstraint_ub)
            self.h.append(-lb)
            self.h.append(ub)
            

    def add_constraint(self, constraint, co_c):
        #input example: solver.addconstraint({'v': 1, 'w': -1, 'd': 0}, 2), var: v, w, d
        newconstraint = np.zeros(3 * self.numofa)
        for name in constraint.keys():
            index = name[1:]
            if name[0] == 'v':
                #agent start from 0
                newconstraint[3 * int(index)] = constraint[name]
            elif name[0] == 'w':
                newconstraint[3 * int(index) + 1] = constraint[name]
            elif name[0] == 'd':
                newconstraint[3 * int(index) + 2] = constraint[name]
        self.G.append(newconstraint)
        self.h.append(co_c)
        return 1
    
    def clear_model(self):
        self.G = self.G[:4 * self.numofa]
        self.h = self.h[:4 * self.numofa]


def main():
    model = Quadmodel(2, 1)
    model.add_constraint({'v0': -9.851038323888194, 'w0': 0.9884006999285284, 'd0': -1}, 5)
    

if __name__ == '__main__':
    main()
