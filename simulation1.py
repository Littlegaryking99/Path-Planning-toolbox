import sys
#print(sys.version)
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from dynamics import Dyn
from agent import Agent
from obstacle import Sphere, Ellipsoid, Wall
#from gurobipy import *
from goal import Goal
from sim_plots import Cbf_data, Clf_data, ColorManager
from gurobisolver import Gurobisolver
from quadprogsolver import Quadprogsolver
from quadmodel import Quadmodel
from params import *

class Simulation(object):

    def __init__(self):
        self.params = Params()
        self.solvernum = self.params.solver
        self.solvers = [Gurobisolver, Quadprogsolver]
        self.solver = self.solvers[self.solvernum]()

def main():
    params = Params('configs.ini')
    sim = Simulation()
    dyn = Dyn.UNICYCLE
    sim.solver.add_agent(Agent((0,0),Goal((8,0)),radius=0.5, dynamics=dyn))
    sim.solver.add_agent(Agent((0,2),Goal((3.5,-1)),radius=0.5, dynamics=dyn))
    sim.solver.add_obstacle(Sphere((2,1), 1))
    # sim.solver.add_obstacle(Sphere((5,0), 1))
    sim.solver.add_obstacle(Ellipsoid((5.5,0.5), (1,2)))


    sim.solver.initiate(dyn)
    # a = 8
    # u_ref = np.hstack((np.ones((2,a)), np.zeros((2,100-a)))) * 1
    # sim.add_agent(Agent((0,0),u_ref,dynamics=Dyn.DOUBLE_INT))
    # sim.add_obstacle(Sphere((3,3.9), 1, dynamics=Dyn.DOUBLE_INT))

    # sim.add_agent((0, 0), goal=(20,0), radius=1, dynamics=Dyn.SINGLE_INT)
    
    # dyn = Dyn.UNICYCLE
    # sim.solver.add_agent(Agent((0,0),Goal((10,0)),dynamics=dyn))
    #sim.add_agent(Agent((10,0),Goal((0,0)),dynamics=dyn))
    # sim.add_agent(Agent((2,2),Goal((-4,8)),dynamics=dyn), priorityVal=0.25)

    # sim.solver.add_obstacle(Wall((0,20),(5,0), np.array([[1],[.4]]), -10))  
    
    #sim.solver.add_obstacle(Sphere((5.1,0), 0.5))
    #sim.solver.add_agent(Agent((5,0.0),Goal((12,0)),dynamics=dyn))
    # sim.solver.add_agent(Agent((0,0),Goal((10,0)),radius=1, dynamics=dyn))
    #sim.solver.add_agent(Agent((0,2),Goal((10,2)),dynamics=dyn))
    # sim.solver.add_obstacle(Sphere((5,0), 1))
    #sim.solver.add_obstacle(Sphere((7,0), 1))
    #sim.solver.add_obstacle(Sphere((3,6), 1))
    #sim.solver.add_obstacle(Sphere((7,5), 1))
    # sim.solver.initiate(dyn)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime: ", (end_time - start_time))