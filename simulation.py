import sys
print(sys.version)
import time
import numpy as np
import matplotlib
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

class Simulation(object):

    def __init__(self):
        self.params = Params()
        self.agents = list()
        self.obsts = list()
        self.cur_timestep = 0
        self.max_timesteps = self.params.max_timesteps
        self.time_vec = [0]

        # CBF Only Fields
        self.u_refs = list()

        # CLF Fields
        self.epsilon = self.params.epsilon

        # Plotting setup
        if self.params.plot_sim:
            self.sim_fig, self.sim_axes = plt.subplots()
            self.xlim = None
            self.ylim = None

    def add_agent(self, agent):
        # Priority values work best in range [0.25, 1.75]
        agent.id = len(self.agents)
        self.agents.append(agent)

    def add_obstacle(self, obst):
        obst.id = len(self.obsts)
        self.obsts.append(obst)

    def plot_scenario(self):
        self.sim_axes.cla()
        [ag.plot(self.sim_axes) for ag in self.agents]
        [ob.plot(self.sim_axes) for ob in self.obsts]

        if self.xlim is None:
            self.xlim = plt.xlim()
            self.ylim = plt.ylim()
        else:
            plt.xlim(self.xlim)
            plt.ylim(self.ylim)
        plt.pause(.01)

    def add_cbf_pair(self, m, agt, obst):
        m.update()
        
        if self.params.decentralized:
            obst_x_dot = obst.get_x_dot((0,0))
            k_cbf = obst.k_cbf
        else:
            obst_x_dot = obst.get_x_dot()
            k_cbf = 1.0
        p_cbf = self.params.p_cbf

        h_val = obst.h.eval(agt)
        lg_h = obst.h.grad(agt).T.dot(agt.get_x_dot() - obst_x_dot)[0][0]

        # Check if you need HOCBF
        if agt.dyn_enum is Dyn.DOUBLE_INT:
            # Could check lg_h to see if u shows up using lg_h.getVar() 
            # and lg_h.getCoeff() but that may be overkill for now
            k1_cbf = 1.0
            x = (agt.state-obst.state)
            x_dot = agt.get_x_dot() - obst.get_x_dot()
            x_ddot = np.vstack((x_dot[2:] , 0,0))
            lg2_h = 2*(x_dot.T).dot(obst.M).dot(x_dot) + 2*(x.T).dot(obst.M).dot(x_ddot)
            
            constr = m.addQConstr((lg2_h+k_cbf*lg_h+k1_cbf*(lg_h+k_cbf*h_val))[0][0]>=0, name="CBF_{}".format(agt.id))
            attr = GRB.Attr.QCRHS
        else:
            constr = m.addConstr((lg_h)>=-k_cbf*h_val**p_cbf, name="CBF_{}".format(agt.id))
            attr = GRB.Attr.RHS

        m.update()

        if self.params.plot_cbf:
            self.cbf_data.add_cbf_val(h_val, agt, obst)

        if self.params.plot_constrs:
            self.cbf_data.add_constr_val(constr.getAttr(attr),agt,obst)
    
    def setup_plots(self):
        clf_used = False
        for agt in self.agents:
            if agt.goal is not None:
                clf_used = True
                break
        if not clf_used:
            self.params.plot_clf=False
            self.params.plot_delta=False

        if len(self.agents)==1 and len(self.obsts)==0:
            self.params.plot_cbf = False
            self.params.plot_constrs =False

        if self.params.plot_clf or self.params.plot_delta:
            self.clf_data = Clf_data(self.agents)
        if self.params.plot_cbf or self.params.plot_constrs:
            self.cbf_data = Cbf_data(self.agents,self.obsts)

        # Set agent colors
        cm = ColorManager()
        for i in range(len(self.agents)):
            self.agents[i].color = cm.get_colors(i)
            

    def show_plots(self, save=False):
        if self.params.plot_clf or self.params.plot_delta:
            self.clf_data.plot(self.time_vec,save=save)
        if self.params.plot_cbf or self.params.plot_constrs:
            self.cbf_data.plot(self.time_vec,save=save)
        
        plt.rcParams.update({'font.size': 18})
        plt.show()

    def goalReached(self):
        goal_reached = False
        if self.cur_timestep == self.max_timesteps:
            goal_reached = True

        if not goal_reached:
            done=[]
            [done.append(agt.done) for agt in self.agents]
            if all(done):
                goal_reached = True

        if goal_reached:
            self.show_plots(save=True)

        if self.params.plot_sim:
            with PdfPages('z_scenario.pdf') as pdf:
                    pdf.savefig(self.sim_fig)

        return goal_reached

    def initiate(self, steps=None):
        # Setup plots
        self.setup_plots()

        if steps is not None:
            x_sol = np.zeros((len(self.agents), 2,steps))
            u_sol = np.zeros((len(self.agents), 2,steps))

        m = Model("CLF_CBF_QP")

        #Stop optimizer from publishing results to console
        m.Params.LogToConsole = 0

        goal_reached = False
        while not goal_reached:
            # Remove constraints and variables from previous loop
            m.remove(m.getConstrs())
            m.remove(m.getQConstrs())
            m.remove(m.getVars())

            for idx in range(len(self.agents)):
                # Get agent for this loop
                agt = self.agents[idx]

                # Add the Gurobi control variables
                agt.add_control(m,idx)
                
                # If goal is specified, add CLF constraints
                if agt.goal is not None:

                    # Control Lyapunov Function (Creates control variables for Model m)
                    v_val = agt.add_clf(m)
                    if self.params.plot_clf:
                        self.clf_data.add_clf_val(v_val, agt.id)

                # Otherwise, apply u_ref for the agent
                else:
                    u_ref = agt.u_ref
                    if u_ref is not None:
                        # Add CBF objective function for this agent
                        u_ref_t = make_column(u_ref[:,self.cur_timestep])
                        cost_func = (agt.u - u_ref_t).T.dot(agt.u - u_ref_t)[0][0]
                        m.setObjective(m.getObjective() + cost_func, GRB.MINIMIZE)
                        m.update()

            # Add Pariwise CBF Constraints
            for i in range(len(self.agents)):
                agt = self.agents[i]

                if self.params.decentralized:
                    agt2 = range(len(self.agents))
                    agt2.remove(i)
                else:
                    agt2 = range(i+1, len(self.agents))
                
                # CBF agent/agent
                for j in agt2:
                    self.add_cbf_pair(m, agt, self.agents[j])
                    
                # CBF agent/obstacle
                for k in range(len(self.obsts)):
                    self.add_cbf_pair(m, agt, self.obsts[k])

            #Solve the optimization problem
            m.optimize()
            solution = m.getVars()

            if self.params.plot_delta:
                [self.clf_data.add_delta_val(m.getVarByName("delta{}".format(agt.id)).x, agt.id) for agt in self.agents]

            [ag.step(None, plot=False) for ag in self.agents]

            if self.params.plot_sim:
                self.plot_scenario()

            
            if steps is not None:
                for i in range(len(self.agents)):
                    ag = self.agents[i]
                    state = ag.state
                    x_sol[i,0,self.cur_timestep] = state[0,0]
                    x_sol[i,1,self.cur_timestep] = state[1,0]
                    u_sol[i,0,self.cur_timestep] = ag.u[0,0].x
                    u_sol[i,1,self.cur_timestep] = ag.u[1,0].x

            self.cur_timestep += 1

            goal_reached = self.goalReached()

            if self.params.live_plots:
                self.show_plots()
            self.time_vec.append(self.cur_timestep*self.params.step_size)

            if self.cur_timestep == steps:
                # return (solution[0].x, solution[1].x)
                return x_sol, u_sol

def make_column(vec):
    vec = np.array(vec)
    vec.shape = (max(vec.shape),1)
    return vec

def main():
    start0 = time.time()
    Params('configs.ini')
    start1 = time.time()
    sim = Simulation()

    # a = 8
    # u_ref = np.hstack((np.ones((2,a)), np.zeros((2,100-a)))) * 1
    # sim.add_agent(Agent((0,0),u_ref,dynamics=Dyn.DOUBLE_INT))
    # sim.add_obstacle(Sphere((3,3.9), 1, dynamics=Dyn.DOUBLE_INT))

    # sim.add_agent((0, 0), goal=(20,0), radius=1, dynamics=Dyn.SINGLE_INT)
    
    dyn = Dyn.UNICYCLE
    sim.add_agent(Agent((0,0),Goal((10,0)),dynamics=dyn))
    #sim.add_agent(Agent((10,0),Goal((0,0)),dynamics=dyn))
    # sim.add_agent(Agent((2,2),Goal((-4,8)),dynamics=dyn), priorityVal=0.25)

    #sim.add_obstacle(Wall((0,20),(5,10), np.array([[1],[.4]]), -10))   
    
    sim.add_obstacle(Sphere((5.1,0), 0.5))

    sim.initiate()
    start2 = time.time()
    print("Initialization time:", start1 - start0)
    print("Simulation time:", start2 - start1)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("Total runtime:",end_time-start_time)