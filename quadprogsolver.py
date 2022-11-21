import sys
print(sys.version)
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

class Quadprogsolver(object):
    def __init__(self):
        self.params = Params('configs.ini')
        self.agents = list()
        self.obsts = list()
        self.cur_timestep = 0
        self.max_timesteps = self.params.max_timesteps
        self.time_vec = [0]

        # CBF Only Fields
        self.u_refs = list()
        self.l = self.params.l

        # CLF Fields
        self.epsilon = self.params.epsilon

        # Plotting setup
        if self.params.plot_sim:
            self.sim_fig, self.sim_axes = plt.subplots()
            self.xlim = None
            self.ylim = None
        
        # QP Solver
        self.P = []
        self.q = []
        self.G = []
        self.h = []
        self.numofa = len(self.agents)
        self.numofo = len(self.obsts)

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
                self.P[3 * i + 2, 3 * i + 2] = 2 * self.params.p
        elif dyn is Dyn.SINGLE_INT:
            self.P = np.identity(3 * self.numofa) * 10
            for i in range(self.numofa):
                self.P[3 * i, 3 * i] = 20
            self.q = np.zeros(3 * self.numofa)
        elif dyn is Dyn.DOUBLE_INT:
            self.P = np.identity(5 * self.numofa) * 10
            for i in range(self.numofa):
                self.P[5 * i, 5 * i] = 20
            self.q = np.zeros(5 * self.numofa)
           

    def add_agent(self, agent):
        # Priority values work best in range [0.25, 1.75]
        agent.id = len(self.agents)
        self.agents.append(agent)
        self.numofa += 1

    def add_obstacle(self, obst):
        obst.id = len(self.obsts)
        self.obsts.append(obst)
        self.numofo += 1

    def add_var(self, lb, ub, name, agt):
        if agt.dyn_enum == Dyn.UNICYCLE or agt.dyn_enum == Dyn.SINGLE_INT:
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
        elif agt.dyn_enum == Dyn.DOUBLE_INT:
            index = agt.id
            
    
    def add_constraint(self, constraint, co_c, dyn_enum):
        #input example: solver.addconstraint({'v': 1, 'w': -1, 'd': 0}, 2), var: v, w, d
        if dyn_enum == Dyn.UNICYCLE or Dyn.SINGLE_INT or Dyn.DOUBLE_INT:
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

    def add_clf(self, agt):
        rot_matrix = np.array([math.cos(agt.dyn.cur_theta), math.sin(agt.dyn.cur_theta)])
        if agt.dyn_enum == Dyn.UNICYCLE:
            x = agt.dyn.cur_state - agt.goal.goal[0:2]
            gamma = self.params.gamma
            rot_matrix1 = np.array([-self.l * math.sin(agt.dyn.cur_theta), self.l * math.cos(agt.dyn.cur_theta)])
            #For the case of V(x) = x.T.dot(x)/2, x = agt.state[0:2] - self.goal[0:2]
            co_v = x.T.dot(rot_matrix.T)
            co_w = x.T.dot(rot_matrix1.T)
            co_d = -1
            co_c = -gamma / 2 * x.T.dot(x)[0][0]
            dict = {'v' + str(agt.id): co_v, 'w' + str(agt.id): co_w, 'd' + str(agt.id): co_d}
        elif agt.dyn_enum == Dyn.SINGLE_INT:
            #v represents u1, w represents u2        
            x = agt.dyn.cur_state - agt.goal.goal[0:2]
            gamma  = .25
            co_v = x[0][0]
            co_w = x[1][0]
            co_d = -1
            co_c = -gamma / 2 * x.T.dot(x)[0][0]
            dict = {'v' + str(agt.id): co_v, 'w' + str(agt.id): co_w, 'd' + str(agt.id): co_d}
        elif agt.dyn_enum == Dyn.DOUBLE_INT:
            x = agt.dyn.cur_state[0:2] - agt.goal.goal[0:2]
            vel = agt.dyn.cur_state[2:]
            gamma = .25
            co_v = x[0][0] + agt.dyn.cur_state[2]
            co_w = x[1][0] + agt.dyn.cur_state[3]
            co_d = -1
            co_c = -vel.T.dot(vel)[0][0] - x.T.dot(vel)[0][0] - gamma / 2 * (vel.T.dot(vel) + x.T.dot(x)) [0][0]
            dict = {'v' + str(agt.id): co_v, 'w' + str(agt.id): co_w, 'd' + str(agt.id): co_d}
        return dict, co_c

    def add_cbf(self, agt, obst):
        rot_matrix = np.array([math.cos(agt.dyn.cur_theta), math.sin(agt.dyn.cur_theta)])
        rot_matrix1 = np.array([math.sin(agt.dyn.cur_theta), -math.cos(agt.dyn.cur_theta)])
        if self.params.decentralized:
            k_cbf = obst.k_cbf
        else:
            k_cbf = 1.0
        p_cbf = self.params.p_cbf
        x = agt.dyn.cur_state[0:2] - obst.state[0:2]
        if agt.dyn_enum is Dyn.UNICYCLE:
            #For the case of h(x) = (x-x_c)^T(x-x_c) - (r+d_s)^2
            co_v = -2 * x.T.dot(rot_matrix.T)
            co_w = 2 * x.T.dot(rot_matrix1.T)* self.l
            co_c = k_cbf * (x.T.dot(x) - (obst.radius + agt.radius) ** 2)[0][0] ** p_cbf
            cdict = {'v' + str(agt.id): co_v, 'w' + str(agt.id): co_w}
        elif agt.dyn_enum == Dyn.SINGLE_INT:
            co_v = -x[0][0] * 2
            co_w = -x[1][0] * 2
            co_c = k_cbf * (x.T.dot(x) - (obst.radius + agt.radius) ** 2)[0][0] ** p_cbf
            cdict = {'v' + str(agt.id): co_v, 'w' + str(agt.id): co_w}
        elif agt.dyn_enum == Dyn.DOUBLE_INT:
            vel = agt.dyn.cur_state[2:]
            k1_cbf = 1.0
            co_v = -x[0][0] * 2
            co_w = -x[1][0] * 2
            co_c = 2 * vel.T.dot(vel)[0][0] + k1_cbf * (2 * x.T.dot(vel) + k_cbf * (x.T.dot(x) - (obst.radius + agt.radius) ** 2))[0][0]
            cdict = {'v' + str(agt.id): co_v, 'w' + str(agt.id): co_w}
        return cdict, co_c
        
    
    def quadprog_one_iteration(self):
        #Calculate P, q, G, h for a stardard form of quadratic programs
        #min            1/2x^TPx + q^Tx
        #subject to     Gx <= h
        #CLF Constraint
        for i in range(self.numofa):
            cdict, co_c = self.add_clf(self.agents[i])
            self.add_constraint(cdict, co_c, self.agents[i].dyn_enum)
        #CBF Constraint
        for i in range(self.numofa):
            for j in range(self.numofa - i - 1):
                cdict, co_c = self.add_cbf(self.agents[i], self.agents[i + j])
                self.add_constraint(cdict, co_c, self.agents[i].dyn_enum)

            for j in range(self.numofo):
                cdict, co_c = self.add_cbf(self.agents[i], self.obsts[j])                
                self.add_constraint(cdict, co_c, self.agents[i].dyn_enum)
        #Transform to Quadprog Standard Form
        meq = 0
        G = np.array(self.P)
        a = -np.array(self.q)
        C = -np.array(self.G).T
        # print(C)
        b = -np.array(self.h).T
        # print(b)
        sol = solve_qp(G, a, C, b, meq)[0].reshape((self.numofa,3))
        # print(sol)
        return sol
        
    def iteration (self, steps=None):
        for i in range(self.numofa):
            if self.agents[i].dyn_enum is Dyn.UNICYCLE:
                self.add_var(0, self.params.v_upper_bound, 'v' + str(i), self.agents[i])
                self.add_var(-self.params.w_upper_bound, self.params.w_upper_bound, 'w' + str(i), self.agents[i])
            elif self.agents[i].dyn_enum is Dyn.SINGLE_INT:
                max_speed = self.params.max_speed ** 0.5
                self.add_var(-max_speed, max_speed, 'v' + str(i), self.agents[i])
                self.add_var(-max_speed, max_speed, 'w' + str(i), self.agents[i])
            elif self.agents[i].dyn_enum is Dyn.DOUBLE_INT:
                max_acc = self.params.max_accel ** 0.5
                self.add_var(-max_acc, max_acc, 'v' + str(i), self.agents[i])
                self.add_var(-max_acc, max_acc, 'w' + str(i), self.agents[i])
        goal_reached = False
        if steps is not None:
            x_sol = np.zeros((len(self.agents), 2,steps))
            u_sol = np.zeros((len(self.agents), 2,steps))
        #while not goal_reached:   
        while not goal_reached:
            # print(self.cur_timestep)
            sol = self.quadprog_one_iteration()
            for i in range(self.numofa):
                self.agents[i].quadstep(sol[i])
            if steps is not None:
                for i in range(len(self.agents)):
                    ag = self.agents[i]
                    state = ag.state
                    x_sol[i,0,self.cur_timestep] = state[0,0]
                    x_sol[i,1,self.cur_timestep] = state[1,0]
                    u_sol[i,0,self.cur_timestep] = ag.u[0,0].x
                    u_sol[i,1,self.cur_timestep] = ag.u[1,0].x
            self.clear_model()
            self.cur_timestep += 1
            goal_reached = self.goalReached()
            # print(self.cur_timestep)
            # goal_reached = True
            #print(self.agents[0].state)
            # if self.cur_timestep == 1:
            if self.cur_timestep == self.max_timesteps:
                # return (solution[0].x, solution[1].x)
                return 1
        return 1

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

        # if goal_reached:
        #     self.show_plots(save=True)

        return goal_reached

    def initiate(self, dyn):
        self.buildModel(dyn)
        self.iteration()

def make_column(vec):
    vec = np.array(vec)
    vec.shape = (max(vec.shape),1)
    return vec

def matrix_test():
    start_time = time.time()
    params = Params('configs.ini')
    end_time = time.time()
    #print(end_time - start_time)
    start_time1 = time.time()
    sparseMatrix  = lil_matrix((100,100),dtype  = np.int8)
    sparseMatrix[0,0] = 1
    sparseMatrix[0,1] = 4
    sparseMatrix[1,0] = 8
    sparseMatrix[1,2] = 11
    sparseMatrix[2,2] = 9
    print(sparseMatrix)
    end_time1 = time.time()
    print('Total runtime: ',end_time1 - start_time1)
    start_time2 = time.time()
    row = list()
    col = list()
    data = list()
    row.append(0)
    col.append(0)
    data.append(1)
    row.append(0)
    col.append(1)
    data.append(4)
    row.append(1)
    col.append(0)
    data.append(8)
    row.append(1)
    col.append(2)
    data.append(11)
    row.append(2)
    col.append(2)
    data.append(9)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    sparse_Matrix1  = csr_matrix((data, (row, col)), shape = (100,100))
    end_time2 = time.time()
    print('Total runtime: ',end_time2 - start_time2)
    start_time3 = time.time()
    arr = np.zeros((100,100))
    arr[0,0] = 1
    arr[0,1] = 4
    arr[1,0] = 8
    arr[1,2] = 11
    arr[2,2] = 9
    end_time3 = time.time()
    print('Total runtime: ',end_time3 - start_time3)
    print(arr)

def main():
    Params('configs.ini')
    start_time = time.time()
    model = Quadprogsolver()
    dyn = Dyn.DOUBLE_INT
    model.add_agent(Agent((2.0,0.0),Goal((10,0)),dynamics=dyn))
    #model.add_agent(Agent((0,2),Goal((10,2)),dynamics=dyn))
    model.add_obstacle(Sphere((5,0), 1))
    # model.add_obstacle(Wall((0,20),(5,10), np.array([[1],[.4]]), -10)) 
    model.buildModel(dyn)
    model.iteration()
    # for j in range(300):
    #     for i in range(2):
    #         model.agents[i].quadstep(sol[i])
    end_time = time.time()
    print('Quadprog Executing time:', end_time - start_time)
    # x = np.array([[1],[2],[3],[4]])
    # y = x - np.linalg.norm(x[2:])
    # print(x)
    # print(y)
    # print(np.linalg.norm(x))
    # print(np.linalg.norm(x[2:]))
    


if __name__ == '__main__':
    main()

