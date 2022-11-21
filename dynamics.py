import sys
import numpy as np
import math
from gurobipy import *
from enum import Enum
import abc
from params import *

class Dyn(Enum):
    UNICYCLE = 0
    SINGLE_INT = 1
    DOUBLE_INT = 2

class Dynamics(object):

    @abc.abstractmethod
    def __init__(self, init_pos):
        pass

    @abc.abstractmethod
    def get_state(self, t_idx=None):
        pass

    @abc.abstractmethod
    def get_x_dot(self):
        pass

    @abc.abstractmethod
    def add_control(self):
        pass

    @abc.abstractmethod
    def step(self):
        pass

    @abc.abstractmethod
    def print_state(self):
        pass

class Unicycle(Dynamics):
    """
    Single Integrator dynamics that are transformed to behave like unicycle dynamics
    x(t) = [x, y, theta]'

    .       | cos(theta)  0 |
    x(t) =  | sin(theta)  0 | | v |
            |     0       1 | | w |
    """
    def __init__(self, init_pos, theta=0):
        self.params = Params()
        self.solver = self.params.solver
        l = self.params.l 
        
        self.rot_mat = lambda theta: (np.array([[math.cos(theta), -l*math.sin(theta)],[math.sin(theta),l*math.cos(theta)]]))

        init_pos = make_column(init_pos)
        
        self.cur_state = init_pos + l * np.array([[math.cos(theta)],[math.sin(theta)]])
        self.cur_theta = theta

        self.cur_time = 0
        
        self.time = np.array([0])
        self.trajectory = np.array(self.cur_state)

        self.time_step = self.params.step_size

    def get_state(self, t_idx=None):
        if t_idx is None:
            return self.cur_state
        else:
            return self.trajectory[:,t_idx]

    def get_x_dot(self, x, u):
        # print('cur_theta: ', self.cur_theta)
        # print('rot_mat: ', self.rot_mat(self.cur_theta))
        # print('x_dot: ', self.rot_mat(self.cur_theta).dot(np.array(u)))
        return self.rot_mat(self.cur_theta).dot(np.array(u))

    def add_control(self, m, id):
        v_ub = self.params.v_upper_bound
        w_ub = self.params.w_upper_bound
        v = m.addVar(lb=0, ub=v_ub, vtype=GRB.CONTINUOUS, name="vel{}".format(id))
        w = m.addVar(lb=-w_ub, ub=w_ub, vtype=GRB.CONTINUOUS, name="omega{}".format(id))
        return np.array([[v],[w]])

    def add_quad_control(self, name, variables, lb, ub):
        if name[0] == 'v':
            variables[name] = {'lb': lb, 'ub': ub}
        elif name[0] == 'w':
            variables[name] = {'lb': -ub, 'ub': ub}
        return variables
    
    def add_constraint(self, constraints, constraint):
        length  = len(constraints)
        return 1

    def step(self, u):
        x0 = self.cur_state
        x_dot = self.get_x_dot(x0, u)
        x1 = x0 + x_dot * self.time_step

        u = make_column(u)

        self.cur_state = x1
        self.cur_theta = self.cur_theta + self.time_step * u[1,0]
        self.cur_time += self.time_step

        self.time = np.append(self.time, self.cur_time)
        self.trajectory = np.append(self.trajectory, self.cur_state, axis=1)

        return self.cur_state

    def __str__(self):
        return 't={}\n'.format(self.cur_time) + np.array2string(self.cur_state)


class SingleIntegrator(Dynamics):

    def __init__(self, init_state=np.zeros((2,1))):
        """
        Single Integrator Dynamics
        
        x(t) = [x1, x2]'

        .      | 0   0 |        | 1  0 |
        x(t) = | 0   0 | x(t) + | 0  1 | u(t)
        
        y(t) = | 1  0 | x(t)
               | 0  1 |
        """
        self.params = Params('configs.ini')
        self.init_state = make_column(init_state)
        self.cur_state = self.init_state
        self.cur_theta = 0

        ndim = self.init_state.shape[0]
        self.A = np.zeros((ndim,ndim))
        self.B = np.identity(ndim)
        self.C = np.identity(ndim)
        
        self.time_step = self.params.step_size
        self.cur_time = 0
        self.time = np.array([0])
        self.trajectory = np.array(self.init_state)

    def get_state(self, t_idx=None):
        if t_idx is None:
            return self.cur_state
        else:
            return self.trajectory[:,t_idx]

    def get_x_dot(self, x, u=(0,0)):
        return make_column(self.B.dot(np.array(u)))

    def add_control(self, m, id):
        v = self.params.max_speed
        u = []
        for u_idx in range(len(self.cur_state)):
            u.append(m.addVar(lb=-v, ub=v, vtype=GRB.CONTINUOUS, name="agt{}_u{}".format(id, u_idx)))
        u = np.array(u)

        m.addConstr(u.transpose().dot(u) <= v**2)
        return make_column(u)

    def step(self, u):
        x0 = self.cur_state
        x_dot = self.get_x_dot(x0, u)
        x1 = x0 + x_dot * self.time_step

        self.cur_state = x1
        self.cur_time += self.time_step

        self.time = np.append(self.time, self.cur_time)
        self.trajectory = np.append(self.trajectory, self.cur_state, axis=1)
      
        return self.cur_state

    def __str__(self):
        return 't={}\n'.format(self.cur_time) + np.array2string(self.cur_state)


class DoubleIntegrator(Dynamics):

    def __init__(self, init_pos, init_vel=None, t0=0):
        """
        Double Integrator Dynamics
        
        x(t) = [x1, x2, v1, v2]'

        .      | 0   0   1   0 |        | 0  0 |
        x(t) = | 0   0   0   1 | x(t) + | 0  0 | u(t)
               | 0   0   0   0 |        | 1  0 |
               | 0   0   0   0 |        | 0  1 |
        
        y(t) = | 1  0  0  0 | x(t)
               | 0  1  0  0 |
        """
        self.params = Params()
        init_pos = make_column(init_pos)
        ndim = init_pos.shape[0]
        Z = np.zeros((ndim,ndim))
        I = np.identity(ndim)

        self.A = np.vstack((np.hstack((Z, I)), np.hstack((Z, Z)) ))
        self.B = np.vstack((Z,I))
        self.C = np.hstack((I,Z))
        
        init_pos = make_column(init_pos)
        if init_vel is None:
            init_vel = np.zeros((ndim,1))

        init_state = np.vstack((init_pos, init_vel))
        self.cur_state = init_state
        self.cur_theta = 0
        self.cur_time = 0
        
        self.time = np.array([0])
        self.trajectory = np.array(init_state)
        self.time_step = self.params.step_size

    def get_state(self, t_idx=None):
        if t_idx is None:
            return self.cur_state
        else:
            return self.trajectory[:,t_idx]

    def get_x_dot(self, x, u=(0,0)):
        return self.A.dot(x) + self.B.dot(make_column(u))

    def add_control(self, m, id):
        a = self.params.max_accel
        u = []
        for u_idx in range(self.B.shape[1]):
            u.append(m.addVar(lb=-a, ub=a, vtype=GRB.CONTINUOUS, name="agt{}_u{}".format(id, u_idx)))
        u = np.array(u)
        m.addConstr(u.transpose().dot(u) <= a**2, name="agt{}_ctrlBound".format(id))
        return make_column(u)

    def step(self, u):
        u = make_column(u)
        x0 = self.cur_state
        x_dot = self.get_x_dot(x0, u)
        x1 = x0 + x_dot * self.time_step + 0.5*np.vstack((u,np.zeros(u.shape)))*self.time_step**2

        self.cur_state = x1
        self.cur_time += self.time_step

        self.time = np.append(self.time, self.cur_time)
        self.trajectory = np.append(self.trajectory, self.cur_state)

        return self.cur_state

    def __str__(self):
        return 't={}\n'.format(self.cur_time) + np.array2string(self.cur_state)

def make_column(vec):
    vec = np.array(vec)
    vec.shape = (max(vec.shape),1)
    return vec
    
def repeat_control(mod, num, u=(1,0)):
    mod.print_state()
    for i in range(num):
        mod.step(u)
        print(mod)

def main():
    params = Params('configs.ini')    
    init = np.array([0,0])
    mod = Unicycle(init)
    repeat_control(mod, 5, (.5, .5))

if __name__ == '__main__':
    main()
        