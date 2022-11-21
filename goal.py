import numpy as np
from dynamics import SingleIntegrator, Unicycle, DoubleIntegrator
from function import Function
from gurobipy import *
from params import *

class Goal(object):

    def __init__(self, pos):
        self.goal = np.array([pos]).T
        self.v = Function(self.v_func, self.del_v)

    def v_func(self, agt):
        x = agt.state[0:2] - self.goal[0:2]
        if not isinstance(agt.dyn, DoubleIntegrator):
            return x.T.dot(x)/2
        else:
            return x.T.dot(x)/2  + np.linalg.norm(agt.state[2:])**2 /2

    def del_v(self, agt):
        if not isinstance(agt.dyn, DoubleIntegrator):
            return agt.state - self.goal
        else:
            return agt.state - self.goal + np.linalg.norm(agt.state[2:])

    def distance_to_goal(self, agt):
        return np.sqrt((agt.state-self.goal).transpose().dot(agt.state-self.goal)[0][0])

    def plot(self, ax=None, color='g'):
        if ax is None:
            _, ax = plt.gca()
        ax.plot(self.goal[0], self.goal[1], 'x', color=color, markersize=10, markeredgewidth=5)
        ax.axis('equal')

    def __repr__(self):
        return "Goal({},{})".format(self.goal[0,0], self.goal[1,0])

    def add_clf(self, m, agt):
        params = Params()
        if type(agt.dyn) is SingleIntegrator:
            H = np.identity(len(agt.u)) * 10
            p = 20
            gamma = .25
        elif type(agt.dyn) is Unicycle:    
            # TODO: MAKE PARAMS TAKE A PARAM FILE. USE CONFIG. SEPARATE PARAMS
            vel_penalty =  params.vel_penalty #2 # was 10
            steer_penalty = params.steer_penalty # 1
            H = np.array([[vel_penalty, 0], [0, steer_penalty]])
            p = params.p
            gamma = params.gamma
        elif type(agt.dyn) is DoubleIntegrator:
            H = np.identity(len(agt.u)) * 10
            p = 20
            gamma = .25

        # Add relaxation variable delta
        delta = m.addVar(vtype=GRB.CONTINUOUS, name="delta{}".format(agt.id))
        m.update()
        v_val = self.v.eval(agt)[0][0]
        #print(v_val)
        lf_v = self.v.grad(agt).transpose().dot(agt.get_x_dot())[0][0]
        # print(lf_v)
        # lf_v = gamma*self.v.grad(agt).transpose().dot(H).dot(agt.get_x_dot())[0][0]
        constraint = (lf_v + gamma*v_val <=  delta)
        # print(constraint)
        cost_func = 0.5 * agt.u.transpose().dot(H).dot(agt.u)[0][0] \
            + gamma*self.v.grad(agt).transpose().dot(H).dot(agt.get_x_dot())[0][0] \
            + p*delta*delta
        if type(agt.dyn) is not Unicycle:
            m.addConstr(constraint)
        m.setObjective(cost_func + m.getObjective(), GRB.MINIMIZE)
        m.update()

        return v_val