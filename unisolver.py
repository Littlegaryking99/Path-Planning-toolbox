import sys
print(sys.version)
import time
import numpy as np
import matplotlib
import quadprog
from quadprog import solve_qp as quad_solve
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from dynamics import Dyn
from agent import Agent
from obstacle import Sphere, Ellipsoid, Wall
import gurobipy as gp
from gurobipy import GRB
from gurobipy import *
from goal import Goal
from sim_plots import Cbf_data, Clf_data, ColorManager
from params import *
import scipy.sparse as spa
from scipy.sparse import csr_matrix, lil_matrix
import constants as const
import math
import re
import warnings
import doctest
from collections import Counter
from collections.abc import Iterable

def listsolvers():
    solverlist = ["quadprog", "Gurobi"]
    print(solverlist)
    return 

class QpElement:
    """
    >>> x = QpElement(name = 'x', value = 5)
    >>> print(x)
    x
    """
    def __init__(self, name, value = 0):
        self.name = name
        self.value = value
        self.modified = True
        self.hash = id(self)

    def __hash__(self):
        return self.hash
    
    def __str__(self):
        return self.name
    
    def __pos__(self):
        return self

    def __neg__(self):
        self.value = -self.value
        return self
    
    def __add__(self, ele):
        # return self.value + ele.value
        return QpExpression(self) + ele

    def __sub__(self, ele):
        return QpExpression(self) - ele

    def __mul__(self, ele):
        return QpExpression(self) * ele

    def __pow__(self, ele):
        """
        >>> x = QpVariable('x_1', -10, 10, 0)
        >>> print(x**2)
        x_1**2
        """
        if isinstance(ele, int):
            n = self.name + '**' + str(ele)
            return QpElement(name = n, value = self.value ** ele)
        else:
            raise TypeError("Non-constant expressions cannot be exponentier")

    def __div__(self, ele):
        return QpExpression(self) / ele

    def __le__(self, ele):
        return QpExpression(self) <= ele

    def __ge__(self, ele):
        return QpExpression(self) >= ele

    def __eq__(self, ele):
        return QpExpression(self) == ele

class QpVariable(QpElement):
    """
    >>> x = QpVariable('x', lowbound = -10, upbound =  10)
    """
    def __init__(self, name = "NoName", lowbound = None, upbound = None, value = 0):
        QpElement.__init__(self, name, value)
        self.name = name
        self.lowbound = lowbound
        self.upbound = upbound
        self.value = value
    
    
    def ToDict(self):
        """
        >>> x = QpVariable('x', lowbound = -10, upbound =  10)
        >>> x.ToDict()
        {'name': 'x', 'lowbound': -10, 'upbound': 10, 'value': 0}
        """
        return dict(
            name = self.name,
            lowbound = self.lowbound,
            upbound=self.upbound,
            value = self.value,
        )

    def getLb(self):
        """
        >>> x = QpVariable('x', lowbound = -10, upbound = 10)
        >>> x.getLb()
        -10
        """
        return self.lowbound

    def getUb(self):
        """
        >>> x = QpVariable('x', lowbound = -10, upbound = 10)
        >>> x.getUb()
        10
        """
        return self.upbound

    def bounds(self, low, up):
        self.lowbound = low
        self.upbound = up
    
class QpExpression(dict):
    """
    A linear(quadratic) combination of class: QpVariable
    >>> x = QpVariable('x', lowbound = -10, upbound =  10)
    >>> y = QpExpression([(x, 2)], constant = 3, name = "c1")
    >>> y
    2*x + 3
    >>> x_name = ['x_0', 'x_1', 'x_2']
    >>> x = [QpVariable(x_name[i], lowbound = 0, upbound = 10) for i in range(3) ]
    >>> c = QpExpression([(x[0],1), (x[1],-3), (x[2],4)], constant = 3)
    >>> c
    1*x_0 + -3*x_1 + 4*x_2 + 3
    """
    def __init__(self, e=None, constant=0, name=None):
        self.name = name
        if e is None:
            e = {}
        if isinstance(e, QpExpression):
            # Will not copy the name
            # print("isexpression")
            self.constant = e.constant
            super().__init__(list(e.items()))
        elif isinstance(e, dict):
            self.constant = constant
            super().__init__(list(e.items()))
        elif isinstance(e, Iterable):
            self.constant = constant
            super().__init__(e)
        elif isinstance(e, QpElement):
            self.constant = 0
            super().__init__([(e, 1)])

    def value(self):
        s = self.constant
        for v, c in self.items():
            if v.value is None:
                return None
            s += v.value * c
        return s
    
    def copy(self):
        """Make a copy of self except the name which is reset"""
        # Will not copy the name
        return QpExpression(self)
    
    def emptyCopy(self):
        """Make an empty QpExpression"""
        return QpExpression()
    
    def addterm(self, key, value):
        y = self.get(key, 0)
        if y:
            y += value
            self[key] = y
        else:
            self[key] = value

    def __str__(self, constant=1):
        s = ""
        for v in self.sorted_keys():
            val = self[v]
            if val < 0:
                if s != "":
                    s += " - "
                else:
                    s += "-"
                val = -val
            elif s != "":
                s += " + "
            if val == 1:
                s += str(v)
            else:
                s += str(val) + "*" + str(v)
        if constant:
            if s == "":
                s = str(self.constant)
            else:
                if self.constant < 0:
                    s += " - " + str(-self.constant)
                elif self.constant > 0:
                    s += " + " + str(self.constant)
        elif s == "":
            s = "0"
        return s

    def sorted_keys(self):
        """
        returns the list of keys sorted by name
        """
        result = [(v.name, v) for v in self.keys()]
        result.sort()
        result = [v for _, v in result]
        # print(result)
        return result

    def __repr__(self):
        l = [str(self[v]) + "*" + str(v) for v in self.sorted_keys()]
        l.append(str(self.constant))
        s = " + ".join(l)
        return s

    def addInPlace(self, other):
        if isinstance(other, int) and (other == 0):
            return self
        if other is None:
            return self
        if isinstance(other, QpElement):
            self.addterm(other, 1)
        elif isinstance(other, QpExpression):
            self.constant += other.constant
            for v, x in other.items():
                self.addterm(v, x)
        elif isinstance(other, dict):
            for e in other.values():
                self.addInPlace(e)
        else:
            self.constant += other
        return self

    def subInPlace(self, other):
        if isinstance(other, int) and (other == 0):
            return self
        if other is None:
            return self
        if isinstance(other, QpElement):
            self.addterm(other, -1)
        elif isinstance(other, QpExpression):
            self.constant -= other.constant
            for v, x in other.items():
                self.addterm(v, -x)
        elif isinstance(other, dict):
            for e in other.values():
                self.subInPlace(e)
        else:
            self.constant -= other
        return self

    def __neg__(self):
        e = self.emptyCopy()
        e.constant = -self.constant
        for v, x in self.items():
            e[v] = -x
        return e

    def __pos__(self):
        return self

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __iadd__(self, other):
        return self.addInPlace(other)
        
    def __sub__(self, other):
        return self.copy().subInPlace(other)

    def __mul__(self, other):
        e = self.emptyCopy()
        if isinstance(other, QpExpression):
            e.constant = self.constant * other.constant
            if len(other):
                if len(self):
                    raise TypeError("Non-constant expressions cannot be multiplied")
                else:
                    c = self.constant
                    if c != 0:
                        for v, x in other.items():
                            e[v] = c * x
            else:
                c = other.constant
                if c != 0:
                    for v, x in self.items():
                        e[v] = c * x
        elif isinstance(other, QpVariable):
            return self * QpExpression(other)
        else:
            if other != 0:
                e.constant = self.constant * other
                for v, x in self.items():
                    e[v] = other * x
        return e   

    def __div__(self, other):
        if isinstance(other, QpExpression) or isinstance(other, QpVariable):
            if len(other):
                raise TypeError(
                    "Expressions cannot be divided by a non-constant expression"
                )
            other = other.constant
        e = self.emptyCopy()
        e.constant = self.constant / other
        for v, x in self.items():
            e[v] = x / other
        return e

    def __le__(self, other):
        return QpConstraint(self - other, -1)

    def __ge__(self, other):
        return QpConstraint(self - other, 1)

    def __eq__(self, other):
        return QpConstraint(self - other, 0)

    def toDict(self):
        """
        exports the :py:class:`LpAffineExpression` into a list of dictionaries with the coefficients
        it does not export the constant
        :return: list of dictionaries with the coefficients
        :rtype: list
        """
        return [dict(name=k.name, value=v) for k, v in self.items()]

    to_dict = toDict

class QpConstraint(QpExpression):
    """
    Combination of QpExpression and constant
    >>> x = QpVariable('x', lowbound = -10, upbound =  10)
    >>> y = QpExpression([(x, 2)], constant = 3, name = "c1")
    >>> z = QpConstraint(e = y, s = 0, rhs = 5)
    >>> print(z)
    2*x = 2
    >>> z
    2*x + -2 = 0
    """
    def __init__(self, e=None, s=0, name = None, rhs=None):
        QpExpression.__init__(self, e, name=name)
        if rhs is not None:
            self.constant -= rhs
        self.s = s
        self.pi = None
        self.slack = None
        self.modified = True
        self.rel = {0:"=", -1:"<=", 1:">="}
    
    def getLb(self):
        if (self.s == 1) or (self.s == 0):
            return -self.constant
        else:
            return None

    def getUb(self):
        if (self.s == -1) or (self.s == 0):
            return -self.constant
        else:
            return None

    def __str__(self):
        s = QpExpression.__str__(self, 0)
        if self.s is not None: 
            s += " " + self.rel[self.s] + " " + str(-self.constant)
        return s

    def __repr__(self):
        s = QpExpression.__repr__(self)
        if self.s is not None:
            s += " " + self.rel[self.s] + " 0"
        return s

    def changeRHS(self, RHS):
        self.constant = -RHS
        self.modified = True

    def copy(self):
        return QpConstraint(self, self.s)

    def emptyCopy(self):
        return QpConstraint(sense=self.s)

    def addInPlace(self, other):
        if isinstance(other, QpConstraint):
            if self.s * other.s >= 0:
                QpExpression.addInPlace(self, other)
                self.s |= other.s
            else:
                QpExpression.subInPlace(self, other)
                self.s |= -other.s
        elif isinstance(other, list):
            for e in other:
                self.addInPlace(e)
        else:
            QpExpression.addInPlace(self, other)
            # raise TypeError, "Constraints and Expressions cannot be added"
        return self

    def subInPlace(self, other):
        if isinstance(other, QpConstraint):
            if self.s * other.s <= 0:
                QpExpression.subInPlace(self, other)
                self.s |= -other.s
            else:
                QpExpression.addInPlace(self, other)
                self.s |= other.s
        elif isinstance(other, list):
            for e in other:
                self.subInPlace(e)
        else:
            QpExpression.subInPlace(self, other)
            # raise TypeError, "Constraints and Expressions cannot be added"
        return self

    def __neg__(self):
        c = QpExpression.__neg__(self)
        c.sense = -c.sense
        return c

    def __add__(self, other):
        return self.copy().addInPlace(other)

    def __radd__(self, other):
        return self.copy().addInPlace(other)

    def __sub__(self, other):
        return self.copy().subInPlace(other)

    def __rsub__(self, other):
        return (-self).addInPlace(other)

    def __mul__(self, other):
        if isinstance(other, QpConstraint):
            c = QpExpression.__mul__(self, other)
            if c.sense == 0:
                c.sense = other.sense
            elif other.sense != 0:
                c.sense *= other.sense
            return c
        else:
            return QpExpression.__mul__(self, other)

    def toDict(self):
        return dict(
            constant=self.constant,
            coefficients=QpExpression.toDict(self),
        )

class QpConstraintVar(QpElement):

    def __init__(self, name=None, sense=None, rhs=None, e=None):
        QpElement.__init__(self, name)
        self.constraint = QpConstraint(name=self.name, sense=sense, rhs=rhs, e=e)

    def addVariable(self, var, coeff):
        self.constraint.addterm(var, coeff)

    def value(self):
        return self.constraint.value()

class QpProblem:
    """
    Main Structure of QP Problem.
    >>> prob = QpProblem("myProblem", "quadprog")
    >>> x = QpVariable("x", 0, 3)
    >>> y = QpVariable("y", 0, 1)
    >>> prob += x ** 2 * 2 + y ** 2 * 5 + x + y * 5
    >>> prob.objective
    1*x + 2*x**2 + 5*y + 5*y**2 + 0
    >>> prob += x + y <= 3
    >>> prob += x + y >= 2
    >>> prob.constraints["c0"]
    1*x + 1*y + -3 <= 0
    >>> print(prob.constraints["c0"])
    x + y <= 3
    >>> prob.solve()
    [[2.]
     [0.]]
    """
    def __init__(self, name="NoName", solver = None):
        self.objective = None
        self.constraints = dict()
        self.name = name    
        self.solver = solver
        self.solvers = ['Gurobi','quadprog']
        self._variables = []
        self._variable_ids = {}
        self.result = None
        self.P = None
        self.q = None
        self.G = None
        self.h = None
        self.numofc = 0
        if self.solver == "Gurobi":
            self.mod = gp.Model("qp")

    def getsolvers(self):
        return self.solvers

    def setsolver(self, name):
        self.solver = name

    def __getstate__(self):
        # Remove transient data prior to pickling.
        state = self.__dict__.copy()
        del state["_variable_ids"]
        return state

    def __setstate__(self, state):
        # Update transient data prior to unpickling.
        self.__dict__.update(state)
        self._variable_ids = {}
        for v in self._variables:
            self._variable_ids[v.hash] = v

    def copy(self):
        lpcopy = QpProblem(name=self.name)
        lpcopy.objective = self.objective
        lpcopy.constraints = self.constraints.copy()

    def addVariable(self, variable):
        if variable.hash not in self._variable_ids:
            self._variables.append(variable)
            self._variable_ids[variable.hash] = variable
    
    def addVariables(self, variables):
        for v in variables:
            self.addVariable(v)
    
    def variables(self):
        if self.objective:
            self.addVariables(list(self.objective.keys()))
        for c in self.constraints.values():
            self.addVariables(list(c.keys()))
        self._variables.sort(key=lambda v: v.name)
        return self._variables

    def variablesDict(self):
        variables = {}
        if self.objective:
            for v in self.objective:
                variables[v.name] = v
        for c in list(self.constraints.values()):
            for v in c:
                variables[v.name] = v
        return variables

    def add(self, constraint, name=None):
        self.addConstraint(constraint, name)

    def addConstraint(self, constraint, name=None, mod = None):
        self.constraints[name] = constraint
        return
        # self.modifiedConstraints.append(constraint)
        # self.addVariables(list(constraint.keys()))

    def setObjective(self, obj):
        if self.solver  == "Gurobi":
            self.mod.setObjective(obj)
        else:
            if isinstance(obj, QpVariable):
                # allows the user to add a LpVariable as an objective
                obj = obj + 0.0
            try:
                obj = obj.constraint
                name = obj.name
            except AttributeError:
                name = None
            self.objective = obj
            self.objective.name = name
    
    def getVariables(self):
        varlist = []
        for i in self._variables:
            varlist.append(i.name[0])
        tmplist = list({}.fromkeys(varlist).keys())
        return tmplist

    def numOfPow(self, exp):
        return str(exp).count("**")

    def solve(self, solver=None, mod = None, **kwargs):
        if not (solver):
            solver = self.solver
        # time it
        self.startClock()
        self.variables()
        list1 = self.getVariables()
        numofvar = len(list1)
        numofmul = self.numOfPow(self.objective)
        self.P = np.zeros((numofvar, numofvar))
        self.q = np.zeros((numofvar, 1))
        self.G = np.zeros((self.numofc, numofvar))
        self.h = np.zeros((self.numofc, 1))
        muldict = dict()
        for k, v in self.objective.items():
            muldict[k.name] = v
        for i in range(numofvar):
            indexP = str(list1[i] + "**2")
            indexq = str(list1[i])
            if indexP in muldict:
                self.P[i][i] = muldict[indexP]
            if str(list1[i]) in muldict:
                self.q[i][0] = muldict[indexq]
        for i in range(self.numofc):
            index = "c" + str(i)
            tmpdict = dict()
            s = QpExpression(self.constraints[index], 0)
            eq = self.constraints[index].s
            for c, d in s.items():
                tmpdict[c.name] = d
            for j in range(numofvar):
                if str(list1[j]) in tmpdict:
                    self.G[i][j] = -eq * tmpdict[str(list1[j])]
                    self.h[i][0] = eq * self.constraints[index].constant
        if self.solver == "quadprog":
            sol = self.quadSolve(numofvar)
        elif self.solver == "Gurobi":
            sol = self.gurobiSolve(numofvar)
        print(sol)
        # print(muldict)
        # if self.solver == "Gurobi":
        #     mod.Params.OutputFlag = 0
        #     mod.optimize()
        #     status = [mod.getVars()[0].X, mod.getVars()[1].X]
        # else:
        #     return
            # status = solver.actualSolve(self, **kwargs)
        self.stopClock()
        self.solver = solver
        return

    def quadSolve(self, numofvar = 0):
        qp_G = self.P
        qp_a = -self.q.flatten()
        qp_C = -self.G.T
        qp_b = -self.h.flatten()
        meq = 0
        sol = quad_solve(qp_G, qp_a, qp_C, qp_b, meq)[0].reshape(numofvar, 1)
        return sol

    def gurobiSolve(self, numofvar = 0):
        ubs, lbs = np.zeros((numofvar,1)), np.zeros((numofvar,1))
        model = gp.Model()
        model.Params.OutputFlag = 0
        identity = spa.eye(numofvar)
        flag = 0
        for item in self._variables:
            if isinstance(item, QpVariable):
                ubs[flag] = item.getUb()
                lbs[flag] = item.getLb()
                flag += 1
        ubs.reshape((numofvar,1))
        lbs.reshape((numofvar,1))
        t = model.addMVar(numofvar, lb = -GRB.INFINITY, ub = GRB.INFINITY, vtype = GRB.CONTINUOUS)
        model.update()
        ineqConstr, lbConstr, ubConstr = None, None, None
        ineqConstr = model.addMConstr(self.G, t, GRB.LESS_EQUAL, self.h)
        lb_constr = model.addMConstr(identity, t, GRB.GREATER_EQUAL, lbs)
        ub_constr = model.addMConstr(identity, t, GRB.LESS_EQUAL, ubs)
        objective = 0.5 * (t @ self.P @ t) + self.q.flatten() @ t
        model.setObjective(objective, sense = GRB.MINIMIZE)
        model.update()
        model.optimize()
        status = []
        for i in range(numofvar):
            status.append(model.getVars()[i].X)
        return status

    def startClock(self):
        self.solutionTime = -time.time()

    def stopClock(self):
        self.solutionTime += time.time()

    def __iadd__(self, other):
        if isinstance(other, tuple):
            other, name = other
        else:
            name = "c0"
        if other is True:
            return self
        elif other is False:
            raise TypeError("A False object cannot be passed as a constraint")
        elif isinstance(other, QpConstraint):
            name = "c" + str(self.numofc)
            self.addConstraint(other, name)
            self.numofc += 1
        elif isinstance(other, QpExpression):
            if self.objective is not None:
                warnings.warn("Overwriting previously set objective.")
            self.objective = other
            if name is not None:
                self.objective.name = name
        elif isinstance(other, QpVariable) or isinstance(other, (int, float)):
            if self.objective is not None:
                warnings.warn("Overwriting previously set objective.")
            self.objective = QpExpression(other)
            self.objective.name = name
        else:
            raise TypeError(
                "Can only add LpConstraintVar, LpConstraint, LpAffineExpression or True objects"
            )
        return self

    def numVariables(self):
        return len(self._variable_ids)

    def numConstraints(self):
        return len(self.constraints)


def main():
    # doctest.testmod(verbose = True)
    # for i in range(2000):
    prob = QpProblem("myProblem", "Gurobi")
    x = QpVariable("x", 0, 3)
    y = QpVariable("y", 0, 1)
    prob += x ** 2 * 2 + y ** 2 * 5 + x + y * 5
    prob += y + x <= 3
    # print(prob.constraints["c0"].toDict())
    prob += x + y >= 2
    prob.solve()

    # print(prob.objective)
    # print(prob.solutionTime)
    # print(prob.objective.toDict())
    # print(str(prob.objective))
    # print(str(prob.objective).count("**"))
    # prob.addVariable("x", 0, 3)
    # prob.addVariable("y", 0, 1)
    # prob.mod.update()
    # print(prob.mod)
    # x = mod.addVar(lb = 0, ub = 3, name = "x")
    # y = mod.addVar(lb = 0, ub = 3, name = "y")
    # mod.addConstr(x + y <= 3)
    # prob += x + y <= 3
    # prob.addConstraint(x + y <= 3)
    # obj = x ** 2 + y
    
    # x = prob.mod.addVar(lb = 1, ub = 3, name = "x")
    # y = prob.mod.addVar(lb = 1, ub = 1, name = "y")
    # prob.setObjective(x**2 + y)
    # mod.addConstr(x + y <= 3)
    # print(prob.mod)
    # prob += x ** 2 + y
    # prob += x + y <= 3
    # sol = prob.solve()
    # print(sol)
    
    

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    # print("Total runtime: ", end_time - start_time)